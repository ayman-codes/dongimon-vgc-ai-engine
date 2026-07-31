"""Tests for MP-based selection scoring and mode integration.

Covers the MP scoring module (src.selection.mp_scoring) and the
selection policy's two modes (mp_only, mp_sim).
Uses mock models for deterministic inference and real PokemonSpecies
objects for feature computation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vgc2.battle_engine.modifiers import Category, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team

from src.data.features import compute_pairwise_features
from src.selection.mp_scoring import (
    _features_to_array,
    _load_feature_order,
    score_pair_mp,
    score_roster_mp,
)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "src" / "models"
_FEATURE_ORDER_PATH = _MODELS_DIR / "feature_order.json"


def _make_move(
    name: str,
    pkm_type: Type = Type.NORMAL,
    base_power: int = 80,
    category: Category = Category.PHYSICAL,
    accuracy: float = 1.0,
) -> Move:
    """Build a minimal damaging Move.

    Args:
        name: Move name.
        pkm_type: vgc2 Type of the move.
        base_power: Base power.
        category: Physical or Special category.
        accuracy: Accuracy fraction.

    Returns:
        A Move instance.
    """
    return Move(
        pkm_type=pkm_type,
        base_power=base_power,
        accuracy=accuracy,
        max_pp=15,
        category=category,
        name=name,
    )


def _make_species(
    name: str,
    types: list[Type] | None = None,
    hp: int = 100,
    atk: int = 80,
    df: int = 70,
    spa: int = 80,
    spd: int = 70,
    spe: int = 80,
    moves: list[Move] | None = None,
) -> PokemonSpecies:
    """Build a PokemonSpecies with explicit stats and moves.

    Args:
        name: Species name.
        types: List of vgc2 Types.
        hp: Base HP.
        atk: Base Attack.
        df: Base Defense.
        spa: Base Special Attack.
        spd: Base Special Defense.
        spe: Base Speed.
        moves: Move list.

    Returns:
        A PokemonSpecies instance.
    """
    if types is None:
        types = [Type.NORMAL]
    if moves is None:
        moves = [_make_move(f"{name}Move")]
    return PokemonSpecies(
        name=name,
        base_stats=(hp, atk, df, spa, spd, spe),
        types=types,
        moves=moves,
    )


def _make_pokemon(species: PokemonSpecies) -> Pokemon:
    """Build a Pokemon from a species with default EVs/IVs.

    Args:
        species: The underlying PokemonSpecies.

    Returns:
        A Pokemon instance with valid stats and moves.
    """
    n_moves = len(species.moves)
    move_idx = list(range(min(4, n_moves)))
    return Pokemon(
        species=species,
        move_indexes=move_idx,
        level=100,
        ivs=(31,) * 6,
        evs=(84, 84, 84, 84, 84, 90),
    )


def _make_team(n: int = 4) -> Team:
    """Build a team of N Pokemon with diverse types.

    Args:
        n: Number of team members.

    Returns:
        A Team with N members.
    """
    type_pool = [
        (Type.FIRE, 110, 90, 70, 80),
        (Type.WATER, 100, 70, 90, 70),
        (Type.GRASS, 90, 80, 80, 60),
        (Type.ELECTRIC, 80, 60, 70, 110),
        (Type.FIGHT, 100, 120, 70, 90),
        (Type.PSYCHIC, 90, 60, 80, 100),
    ]
    members = []
    for i in range(n):
        t, atk, df, spd, spe = type_pool[i % len(type_pool)]
        species = _make_species(
            name=f"Mon{i}",
            types=[t],
            hp=100,
            atk=atk,
            df=df,
            spa=atk - 10,
            spd=spd,
            spe=spe,
            moves=[
                _make_move(f"Move{i}A", pkm_type=t, base_power=90),
                _make_move(f"Move{i}B", pkm_type=Type.NORMAL, base_power=70),
            ],
        )
        members.append(_make_pokemon(species))
    return Team(members)


class _MockModel:
    """Mock classifier returning a fixed probability."""

    def __init__(self, prob: float = 0.6) -> None:
        self._prob = prob

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fixed probability for all inputs.

        Args:
            X: Feature array of shape (n, 56).

        Returns:
            Array of shape (n, 2) with [1-prob, prob].
        """
        n = X.shape[0]
        return np.array([[1.0 - self._prob, self._prob]] * n)


def _make_opp_views(n: int = 4) -> list[MagicMock]:
    """Build mock opponent PokemonView objects.

    Args:
        n: Number of opponent views.

    Returns:
        List of MagicMock objects with species attributes.
    """
    views = []
    type_pool = [Type.DARK, Type.GHOST, Type.STEEL, Type.DRAGON, Type.FAIRY, Type.ICE]
    for i in range(n):
        t = type_pool[i % len(type_pool)]
        species = _make_species(
            name=f"Opp{i}",
            types=[t],
            hp=90,
            atk=85,
            df=75,
            spa=85,
            spd=75,
            spe=95,
            moves=[_make_move(f"OppMove{i}", pkm_type=t, base_power=85)],
        )
        view = MagicMock()
        view.species = species
        views.append(view)
    return views


def _make_predicted_builds(opp_views: list[MagicMock]) -> dict[MagicMock, list[Pokemon]]:
    """Build predicted Pokemon for each opponent view.

    Args:
        opp_views: List of opponent view mocks.

    Returns:
        Dict mapping each view to a list with one predicted Pokemon build.
    """
    builds: dict[MagicMock, list[Pokemon]] = {}
    for view in opp_views:
        pkm = _make_pokemon(view.species)
        builds[view] = [pkm]
    return builds


class TestFeatureOrder:
    """Tests for feature order alignment."""

    def test_feature_order_file_exists(self) -> None:
        """feature_order.json exists in src/models/."""
        assert _FEATURE_ORDER_PATH.exists()

    def test_feature_order_has_56_entries(self) -> None:
        """Feature order contains exactly 56 names."""
        order = _load_feature_order()
        assert len(order) == 56

    def test_feature_order_matches_compute(self) -> None:
        """Feature order matches keys from compute_pairwise_features."""
        team_a = _make_team(4)
        team_b = _make_team(4)
        features = compute_pairwise_features(
            list(team_a.members), list(team_b.members)
        )
        order = _load_feature_order()
        assert set(order) == set(features.keys())

    def test_features_to_array_length(self) -> None:
        """_features_to_array produces a 56-element array."""
        team_a = _make_team(4)
        team_b = _make_team(4)
        features = compute_pairwise_features(
            list(team_a.members), list(team_b.members)
        )
        arr = _features_to_array(features)
        assert arr.shape == (56,)
        assert arr.dtype == np.float32


class TestScorePairMp:
    """Tests for score_pair_mp function."""

    def test_output_range(self) -> None:
        """Score is in [0, 1]."""
        team = _make_team(4)
        opp_views = _make_opp_views(4)
        builds = _make_predicted_builds(opp_views)
        model = _MockModel(prob=0.7)

        score = score_pair_mp(
            pair_indices=(0, 1),
            roster_indices=(0, 1, 2, 3),
            my_full_team=team,
            predicted_builds=builds,
            opp_views=opp_views,
            model=model,
        )
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        """Same inputs produce the same score."""
        team = _make_team(4)
        opp_views = _make_opp_views(4)
        builds = _make_predicted_builds(opp_views)
        model = _MockModel(prob=0.55)

        score1 = score_pair_mp((0, 1), (0, 1, 2, 3), team, builds, opp_views, model)
        score2 = score_pair_mp((0, 1), (0, 1, 2, 3), team, builds, opp_views, model)
        assert score1 == score2

    def test_mock_prob_returned(self) -> None:
        """With a fixed-prob mock model, score equals that probability."""
        team = _make_team(4)
        opp_views = _make_opp_views(4)
        builds = _make_predicted_builds(opp_views)
        model = _MockModel(prob=0.42)

        score = score_pair_mp(
            (0, 1), (0, 1, 2, 3), team, builds, opp_views, model
        )
        assert abs(score - 0.42) < 1e-6


class TestScoreRosterMp:
    """Tests for score_roster_mp function."""

    def test_output_range(self) -> None:
        """Roster score is in [0, 1]."""
        team = _make_team(4)
        opp_views = _make_opp_views(4)
        builds = _make_predicted_builds(opp_views)
        model = _MockModel(prob=0.65)

        score = score_roster_mp(
            roster_indices=(0, 1, 2, 3),
            my_full_team=team,
            predicted_builds=builds,
            opp_views=opp_views,
            model=model,
        )
        assert 0.0 <= score <= 1.0

    def test_averages_over_pairs(self) -> None:
        """Roster score equals mock prob (uniform model → same average)."""
        team = _make_team(4)
        opp_views = _make_opp_views(4)
        builds = _make_predicted_builds(opp_views)
        model = _MockModel(prob=0.5)

        score = score_roster_mp(
            (0, 1, 2, 3), team, builds, opp_views, model
        )
        assert abs(score - 0.5) < 1e-6


class TestSelectionPolicyModes:
    """Tests for DongimonSelectionPolicy mode integration."""

    def test_invalid_mode_raises(self) -> None:
        """ValueError on unrecognized selection_mode."""
        from src.selection.policy import DongimonSelectionPolicy

        with pytest.raises(ValueError, match="selection_mode must be one of"):
            DongimonSelectionPolicy(selection_mode="invalid_mode")

    @patch("src.selection.policy.load_mp_model")
    def test_mp_only_returns_valid_indices(self, mock_load: MagicMock) -> None:
        """mp_only mode returns correct length, no duplicates, all in range."""
        mock_load.return_value = _MockModel(prob=0.6)

        from src.selection.policy import DongimonSelectionPolicy

        policy = DongimonSelectionPolicy(selection_mode="mp_only")

        team = _make_team(4)
        opp_views = _make_opp_views(4)
        opp_team = MagicMock()
        opp_team.members = opp_views

        with patch("src.selection.policy.predict_opponent_builds") as mock_pred:
            mock_pred.side_effect = lambda pokemon_view, **kw: [
                _make_pokemon(pokemon_view.species)
            ]
            result = policy.decision((team, opp_team), 4)

        assert len(result) == 4
        assert len(set(result)) == 4
        assert all(0 <= i < 4 for i in result)

    @patch("src.selection.policy.load_mp_model")
    def test_full_pipeline_team_size_6(self, mock_load: MagicMock) -> None:
        """mp_only mode with 6-member team returns 4 valid indices."""
        mock_load.return_value = _MockModel(prob=0.55)

        from src.selection.policy import DongimonSelectionPolicy

        policy = DongimonSelectionPolicy(
            selection_mode="mp_only", n_top_candidates=3
        )

        team = _make_team(6)
        opp_views = _make_opp_views(4)
        opp_team = MagicMock()
        opp_team.members = opp_views

        with patch("src.selection.policy.predict_opponent_builds") as mock_pred:
            mock_pred.side_effect = lambda pokemon_view, **kw: [
                _make_pokemon(pokemon_view.species)
            ]
            result = policy.decision((team, opp_team), 4)

        assert len(result) == 4
        assert len(set(result)) == 4
        assert all(0 <= i < 6 for i in result)

    def test_mp_only_mode_default(self) -> None:
        """Default mode is mp_only (MP model loaded)."""
        from src.selection.policy import DongimonSelectionPolicy

        policy = DongimonSelectionPolicy()
        assert policy._mode == "mp_only"
        assert policy._mp_model is not None
