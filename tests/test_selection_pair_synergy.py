"""Tests for the analytical pair-synergy scorer and the selection fast path.

Covers the four teamwork terms in ``src.selection.pair_synergy`` and the
pure-ordering fast path in ``DongimonSelectionPolicy`` that activates when
the team is already final size (no subset to choose, only which pair starts
active). Fixtures use real ``PokemonSpecies`` / ``Move`` objects so the
type-effectiveness and fitness-operator code paths run for real.
"""

from unittest.mock import MagicMock

import pytest
from vgc2.battle_engine.modifiers import Category, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies
from vgc2.battle_engine.team import Team

from src.config.loader import load_selection_synergy
from src.config.models import SelectionSynergyWeights
from src.selection.pair_synergy import (
    defensive_synergy,
    offensive_coverage_vs,
    pair_synergy_terms,
    role_balance,
    score_pair_synergy,
    speed_control,
)
from src.selection.policy import DongimonSelectionPolicy


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
        base_power: Base power (0 for status moves).
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
        types: List of vgc2 Types (defaults to Normal).
        hp: Base HP.
        atk: Base Attack.
        df: Base Defense.
        spa: Base Special Attack.
        spd: Base Special Defense.
        spe: Base Speed.
        moves: Move list (defaults to one Normal physical move).

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


def _make_member(species: PokemonSpecies) -> MagicMock:
    """Wrap a species in a member-like mock exposing ``.species``.

    Args:
        species: The underlying PokemonSpecies.

    Returns:
        A MagicMock with a ``species`` attribute.
    """
    member = MagicMock()
    member.species = species
    return member


def _make_team(species_list: list[PokemonSpecies]) -> MagicMock:
    """Build a team-like mock whose members wrap the given species.

    Args:
        species_list: Species for each team member, in order.

    Returns:
        A MagicMock with a ``members`` list of member mocks.
    """
    team = MagicMock(spec=Team)
    team.members = [_make_member(s) for s in species_list]
    return team


def _fire_species(name: str, spe: int = 100, moves: list[Move] | None = None) -> PokemonSpecies:
    """A Fire-type with a strong STAB Fire move.

    Args:
        name: Species name.
        spe: Base Speed.
        moves: Optional move list override (defaults to a strong Fire move).

    Returns:
        Fire-type PokemonSpecies.
    """
    if moves is None:
        moves = [_make_move(f"{name}Flamethrower", pkm_type=Type.FIRE, base_power=90, category=Category.SPECIAL)]
    return _make_species(
        name,
        types=[Type.FIRE],
        atk=120,
        spe=spe,
        moves=moves,
    )


def _grass_species(name: str, spe: int = 60) -> PokemonSpecies:
    """A Grass-type opponent with a STAB Grass move.

    Args:
        name: Species name.
        spe: Base Speed.

    Returns:
        Grass-type PokemonSpecies.
    """
    return _make_species(
        name,
        types=[Type.GRASS],
        spe=spe,
        moves=[_make_move(f"{name}Leaf", pkm_type=Type.GRASS, base_power=80, category=Category.SPECIAL)],
    )


class TestPairSynergyTerms:
    """Tests for pair_synergy_terms and its component functions."""

    def test_returns_four_keys(self) -> None:
        """The terms dict exposes coverage, defense, speed, and role."""
        pair = [_fire_species("A"), _fire_species("B")]
        opps = [_grass_species("G1"), _grass_species("G2")]
        terms = pair_synergy_terms(pair, opps)
        assert set(terms) == {"coverage", "defense", "speed", "role"}

    def test_terms_are_finite_floats(self) -> None:
        """Every term is a finite float."""
        pair = [_fire_species("A"), _fire_species("B")]
        opps = [_grass_species("G1"), _grass_species("G2")]
        terms = pair_synergy_terms(pair, opps)
        for value in terms.values():
            assert isinstance(value, float)
            assert value == value  # not NaN

    def test_coverage_zero_without_opponents(self) -> None:
        """Coverage is zero when there are no opponents to threaten."""
        pair = [_fire_species("A"), _fire_species("B")]
        assert offensive_coverage_vs(pair, []) == 0.0

    def test_coverage_super_effective_beats_neutral(self) -> None:
        """A super-effective pair outscores a neutral pair vs the same field."""
        grass_field = [_grass_species("G1"), _grass_species("G2")]
        fire_pair = [_fire_species("F1"), _fire_species("F2")]
        normal_pair = [_make_species("N1"), _make_species("N2")]
        assert offensive_coverage_vs(fire_pair, grass_field) > offensive_coverage_vs(
            normal_pair, grass_field
        )

    def test_speed_control_rewards_outspeeding(self) -> None:
        """A fast pair scores higher speed control than a slow pair."""
        grass_field = [_grass_species("G1", spe=60), _grass_species("G2", spe=60)]
        fast_pair = [_fire_species("F1", spe=120), _fire_species("F2", spe=110)]
        slow_pair = [_fire_species("S1", spe=30), _fire_species("S2", spe=40)]
        assert speed_control(fast_pair, grass_field) > speed_control(slow_pair, grass_field)

    def test_speed_control_in_unit_interval(self) -> None:
        """Speed control is bounded within [0, 1]."""
        grass_field = [_grass_species("G1"), _grass_species("G2")]
        pair = [_fire_species("F1", spe=120), _fire_species("F2", spe=30)]
        score = speed_control(pair, grass_field)
        assert 0.0 <= score <= 1.0

    def test_defensive_synergy_in_unit_interval(self) -> None:
        """Defensive synergy is a fraction in [0, 1]."""
        pair = [_make_species("W", types=[Type.WATER]), _make_species("G", types=[Type.GRASS])]
        score = defensive_synergy(pair)
        assert 0.0 <= score <= 1.0

    def test_role_balance_in_unit_interval(self) -> None:
        """Role balance is bounded within [0, 1]."""
        pair = [_fire_species("F1"), _make_species("N1")]
        score = role_balance(pair)
        assert 0.0 <= score <= 1.0

    def test_score_pair_synergy_is_mean_of_terms(self) -> None:
        """The aggregate score equals the equal-weighted mean of the terms."""
        pair = [_fire_species("A"), _fire_species("B")]
        opps = [_grass_species("G1"), _grass_species("G2")]
        terms = pair_synergy_terms(pair, opps)
        expected = 0.25 * (terms["coverage"] + terms["defense"] + terms["speed"] + terms["role"])
        assert score_pair_synergy(pair, opps) == pytest.approx(expected)


class TestSelectionSynergyConfig:
    """Tests for the SelectionSynergyWeights model and its loader."""

    def test_model_defaults(self) -> None:
        """Default blend weights are the agreed 0.6 / 0.4 split."""
        weights = SelectionSynergyWeights()
        assert weights.avg_weight == pytest.approx(0.6)
        assert weights.worst_weight == pytest.approx(0.4)

    def test_synergy_dict_has_five_terms(self) -> None:
        """synergy_dict exposes exactly the five tunable term weights."""
        weights = SelectionSynergyWeights()
        assert set(weights.synergy_dict()) == {
            "w_matchup",
            "w_defense",
            "w_speed",
            "w_role",
            "w_coverage",
        }

    def test_loader_reads_yaml(self) -> None:
        """The loader returns a validated model from the shipped YAML."""
        weights = load_selection_synergy()
        assert isinstance(weights, SelectionSynergyWeights)
        assert weights.avg_weight == pytest.approx(0.6)
        assert weights.worst_weight == pytest.approx(0.4)


class TestFastPathOrdering:
    """Tests for the pure-ordering fast path in DongimonSelectionPolicy."""

    def _grass_field_team(self) -> MagicMock:
        """An opponent view of four Grass-types.

        Returns:
            Team mock with four Grass-type members.
        """
        return _make_team([_grass_species(f"G{i}") for i in range(4)])

    def test_returns_valid_permutation(self) -> None:
        """Fast path returns every index exactly once, capped at max_size."""
        my_team = _make_team(
            [_fire_species("F1"), _fire_species("F2"), _make_species("N1"), _make_species("N2")]
        )
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy()
        ordered = policy.decision((my_team, opp_team), 4)
        assert sorted(ordered) == [0, 1, 2, 3]

    def test_active_pair_size_two(self) -> None:
        """The first two returned indices form the active pair."""
        my_team = _make_team(
            [_fire_species("F1"), _fire_species("F2"), _make_species("N1"), _make_species("N2")]
        )
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy()
        ordered = policy.decision((my_team, opp_team), 4)
        assert len(ordered) == 4
        assert len(set(ordered[:2])) == 2

    def test_prefers_strong_super_effective_pair(self) -> None:
        """The pair with strong super-effective moves starts active.

        All four members are Fire-types with identical stats so the
        defensive, speed, and role terms are constant across pairs; only
        the matchup and coverage terms differ. The two carrying strong
        STAB Fire moves must therefore be chosen as the active pair.
        """
        weak = [_make_move("WeakFire", pkm_type=Type.FIRE, base_power=30, category=Category.SPECIAL)]
        my_team = _make_team(
            [
                _fire_species("F1"),
                _fire_species("F2"),
                _fire_species("F3", moves=weak),
                _fire_species("F4", moves=weak),
            ]
        )
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy()
        ordered = policy.decision((my_team, opp_team), 4)
        assert set(ordered[:2]) == {0, 1}

    def test_reserves_are_remaining_members(self) -> None:
        """Reserve indices are exactly the non-active members."""
        my_team = _make_team(
            [_fire_species("F1"), _fire_species("F2"), _make_species("N1"), _make_species("N2")]
        )
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy()
        ordered = policy.decision((my_team, opp_team), 4)
        assert set(ordered[2:]) == set(range(4)) - set(ordered[:2])

    def test_deterministic(self) -> None:
        """The same inputs produce the same ordering across calls."""
        my_team = _make_team(
            [_fire_species("F1"), _fire_species("F2"), _make_species("N1"), _make_species("N2")]
        )
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy()
        first = policy.decision((my_team, opp_team), 4)
        second = policy.decision((my_team, opp_team), 4)
        assert first == second

    def test_empty_own_team_raises(self) -> None:
        """An empty own team raises RuntimeError."""
        my_team = _make_team([])
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy()
        with pytest.raises(RuntimeError):
            policy.decision((my_team, opp_team), 4)

    def test_empty_opponent_raises(self) -> None:
        """An empty opponent view raises RuntimeError."""
        my_team = _make_team([_fire_species("F1"), _fire_species("F2")])
        opp_team = _make_team([])
        policy = DongimonSelectionPolicy()
        with pytest.raises(RuntimeError):
            policy.decision((my_team, opp_team), 4)

    def test_custom_weights_are_used(self) -> None:
        """Injecting matchup-only weights still picks the strong-move pair."""
        weights = SelectionSynergyWeights(
            w_matchup=1.0, w_defense=0.0, w_speed=0.0, w_role=0.0, w_coverage=0.0
        )
        weak = [_make_move("WeakFire", pkm_type=Type.FIRE, base_power=30, category=Category.SPECIAL)]
        my_team = _make_team(
            [
                _fire_species("F1"),
                _fire_species("F2"),
                _fire_species("F3", moves=weak),
                _fire_species("F4", moves=weak),
            ]
        )
        opp_team = self._grass_field_team()
        policy = DongimonSelectionPolicy(synergy_weights=weights)
        ordered = policy.decision((my_team, opp_team), 4)
        assert set(ordered[:2]) == {0, 1}
