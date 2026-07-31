"""Matchup Predictor scoring for the Selection Policy.

Provides MP-based roster and pair scoring using the trained XGBoost
champion model. Replaces the damage-ratio matrix pre-filter with
learned win-probability inference over the same 56-feature schema
used during training.

Scoring mirrors the sub-tournament's exhaustive pair-vs-pair structure:
for each candidate active pair, the model evaluates all opponent pair
combinations and averages P(win) across them. This is orders of
magnitude faster than battle simulation while capturing the same
matchup dynamics the model learned from 24K labeled pairings.
"""

import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.team import Team

from src.data.features import compute_pairwise_features

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "xgboost_model.joblib"
_FEATURE_ORDER_PATH = _MODEL_DIR / "feature_order.json"


@lru_cache(maxsize=1)
def _load_feature_order() -> tuple[str, ...]:
    """Load the canonical feature order from training metadata.

    Returns:
        Tuple of 56 feature names in the order the model expects.

    Raises:
        FileNotFoundError: If feature_order.json is missing.
    """
    if not _FEATURE_ORDER_PATH.exists():
        raise FileNotFoundError(
            f"Feature order file not found: {_FEATURE_ORDER_PATH}. "
            "Generate it from data/MP/mp_meta_*.json → feature_names."
        )
    with open(_FEATURE_ORDER_PATH) as f:
        names: list[str] = json.load(f)
    return tuple(names)


@lru_cache(maxsize=1)
def load_mp_model(model_path: Path | None = None) -> Any:
    """Load the champion XGBoost model for MP inference.

    Cached via lru_cache so repeated calls reuse the same instance.

    Args:
        model_path: Path to the joblib model file. Uses the default
            champion model if None.

    Returns:
        Fitted classifier with predict_proba method.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = model_path or _DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"MP model not found: {path}. "
            "Train via: uv run python -m src.data.train"
        )
    return joblib.load(path)


def _features_to_array(features: dict[str, float]) -> npt.NDArray[np.float32]:
    """Convert a feature dict to an ordered array matching training schema.

    Args:
        features: Dict of feature_name -> value from compute_pairwise_features.

    Returns:
        Float32 array of shape (56,) in canonical training order.

    Raises:
        KeyError: If a required feature is missing from the dict.
    """
    order = _load_feature_order()
    return np.array([features[name] for name in order], dtype=np.float32)


def _predict_win_probability(model: Any, features: dict[str, float]) -> float:
    """Run MP model inference on a single feature vector.

    Args:
        model: Fitted classifier with predict_proba.
        features: Pairwise feature dict (56 features).

    Returns:
        P(side A wins) in [0, 1].
    """
    arr = _features_to_array(features).reshape(1, -1)
    proba = model.predict_proba(arr)
    return float(proba[0, 1])


def _pick_opp_reserve_builds(
    predicted_builds: dict[Any, list[Any]],
    active_pair_views: tuple[Any, ...],
) -> list[Any]:
    """Select up to 2 reserve Pokemon for the opponent from non-active views.

    Picks the best predicted build from each opponent view not in the
    active pair, ranked by bulk (sum of defensive stats).

    Args:
        predicted_builds: Dict mapping PokemonView to list of predicted builds.
        active_pair_views: The two active opponent views (excluded from reserve).

    Returns:
        List of up to 2 Pokemon for the opponent reserve.
    """
    active_set = {id(v) for v in active_pair_views}
    candidates: list[Any] = []
    for view, builds in predicted_builds.items():
        if id(view) in active_set or not builds:
            continue
        best = builds[0]
        if hasattr(best, "moves") and best.moves:
            candidates.append(best)

    candidates.sort(
        key=lambda p: (
            p.stats[Stat.MAX_HP] + p.stats[Stat.DEFENSE] + p.stats[Stat.SPECIAL_DEFENSE]
        ),
        reverse=True,
    )
    return candidates[:2]


def score_pair_mp(
    pair_indices: tuple[int, ...],
    roster_indices: tuple[int, ...],
    my_full_team: Team,
    predicted_builds: dict[Any, list[Any]],
    opp_views: list[Any],
    model: Any,
) -> float:
    """Score a single active pair using MP P(win) averaged over opponent pairs.

    Forms our 4-member subteam (active pair + reserves from roster) and
    evaluates against every possible opponent pair from predicted builds.

    Args:
        pair_indices: Indices of the two active Pokemon in my_full_team.
        roster_indices: Indices of the full 4-member roster in my_full_team.
        my_full_team: Our full team object.
        predicted_builds: Dict mapping opponent PokemonView to predicted builds.
        opp_views: List of all opponent PokemonView objects.
        model: Fitted MP classifier.

    Returns:
        Average P(win) across all opponent pair matchups, in [0, 1].

    Raises:
        RuntimeError: If no opponent pairs can be formed.
    """
    reserve_indices = [i for i in roster_indices if i not in pair_indices]
    my_subteam = [my_full_team.members[i] for i in pair_indices]
    my_subteam += [my_full_team.members[i] for i in reserve_indices]

    opp_pairs = list(itertools.combinations(opp_views, 2))
    if not opp_pairs:
        raise RuntimeError(
            "score_pair_mp: no opponent pairs from "
            f"{len(opp_views)} views"
        )

    total_prob = 0.0
    n_matchups = 0

    for opp_pair in opp_pairs:
        opp_active = [predicted_builds[v][0] for v in opp_pair if predicted_builds.get(v)]
        if len(opp_active) < 2:
            continue

        opp_reserve = _pick_opp_reserve_builds(predicted_builds, opp_pair)
        opp_subteam = opp_active + opp_reserve

        if len(opp_subteam) < 4:
            opp_subteam = opp_active + opp_reserve
            while len(opp_subteam) < 4 and opp_active:
                opp_subteam.append(opp_active[0])

        features = compute_pairwise_features(my_subteam, opp_subteam[:4])
        prob = _predict_win_probability(model, features)
        total_prob += prob
        n_matchups += 1

    if n_matchups == 0:
        raise RuntimeError(
            "score_pair_mp: zero valid opponent matchups formed"
        )

    return total_prob / n_matchups


def score_roster_mp(
    roster_indices: tuple[int, ...],
    my_full_team: Team,
    predicted_builds: dict[Any, list[Any]],
    opp_views: list[Any],
    model: Any,
    n_active: int = 2,
) -> float:
    """Score a roster by averaging MP P(win) over all pair-vs-pair matchups.

    For each C(4,2) active pair within the roster, computes the MP score
    against all opponent pairs. Returns the average across all pairs,
    representing the roster's overall matchup quality.

    Args:
        roster_indices: Tuple of 4 indices into my_full_team forming the roster.
        my_full_team: Our full team object.
        predicted_builds: Dict mapping opponent PokemonView to predicted builds.
        opp_views: List of all opponent PokemonView objects.
        model: Fitted MP classifier.
        n_active: Number of active Pokemon per side (default 2).

    Returns:
        Average P(win) across all pair-vs-pair matchups, in [0, 1].

    Raises:
        RuntimeError: If no pairs can be formed from the roster.
    """
    my_pairs = list(itertools.combinations(roster_indices, n_active))
    if not my_pairs:
        raise RuntimeError(
            f"score_roster_mp: no pairs of size {n_active} from "
            f"roster of size {len(roster_indices)}"
        )

    total_prob = 0.0
    for pair in my_pairs:
        pair_score = score_pair_mp(
            pair_indices=pair,
            roster_indices=roster_indices,
            my_full_team=my_full_team,
            predicted_builds=predicted_builds,
            opp_views=opp_views,
            model=model,
        )
        total_prob += pair_score

    return total_prob / len(my_pairs)
