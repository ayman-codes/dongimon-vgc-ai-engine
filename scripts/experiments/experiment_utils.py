"""Shared utilities for experiment scripts.

Provides stratified team generation, data profiling, battle running,
feature computation, statistical testing, and model evaluation helpers.
Imported by all experiment scripts to eliminate code duplication.
"""

import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_move_set, gen_pkm_roster, gen_team

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.shared.archetypes import create_generic_build_for_species
from src.shared.types import type_effectiveness, vgc2_type_to_name
from src.teambuild.builds import create_single_optimal_build, species_power, species_role
from src.teambuild.evolution import run_evolution
from src.teambuild.operators import seed_coverage_teams

N_TYPES = 18
TYPE_NAMES = [
    "normal", "fire", "water", "electric", "grass",
    "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon",
    "dark", "steel", "fairy",
]

_POOL_CACHE: tuple[list[Any], dict[Any, float]] | None = None


def _get_pool() -> tuple[list[Any], dict[Any, float]]:
    """Lazily generate and cache the species pool with viability scores.

    Returns:
        Tuple of (pool_species, viability_scores) where pool_species is
        a list of PokemonSpecies and viability_scores maps species -> float.
    """
    global _POOL_CACHE
    if _POOL_CACHE is not None:
        return _POOL_CACHE

    move_set = gen_move_set(400)
    roster = gen_pkm_roster(200, move_set)
    pool: list[Any] = [sp for sp in list(roster) if species_power(sp) > 0 and len(sp.moves) > 0]
    viability: dict[Any, float] = {sp: species_power(sp) for sp in pool}
    _POOL_CACHE = (pool, viability)
    return _POOL_CACHE


def _try_build_species(species: Any) -> Any | None:
    """Try to create a Pokemon build from a species.

    Attempts create_single_optimal_build first, then falls back to
    create_generic_build_for_species. Catches and logs any exceptions
    during build to prevent silent failures.

    Args:
        species: A PokemonSpecies object.

    Returns:
        A Pokemon object, or None if both build attempts fail.
    """
    try:
        build = create_single_optimal_build(species)
        if build is None:
            build = create_generic_build_for_species(species)
        return build
    except Exception:
        sp_name = getattr(species, "name", str(species))
        print(f"[WARNING] Build threw exception for species '{sp_name}':")
        traceback.print_exc()
        return None


def _build_team_from_indices(indices: list[int], pool_species: list[Any]) -> Team | None:
    """Convert species pool indices into a vgc2 Team.

    Uses create_single_optimal_build with create_generic_build_for_species
    fallback for each member. If a species fails both builds, scans the
    pool for an unused alternative before giving up.

    Args:
        indices: List of indices into pool_species.
        pool_species: List of PokemonSpecies objects.

    Returns:
        A Team object, or None if no buildable species found for a slot.
    """
    members: list[Any] = []
    used: set[int] = set(indices)

    for idx in indices:
        build = _try_build_species(pool_species[idx])
        if build is None:
            for alt_idx, alt_sp in enumerate(pool_species):
                if alt_idx not in used:
                    alt_build = _try_build_species(alt_sp)
                    if alt_build is not None:
                        build = alt_build
                        used.add(alt_idx)
                        break
        if build is None:
            return None
        members.append(build)

    try:
        return Team(members)
    except Exception:
        traceback.print_exc()
        return None


def generate_stratified_teams(
    n_teams: int,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.33, 0.34, 0.33),
) -> tuple[list[Any], list[str]]:
    """Generate teams via triad: random, GA-evolved, JJJ-coverage.

    Logs every fallback to random team generation. Raises RuntimeError
    if the total fallback rate exceeds 10%.

    Args:
        n_teams: Total number of teams to generate.
        seed: Base RNG seed.
        fractions: Tuple of (random_fraction, ga_fraction, coverage_fraction).

    Returns:
        Tuple of (teams, tier_labels) where tier_labels[i] is one of
        {"random", "ga", "coverage"}.

    Raises:
        RuntimeError: If fallback rate exceeds 10%.
    """
    rng = np.random.default_rng(seed)
    pool_species, viability = _get_pool()

    n_random = int(n_teams * fractions[0])
    n_ga = int(n_teams * fractions[1])
    n_coverage = n_teams - n_random - n_ga

    fallbacks = 0
    teams: list[Any] = []
    labels: list[str] = []

    random_teams: list[Any] = []
    for _ in range(n_random):
        random_teams.append(gen_team(6, 4, rng))
    teams.extend(random_teams)
    labels.extend(["random"] * len(random_teams))

    ga_teams: list[Any] = []
    ga_runs = max(1, (n_ga + 4) // 5)
    try:
        for run_i in range(ga_runs):
            ga_run_rng = np.random.default_rng(seed + 100 + run_i * 10)
            evo_results = run_evolution(
                pool_species=pool_species,
                viability_scores=viability,
                team_size=6,
                pop_size=50,
                generations=10,
                mutation_rate=0.10,
                elite_fraction=0.10,
                rng=ga_run_rng,
            )
            for team_indices in evo_results[: n_ga - len(ga_teams)]:
                team = _build_team_from_indices(list(team_indices), pool_species)
                if team is not None:
                    ga_teams.append(team)
                else:
                    fallbacks += 1
                    print(f"[WARNING] Failed to build GA team (indices {team_indices[:3]}...) — fallbacks={fallbacks}")
            if len(ga_teams) >= n_ga:
                break
        while len(ga_teams) < n_ga:
            ga_teams.append(gen_team(6, 4, rng))
    except Exception:
        print("[ERROR] GA evolution crashed:")
        traceback.print_exc()
        ga_teams = [gen_team(6, 4, rng) for _ in range(n_ga)]
        fallbacks += n_ga
    teams.extend(ga_teams)
    labels.extend(["ga"] * len(ga_teams))

    coverage_teams: list[Any] = []
    cov_rng = np.random.default_rng(seed + 200)
    try:
        cov_results = seed_coverage_teams(
            pool_species=pool_species,
            viability_scores=viability,
            team_size=6,
            n_seeds=n_coverage,
            rng=cov_rng,
        )
        for team_indices in cov_results[:n_coverage]:
            team = _build_team_from_indices(list(team_indices), pool_species)
            if team is not None:
                coverage_teams.append(team)
            else:
                fallbacks += 1
                print(
                    f"[WARNING] Failed to build coverage team (indices {team_indices[:3]}...) — fallbacks={fallbacks}"
                )
        while len(coverage_teams) < n_coverage:
            coverage_teams.append(gen_team(6, 4, rng))
    except Exception:
        print("[ERROR] Coverage team generation crashed:")
        traceback.print_exc()
        coverage_teams = [gen_team(6, 4, rng) for _ in range(n_coverage)]
        fallbacks += n_coverage
    teams.extend(coverage_teams)
    labels.extend(["coverage"] * len(coverage_teams))

    fallback_rate = fallbacks / max(n_teams, 1)
    if fallback_rate > 0.10:
        raise RuntimeError(
            f"Fallback rate {fallback_rate:.2%} exceeds 10% threshold (n_fallbacks={fallbacks}, n_teams={n_teams})"
        )

    combined = list(zip(teams, labels, strict=False))
    rng = np.random.default_rng(seed + 300)
    rng.shuffle(combined)
    teams[:], labels[:] = zip(*combined, strict=True) if combined else ((), ())

    return list(teams), list(labels)


def _team_mean_bst(team: Any) -> float:
    """Compute the mean base stat total of a team's members.

    Args:
        team: A vgc2 Team object with members having .species.base_stats.

    Returns:
        Mean BST across all 6 members.
    """
    members = team.members
    bsts = [sum(m.species.base_stats) for m in members]
    return float(np.mean(bsts)) if bsts else 0.0


def validate_stratification(
    teams: list[Any],
    tier_labels: list[str],
) -> dict[str, Any]:
    """Verify tiers are statistically separable via Kruskal-Wallis H-test.

    Args:
        teams: List of Team objects.
        tier_labels: Tier label per team ("random", "ga", "coverage").

    Returns:
        Dict with mean_bst_per_tier, std_bst_per_tier, kruskal_wallis_H,
        kruskal_wallis_p, fallback_count, fallback_rate.
    """
    tier_bsts: dict[str, list[float]] = {}
    for team, label in zip(teams, tier_labels, strict=False):
        if label not in tier_bsts:
            tier_bsts[label] = []
        tier_bsts[label].append(_team_mean_bst(team))

    mean_per_tier = {t: float(np.mean(b)) for t, b in tier_bsts.items()}
    std_per_tier = {t: float(np.std(b)) for t, b in tier_bsts.items()}

    tier_values = list(tier_bsts.values())
    if len(tier_values) >= 2 and all(len(v) > 0 for v in tier_values):
        h_stat, p_val = stats.kruskal(*tier_values)
    else:
        h_stat, p_val = 0.0, 1.0

    return {
        "mean_bst_per_tier": mean_per_tier,
        "std_bst_per_tier": std_per_tier,
        "kruskal_wallis_H": float(h_stat),
        "kruskal_wallis_p": float(p_val),
    }


def profile_teams(teams: list[Any]) -> dict[str, Any]:
    """Compute output distribution statistics for generated teams.

    Args:
        teams: List of Team objects.

    Returns:
        Dict with bst_mean, bst_std, bst_min, bst_max, n_unique_species,
        type_distribution (18-dim), role_sweeper, role_wall, role_mixed.
    """
    bsts = [_team_mean_bst(t) for t in teams]

    type_counts = [0] * N_TYPES
    role_counts: dict[str, int] = {"sweeper": 0, "wall": 0, "mixed": 0}
    species_seen: set[str] = set()

    for team in teams:
        for member in team.members:
            spec_obj = member.species if hasattr(member, "species") else member
            role = species_role(spec_obj)
            role_counts[role] = role_counts.get(role, 0) + 1

            for t_enum in spec_obj.types:
                t_name = vgc2_type_to_name(int(t_enum))
                if t_name in TYPE_NAMES:
                    type_counts[TYPE_NAMES.index(t_name)] += 1

            species_seen.add(spec_obj.name if hasattr(spec_obj, "name") else str(id(spec_obj)))

    return {
        "bst_mean": float(np.mean(bsts)) if bsts else 0.0,
        "bst_std": float(np.std(bsts)) if len(bsts) > 1 else 0.0,
        "bst_min": float(np.min(bsts)) if bsts else 0.0,
        "bst_max": float(np.max(bsts)) if bsts else 0.0,
        "n_unique_species": len(species_seen),
        "type_distribution": dict(zip(TYPE_NAMES, type_counts, strict=False)),
        "role_sweeper": role_counts.get("sweeper", 0),
        "role_wall": role_counts.get("wall", 0),
        "role_mixed": role_counts.get("mixed", 0),
    }


def run_pair_battles(
    team_a: Any,
    team_b: Any,
    bp_side_a: Any,
    bp_side_b: Any,
    n_battles: int,
    pair_seed: int,
    params: BattleRuleParam | None = None,
    sel: BasicSelectionPolicy | None = None,
) -> tuple[int, int, list[int], list[int]]:
    """Run n_battles between team_a (side A) and team_b (side B).

    If bp_side_b is a list, a policy is selected per battle via
    deterministic RNG seeded by pair_seed + 5000 + b_idx.

    Args:
        team_a: vgc2 Team for side A.
        team_b: vgc2 Team for side B.
        bp_side_a: BattlePolicy for side A.
        bp_side_b: BattlePolicy or list[BattlePolicy] for side B.
        n_battles: Number of battles to run.
        pair_seed: Seed for deterministic RNG.
        params: BattleRuleParam (uses default if None).
        sel: SelectionPolicy (uses BasicSelectionPolicy if None).

    Returns:
        Tuple of (wins_a, wins_b, selected_indices_a, selected_indices_b).
        selected_indices_* are the 4-member subteam indices from the
        FIRST battle (representative, since BasicSelectionPolicy is deterministic).
    """
    _sel = sel if sel is not None else BasicSelectionPolicy()
    _params = params if params is not None else BattleRuleParam()
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)

    wins_a = 0
    wins_b = 0
    first_idx_a = list(_sel.decision((team_a, view_b), 4))
    first_idx_b = list(_sel.decision((team_b, view_a), 4))

    for b_idx in range(n_battles):
        gen_rng = np.random.default_rng(pair_seed + 2000 + b_idx)

        sub_a, sub_view_a = subteam(team_a, view_a, first_idx_a)
        sub_b, sub_view_b = subteam(team_b, view_b, first_idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), 2)
        state = State(battle_teams)
        rng_tuple = ((gen_rng, gen_rng), (gen_rng, gen_rng))
        engine = BattleEngine(
            state,
            params=_params,
            acc_rng=rng_tuple,
            eff_rng=rng_tuple,
            sta_rng=rng_tuple,
        )

        bp_b = bp_side_b
        if isinstance(bp_side_b, list):
            choice_rng = np.random.default_rng(pair_seed + 5000 + b_idx)
            bp_b = bp_side_b[choice_rng.integers(0, len(bp_side_b))]

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            cmd0 = bp_side_a.decision(sv0, sub_view_b)
            cmd1 = bp_b.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
        elif engine.winning_side == 1:
            wins_b += 1

    return wins_a, wins_b, first_idx_a, first_idx_b


def compute_subteam_features(members: list[Any]) -> dict[str, float]:
    """Compute 53 features on the 4 Pokemon that fought.

    Features (in insertion order):
        7 stats × 4 aggregations — avg, max, min, std (28):
            bst_avg, bst_max, bst_min, bst_std,
            hp_avg, hp_max, hp_min, hp_std,
            atk_avg, ..., spe_std
        Speed brackets (4): spd_b0_50, spd_b51_80, spd_b81_110, spd_b111_p
        Type coverage SE vector (18): type_cov_{name}_se per type
        Weakness count (1): weakness_count
        Move BP (2): avg_move_bp, max_move_bp

    BST-only ablation indices: [0, 1, 2, 3] corresponding to
    bst_avg, bst_max, bst_min, bst_std.

    Args:
        members: List of 4 Pokemon objects.

    Returns:
        Dict of feature_name -> float value.
    """
    feat: dict[str, float] = {}

    bst_list: list[int] = []
    hp_list: list[int] = []
    atk_list: list[int] = []
    def_list: list[int] = []
    spa_list: list[int] = []
    spd_list: list[int] = []
    spe_list: list[int] = []

    for pkm in members:
        if hasattr(pkm, "species"):
            base = pkm.species.base_stats
        elif hasattr(pkm, "constants"):
            base = pkm.constants.base
        else:
            base = (0,) * 8

        bst = int(sum(base))
        bst_list.append(bst)
        hp_list.append(int(base[0]))
        atk_list.append(int(base[1]))
        def_list.append(int(base[2]))
        spa_list.append(int(base[3]))
        spd_list.append(int(base[4]))
        spe_list.append(int(base[5]))

    for label, lst in [
        ("bst", bst_list), ("hp", hp_list), ("atk", atk_list),
        ("def", def_list), ("spa", spa_list), ("spd", spd_list),
        ("spe", spe_list),
    ]:
        vals = lst if lst else [0]
        feat[f"{label}_avg"] = float(np.mean(vals))
        feat[f"{label}_max"] = float(np.max(vals))
        feat[f"{label}_min"] = float(np.min(vals))
        feat[f"{label}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    speed_brackets = {"spd_b0_50": 0, "spd_b51_80": 0, "spd_b81_110": 0, "spd_b111_p": 0}
    for s in spe_list:
        if s <= 50:
            speed_brackets["spd_b0_50"] += 1
        elif s <= 80:
            speed_brackets["spd_b51_80"] += 1
        elif s <= 110:
            speed_brackets["spd_b81_110"] += 1
        else:
            speed_brackets["spd_b111_p"] += 1
    for bracket, count in speed_brackets.items():
        feat[bracket] = float(count)

    type_se_coverage = [0] * N_TYPES
    for pkm in members:
        covered_by_pkm: set[int] = set()
        moves = pkm.moves if hasattr(pkm, "moves") else []
        for move in moves:
            if not hasattr(move, "base_power") or move.base_power <= 0:
                continue
            atk_type = int(move.pkm_type) if hasattr(move.pkm_type, "value") else int(move.pkm_type)
            if atk_type < 0 or atk_type >= N_TYPES:
                continue
            atk_name = vgc2_type_to_name(atk_type)
            for def_type in range(N_TYPES):
                def_name = vgc2_type_to_name(def_type)
                eff = type_effectiveness(atk_name, [def_name])
                if eff > 1.0:
                    covered_by_pkm.add(def_type)
        for def_type in covered_by_pkm:
            type_se_coverage[def_type] += 1

    for t_idx, count in enumerate(type_se_coverage):
        feat[f"type_cov_{TYPE_NAMES[t_idx]}_se"] = float(count)

    total_weaknesses = 0.0
    member_type_lists: list[list[str]] = []
    for pkm in members:
        spec_obj = pkm.species if hasattr(pkm, "species") else pkm
        tl = [vgc2_type_to_name(int(t)) for t in spec_obj.types]
        member_type_lists.append(tl)

    for tl in member_type_lists:
        for atk_name in TYPE_NAMES:
            eff = type_effectiveness(atk_name, tl)
            if eff > 1.0:
                total_weaknesses += 1.0
    feat["weakness_count"] = total_weaknesses

    move_bp_sum = 0
    move_count = 0
    max_bp = 0
    for pkm in members:
        moves = pkm.moves if hasattr(pkm, "moves") else []
        for move in moves:
            if hasattr(move, "base_power") and move.base_power > 0:
                bp = move.base_power
                move_bp_sum += bp
                move_count += 1
                if bp > max_bp:
                    max_bp = bp
    feat["avg_move_bp"] = float(move_bp_sum / max(move_count, 1)) if move_count > 0 else 0.0
    feat["max_move_bp"] = float(max_bp)

    return feat


def _count_se_moves_against(subteam_a: list[Any], subteam_b: list[Any]) -> float:
    """Count super-effective offensive types A has against B's defensive types.

    Higher = A has better type matchup against B.

    Args:
        subteam_a: List of 4 Pokemon (attackers).
        subteam_b: List of 4 Pokemon (defenders).

    Returns:
        Number of SE hits A has against B's collected types.
    """
    defender_types: list[list[str]] = []
    for pkm in subteam_b:
        spec_obj = pkm.species if hasattr(pkm, "species") else pkm
        tl = [vgc2_type_to_name(int(t)) for t in spec_obj.types]
        defender_types.append(tl)

    count = 0.0
    seen: set[tuple[int, int]] = set()
    for pkm in subteam_a:
        moves = pkm.moves if hasattr(pkm, "moves") else []
        for move in moves:
            if not hasattr(move, "base_power") or move.base_power <= 0:
                continue
            atk_type = int(move.pkm_type) if hasattr(move.pkm_type, "value") else int(move.pkm_type)
            if atk_type < 0 or atk_type >= N_TYPES:
                continue
            atk_name = vgc2_type_to_name(atk_type)
            for d_idx, dtl in enumerate(defender_types):
                if (atk_type, d_idx) in seen:
                    continue
                eff = type_effectiveness(atk_name, dtl)
                if eff > 1.0:
                    count += 1.0
                    seen.add((atk_type, d_idx))
    return count


def compute_pairwise_features(
    subteam_a: list[Any],
    subteam_b: list[Any],
) -> dict[str, float]:
    """Compute pairwise delta features between two subteams.

    Computes per-team features and delta (A - B) for each, plus
    type advantage metrics.

    Args:
        subteam_a: List of 4 Pokemon (side A).
        subteam_b: List of 4 Pokemon (side B).

    Returns:
        Dict of pairwise feature_name -> float value.
    """
    feat_a = compute_subteam_features(subteam_a)
    feat_b = compute_subteam_features(subteam_b)

    deltas: dict[str, float] = {}
    for key in feat_a:
        val_a = feat_a.get(key, 0.0)
        val_b = feat_b.get(key, 0.0)
        deltas[f"{key}_diff"] = val_a - val_b

    type_adv_a = _count_se_moves_against(subteam_a, subteam_b)
    type_adv_b = _count_se_moves_against(subteam_b, subteam_a)
    deltas["type_advantage_net"] = type_adv_a - type_adv_b
    deltas["type_adv_a"] = type_adv_a
    deltas["type_adv_b"] = type_adv_b

    return deltas


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metric_fn: Function taking (y_true, y_pred) and returning a scalar.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence interval width (e.g. 0.95 for 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_pred)
    n = len(y_true)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        est = metric_fn(y_true[idx], y_pred[idx])
        estimates.append(est)

    alpha = 1.0 - ci
    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return point, lower, upper


def attenuation_corrected_rho(
    x: np.ndarray,
    y: np.ndarray,
    n_battles: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute raw and attenuation-corrected Spearman rho.

    The attenuation correction accounts for measurement noise in the
    win-rate labels, which is binomial with variance p*(1-p)/n_battles.

    Args:
        x: First set of win rates (e.g. Greedy).
        y: Second set of win rates (e.g. Dongimon).
        n_battles: Number of battles per pairing (used for noise estimate).
        n_bootstrap: Number of bootstrap resamples for CIs.
        seed: RNG seed.

    Returns:
        Dict with rho_raw, rho_corrected, reliability, sigma2_noise,
        sigma2_total, p_value, rho_raw_ci, rho_corrected_ci.
    """
    result = stats.spearmanr(x, y)
    rho_raw: float = float(result.statistic)
    p_value: float = float(result.pvalue)

    sigma2_total = float(np.var(x) + np.var(y)) / 2.0
    sigma2_noise = 0.25 / n_battles

    if sigma2_total > sigma2_noise and sigma2_total > 1e-10:
        reliability = 1.0 - (sigma2_noise / sigma2_total)
        reliability = max(0.1, min(1.0, reliability))
        rho_corrected = rho_raw / reliability
    else:
        reliability = 1.0
        rho_corrected = rho_raw

    def _spearmanr(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
        r = stats.spearmanr(x_arr, y_arr)
        return float(r.statistic)

    _, raw_lo, raw_hi = bootstrap_ci(x, y, _spearmanr, n_bootstrap=n_bootstrap, seed=seed)
    rho_raw_ci: tuple[float, float] = (raw_lo, raw_hi)

    if abs(rho_raw) >= 0.05 and reliability > 0.1:
        rho_corrected_ci = (rho_corrected * raw_lo / max(abs(rho_raw), 1e-10),
                            rho_corrected * raw_hi / max(abs(rho_raw), 1e-10))
    else:
        rho_corrected_ci = (rho_corrected, rho_corrected)

    return {
        "rho_raw": rho_raw,
        "rho_corrected": float(rho_corrected),
        "reliability": float(reliability),
        "sigma2_noise": float(sigma2_noise),
        "sigma2_total": float(sigma2_total),
        "p_value": p_value,
        "rho_raw_ci": list(rho_raw_ci),
        "rho_corrected_ci": list(rho_corrected_ci),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from N tests.
        alpha: Family-wise significance level.

    Returns:
        List of reject (True) / do-not-reject (False) decisions.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda pair: pair[0])

    decisions: list[tuple[bool, int]] = []
    for rank, (p, original_idx) in enumerate(indexed):
        threshold = alpha / (n - rank)
        decisions.append((p < threshold, original_idx))

    decisions.sort(key=lambda pair: pair[1])
    return [d[0] for d in decisions]


def run_ablation_tracks(
    x_full: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    bst_feature_indices: list[int],
    model_factory: Callable[[], BaseEstimator],
    task: str = "regression",
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run 3-track ablation: BST-only, features-only (no-BST), full.

    For each track, performs stratified k-fold CV, bootstrap 95% CI,
    and a paired bootstrap test of Track C vs Track A.
    Applies Holm-Bonferroni correction across the 3 tracks.

    Args:
        x_full: Feature matrix (n_samples, n_features).
        y: Target values.
        feature_names: Names of all features in x_full columns.
        bst_feature_indices: Column indices of BST features.
        model_factory: Zero-argument callable returning a sklearn estimator.
        task: "regression" or "classification".
        n_folds: Number of CV folds.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        Dict with keys "bst_only", "features_only", "full", each containing
        scores, ci, p_value_vs_bst, cv_scores, and feature_indices_used.
    """
    n_features = x_full.shape[1]

    bst_indices = sorted({i for i in bst_feature_indices if 0 <= i < n_features})
    non_bst_indices = sorted(set(range(n_features)) - set(bst_indices))

    tracks: dict[str, np.ndarray] = {
        "bst_only": x_full[:, bst_indices],
        "features_only": x_full[:, non_bst_indices],
        "full": x_full,
    }

    feature_indices_map = {
        "bst_only": bst_indices,
        "features_only": non_bst_indices,
        "full": list(range(n_features)),
    }

    def _metric(t: np.ndarray, p: np.ndarray) -> float:
        if task == "classification":
            return float(roc_auc_score(t, p))
        return float(r2_score(t, p))

    results: dict[str, dict[str, Any]] = {}
    cv_splitter: Any = (
        StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        if task == "classification"
        else KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    )

    for track_name, x_t in tracks.items():
        if x_t.shape[1] == 0:
            results[track_name] = {
                "scores": {"mean": 0.0, "std": 0.0},
                "ci": [0.0, 0.0, 0.0],
                "cv_scores": [],
                "feature_indices_used": [],
            }
            continue

        cv_scores: list[float] = []
        all_y_test: list[float] = []
        all_y_pred: list[float] = []

        for _fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(x_t, y)):
            x_train_cv = x_t[train_idx]
            x_test_cv = x_t[test_idx]
            y_train_cv = y[train_idx]
            y_test_cv = y[test_idx]

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_cv)
            x_test_scaled = scaler.transform(x_test_cv)

            model = clone(model_factory())
            model.fit(x_train_scaled, y_train_cv)
            y_pred_cv = model.predict(x_test_scaled)

            score = _metric(y_test_cv, y_pred_cv)
            cv_scores.append(score)
            all_y_test.extend(y_test_cv.tolist())
            all_y_pred.extend(y_pred_cv.tolist())

        y_test_arr = np.array(all_y_test)
        y_pred_arr = np.array(all_y_pred)
        point, ci_lo, ci_hi = bootstrap_ci(
            y_test_arr, y_pred_arr, _metric, n_bootstrap=n_bootstrap, seed=seed
        )

        results[track_name] = {
            "scores": {
                "mean": float(np.mean(cv_scores)),
                "std": float(np.std(cv_scores)),
            },
            "ci": [point, ci_lo, ci_hi],
            "cv_scores": cv_scores,
            "feature_indices_used": feature_indices_map.get(track_name, []),
            "feature_names_used": [feature_names[i] for i in feature_indices_map.get(track_name, [])],
        }

    track_seed_offset: dict[str, int] = {"bst_only": 1, "features_only": 2, "full": 3}

    p_values: list[float] = []
    for track_name in ["bst_only", "features_only"]:
        if track_name not in results or "cv_scores" not in results[track_name]:
            p_values.append(1.0)
            results.setdefault(track_name, {})["p_value_vs_bst"] = 1.0
            continue

        full_scores = results["full"]["cv_scores"]
        track_scores = results[track_name]["cv_scores"]

        diffs: list[float] = []
        rng_bs = np.random.default_rng(seed + track_seed_offset[track_name])
        for _ in range(n_bootstrap):
            idx = rng_bs.integers(0, len(full_scores), size=len(full_scores))
            full_boot = np.mean([full_scores[i] for i in idx])
            track_boot = np.mean([track_scores[i] for i in idx])
            diffs.append(full_boot - track_boot)

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        p_val: float = 1.0
        if std_diff > 1e-10:
            z = abs(mean_diff) / (std_diff / np.sqrt(n_bootstrap))
            p_val = float(2 * (1 - stats.norm.cdf(z)))
        elif abs(mean_diff) > 1e-10:
            p_val = 0.00001

        p_values.append(p_val)
        results[track_name]["p_value_vs_bst"] = p_val
        results[track_name]["paired_bootstrap_mean_diff"] = float(mean_diff)

    reject = holm_bonferroni(p_values)
    for i, track_name in enumerate(["bst_only", "features_only"]):
        if track_name in results:
            results[track_name]["holm_significant"] = reject[i] if i < len(reject) else False

    return results


def fit_bradley_terry(
    pair_outcomes: list[tuple[int, int, int, int]],
    n_teams: int,
    n_iter: int = 200,
    lr: float = 0.01,
) -> np.ndarray:
    """Iterative MLE for Bradley-Terry model.

    Args:
        pair_outcomes: List of (i, j, wins_i, wins_j) tuples.
        n_teams: Total number of teams.
        n_iter: Number of MLE iterations.
        lr: Learning rate for gradient ascent.

    Returns:
        theta: Array of shape (n_teams,) with Bradley-Terry strengths,
        normalized to zero mean.
    """
    theta = np.zeros(n_teams)
    for _ in range(n_iter):
        for i, j, w_i, w_j in pair_outcomes:
            n = w_i + w_j
            if n == 0:
                continue
            exp_i = np.exp(theta[i])
            exp_j = np.exp(theta[j])
            denom = exp_i + exp_j
            if denom < 1e-15:
                continue
            p_i = exp_i / denom
            theta[i] += lr * (w_i / n - p_i)
            theta[j] += lr * (w_j / n - (1.0 - p_i))
        theta = theta - theta.mean()
    return theta


def swiss_pairings_round(
    scores: np.ndarray,
    history: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate one round of Swiss-system pairings given current standings.

    Sorts by score descending (ties broken by random shuffle), pairs
    adjacent teams avoiding rematches in history, and handles byes
    for odd team counts (unpaired team gets no match in this round).

    Args:
        scores: Array of current match scores per team (wins, draws, etc.).
        history: Set of (min, max) tuples of pairs already matched.
        rng: NumPy Generator for reproducibility.

    Returns:
        List of (team_i, team_j) pairs for this round. Caller runs
        battles, updates scores, adds pairs to history, then calls again.
    """
    n_teams = len(scores)
    order = list(range(n_teams))
    rng.shuffle(order)
    order.sort(key=lambda t: (-scores[t], rng.random()))

    paired: set[int] = set()
    round_pairs: list[tuple[int, int]] = []

    for i in range(len(order)):
        a = order[i]
        if a in paired:
            continue
        for j in range(i + 1, len(order)):
            b = order[j]
            if b in paired:
                continue
            pair_key = (min(a, b), max(a, b))
            if pair_key not in history:
                round_pairs.append((a, b))
                history.add(pair_key)
                paired.add(a)
                paired.add(b)
                break

    return round_pairs
