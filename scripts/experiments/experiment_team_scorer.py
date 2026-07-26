"""Experiment: Team Quality Scorer viability (ELO-based).

Generates stratified teams (top 25% / middle 50% / bottom 25% by BST),
runs all-vs-all round-robin with Greedy on both sides, labels via ELO,
extracts rich features, and tests if LinearRegression or MLPRegressor
can predict team quality from features.

Usage:
    uv run python scripts/experiments/experiment_team_scorer.py --n-teams=200 --n-battles=5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_move_set, gen_pkm_roster, gen_team

from src.shared.archetypes import create_generic_build_for_species
from src.shared.types import type_effectiveness, vgc2_type_to_name
from src.teambuild.builds import create_single_optimal_build, species_power, species_role
from src.teambuild.evolution import run_evolution
from src.teambuild.operators import seed_coverage_teams
from src.tuning.elo_rating import update_elo

N_TYPES = 18
TYPE_NAMES = [
    "normal", "fire", "water", "electric", "grass",
    "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon",
    "dark", "steel", "fairy",
]


def _extract_team_features(team: Any) -> dict[str, float]:
    """Extract predictive features from a team of 6 Pokemon.

    Covers BST stats, speed tiers, type coverage, defensive synergy,
    move quality, role composition, and type diversity.

    Args:
        team: A vgc2 Team object.

    Returns:
        Dict of feature name -> float value.
    """
    members = team.members

    def _bs(m: Any, idx: int) -> int:
        return m.species.base_stats[idx] if hasattr(m, 'species') else 0

    bst_list = [sum(m.species.base_stats) for m in members]
    hp_list = [_bs(m, Stat.MAX_HP) for m in members]
    atk_list = [_bs(m, Stat.ATTACK) for m in members]
    def_list = [_bs(m, Stat.DEFENSE) for m in members]
    spa_list = [_bs(m, Stat.SPECIAL_ATTACK) for m in members]
    spd_list = [_bs(m, Stat.SPECIAL_DEFENSE) for m in members]
    spe_list = [_bs(m, Stat.SPEED) for m in members]
    n = len(members)

    speed_brackets = {"bracket_0_50": 0, "bracket_51_80": 0, "bracket_81_110": 0, "bracket_111_plus": 0}
    for s in spe_list:
        if s <= 50:
            speed_brackets["bracket_0_50"] += 1
        elif s <= 80:
            speed_brackets["bracket_51_80"] += 1
        elif s <= 110:
            speed_brackets["bracket_81_110"] += 1
        else:
            speed_brackets["bracket_111_plus"] += 1

    feat: dict[str, float] = {}
    feat["n_members"] = n

    for label, lst in [("bst", bst_list), ("hp", hp_list), ("atk", atk_list),
                        ("def", def_list), ("spa", spa_list), ("spd", spd_list),
                        ("spe", spe_list)]:
        vals = lst if lst else [0]
        feat[f"{label}_avg"] = float(np.mean(vals))
        feat[f"{label}_max"] = float(np.max(vals))
        feat[f"{label}_min"] = float(np.min(vals))
        feat[f"{label}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    for bracket, count in speed_brackets.items():
        feat[bracket] = float(count)

    covered_types = set()
    for species in members:
        for move in species.moves:
            if not hasattr(move, 'base_power') or move.base_power <= 0:
                continue
            atk_type = move.pkm_type
            atk_name = vgc2_type_to_name(atk_type.value if hasattr(atk_type, 'value') else atk_type)
            for def_name in TYPE_NAMES:
                eff = type_effectiveness(atk_name, [def_name])
                if eff > 1.0:
                    covered_types.add(def_name)
    feat["type_coverage_se_count"] = float(len(covered_types))
    feat["type_coverage_ratio"] = float(len(covered_types)) / N_TYPES if n > 0 else 0.0

    all_types_seen = set()
    for species in members:
        for t in species.species.types:
            t_name = vgc2_type_to_name(t.value if hasattr(t, 'value') else t)
            all_types_seen.add(t_name)
    feat["type_diversity"] = float(len(all_types_seen))

    total_weaknesses = 0
    covered_weaknesses = 0
    member_type_lists = []
    for species in members:
        spec_obj = species.species if hasattr(species, 'species') else species
        tl = [vgc2_type_to_name(t.value) for t in spec_obj.types]
        member_type_lists.append(tl)
    for i, _species in enumerate(members):
        for atk_name in TYPE_NAMES:
            eff = type_effectiveness(atk_name, member_type_lists[i])
            if eff > 1.0:
                total_weaknesses += 1
                for j, ally_types in enumerate(member_type_lists):
                    if i == j:
                        continue
                    ally_eff = type_effectiveness(atk_name, ally_types)
                    if ally_eff < 1.0:
                        covered_weaknesses += 1
                        break
    feat["defensive_synergy"] = float(covered_weaknesses / max(total_weaknesses, 1))

    move_bp_sum = 0
    move_count = 0
    for species in members:
        for move in species.moves:
            if hasattr(move, 'base_power') and move.base_power > 0:
                move_bp_sum += move.base_power
                move_count += 1
    feat["avg_move_bp"] = float(move_bp_sum / max(move_count, 1)) if move_count > 0 else 0.0

    role_counts: dict[str, int] = {"sweeper": 0, "wall": 0, "mixed": 0}
    for member in members:
        spec_obj = member.species if hasattr(member, 'species') else member
        role = species_role(spec_obj)
        role_counts[role] = role_counts.get(role, 0) + 1
    feat["role_sweeper"] = float(role_counts.get("sweeper", 0))
    feat["role_wall"] = float(role_counts.get("wall", 0))
    feat["role_mixed"] = float(role_counts.get("mixed", 0))

    return feat


def _species_indices_to_team(
    indices: list[int],
    pool_species: list[Any],
) -> Team | None:
    """Convert a list of species pool indices into a vgc2 Team.

    Uses create_single_optimal_build with create_generic_build_for_species
    fallback for each member.

    Args:
        indices: List of indices into pool_species.
        pool_species: List of PokemonSpecies objects.

    Returns:
        A Team object, or None if any build fails.
    """
    members: list[Any] = []
    for idx in indices:
        sp = pool_species[idx]
        build = create_single_optimal_build(sp)
        if build is None:
            build = create_generic_build_for_species(sp)
            if build is None:
                return None
        members.append(build)
    return Team(members)


def _generate_stratified_teams(
    target_count: int,
    seed: int,
) -> list[Any]:
    """Generate teams via triad approach: random, GA-evolved, JJJ-coverage.

    Produces a mixed dataset:
        33% uniform random (baseline noise / negative labels),
        33% GA-evolved via run_evolution (mid-tier labels),
        33% JJJ-coverage-selected via seed_coverage_teams (high-tier labels).

    Args:
        target_count: Desired total number of teams.
        seed: Base RNG seed.

    Returns:
        List of vgc2 Team objects.
    """
    rng = np.random.default_rng(seed)
    n_random = target_count // 3
    n_ga = target_count // 3
    n_coverage = target_count - n_random - n_ga

    random_teams = [gen_team(6, 4, rng) for _i in range(n_random)]

    pool_size = max(100, n_ga * 6 + n_coverage * 6)
    move_set = gen_move_set(200)
    pool_species = gen_pkm_roster(pool_size, move_set)

    viability: dict[Any, float] = {}
    for sp in pool_species:
        viability[sp] = species_power(sp)

    ga_teams: list[Any] = []
    ga_rng = np.random.default_rng(seed + 100)
    try:
        evo_results = run_evolution(
            pool_species=pool_species,
            viability_scores=viability,
            team_size=6,
            pop_size=50,
            generations=10,
            mutation_rate=0.10,
            elite_fraction=0.10,
            rng=ga_rng,
        )
        for team_indices in evo_results[:n_ga]:
            team = _species_indices_to_team(list(team_indices), pool_species)
            if team is not None:
                ga_teams.append(team)
        while len(ga_teams) < n_ga:
            ga_teams.append(gen_team(6, 4, rng))
    except Exception:
        ga_teams = [gen_team(6, 4, rng) for _i in range(n_ga)]

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
            team = _species_indices_to_team(list(team_indices), pool_species)
            if team is not None:
                coverage_teams.append(team)
        while len(coverage_teams) < n_coverage:
            coverage_teams.append(gen_team(6, 4, rng))
    except Exception:
        coverage_teams = [gen_team(6, 4, rng) for _i in range(n_coverage)]

    result = random_teams + ga_teams + coverage_teams
    rng.shuffle(result)
    return result


def _import_bp(module_path: str, class_name: str) -> Any:
    """Import a competitor and return its battle policy."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls().battlepolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Team Quality Scorer viability experiment (ELO-based).")
    parser.add_argument("--n-teams", type=int, default=200, help="Total teams to generate")
    parser.add_argument("--n-battles", type=int, default=5, help="Battles per round-robin pairing")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--elo-k", type=float, default=32.0, help="ELO K-factor")
    parser.add_argument("--initial-elo", type=float, default=1500.0, help="Initial ELO rating")
    parser.add_argument("--policy", type=str, default="greedy", choices=["greedy", "jjj", "minimon"],
                        help="Battle policy for labeling (default: greedy, fastest)")
    args = parser.parse_args()

    policy_map: dict[str, Any] = {
        "greedy": GreedyBattlePolicy(),
        "jjj": _import_bp("competitors.competitor1_jjj", "JJJ_Competitor"),
        "minimon": _import_bp("competitors.competitor2_minimon", "minimon"),
    }
    bp = policy_map.get(args.policy, GreedyBattlePolicy())
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    print("=" * 60)
    print("Team Quality Scorer — ELO-based Experiment (stratified BST)")
    print(f"  seed={args.seed}, total_teams={args.n_teams} (33% random + 33% GA-evolved + 33% JJJ-coverage)")
    print(f"  battle_policy={args.policy} (fastest: greedy=124.8/s, jjj=90.2/s, minimon=71.0/s)")
    print(f"  battles_per_pairing={args.n_battles}")
    total_pairs = args.n_teams * (args.n_teams - 1) // 2
    total_battles = total_pairs * args.n_battles
    print(f"  total pairings={total_pairs}, total battles={total_battles}")
    print("=" * 60)

    start = time.perf_counter()

    teams = _generate_stratified_teams(args.n_teams, args.seed)
    n_teams = len(teams)

    features_list: list[dict[str, float]] = [_extract_team_features(t) for t in teams]

    elos: dict[int, float] = dict.fromkeys(range(n_teams), args.initial_elo)
    computed = 0

    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            pairing_seed = args.seed + i * 1000 + j
            wins_i = 0

            for b_idx in range(args.n_battles):
                battle_seed = pairing_seed + b_idx
                gen = np.random.default_rng(battle_seed)

                idx_a = sel.decision((teams[i], TeamView(teams[j])), 4)
                idx_b = sel.decision((teams[j], TeamView(teams[i])), 4)

                sub_a, sub_view_a = subteam(teams[i], TeamView(teams[i]), idx_a)
                sub_b, sub_view_b = subteam(teams[j], TeamView(teams[j]), idx_b)

                battle_teams = get_battle_teams((sub_a, sub_b), 2)
                state = State(battle_teams)
                rng_tuple = ((gen, gen), (gen, gen))
                engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

                while not engine.finished():
                    sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
                    sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
                    cmd0 = bp.decision(sv0, sub_view_b)
                    cmd1 = bp.decision(sv1, sub_view_a)
                    engine.run_turn((cmd0, cmd1))

                if engine.winning_side == 0:
                    wins_i += 1

            i_won = wins_i > args.n_battles // 2
            elos[i], elos[j] = update_elo(elos[i], elos[j], i_won, args.elo_k)
            computed += 1

        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - start
            pct = (i + 1) / n_teams * 100
            print(f"  Round robin: {i + 1}/{n_teams} teams ({pct:.0f}%), {computed} pairings, {elapsed:.1f}s")

    elapsed = time.perf_counter() - start
    elo_labels = [elos[i] for i in range(n_teams)]
    print(f"\nData generation complete: {n_teams} teams, {computed} pairings in {elapsed:.1f}s")
    print(f"  ELO range: {min(elo_labels):.1f} - {max(elo_labels):.1f}, mean: {np.mean(elo_labels):.1f}")

    feature_names = list(features_list[0].keys())
    x_data = np.array([[f[n] for n in feature_names] for f in features_list], dtype=np.float64)
    y_data = np.array(elo_labels, dtype=np.float64)

    x_tr, x_te, y_tr, y_te = train_test_split(x_data, y_data, test_size=0.2, random_state=args.seed)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    print("\n--- Feature Selection (Random Forest, 250 trees) ---")
    rf = RandomForestRegressor(n_estimators=250, random_state=args.seed, n_jobs=-1)
    rf.fit(x_tr_s, y_tr)
    importances = rf.feature_importances_
    threshold = 0.02
    important_mask = importances >= threshold
    important_indices = [i for i, ok in enumerate(important_mask) if ok]

    importance_snapshot = {
        "experiment": "team_scorer",
        "n_features_total": len(feature_names),
        "n_features_kept": len(important_indices),
        "threshold": threshold,
        "features": [
            {"name": feature_names[i], "importance": float(importances[i]), "kept": bool(important_mask[i])}
            for i in range(len(feature_names))
        ],
    }
    snapshot_path = Path("data/gini_team_scorer.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(importance_snapshot, f, indent=2)
    print(f"  Saved feature importances to {snapshot_path}")

    dropped = [feature_names[i] for i in range(len(feature_names)) if not important_mask[i]]
    kept_names = [feature_names[i] for i in important_indices]
    print(f"  Kept {len(important_indices)}/{len(feature_names)} features")
    print(f"  Dropped: {dropped}")
    for name in kept_names:
        idx = feature_names.index(name)
        print(f"    {name}: {importances[idx]:.4f}")

    x_tr_clean = x_tr_s[:, important_indices]
    x_te_clean = x_te_s[:, important_indices]

    print("\n--- Training Models ---")
    print(f"  Features: {len(kept_names)}, Train: {len(x_tr)}, Test: {len(x_te)}")

    linear = LinearRegression()
    linear.fit(x_tr_clean, y_tr)
    y_pred_lin = linear.predict(x_te_clean)
    r2_lin = r2_score(y_te, y_pred_lin)
    print(f"\n  LinearRegression R² on holdout: {r2_lin:.4f}")

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(x_tr_clean, y_tr)
    y_pred_mlp = mlp.predict(x_te_clean)
    r2_mlp = r2_score(y_te, y_pred_mlp)
    print(f"\n  MLPRegressor (64->32) R² on holdout: {r2_mlp:.4f}")

    best_r2 = max(r2_lin, r2_mlp)
    threshold = 0.30
    print("\n--- Verdict ---")
    print(f"  Best R²: {best_r2:.4f} (threshold: {threshold})")
    print(f"  Viable: {'YES' if best_r2 > threshold else 'NO'}")

    if best_r2 > threshold:
        print("  Proceed to full-scale data generation (5-10 hours).")
    else:
        print("  Suggestion: Increase n_teams, add features, or increase ELO resolution.")


if __name__ == "__main__":
    main()
