"""Experiment: Matchup Predictor viability.

Generates pairs of different random teams, runs battles with a chosen
side-A policy (Greedy or Dongimon) vs opponent policies (Greedy, JJJ),
extracts rich pairwise features, and tests if LogisticRegression or
MLPClassifier can predict outcomes.

When side-A is Greedy, also runs a correlation test on holdout pairs
comparing Greedy outcomes to Dongimon outcomes, measuring whether
Greedy-labeled data generalizes to Dongimon.

Usage:
    uv run python scripts/experiments/experiment_matchup_predictor.py --n-pairs=2000 --n-battles=10
    uv run python scripts/experiments/experiment_matchup_predictor.py --side-a-policy=greedy --n-pairs=5000
"""


import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights
from src.shared.types import type_effectiveness, vgc2_type_to_name

TYPE_NAMES = [
    "normal", "fire", "water", "electric", "grass",
    "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon",
    "dark", "steel", "fairy",
]


def _team_stat_vectors(team: Any) -> dict[str, float]:
    """Compute per-team aggregate stat features.

    Args:
        team: vgc2 Team object.

    Returns:
        Dict of stat features.
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

    def _bracket_counts(speeds: list[int]) -> dict[str, int]:
        return {
            "b0_50": sum(1 for s in speeds if s <= 50),
            "b51_80": sum(1 for s in speeds if 51 <= s <= 80),
            "b81_110": sum(1 for s in speeds if 81 <= s <= 110),
            "b111_plus": sum(1 for s in speeds if s >= 111),
        }

    buckets_a = _bracket_counts(spe_list)
    se_count = 0
    for member in members:
        spec_obj = member.species if hasattr(member, 'species') else member
        spec_types = [vgc2_type_to_name(t.value) for t in spec_obj.types]
        for atk_name in TYPE_NAMES:
            eff = type_effectiveness(atk_name, spec_types)
            if eff > 1.0:
                se_count += 1

    move_bp = [move.base_power for m in members for move in m.moves
               if hasattr(move, 'base_power') and move.base_power > 0]

    return {
        "bst_avg": float(np.mean(bst_list)) if bst_list else 0,
        "bst_max": float(np.max(bst_list)) if bst_list else 0,
        "hp_avg": float(np.mean(hp_list)) if hp_list else 0,
        "atk_avg": float(np.mean(atk_list)) if atk_list else 0,
        "def_avg": float(np.mean(def_list)) if def_list else 0,
        "spa_avg": float(np.mean(spa_list)) if spa_list else 0,
        "spd_avg": float(np.mean(spd_list)) if spd_list else 0,
        "spe_avg": float(np.mean(spe_list)) if spe_list else 0,
        "spd_b0_50": float(buckets_a["b0_50"]),
        "spd_b51_80": float(buckets_a["b51_80"]),
        "spd_b81_110": float(buckets_a["b81_110"]),
        "spd_b111_p": float(buckets_a["b111_plus"]),
        "weakness_count": float(se_count),
        "avg_move_bp": float(np.mean(move_bp)) if move_bp else 0,
        "max_move_bp": float(np.max(move_bp)) if move_bp else 0,
    }


def _extract_pairwise_features(team_a: Any, team_b: Any) -> dict[str, float]:
    """Extract pairwise features for team A vs team B.

    All features are differences or net advantages (A - B).

    Args:
        team_a: First team (side 0).
        team_b: Second team (side 1).

    Returns:
        Dict of pairwise features.
    """
    sa = _team_stat_vectors(team_a)
    sb = _team_stat_vectors(team_b)

    feat: dict[str, float] = {}
    for key in sa:
        feat[f"{key}_diff"] = sa[key] - sb.get(key, 0)

    members_a = team_a.members
    members_b = team_b.members

    type_advantage_a = 0.0
    for mx in members_a:
        for move in mx.moves:
            if not hasattr(move, 'base_power') or move.base_power <= 0:
                continue
            atk_type = move.pkm_type
            atk_name = vgc2_type_to_name(atk_type.value if hasattr(atk_type, 'value') else atk_type)
            for my in members_b:
                spec_types_b = [vgc2_type_to_name(t.value) for t in my.species.types]
                eff = type_effectiveness(atk_name, spec_types_b)
                if eff > 1.0:
                    type_advantage_a += 1.0

    type_advantage_b = 0.0
    for my in members_b:
        for move in my.moves:
            if not hasattr(move, 'base_power') or move.base_power <= 0:
                continue
            atk_type = move.pkm_type
            atk_name = vgc2_type_to_name(atk_type.value if hasattr(atk_type, 'value') else atk_type)
            for mx in members_a:
                spec_types_a = [vgc2_type_to_name(t.value) for t in mx.species.types]
                eff = type_effectiveness(atk_name, spec_types_a)
                if eff > 1.0:
                    type_advantage_b += 1.0

    feat["type_advantage_net"] = type_advantage_a - type_advantage_b
    feat["type_adv_a"] = type_advantage_a
    feat["type_adv_b"] = type_advantage_b

    for atk_name in TYPE_NAMES:
        a_se = 0
        b_se = 0
        for mx in members_a:
            spec_types_a = [vgc2_type_to_name(t.value) for t in mx.species.types]
            eff = type_effectiveness(atk_name, spec_types_a)
            if eff > 1.0:
                a_se += 1
        for my in members_b:
            spec_types_b = [vgc2_type_to_name(t.value) for t in my.species.types]
            eff = type_effectiveness(atk_name, spec_types_b)
            if eff > 1.0:
                b_se += 1
        feat[f"type_{atk_name}_diff"] = float(a_se - b_se)

    return feat


def _import_bp(module_path: str, class_name: str) -> Any:
    """Import a competitor and return its battle policy."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls().battlepolicy


def _dongimon_bp() -> Any:
    """Return Dongimon's battle policy with current weights."""
    return DongimonCompetitor(custom_weights=load_battle_weights().model_dump()).battlepolicy


def _run_battles_with_side_a(
    bp_side_a: Any,
    opp_policies: list[Any],
    pairs_teams_a: list[Any],
    pairs_teams_b: list[Any],
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    base_seed: int,
    n_battles: int,
) -> list[int]:
    """Run battles for a list of (team_a, team_b) pairs with side-A policy.

    Returns:
        List of wins for side A (same length as input pairs).
    """
    wins_list = []
    for p_idx, (team_a, team_b) in enumerate(zip(pairs_teams_a, pairs_teams_b, strict=False)):
        pair_seed = base_seed + p_idx * 100
        view_a = TeamView(team_a)
        view_b = TeamView(team_b)
        wins_a = 0
        policy_rng = np.random.default_rng(pair_seed + 5000)
        for b_idx in range(n_battles):
            battle_seed = pair_seed + b_idx + 2000
            bp_b = opp_policies[int(policy_rng.integers(0, len(opp_policies)))]

            idx_a = sel.decision((team_a, view_b), 4)
            idx_b = sel.decision((team_b, view_a), 4)

            sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
            sub_b, sub_view_b = subteam(team_b, view_b, idx_b)

            battle_teams = get_battle_teams((sub_a, sub_b), 2)
            state = State(battle_teams)
            gen = np.random.default_rng(battle_seed)
            rng_tuple = ((gen, gen), (gen, gen))
            engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

            while not engine.finished():
                sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
                sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
                cmd0 = bp_side_a.decision(sv0, sub_view_b)
                cmd1 = bp_b.decision(sv1, sub_view_a)
                engine.run_turn((cmd0, cmd1))

            if engine.winning_side == 0:
                wins_a += 1
        wins_list.append(wins_a)
    return wins_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Matchup Predictor viability experiment.")
    parser.add_argument("--n-pairs", type=int, default=2000, help="Number of team pairings")
    parser.add_argument(
        "--n-battles", type=int, default=30,
        help="Battles per pairing (30+ recommended for statistical robustness; requires LRU cache)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--side-a-policy", type=str, default="greedy", choices=["greedy", "dongimon"],
                        help="Which policy pilots side A (greedy=dongimon)")
    parser.add_argument("--corr-pairs", type=int, default=200, help="Holdout pairs for policy correlation test")
    parser.add_argument("--corr-battles", type=int, default=10, help="Battles per holdout pair for correlation test")
    args = parser.parse_args()

    opp_policies: list[Any] = [
        GreedyBattlePolicy(),
        _import_bp("competitors.competitor1_jjj", "JJJ_Competitor"),
    ]
    opp_names = ["Greedy", "JJJ"]

    if args.side_a_policy == "greedy":
        bp_side_a = GreedyBattlePolicy()
        side_a_name = "Greedy"
        effective_n_pairs = args.n_pairs
    else:
        bp_side_a = _dongimon_bp()
        side_a_name = "Dongimon"
        effective_n_pairs = min(args.n_pairs, 500)

    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    total_battles = effective_n_pairs * args.n_battles
    print("=" * 60)
    print(f"Matchup Predictor — Experiment (side A = {side_a_name})")
    print(f"  seed={args.seed}, n_pairs={effective_n_pairs}, n_battles_per_pair={args.n_battles}")
    print(f"  opponent policies: {', '.join(opp_names)}")
    print(f"  side-A policy: {side_a_name}")
    print(f"  total simulated battles: {total_battles}")
    print("=" * 60)

    start = time.perf_counter()

    gen_pairs_teams_a = []
    gen_pairs_teams_b = []
    all_pair_seeds = []
    for p_idx in range(effective_n_pairs):
        pair_seed = args.seed + p_idx * 100
        gen_a = np.random.default_rng(pair_seed)
        gen_b = np.random.default_rng(pair_seed + 1000)
        all_pair_seeds.append(pair_seed)
        gen_pairs_teams_a.append(gen_team(6, 4, gen_a))
        gen_pairs_teams_b.append(gen_team(6, 4, gen_b))

    wins_list = _run_battles_with_side_a(
        bp_side_a, opp_policies,
        gen_pairs_teams_a, gen_pairs_teams_b,
        sel, params, args.seed, args.n_battles,
    )

    features_list: list[dict[str, float]] = []
    labels: list[int] = []
    for p_idx in range(effective_n_pairs):
        feat = _extract_pairwise_features(gen_pairs_teams_a[p_idx], gen_pairs_teams_b[p_idx])
        features_list.append(feat)
        labels.append(1 if wins_list[p_idx] > args.n_battles // 2 else 0)

        if (p_idx + 1) % 500 == 0:
            elapsed = time.perf_counter() - start
            pct = (p_idx + 1) / effective_n_pairs * 100
            print(f"  Evaluated {p_idx + 1}/{effective_n_pairs} ({pct:.0f}%), {elapsed:.1f}s")

    elapsed = time.perf_counter() - start
    win_rate_a = sum(labels) / len(labels) if labels else 0
    print(f"\nData complete: {effective_n_pairs} pairings in {elapsed:.1f}s")
    print(f"  Side A ({side_a_name}) win rate: {win_rate_a:.3f}")

    if not features_list:
        print("Error: No features generated.")
        return

    feature_names = list(features_list[0].keys())
    x_data = np.array([[f[n] for n in feature_names] for f in features_list], dtype=np.float64)
    y_data = np.array(labels, dtype=np.int32)

    if len(np.unique(y_data)) < 2:
        print("\nError: All labels are the same class. Try more battles per pairing.")
        return

    x_tr, x_te, y_tr, y_te = train_test_split(x_data, y_data, test_size=0.2, random_state=args.seed)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    print("\n--- Feature Selection (Random Forest, 250 trees) ---")
    rf = RandomForestClassifier(n_estimators=250, random_state=args.seed, n_jobs=-1)
    rf.fit(x_tr_s, y_tr)
    importances = rf.feature_importances_
    threshold = 0.02
    important_mask = importances >= threshold
    important_indices = [i for i, ok in enumerate(important_mask) if ok]

    importance_snapshot = {
        "experiment": "matchup_predictor",
        "n_features_total": len(feature_names),
        "n_features_kept": len(important_indices),
        "threshold": threshold,
        "features": [
            {"name": feature_names[i], "importance": float(importances[i]), "kept": bool(important_mask[i])}
            for i in range(len(feature_names))
        ],
    }
    snapshot_path = Path("data/gini_matchup_predictor.json")
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

    logreg = LogisticRegression(max_iter=1000, random_state=args.seed)
    logreg.fit(x_tr_clean, y_tr)
    y_prob = logreg.predict_proba(x_te_clean)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print(f"\n  LogisticRegression AUROC on holdout: {auc:.4f}")

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(x_tr_clean, y_tr)
    y_prob_mlp = mlp.predict_proba(x_te_clean)[:, 1]
    auc_mlp = roc_auc_score(y_te, y_prob_mlp)
    print(f"\n  MLPClassifier (64->32) AUROC on holdout: {auc_mlp:.4f}")

    best_auc = max(auc, auc_mlp)
    threshold = 0.65
    print("\n--- Verdict ---")
    print(f"  Best AUROC: {best_auc:.4f} (threshold: {threshold})")
    print(f"  Viable: {'YES' if best_auc > threshold else 'NO'}")

    if args.side_a_policy == "greedy" and effective_n_pairs >= args.corr_pairs + 100:
        print("\n" + "=" * 60)
        print("Policy Correlation Test: Greedy vs Dongimon")
        print(f"  Holding out {args.corr_pairs} pairs for comparison")
        print("  Re-running with Dongimon as side A...")
        print("=" * 60)

        dongimon_bp = _dongimon_bp()
        holdout_start = effective_n_pairs - args.corr_pairs

        hl_teams_a = gen_pairs_teams_a[holdout_start:]
        hl_teams_b = gen_pairs_teams_b[holdout_start:]

        dongimon_wins = _run_battles_with_side_a(
            dongimon_bp, opp_policies,
            hl_teams_a, hl_teams_b,
            sel, params, args.seed + 99999, args.corr_battles,
        )

        greedy_wins_hl = wins_list[holdout_start:]

        greedy_wr = [w / max(args.n_battles, 1) for w in greedy_wins_hl]
        dongimon_wr = [w / max(args.corr_battles, 1) for w in dongimon_wins]

        rho, pval = spearmanr(greedy_wr, dongimon_wr)
        print(f"\n  Spearman rho: {rho:.4f} (p={pval:.4f})")
        print(f"  Greedy win rate mean: {np.mean(greedy_wr):.3f}")
        print(f"  Dongimon win rate mean: {np.mean(dongimon_wr):.3f}")

        corr_threshold = 0.70
        if rho > corr_threshold and pval < 0.05:
            print(f"\n  Correlation is strong (rho > {corr_threshold}): Greedy labels can substitute for Dongimon.")
            print("  Proceed to full-scale data generation with Greedy as side A.")
        else:
            print(f"\n  Correlation is weak (rho <= {corr_threshold}): Dongimon data is needed in training.")
            print("  Full-scale data gen should include Dongimon battles.")

    if best_auc > threshold:
        print("\n  Proceed to full-scale data generation (5-10 hours).")
    else:
        print("\n  Suggestion: Increase n_pairs, improve pairwise features, or use a pool of opponents.")


if __name__ == "__main__":
    main()
