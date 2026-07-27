"""Experiment: Greedy vs Dongimon policy correlation test.

Determines whether fast Greedy-labeled training data generalizes
to Dongimon battle outcomes. The result dictates the labeling
policy for downstream Matchup Predictor experiments.

Uses stratified team generation, same opponent-selection seed for
both policy runs, attenuation-corrected Spearman rho, and BST-binned
stratification analysis.

Usage:
    uv run python scripts/experiments/experiment_policy_correlation.py \
        --n-pairs=500 --n-battles=30 --seed=42
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.stats import spearmanr
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleRuleParam

from competitor import DongimonCompetitor
from scripts.experiments.experiment_utils import (
    attenuation_corrected_rho,
    compute_subteam_features,
    generate_stratified_teams,
    profile_teams,
    run_pair_battles,
)
from src.config.loader import load_battle_weights


def _dongimon_bp() -> Any:
    """Return Dongimon's battle policy with current tuned weights.

    Returns:
        DongimonBattlePolicy instance loaded from battle_weights.yaml.
    """
    return DongimonCompetitor(custom_weights=load_battle_weights().model_dump()).battlepolicy


def _compute_bst_delta(
    subteam_a: list[Any],
    subteam_b: list[Any],
) -> float:
    """Compute the mean BST difference between two subteams.

    Args:
        subteam_a: List of 4 Pokemon for side A.
        subteam_b: List of 4 Pokemon for side B.

    Returns:
        |mean_bst_a - mean_bst_b|.
    """
    feats_a = compute_subteam_features(subteam_a)
    feats_b = compute_subteam_features(subteam_b)
    return abs(feats_a.get("bst_avg", 0.0) - feats_b.get("bst_avg", 0.0))


def main() -> None:
    """Run the Greedy-vs-Dongimon policy correlation experiment."""
    parser = argparse.ArgumentParser(
        description="Greedy vs Dongimon policy correlation experiment"
    )
    parser.add_argument("--n-pairs", type=int, default=500, help="Number of team pairings")
    parser.add_argument("--n-battles", type=int, default=30, help="Battles per pairing per policy")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/experiments/correlation"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = Path("data/experiments/meta")
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Policy Correlation Experiment — Greedy vs Dongimon")
    print(f"  seed={args.seed}, n_pairs={args.n_pairs}, n_battles={args.n_battles}")
    print(f"  output={output_dir.resolve()}")
    print("=" * 60)

    start = time.perf_counter()

    print("Generating stratified teams...")
    all_teams, tier_labels = generate_stratified_teams(
        n_teams=args.n_pairs * 2, seed=args.seed
    )

    teams_a = all_teams[: args.n_pairs]
    teams_b = all_teams[args.n_pairs : args.n_pairs * 2]

    tier_names = ["random", "ga", "coverage"]
    tier_counts_a = dict.fromkeys(tier_names, 0)
    tier_counts_b = dict.fromkeys(tier_names, 0)
    for i in range(args.n_pairs):
        if i < len(tier_labels):
            tier_counts_a[tier_labels[i]] = tier_counts_a.get(tier_labels[i], 0) + 1
        idx_b = args.n_pairs + i
        if idx_b < len(tier_labels):
            tier_counts_b[tier_labels[idx_b]] = tier_counts_b.get(tier_labels[idx_b], 0) + 1

    tier_counts = {"team_a": tier_counts_a, "team_b": tier_counts_b}

    print(f"  Teams generated: {len(all_teams)} total, {args.n_pairs} pairs")
    print(f"  Tier distribution A: {tier_counts_a}")
    print(f"  Tier distribution B: {tier_counts_b}")

    print("Profiling teams...")
    profile = profile_teams(teams_a + teams_b)
    profile_path = meta_dir / f"correlation_profile_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  Profile saved to {profile_path}")

    print("Building policies...")
    dongimon_policy = _dongimon_bp()
    greedy_policy = GreedyBattlePolicy()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    print(f"Running {args.n_pairs} pairs x {args.n_battles} battles x 2 policies...")
    print("  Opponent: Greedy (fixed)")

    wr_greedy: list[float] = []
    wr_dongimon: list[float] = []
    bst_deltas: list[float] = []
    subteam_as: list[list[Any]] = []
    subteam_bs: list[list[Any]] = []

    for pair_idx in range(args.n_pairs):
        team_a = teams_a[pair_idx]
        team_b = teams_b[pair_idx]
        pair_seed = args.seed + pair_idx * 100

        wins_g, _, idx_g_a, idx_g_b = run_pair_battles(
            team_a=team_a,
            team_b=team_b,
            bp_side_a=greedy_policy,
            bp_side_b=GreedyBattlePolicy(),
            n_battles=args.n_battles,
            pair_seed=pair_seed,
            params=params,
            sel=sel,
        )

        sub_a = [team_a.members[i] for i in idx_g_a] if idx_g_a else list(team_a.members[:4])
        sub_b = [team_b.members[i] for i in idx_g_b] if idx_g_b else list(team_b.members[:4])
        subteam_as.append(sub_a)
        subteam_bs.append(sub_b)

        wins_d, _, _, _ = run_pair_battles(
            team_a=team_a,
            team_b=team_b,
            bp_side_a=dongimon_policy,
            bp_side_b=GreedyBattlePolicy(),
            n_battles=args.n_battles,
            pair_seed=pair_seed,
            params=params,
            sel=sel,
        )

        wr_g = wins_g / args.n_battles
        wr_d = wins_d / args.n_battles
        wr_greedy.append(wr_g)
        wr_dongimon.append(wr_d)

        bst_delta = _compute_bst_delta(sub_a, sub_b)
        bst_deltas.append(bst_delta)

        if (pair_idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {pair_idx + 1}/{args.n_pairs} pairs done ({elapsed:.0f}s)")

    wr_g = np.array(wr_greedy)
    wr_d = np.array(wr_dongimon)
    bst_d_arr = np.array(bst_deltas)

    print("Computing correlation metrics...")
    corr_result = attenuation_corrected_rho(
        wr_g, wr_d, n_battles=args.n_battles, n_bootstrap=1000, seed=args.seed
    )

    low_mask = bst_d_arr < 20.0
    med_mask = (bst_d_arr >= 20.0) & (bst_d_arr < 60.0)
    high_mask = bst_d_arr >= 60.0

    stratified: dict[str, dict[str, Any]] = {}
    for label, mask in [("low_bst_diff", low_mask), ("medium_bst_diff", med_mask), ("high_bst_diff", high_mask)]:
        n_in_bin = int(np.sum(mask))
        if n_in_bin >= 5:
            r = spearmanr(wr_g[mask], wr_d[mask])
            stratified[label] = {
                "n": n_in_bin,
                "rho": float(r.statistic),
                "p_value": float(r.pvalue),
            }
        else:
            stratified[label] = {"n": n_in_bin, "rho": 0.0, "p_value": 1.0}

    rho_corrected = corr_result["rho_corrected"]
    p_value = corr_result["p_value"]
    if rho_corrected >= 0.70 and p_value < 0.05:
        verdict = "GREEDY VIABLE"
    elif rho_corrected >= 0.50:
        verdict = "GREEDY PARTIALLY VIABLE"
    else:
        verdict = "DONGIMON REQUIRED"

    mean_wr_g = float(np.mean(wr_g))
    mean_wr_d = float(np.mean(wr_d))

    elapsed = time.perf_counter() - start
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    result = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "n_battles": args.n_battles,
        "duration_seconds": round(elapsed, 1),
        "team_generation": tier_counts,
        "data_profile": profile,
        "overall": corr_result,
        "stratified": stratified,
        "mean_win_rates": {
            "greedy": mean_wr_g,
            "dongimon": mean_wr_d,
        },
        "verdict": verdict,
    }

    result_path = output_dir / f"correlation_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
            default=lambda o: str(o)
            if not isinstance(o, (int, float, str, bool, list, dict, type(None)))
            else o,
        )

    verdict_path = output_dir / "VERDICT.txt"
    with open(verdict_path, "w") as f:
        f.write(f"{verdict}\n")
        f.write(f"rho_corrected={rho_corrected:.4f} p={p_value:.6f}\n")
        f.write(f"rho_raw={corr_result['rho_raw']:.4f}\n")
        f.write(f"reliability={corr_result['reliability']:.4f}\n")

    print("=" * 60)
    print("Results:")
    print(f"  rho_raw:         {corr_result['rho_raw']:.4f}")
    print(f"  rho_corrected:   {rho_corrected:.4f}")
    print(f"  reliability:     {corr_result['reliability']:.4f}")
    print(f"  p_value:         {p_value:.6f}")
    print(f"  rho_raw_ci:      [{corr_result['rho_raw_ci'][0]:.4f}, {corr_result['rho_raw_ci'][1]:.4f}]")
    print(f"  rho_corrected_ci: [{corr_result['rho_corrected_ci'][0]:.4f}, {corr_result['rho_corrected_ci'][1]:.4f}]")
    print(f"  mean_wr_greedy:  {mean_wr_g:.4f}")
    print(f"  mean_wr_dongimon:{mean_wr_d:.4f}")
    print("  stratified:")
    for label, stats_dict in stratified.items():
        print(f"    {label}: n={stats_dict['n']}, rho={stats_dict['rho']:.4f}")
    print(f"  VERDICT: {verdict}")
    print(f"  Duration: {elapsed:.0f}s")
    print(f"  Results saved to {result_path}")
    print("=" * 60)

    label_advice = (
        "GREEDY can be used for MP data labeling."
        if verdict == "GREEDY VIABLE"
        else "GREEDY partially viable; use Greedy with caution for close matchups."
        if verdict == "GREEDY PARTIALLY VIABLE"
        else "Dongimon labels required for all MP data."
    )
    print(f"\nNext step: {label_advice}")


if __name__ == "__main__":
    main()
