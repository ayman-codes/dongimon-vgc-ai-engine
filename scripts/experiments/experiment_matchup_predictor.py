"""Experiment: Matchup Predictor viability (MP-1 through MP-6).

Generates stratified team pairs, runs N=50 battles with a fixed
side-A policy vs Greedy, extracts features on the 4-member subteam
that actually fought, trains Ridge regression on continuous win_rate_a
target, and runs 3-track ablation (BST-only / features-only / full)
with 5-fold CV and bootstrap CIs.

Uses experiment_utils for team generation, battle execution, feature
computation, and statistical evaluation.

Usage:
    uv run python scripts/experiments/experiment_matchup_predictor.py \
        --n-pairs=3000 --n-battles=50 --seed=42 --side-a-policy=greedy
"""

import argparse
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleRuleParam
from xgboost import XGBRegressor

from competitor import DongimonCompetitor
from scripts.experiments.experiment_utils import (
    compute_pairwise_features,
    compute_subteam_features,
    discover_latest_jsonl,
    generate_stratified_teams,
    profile_teams,
    run_ablation_tracks,
    run_pair_battles,
)
from src.config.loader import load_battle_weights


def _dongimon_bp() -> Any:
    """Return Dongimon's battle policy with current tuned weights.

    Returns:
        DongimonBattlePolicy instance loaded from battle_weights.yaml.
    """
    return DongimonCompetitor(custom_weights=load_battle_weights().model_dump()).battlepolicy


def _compute_bst_delta_from_feats(feats_a: dict[str, float], feats_b: dict[str, float]) -> float:
    """Compute BST difference from pre-computed feature dicts.

    Args:
        feats_a: Features for subteam A.
        feats_b: Features for subteam B.

    Returns:
        |mean_bst_a - mean_bst_b|.
    """
    return abs(feats_a.get("bst_avg", 0.0) - feats_b.get("bst_avg", 0.0))


def _save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Save list of dicts as JSONL file.

    Args:
        rows: List of row dicts to write.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _binarize_label(win_rate: float) -> int:
    """Convert continuous win rate to binary label.

    Args:
        win_rate: Continuous win rate in [0, 1].

    Returns:
        1 if win_rate > 0.5, else 0.
    """
    return 1 if win_rate > 0.5 else 0


def main() -> None:
    """Run the Matchup Predictor viability experiment."""
    parser = argparse.ArgumentParser(description="Matchup Predictor viability experiment")
    parser.add_argument("--n-pairs", type=int, default=3000, help="Number of team pairings")
    parser.add_argument(
        "--n-battles", type=int, default=50,
        help="Battles per pairing (N=50 for label stability)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--side-a-policy", type=str, default="greedy", choices=["greedy", "dongimon"],
        help="Policy piloting side A (determined by Phase 1 verdict)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/experiments/matchup_predictor"),
        help="Output directory for data, model, and metrics",
    )
    parser.add_argument(
        "--data", type=str, default="generate", choices=["generate", "load"],
        help="generate: run full experiment. load: skip to feature selection from saved JSONL.",
    )
    parser.add_argument(
        "--data-path", type=Path, default=None,
        help="Explicit JSONL path for --data=load (auto-discovers latest if omitted).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = Path("data/experiments/meta")
    meta_dir.mkdir(parents=True, exist_ok=True)

    if args.side_a_policy == "greedy":
        side_a_policy = GreedyBattlePolicy()
        side_a_name = "Greedy"
    else:
        side_a_policy = _dongimon_bp()
        side_a_name = "Dongimon"

    opponent_policy = GreedyBattlePolicy()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    print("=" * 60)
    print("Matchup Predictor Experiment")
    mode_label = "GENERATE" if args.data == "generate" else "LOAD"
    print(f"  seed={args.seed}, mode={mode_label}, n_pairs={args.n_pairs}, n_battles={args.n_battles}")
    print(f"  side-A policy: {side_a_name}")
    print("  opponent: Greedy (fixed)")
    print(f"  output={output_dir.resolve()}")
    print("=" * 60)

    start = time.perf_counter()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    win_rates: list[float] = []
    pairwise_feats: list[dict[str, float]] = []
    profile: dict[str, Any] = {}
    tier_counts_final: dict[str, int] = {}
    jsonl_path: Path | None = None

    if args.data == "generate":
        print("Generating stratified teams...")
        all_teams, tier_labels = generate_stratified_teams(
            n_teams=args.n_pairs * 2, seed=args.seed
        )
        teams_a = all_teams[: args.n_pairs]
        teams_b = all_teams[args.n_pairs : args.n_pairs * 2]

        print(f"  {len(all_teams)} teams, {args.n_pairs} pairs")
        tier_counts: dict[str, int] = {}
        for t in tier_labels:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        print(f"  Tier distribution: {tier_counts}")

        print("Profiling teams...")
        profile = profile_teams(teams_a + teams_b)
        profile_path = meta_dir / f"mp_profile_{timestamp}.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        total_battles = args.n_pairs * args.n_battles
        print(f"Running {args.n_pairs} pairs x {args.n_battles} battles ({total_battles} total)...")

        raw_rows: list[dict[str, Any]] = []
        flush_interval = 100
        jsonl_path = output_dir / f"mp_data_{timestamp}.jsonl"

        for pair_idx in range(args.n_pairs):
            team_a = teams_a[pair_idx]
            team_b = teams_b[pair_idx]
            pair_seed = args.seed + pair_idx * 100

            wins_a, _, idx_a, idx_b = run_pair_battles(
                team_a=team_a,
                team_b=team_b,
                bp_side_a=side_a_policy,
                bp_side_b=opponent_policy,
                n_battles=args.n_battles,
                pair_seed=pair_seed,
                params=params,
                sel=sel,
            )

            win_rate = wins_a / args.n_battles
            win_rates.append(win_rate)

            subteam_a_members = [team_a.members[i] for i in idx_a] if idx_a else list(team_a.members[:4])
            subteam_b_members = [team_b.members[i] for i in idx_b] if idx_b else list(team_b.members[:4])

            sub_feats_a = compute_subteam_features(subteam_a_members)
            sub_feats_b = compute_subteam_features(subteam_b_members)
            pair_feats = compute_pairwise_features(subteam_a_members, subteam_b_members)
            pairwise_feats.append(pair_feats)

            bst_delta = _compute_bst_delta_from_feats(sub_feats_a, sub_feats_b)

            raw_rows.append({
                "pair_id": pair_idx,
                "seed": pair_seed,
                "wins_a": wins_a,
                "n_battles": args.n_battles,
                "win_rate_a": win_rate,
                "selected_indices_a": idx_a,
                "selected_indices_b": idx_b,
                "bst_delta": bst_delta,
                "features": pair_feats,
            })

            if (pair_idx + 1) % flush_interval == 0:
                _save_jsonl(raw_rows, jsonl_path)
                elapsed = time.perf_counter() - start
                rate = (pair_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  {pair_idx + 1}/{args.n_pairs} pairs ({rate:.1f} pairs/s, {elapsed:.0f}s)")

        _save_jsonl(raw_rows, jsonl_path)
        elapsed = time.perf_counter() - start
        print(f"\nData complete: {args.n_pairs} pairs in {elapsed:.0f}s")
        tier_counts_final = tier_counts
    else:
        data_path = args.data_path or discover_latest_jsonl(output_dir, "mp_data")
        if data_path is None:
            print("ERROR: No mp_data_*.jsonl found and --data-path not provided.")
            sys.exit(1)
        print(f"[LOAD] Reading data from {data_path}")
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                win_rates.append(float(row["win_rate_a"]))
                pairwise_feats.append(row["features"])
        print(f"[LOAD] Loaded {len(win_rates)} pairs")
        tier_counts_final = {"loaded": len(win_rates)}
        elapsed = time.perf_counter() - start
        profile = {}
        jsonl_path = data_path

    print(f"  Mean win rate (side A, {side_a_name}): {np.mean(win_rates):.4f}")

    feature_names = list(pairwise_feats[0].keys())
    x_data = np.array([[f[n] for n in feature_names] for f in pairwise_feats], dtype=np.float64)
    y_data = np.array(win_rates, dtype=np.float64)

    bst_keys = ["bst_avg_diff", "bst_max_diff", "bst_min_diff", "bst_std_diff"]
    bst_indices = [feature_names.index(k) for k in bst_keys if k in feature_names]
    print(f"  Total features: {len(feature_names)}")
    print(f"  BST feature indices: {bst_indices} ({[feature_names[i] for i in bst_indices]})")

    print("\n--- Feature Selection (Random Forest, 250 trees) ---")
    rf = RandomForestRegressor(n_estimators=250, random_state=args.seed, n_jobs=-1)
    rf.fit(x_data, y_data)
    importances = rf.feature_importances_
    threshold = 0.02
    important_mask = importances >= threshold
    important_indices = [int(i) for i, ok in enumerate(important_mask) if ok]

    gini_snapshot = {
        "experiment": "matchup_predictor",
        "n_features_total": len(feature_names),
        "n_features_kept": len(important_indices),
        "threshold": threshold,
        "features": [
            {"name": feature_names[i], "importance": float(importances[i]), "kept": bool(important_mask[i])}
            for i in range(len(feature_names))
        ],
    }
    gini_path = output_dir / f"mp_gini_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(gini_path, "w") as f:
        json.dump(gini_snapshot, f, indent=2)
    print(f"  Saved Gini importances to {gini_path}")
    print(f"  Kept {len(important_indices)}/{len(feature_names)} features (threshold={threshold})")

    dropped = [feature_names[i] for i in range(len(feature_names)) if not important_mask[i]]
    if len(dropped) <= 15:
        print(f"  Dropped: {dropped}")

    print("\n--- 3-Track Ablation: Ridge Regression (continuous target) ---")
    def _ridge_factory() -> Ridge:
        return Ridge(alpha=1.0, random_state=args.seed)

    ridge_results = run_ablation_tracks(
        x_full=x_data,
        y=y_data,
        feature_names=feature_names,
        bst_feature_indices=bst_indices,
        model_factory=_ridge_factory,
        task="regression",
        n_folds=5,
        n_bootstrap=1000,
        seed=args.seed,
    )

    for track_name in ["bst_only", "features_only", "full"]:
        if track_name in ridge_results:
            r = ridge_results[track_name]
            ci = r.get("ci", [0, 0, 0])
            p_vs_bst = r.get("p_value_vs_bst", 1.0)
            sig = r.get("holm_significant", False)
            print(f"  {track_name}: R\u00b2={r['scores']['mean']:.4f}\u00b1{r['scores']['std']:.4f} "
                  f"CI=[{ci[1]:.4f},{ci[2]:.4f}] p_vs_bst={p_vs_bst:.4f} sig={sig}")

    print("\n--- 3-Track Ablation: LogisticRegression (binarized target) ---")
    y_binary = np.array([_binarize_label(wr) for wr in win_rates], dtype=np.float64)
    if len(np.unique(y_binary)) < 2:
        print("  Skipping: only one class in binarized labels")
        logreg_results: dict[str, dict[str, Any]] = {}
    else:
        def _logreg_factory() -> LogisticRegression:
            return LogisticRegression(max_iter=2000, random_state=args.seed)

        logreg_results = run_ablation_tracks(
            x_full=x_data,
            y=y_binary,
            feature_names=feature_names,
            bst_feature_indices=bst_indices,
            model_factory=_logreg_factory,
            task="classification",
            n_folds=5,
            n_bootstrap=1000,
            seed=args.seed,
        )

        for track_name in ["bst_only", "features_only", "full"]:
            if track_name in logreg_results:
                r = logreg_results[track_name]
                ci = r.get("ci", [0, 0, 0])
                p_vs_bst = r.get("p_value_vs_bst", 1.0)
                sig = r.get("holm_significant", False)
                print(f"  {track_name}: AUROC={r['scores']['mean']:.4f}\u00b1{r['scores']['std']:.4f} "
                      f"CI=[{ci[1]:.4f},{ci[2]:.4f}] p_vs_bst={p_vs_bst:.4f} sig={sig}")

    full_ridge_r2 = ridge_results.get("full", {}).get("scores", {}).get("mean", 0.0)
    full_logreg_auc = logreg_results.get("full", {}).get("scores", {}).get("mean", 0.0)
    logreg_p_vs_bst = logreg_results.get("bst_only", {}).get("p_value_vs_bst", 1.0)

    bst_auc = logreg_results.get("bst_only", {}).get("scores", {}).get("mean", 0.0)
    full_auc = logreg_results.get("full", {}).get("scores", {}).get("mean", 0.0)
    auc_delta = full_auc - bst_auc

    print("\n--- Verdict ---")
    kill_switch_triggered = False
    kill_reason = ""

    if full_logreg_auc < 0.65:
        kill_switch_triggered = True
        kill_reason = f"Track C AUROC ({full_logreg_auc:.4f}) < 0.65"
    elif logreg_p_vs_bst > 0.05 or auc_delta <= 0.005:
        kill_switch_triggered = True
        kill_reason = (
            f"Track C not significantly better than Track A "
            f"(p={logreg_p_vs_bst:.4f}, delta_AUC={auc_delta:.4f})"
        )

    if kill_switch_triggered:
        print(f"  NOT VIABLE: {kill_reason}")
    else:
        print(f"  VIABLE: Track C AUROC={full_logreg_auc:.4f}, R2={full_ridge_r2:.4f}")

    elapsed = time.perf_counter() - start

    metrics = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_pairs": len(win_rates),
        "n_battles": args.n_battles,
        "side_a_policy": side_a_name,
        "duration_seconds": round(elapsed, 1),
        "mean_win_rate_a": float(np.mean(win_rates)),
        "n_features_total": len(feature_names),
        "bst_feature_names": [feature_names[i] for i in bst_indices],
        "bst_feature_indices": bst_indices,
        "ridge_ablation": ridge_results,
        "logreg_ablation": logreg_results,
        "gini_importances": gini_snapshot,
        "viable": not kill_switch_triggered,
        "kill_reason": kill_reason if kill_switch_triggered else "",
        "bst_only_auc": bst_auc,
        "full_auc": full_auc,
        "full_r2": full_ridge_r2,
    }

    metrics_path = output_dir / f"mp_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    meta = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_pairs": len(win_rates),
        "n_battles": args.n_battles,
        "side_a_policy": side_a_name,
        "data_profile": profile,
        "tier_counts": tier_counts_final,
        "feature_names": feature_names,
        "bst_feature_indices": bst_indices,
    }
    meta_path = output_dir / f"mp_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  Metrics saved to {metrics_path}")
    print(f"  Metadata saved to {meta_path}")
    print(f"  Raw data saved to {jsonl_path}")

    if not kill_switch_triggered:
        print("Saving all models...")
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_data)

        def _save_model(model_obj: Any, name: str) -> None:
            path = output_dir / f"MP_{name}.pkl"
            with open(path, "wb") as bf:
                pickle.dump({
                    "model": model_obj,
                    "scaler": scaler,
                    "feature_names": feature_names,
                    "config": {
                        "side_a_policy": side_a_name,
                        "n_battles": args.n_battles,
                        "seed": args.seed,
                    },
                }, bf)
            print(f"  Saved {path}")

        ridge_save = Ridge(alpha=1.0, random_state=args.seed)
        ridge_save.fit(x_scaled, y_data)
        _save_model(ridge_save, "Ridge")

        logreg_save = LogisticRegression(max_iter=2000, random_state=args.seed)
        logreg_save.fit(x_scaled, y_binary)
        _save_model(logreg_save, "LogReg")

        xgb_save = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=args.seed, verbosity=0,
        )
        xgb_save.fit(x_scaled, y_data)
        _save_model(xgb_save, "XGBoost")

    print("=" * 60)
    print(f"Done in {elapsed:.0f}s. Viable: {not kill_switch_triggered}")


if __name__ == "__main__":
    main()
