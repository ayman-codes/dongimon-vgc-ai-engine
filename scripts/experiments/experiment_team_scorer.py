"""Experiment: Team Quality Scorer viability (TQS-1 through TQS-10).

Generates 250 stratified teams, runs a Swiss tournament (Greedy-vs-Greedy)
to derive Bradley-Terry strength labels, residualizes against BST,
trains 4 regression models (LinearRegression, Ridge, Lasso, XGBoost),
and runs 3-track ablation with 5-fold CV, bootstrap CIs, and a
species-disjoint quarantine split.

Uses experiment_utils for team generation, swiss pairings, battle
execution, feature computation, and BT fitting.

Usage:
    uv run python scripts/experiments/experiment_team_scorer.py \
        --n-teams=250 --n-battles=30 --n-rounds=7 --seed=42 --policy=greedy
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
from scipy.stats import norm, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleRuleParam
from xgboost import XGBRegressor

from competitor import DongimonCompetitor
from scripts.experiments.experiment_utils import (
    bootstrap_ci,
    compute_subteam_features,
    discover_latest_jsonl,
    fit_bradley_terry,
    generate_ga_only_teams,
    generate_stratified_teams,
    holm_bonferroni,
    profile_teams,
    run_pair_battles,
    swiss_pairings_round,
    validate_stratification,
)
from src.config.loader import load_battle_weights

STAT_FEATURE_COUNT = 28
NO_STAT_START = STAT_FEATURE_COUNT


def _per_team_raw_win_rates(
    pair_outcomes: list[tuple[int, int, int, int]],
    n_teams: int,
) -> np.ndarray:
    """Compute raw win rate per team from all pair outcomes.

    Args:
        pair_outcomes: List of (i, j, wins_i, wins_j).
        n_teams: Total number of teams.

    Returns:
        Array of shape (n_teams,) with win rate per team.
    """
    total_wins = np.zeros(n_teams)
    total_battles_arr = np.zeros(n_teams)
    for i, j, w_i, w_j in pair_outcomes:
        total_wins[i] += w_i
        total_wins[j] += w_j
        total_battles_arr[i] += w_i + w_j
        total_battles_arr[j] += w_i + w_j
    wr = np.zeros(n_teams)
    for t in range(n_teams):
        if total_battles_arr[t] > 0:
            wr[t] = total_wins[t] / total_battles_arr[t]
    return wr


def _residualize(theta: np.ndarray, bst_avg: np.ndarray) -> tuple[np.ndarray, float]:
    """Residualize BT strengths against bst_avg via OLS.

    Args:
        theta: Bradley-Terry strengths (n_teams,).
        bst_avg: Mean BST per team (n_teams,).

    Returns:
        Tuple of (residuals, r2_of_ols_fit).
    """
    x_ols = bst_avg.reshape(-1, 1)
    ols = LinearRegression()
    ols.fit(x_ols, theta)
    pred = ols.predict(x_ols)
    residuals = theta - pred
    return residuals, float(r2_score(theta, pred))


def _compute_all_team_features(
    teams: list[Any],
) -> tuple[np.ndarray, list[str], list[int], list[int]]:
    """Compute feature matrix for all teams.

    Args:
        teams: List of vgc2 Team objects.

    Returns:
        Tuple of (x_mat, feature_names, bst_all_indices, no_bst_indices).
    """
    feat_dicts = [compute_subteam_features(list(t.members)) for t in teams]
    feature_names = list(feat_dicts[0].keys())
    x_mat = np.array(
        [[f[n] for n in feature_names] for f in feat_dicts], dtype=np.float64
    )

    bst_all_indices = list(range(STAT_FEATURE_COUNT))
    no_bst_indices = list(range(NO_STAT_START, len(feature_names)))
    return x_mat, feature_names, bst_all_indices, no_bst_indices


def _species_disjoint_split(
    teams: list[Any],
    train_size: int,
    test_size: int,
    quarantine_size: int,
    seed: int,
    max_attempts: int = 100,
) -> tuple[list[int], list[int], list[int]]:
    """Split team indices ensuring quarantine shares no species with train.

    Args:
        teams: List of Team objects.
        train_size: Number of training teams.
        test_size: Number of test teams.
        quarantine_size: Number of quarantine teams.
        seed: RNG seed.
        max_attempts: Maximum reshuffle attempts.

    Returns:
        Tuple of (train_indices, test_indices, quarantine_indices).

    Raises:
        RuntimeError: If no disjoint split found within max_attempts.
    """
    rng = np.random.default_rng(seed)

    def _species_set(indices: list[int]) -> set[str]:
        species: set[str] = set()
        for idx in indices:
            for m in teams[idx].members:
                spec = m.species if hasattr(m, "species") else m
                species.add(
                    spec.name if hasattr(spec, "name") else str(id(spec))
                )
        return species

    for _ in range(max_attempts):
        all_idx = list(range(len(teams)))
        rng.shuffle(all_idx)
        train_idx = all_idx[:train_size]
        test_idx = all_idx[train_size : train_size + test_size]
        quarantine_idx = all_idx[
            train_size + test_size : train_size + test_size + quarantine_size
        ]

        train_species = _species_set(train_idx)
        quarantine_species = _species_set(quarantine_idx)
        if train_species.isdisjoint(quarantine_species):
            return train_idx, test_idx, quarantine_idx

    raise RuntimeError(
        f"Could not find species-disjoint split within {max_attempts} attempts"
    )


def _evaluate_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_quarantine: np.ndarray,
    y_quarantine: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a model and evaluate on test + quarantine.

    Args:
        model: sklearn-compatible regressor.
        x_train: Training features.
        y_train: Training targets.
        x_test: Test features.
        y_test: Test targets.
        x_quarantine: Quarantine features.
        y_quarantine: Quarantine targets.
        n_bootstrap: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Dict with r2_test, r2_quarantine, ci_test, ci_quarantine.
    """
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    y_pred_quarantine = model.predict(x_quarantine)

    r2_test_val = r2_score(y_test, y_pred_test)
    r2_quar_val = r2_score(y_quarantine, y_pred_quarantine)

    def _r2_fn(t: np.ndarray, p: np.ndarray) -> float:
        return float(r2_score(t, p))

    _, ci_lo_test, ci_hi_test = bootstrap_ci(
        y_test, y_pred_test, _r2_fn, n_bootstrap=n_bootstrap, seed=seed,
    )
    _, ci_lo_quar, ci_hi_quar = bootstrap_ci(
        y_quarantine, y_pred_quarantine, _r2_fn, n_bootstrap=n_bootstrap, seed=seed + 1,
    )

    return {
        "r2_test": float(r2_test_val),
        "r2_quarantine": float(r2_quar_val),
        "ci_test": [float(r2_test_val), ci_lo_test, ci_hi_test],
        "ci_quarantine": [float(r2_quar_val), ci_lo_quar, ci_hi_quar],
    }


def _ablation_track(
    x_full: np.ndarray,
    y_full: np.ndarray,
    col_indices: list[int],
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_quarantine: np.ndarray,
    y_quarantine: np.ndarray,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run 5-fold CV Ridge ablation on a feature subset.

    Also evaluates held-out test and quarantine sets.

    Args:
        x_full: Full training features.
        y_full: Training targets.
        col_indices: Column indices for this track's feature subset.
        x_test: Test features (full).
        y_test: Test targets.
        x_quarantine: Quarantine features (full).
        y_quarantine: Quarantine targets.
        n_folds: CV folds.
        n_bootstrap: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Dict with cv_mean, cv_std, cv_scores, r2_test, r2_quarantine,
        ci_test, ci_quarantine, n_features.
    """
    x_sub = x_full[:, col_indices]
    x_test_sub = x_test[:, col_indices]
    x_quarantine_sub = x_quarantine[:, col_indices]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_scores: list[float] = []
    for train_idx_cv, val_idx_cv in kf.split(x_sub):
        x_tr = x_sub[train_idx_cv]
        x_val = x_sub[val_idx_cv]
        y_tr = y_full[train_idx_cv]
        y_val = y_full[val_idx_cv]
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_val_s = scaler.transform(x_val)
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(x_tr_s, y_tr)
        y_pred_cv = ridge.predict(x_val_s)
        cv_scores.append(float(r2_score(y_val, y_pred_cv)))

    scaler_full = StandardScaler()
    x_sub_s = scaler_full.fit_transform(x_sub)
    ridge_final = Ridge(alpha=1.0, random_state=seed)
    ridge_final.fit(x_sub_s, y_full)

    x_test_s = scaler_full.transform(x_test_sub)
    x_quarantine_s = scaler_full.transform(x_quarantine_sub)
    y_pred_test = ridge_final.predict(x_test_s)
    y_pred_quarantine = ridge_final.predict(x_quarantine_s)

    r2_test_val = r2_score(y_test, y_pred_test)
    r2_quar_val = r2_score(y_quarantine, y_pred_quarantine)

    def _r2_fn(t: np.ndarray, p: np.ndarray) -> float:
        return float(r2_score(t, p))

    _, ci_lo_t, ci_hi_t = bootstrap_ci(
        y_test, y_pred_test, _r2_fn, n_bootstrap=n_bootstrap, seed=seed,
    )
    _, ci_lo_q, ci_hi_q = bootstrap_ci(
        y_quarantine, y_pred_quarantine, _r2_fn, n_bootstrap=n_bootstrap, seed=seed + 1,
    )

    return {
        "cv_mean": float(np.mean(cv_scores)) if cv_scores else 0.0,
        "cv_std": float(np.std(cv_scores)) if len(cv_scores) > 1 else 0.0,
        "cv_scores": cv_scores,
        "r2_test": float(r2_test_val),
        "r2_quarantine": float(r2_quar_val),
        "ci_test": [float(r2_test_val), float(ci_lo_t), float(ci_hi_t)],
        "ci_quarantine": [float(r2_quar_val), float(ci_lo_q), float(ci_hi_q)],
        "n_features": len(col_indices),
    }


def _paired_bootstrap_pval(
    cv_a: list[float],
    cv_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test: is the mean of cv_b greater than cv_a?

    Args:
        cv_a: CV scores for baseline track.
        cv_b: CV scores for comparison track.
        n_bootstrap: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        p-value from bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    n = len(cv_a)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs.append(
            np.mean([cv_b[i] for i in idx]) - np.mean([cv_a[i] for i in idx])
        )
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    if std_diff > 1e-10:
        z = abs(mean_diff) / (std_diff / np.sqrt(n_bootstrap))
        return float(2 * (1 - norm.cdf(z)))
    return 1.0


def main() -> None:
    """Run the Team Quality Scorer viability experiment."""
    parser = argparse.ArgumentParser(
        description="Team Quality Scorer viability experiment"
    )
    parser.add_argument(
        "--n-teams", type=int, default=250, help="Total teams to generate"
    )
    parser.add_argument(
        "--n-battles", type=int, default=30, help="Battles per Swiss pairing"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=7, help="Number of Swiss rounds"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--policy", type=str, default="greedy",
        choices=["greedy", "dongimon"],
        help="Battle policy for labeling (Phase 3a: greedy; Phase 3b: dongimon)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/experiments/team_scorer"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--generation-mode", type=str, default="stratified",
        choices=["stratified", "ga_only"],
        help=(
            "Team generation strategy. stratified: 28%% random + 40%% GA + "
            "28%% coverage. ga_only: 80%% GA-evolved + 20%% random."
        ),
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

    if args.policy == "greedy":
        labeling_policy: Any = GreedyBattlePolicy()
        policy_name = "greedy"
    else:
        labeling_policy = DongimonCompetitor(
            custom_weights=load_battle_weights().model_dump()
        ).battlepolicy
        policy_name = "dongimon"

    opponent_policy = labeling_policy
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    print("=" * 60)
    mode_label = "GENERATE" if args.data == "generate" else "LOAD"
    print("Team Quality Scorer — Experiment (Phase 3a)")
    print(
        f"  seed={args.seed}, n_teams={args.n_teams}, "
        f"n_battles={args.n_battles}, policy={policy_name}, mode={mode_label}"
    )
    print(f"  generation_mode={args.generation_mode}")
    print(f"  swiss_rounds={args.n_rounds}")
    print(f"  output={output_dir.resolve()}")
    print("=" * 60)

    start = time.perf_counter()
    n_rounds = args.n_rounds
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    profile: dict[str, Any] = {}
    tier_counts: dict[str, int] = {}
    strat_result: dict[str, Any] = {}
    jsonl_path: Path | None = None

    if args.data == "generate":
        if args.generation_mode == "ga_only":
            print("Generating GA-only teams (80% GA + 20% random)...")
            teams, tier_labels = generate_ga_only_teams(
                n_teams=args.n_teams,
                seed=args.seed,
                ga_fraction=0.80,
            )
        else:
            print("Generating stratified teams...")
            teams, tier_labels = generate_stratified_teams(
                n_teams=args.n_teams,
                seed=args.seed,
                fractions=(0.28, 0.40, 0.28),
            )
        n_teams_actual = len(teams)
        print(f"  {n_teams_actual} teams generated (mode={args.generation_mode})")
        tier_counts = {}
        for t in tier_labels:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        print(f"  Tier distribution: {tier_counts}")

        print("Validating stratification...")
        strat_result = validate_stratification(teams, tier_labels)
        print(f"  Mean BST per tier: {strat_result['mean_bst_per_tier']}")
        print(
            f"  Kruskal-Wallis: H={strat_result['kruskal_wallis_H']:.2f}, "
            f"p={strat_result['kruskal_wallis_p']:.4f}"
        )

        print("Profiling teams...")
        profile = profile_teams(teams)

        print(
            f"Running Swiss tournament ({n_rounds} rounds, "
            f"{args.n_battles} battles/pair)..."
        )
        swiss_rng = np.random.default_rng(args.seed + 10)
        scores = np.zeros(n_teams_actual)
        history: set[tuple[int, int]] = set()
        pair_outcomes: list[tuple[int, int, int, int]] = []

        for round_idx in range(n_rounds):
            round_pairs = swiss_pairings_round(scores, history, swiss_rng)
            for team_i, team_j in round_pairs:
                seed_offset = (
                    args.seed + round_idx * 1000 + team_i * 10 + team_j
                )
                wins_i, wins_j, _, _ = run_pair_battles(
                    team_a=teams[team_i],
                    team_b=teams[team_j],
                    bp_side_a=labeling_policy,
                    bp_side_b=opponent_policy,
                    n_battles=args.n_battles,
                    pair_seed=seed_offset,
                    params=params,
                    sel=sel,
                )
                pair_outcomes.append((team_i, team_j, wins_i, wins_j))
                if wins_i > wins_j:
                    scores[team_i] += 1.0
                elif wins_j > wins_i:
                    scores[team_j] += 1.0
                else:
                    scores[team_i] += 0.5
                    scores[team_j] += 0.5
            total_pairs_so_far = len(pair_outcomes)
            print(
                f"  Round {round_idx + 1}/{n_rounds}: "
                f"{len(round_pairs)} pairs, {total_pairs_so_far} total"
            )

        total_pairs = len(pair_outcomes)
        print(f"  Total pairs: {total_pairs}")

        print("Fitting Bradley-Terry...")
        theta = fit_bradley_terry(pair_outcomes, n_teams_actual, n_iter=200, lr=0.01)
        raw_wr = _per_team_raw_win_rates(pair_outcomes, n_teams_actual)
        bt_rho, bt_pval = spearmanr(theta, raw_wr)
        print(f"  BT vs raw WR: Spearman rho={bt_rho:.4f}, p={bt_pval:.6f}")
        if bt_rho < 0.8:
            print(
                "  WARNING: BT validation rho < 0.8. "
                "Labels may be unreliable. Continuing."
            )

        print("Computing team features + residualizing...")
        x_mat, feature_names, bst_all_indices, no_bst_indices = (
            _compute_all_team_features(teams)
        )
        bst_avg = x_mat[:, 0]
        y_residual, ols_r2 = _residualize(theta, bst_avg)
        jsonl_path = output_dir / f"tqs_data_{timestamp}.jsonl"

        data_rows: list[dict[str, Any]] = [
            {
                "team_idx": i,
                "bt_theta": float(theta[i]),
                "bt_residual": float(y_residual[i]),
                "bst_avg": float(bst_avg[i]),
                "features": compute_subteam_features(list(teams[i].members)),
            }
            for i in range(n_teams_actual)
        ]
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w") as f:
            for row in data_rows:
                f.write(json.dumps(row, default=str) + "\n")
    else:
        data_path = args.data_path or discover_latest_jsonl(output_dir, "tqs_data")
        if data_path is None:
            print("ERROR: No tqs_data_*.jsonl found and --data-path not provided.")
            sys.exit(1)
        print(f"[LOAD] Reading data from {data_path}")
        y_residual_list: list[float] = []
        bst_avg_list: list[float] = []
        all_feat_dicts: list[dict[str, float]] = []
        with open(data_path) as f:
            for line in f:
                row = json.loads(line)
                y_residual_list.append(float(row["bt_residual"]))
                bst_avg_list.append(float(row["bst_avg"]))
                all_feat_dicts.append(row["features"])
        y_residual = np.array(y_residual_list, dtype=np.float64)
        bst_avg = np.array(bst_avg_list, dtype=np.float64)
        feature_names = list(all_feat_dicts[0].keys())
        x_mat = np.array(
            [[fd.get(n, 0.0) for n in feature_names] for fd in all_feat_dicts],
            dtype=np.float64,
        )
        n_teams_actual = len(y_residual_list)
        bt_rho, bt_pval = 0.0, 1.0
        ols_r2 = 0.0
        profile = {}
        tier_counts = {"loaded": n_teams_actual}
        strat_result = {}
        jsonl_path = data_path
        bst_all_indices = list(range(28))
        no_bst_indices = list(range(28, len(feature_names)))
        print(f"[LOAD] Loaded {n_teams_actual} teams, features={len(feature_names)}")
        print("[LOAD] BT validation skipped (data loaded from file)")

    print(f"  OLS(BT ~ bst_avg) R2={ols_r2:.4f}")
    print(f"  Feature matrix: ({x_mat.shape[0]}, {x_mat.shape[1]})")

    print("RF Gini feature selection on no-BST features...")
    rf_gini = RandomForestRegressor(
        n_estimators=250, random_state=args.seed, n_jobs=-1,
    )
    x_no_bst = x_mat[:, no_bst_indices]
    no_bst_names = [feature_names[i] for i in no_bst_indices]
    rf_gini.fit(x_no_bst, y_residual)
    importances_no_bst = rf_gini.feature_importances_
    gini_threshold = 0.02
    top_k = min(12, len(no_bst_names))
    ranked = sorted(
        zip(no_bst_names, importances_no_bst, no_bst_indices, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    top_12_indices = [
        r[2] for r in ranked[:top_k] if r[1] >= gini_threshold
    ]
    if len(top_12_indices) < 3:
        top_12_indices = [r[2] for r in ranked[:top_k]]
    top_12_names = [feature_names[i] for i in top_12_indices]
    print(
        f"  Top {len(top_12_indices)} structural features: {top_12_names}"
    )

    rf_full = RandomForestRegressor(
        n_estimators=250, random_state=args.seed, n_jobs=-1,
    )
    rf_full.fit(x_mat, y_residual)
    importances_full = rf_full.feature_importances_

    print("Splitting 70/15/15 by team ID...")
    n_train = int(n_teams_actual * 0.70)
    n_test = int(n_teams_actual * 0.15)
    n_quarantine = n_teams_actual - n_train - n_test

    split_rng = np.random.default_rng(args.seed + 100)
    all_idx = list(range(n_teams_actual))
    split_rng.shuffle(all_idx)
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train : n_train + n_test]
    quarantine_idx = all_idx[n_train + n_test : n_train + n_test + n_quarantine]

    print(
        f"  Train: {len(train_idx)} teams, Test: {len(test_idx)}, "
        f"Quarantine: {len(quarantine_idx)}"
    )

    x_train_full = x_mat[train_idx]
    y_train_full = y_residual[train_idx]
    x_test_full = x_mat[test_idx]
    y_test_full = y_residual[test_idx]
    x_quarantine_full = x_mat[quarantine_idx]
    y_quarantine_full = y_residual[quarantine_idx]

    print("\n--- 3-Track Ablation (Ridge, 5-fold CV) ---")
    ab_results: dict[str, dict[str, Any]] = {}

    for track_label, col_idx in [
        ("track_a", [0]),
        ("track_b", top_12_indices),
        ("track_c", list(range(len(feature_names)))),
    ]:
        ab_results[track_label] = _ablation_track(
            x_train_full, y_train_full, col_idx,
            x_test_full, y_test_full,
            x_quarantine_full, y_quarantine_full,
            n_bootstrap=1000, seed=args.seed,
        )
        r = ab_results[track_label]
        print(
            f"  {track_label}: CV R2={r['cv_mean']:.4f}+-{r['cv_std']:.4f} "
            f"n={r['n_features']} "
            f"test_R2={r['r2_test']:.4f} quar_R2={r['r2_quarantine']:.4f}"
        )

    cv_a = ab_results["track_a"].get("cv_scores", [0.0])
    cv_b = ab_results["track_b"].get("cv_scores", [0.0])
    cv_c = ab_results["track_c"].get("cv_scores", [0.0])
    if not cv_a:
        cv_a = [ab_results["track_a"]["cv_mean"]]
    if not cv_b:
        cv_b = [ab_results["track_b"]["cv_mean"]]
    if not cv_c:
        cv_c = [ab_results["track_c"]["cv_mean"]]

    p_c_vs_a = _paired_bootstrap_pval(cv_a, cv_c, n_bootstrap=1000, seed=args.seed)
    p_c_vs_b = _paired_bootstrap_pval(cv_b, cv_c, n_bootstrap=1000, seed=args.seed)
    holm_reject = holm_bonferroni([p_c_vs_a, p_c_vs_b])
    print(f"  Paired bootstrap: C vs A p={p_c_vs_a:.4f} sig={holm_reject[0]}")
    print(f"  Paired bootstrap: C vs B p={p_c_vs_b:.4f} sig={holm_reject[1]}")

    print("\n--- Models on Full Features (Track C) ---")
    scaler_train = StandardScaler()
    x_tr_s = scaler_train.fit_transform(x_train_full)
    x_te_s = scaler_train.transform(x_test_full)
    x_qu_s = scaler_train.transform(x_quarantine_full)

    model_results: dict[str, Any] = {}

    lr_model = LinearRegression()
    model_results["linear_regression"] = _evaluate_model(
        lr_model, x_tr_s, y_train_full, x_te_s, y_test_full, x_qu_s, y_quarantine_full,
        n_bootstrap=1000, seed=args.seed,
    )
    mr = model_results["linear_regression"]
    print(f"  LinearRegression: test_R2={mr['r2_test']:.4f} quar_R2={mr['r2_quarantine']:.4f}")

    ridge_model = Ridge(alpha=1.0, random_state=args.seed)
    model_results["ridge"] = _evaluate_model(
        ridge_model, x_tr_s, y_train_full, x_te_s, y_test_full, x_qu_s, y_quarantine_full,
        n_bootstrap=1000, seed=args.seed,
    )
    mr = model_results["ridge"]
    print(f"  Ridge:            test_R2={mr['r2_test']:.4f} quar_R2={mr['r2_quarantine']:.4f}")

    lasso_model = Lasso(alpha=0.01, random_state=args.seed, max_iter=5000)
    model_results["lasso"] = _evaluate_model(
        lasso_model, x_tr_s, y_train_full, x_te_s, y_test_full, x_qu_s, y_quarantine_full,
        n_bootstrap=1000, seed=args.seed,
    )
    n_nonzero = int(np.sum(np.abs(lasso_model.coef_) > 1e-8))
    model_results["lasso"]["n_nonzero_coef"] = n_nonzero
    mr = model_results["lasso"]
    print(
        f"  Lasso:            test_R2={mr['r2_test']:.4f} "
        f"quar_R2={mr['r2_quarantine']:.4f} n_nonzero={n_nonzero}"
    )

    xgb_model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=args.seed, verbosity=0,
    )
    model_results["xgboost"] = _evaluate_model(
        xgb_model, x_tr_s, y_train_full, x_te_s, y_test_full, x_qu_s, y_quarantine_full,
        n_bootstrap=1000, seed=args.seed,
    )
    mr = model_results["xgboost"]
    print(f"  XGBoost:          test_R2={mr['r2_test']:.4f} quar_R2={mr['r2_quarantine']:.4f}")

    track_c_quarantine_r2 = ab_results["track_c"]["r2_quarantine"]
    viable = track_c_quarantine_r2 >= 0.10

    print("\n--- Verdict ---")
    if viable:
        print(
            f"  VIABLE: Track C Ridge R2_residual on quarantine = "
            f"{track_c_quarantine_r2:.4f} >= 0.10"
        )
    else:
        print(
            f"  NOT VIABLE: Track C Ridge R2_residual on quarantine = "
            f"{track_c_quarantine_r2:.4f} < 0.10"
        )

    elapsed = time.perf_counter() - start
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    metrics: dict[str, Any] = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_teams": n_teams_actual,
        "n_battles": args.n_battles,
        "n_rounds": n_rounds,
        "policy": policy_name,
        "generation_mode": args.generation_mode,
        "duration_seconds": round(elapsed, 1),
        "bt_validation": {"rho": float(bt_rho), "p_value": float(bt_pval)},
        "ols_bt_vs_bst_r2": ols_r2,
        "ablation": {
            "track_a": ab_results.get("track_a", {}),
            "track_b": ab_results.get("track_b", {}),
            "track_c": ab_results.get("track_c", {}),
        },
        "paired_bootstrap": {
            "c_vs_a_p": p_c_vs_a,
            "c_vs_b_p": p_c_vs_b,
            "holm_sig_c_vs_a": holm_reject[0],
            "holm_sig_c_vs_b": holm_reject[1],
        },
        "models": model_results,
        "viable": viable,
        "track_c_quarantine_r2": track_c_quarantine_r2,
    }

    metrics_path = output_dir / f"tqs_metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    meta: dict[str, Any] = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_teams": n_teams_actual,
        "n_battles": args.n_battles,
        "n_rounds": n_rounds,
        "policy": policy_name,
        "data_profile": profile,
        "tier_counts": tier_counts,
        "stratification": strat_result,
        "feature_names": feature_names,
        "bst_all_indices": bst_all_indices,
        "no_bst_indices": no_bst_indices,
        "top_12_features": top_12_names,
        "top_12_indices": top_12_indices,
        "train_teams": len(train_idx),
        "test_teams": len(test_idx),
        "quarantine_teams": len(quarantine_idx),
    }
    meta_path = output_dir / f"tqs_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    gini_data: dict[str, Any] = {
        "timestamp": timestamp,
        "no_bst_features": [
            {"name": no_bst_names[i], "importance": float(importances_no_bst[i])}
            for i in range(len(no_bst_names))
        ],
        "full_features": [
            {"name": feature_names[i], "importance": float(importances_full[i])}
            for i in range(len(feature_names))
        ],
        "top_k": top_k,
        "threshold": gini_threshold,
    }
    gini_path = output_dir / f"tqs_gini_{timestamp}.json"
    with open(gini_path, "w") as f:
        json.dump(gini_data, f, indent=2)

    print(f"\n  Metrics saved to {metrics_path}")
    print(f"  Metadata saved to {meta_path}")
    print(f"  Gini importance saved to {gini_path}")
    print(f"  Data saved to {jsonl_path}")

    print("Saving all models...")
    scaler_save = StandardScaler()
    x_all_s = scaler_save.fit_transform(x_mat)

    def _save_tqs_model(model_obj: Any, name: str) -> None:
        path = output_dir / f"TQS_{name}.pkl"
        with open(path, "wb") as bf:
            pickle.dump(
                {
                    "model": model_obj,
                    "scaler": scaler_save,
                    "feature_names": feature_names,
                    "config": {"policy": policy_name, "seed": args.seed},
                },
                bf,
            )
        print(f"  Saved {path}")

    ridge_save = Ridge(alpha=1.0, random_state=args.seed)
    ridge_save.fit(x_all_s, y_residual)
    _save_tqs_model(ridge_save, "Ridge")

    xgb_save = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=args.seed, verbosity=0,
    )
    xgb_save.fit(x_all_s, y_residual)
    _save_tqs_model(xgb_save, "XGBoost")

    lasso_save = Lasso(alpha=0.01, random_state=args.seed, max_iter=5000)
    lasso_save.fit(x_all_s, y_residual)
    _save_tqs_model(lasso_save, "Lasso")

    lr_save = LinearRegression()
    lr_save.fit(x_all_s, y_residual)
    _save_tqs_model(lr_save, "LinearReg")

    print("=" * 60)
    print(f"Done in {elapsed:.0f}s. Viable: {viable}")


if __name__ == "__main__":
    main()
