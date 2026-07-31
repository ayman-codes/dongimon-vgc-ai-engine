"""BC model training with fixed Optuna-tuned hyperparameters.

Trains an XGBoost multi-class classifier on behavioral cloning data
using the best hyperparameters found by Optuna (trial #4, val accuracy
0.5810). Evaluates via stratified k-fold cross-validation and a
separately-generated holdout set to prevent data leakage.

Usage:
    uv run python -m src.tree_bc.train \
        --data-dir=data/BC --holdout-dir=data/BC_holdout --output-dir=src/models
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tree_bc.encoder import FEATURE_DIM

EXPERIMENT_NAME = "bc_tree_training"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
N_CLASSES = 100
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 30
CV_FOLDS = 5

BEST_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.2561025638713225,
    "n_estimators": 600,
    "subsample": 0.6973875333307435,
    "colsample_bytree": 0.40894110290282437,
    "min_child_weight": 11,
    "reg_alpha": 2.0949430946634093,
    "reg_lambda": 4.162828947124134,
    "objective": "multi:softprob",
    "random_state": RANDOM_STATE,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "tree_method": "hist",
}

EXPERT_SAMPLE_WEIGHTS: dict[str, float] = {
    "Greedy": 1.0,
    "caaaden": 0.75,
    "minimon": 0.5,
    "JJJ": 0.25,
    "dongimon": 0.15,
}


def load_bc_data(
    data_dir: Path, win_filter: bool = True
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float32],
]:
    """Load BC JSONL data into arrays for training.

    Args:
        data_dir: Directory containing bc_data_*.jsonl files.
        win_filter: If True, only keep records where won=True.

    Returns:
        Tuple of (features, actions, valid_masks, sample_weights).
        features: shape (N, FEATURE_DIM) float32.
        actions: shape (N,) int32, values 0-99.
        valid_masks: shape (N, N_CLASSES) bool.
        sample_weights: shape (N,) float32.

    Raises:
        FileNotFoundError: If no JSONL files found in data_dir.
    """
    jsonl_files = sorted(data_dir.glob("bc_data_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No bc_data_*.jsonl files in {data_dir}")

    features_list: list[list[float]] = []
    actions_list: list[int] = []
    masks_list: list[list[bool]] = []
    weights_list: list[float] = []

    for jf in jsonl_files:
        with open(jf) as f:
            for line in f:
                row = json.loads(line)
                if win_filter and not row.get("won", True):
                    continue
                features_list.append(row["features"])
                actions_list.append(row["action_idx"])
                valid_indices = row["valid_actions"]
                mask = [False] * N_CLASSES
                for idx in valid_indices:
                    mask[idx] = True
                masks_list.append(mask)
                expert = row.get("expert", "JJJ")
                weights_list.append(EXPERT_SAMPLE_WEIGHTS.get(expert, 0.5))

    features = np.array(features_list, dtype=np.float32)
    actions = np.array(actions_list, dtype=np.int32)
    valid_masks = np.array(masks_list, dtype=bool)
    sample_weights = np.array(weights_list, dtype=np.float32)

    return features, actions, valid_masks, sample_weights


def _remap_labels(
    actions: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Remap sparse action labels to consecutive 0..N-1 for XGBoost.

    Args:
        actions: Original action labels (may have gaps).

    Returns:
        Tuple of (remapped_actions, inverse_map) where inverse_map[i]
        gives the original action index for remapped class i.
    """
    unique_sorted = np.unique(actions)
    inverse_map = unique_sorted.astype(np.int32)
    mapping: npt.NDArray[np.int32] = np.full(N_CLASSES, -1, dtype=np.int32)
    for new_idx, orig_idx in enumerate(unique_sorted):
        mapping[orig_idx] = new_idx
    remapped = mapping[actions]
    return remapped, inverse_map


def _remap_masks(
    valid_masks: npt.NDArray[np.bool_],
    inverse_map: npt.NDArray[np.int32],
) -> npt.NDArray[np.bool_]:
    """Remap valid-action masks to the consecutive class space.

    Args:
        valid_masks: Original masks, shape (N, N_CLASSES).
        inverse_map: Mapping from new class index to original class index.

    Returns:
        Remapped masks, shape (N, len(inverse_map)).
    """
    n_present = len(inverse_map)
    remapped: npt.NDArray[np.bool_] = np.zeros(
        (valid_masks.shape[0], n_present), dtype=bool
    )
    for new_idx in range(n_present):
        orig_idx = inverse_map[new_idx]
        remapped[:, new_idx] = valid_masks[:, orig_idx]
    return remapped


def valid_action_accuracy(
    y_true: npt.NDArray[np.int32],
    y_proba: npt.NDArray[np.float64],
    valid_masks: npt.NDArray[np.bool_],
) -> float:
    """Compute top-1 accuracy restricted to valid actions.

    Zeroes out probability mass on invalid actions before taking argmax.

    Args:
        y_true: True action indices, shape (N,).
        y_proba: Predicted probabilities, shape (N, C).
        valid_masks: Boolean masks, shape (N, C).

    Returns:
        Fraction of samples where masked argmax matches true action.
    """
    masked_proba = y_proba.copy()
    masked_proba[~valid_masks] = 0.0
    y_pred = masked_proba.argmax(axis=1)
    return float((y_pred == y_true).mean())


def _filter_rare_classes(
    features: npt.NDArray[np.float32],
    actions: npt.NDArray[np.int32],
    valid_masks: npt.NDArray[np.bool_],
    sample_weights: npt.NDArray[np.float32],
    min_count: int = 3,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float32],
    set[int],
]:
    """Remove samples belonging to action classes with fewer than min_count.

    Args:
        features: Feature matrix, shape (N, D).
        actions: Action labels, shape (N,).
        valid_masks: Valid action masks, shape (N, C).
        sample_weights: Per-sample weights, shape (N,).
        min_count: Minimum samples required per class.

    Returns:
        Tuple of (features, actions, valid_masks, sample_weights, rare_classes).
    """
    unique, counts = np.unique(actions, return_counts=True)
    rare_classes = set(unique[counts < min_count].tolist())
    if not rare_classes:
        return features, actions, valid_masks, sample_weights, rare_classes
    keep = np.array([a not in rare_classes for a in actions])
    return (
        features[keep],
        actions[keep],
        valid_masks[keep],
        sample_weights[keep],
        rare_classes,
    )


def train(
    data_dir: Path,
    holdout_dir: Path,
    output_dir: Path,
    cv_folds: int,
    win_filter: bool,
) -> None:
    """Execute the fixed-hyperparameter BC training pipeline.

    Loads training and holdout data from separate directories (different
    seeds to prevent leakage), runs stratified k-fold CV on training
    data, trains final model on all training data with early stopping
    on holdout, evaluates, logs to MLflow, and saves model via joblib.

    Args:
        data_dir: Directory containing training bc_data_*.jsonl files.
        holdout_dir: Directory containing holdout bc_data_*.jsonl files.
        output_dir: Directory for model artifacts and MLflow db.
        cv_folds: Number of stratified CV folds.
        win_filter: If True, only train on winning records.
    """
    print("=" * 60)
    print("BC Tree Model Training (XGBoost, fixed hyperparams)")
    print(f"  data_dir={data_dir}")
    print(f"  holdout_dir={holdout_dir}")
    print(f"  cv_folds={cv_folds}")
    print(f"  win_filter={win_filter}")
    print(f"  early_stopping_rounds={EARLY_STOPPING_ROUNDS}")
    print("=" * 60)

    tr_feat, tr_act, tr_mask, tr_wt = load_bc_data(data_dir, win_filter=win_filter)
    ho_feat, ho_act, ho_mask, ho_wt = load_bc_data(holdout_dir, win_filter=win_filter)

    print(f"\nTraining data:   {len(tr_act)} samples, {tr_feat.shape[1]} features")
    print(f"  Unique actions: {len(np.unique(tr_act))}/{N_CLASSES}")
    print(f"Holdout data:    {len(ho_act)} samples")
    print(f"  Unique actions: {len(np.unique(ho_act))}/{N_CLASSES}")

    tr_feat, tr_act, tr_mask, tr_wt, rare_tr = _filter_rare_classes(
        tr_feat, tr_act, tr_mask, tr_wt
    )
    ho_feat, ho_act, ho_mask, ho_wt, rare_ho = _filter_rare_classes(
        ho_feat, ho_act, ho_mask, ho_wt
    )
    if rare_tr or rare_ho:
        print(f"  Dropped rare classes — train: {sorted(rare_tr)}, holdout: {sorted(rare_ho)}")

    all_actions = np.concatenate([tr_act, ho_act])
    _, inverse_map = _remap_labels(all_actions)
    n_present = len(inverse_map)

    tr_act_remapped, _ = _remap_labels(tr_act)
    ho_act_remapped, _ = _remap_labels(ho_act)

    tr_mask = _remap_masks(tr_mask, inverse_map)
    ho_mask = _remap_masks(ho_mask, inverse_map)

    print(f"  Remapped to {n_present} consecutive classes")
    print(f"  Train: {len(tr_act_remapped)} | Holdout: {len(ho_act_remapped)}")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_accs: list[float] = []

    print(f"\nRunning {cv_folds}-fold stratified CV...")
    t0 = time.perf_counter()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(tr_feat, tr_act_remapped)):
        X_tr, X_val = tr_feat[train_idx], tr_feat[val_idx]
        y_tr, y_val = tr_act_remapped[train_idx], tr_act_remapped[val_idx]
        m_val = tr_mask[val_idx]
        w_tr = tr_wt[train_idx]

        model = XGBClassifier(**BEST_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        proba = model.predict_proba(X_val).astype(np.float64)
        fold_acc = valid_action_accuracy(y_val, proba, m_val)
        cv_accs.append(fold_acc)
        print(f"  Fold {fold_idx + 1}/{cv_folds}: {fold_acc:.4f}")

    cv_time = time.perf_counter() - t0
    cv_mean = float(np.mean(cv_accs))
    cv_std = float(np.std(cv_accs))
    print(f"\n  CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f}  ({cv_time:.1f}s)")

    print("\nTraining final model on all training data...")
    t1 = time.perf_counter()

    final_params = dict(BEST_PARAMS)
    final_model = XGBClassifier(**final_params)
    final_model.fit(
        tr_feat, tr_act_remapped,
        sample_weight=tr_wt,
        eval_set=[(ho_feat, ho_act_remapped)],
        verbose=False,
    )

    train_time = time.perf_counter() - t1

    ho_proba = _full_proba(final_model, ho_feat, n_present)
    holdout_acc = valid_action_accuracy(ho_act_remapped, ho_proba, ho_mask)

    tr_proba = _full_proba(final_model, tr_feat, n_present)
    train_acc = valid_action_accuracy(tr_act_remapped, tr_proba, tr_mask)

    print(f"  Train accuracy (valid-masked):   {train_acc:.4f}")
    print(f"  Holdout accuracy (valid-masked): {holdout_acc:.4f}")
    print(f"  Final training time: {train_time:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(output_dir / 'mlflow_bc.db').resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"bc_tree_{time.strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in BEST_PARAMS.items() if k != "verbosity"})
        mlflow.log_metric("cv_accuracy_mean", cv_mean)
        mlflow.log_metric("cv_accuracy_std", cv_std)
        mlflow.log_metric("holdout_accuracy", holdout_acc)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("n_train_samples", len(tr_act_remapped))
        mlflow.log_metric("n_holdout_samples", len(ho_act_remapped))
        mlflow.log_metric("n_classes", n_present)
        mlflow.log_metric("cv_folds", cv_folds)
        mlflow.log_metric("cv_time_s", round(cv_time, 1))
        mlflow.log_metric("train_time_s", round(train_time, 1))

        model_path = output_dir / "bc_xgboost_model.joblib"
        joblib.dump({"model": final_model, "inverse_map": inverse_map}, model_path)
        mlflow.log_artifact(str(model_path))

    meta = {
        "model_type": "xgboost_multiclass_fixed_params",
        "n_classes": N_CLASSES,
        "n_present_classes": n_present,
        "feature_dim": FEATURE_DIM,
        "holdout_accuracy": holdout_acc,
        "train_accuracy": train_acc,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "cv_folds": cv_folds,
        "cv_fold_accuracies": cv_accs,
        "best_params": {k: v for k, v in BEST_PARAMS.items() if isinstance(v, (int, float, str))},
        "n_train_samples": int(len(tr_act_remapped)),
        "n_holdout_samples": int(len(ho_act_remapped)),
        "win_filter": win_filter,
        "expert_weights": EXPERT_SAMPLE_WEIGHTS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = output_dir / "bc_champion_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved: {model_path}")
    print(f"  Metadata saved: {meta_path}")
    print(f"  MLflow run: {run_name}")
    print("=" * 60)


def _full_proba(
    model: XGBClassifier,
    X: npt.NDArray[np.float32],
    n_classes: int,
) -> npt.NDArray[np.float64]:
    """Get predict_proba padded to full n_classes width.

    XGBClassifier.predict_proba only returns columns for classes seen
    during fit. This maps them back to a (N, n_classes) array using
    the model's learned classes_ attribute.

    Args:
        model: Fitted XGBClassifier.
        X: Feature matrix.
        n_classes: Total number of consecutive classes.

    Returns:
        Probability array of shape (N, n_classes).
    """
    raw = model.predict_proba(X)
    if raw.shape[1] == n_classes:
        return raw.astype(np.float64)
    full = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
    classes: npt.NDArray[np.int32] = model.classes_.astype(np.int32)
    full[:, classes] = raw
    return full


def main() -> None:
    """CLI entry point for fixed-hyperparameter BC model training."""
    parser = argparse.ArgumentParser(
        description="Train BC model with fixed Optuna-tuned hyperparameters"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/BC"),
        help="Directory containing training bc_data_*.jsonl files",
    )
    parser.add_argument(
        "--holdout-dir", type=Path, default=Path("data/BC_holdout"),
        help="Directory containing holdout bc_data_*.jsonl files (different seed)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODEL_DIR,
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=CV_FOLDS,
        help=f"Number of stratified CV folds (default: {CV_FOLDS})",
    )
    parser.add_argument(
        "--no-win-filter", action="store_true",
        help="Include losing records in training",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        holdout_dir=args.holdout_dir,
        output_dir=args.output_dir,
        cv_folds=args.cv_folds,
        win_filter=not args.no_win_filter,
    )


if __name__ == "__main__":
    main()
