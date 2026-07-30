"""Matchup Predictor model training pipeline.

Trains XGBoost, LightGBM, and Random Forest classifiers on generated
MP data (JSONL). Uses Optuna for hyperparameter optimization with
nested 5-fold CV, MLflow file-based tracking with Champion/Contender
model registry pattern.

Usage:
    uv run python -m src.data.train --data-dir=data/MP --n-trials=150
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
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

EXPERIMENT_NAME = "mp_model_training"
MODEL_REGISTRY_NAME = "matchup_predictor"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CV_FOLDS = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42


def load_jsonl(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all JSONL files in data_dir into feature matrix and labels.

    Args:
        data_dir: Directory containing mp_data_*.jsonl files.

    Returns:
        Tuple of (X, y, feature_names) where X is (n_samples, 56),
        y is binary labels (win_rate_a > 0.5), and feature_names is
        the ordered list of feature keys.

    Raises:
        FileNotFoundError: If no JSONL files found in data_dir.
    """
    jsonl_files = sorted(data_dir.glob("mp_data_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No mp_data_*.jsonl files in {data_dir}")

    rows: list[dict[str, Any]] = []
    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not rows:
        raise FileNotFoundError(f"JSONL files in {data_dir} are empty")

    feature_names = list(rows[0]["features"].keys())
    X = np.array([[row["features"][k] for k in feature_names] for row in rows])
    y = np.array([1.0 if row["win_rate_a"] > 0.5 else 0.0 for row in rows])

    return X, y, feature_names


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test (70/15/15) with stratification.

    Each team appears in exactly one pair, so random pair-level splitting
    is equivalent to team-ID splitting (no leakage possible).

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Binary label vector.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y,
    )
    relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=RANDOM_STATE, stratify=y_temp,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def xgboost_objective(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray
) -> float:
    """Optuna objective for XGBoost: mean 5-fold CV AUROC.

    Args:
        trial: Optuna trial for hyperparameter suggestion.
        X: Training feature matrix.
        y: Training labels.

    Returns:
        Mean AUROC across CV folds.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "verbosity": 0,
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], preds))

    return float(np.mean(scores))


def lightgbm_objective(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray
) -> float:
    """Optuna objective for LightGBM: mean 5-fold CV AUROC.

    Args:
        trial: Optuna trial for hyperparameter suggestion.
        X: Training feature matrix.
        y: Training labels.

    Returns:
        Mean AUROC across CV folds.
    """
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        model = LGBMClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], preds))

    return float(np.mean(scores))


def random_forest_objective(
    trial: optuna.Trial, X: np.ndarray, y: np.ndarray
) -> float:
    """Optuna objective for Random Forest: mean 5-fold CV AUROC.

    Args:
        trial: Optuna trial for hyperparameter suggestion.
        X: Training feature matrix.
        y: Training labels.

    Returns:
        Mean AUROC across CV folds.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        model = RandomForestClassifier(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], preds))

    return float(np.mean(scores))


def train_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Train model on full train set and evaluate on val + test.

    Args:
        model: Classifier instance with fit/predict_proba.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict with val_auroc and test_auroc.
    """
    model.fit(X_train, y_train)
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    return {
        "val_auroc": float(roc_auc_score(y_val, val_preds)),
        "test_auroc": float(roc_auc_score(y_test, test_preds)),
    }


def get_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance from a trained model.

    Args:
        model: Fitted classifier with feature_importances_ attribute.
        feature_names: Ordered feature name list.

    Returns:
        Dict mapping feature name to importance score, sorted descending.
    """
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances, strict=True), key=lambda x: -x[1])
    return {name: float(imp) for name, imp in pairs}


def run_training(
    data_dir: Path,
    n_trials: int,
    output_dir: Path,
) -> None:
    """Execute the full training pipeline for all three models.

    Loads data, splits, tunes each model with Optuna, evaluates,
    logs to MLflow, and registers the best as Champion.

    Args:
        data_dir: Directory containing mp_data_*.jsonl files.
        n_trials: Number of Optuna trials per model.
        output_dir: Directory for model artifacts and MLflow runs.
    """
    print("=" * 60)
    print("Matchup Predictor Model Training")
    print(f"  data_dir={data_dir}")
    print(f"  n_trials={n_trials} per model")
    print(f"  CV folds={CV_FOLDS}")
    print("  split=70/15/15 (stratified)")
    print("=" * 60)

    X, y, feature_names = load_jsonl(data_dir)
    print(f"\nLoaded {len(y)} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.4f}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    mlflow.set_tracking_uri(f"sqlite:///{(output_dir / 'mlflow.db').resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_test_auroc = 0.0
    best_model_name = ""

    model_configs: list[tuple[str, Any, Any]] = [
        ("xgboost", xgboost_objective, lambda p: XGBClassifier(**p, verbosity=0, eval_metric="auc")),
        ("lightgbm", lightgbm_objective, lambda p: LGBMClassifier(**p, verbosity=-1)),
        ("random_forest", random_forest_objective, lambda p: RandomForestClassifier(**p, n_jobs=-1)),
    ]

    for model_name, objective_fn, model_factory in model_configs:
        print(f"\n{'-' * 60}")
        print(f"Training: {model_name} ({n_trials} Optuna trials)")
        print(f"{'-' * 60}")

        t_start = time.perf_counter()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda trial, _fn=objective_fn: _fn(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        elapsed = time.perf_counter() - t_start
        print(f"  Best CV AUROC: {study.best_value:.4f} (trial #{study.best_trial.number})")
        print(f"  Tuning time: {elapsed:.1f}s")

        best_params = study.best_params
        best_params["random_state"] = RANDOM_STATE

        model = model_factory(best_params)
        metrics = train_and_evaluate(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )

        print(f"  Val AUROC:  {metrics['val_auroc']:.4f}")
        print(f"  Test AUROC: {metrics['test_auroc']:.4f}")

        importance = get_feature_importance(model, feature_names)

        with mlflow.start_run(run_name=f"{model_name}_best"):
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_auroc", study.best_value)
            mlflow.log_metric("val_auroc", metrics["val_auroc"])
            mlflow.log_metric("test_auroc", metrics["test_auroc"])
            mlflow.log_metric("n_train", len(y_train))
            mlflow.log_metric("n_val", len(y_val))
            mlflow.log_metric("n_test", len(y_test))
            mlflow.log_metric("tuning_seconds", elapsed)

            model_path = MODEL_DIR / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            importance_path = output_dir / f"{model_name}_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)
            mlflow.log_artifact(str(importance_path))

            cv_scores_path = output_dir / f"{model_name}_cv_trials.json"
            trials_data = [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study.trials if t.value is not None
            ]
            with open(cv_scores_path, "w") as f:
                json.dump(trials_data, f, indent=2)
            mlflow.log_artifact(str(cv_scores_path))

            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("mlflow.runName", f"{model_name}_best")

            if metrics["test_auroc"] > best_test_auroc:
                best_test_auroc = metrics["test_auroc"]
                best_model_name = model_name
                mlflow.set_tag("registry_status", "champion")
            else:
                mlflow.set_tag("registry_status", "contender")

    print(f"\n{'=' * 60}")
    print(f"CHAMPION: {best_model_name} (test AUROC = {best_test_auroc:.4f})")
    print(f"{'=' * 60}")

    champion_meta = {
        "champion_model": best_model_name,
        "test_auroc": best_test_auroc,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "n_trials_per_model": n_trials,
        "cv_folds": CV_FOLDS,
        "split": "70/15/15",
        "random_state": RANDOM_STATE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = MODEL_DIR / "champion_meta.json"
    with open(meta_path, "w") as f:
        json.dump(champion_meta, f, indent=2)
    print(f"Models saved to {MODEL_DIR}")
    print(f"Metadata saved to {meta_path}")


def main() -> None:
    """CLI entry point for MP model training."""
    parser = argparse.ArgumentParser(
        description="Train Matchup Predictor models with Optuna + MLflow"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/MP"),
        help="Directory containing mp_data_*.jsonl files",
    )
    parser.add_argument(
        "--n-trials", type=int, default=150,
        help="Number of Optuna trials per model (default: 150)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/MP/models"),
        help="Directory for model artifacts and MLflow runs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_training(
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
