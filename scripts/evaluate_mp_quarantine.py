"""Evaluate MP models on quarantine data (unseen species, different seed).

Loads pre-trained models (XGBoost, LightGBM, Random Forest) and evaluates
AUROC against data generated from a different seed — species never seen
during training. Reports quarantine AUROC vs claimed test AUROC.

Usage:
    uv run python scripts/evaluate_mp_quarantine.py \
        --data=data/MP/quarantine/mp_data_20260730_130816.jsonl \
        --model-dir=src/models/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.s3 import sync_from_s3

RANDOM_STATE = 42


def load_jsonl(path: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    """Load JSONL into features and binary labels.

    Args:
        path: Path to mp_data_*.jsonl file.

    Returns:
        Tuple of (X, y, feature_names).
    """
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    feature_names = list(rows[0]["features"].keys())
    X = np.array([[row["features"][k] for k in feature_names] for row in rows])
    y = np.array([1.0 if row["win_rate_a"] > 0.5 else 0.0 for row in rows])
    return X, y, feature_names


def load_claimed_auroc(model_dir: Path) -> dict[str, float | None]:
    """Read champion_meta.json for claimed test AUROC values.

    Args:
        model_dir: Directory containing champion_meta.json and model files.

    Returns:
        Dict with claimed AUROC per model (from train.py champion tracking).
    """
    meta_path = model_dir / "champion_meta.json"
    claimed: dict[str, float | None] = {
        "xgboost": None,
        "lightgbm": None,
        "random_forest": None,
    }
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        champion = meta.get("champion_model", "")
        test_auroc = meta.get("test_auroc")
        if champion == "xgboost" and test_auroc is not None:
            claimed["xgboost"] = float(test_auroc)
    return claimed


def main() -> None:
    """Load models and evaluate against quarantine data."""
    parser = argparse.ArgumentParser(
        description="Evaluate MP models against unseen quarantine data"
    )
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Path to quarantine JSONL data file",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("src/models"),
        help="Directory containing {name}_model.joblib files",
    )
    parser.add_argument(
        "--s3-bucket", type=str, default="",
        help="S3 bucket to sync data and models from (skipped if empty)",
    )
    parser.add_argument(
        "--s3-prefix", type=str, default="data_MP/",
        help="S3 key prefix to sync into data/MP (default: data_MP/)",
    )
    parser.add_argument(
        "--s3-model-prefix", type=str, default="models/",
        help="S3 key prefix to sync into --model-dir (default: models/)",
    )
    args = parser.parse_args()

    if args.s3_bucket:
        n_data = sync_from_s3(Path("data/MP"), args.s3_prefix, args.s3_bucket)
        print(f"Synced {n_data} files from s3://{args.s3_bucket}/{args.s3_prefix} to data/MP")
        n_models = sync_from_s3(args.model_dir, args.s3_model_prefix, args.s3_bucket)
        print(f"Synced {n_models} files from s3://{args.s3_bucket}/{args.s3_model_prefix} to {args.model_dir}")

    print("=" * 60)
    print("MP Model — Quarantine Generalization Test")
    print(f"  quarantine data: {args.data}")
    print(f"  model dir:       {args.model_dir}")
    print("=" * 60)

    X, y, feature_names = load_jsonl(args.data)
    n_pairs = len(y)
    print(f"\nLoaded {n_pairs} quarantine pairs")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive rate: {y.mean():.4f}")

    claimed = load_claimed_auroc(args.model_dir)

    model_configs = [
        ("XGBoost (champion)", "xgboost_model.joblib", claimed.get("xgboost")),
        ("LightGBM", "lightgbm_model.joblib", claimed.get("lightgbm")),
        ("Random Forest", "random_forest_model.joblib", claimed.get("random_forest")),
    ]

    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'Claimed Test':>12} {'Quarantine':>12} {'Drop':>10} {'Status'}")
    print(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10}")

    results: list[dict[str, Any]] = []

    for display_name, filename, claimed_auroc in model_configs:
        model_path = args.model_dir / filename
        if not model_path.exists():
            print(f"{display_name:<25} {'—':>12} {'—':>12} {'—':>10} NOT FOUND")
            continue

        model = joblib.load(model_path)
        probs = model.predict_proba(X)[:, 1]
        quarantine_auroc = float(roc_auc_score(y, probs))

        claimed_str = f"{claimed_auroc:.4f}" if claimed_auroc else "—"
        drop = claimed_auroc - quarantine_auroc if claimed_auroc else None
        drop_str = f"{drop:+.4f}" if drop is not None else "—"

        if claimed_auroc and drop is not None:
            if drop < 0.03:
                status = "GENERALIZES"
            elif drop < 0.07:
                status = "MODERATE"
            else:
                status = "OVERFIT"
        else:
            status = "—"

        print(f"{display_name:<25} {claimed_str:>12} {quarantine_auroc:>12.4f} "
              f"{drop_str:>10} {status}")

        results.append({
            "model": display_name,
            "claimed_test_auroc": claimed_auroc,
            "quarantine_auroc": quarantine_auroc,
            "drop": drop,
            "status": status,
        })

    print(f"{'=' * 60}")
    print(f"\nQuarantine data: {n_pairs} pairs from seed=43")
    print("Training data:   24,000 pairs from seed=42")
    print("Different species, types, moves, and base stats between seeds.")
    print("Quarantine AUROC = model's ability to generalize to unseen species.")

    if results:
        champion = next((r for r in results if r.get("claimed_test_auroc")), results[0])
        print("\nKey question answered:")
        print("  Champion model (from seed=42 data):")
        print(f"    Claimed test AUROC:  {champion['claimed_test_auroc']:.4f}")
        print(f"    Quarantine AUROC:    {champion['quarantine_auroc']:.4f}")
        if champion["drop"] is not None:
            print(f"    Drop:                {champion['drop']:+.4f}")
        print(f"    Verdict:             {champion['status']}")

    out_path = args.model_dir / "quarantine_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "quarantine_data_file": str(args.data),
            "n_quarantine_pairs": n_pairs,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
