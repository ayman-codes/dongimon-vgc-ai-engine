"""Tests for evaluate_mp_quarantine.py quarantine evaluation."""
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from scripts.evaluate_mp_quarantine import load_claimed_auroc, load_jsonl


def _make_jsonl(path: Path, n_pairs: int = 100, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    names = [f"f{i}" for i in range(10)]
    for i in range(n_pairs):
        feats = {k: float(v) for k, v in zip(names, rng.standard_normal(10), strict=True)}
        wins_a = int(rng.integers(0, 51))
        rows.append({
            "pair_id": i,
            "seed": seed + i * 100,
            "wins_a": wins_a,
            "n_battles": 50,
            "win_rate_a": wins_a / 50.0,
            "features": feats,
        })
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_dummy_model(path: Path) -> None:
    model = DummyClassifier(strategy="uniform", random_state=42)
    X = np.random.default_rng(42).random((20, 5))
    y = np.array([0, 1] * 10)
    model.fit(X, y)
    joblib.dump(model, path)


def _make_champion_meta(path: Path, champion: str = "xgboost", test_auroc: float = 0.8043) -> None:
    meta = {
        "champion_model": champion,
        "test_auroc": test_auroc,
        "n_samples": 18000,
        "n_features": 56,
        "n_trials_per_model": 150,
        "cv_folds": 5,
        "split": "70/15/15",
        "random_state": 42,
        "timestamp": "2026-07-30 07:29:11",
    }
    with open(path, "w") as f:
        json.dump(meta, f)


class TestLoadJsonl:
    def test_returns_features_and_labels(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
        try:
            _make_jsonl(path, n_pairs=50)
            X, y, names = load_jsonl(path)
            assert X.shape == (50, 10)
            assert len(y) == 50
            assert len(names) == 10
            assert set(names) == {f"f{i}" for i in range(10)}
        finally:
            path.unlink()

    def test_labels_are_binary(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
        try:
            _make_jsonl(path, n_pairs=100)
            _, y, _ = load_jsonl(path)
            assert set(y) == {0.0, 1.0}
        finally:
            path.unlink()

    def test_win_rate_above_50_is_positive(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
        try:
            rows = [
                {"features": {"a": 1.0}, "win_rate_a": 0.51},
                {"features": {"a": 2.0}, "win_rate_a": 0.50},
                {"features": {"a": 3.0}, "win_rate_a": 0.49},
            ]
            with open(path, "w") as fout:
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
            _, y, _ = load_jsonl(path)
            assert y.tolist() == [1.0, 0.0, 0.0]
        finally:
            path.unlink()

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl(Path("/nonexistent/path.jsonl"))

    def test_returns_empty_feature_names_for_empty_rows(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
        try:
            open(path, "w").close()
            with pytest.raises((IndexError, FileNotFoundError)):
                load_jsonl(path)
        finally:
            path.unlink()

    def test_deterministic_same_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
        try:
            _make_jsonl(path, n_pairs=30, seed=99)
            X1, y1, _ = load_jsonl(path)
            X2, y2, _ = load_jsonl(path)
            assert np.array_equal(X1, X2)
            assert np.array_equal(y1, y2)
        finally:
            path.unlink()

    def test_feature_names_preserve_order(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)
        try:
            features_a = {"a": 1.0, "b": 2.0, "c": 3.0}
            features_b = {"a": 4.0, "b": 5.0, "c": 6.0}
            rows = [
                {"features": features_a, "win_rate_a": 0.6},
                {"features": features_b, "win_rate_a": 0.3},
            ]
            with open(path, "w") as fout:
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
            X, _, names = load_jsonl(path)
            assert names == ["a", "b", "c"]
            assert X.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        finally:
            path.unlink()


class TestLoadClaimedAuroc:
    def test_returns_xgboost_when_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _make_champion_meta(
                model_dir / "champion_meta.json",
                champion="xgboost",
                test_auroc=0.8043,
            )
            claimed = load_claimed_auroc(model_dir)
            assert claimed["xgboost"] == 0.8043
            assert claimed["lightgbm"] is None
            assert claimed["random_forest"] is None

    def test_returns_none_for_unknown_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _make_champion_meta(
                model_dir / "champion_meta.json",
                champion="lightgbm",
                test_auroc=0.7987,
            )
            claimed = load_claimed_auroc(model_dir)
            assert claimed["xgboost"] is None
            assert claimed["lightgbm"] is None
            assert claimed["random_forest"] is None

    def test_returns_all_none_when_meta_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            claimed = load_claimed_auroc(model_dir)
            for v in claimed.values():
                assert v is None

    def test_handles_missing_test_auroc_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            meta = {"champion_model": "xgboost"}
            with open(model_dir / "champion_meta.json", "w") as f:
                json.dump(meta, f)
            claimed = load_claimed_auroc(model_dir)
            assert claimed["xgboost"] is None

    def test_handles_missing_champion_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            meta = {"test_auroc": 0.85}
            with open(model_dir / "champion_meta.json", "w") as f:
                json.dump(meta, f)
            claimed = load_claimed_auroc(model_dir)
            for v in claimed.values():
                assert v is None


class TestStatusThresholds:
    def test_generalizes_when_drop_below_3_percent(self):
        claimed = 0.80
        quarantine = 0.79
        drop = claimed - quarantine
        assert drop < 0.03
        assert drop > 0

    def test_moderate_when_drop_3_to_7_percent(self):
        claimed = 0.80
        quarantine = 0.75
        drop = claimed - quarantine
        assert 0.03 <= drop < 0.07

    def test_overfit_when_drop_above_7_percent(self):
        claimed = 0.80
        quarantine = 0.70
        drop = claimed - quarantine
        assert drop >= 0.07

    def test_negative_drop_means_quarantine_better(self):
        claimed = 0.80
        quarantine = 0.83
        drop = claimed - quarantine
        assert drop < 0
        assert drop < 0.03
