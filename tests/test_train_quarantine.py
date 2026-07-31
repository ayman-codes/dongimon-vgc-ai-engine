"""Smoke tests for train.py model training pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.train import get_feature_importance, load_jsonl, split_data


def _make_jsonl(path: Path, n_pairs: int = 50, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(10)]
    rows = []
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


class TestLoadJsonl:
    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            path = data_dir / "mp_data_test.jsonl"
            _make_jsonl(path, n_pairs=30)
            X, y, names = load_jsonl(data_dir)
            assert X.shape == (30, 10)
            assert len(y) == 30

    def test_labels_are_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            path = data_dir / "mp_data_test.jsonl"
            _make_jsonl(path, n_pairs=100)
            _, y, _ = load_jsonl(data_dir)
            assert set(y) == {0.0, 1.0}

    def test_win_rate_above_50_is_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            path = data_dir / "mp_data_test.jsonl"
            rows = [
                {"features": {"a": 1.0}, "win_rate_a": 0.51},
                {"features": {"a": 2.0}, "win_rate_a": 0.50},
                {"features": {"a": 3.0}, "win_rate_a": 0.49},
            ]
            with open(path, "w") as fout:
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
            _, y, _ = load_jsonl(data_dir)
            assert y.tolist() == [1.0, 0.0, 0.0]

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl(Path("/nonexistent/path.jsonl"))


class TestSplitData:
    def test_returns_correct_split_sizes(self):
        rng = np.random.default_rng(42)
        X = rng.random((1000, 5))
        y = np.array([0, 1] * 500)
        X_tr, y_tr, X_val, y_val, X_test, y_test = split_data(X, y)
        assert len(y_tr) == 700
        assert len(y_val) == 150
        assert len(y_test) == 150

    def test_stratification_preserves_label_ratio(self):
        rng = np.random.default_rng(42)
        X = rng.random((2000, 5))
        y = np.array([0] * 600 + [1] * 1400)
        overall = y.mean()
        _, y_tr, _, y_val, _, y_test = split_data(X, y)
        assert abs(y_tr.mean() - overall) < 0.05
        assert abs(y_val.mean() - overall) < 0.05
        assert abs(y_test.mean() - overall) < 0.05


class TestGetFeatureImportance:
    def test_returns_sorted_descending(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.random((50, 4))
        y = np.array([0, 1] * 25)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        names = ["a", "b", "c", "d"]
        imp = get_feature_importance(model, names)
        values = list(imp.values())
        assert len(imp) == len(names)
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_all_values_positive(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.random((50, 4))
        y = np.array([0, 1] * 25)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        imp = get_feature_importance(model, ["x", "y", "z", "w"])
        for v in imp.values():
            assert v >= 0.0
