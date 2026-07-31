"""Tests for BC model training pipeline (Milestone 4)."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pytest

from src.tree_bc.actions import JOINT_ACTION_COUNT
from src.tree_bc.encoder import FEATURE_DIM
from src.tree_bc.optuna import (
    EXPERT_SAMPLE_WEIGHTS,
    load_bc_data,
    split_data,
    valid_action_accuracy,
    xgboost_objective,
)

N_JOINT_ACTIONS = JOINT_ACTION_COUNT


def _make_record(
    action_idx: int = 0,
    expert: str = "JJJ",
    won: bool = True,
    features: list[float] | None = None,
) -> dict[str, Any]:
    """Create a synthetic BC data record.

    Args:
        action_idx: Joint action index (0-99).
        expert: Expert policy name.
        won: Whether the battle was won.
        features: Optional feature vector (random if None).

    Returns:
        Dict matching the BC JSONL record schema.
    """
    if features is None:
        features = np.random.default_rng(42).random(FEATURE_DIM).tolist()
    valid = list(range(N_JOINT_ACTIONS))
    return {
        "battle_id": 0,
        "turn": 0,
        "side": 0,
        "expert": expert,
        "features": features,
        "action_idx": action_idx,
        "valid_actions": valid,
        "won": won,
    }


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file.

    Args:
        records: List of record dicts.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestLoadBcData:
    """Tests for load_bc_data function."""

    def test_loads_correct_shapes(self, tmp_path: Path) -> None:
        """Loaded arrays have correct shapes and dtypes."""
        records = [_make_record(action_idx=i % 10) for i in range(20)]
        _write_jsonl(records, tmp_path / "bc_data_test.jsonl")

        features, actions, masks, weights = load_bc_data(tmp_path)

        assert features.shape == (20, FEATURE_DIM)
        assert features.dtype == np.float32
        assert actions.shape == (20,)
        assert actions.dtype == np.int32
        assert masks.shape == (20, N_JOINT_ACTIONS)
        assert masks.dtype == bool
        assert weights.shape == (20,)
        assert weights.dtype == np.float32

    def test_win_filter(self, tmp_path: Path) -> None:
        """Win filter removes losing records."""
        records = [
            _make_record(won=True),
            _make_record(won=False),
            _make_record(won=True),
        ]
        _write_jsonl(records, tmp_path / "bc_data_test.jsonl")

        features, actions, masks, weights = load_bc_data(tmp_path, win_filter=True)
        assert len(actions) == 2

        features, actions, masks, weights = load_bc_data(tmp_path, win_filter=False)
        assert len(actions) == 3

    def test_expert_weights_applied(self, tmp_path: Path) -> None:
        """Sample weights correspond to expert quality mapping."""
        records = [
            _make_record(expert="JJJ"),
            _make_record(expert="minimon"),
            _make_record(expert="caaaden"),
            _make_record(expert="dongimon"),
        ]
        _write_jsonl(records, tmp_path / "bc_data_test.jsonl")

        _, _, _, weights = load_bc_data(tmp_path)

        assert weights[0] == pytest.approx(EXPERT_SAMPLE_WEIGHTS["JJJ"])
        assert weights[1] == pytest.approx(EXPERT_SAMPLE_WEIGHTS["minimon"])
        assert weights[2] == pytest.approx(EXPERT_SAMPLE_WEIGHTS["caaaden"])
        assert weights[3] == pytest.approx(EXPERT_SAMPLE_WEIGHTS["dongimon"])

    def test_valid_mask_encoding(self, tmp_path: Path) -> None:
        """Valid actions list is converted to boolean mask correctly."""
        record = _make_record()
        record["valid_actions"] = [0, 5, 10, 99]
        _write_jsonl([record], tmp_path / "bc_data_test.jsonl")

        _, _, masks, _ = load_bc_data(tmp_path)

        assert masks[0, 0] is np.True_
        assert masks[0, 5] is np.True_
        assert masks[0, 10] is np.True_
        assert masks[0, 99] is np.True_
        assert masks[0, 1] is np.False_
        assert masks[0].sum() == 4

    def test_no_files_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when no JSONL files exist."""
        with pytest.raises(FileNotFoundError):
            load_bc_data(tmp_path)


class TestValidActionAccuracy:
    """Tests for valid_action_accuracy metric."""

    def test_perfect_predictions(self) -> None:
        """All correct predictions yield accuracy 1.0."""
        y_true = np.array([0, 1, 2], dtype=np.int32)
        proba = np.zeros((3, N_JOINT_ACTIONS), dtype=np.float64)
        proba[0, 0] = 1.0
        proba[1, 1] = 1.0
        proba[2, 2] = 1.0
        masks = np.ones((3, N_JOINT_ACTIONS), dtype=bool)

        assert valid_action_accuracy(y_true, proba, masks) == pytest.approx(1.0)

    def test_mask_overrides_higher_prob(self) -> None:
        """Invalid action with higher prob is masked out."""
        y_true = np.array([5], dtype=np.int32)
        proba = np.zeros((1, N_JOINT_ACTIONS), dtype=np.float64)
        proba[0, 3] = 0.9
        proba[0, 5] = 0.1
        masks = np.ones((1, N_JOINT_ACTIONS), dtype=bool)
        masks[0, 3] = False

        assert valid_action_accuracy(y_true, proba, masks) == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """All wrong predictions yield accuracy 0.0."""
        y_true = np.array([0, 1], dtype=np.int32)
        proba = np.zeros((2, N_JOINT_ACTIONS), dtype=np.float64)
        proba[0, 1] = 1.0
        proba[1, 0] = 1.0
        masks = np.ones((2, N_JOINT_ACTIONS), dtype=bool)

        assert valid_action_accuracy(y_true, proba, masks) == pytest.approx(0.0)


class TestSplitData:
    """Tests for stratified train/val/test split."""

    def test_split_sizes(self) -> None:
        """Split produces approximately correct proportions."""
        n = 1000
        rng = np.random.default_rng(42)
        features = rng.random((n, FEATURE_DIM)).astype(np.float32)
        actions = rng.integers(0, 10, size=n).astype(np.int32)
        masks = np.ones((n, N_JOINT_ACTIONS), dtype=bool)
        weights = np.ones(n, dtype=np.float32)

        result = split_data(features, actions, masks, weights)
        _X_train, y_train = result[0], result[1]
        _X_val, y_val = result[4], result[5]
        _X_test, y_test = result[8], result[9]

        assert len(y_test) == pytest.approx(n * 0.10, abs=5)
        assert len(y_val) == pytest.approx(n * 0.10, abs=5)
        assert len(y_train) == pytest.approx(n * 0.80, abs=10)

    def test_no_overlap(self) -> None:
        """Train/val/test sets are disjoint."""
        n = 500
        rng = np.random.default_rng(7)
        features = rng.random((n, FEATURE_DIM)).astype(np.float32)
        actions = rng.integers(0, 5, size=n).astype(np.int32)
        masks = np.ones((n, N_JOINT_ACTIONS), dtype=bool)
        weights = np.ones(n, dtype=np.float32)

        result = split_data(features, actions, masks, weights)
        total = len(result[1]) + len(result[5]) + len(result[9])
        assert total == n


class TestXGBoostObjective:
    """Tests for the Optuna objective function."""

    def test_overfit_tiny_dataset(self) -> None:
        """XGBoost can overfit a tiny dataset (sanity check)."""
        rng = np.random.default_rng(42)
        n = 50
        n_classes = 5
        X = rng.random((n, FEATURE_DIM)).astype(np.float32)
        y = rng.integers(0, n_classes, size=n).astype(np.int32)
        masks = np.ones((n, n_classes), dtype=bool)
        w = np.ones(n, dtype=np.float32)

        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        m_train, m_val = masks[:40], masks[40:]
        w_train = w[:40]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: xgboost_objective(
                trial, X_train, y_train, m_train, w_train, X_val, y_val, m_val
            ),
            n_trials=3,
        )

        assert study.best_value > 0.1


class TestModelSaveLoad:
    """Tests for model serialization."""

    def test_model_saves_loads(self, tmp_path: Path) -> None:
        """Joblib roundtrip preserves predictions."""
        from xgboost import XGBClassifier

        rng = np.random.default_rng(42)
        n = 100
        X = rng.random((n, FEATURE_DIM)).astype(np.float32)
        y = rng.integers(0, 5, size=n).astype(np.int32)

        model = XGBClassifier(
            n_estimators=10, max_depth=3, verbosity=0,
            objective="multi:softprob", num_class=N_JOINT_ACTIONS,
            tree_method="hist",
        )
        model.fit(X, y)
        proba_before = model.predict_proba(X)

        path = tmp_path / "test_model.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        proba_after = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(proba_before, proba_after)

    def test_feature_importance_nonzero(self) -> None:
        """Trained model uses features (not degenerate)."""
        from xgboost import XGBClassifier

        rng = np.random.default_rng(42)
        n = 200
        X = rng.random((n, FEATURE_DIM)).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.int32)

        model = XGBClassifier(
            n_estimators=50, max_depth=4, verbosity=0,
            objective="multi:softprob", num_class=N_JOINT_ACTIONS,
            tree_method="hist",
        )
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.sum() > 0
        assert (importances > 0).sum() >= 1
