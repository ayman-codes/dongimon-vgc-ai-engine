"""Tests for shared move utility functions."""

from src.shared.move_utils import accuracy_factor, is_damaging_move, is_status_move, pp_penalty


class TestAccuracyFactor:
    """Tests for accuracy_factor()."""

    def test_none_returns_one(self) -> None:
        """Never-miss moves (None accuracy) return 1.0."""
        assert accuracy_factor(None) == 1.0

    def test_full_accuracy(self) -> None:
        """100% accuracy returns 1.0."""
        assert accuracy_factor(1.0) == 1.0

    def test_partial_accuracy(self) -> None:
        """Partial accuracy returns the value as-is."""
        assert accuracy_factor(0.85) == 0.85
        assert accuracy_factor(0.7) == 0.7


class TestPPPenalty:
    """Tests for pp_penalty()."""

    def test_no_penalty_when_pp_high(self) -> None:
        """High PP ratio applies no penalty."""
        assert pp_penalty(100.0, 8, 10) == 100.0

    def test_penalty_when_pp_low(self) -> None:
        """Low PP ratio reduces the score."""
        assert pp_penalty(100.0, 2, 10) == 80.0

    def test_zero_max_pp_no_penalty(self) -> None:
        """Zero max PP means no penalty (guards against division by zero)."""
        assert pp_penalty(100.0, 0, 0) == 100.0

    def test_custom_threshold(self) -> None:
        """Custom threshold/multiplier parameters work correctly."""
        assert pp_penalty(100.0, 4, 10, threshold=0.5, multiplier=0.5) == 50.0
        assert pp_penalty(100.0, 5, 10, threshold=0.5, multiplier=0.5) == 100.0


class TestMoveCategories:
    """Tests for move category checkers."""

    def test_is_status_move(self) -> None:
        """Category 2 (OTHER) is a status move."""
        assert is_status_move(2)
        assert not is_status_move(0)
        assert not is_status_move(1)

    def test_is_damaging_move(self) -> None:
        """Physical/Special moves with BP > 0 are damaging."""
        assert is_damaging_move(0, 80)
        assert is_damaging_move(1, 90)
        assert not is_damaging_move(0, 0)
        assert not is_damaging_move(2, 0)
        assert not is_damaging_move(2, 100)
