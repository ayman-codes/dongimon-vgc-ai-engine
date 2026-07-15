"""Shared move evaluation utilities used across all three policy engines.

Provides common primitives for accuracy scaling, PP conservation penalties,
and move category checks that were duplicated across battle, selection, and
teambuild modules.
"""


def accuracy_factor(accuracy: float | None) -> float:
    """Convert a move's accuracy to a scalar multiplier.

    Args:
        accuracy: Move accuracy value (0.0–1.0 or None for never-miss).

    Returns:
        1.0 for never-miss moves, otherwise the accuracy value.
    """
    return 1.0 if accuracy is None else accuracy


def pp_penalty(score: float, current_pp: int, max_pp: int, threshold: float = 0.3, multiplier: float = 0.8) -> float:
    """Apply a PP conservation penalty to a move score.

    Reduces the score of moves with low remaining PP to discourage
    wasteful usage.

    Args:
        score: Current move score before penalty.
        current_pp: Remaining PP.
        max_pp: Maximum PP for the move.
        threshold: Fraction of max PP below which the penalty applies.
        multiplier: Score multiplier when penalized.

    Returns:
        Penalized score if PP is below threshold, otherwise unchanged score.
    """
    if max_pp > 0 and (current_pp / max_pp) < threshold:
        return score * multiplier
    return score


def is_status_move(category: int) -> bool:
    """Check if a move category represents a status (OTHER) move.

    Args:
        category: Integer category value (0=PHYSICAL, 1=SPECIAL, 2=OTHER).

    Returns:
        True if the move is a status/non-damaging move.
    """
    return category == 2


def is_damaging_move(category: int, base_power: int) -> bool:
    """Check if a move can deal damage.

    Args:
        category: Integer category value.
        base_power: Move base power.

    Returns:
        True if the move is physical or special with non-zero base power.
    """
    return category in (0, 1) and base_power > 0
