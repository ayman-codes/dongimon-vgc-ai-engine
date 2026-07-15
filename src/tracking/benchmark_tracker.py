"""Lightweight benchmark result logger.

Logs parameters, metrics, and results to JSON files in ``mlruns/``.
Functional replacement for MLflow when the full server is unavailable.
"""

import json
import os
import time
from typing import Any


class BenchmarkTracker:
    """Context manager that logs benchmark results to a JSON run file.

    Usage::

        with BenchmarkTracker("weight_tuning_test", weights=my_weights) as bt:
            bt.log_result("JJJ", dongimon_wins, jjj_wins)
    """

    def __init__(
        self,
        run_name: str,
        seed: int = 42,
        weights: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Initialise a benchmark run.

        Args:
            run_name: Human-readable name for this run.
            seed: Random seed used for reproducibility.
            weights: Current battle policy weights to snapshot.
            tags: Optional tags (e.g., ``{"phase": "c1_normalization"}``).
        """
        self._run_name = run_name
        self._seed = seed
        self._weights = weights or {}
        self._tags = tags or {}
        self._results: dict[str, tuple[int, int]] = {}
        self._start_time: float | None = None

    def __enter__(self) -> "BenchmarkTracker":
        os.makedirs("mlruns", exist_ok=True)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - (self._start_time or 0)
        total_dongimon = sum(d for d, _ in self._results.values())
        total_opponent = sum(o for _, o in self._results.values())
        grand_total = total_dongimon + total_opponent

        record = {
            "run_name": self._run_name,
            "seed": self._seed,
            "tags": self._tags,
            "weights": self._weights,
            "results": {k: {"dongimon": v[0], "opponent": v[1]} for k, v in self._results.items()},
            "aggregate": {
                "dongimon_total": total_dongimon,
                "opponent_total": total_opponent,
                "win_rate": total_dongimon / max(grand_total, 1),
            },
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        fname = f"mlruns/{time.strftime('%Y%m%d_%H%M%S')}_{self._run_name.replace(' ', '_')}.json"
        with open(fname, "w") as f:
            json.dump(record, f, indent=2)

    def log_result(self, opponent_name: str, dongimon_wins: int, opponent_wins: int) -> None:
        """Record the result of a single matchup.

        Args:
            opponent_name: Name of the opponent.
            dongimon_wins: Battles won by Dongimon.
            opponent_wins: Battles won by the opponent.
        """
        self._results[opponent_name] = (dongimon_wins, opponent_wins)

    @property
    def results(self) -> dict[str, tuple[int, int]]:
        """Return recorded results as a dict."""
        return dict(self._results)
