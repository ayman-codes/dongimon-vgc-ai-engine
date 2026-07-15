"""Benchmark competitor wrapper: StocKarpador (ELO 1211).

Top 3 competitor — 4th place (Wolfe is 2nd but no source available).
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\StocKarpadorSubmission"
)
sys.path.insert(0, str(_submission))

from StocKarpadorCompetitor import StocKarpadorCompetitor  # noqa: E402

__all__ = ["StocKarpadorCompetitor"]
