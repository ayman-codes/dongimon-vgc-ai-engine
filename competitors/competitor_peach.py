"""Benchmark competitor wrapper: Peach (ELO 1179).

Authors: Lilly Gerlach, Anna-Lena Penk.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\PeachSubmission"
)
sys.path.insert(0, str(_submission))

from PeachCompetitor import PeachCompetitor  # noqa: E402

__all__ = ["PeachCompetitor"]
