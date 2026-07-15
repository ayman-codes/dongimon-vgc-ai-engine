"""Benchmark competitor wrapper: IceMonte (ELO 1187).
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\iceMonteSubmission"
)
sys.path.insert(0, str(_submission))

from iceMonteCompetitor import IceMonteCompetitor  # noqa: E402

__all__ = ["IceMonteCompetitor"]
