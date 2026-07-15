"""Benchmark competitor wrapper: Laze (ELO 1095).
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\LazeComp"
)
sys.path.insert(0, str(_submission))

from LazeCompetitor import LazeCompetitor  # noqa: E402

__all__ = ["LazeCompetitor"]
