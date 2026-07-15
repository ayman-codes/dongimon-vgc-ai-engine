"""Benchmark competitor wrapper: Caaaden (ELO 1162).

Custom battle policy with Korean comments, damage-based move selection.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\caaaden_competitor"
)
sys.path.insert(0, str(_submission))

from caaaden_competitor import CaaadenCompetitor  # noqa: E402

__all__ = ["CaaadenCompetitor"]
