"""Benchmark competitor wrapper: Botzilla (ELO 1211).

Uses Q-learning with a pre-trained Q-table.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\BotzillaSubmission"
)
sys.path.insert(0, str(_submission))

from botzillaCompetitor import BotzillaCompetitor  # noqa: E402

__all__ = ["BotzillaCompetitor"]
