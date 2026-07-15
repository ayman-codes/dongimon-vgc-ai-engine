"""Benchmark competitor wrapper: minimon (ELO 1215).

Top 3 competitor — 3rd place.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\minimon_02 - Leon Brunke"
)
sys.path.insert(0, str(_submission))

from minimon import minimon  # noqa: E402

__all__ = ["minimon"]
