"""Benchmark competitor wrapper: JJJ (Championship winner, ELO 1606).

Top 3 competitor — champion.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\JJJ - JunSung - wfd gfd"
)
sys.path.insert(0, str(_submission))

from JJJCompetitor import JJJ_Competitor  # noqa: E402

__all__ = ["JJJ_Competitor"]
