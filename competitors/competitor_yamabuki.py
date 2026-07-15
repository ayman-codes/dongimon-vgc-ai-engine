"""Benchmark competitor wrapper: Yamabuki (Battle Track winner).

Uses Monte Carlo Tree Search with a trained LogisticRegression win-rate
predictor (164 features). Battle track champion, not championship competitor.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\submission"
)
sys.path.insert(0, str(_submission))

from competitor import S7Competitor  # noqa: E402

__all__ = ["S7Competitor"]
