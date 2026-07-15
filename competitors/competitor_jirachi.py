"""Benchmark competitor wrapper: jirachi / Smart_Jirachi_Championship_AI (ELO 1159).

Uses Always Smart Beam Search for battle, Max Firepower for selection and team building.
"""

import sys
from pathlib import Path

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\jirachi - DONGMIN KIM"
)
sys.path.insert(0, str(_submission))

from jirachi_championship_competitor import SmartJirachiChampionshipCompetitor  # noqa: E402

__all__ = ["SmartJirachiChampionshipCompetitor"]
