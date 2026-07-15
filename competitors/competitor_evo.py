"""Benchmark competitor wrapper: EvoTrainer (battle track only).

Evolutionary strategy battle policy with evolved hyperparameters.
No team build policy (uses random). HANDLES genes.npy relative path
by changing directory during instantiation.
"""

import os
import sys
from pathlib import Path

from vgc2.competition import Competitor

_submission = Path(
    r"C:\Users\Mohammed Ayman PC\pokemon-vgc-engine\edition\vgc2025\submissions\evoTrainer"
)
sys.path.insert(0, str(_submission))

from EvoCompetitor import EvoCompetitor as _EvoCompetitor  # noqa: E402


class EvoCompetitor(Competitor):
    """Wrapper that handles genes.npy relative path by chdir during init."""

    def __init__(self, name: str = "Evo"):
        _old_cwd = os.getcwd()
        os.chdir(str(_submission))
        self._inner = _EvoCompetitor(name)
        os.chdir(_old_cwd)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def battlepolicy(self):
        return self._inner.battlepolicy

    @property
    def selectionpolicy(self):
        return self._inner.selectionpolicy

    @property
    def teambuildpolicy(self):
        return self._inner.teambuildpolicy


__all__ = ["EvoCompetitor"]
