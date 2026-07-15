"""Benchmark Dongimon against championship competitors and baseline.

Runs 10-battle matches between Dongimon and each opponent.
Reports per-matchup and aggregate win rates.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor


class BaselineCompetitor:
    """Baseline: GreedyBattle + RandomSelection + RandomTeambuild."""

    def __init__(self, name: str = "Baseline"):
        from vgc2.agent.battle import GreedyBattlePolicy
        from vgc2.agent.selection import RandomSelectionPolicy
        from vgc2.agent.teambuild import RandomTeamBuildPolicy
        from vgc2.competition import Competitor as Comp

        class _B(Comp):
            @property
            def name(self): return name
            @property
            def battlepolicy(self): return GreedyBattlePolicy()
            @property
            def selectionpolicy(self): return RandomSelectionPolicy()
            @property
            def teambuildpolicy(self): return RandomTeamBuildPolicy()

        self._cls = _B

    def __call__(self):
        return self._cls()


def main():
    opponents = [
        ("Baseline",     BaselineCompetitor()),
        ("JJJ",          _import_competitor("competitors.competitor1_jjj", "JJJ_Competitor")),
        ("minimon",      _import_competitor("competitors.competitor2_minimon", "minimon")),
        ("StocKarpador", _import_competitor("competitors.competitor3_stockarpador", "StocKarpadorCompetitor")),
    ]

    print("=" * 60)
    print("Dongimon Benchmark — 10 battles per opponent")
    print("=" * 60)

    results = {}
    for opp_name, opp_factory in opponents:
        print(f"\n  vs {opp_name}...", end=" ", flush=True)
        start = time.perf_counter()

        dongimon = CompetitorManager(DongimonCompetitor())
        opponent = CompetitorManager(opp_factory())
        match = Match((dongimon, opponent), n_battles=10, gen=gen_team)
        match.run()

        elapsed = time.perf_counter() - start
        wins = match.wins
        d_wins, o_wins = wins[0], wins[1]
        total = d_wins + o_wins
        results[opp_name] = (d_wins, o_wins)
        print(f"done  ({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    total_d = 0
    total_o = 0
    for opp_name, (dw, ow) in results.items():
        total = dw + ow
        d_pct = dw / total * 100 if total else 0.0
        o_pct = ow / total * 100 if total else 0.0
        winner = "Dongimon" if dw > ow else opp_name
        print(f"  vs {opp_name:<12}  Winner: {winner:>12} ({d_pct:5.1f}% Dongimon vs {o_pct:5.1f}%)")
        total_d += dw
        total_o += ow

    gt = total_d + total_o
    if gt > 0:
        pct = total_d / gt * 100
        print(f"\n  ALL-TIME: Dongimon {total_d} wins — Opponents {total_o} wins ({pct:.1f}% aggregate)")
    print("=" * 60)


def _import_competitor(module_path: str, class_name: str):
    """Dynamically import a competitor class, returning a factory callable.

    Args:
        module_path: Dot-separated module path.
        class_name: Name of the competitor class.

    Returns:
        Callable that creates a new instance of the competitor.
    """
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


if __name__ == "__main__":
    main()
