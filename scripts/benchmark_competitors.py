"""Benchmark Dongimon against championship competitors and baseline.

Runs 10-battle matches between Dongimon and each opponent.
Reports win/loss for every pairing.
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
        results[opp_name] = (d_wins, o_wins)
        print(f"Dongimon {d_wins} — {opp_name} {o_wins}  ({elapsed:.1f}s)")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Opponent':<20} {'Dongimon':>10} {'Opponent':>10}")
    print("-" * 42)
    for opp_name, (dw, ow) in results.items():
        print(f"{opp_name:<20} {dw:>10} {ow:>10}")
    total_d = sum(r[0] for r in results.values())
    total_o = sum(r[1] for r in results.values())
    print("-" * 42)
    print(f"{'TOTAL':<20} {total_d:>10} {total_o:>10}")


def _import_competitor(module_path: str, class_name: str):
    """Dynamically import a competitor class, returning a factory callable."""

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


if __name__ == "__main__":
    main()
