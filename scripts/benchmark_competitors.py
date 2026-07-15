"""Benchmark Dongimon against championship competitors and baseline.

Runs 10-battle matches between Dongimon and each opponent.
Reports per-matchup and aggregate win rates.

Results are logged to MLflow under the experiment ``dongimon_benchmarks``.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights
from src.tracking.benchmark_tracker import BenchmarkTracker


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-battles", type=int, default=10, help="Battles per matchup")
    parser.add_argument("--tag", type=str, default="", help="Optional MLflow tag")
    parser.add_argument("--no-tracking", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    np.random.default_rng(args.seed)
    weights = load_battle_weights().model_dump()

    opponents = [
        ("JJJ",          _import_competitor("competitors.competitor1_jjj", "JJJ_Competitor")),
        ("minimon",      _import_competitor("competitors.competitor2_minimon", "minimon")),
        ("StocKarpador", _import_competitor("competitors.competitor3_stockarpador", "StocKarpadorCompetitor")),
    ]

    run_name = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}"
    tags = {}
    if args.tag:
        tags["tag"] = args.tag

    tracker = BenchmarkTracker(run_name, seed=args.seed, weights=weights, tags=tags)

    if args.no_tracking:
        print("=" * 60)
        print("Dongimon Benchmark — Tracking disabled")
        print("=" * 60)
    else:
        tracker.__enter__()

    print(f"  seed={args.seed}, n_battles={args.n_battles}")

    for opp_name, opp_factory in opponents:
        print(f"\n  vs {opp_name}...", end=" ", flush=True)
        start = time.perf_counter()

        dongimon = CompetitorManager(DongimonCompetitor())
        opponent = CompetitorManager(opp_factory())
        match = Match((dongimon, opponent), n_battles=args.n_battles, gen=gen_team)
        match.run()

        elapsed = time.perf_counter() - start
        wins = match.wins
        d_wins, o_wins = wins[0], wins[1]
        total = d_wins + o_wins
        d_pct = d_wins / total * 100 if total else 0.0
        o_pct = o_wins / total * 100 if total else 0.0
        winner = "Dongimon" if d_wins > o_wins else opp_name
        print(f"done  ({elapsed:.1f}s)")
        print(f"         Winner: {winner:>12} ({d_pct:5.1f}% Dongimon vs {o_pct:5.1f}%)")

        if not args.no_tracking:
            tracker.log_result(opp_name, d_wins, o_wins)

    if not args.no_tracking:
        tracker.__exit__()

    total_d = 0
    total_o = 0
    if hasattr(tracker, '_results'):
        total_d = sum(r[0] for r in tracker._results.values())
        total_o = sum(r[1] for r in tracker._results.values())

    gt = total_d + total_o
    print("\n" + "=" * 60)
    if gt > 0:
        print(f"  ALL-TIME: Dongimon {total_d} wins — Opponents {total_o} wins ({total_d/gt*100:.1f}% aggregate)")
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
