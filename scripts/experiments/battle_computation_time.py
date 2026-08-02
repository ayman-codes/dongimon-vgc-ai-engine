"""Measure per-policy battle computation speed.

Runs 100 battles per battle policy (same team, same selection, both sides
use the same policy) and reports battles/second for each. Identifies the
fastest policies for data generation.

Usage:
    uv run python scripts/experiments/battle_computation_time.py --n-battles=100
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.battle.greedy_dongi import GreedyDongiPolicy


def _import_bp(module_path: str, class_name: str) -> Any:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls().battlepolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure per-policy battle computation speed.")
    parser.add_argument("--n-battles", type=int, default=100, help="Battles per policy")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    team_rng = np.random.default_rng(args.seed)
    team = gen_team(4, 4, team_rng)
    view = TeamView(team)
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    policies: list[tuple[str, Any]] = [
        ("Greedy", GreedyBattlePolicy()),
        ("JJJ", _import_bp("competitors.competitor1_jjj", "JJJ_Competitor")),
        ("minimon", _import_bp("competitors.competitor2_minimon", "minimon")),
        ("caaaden", _import_bp("competitors.competitor_caaaden", "CaaadenCompetitor")),
        ("GreedyDongi", GreedyDongiPolicy()),
        ("botzilla", _import_bp("competitors.competitor_botzilla", "BotzillaCompetitor")),
        ("laze", _import_bp("competitors.competitor_laze", "LazeCompetitor")),
        ("peach", _import_bp("competitors.competitor_peach", "PeachCompetitor")),
    ]

    print("=" * 60)
    print(f"Battle Policy Computation Speed Test ({args.n_battles} battles each)")
    print("=" * 60)

    results: list[tuple[str, float]] = []

    for name, bp in policies:
        wins = 0
        start = time.perf_counter()

        for b_idx in range(args.n_battles):
            battle_seed = args.seed + b_idx + 1000
            gen = np.random.default_rng(battle_seed)

            idx_a = sel.decision((team, view), 4)
            idx_b = sel.decision((team, view), 4)

            sub_a, sub_view_a = subteam(team, view, idx_a)
            sub_b, sub_view_b = subteam(team, view, idx_b)

            battle_teams = get_battle_teams((sub_a, sub_b), 2)
            state = State(battle_teams)
            rng_tuple = ((gen, gen), (gen, gen))
            engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

            while not engine.finished():
                sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
                sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
                cmd0 = bp.decision(sv0, sub_view_b)
                cmd1 = bp.decision(sv1, sub_view_a)
                engine.run_turn((cmd0, cmd1))

            if engine.winning_side == 0:
                wins += 1

        elapsed = time.perf_counter() - start
        bps = args.n_battles / max(elapsed, 0.001)
        results.append((name, bps))
        marker = "  <- fastest so far" if bps == max(r[1] for r in results) else ""
        print(f"  {name:<20} {bps:>8.1f} battles/s{marker}")

    print("\n" + "=" * 60)
    print("Rankings (fastest first)")
    print("=" * 60)
    results.sort(key=lambda x: -x[1])
    for i, (name, bps) in enumerate(results, 1):
        print(f"  {i}. {name:<20} {bps:>8.1f} battles/s")

    fastest = [r[0] for r in results[:3]]
    slowest = [r[0] for r in results[-2:]]
    print(f"\n  Recommended for data gen (top 3): {', '.join(fastest)}")
    print(f"  Avoid for data gen (slowest): {', '.join(slowest)}")


if __name__ == "__main__":
    main()
