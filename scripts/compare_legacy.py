"""Compare legacy vs extracted battle policy head-to-head on the same team.

Runs 20 battles between each policy and GreedyBattlePolicy using
the same shared team + BasicSelection for isolation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from legacy.my_battle_policy import MyBattlePolicy
from src.battle.policy import DongimonBattlePolicy


def run_battles(bp: Any, n: int, team: Any, view: Any, sel: Any, params: Any) -> int:
    wins = 0
    opp = GreedyBattlePolicy()
    for _ in range(n):
        idx_a = sel.decision((team, view), 4)
        idx_b = sel.decision((team, view), 4)
        ta, va = subteam(team, view, idx_a)
        tb, vb = subteam(team, view, idx_b)
        bt = get_battle_teams((ta, tb), 2)
        state = State(bt)
        eng = BattleEngine(state, params=params)
        while not eng.finished():
            s0 = StateView(eng.state, 0, (va, vb))
            s1 = StateView(eng.state, 1, (vb, va))
            c0 = bp.decision(s0, vb)
            c1 = opp.decision(s1, va)
            eng.run_turn((c0, c1))
        if eng.winning_side == 0:
            wins += 1
    return wins


def main() -> None:
    seed = 42
    n = 20
    rng = np.random.default_rng(seed)
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    print("Generating shared team...", end=" ", flush=True)
    team = gen_team(6, 4, rng)
    view = TeamView(team)
    print("done")

    print("=" * 60)
    print("Legacy vs Extracted Battle Policy comparison")
    print("=" * 60)

    legacy = MyBattlePolicy()
    extracted = DongimonBattlePolicy()

    print("\n  Legacy MyBattlePolicy vs GreedyBattlePolicy...", end=" ", flush=True)
    t0 = time.perf_counter()
    lw = run_battles(legacy, n, team, view, sel, params)
    lt = time.perf_counter() - t0
    print(f"  {lw}/{n} wins ({lw / n * 100:.0f}%)  [{lt:.0f}s]")

    print("  Extracted DongimonBattlePolicy vs GreedyBattlePolicy...", end=" ", flush=True)
    t0 = time.perf_counter()
    ew = run_battles(extracted, n, team, view, sel, params)
    et = time.perf_counter() - t0
    print(f"  {ew}/{n} wins ({ew / n * 100:.0f}%)  [{et:.0f}s]")

    print()
    print(f"  Legacy:    {lw}/{n} ({lw / n * 100:.0f}%)  [{lt:.0f}s]")
    print(f"  Extracted: {ew}/{n} ({ew / n * 100:.0f}%)  [{et:.0f}s]")
    diff = lw - ew
    if diff > 0:
        print(f"\n  Legacy outperforms extracted by {diff}/{n} wins ({(diff / n) * 100:.0f}%)")
    elif diff < 0:
        print(f"\n  Extracted outperforms legacy by {-diff}/{n} wins ({(-diff / n) * 100:.0f}%)")
    else:
        print(f"\n  Identical performance at {lw}/{n} wins")


if __name__ == "__main__":
    main()
