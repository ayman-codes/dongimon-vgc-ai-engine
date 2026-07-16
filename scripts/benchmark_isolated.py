"""Isolated benchmarks testing battle policy in isolation and
selection+battle policy together, both with identical shared teams.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition import Competitor
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor


def _greedy_baseline():
    class GreedyBaseline(Competitor):
        @property
        def name(self):
            return "Greedy"
        @property
        def battlepolicy(self):
            return GreedyBattlePolicy()
        @property
        def selectionpolicy(self):
            return None
        @property
        def teambuildpolicy(self):
            return None
    return GreedyBaseline()


def run_battles(
    battle_policy_a,
    battle_policy_b,
    base_team,
    base_view,
    selection_policy_a,
    selection_policy_b,
    n_battles,
    params,
) -> tuple[int, int]:
    """Run N battles between two policies using the same team pool.

    Args:
        battle_policy_a, battle_policy_b: The two battle policies.
        base_team: The shared team object.
        base_view: The shared team view.
        selection_policy_a, selection_policy_b: Selection policies.
        n_battles: How many battles to run.
        params: Battle rule parameters.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    wins_a = 0
    wins_b = 0

    for _ in range(n_battles):
        idx_a = selection_policy_a.decision((base_team, base_view), 4)
        idx_b = selection_policy_b.decision((base_team, base_view), 4)

        sub_a, sub_view_a = subteam(base_team, base_view, idx_a)
        sub_b, sub_view_b = subteam(base_team, base_view, idx_b)

        team_a = sub_a
        team_b = sub_b
        view_a = sub_view_a
        view_b = sub_view_b

        battle_teams = get_battle_teams((team_a, team_b), 2)
        state = State(battle_teams)
        engine = BattleEngine(state, params=params)

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (view_a, view_b))
            sv1 = StateView(engine.state, 1, (view_b, view_a))
            cmd0 = battle_policy_a.decision(sv0, view_b)
            cmd1 = battle_policy_b.decision(sv1, view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
        elif engine.winning_side == 1:
            wins_b += 1

    return wins_a, wins_b


def main():
    seed = 42
    n_battles = 125
    rng = np.random.default_rng(seed)

    print("Generating shared team...", end=" ", flush=True)
    shared_team = gen_team(6, 4, rng)
    shared_view = TeamView(shared_team)
    print("done")

    dongimon = DongimonCompetitor()
    basic_sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    opponents = [
        ("Greedy",       _greedy_baseline),
        ("JJJ",          _import("competitors.competitor1_jjj", "JJJ_Competitor")),
        ("minimon",      _import("competitors.competitor2_minimon", "minimon")),
        ("StocKarpador", _import("competitors.competitor3_stockarpador", "StocKarpadorCompetitor")),
    ]

    print("=" * 60)
    print("Battle policy only (same team + BasicSelection for all)")
    print("=" * 60)

    for opp_name, opp_factory in opponents:
        opp = opp_factory() if callable(opp_factory) else opp_factory
        print(f"  Dongimon vs {opp_name:<12}", end=" ", flush=True)
        t0 = time.perf_counter()
        dw, ow = run_battles(
            dongimon.battlepolicy,
            opp.battlepolicy,
            shared_team, shared_view,
            basic_sel, basic_sel,
            n_battles, params,
        )
        dt = time.perf_counter() - t0
        pct = dw / max(dw + ow, 1) * 100
        winner = "Dongimon" if dw > ow else opp_name
        print(f"Winner: {winner:>12} ({pct:5.1f}%)  [{dt:.0f}s]")


def _import(module_path: str, class_name: str):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


if __name__ == "__main__":
    main()
