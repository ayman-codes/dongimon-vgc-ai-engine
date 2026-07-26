"""Experiment: Measure data generation speed for ML model training.

Generates teams, runs battles against Greedy using the vgc2 engine,
and reports how many battles can be generated per hour. This informs
how much training data we can produce in a 10-hour window.

Usage:
    uv run python scripts/experiment_data_speed.py --n-teams=10 --n-battles=10
"""

import argparse
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
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure battle generation speed for ML data.")
    parser.add_argument("--n-teams", type=int, default=10, help="Number of random teams to test")
    parser.add_argument("--n-battles", type=int, default=10, help="Battles per team vs Greedy")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    bp = GreedyBattlePolicy()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    total_battles = args.n_teams * args.n_battles
    completed = 0

    print("=" * 60)
    print("Data Generation Speed Test")
    print(f"  seed={args.seed}, n_teams={args.n_teams}, n_battles_per_team={args.n_battles}")
    print(f"  total battles: {total_battles}")
    print("=" * 60)

    start = time.perf_counter()

    for t_idx in range(args.n_teams):
        team_seed = args.seed + t_idx * 100
        team_rng = np.random.default_rng(team_seed)
        team = gen_team(6, 4, team_rng)
        view = TeamView(team)

        for b_idx in range(args.n_battles):
            battle_seed = team_seed + b_idx
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

            completed += 1

    elapsed = time.perf_counter() - start
    bps = completed / max(elapsed, 0.001)
    bph = bps * 3600
    bph_10h = bph * 10

    print("\nResults:")
    print(f"  Battles completed: {completed}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  Battles/second: {bps:.1f}")
    print(f"  Battles/hour: {bph:.0f}")
    print(f"  Estimated in 10 hours: {bph_10h:.0f}")
    print(f"  Estimated teams × 10 battles in 10 hours: {bph_10h / args.n_battles:.0f}")
    target_battles = 100000
    hours_needed = target_battles / max(bph, 1)
    print(f"\nTo generate {target_battles} battles (for ~10K teams × 10 battles): ~{hours_needed:.1f} hours")


if __name__ == "__main__":
    main()
