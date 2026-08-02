"""Head-to-head ELO benchmark — Dongimon battle policy vs GreedyBattlePolicy.

Same team, same selection. Only battle policy differs.
Quick diagnostic to isolate battle-policy quality.

Usage:
    uv run python scripts/benchmark/benchmark_greedy_vs_dongimon.py --n-matches=20 --n-battles=25
"""

import argparse
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
from src.tuning.elo_rating import update_elo

INITIAL_ELO = 1500.0
ELO_K = 32.0


def _run_match(
    bp_a: Any,
    bp_b: Any,
    base_team: Any,
    base_view: Any,
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
    seed: int,
) -> tuple[int, int]:
    """Run N battles between two battle policies using the same team.

    Sides are swapped each battle to eliminate the engine's
    side-0 speed-tie advantage.
    """
    wins_a = 0
    wins_b = 0

    for b_idx in range(n_battles):
        battle_seed = seed + b_idx
        gen = np.random.default_rng(battle_seed)

        idx_a = sel.decision((base_team, base_view), 4)
        idx_b = sel.decision((base_team, base_view), 4)

        sub_a, sub_view_a = subteam(base_team, base_view, idx_a)
        sub_b, sub_view_b = subteam(base_team, base_view, idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), 2)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

        a_is_side0 = (b_idx % 2 == 0)
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            if a_is_side0:
                cmd0 = bp_a.decision(sv0, sub_view_b)
                cmd1 = bp_b.decision(sv1, sub_view_a)
            else:
                cmd0 = bp_b.decision(sv0, sub_view_b)
                cmd1 = bp_a.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            if a_is_side0:
                wins_a += 1
            else:
                wins_b += 1
        elif engine.winning_side == 1:
            if a_is_side0:
                wins_b += 1
            else:
                wins_a += 1

    return wins_a, wins_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Dongimon vs Greedy head-to-head battle benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-matches", type=int, default=20, help="Number of rounds (different teams)")
    parser.add_argument("--n-battles", type=int, default=25, help="Battles per round")
    args = parser.parse_args()

    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    elo_dongimon = INITIAL_ELO
    elo_greedy = INITIAL_ELO
    total_wins_d = 0
    total_wins_g = 0
    total_draws = 0

    print("=" * 60)
    print("Dongimon vs Greedy — Battle Policy Head-to-Head")
    print(f"  seed={args.seed}, n_matches={args.n_matches}, n_battles={args.n_battles}")
    print(f"  total battles: {args.n_matches * args.n_battles}")
    print("=" * 60)

    t0 = time.perf_counter()

    for r_idx in range(args.n_matches):
        round_seed = args.seed + r_idx * 1000
        team_rng = np.random.default_rng(round_seed)

        base_team = gen_team(4, 4, team_rng)
        base_view = TeamView(base_team)

        bp_dongimon = GreedyDongiPolicy()
        bp_greedy = GreedyBattlePolicy()

        matchup_seed = round_seed + 1
        wins_d, wins_g = _run_match(
            bp_dongimon, bp_greedy,
            base_team, base_view, sel, params,
            args.n_battles, matchup_seed,
        )

        draws = args.n_battles - wins_d - wins_g
        total_wins_d += wins_d
        total_wins_g += wins_g
        total_draws += draws

        d_won = wins_d > wins_g
        elo_dongimon, elo_greedy = update_elo(elo_dongimon, elo_greedy, d_won, ELO_K)

        print(
            f"  Round {r_idx + 1:2d}/{args.n_matches}: "
            f"Dongimon {wins_d:2d} - {wins_g:2d} Greedy  "
            f"(ELO: {elo_dongimon:.1f} / {elo_greedy:.1f})"
        )

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    total_decisive = total_wins_d + total_wins_g
    win_rate = total_wins_d / total_decisive * 100 if total_decisive > 0 else 0.0
    print(f"  Dongimon: {total_wins_d} wins  |  Greedy: {total_wins_g} wins  |  Draws: {total_draws}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Final ELO: Dongimon {elo_dongimon:.1f} / Greedy {elo_greedy:.1f}")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
