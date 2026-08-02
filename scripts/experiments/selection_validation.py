"""Validate the size-4 active-pair ordering of the selection policy.

At ``max_team_size=4`` the selection decision routes to the analytical
``_order_by_pair_synergy`` fast path (``policy.py``); the MP model and the
``mp_only``/``mp_sim`` modes only run for teams larger than the roster cut,
and ``_order_by_mp`` is currently dead code.

This experiment:
1. Measures the ground-truth win rate of every ordered active-pair of
   Dongimon's ga_seed=5 team under the GreedyDongi pilot (both sides),
   sides swapped, against each competitor's team.
2. Reports the pair chosen by the fast path, the (unused) MP ordering
   path, and BasicSelectionPolicy, with the ground-truth WR of each pick.

Usage:
    uv run python scripts/experiments/selection_validation.py --n-battles=30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.ecosystem import build_team, label_roster, sanitized_team_build_decision
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_move_set, gen_pkm_roster

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.battle.greedy_dongi import GreedyDongiPolicy
from src.selection.policy import DongimonSelectionPolicy
from src.teambuild.policy import HesfTeamBuildPolicy

N_MOVES = 100
ROSTER_SIZE = 50
MAX_TEAM_SIZE = 4
MAX_PKM_MOVES = 4
N_ACTIVE = 2


def _build_roster(seed: int) -> tuple[list[Any], list[Any]]:
    """Generate and label the championship roster deterministically.

    Args:
        seed: RNG seed for roster generation.

    Returns:
        Tuple of (move_set, roster) with ids labelled.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    move_set = gen_move_set(N_MOVES, rng)
    roster = gen_pkm_roster(ROSTER_SIZE, move_set, MAX_PKM_MOVES, rng)
    label_roster(move_set, roster)
    BasicMeta(move_set, roster)
    return move_set, roster


def _build_team(policy: Any, roster: list[Any]) -> Any | None:
    """Build a vgc2 Team for a teambuild policy.

    Args:
        policy: A TeamBuildPolicy instance.
        roster: Labelled championship roster.

    Returns:
        The built Team, or None on failure.
    """
    try:
        commands = sanitized_team_build_decision(policy, roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
        if not commands:
            return None
        return build_team(commands, roster)
    except Exception:
        return None


def _run_pair_match(
    my_team: Any,
    my_order: list[int],
    opp_team: Any,
    opp_order: list[int],
    n_battles: int,
    seed: int,
    bp: Any,
) -> tuple[int, int]:
    """Battle two ordered subteams under the shared pilot, sides swapped.

    Args:
        my_team: Dongimon full team.
        my_order: Ordered member indices (active pair first, then reserves).
        opp_team: Opponent full team.
        opp_order: Opponent ordered member indices.
        n_battles: Number of battles to run.
        seed: Base RNG seed.
        bp: Shared battle policy for both sides.

    Returns:
        Tuple of (wins_my, wins_opp).
    """
    my_view = TeamView(my_team)
    opp_view = TeamView(opp_team)
    sub_a, sub_view_a = subteam(my_team, my_view, my_order)
    sub_b, sub_view_b = subteam(opp_team, opp_view, opp_order)
    params = BattleRuleParam()
    wins_a = 0
    wins_b = 0

    for b_idx in range(n_battles):
        gen = np.random.default_rng(seed + b_idx)
        a_is_side0 = b_idx % 2 == 0
        if a_is_side0:
            s0, v0 = sub_a, sub_view_a
            s1, v1 = sub_b, sub_view_b
        else:
            s0, v0 = sub_b, sub_view_b
            s1, v1 = sub_a, sub_view_a

        battle_teams = get_battle_teams((s0, s1), N_ACTIVE)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (v0, v1))
            sv1 = StateView(engine.state, 1, (v1, v0))
            cmd0 = bp.decision(sv0, v1)
            cmd1 = bp.decision(sv1, v0)
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


def _ordered_pairs(n: int) -> list[list[int]]:
    """Enumerate every ordered active pair plus reserves.

    Args:
        n: Team size.

    Returns:
        List of orderings: [pair[0], pair[1], *reserves].
    """
    orderings = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            rest = [i for i in range(n) if i not in (a, b)]
            orderings.append([a, b] + rest)
    return orderings


def main() -> None:
    """Run the selection ordering validation and print findings."""
    parser = argparse.ArgumentParser(description="Selection active-pair ordering validation.")
    parser.add_argument("--roster-seed", type=int, default=42, help="RNG seed for the roster")
    parser.add_argument("--n-battles", type=int, default=20, help="Battles per pair per opponent")
    parser.add_argument("--battle-seed", type=int, default=11, help="Base seed for battles")
    args = parser.parse_args()

    _move_set, roster = _build_roster(args.roster_seed)
    my_team = _build_team(HesfTeamBuildPolicy(ga_seed=5), roster)
    assert my_team is not None
    my_species = [m.species.id for m in my_team.members]

    from vgc2.agent.teambuild import RandomTeamBuildPolicy

    from competitors.competitor1_jjj import JJJ_Competitor
    from competitors.competitor2_minimon import minimon
    from competitors.competitor_caaaden import CaaadenCompetitor

    opp_teams: list[tuple[str, Any]] = []
    for name, tp in [
        ("JJJ", JJJ_Competitor().teambuildpolicy),
        ("minimon", minimon().teambuildpolicy),
        ("caaaden", CaaadenCompetitor().teambuildpolicy),
        ("Greedy", RandomTeamBuildPolicy()),
    ]:
        t = _build_team(tp, roster)
        if t is not None:
            opp_teams.append((name, t))

    sel = DongimonSelectionPolicy(selection_mode="mp_only")
    my_view = TeamView(my_team)
    opp_views = list(my_view.members)
    bp_gd = GreedyDongiPolicy()

    print("=" * 72)
    print("Selection active-pair ordering validation")
    print(f"  Dongimon team species: {my_species}")
    print(f"  opponents: {[n for n, _ in opp_teams]}")
    print("=" * 72)

    fast_order = list(sel._order_by_pair_synergy(my_team, opp_views, MAX_TEAM_SIZE))
    try:
        mp_order = list(sel._order_by_mp(my_team, opp_views, MAX_TEAM_SIZE))
    except Exception as exc:
        mp_order = []
        print(f"  [WARN] _order_by_mp failed: {type(exc).__name__}: {exc}")
    basic_order = list(BasicSelectionPolicy().decision((my_team, my_view), MAX_TEAM_SIZE))
    print(f"\n  fast-path order  : {fast_order}")
    print(f"  MP order (dead)  : {mp_order}")
    print(f"  Basic order      : {basic_order}")

    print("\n  Ground truth (mean WR of each ordered active pair vs field):")
    pair_wrs: dict[tuple[int, int], list[float]] = {}
    for order in _ordered_pairs(len(my_team.members)):
        pair = (order[0], order[1])
        opp_wrs: list[float] = []
        for _opp_idx, (_opp_name, opp_team) in enumerate(opp_teams):
            opp_view = TeamView(opp_team)
            opp_basic = list(BasicSelectionPolicy().decision((opp_team, opp_view), MAX_TEAM_SIZE))
            w_my, w_opp = _run_pair_match(
                my_team, order, opp_team, opp_basic, args.n_battles,
                args.battle_seed + pair[0] * 100 + pair[1] * 10 + _opp_idx, bp_gd,
            )
            dec = w_my + w_opp
            opp_wrs.append(w_my / dec if dec > 0 else 0.5)
        mean_wr = float(np.mean(opp_wrs))
        pair_wrs[pair] = opp_wrs
        print(f"    pair {pair[0]}x{pair[1]} species=({my_species[pair[0]]},{my_species[pair[1]]}) "
              f"mean_WR={mean_wr:.3f}")

    def _pick_wr(order: list[int]) -> float:
        pair = (order[0], order[1])
        return float(np.mean(pair_wrs[pair]))

    print("\n  Verdict:")
    print(f"    fast-path pick  {tuple(fast_order[:2])} -> WR {_pick_wr(fast_order):.3f}")
    print(f"    MP pick         {tuple(mp_order[:2])} -> WR {_pick_wr(mp_order):.3f}")
    print(f"    Basic pick      {tuple(basic_order[:2])} -> WR {_pick_wr(basic_order):.3f}")
    best_pair = max(pair_wrs, key=lambda p: np.mean(pair_wrs[p]))
    print(f"    ground-truth best pair: {best_pair} -> WR {np.mean(pair_wrs[best_pair]):.3f}")

    result = {
        "dongimon_team_species": my_species,
        "fast_path_order": fast_order,
        "mp_order": mp_order,
        "basic_order": basic_order,
        "pair_wrs": {f"{a}x{b}": [round(v, 3) for v in vs] for (a, b), vs in pair_wrs.items()},
    }
    out = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "selection_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
