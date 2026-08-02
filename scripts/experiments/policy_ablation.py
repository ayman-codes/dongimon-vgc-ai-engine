"""Isolate which Dongimon policy is weakest via component ablation.

A. Selection isolation: Dongimon's ga_seed=5 team is piloted by GreedyDongi
   on both sides while the active-pair selection is swapped between
   DongimonSelectionPolicy and BasicSelectionPolicy (sides swapped). Also
   records the active-pair indices each selection returns.

B. Teambuild isolation: all six competitors' teams are battled round-robin
   with BasicSelectionPolicy for everyone and the Greedy pilot on both
   sides. The resulting ranking isolates pure teambuild quality (selection
   neutralized), and can be compared against benchmark_team's own-selection
   result to expose the selection contribution.

Usage:
    uv run python scripts/experiments/policy_ablation.py --n-battles=30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
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


def _run_match(
    team_a: Any,
    team_b: Any,
    sel_a: Any,
    sel_b: Any,
    n_battles: int,
    seed: int,
    bp: Any,
) -> tuple[int, int]:
    """Battle two teams under a shared pilot, sides swapped.

    Args:
        team_a: Team for side A.
        team_b: Team for side B.
        sel_a: Selection policy for side A.
        sel_b: Selection policy for side B.
        n_battles: Number of battles to run.
        seed: Base RNG seed.
        bp: Shared battle policy for both sides.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)
    params = BattleRuleParam()
    idx_a = list(sel_a.decision((team_a, view_b), MAX_TEAM_SIZE))
    idx_b = list(sel_b.decision((team_b, view_a), MAX_TEAM_SIZE))
    sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
    sub_b, sub_view_b = subteam(team_b, view_b, idx_b)
    wins_a = 0
    wins_b = 0

    for b_idx in range(n_battles):
        gen = np.random.default_rng(seed + b_idx)
        a_is_side0 = b_idx % 2 == 0
        if a_is_side0:
            sub_0, sub_view_0 = sub_a, sub_view_a
            sub_1, sub_view_1 = sub_b, sub_view_b
        else:
            sub_0, sub_view_0 = sub_b, sub_view_b
            sub_1, sub_view_1 = sub_a, sub_view_a

        battle_teams = get_battle_teams((sub_0, sub_1), N_ACTIVE)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_0, sub_view_1))
            sv1 = StateView(engine.state, 1, (sub_view_1, sub_view_0))
            cmd0 = bp.decision(sv0, sub_view_1)
            cmd1 = bp.decision(sv1, sub_view_0)
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


def _competitor_teams(roster: list[Any]) -> list[tuple[str, Any | None]]:
    """Build the competitor field's teams from the roster.

    Args:
        roster: Labelled championship roster.

    Returns:
        List of (name, team) tuples; team may be None if teambuild failed.
    """
    from vgc2.agent.teambuild import RandomTeamBuildPolicy

    from competitors.competitor1_jjj import JJJ_Competitor
    from competitors.competitor2_minimon import minimon
    from competitors.competitor_caaaden import CaaadenCompetitor
    from competitors.competitor_peach import PeachCompetitor

    builders: list[tuple[str, Any]] = [
        ("Dongimon", HesfTeamBuildPolicy(ga_seed=5)),
        ("JJJ", JJJ_Competitor().teambuildpolicy),
        ("minimon", minimon().teambuildpolicy),
        ("caaaden", CaaadenCompetitor().teambuildpolicy),
        ("Greedy", RandomTeamBuildPolicy()),
    ]
    peach = PeachCompetitor()
    peach_tp = getattr(peach, "teambuildpolicy", None) or RandomTeamBuildPolicy()
    builders.append(("peach", peach_tp))

    teams: list[tuple[str, Any | None]] = []
    for name, tp in builders:
        teams.append((name, _build_team(tp, roster)))
    return teams


def main() -> None:
    """Run the policy ablation and print the weakest-policy verdict."""
    parser = argparse.ArgumentParser(description="Dongimon policy weakness ablation.")
    parser.add_argument("--roster-seed", type=int, default=42, help="RNG seed for the roster")
    parser.add_argument("--n-battles", type=int, default=30, help="Battles per matchup")
    parser.add_argument("--battle-seed", type=int, default=7, help="Base seed for battles")
    args = parser.parse_args()

    from competitor import DongimonCompetitor

    _move_set, roster = _build_roster(args.roster_seed)
    dongimon = DongimonCompetitor()
    team_d = _build_team(HesfTeamBuildPolicy(ga_seed=5), roster)
    sel_dongimon = dongimon.selectionpolicy
    sel_basic = BasicSelectionPolicy()
    bp_gd = GreedyDongiPolicy()
    bp_greedy = GreedyBattlePolicy()

    print("=" * 64)
    print("Policy ablation: which Dongimon policy is weakest?")
    print(f"  roster_seed={args.roster_seed}  n_battles={args.n_battles}")
    print("=" * 64)

    print("\n[A] Selection isolation (Dongimon ga_seed=5 team, GreedyDongi pilot):")
    team_view = TeamView(team_d)
    idx_d = list(sel_dongimon.decision((team_d, team_view), MAX_TEAM_SIZE))
    idx_basic = list(sel_basic.decision((team_d, team_view), MAX_TEAM_SIZE))
    print(f"    Dongimon selection active order : {idx_d}")
    print(f"    Basic selection active order    : {idx_basic}")

    w_d, w_b = _run_match(team_d, team_d, sel_dongimon, sel_basic, args.n_battles, args.battle_seed, bp_gd)
    decisive = w_d + w_b
    wr_dongimon_sel = w_d / decisive if decisive > 0 else 0.5
    print(f"    Dongimon-selection WR vs Basic-selection: {wr_dongimon_sel:.3f} "
          f"({w_d}-{w_b})  -> {'selection HELPS' if wr_dongimon_sel > 0.5 else 'selection HURTS'}")

    print("\n[B] Teambuild isolation (BasicSelection for all, Greedy pilot):")
    teams = _competitor_teams(roster)
    names = [n for n, t in teams if t is not None]
    pair_wrs: dict[str, list[float]] = {n: [] for n in names}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            n_a, t_a = teams[i]
            n_b, t_b = teams[j]
            if t_a is None or t_b is None:
                continue
            w_a, w_b = _run_match(t_a, t_b, sel_basic, sel_basic, args.n_battles,
                                  args.battle_seed + i * 100 + j, bp_greedy)
            dec = w_a + w_b
            pair_wrs[n_a].append(w_a / dec if dec > 0 else 0.5)
            pair_wrs[n_b].append(w_b / dec if dec > 0 else 0.5)

    ranked = sorted(pair_wrs.items(), key=lambda kv: -np.mean(kv[1]))
    for rank, (name, wrs) in enumerate(ranked, 1):
        print(f"    {rank}. {name:<10} mean_WR={np.mean(wrs):.3f}")

    result = {
        "selection_isolation": {
            "dongimon_active_order": idx_d,
            "basic_active_order": idx_basic,
            "dongimon_sel_wr_vs_basic": round(wr_dongimon_sel, 4),
            "verdict": "selection HELPS" if wr_dongimon_sel > 0.5 else "selection HURTS",
        },
        "teambuild_isolation_rankings": {n: round(float(np.mean(v)), 4) for n, v in ranked},
    }
    out = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "policy_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
