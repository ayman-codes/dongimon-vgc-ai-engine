"""Teambuild diagnosis: why is Dongimon's team only mid-tier?

Dumps each competitor's team composition on the championship roster and
measures team strength under both the neutral Greedy pilot and Dongimon's
own GreedyDongi pilot (Basic selection for everyone, round-robin, sides
swapped). This exposes:

1. The synergy-vs-BST tradeoff: does Dongimon's HESF pipeline sacrifice
   raw stats for coverage/roles, and does that hurt under a neutral pilot?
2. Pilot sensitivity: is Dongimon's team stronger under its own BP
   (GreedyDongi) than under Greedy?

Usage:
    uv run python scripts/experiments/teambuild_diagnosis.py --n-battles=20
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


def _all_teams(roster: list[Any]) -> list[tuple[str, Any | None]]:
    """Build the field's teams from the roster.

    Args:
        roster: Labelled championship roster.

    Returns:
        List of (name, team) tuples.
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

    return [(name, _build_team(tp, roster)) for name, tp in builders]


def _run_match(
    team_a: Any,
    team_b: Any,
    sel: Any,
    n_battles: int,
    seed: int,
    bp: Any,
) -> tuple[int, int]:
    """Battle two teams under a shared pilot, sides swapped.

    Args:
        team_a: Team for side A.
        team_b: Team for side B.
        sel: Shared selection policy.
        n_battles: Number of battles to run.
        seed: Base RNG seed.
        bp: Shared battle policy for both sides.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)
    idx_a = list(sel.decision((team_a, view_b), MAX_TEAM_SIZE))
    idx_b = list(sel.decision((team_b, view_a), MAX_TEAM_SIZE))
    sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
    sub_b, sub_view_b = subteam(team_b, view_b, idx_b)
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


def _mean_wr(team: Any, others: list[tuple[str, Any]], sel: Any, n_battles: int, seed: int, bp: Any) -> float:
    """Compute the mean win rate of a team vs the field.

    Args:
        team: The team under evaluation.
        others: List of (name, team) opponents.
        sel: Shared selection policy.
        n_battles: Battles per matchup.
        seed: Base seed.
        bp: Shared pilot.

    Returns:
        Mean win rate over opponents.
    """
    wrs = []
    for _i, (_name, opp) in enumerate(others):
        if opp is None:
            continue
        w_a, w_b = _run_match(team, opp, sel, n_battles, seed, bp)
        dec = w_a + w_b
        wrs.append(w_a / dec if dec > 0 else 0.5)
    return float(np.mean(wrs)) if wrs else 0.0


def main() -> None:
    """Run the teambuild diagnosis and print composition + pilot results."""
    parser = argparse.ArgumentParser(description="Teambuild composition and pilot-sensitivity diagnosis.")
    parser.add_argument("--roster-seed", type=int, default=42, help="RNG seed for the roster")
    parser.add_argument("--n-battles", type=int, default=20, help="Battles per matchup")
    parser.add_argument("--battle-seed", type=int, default=13, help="Base seed for battles")
    args = parser.parse_args()

    _move_set, roster = _build_roster(args.roster_seed)
    teams = _all_teams(roster)
    sel = BasicSelectionPolicy()
    bp_greedy = GreedyBattlePolicy()
    bp_gd = GreedyDongiPolicy()

    print("=" * 72)
    print("Teambuild diagnosis")
    print(f"  roster_seed={args.roster_seed}  n_battles={args.n_battles}")
    print("=" * 72)

    print("\n  Team composition:")
    comps: dict[str, Any] = {}
    for name, team in teams:
        if team is None:
            print(f"    {name:<10} BUILD_FAILED")
            continue
        members = team.members
        species_ids = [m.species.id for m in members]
        bsts = [sum(m.species.base_stats) for m in members]
        n_physical = [sum(1 for mv in m.moves if mv.category.value in (1, 2)) for m in members]
        ev_spe = [int(m.evs[5]) for m in members]
        comps[name] = {
            "species": species_ids,
            "bst": bsts,
            "mean_bst": float(np.mean(bsts)),
            "physical_count": n_physical,
            "spe_ev": ev_spe,
        }
        print(f"    {name:<10} species={species_ids} BST={bsts} mean={np.mean(bsts):.0f} "
              f"phys={n_physical} speEV={ev_spe}")

    print("\n  Team strength under each pilot (Basic selection, vs field):")
    results: dict[str, Any] = {}
    for pilot_name, bp in [("Greedy", bp_greedy), ("GreedyDongi", bp_gd)]:
        wrs = {}
        for name, team in teams:
            if team is None:
                continue
            others = [(n, t) for n, t in teams if n != name]
            wrs[name] = round(_mean_wr(team, others, sel, args.n_battles, args.battle_seed, bp), 3)
        results[pilot_name] = wrs
        ranked = sorted(wrs.items(), key=lambda kv: -kv[1])
        line = "  ".join(f"{n}: {v:.3f}" for n, v in ranked)
        print(f"    {pilot_name:<12} -> {line}")

    verdict = {
        "composition": comps,
        "pilot_wrs": results,
        "note": "Dongimon ga_seed=5 team; synergy-vs-BST tradeoff visible via composition.",
    }
    out = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "teambuild_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
