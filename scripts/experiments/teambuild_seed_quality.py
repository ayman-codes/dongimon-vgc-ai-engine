"""Evaluate teambuild GA seeds by team strength against the field.

For each candidate ``ga_seed``, builds Dongimon's team deterministically
from the championship roster (seeded stdlib/numpy RNG + seeded battle
royale) and battles it against every competitor's team under the
GreedyDongi pilot on both sides. The seed whose team has the highest mean
win rate is the one that most consistently produces a winning team.

Usage:
    uv run python scripts/experiments/teambuild_seed_quality.py \
        --roster-seed=42 --n-seeds=10 --n-battles=20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.ecosystem import build_team, label_roster, sanitized_team_build_decision
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


def _run_match(
    team_a: Any,
    team_b: Any,
    n_battles: int,
    seed: int,
    bp: Any,
) -> tuple[int, int]:
    """Battle two teams under the shared pilot with sides swapped.

    Args:
        team_a: Team for side A.
        team_b: Team for side B.
        n_battles: Number of battles to run.
        seed: Base RNG seed.
        bp: Shared battle policy for both sides.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)
    params = BattleRuleParam()
    wins_a = 0
    wins_b = 0

    for b_idx in range(n_battles):
        gen = np.random.default_rng(seed + b_idx)
        a_is_side0 = b_idx % 2 == 0
        if a_is_side0:
            sub_0, sub_view_0 = team_a, view_a
            sub_1, sub_view_1 = team_b, view_b
        else:
            sub_0, sub_view_0 = team_b, view_b
            sub_1, sub_view_1 = team_a, view_a

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


def main() -> None:
    """Run the seed-quality evaluation and print the best seed."""
    parser = argparse.ArgumentParser(description="Teambuild GA seed quality evaluation.")
    parser.add_argument("--roster-seed", type=int, default=42, help="RNG seed for the roster")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of GA seeds to evaluate")
    parser.add_argument("--n-battles", type=int, default=20, help="Battles per matchup")
    parser.add_argument("--battle-seed", type=int, default=7, help="Base seed for evaluation battles")
    args = parser.parse_args()

    _move_set, roster = _build_roster(args.roster_seed)
    opponents = _competitor_teams(roster)
    bp = GreedyDongiPolicy()

    print("=" * 64)
    print("Teambuild seed quality evaluation")
    print(f"  roster_seed={args.roster_seed}  n_seeds={args.n_seeds}  n_battles={args.n_battles}")
    print(f"  opponents: {', '.join(n for n, _ in opponents)}")
    print("=" * 64)

    seed_results: dict[str, Any] = {}
    for seed in range(args.n_seeds):
        t0 = time.perf_counter()
        team = _build_team(HesfTeamBuildPolicy(ga_seed=seed), roster)
        if team is None:
            print(f"  seed {seed}: BUILD_FAILED")
            continue
        species = [m.species.id for m in team.members]

        match_wrs: dict[str, float] = {}
        wins_total = 0
        battles_total = 0
        for opp_name, opp_team in opponents:
            if opp_team is None:
                continue
            w_a, w_b = _run_match(team, opp_team, args.n_battles, args.battle_seed + seed * 1000, bp)
            decisive = w_a + w_b
            wr = w_a / decisive if decisive > 0 else 0.5
            match_wrs[opp_name] = round(wr, 3)
            wins_total += w_a
            battles_total += decisive

        mean_wr = wins_total / battles_total if battles_total > 0 else 0.0
        seed_results[str(seed)] = {
            "team_species": species,
            "mean_wr": round(mean_wr, 4),
            "matchup_wr": match_wrs,
        }
        elapsed = time.perf_counter() - t0
        print(f"  seed {seed:2d} mean_wr={mean_wr:.3f} team={species}  ({elapsed:.1f}s)")

    ranked = sorted(seed_results.items(), key=lambda kv: -kv[1]["mean_wr"])
    print("\n" + "=" * 64)
    print("Seed ranking (best first)")
    print("=" * 64)
    for rank, (seed_key, rec) in enumerate(ranked, 1):
        marker = "  <-- best" if rank == 1 else ""
        print(f"  {rank}. ga_seed={seed_key}  mean_wr={rec['mean_wr']:.4f}  team={rec['team_species']}{marker}")

    best_key, best_rec = ranked[0]
    result = {
        "roster_seed": args.roster_seed,
        "n_battles": args.n_battles,
        "battle_seed": args.battle_seed,
        "seed_results": seed_results,
        "best_seed": int(best_key),
        "best_mean_wr": best_rec["mean_wr"],
    }
    out = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "teambuild_seed_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
