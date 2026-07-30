"""All-vs-all ELO benchmark — battle policy only.

Every player shares a procedurally-generated team and uses
BasicSelectionPolicy. Only the BattlePolicy differs, isolating
in-battle decision quality via ELO ratings.

Usage:
    uv run python scripts/benchmark_battle.py --seed=42 --n-matches=10 --n-battles=25
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights
from src.tuning.elo_rating import update_elo

INITIAL_ELO = 1500.0
ELO_K = 32.0


def _import_bp_factory(module_path: str, class_name: str) -> Any:
    """Import a competitor and return a factory for its battle policy."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    def factory() -> Any:
        return cls().battlepolicy

    return factory


def _greedy_bp_factory() -> Any:
    return GreedyBattlePolicy()


_PLAYER_ROSTER: list[tuple[str, Any]] = [
    ("Dongimon", None),
    ("Greedy", _greedy_bp_factory),
    ("JJJ", _import_bp_factory("competitors.competitor1_jjj", "JJJ_Competitor")),
    ("minimon", _import_bp_factory("competitors.competitor2_minimon", "minimon")),
    ("caaaden", _import_bp_factory("competitors.competitor_caaaden", "CaaadenCompetitor"))
]


def _run_match(
    bp_a: Any,
    bp_b: Any,
    base_team: Any,
    base_view: Any,
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
    seed: int,
    name_a: str = "",
    name_b: str = "",
    round_idx: int = 0,
    pair_info: str = "",
    battle_log: TextIO | None = None,
) -> tuple[int, int]:
    """Run N battles between two battle policies using the same team.

    Args:
        bp_a: Battle policy for side A.
        bp_b: Battle policy for side B.
        base_team: Shared team object.
        base_view: Shared team view.
        sel: Selection policy for both sides.
        params: Battle rule parameters.
        n_battles: Number of battles to run.
        seed: Base RNG seed.
        name_a: Player name for side A (logging).
        name_b: Player name for side B (logging).
        round_idx: Current round identifier (logging).
        pair_info: Pair description (logging).
        battle_log: Optional JSONL file for per-battle logging.

    Returns:
        Tuple of (wins_a, wins_b).
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

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            cmd0 = bp_a.decision(sv0, sub_view_b)
            cmd1 = bp_b.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
            winner_name = name_a
        elif engine.winning_side == 1:
            wins_b += 1
            winner_name = name_b
        else:
            winner_name = "draw"

        if battle_log is not None:
            record = {
                "round": round_idx,
                "pair": pair_info,
                "player_a": name_a,
                "player_b": name_b,
                "winner": winner_name,
                "seed": battle_seed,
            }
            json.dump(record, battle_log)
            battle_log.write("\n")

    return wins_a, wins_b


def main() -> None:
    parser = argparse.ArgumentParser(description="All-vs-all ELO benchmark for Dongimon battle policy.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-matches", type=int, default=10, help="Number of all-vs-all rounds (ELO epochs)")
    parser.add_argument("--n-battles", type=int, default=25, help="Battles per head-to-head matchup")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    args = parser.parse_args()

    weights_dict = load_battle_weights().model_dump()
    player_names = [p[0] for p in _PLAYER_ROSTER]

    elos: dict[str, float] = dict.fromkeys(player_names, INITIAL_ELO)
    n_players = len(player_names)

    sel = BasicSelectionPolicy()
    params = BattleRuleParam()
    history: list[dict[str, Any]] = []

    total_matchups = args.n_matches * n_players * (n_players - 1) // 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_battle")
    os.makedirs(results_dir, exist_ok=True)

    battle_log_path = os.path.join(results_dir, f"battle_log_{timestamp}.jsonl")
    team_log_path = os.path.join(results_dir, f"team_log_{timestamp}.jsonl")
    results_path = os.path.join(results_dir, f"elo_battle_{timestamp}.json")

    print("=" * 60)
    print("Dongimon Battle Royale ELO — mode=battle")
    print(f"  seed={args.seed}, n_matches={args.n_matches}, n_battles={args.n_battles}")
    print(f"  players: {', '.join(player_names)}")
    print(f"  total matchups: {total_matchups} ({args.n_matches} rounds x {n_players * (n_players - 1) // 2} pairs)")
    print(f"  battle log: {battle_log_path}")
    print(f"  team log: {team_log_path}")
    print("=" * 60)

    with (
        open(battle_log_path, "w", encoding="utf-8") as battle_log,
        open(team_log_path, "w", encoding="utf-8") as team_log,
    ):
        for r_idx in range(args.n_matches):
            round_seed = args.seed + r_idx * 1000
            team_rng = np.random.default_rng(round_seed)

            base_team = gen_team(4, 4, team_rng)
            base_view = TeamView(base_team)

            team_record = {
                "round": r_idx,
                "species_names": [str(m) for m in base_team.members],
                "base_stats": [list(m.base_stats) if hasattr(m, 'base_stats') else [] for m in base_team.members],
            }
            json.dump(team_record, team_log)
            team_log.write("\n")

            bp_cache: dict[str, Any] = {}
            for name, factory in _PLAYER_ROSTER:
                if name == "Dongimon":
                    comp = DongimonCompetitor(custom_weights=weights_dict)
                    bp_cache[name] = comp.battlepolicy
                else:
                    bp_cache[name] = factory()

            new_elos = dict(elos)
            pair_idx = 0
            for i in range(n_players):
                for j in range(i + 1, n_players):
                    p1, p2 = player_names[i], player_names[j]
                    matchup_seed = round_seed + pair_idx * 100 + 1
                    wins_p1, wins_p2 = _run_match(
                        bp_cache[p1], bp_cache[p2],
                        base_team, base_view, sel, params, args.n_battles, matchup_seed,
                        name_a=p1, name_b=p2,
                        round_idx=r_idx, pair_info=f"{i}_{j}",
                        battle_log=battle_log,
                    )
                    p1_won = wins_p1 > wins_p2
                    new_elos[p1], new_elos[p2] = update_elo(new_elos[p1], new_elos[p2], p1_won, ELO_K)
                    pair_idx += 1

            elos = new_elos
            history.append({"round": r_idx, "elos": dict(elos)})

            top = sorted(elos.items(), key=lambda x: -x[1])
            top_str = " | ".join(f"{n}: {r:.1f}" for n, r in top)
            print(f"  Round {r_idx + 1:2d}/{args.n_matches}: {top_str}")

    rankings = sorted(elos.items(), key=lambda x: -x[1])
    print("\n" + "=" * 60)
    print("Final ELO Standings")
    print("=" * 60)
    for rank, (name, elo) in enumerate(rankings, 1):
        marker = "  ← Dongimon" if name == "Dongimon" else ""
        print(f"  {rank}. {name:<20} {elo:>8.1f}{marker}")
    print("=" * 60)

    output = {
        "mode": "battle",
        "seed": args.seed,
        "n_matches": args.n_matches,
        "n_battles": args.n_battles,
        "k_factor": ELO_K,
        "initial_elo": INITIAL_ELO,
        "tag": args.tag,
        "players": player_names,
        "history": history,
        "final_elos": dict(elos),
        "final_rankings": [name for name, _ in rankings],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Battle log: {battle_log_path}")
    print(f"Team log: {team_log_path}")


if __name__ == "__main__":
    main()
