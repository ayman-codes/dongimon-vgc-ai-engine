"""All-vs-all ELO benchmark — teambuild + selection quality (battle policy neutralized).

All competitors use GreedyBattlePolicy to eliminate battle-policy bias.
ELO differences reflect ONLY teambuild + selection quality.

Each pairing creates fresh teams via each competitor's TeambuildPolicy.
ELO is updated after each head-to-head matchup based on series winner.

Usage:
    uv run python scripts/benchmark_team.py --seed=42 --n-matches=5 --n-battles=25
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import BattleRuleParam
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights
from src.tuning.elo_rating import update_elo

INITIAL_ELO = 1500.0
ELO_K = 32.0


class GreedyBattleWrapper(Competitor):  # type: ignore[misc]
    """Wraps any competitor, overriding battlepolicy with GreedyBattlePolicy.

    Preserves the wrapped competitor's teambuildpolicy and selectionpolicy
    so that ELO differences reflect only team-building and selection quality.
    """

    def __init__(self, inner: Competitor) -> None:
        self._inner = inner
        self._greedy_bp = GreedyBattlePolicy()

    @property
    def name(self) -> str:
        return str(self._inner.name)

    @property
    def battlepolicy(self) -> Any:
        return self._greedy_bp

    @property
    def selectionpolicy(self) -> Any:
        return self._inner.selectionpolicy

    @property
    def teambuildpolicy(self) -> Any:
        return self._inner.teambuildpolicy


def _import_competitor_cls(module_path: str, class_name: str) -> Any:
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


_PLAYER_ROSTER: list[tuple[str, Any]] = [
    ("Dongimon", None),
    ("JJJ", _import_competitor_cls("competitors.competitor1_jjj", "JJJ_Competitor")),
    ("minimon", _import_competitor_cls("competitors.competitor2_minimon", "minimon")),
    ("caaaden", _import_competitor_cls("competitors.competitor_caaaden", "CaaadenCompetitor")),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="All-vs-all ELO benchmark for Dongimon (full pipeline).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-matches", type=int, default=5, help="Number of all-vs-all rounds (ELO epochs)")
    parser.add_argument("--n-battles", type=int, default=25, help="Battles per head-to-head matchup")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--fitness-mode", type=str, default="heuristic", choices=["heuristic", "model"],
                        help="Teambuild fitness mode")
    parser.add_argument("--selection-mode", type=str, default="hybrid", choices=["hybrid", "matrix", "simulate"],
                        help="Selection mode")
    args = parser.parse_args()

    weights_dict = load_battle_weights().model_dump()
    player_names = [p[0] for p in _PLAYER_ROSTER]
    n_players = len(player_names)

    elos: dict[str, float] = dict.fromkeys(player_names, INITIAL_ELO)
    params = BattleRuleParam()
    history: list[dict[str, Any]] = []

    total_matchups = args.n_matches * n_players * (n_players - 1) // 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Dongimon Battle Royale ELO — mode=team (battle policy neutralized)")
    print(f"  seed={args.seed}, n_matches={args.n_matches}, n_battles={args.n_battles}")
    print(f"  players: {', '.join(player_names)}")
    print("  battle policy: GreedyBattlePolicy (ALL competitors)")
    print(f"  total matchups: {total_matchups} ({args.n_matches} rounds x {n_players * (n_players - 1) // 2} pairs)")
    print(f"  fitness_mode={args.fitness_mode}, selection_mode={args.selection_mode}")
    print("=" * 60)

    for r_idx in range(args.n_matches):
        round_seed = args.seed + r_idx * 2000
        new_elos = dict(elos)
        pair_idx = 0

        for i in range(n_players):
            for j in range(i + 1, n_players):
                p1_name, p2_name = player_names[i], player_names[j]

                p1_cls: Any = _PLAYER_ROSTER[i][1]
                p2_cls: Any = _PLAYER_ROSTER[j][1]

                p1_inner = DongimonCompetitor(custom_weights=weights_dict) if p1_name == "Dongimon" else p1_cls()
                p1_cm = CompetitorManager(GreedyBattleWrapper(p1_inner))

                p2_inner = DongimonCompetitor(custom_weights=weights_dict) if p2_name == "Dongimon" else p2_cls()
                p2_cm = CompetitorManager(GreedyBattleWrapper(p2_inner))

                matchup_seed = round_seed + pair_idx * 100
                np.random.seed(matchup_seed)

                match = Match(
                    (p1_cm, p2_cm),
                    n_battles=args.n_battles,
                    gen=gen_team,
                    params=params,
                )
                match.run()

                wins_p1, wins_p2 = match.wins
                p1_won = wins_p1 > wins_p2
                new_elos[p1_name], new_elos[p2_name] = update_elo(new_elos[p1_name], new_elos[p2_name], p1_won, ELO_K)
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
        marker = "  <-- Dongimon" if name == "Dongimon" else ""
        print(f"  {rank}. {name:<20} {elo:>8.1f}{marker}")
    print("=" * 60)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"elo_team_{timestamp}.json")

    output = {
        "mode": "team_greedy_battle",
        "description": "Battle policy neutralized (all Greedy). ELO reflects teambuild+selection only.",
        "seed": args.seed,
        "n_matches": args.n_matches,
        "n_battles": args.n_battles,
        "k_factor": ELO_K,
        "initial_elo": INITIAL_ELO,
        "tag": args.tag,
        "players": player_names,
        "battle_policy": "GreedyBattlePolicy (uniform)",
        "fitness_mode": args.fitness_mode,
        "selection_mode": args.selection_mode,
        "history": history,
        "final_elos": dict(elos),
        "final_rankings": [name for name, _ in rankings],
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
