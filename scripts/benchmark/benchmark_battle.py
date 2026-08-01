"""Pure battle-policy benchmark — BP quality isolation.

Every player shares the same deterministically-generated 4-member team.
No selection policy is involved: all 4 members are used in fixed index
order.  Only the BattlePolicy differs between players, isolating
in-battle decision quality.

Fully seeded and reproducible: same seed → same teams → same battles.

Players:
    Greedy, JJJ, minimon, caaaden, Dongimon (heuristic), TreeBC (XGBoost)

Usage:
    uv run python scripts/benchmark/benchmark_battle.py --seed=42 --n-rounds=5 --n-battles=20
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.tuning.elo_rating import update_elo

INITIAL_ELO = 1500.0
ELO_K = 32.0
BC_MODEL_PATH = Path(__file__).parent.parent.parent / "src" / "models" / "bc_xgboost_model.joblib"


def _try_import_bp(module_path: str, class_name: str) -> Any | None:
    """Import a competitor and return a factory for its battle policy.

    Args:
        module_path: Dotted module path (e.g. "competitors.competitor1_jjj").
        class_name: Competitor class name inside the module.

    Returns:
        A zero-arg factory returning the battle policy, or None on failure.
    """
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        def factory() -> Any:
            return cls().battlepolicy

        return factory
    except Exception as exc:
        print(f"  [WARN] Could not import {class_name} from {module_path}: {exc}")
        return None


def _greedy_bp_factory() -> Any:
    """Return a fresh GreedyBattlePolicy instance."""
    return GreedyBattlePolicy()


def _dongimon_bp_factory() -> Any:
    """Return the Dongimon heuristic battle policy."""
    from competitor import DongimonCompetitor

    return DongimonCompetitor().battlepolicy


def _greedy_dongi_bp_factory() -> Any:
    """Return a fresh GreedyDongi net-damage policy instance."""
    from src.battle.greedy_dongi import GreedyDongiPolicy

    return GreedyDongiPolicy()


def _tree_bc_bp_factory() -> Any:
    """Return a TreeBC XGBoost battle policy wrapper.

    Loads the trained XGBoost model and wraps it with valid-action
    masking inference via the extracted ``TreeBCBattlePolicy``.

    Returns:
        A policy object with a decision(state, opp_view) method.
    """
    from PPO_trainers.tree_bc_policy.policy import TreeBCBattlePolicy

    return TreeBCBattlePolicy(BC_MODEL_PATH)


def _build_roster() -> list[tuple[str, Any]]:
    """Build the player roster, skipping unavailable competitors.

    Returns:
        List of (name, bp_factory) tuples.
    """
    candidates: list[tuple[str, str, str]] = [
        ("Greedy", "", ""),
        ("JJJ", "competitors.competitor1_jjj", "JJJ_Competitor"),
        ("minimon", "competitors.competitor2_minimon", "minimon"),
        ("caaaden", "competitors.competitor_caaaden", "CaaadenCompetitor"),
        ("Dongimon", "competitor", "DongimonCompetitor"),
        ("GreedyDongi", "", ""),
    ]

    roster: list[tuple[str, Any]] = []
    for name, mod_path, cls_name in candidates:
        if name == "Greedy":
            roster.append((name, _greedy_bp_factory))
        elif name == "GreedyDongi":
            roster.append((name, _greedy_dongi_bp_factory))
        else:
            factory = _try_import_bp(mod_path, cls_name)
            if factory is not None:
                roster.append((name, factory))

    if BC_MODEL_PATH.exists():
        roster.append(("TreeBC", _tree_bc_bp_factory))
    else:
        print(f"  [WARN] TreeBC model not found at {BC_MODEL_PATH}, skipping")

    return roster


def _run_match(
    bp_a: Any,
    bp_b: Any,
    base_team: Any,
    base_view: Any,
    params: BattleRuleParam,
    n_battles: int,
    seed: int,
    name_a: str = "",
    name_b: str = "",
    round_idx: int = 0,
    battle_log: TextIO | None = None,
) -> tuple[int, int]:
    """Run N battles between two battle policies on the same team.

    Selection is deterministic: indices [0, 1, 2, 3] in order.
    Both sides use the identical 4-member team.

    Args:
        bp_a: Battle policy for side A.
        bp_b: Battle policy for side B.
        base_team: Shared 4-member team.
        base_view: Shared team view.
        params: Battle rule parameters.
        n_battles: Number of battles to run.
        seed: Base RNG seed.
        name_a: Player name for side A (logging).
        name_b: Player name for side B (logging).
        round_idx: Current round identifier (logging).
        battle_log: Optional JSONL file for per-battle logging.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    wins_a = 0
    wins_b = 0
    fixed_indices = list(range(len(base_team.members)))

    for b_idx in range(n_battles):
        battle_seed = seed + b_idx
        gen = np.random.default_rng(battle_seed)

        sub_a, sub_view_a = subteam(base_team, base_view, fixed_indices)
        sub_b, sub_view_b = subteam(base_team, base_view, fixed_indices)

        battle_teams = get_battle_teams((sub_a, sub_b), 2)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(
            state, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )

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
                "player_a": name_a,
                "player_b": name_b,
                "winner": winner_name,
                "seed": battle_seed,
            }
            json.dump(record, battle_log)
            battle_log.write("\n")

    return wins_a, wins_b


def main() -> None:
    """Run the pure-BP benchmark and print ELO + win-rate matrix."""
    parser = argparse.ArgumentParser(
        description="Pure battle-policy benchmark (no selection, deterministic)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument(
        "--n-rounds", type=int, default=5,
        help="Number of team-generation rounds (default: 5)",
    )
    parser.add_argument(
        "--n-battles", type=int, default=20,
        help="Battles per head-to-head matchup per round (default: 20)",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    args = parser.parse_args()

    roster = _build_roster()
    player_names = [name for name, _ in roster]
    n_players = len(player_names)

    if n_players < 2:
        print("ERROR: Need at least 2 players. Check competitor imports.")
        sys.exit(1)

    elos: dict[str, float] = dict.fromkeys(player_names, INITIAL_ELO)
    pair_wins: dict[tuple[str, str], int] = {}
    pair_total: dict[tuple[str, str], int] = {}
    for i in range(n_players):
        for j in range(i + 1, n_players):
            pair_wins[(player_names[i], player_names[j])] = 0
            pair_total[(player_names[i], player_names[j])] = 0

    params = BattleRuleParam()
    history: list[dict[str, Any]] = []

    total_matchups = args.n_rounds * n_players * (n_players - 1) // 2
    total_battles = total_matchups * args.n_battles

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "benchmark_battle")
    os.makedirs(results_dir, exist_ok=True)

    battle_log_path = os.path.join(results_dir, f"battle_log_{timestamp}.jsonl")
    results_path = os.path.join(results_dir, f"elo_battle_{timestamp}.json")

    print("=" * 60)
    print("Pure Battle-Policy Benchmark (BP only, no selection)")
    print(f"  seed={args.seed}  rounds={args.n_rounds}  battles/pair/round={args.n_battles}")
    print(f"  players ({n_players}): {', '.join(player_names)}")
    print(f"  total matchups: {total_matchups}  total battles: {total_battles}")
    print(f"  battle log: {battle_log_path}")
    print("=" * 60)

    t_start = time.time()

    with open(battle_log_path, "w", encoding="utf-8") as battle_log:
        for r_idx in range(args.n_rounds):
            round_seed = args.seed + r_idx * 10000
            team_rng = np.random.default_rng(round_seed)

            base_team = gen_team(4, 4, team_rng)
            base_view = TeamView(base_team)

            bp_cache: dict[str, Any] = {}
            for name, factory in roster:
                bp_cache[name] = factory()

            new_elos = dict(elos)
            pair_idx = 0
            for i in range(n_players):
                for j in range(i + 1, n_players):
                    p1, p2 = player_names[i], player_names[j]
                    matchup_seed = round_seed + pair_idx * 1000 + 1
                    wins_p1, wins_p2 = _run_match(
                        bp_cache[p1], bp_cache[p2],
                        base_team, base_view, params,
                        args.n_battles, matchup_seed,
                        name_a=p1, name_b=p2,
                        round_idx=r_idx,
                        battle_log=battle_log,
                    )
                    pair_wins[(p1, p2)] += wins_p1
                    pair_total[(p1, p2)] += wins_p1 + wins_p2

                    p1_won = wins_p1 > wins_p2
                    new_elos[p1], new_elos[p2] = update_elo(
                        new_elos[p1], new_elos[p2], p1_won, ELO_K
                    )
                    pair_idx += 1

            elos = new_elos
            history.append({"round": r_idx, "elos": dict(elos)})

            top = sorted(elos.items(), key=lambda x: -x[1])
            top_str = " | ".join(f"{n}: {r:.0f}" for n, r in top)
            elapsed = time.time() - t_start
            print(f"  Round {r_idx + 1:2d}/{args.n_rounds}  [{elapsed:.0f}s]  {top_str}")

    elapsed_total = time.time() - t_start

    rankings = sorted(elos.items(), key=lambda x: -x[1])

    print("\n" + "=" * 60)
    print("Final ELO Standings")
    print("=" * 60)
    for rank, (name, elo) in enumerate(rankings, 1):
        print(f"  {rank}. {name:<20} {elo:>8.1f}")

    print("\n" + "=" * 60)
    print("Pairwise Win Rates (row vs column)")
    print("=" * 60)

    header = f"{'':>16}" + "".join(f"{n:>14}" for n in player_names)
    print(header)
    for p1 in player_names:
        row = f"{p1:>16}"
        for p2 in player_names:
            if p1 == p2:
                row += f"{'—':>14}"
            else:
                key = (p1, p2) if (p1, p2) in pair_total else (p2, p1)
                total = pair_total.get(key, 0)
                if total == 0:
                    row += f"{'?':>14}"
                else:
                    wr = pair_wins[key] / total if key == (p1, p2) else 1.0 - pair_wins[key] / total
                    row += f"{wr:>13.1%} "
        print(row)

    print("=" * 60)
    print(f"  Elapsed: {elapsed_total:.1f}s")

    output = {
        "mode": "battle_policy_only",
        "description": (
            "Pure BP benchmark. Same 4-member team for both sides, "
            "deterministic ordering [0,1,2,3], no selection policy. "
            "Isolates in-battle decision quality."
        ),
        "seed": args.seed,
        "n_rounds": args.n_rounds,
        "n_battles_per_pair_per_round": args.n_battles,
        "total_battles_per_pair": args.n_rounds * args.n_battles,
        "k_factor": ELO_K,
        "initial_elo": INITIAL_ELO,
        "tag": args.tag,
        "players": player_names,
        "history": history,
        "final_elos": dict(elos),
        "final_rankings": [name for name, _ in rankings],
        "pairwise_win_rates": {
            f"{p1}_vs_{p2}": (
                pair_wins[(p1, p2)] / pair_total[(p1, p2)]
                if pair_total.get((p1, p2), 0) > 0 else None
            )
            for p1 in player_names
            for p2 in player_names
            if p1 != p2 and (p1, p2) in pair_total
        },
        "elapsed_sec": round(elapsed_total, 1),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Battle log: {battle_log_path}")


if __name__ == "__main__":
    main()
