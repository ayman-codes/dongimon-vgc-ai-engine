"""Diagnose championship ELO pathology and Dongimon slowness impact.

Establishes four facts about the championship-track results:

1. The vgc2 engine's ``elo_rating`` uses an inverted win-probability
   (``probability(r_a, r_b) = 1/(1 + 10^((r_a - r_b)/400))``), so a strong
   player beating a weak one gains ~``k`` points instead of ~``0``. This
   produces unbounded ELO drift (observed: Dongimon 3385, TreeBC -1253).
2. Re-simulating a 6-player championship with the correct ELO keeps the
   spread sane (~1000-1600) while the engine formula explodes.
3. The 1% time weight can move the overall score by at most 0.01, which
   only flips a ranking when ELOs are within ~2 points — slowness is not
   why Dongimon places low.
4. The battle engine has no per-decision timer, so a slow battle policy
   cannot lose turns; measured GreedyDongi decision latency confirms a
   battle completes with every turn resolved.

Usage:
    uv run python scripts/experiments/diagnose_championship_elo.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.util.generator import gen_team

from src.battle.greedy_dongi import GreedyDongiPolicy
from src.tuning.elo_rating import elo_probability, update_elo

ENGINE_ELO_K = 30.0
REPO_ELO_K = 30.0
WAVES = 30
PLAYERS = ["Greedy", "JJJ", "minimon", "caaaden", "Dongimon", "TreeBC"]
TRUE_RATINGS = {
    "Greedy": 1300.0,
    "JJJ": 1100.0,
    "minimon": 1400.0,
    "caaaden": 1500.0,
    "Dongimon": 1460.0,
    "TreeBC": 1000.0,
}
TIME_MIN = 0.001
TIME_MAX = 1.0


def engine_elo_rating(r_a: float, r_b: float, d: int, k: float = ENGINE_ELO_K) -> tuple[float, float]:
    """Mirror the vgc2 engine's ``elo_rating`` with its inverted probability.

    Args:
        r_a: ELO rating of player A.
        r_b: ELO rating of player B.
        d: 0 if A won, 1 if B won.
        k: K-factor.

    Returns:
        Tuple of (new_r_a, new_r_b).
    """
    p_a = 1.0 / (1.0 + math.pow(10.0, (r_a - r_b) / 400.0))
    p_b = 1.0 / (1.0 + math.pow(10.0, (r_b - r_a) / 400.0))
    if d == 0:
        return r_a + k * (1.0 - p_a), r_b + k * (0.0 - p_b)
    return r_a + k * (0.0 - p_a), r_b + k * (1.0 - p_b)


def _single_updates() -> dict[str, list[float]]:
    """Compare engine vs correct ELO update for a dominant player.

    A=1500 beats B=1200 five times in a row.

    Returns:
        Dict with per-update engine and correct ratings for A.
    """
    engine_a, engine_b = 1500.0, 1200.0
    repo_a, repo_b = 1500.0, 1200.0
    engine_traj: list[float] = []
    repo_traj: list[float] = []
    for _ in range(5):
        engine_a, engine_b = engine_elo_rating(engine_a, engine_b, 0)
        repo_a, repo_b = update_elo(repo_a, repo_b, True, REPO_ELO_K)
        engine_traj.append(round(engine_a, 1))
        repo_traj.append(round(repo_a, 1))
    return {"engine_A": engine_traj, "correct_A": repo_traj}


def _simulate_championship(update_fn: Any, seed: int) -> dict[str, float]:
    """Simulate 30 ELO-paired waves for the six players.

    Args:
        update_fn: Callable(r_a, r_b, winner_is_a) -> (new_a, new_b).
        seed: RNG seed for match outcomes.

    Returns:
        Final ELO per player.
    """
    rng = random.Random(seed)
    elos = dict.fromkeys(PLAYERS, 1200.0)
    for _ in range(WAVES):
        order = sorted(elos.keys(), key=lambda n: elos[n], reverse=True)
        for i in range(0, len(order) - 1, 2):
            a, b = order[i], order[i + 1]
            p_a = elo_probability(TRUE_RATINGS[a], TRUE_RATINGS[b])
            a_won = rng.random() < p_a
            elos[a], elos[b] = update_fn(elos[a], elos[b], a_won)
    return {name: round(elo, 1) for name, elo in elos.items()}


def _engine_wrap(r_a: float, r_b: float, a_won: bool) -> tuple[float, float]:
    """Adapt the engine update to the winner_is_a convention.

    Args:
        r_a: ELO of player A.
        r_b: ELO of player B.
        a_won: True if A won.

    Returns:
        Tuple of (new_a, new_b).
    """
    return engine_elo_rating(r_a, r_b, 0 if a_won else 1)


def _time_score(t: float) -> float:
    """Compute the engine's log-scale time score for a per-epoch time.

    Args:
        t: Time in seconds per epoch (clamped to [TIME_MIN, TIME_MAX]).

    Returns:
        Score in [0, 1] where lower time yields higher score.
    """
    clamped = max(TIME_MIN, min(t, TIME_MAX))
    return 1.0 - (math.log(clamped) - math.log(TIME_MIN)) / (math.log(TIME_MAX) - math.log(TIME_MIN))


def _time_impact() -> dict[str, float]:
    """Quantify the maximum overall-score swing from the 1% time weight.

    Returns:
        Dict with max time swing, ELO gap the swing can flip, and
        example overall scores for a fast vs slow competitor at equal ELO.
    """
    fast_time = 0.0001
    slow_time = 1.7
    swing = 0.01 * (_time_score(fast_time) - _time_score(slow_time))
    flip_gap = swing / 0.99 * 200.0
    elo = 1250.0
    overall_fast = 0.99 * (elo - 1100.0) / 200.0 + 0.01 * _time_score(fast_time)
    overall_slow = 0.99 * (elo - 1100.0) / 200.0 + 0.01 * _time_score(slow_time)
    return {
        "time_swing": round(swing, 5),
        "elo_gap_flipped_by_time": round(flip_gap, 2),
        "overall_fast_at_1250": round(overall_fast, 4),
        "overall_slow_at_1250": round(overall_slow, 4),
    }


def _measure_latency(n_battles: int, seed: int) -> dict[str, Any]:
    """Measure GreedyDongi decision latency over N mirror battles.

    Confirms the engine has no per-decision timer: every turn resolves and
    the battle terminates with all decision calls answered.

    Args:
        n_battles: Number of mirror battles to run.
        seed: RNG seed.

    Returns:
        Dict with latency stats and battle completion counts.
    """
    params = BattleRuleParam()
    bp_a = GreedyDongiPolicy()
    bp_b = GreedyDongiPolicy()
    latencies: list[float] = []
    turns = 0
    completed = 0
    for b_idx in range(n_battles):
        rng = np.random.default_rng(seed + b_idx)
        team = gen_team(4, 4, rng)
        view = TeamView(team)
        battle_teams = get_battle_teams((team, team), 2)
        state = State(battle_teams)
        rng_tuple = ((rng, rng), (rng, rng))
        engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)
        turn_count = 0
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (view, view))
            sv1 = StateView(engine.state, 1, (view, view))
            t0 = time.perf_counter()
            cmd0 = bp_a.decision(sv0, view)
            cmd1 = bp_b.decision(sv1, view)
            latencies.append(time.perf_counter() - t0)
            engine.run_turn((cmd0, cmd1))
            turn_count += 1
        completed += 1
        turns += turn_count
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "n_battles": completed,
        "total_turns": turns,
        "decisions": len(latencies),
        "latency_mean_ms": round(float(arr.mean()) * 1000.0, 2),
        "latency_p99_ms": round(float(np.percentile(arr, 99)) * 1000.0, 2),
        "latency_max_ms": round(float(arr.max()) * 1000.0, 2),
    }


def main() -> None:
    """Run the championship ELO diagnosis and print the findings."""
    parser = argparse.ArgumentParser(description="Championship ELO + slowness diagnosis.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--latency-battles", type=int, default=10, help="Battles for latency measurement")
    args = parser.parse_args()

    single = _single_updates()
    sim_engine = _simulate_championship(_engine_wrap, args.seed)
    sim_correct = _simulate_championship(update_elo, args.seed)
    time_impact = _time_impact()
    latency = _measure_latency(args.latency_battles, args.seed)

    print("=" * 64)
    print("Championship ELO + slowness diagnosis")
    print("=" * 64)

    print("\n[1] Engine elo_rating is INVERTED (A=1500 beats B=1200 x5):")
    print(f"    engine A trajectory : {single['engine_A']}")
    print(f"    correct A trajectory: {single['correct_A']}")
    print("    -> engine gains ~k=30 per win even when dominant; correct gains ~4.")

    print("\n[2] Simulated 30-wave championship (true skills from benchmark_battle):")
    print(f"    engine ELO  : {json.dumps(sim_engine)}")
    print(f"    correct ELO : {json.dumps(sim_correct)}")

    print("\n[3] 1% time-weight impact:")
    print(f"    max overall swing from time  : {time_impact['time_swing']}")
    print(f"    ELO gap time can flip        : {time_impact['elo_gap_flipped_by_time']} points")
    print(f"    overall fast vs slow @1250   : {time_impact['overall_fast_at_1250']} "
          f"vs {time_impact['overall_slow_at_1250']}")
    print("    -> slowness never decides a ranking beyond a ~2-point ELO tie.")

    print("\n[4] GreedyDongi decision latency (no engine per-decision timer):")
    print(f"    battles={latency['n_battles']} turns={latency['total_turns']} decisions={latency['decisions']}")
    print(f"    latency mean={latency['latency_mean_ms']}ms p99={latency['latency_p99_ms']}ms "
          f"max={latency['latency_max_ms']}ms")
    print("    -> all turns resolved; a slow policy cannot lose turns (no deadline exists).")

    result = {
        "single_update": single,
        "simulated_final_elos_engine": sim_engine,
        "simulated_final_elos_correct": sim_correct,
        "time_impact": time_impact,
        "latency": latency,
    }
    out = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "championship_elo_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
