"""Teambuild seed determinism and diversity diagnosis.

Builds the championship roster exactly as ``scripts/championship_track.py``
does and asks three questions about ``HesfTeamBuildPolicy(ga_seed=X)``:

1. Is the build deterministic for a fixed seed? (Stage 3 battle royale
   uses an unseeded ``default_rng``, so this is expected to fail today.)
2. Do different seeds produce different teams?
3. How diverse are the teams across seeds?

If seeds diverge AND a deterministic build is achievable (after seeding
the battle royale), a downstream seed-quality search becomes meaningful.

Usage:
    uv run python scripts/experiments/teambuild_seed_search.py --seeds=0,1,2,3,4,5,6,7,8,9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.balance.meta import BasicMeta
from vgc2.competition.ecosystem import label_roster, sanitized_team_build_decision
from vgc2.util.generator import gen_move_set, gen_pkm_roster

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.teambuild.policy import HesfTeamBuildPolicy

N_MOVES = 100
ROSTER_SIZE = 50
MAX_TEAM_SIZE = 4
MAX_PKM_MOVES = 4
N_ACTIVE = 2


def _build_roster(seed: int) -> tuple[list[Any], list[Any]]:
    """Generate and label the championship roster.

    Args:
        seed: RNG seed for roster generation.

    Returns:
        Tuple of (move_set, roster) with ids labelled.
    """
    rng = np.random.default_rng(seed)
    move_set = gen_move_set(N_MOVES, rng)
    roster = gen_pkm_roster(ROSTER_SIZE, move_set, MAX_PKM_MOVES, rng)
    label_roster(move_set, roster)
    BasicMeta(move_set, roster)
    return move_set, roster


def _build_commands(ga_seed: int, roster: list[Any]) -> list[Any]:
    """Build a team command for a given GA seed.

    Args:
        ga_seed: Seed for the evolutionary stage.
        roster: Labelled championship roster.

    Returns:
        The sanitized TeamBuildCommand list.
    """
    policy = HesfTeamBuildPolicy(ga_seed=ga_seed)
    return list(sanitized_team_build_decision(policy, roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE))


def _species_fingerprint(commands: list[Any]) -> tuple[int, ...]:
    """Extract the ordered species roster indices from a build command.

    Args:
        commands: TeamBuildCommand list.

    Returns:
        Tuple of roster indices in team order.
    """
    return tuple(int(cmd[0]) for cmd in commands)


def main() -> None:
    """Run the seed determinism/diversity diagnosis and print findings."""
    parser = argparse.ArgumentParser(description="Teambuild seed determinism and diversity diagnosis.")
    parser.add_argument("--roster-seed", type=int, default=42, help="RNG seed for the championship roster")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated GA seeds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    _move_set, roster = _build_roster(args.roster_seed)

    print("=" * 64)
    print("Teambuild seed determinism / diversity diagnosis")
    print(f"  roster_seed={args.roster_seed}  roster_size={len(roster)}  ga_seeds={seeds}")
    print("=" * 64)

    per_seed: dict[str, Any] = {}
    unique_fingerprints: dict[tuple[int, ...], list[int]] = {}
    build_times: list[float] = []

    for seed in seeds:
        t0 = time.perf_counter()
        cmd_a = _build_commands(seed, roster)
        cmd_b = _build_commands(seed, roster)
        build_times.append(time.perf_counter() - t0)
        fp_a = _species_fingerprint(cmd_a)
        fp_b = _species_fingerprint(cmd_b)
        deterministic = fp_a == fp_b
        per_seed[str(seed)] = {
            "call_a": list(fp_a),
            "call_b": list(fp_b),
            "deterministic": deterministic,
        }
        unique_fingerprints.setdefault(fp_a, []).append(seed)
        if not deterministic:
            unique_fingerprints.setdefault(fp_b, []).append(seed)

    print(f"\n  mean build time per call: {sum(build_times) / len(build_times):.3f}s")

    print("\n  seed -> team species (call A == call B?):")
    for seed in seeds:
        rec = per_seed[str(seed)]
        marker = "DET" if rec["deterministic"] else "STOCHASTIC"
        print(f"    seed {seed:2d} -> {rec['call_a']}  call_b={rec['call_b']}  [{marker}]")

    n_distinct = len(unique_fingerprints)
    print(f"\n  distinct teams across seeds: {n_distinct} / {len(seeds)}")
    print("  -> if seeds diverge AND builds are stochastic, the battle-royale")
    print("     (unseeded default_rng) must be seeded before a seed search is meaningful.")

    result = {"roster_seed": args.roster_seed, "seeds": seeds, "per_seed": per_seed, "distinct_teams": n_distinct}
    out = Path(__file__).resolve().parent.parent.parent / "data" / "experiments" / "teambuild_seed_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
