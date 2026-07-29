"""Matchup Predictor full-scale data generation pipeline.

Generates stratified team pairs, runs N battles per pair with Greedy
labels on both sides, computes 56 pairwise features on the 4-member
subteams, and saves to JSONL with checkpoint/resume support for EC2.

Usage:
    uv run python -m src.data.generate \
        --n-pairs=24000 --n-battles=50 --seed=42 --output-dir=data/MP

Resume after crash:
    uv run python -m src.data.generate --resume --output-dir=data/MP
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleRuleParam

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiments.experiment_utils import (
    generate_stratified_teams,
    profile_teams,
    run_pair_battles,
)
from src.data.features import (
    BST_FEATURE_NAMES,
    compute_pairwise_features,
    compute_subteam_features,
)

DEFAULT_N_PAIRS = 24000
DEFAULT_N_BATTLES = 50
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("data/MP")
DEFAULT_FLUSH_INTERVAL = 200


def _save_jsonl_append(rows: list[dict[str, Any]], path: Path) -> None:
    """Append list of row dicts as JSONL to an existing or new file.

    Args:
        rows: List of row dicts to append.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _save_checkpoint(
    output_dir: Path,
    timestamp: str,
    seed: int,
    n_pairs: int,
    n_battles: int,
    data_file: str,
    last_completed_pair: int,
    status: str,
) -> None:
    """Write checkpoint state to mp_checkpoint.json.

    Args:
        output_dir: Directory for checkpoint file.
        timestamp: Run timestamp string.
        seed: Master RNG seed.
        n_pairs: Total number of pairs.
        n_battles: Battles per pair.
        data_file: Filename of JSONL data file.
        last_completed_pair: Index of last completed pair (-1 if none).
        status: One of "in_progress" or "complete".
    """
    checkpoint = {
        "timestamp": timestamp,
        "seed": seed,
        "n_pairs": n_pairs,
        "n_battles": n_battles,
        "data_file": data_file,
        "last_completed_pair": last_completed_pair,
        "status": status,
    }
    path = output_dir / "mp_checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def _load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    """Load checkpoint from mp_checkpoint.json if it exists.

    Args:
        output_dir: Directory containing checkpoint file.

    Returns:
        Checkpoint dict, or None if no checkpoint found.
    """
    path = output_dir / "mp_checkpoint.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _count_jsonl_lines(path: Path) -> int:
    """Count lines in a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        Number of lines (0 if file does not exist).
    """
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def _generate_teams(
    n_pairs: int,
    seed: int,
) -> tuple[list[Any], list[Any], list[str]]:
    """Generate stratified teams deterministically from seed.

    Teams are reproducible from the same seed, so on resume we simply
    re-generate rather than persisting to disk.

    Args:
        n_pairs: Number of pairs (generates n_pairs * 2 teams).
        seed: RNG seed for reproducible generation.

    Returns:
        Tuple of (teams_a, teams_b, tier_labels).
    """
    n_teams = n_pairs * 2
    print(f"Generating {n_teams} stratified teams...")
    all_teams, tier_labels = generate_stratified_teams(n_teams=n_teams, seed=seed)

    teams_a = all_teams[:n_pairs]
    teams_b = all_teams[n_pairs:]

    tier_counts: dict[str, int] = {}
    for t in tier_labels:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"  Tier distribution: {tier_counts}")
    print(f"  Generated {len(teams_a)} pairs")

    return teams_a, teams_b, tier_labels


def run_pipeline(
    n_pairs: int,
    n_battles: int,
    seed: int,
    output_dir: Path,
    flush_interval: int,
    resume: bool,
) -> None:
    """Execute the full data generation pipeline.

    Generates team pairs, runs battles with Greedy labels, computes
    pairwise features, and saves to JSONL with periodic checkpointing.

    Args:
        n_pairs: Number of team pairings to generate.
        n_battles: Number of battles per pairing.
        seed: Master RNG seed.
        output_dir: Output directory for all artifacts.
        flush_interval: Flush JSONL buffer every N pairs.
        resume: If True, attempt to resume from checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    greedy_a = GreedyBattlePolicy()
    greedy_b = GreedyBattlePolicy()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    checkpoint = _load_checkpoint(output_dir) if resume else None
    start_pair = 0

    if checkpoint is not None and checkpoint.get("status") == "in_progress":
        timestamp = checkpoint["timestamp"]
        seed = checkpoint["seed"]
        n_pairs = checkpoint["n_pairs"]
        n_battles = checkpoint["n_battles"]
        data_filename = checkpoint["data_file"]
        start_pair = checkpoint["last_completed_pair"] + 1

        print("=" * 60)
        print("Matchup Predictor Data Generation (RESUME)")
        print(f"  Resuming from pair {start_pair}/{n_pairs}")
        print(f"  seed={seed}, n_battles={n_battles}")
        print("=" * 60)

        teams_a, teams_b, tier_labels = _generate_teams(n_pairs=n_pairs, seed=seed)
        existing_lines = _count_jsonl_lines(output_dir / data_filename)
        print(f"  Existing JSONL lines: {existing_lines}")
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        data_filename = f"mp_data_{timestamp}.jsonl"

        print("=" * 60)
        print("Matchup Predictor Data Generation")
        print(f"  seed={seed}, n_pairs={n_pairs}, n_battles={n_battles}")
        print("  side-A policy: Greedy")
        print("  side-B policy: Greedy")
        print(f"  output={output_dir.resolve()}")
        print(f"  flush_interval={flush_interval}")
        print("=" * 60)

        teams_a, teams_b, tier_labels = _generate_teams(n_pairs=n_pairs, seed=seed)

        profile = profile_teams(teams_a + teams_b)
        _save_checkpoint(
            output_dir=output_dir,
            timestamp=timestamp,
            seed=seed,
            n_pairs=n_pairs,
            n_battles=n_battles,
            data_file=data_filename,
            last_completed_pair=-1,
            status="in_progress",
        )

    jsonl_path = output_dir / data_filename
    buffer: list[dict[str, Any]] = []

    print(f"\nRunning {n_pairs - start_pair} remaining pairs "
          f"({(n_pairs - start_pair) * n_battles} battles)...")

    for pair_idx in range(start_pair, n_pairs):
        team_a = teams_a[pair_idx]
        team_b = teams_b[pair_idx]
        pair_seed = seed + pair_idx * 100

        wins_a, _, idx_a, idx_b = run_pair_battles(
            team_a=team_a,
            team_b=team_b,
            bp_side_a=greedy_a,
            bp_side_b=greedy_b,
            n_battles=n_battles,
            pair_seed=pair_seed,
            params=params,
            sel=sel,
        )

        win_rate = wins_a / n_battles

        subteam_a_members = (
            [team_a.members[i] for i in idx_a] if idx_a else list(team_a.members[:4])
        )
        subteam_b_members = (
            [team_b.members[i] for i in idx_b] if idx_b else list(team_b.members[:4])
        )

        sub_feats_a = compute_subteam_features(subteam_a_members)
        sub_feats_b = compute_subteam_features(subteam_b_members)
        pair_feats = compute_pairwise_features(subteam_a_members, subteam_b_members)

        bst_delta = abs(
            sub_feats_a.get("bst_avg", 0.0) - sub_feats_b.get("bst_avg", 0.0)
        )

        buffer.append({
            "pair_id": pair_idx,
            "seed": pair_seed,
            "wins_a": wins_a,
            "n_battles": n_battles,
            "win_rate_a": win_rate,
            "selected_indices_a": idx_a,
            "selected_indices_b": idx_b,
            "bst_delta": bst_delta,
            "features": pair_feats,
        })

        if len(buffer) >= flush_interval:
            _save_jsonl_append(buffer, jsonl_path)
            buffer.clear()
            _save_checkpoint(
                output_dir=output_dir,
                timestamp=timestamp,
                seed=seed,
                n_pairs=n_pairs,
                n_battles=n_battles,
                data_file=data_filename,
                last_completed_pair=pair_idx,
                status="in_progress",
            )
            elapsed = time.perf_counter() - start
            completed = pair_idx - start_pair + 1
            rate = completed / elapsed if elapsed > 0 else 0
            eta_s = (n_pairs - pair_idx - 1) / rate if rate > 0 else 0
            print(
                f"  {pair_idx + 1}/{n_pairs} pairs "
                f"({rate:.2f} pairs/s, elapsed={elapsed:.0f}s, ETA={eta_s:.0f}s)"
            )

    if buffer:
        _save_jsonl_append(buffer, jsonl_path)
        buffer.clear()

    elapsed = time.perf_counter() - start
    print(f"\nData generation complete: {n_pairs} pairs in {elapsed:.0f}s")

    tier_counts: dict[str, int] = {}
    for t in tier_labels:
        tier_counts[t] = tier_counts.get(t, 0) + 1

    win_rates: list[float] = []
    feature_names: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            win_rates.append(float(row["win_rate_a"]))
            if not feature_names:
                feature_names = list(row["features"].keys())

    bst_indices = [
        feature_names.index(k) for k in BST_FEATURE_NAMES if k in feature_names
    ]

    meta = {
        "timestamp": timestamp,
        "config": {
            "n_pairs": n_pairs,
            "n_battles": n_battles,
            "seed": seed,
            "side_a_policy": "Greedy",
            "side_b_policy": "Greedy",
            "flush_interval": flush_interval,
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "bst_feature_names": BST_FEATURE_NAMES,
        "bst_feature_indices": bst_indices,
        "tier_counts": tier_counts,
        "label_stats": {
            "mean_win_rate_a": float(np.mean(win_rates)) if win_rates else 0.0,
            "std_win_rate_a": float(np.std(win_rates)) if win_rates else 0.0,
            "n_pairs": len(win_rates),
        },
        "duration_seconds": round(elapsed, 1),
        "generation_rate_pairs_per_sec": round(n_pairs / elapsed, 3) if elapsed > 0 else 0.0,
    }

    if "profile" in dir():
        meta["data_profile"] = profile

    meta_path = output_dir / f"mp_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    _save_checkpoint(
        output_dir=output_dir,
        timestamp=timestamp,
        seed=seed,
        n_pairs=n_pairs,
        n_battles=n_battles,
        data_file=data_filename,
        last_completed_pair=n_pairs - 1,
        status="complete",
    )

    print(f"  Mean win rate (side A): {meta['label_stats']['mean_win_rate_a']:.4f}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Metadata saved to {meta_path}")
    print(f"  Data saved to {jsonl_path}")
    print("=" * 60)


def main() -> None:
    """CLI entry point for MP data generation."""
    parser = argparse.ArgumentParser(
        description="Matchup Predictor full-scale data generation"
    )
    parser.add_argument(
        "--n-pairs", type=int, default=DEFAULT_N_PAIRS,
        help="Number of team pairings (default: 24000)",
    )
    parser.add_argument(
        "--n-battles", type=int, default=DEFAULT_N_BATTLES,
        help="Battles per pairing for label stability (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Master RNG seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/MP)",
    )
    parser.add_argument(
        "--flush-interval", type=int, default=DEFAULT_FLUSH_INTERVAL,
        help="Flush JSONL every N pairs (default: 200)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    run_pipeline(
        n_pairs=args.n_pairs,
        n_battles=args.n_battles,
        seed=args.seed,
        output_dir=args.output_dir,
        flush_interval=args.flush_interval,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
