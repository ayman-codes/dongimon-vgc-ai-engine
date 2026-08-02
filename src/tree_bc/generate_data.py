"""BC data generation pipeline.

Runs self-play battles with expert policies (JJJ, minimon, Caaaden,
Dongimon), recording per-turn (state_vector, joint_action, valid_actions,
battle_won) tuples for supervised behavioral cloning.

Usage:
    uv run python -m src.tree_bc.generate_data \
        --n-battles=10000 --seed=42 --output-dir=data/BC --win-filter
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.util.generator import gen_move_set, gen_pkm_roster, gen_team

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tree_bc.actions import encode_action, get_valid_actions
from src.tree_bc.encoder import FEATURE_DIM, encode_state

DEFAULT_N_BATTLES = 10000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("data/BC")
DEFAULT_FLUSH_INTERVAL = 500
EXPERT_WEIGHTS = {"Greedy": 0.40, "caaaden": 0.25, "minimon": 0.20, "JJJ": 0.10, "dongimon": 0.05}


def _save_jsonl_append(rows: list[dict[str, Any]], path: Path) -> None:
    """Append list of row dicts as JSONL to an existing or new file.

    Args:
        rows: List of row dicts to append.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(obj: object) -> Any:
    """JSON serialization fallback for numpy arrays and other types.

    Args:
        obj: Object that is not JSON-serializable by default.

    Returns:
        JSON-compatible representation.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_checkpoint(
    output_dir: Path,
    timestamp: str,
    seed: int,
    n_battles: int,
    data_file: str,
    last_completed_battle: int,
    win_filter: bool,
    status: str,
) -> None:
    """Write checkpoint state to bc_checkpoint.json.

    Args:
        output_dir: Directory for checkpoint file.
        timestamp: Run timestamp string.
        seed: Master RNG seed.
        n_battles: Total number of battles.
        data_file: Filename of JSONL data file.
        last_completed_battle: Index of last completed battle (-1 if none).
        win_filter: Whether win-only filtering is enabled.
        status: One of "in_progress" or "complete".
    """
    checkpoint = {
        "timestamp": timestamp,
        "seed": seed,
        "n_battles": n_battles,
        "data_file": data_file,
        "last_completed_battle": last_completed_battle,
        "win_filter": win_filter,
        "status": status,
    }
    path = output_dir / "bc_checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def _load_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    """Load checkpoint from bc_checkpoint.json if it exists.

    Args:
        output_dir: Directory containing checkpoint file.

    Returns:
        Checkpoint dict, or None if no checkpoint found.
    """
    path = output_dir / "bc_checkpoint.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _load_expert_policies() -> dict[str, Any]:
    """Import and return all expert battle policies.

    Returns:
        Dict mapping expert name to battle policy instance.
    """
    from vgc2.agent.battle import GreedyBattlePolicy

    from competitor import DongimonCompetitor
    from competitors.competitor1_jjj import JJJ_Competitor
    from competitors.competitor2_minimon import minimon
    from competitors.competitor_caaaden import CaaadenCompetitor

    return {
        "Greedy": GreedyBattlePolicy(),
        "caaaden": CaaadenCompetitor().battlepolicy,
        "minimon": minimon().battlepolicy,
        "JJJ": JJJ_Competitor().battlepolicy,
        "dongimon": DongimonCompetitor().battlepolicy,
    }


def run_pipeline(
    n_battles: int,
    seed: int,
    output_dir: Path,
    flush_interval: int,
    win_filter: bool,
    resume: bool,
) -> None:
    """Execute the full BC data generation pipeline.

    Runs self-play battles with weighted expert selection, recording
    per-turn state-action-outcome tuples to JSONL with checkpointing.

    The global numpy and stdlib RNGs are seeded with ``seed`` so expert
    fallbacks (which draw from module-level RNGs) are reproducible, and
    expert commands that fall outside the valid-action mask are repaired
    to a legal action so every recorded training label is usable.

    Args:
        n_battles: Number of battles to generate.
        seed: Master RNG seed.
        output_dir: Output directory for all artifacts.
        flush_interval: Flush JSONL buffer every N battles.
        win_filter: If True, only keep turns from battles the expert won.
        resume: If True, attempt to resume from checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    np.random.seed(seed)
    random.seed(seed)

    params = BattleRuleParam()
    experts = _load_expert_policies()
    expert_names = list(EXPERT_WEIGHTS.keys())
    expert_probs = list(EXPERT_WEIGHTS.values())

    checkpoint = _load_checkpoint(output_dir) if resume else None
    start_battle = 0

    if checkpoint is not None and checkpoint.get("status") == "in_progress":
        timestamp = checkpoint["timestamp"]
        seed = checkpoint["seed"]
        n_battles = checkpoint["n_battles"]
        data_filename = checkpoint["data_file"]
        start_battle = checkpoint["last_completed_battle"] + 1
        win_filter = checkpoint.get("win_filter", win_filter)

        print("=" * 60)
        print("BC Data Generation (RESUME)")
        print(f"  Resuming from battle {start_battle}/{n_battles}")
        print(f"  seed={seed}, win_filter={win_filter}")
        print("=" * 60)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        data_filename = f"bc_data_{timestamp}.jsonl"

        print("=" * 60)
        print("BC Data Generation")
        print(f"  seed={seed}, n_battles={n_battles}")
        print(f"  win_filter={win_filter}")
        print(f"  expert weights: {EXPERT_WEIGHTS}")
        print(f"  output={output_dir.resolve()}")
        print(f"  flush_interval={flush_interval}")
        print("=" * 60)

        _save_checkpoint(
            output_dir=output_dir,
            timestamp=timestamp,
            seed=seed,
            n_battles=n_battles,
            data_file=data_filename,
            last_completed_battle=-1,
            win_filter=win_filter,
            status="in_progress",
        )

    jsonl_path = output_dir / data_filename
    buffer: list[dict[str, Any]] = []
    total_records = 0
    total_filtered = 0

    print(f"\nRunning {n_battles - start_battle} remaining battles...")

    for battle_idx in range(start_battle, n_battles):
        battle_seed = seed + battle_idx * 1000
        battle_rng = np.random.default_rng(battle_seed)

        expert_name = battle_rng.choice(expert_names, p=expert_probs)
        expert = experts[expert_name]

        move_set = gen_move_set(200, battle_rng)
        gen_pkm_roster(30, move_set)
        team_a = gen_team(4, 4, battle_rng)
        team_b = gen_team(4, 4, battle_rng)

        battle_teams = get_battle_teams((team_a, team_b), 2)
        state_obj = State(battle_teams)
        rng_tuple = ((battle_rng, battle_rng), (battle_rng, battle_rng))
        engine = BattleEngine(
            state_obj, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )

        view_a = TeamView(team_a)
        view_b = TeamView(team_b)

        battle_records: list[dict[str, Any]] = []
        turn = 0

        while not engine.finished():
            sv_a = StateView(engine.state, 0, (view_a, view_b))
            sv_b = StateView(engine.state, 1, (view_b, view_a))

            cmd_a = list(expert.decision(sv_a, view_b))
            cmd_b = list(expert.decision(sv_b, view_a))

            state_vec_a = encode_state(sv_a)
            action_idx_a = encode_action(cmd_a)
            valid_actions_a = get_valid_actions(sv_a)

            state_vec_b = encode_state(sv_b)
            action_idx_b = encode_action(cmd_b)
            valid_actions_b = get_valid_actions(sv_b)

            if action_idx_a not in valid_actions_a and valid_actions_a:
                action_idx_a = valid_actions_a[0]
            if action_idx_b not in valid_actions_b and valid_actions_b:
                action_idx_b = valid_actions_b[0]

            battle_records.append({
                "battle_id": battle_idx,
                "turn": turn,
                "side": 0,
                "expert": expert_name,
                "features": state_vec_a,
                "action_idx": action_idx_a,
                "valid_actions": valid_actions_a,
                "won": False,
            })
            battle_records.append({
                "battle_id": battle_idx,
                "turn": turn,
                "side": 1,
                "expert": expert_name,
                "features": state_vec_b,
                "action_idx": action_idx_b,
                "valid_actions": valid_actions_b,
                "won": False,
            })

            engine.run_turn((cmd_a, cmd_b))
            turn += 1

        winner = engine.winning_side
        for rec in battle_records:
            rec["won"] = (rec["side"] == winner)

        if win_filter:
            filtered = [r for r in battle_records if r["won"]]
            total_filtered += len(battle_records) - len(filtered)
            battle_records = filtered

        buffer.extend(battle_records)
        total_records += len(battle_records)

        if (battle_idx - start_battle + 1) % flush_interval == 0:
            _save_jsonl_append(buffer, jsonl_path)
            buffer.clear()
            _save_checkpoint(
                output_dir=output_dir,
                timestamp=timestamp,
                seed=seed,
                n_battles=n_battles,
                data_file=data_filename,
                last_completed_battle=battle_idx,
                win_filter=win_filter,
                status="in_progress",
            )
            elapsed = time.perf_counter() - start
            completed = battle_idx - start_battle + 1
            rate = completed / elapsed if elapsed > 0 else 0
            eta_s = (n_battles - battle_idx - 1) / rate if rate > 0 else 0
            print(
                f"  {battle_idx + 1}/{n_battles} battles "
                f"({rate:.2f} battles/s, {total_records} records, "
                f"elapsed={elapsed:.0f}s, ETA={eta_s:.0f}s)"
            )

    if buffer:
        _save_jsonl_append(buffer, jsonl_path)
        buffer.clear()

    elapsed = time.perf_counter() - start

    _save_checkpoint(
        output_dir=output_dir,
        timestamp=timestamp,
        seed=seed,
        n_battles=n_battles,
        data_file=data_filename,
        last_completed_battle=n_battles - 1,
        win_filter=win_filter,
        status="complete",
    )

    win_rates: list[float] = []
    action_counts: dict[int, int] = {}
    expert_counts: dict[str, int] = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            if row["won"]:
                win_rates.append(1.0)
            else:
                win_rates.append(0.0)
            action_counts[row["action_idx"]] = action_counts.get(row["action_idx"], 0) + 1
            expert_counts[row["expert"]] = expert_counts.get(row["expert"], 0) + 1

    meta = {
        "timestamp": timestamp,
        "config": {
            "n_battles": n_battles,
            "seed": seed,
            "win_filter": win_filter,
            "expert_weights": EXPERT_WEIGHTS,
            "flush_interval": flush_interval,
        },
        "record_stats": {
            "total_records": total_records,
            "total_filtered_out": total_filtered,
            "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
        },
        "expert_distribution": expert_counts,
        "feature_dim": FEATURE_DIM,
        "duration_seconds": round(elapsed, 1),
        "generation_rate_battles_per_sec": round(n_battles / elapsed, 3) if elapsed > 0 else 0.0,
    }

    meta_path = output_dir / f"bc_meta_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nData generation complete: {n_battles} battles in {elapsed:.0f}s")
    print(f"  Records: {total_records} | Filtered out: {total_filtered}")
    print(f"  Expert distribution: {expert_counts}")
    print(f"  Metadata saved to {meta_path}")
    print(f"  Data saved to {jsonl_path}")
    print("=" * 60)


def main() -> None:
    """CLI entry point for BC data generation."""
    parser = argparse.ArgumentParser(
        description="BC data generation: self-play battles with expert policies"
    )
    parser.add_argument(
        "--n-battles", type=int, default=DEFAULT_N_BATTLES,
        help="Number of battles to generate (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Master RNG seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/BC)",
    )
    parser.add_argument(
        "--flush-interval", type=int, default=DEFAULT_FLUSH_INTERVAL,
        help="Flush JSONL every N battles (default: 500)",
    )
    parser.add_argument(
        "--win-filter", action="store_true",
        help="Only keep turns from battles the recorded side won",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    run_pipeline(
        n_battles=args.n_battles,
        seed=args.seed,
        output_dir=args.output_dir,
        flush_interval=args.flush_interval,
        win_filter=args.win_filter,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
