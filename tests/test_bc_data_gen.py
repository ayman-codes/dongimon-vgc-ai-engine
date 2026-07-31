"""Tests for BC data generation pipeline."""

import json
import tempfile
from pathlib import Path

from src.tree_bc.actions import decode_action
from src.tree_bc.encoder import FEATURE_DIM
from src.tree_bc.generate_data import (
    _load_checkpoint,
    _save_checkpoint,
    run_pipeline,
)


class TestCheckpoint:
    """Validates checkpoint save/load/resume functionality."""

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _save_checkpoint(
                output_dir=output_dir,
                timestamp="20260101_120000",
                seed=42,
                n_battles=100,
                data_file="bc_data_test.jsonl",
                last_completed_battle=49,
                win_filter=True,
                status="in_progress",
            )
            ckpt = _load_checkpoint(output_dir)
            assert ckpt is not None
            assert ckpt["seed"] == 42
            assert ckpt["n_battles"] == 100
            assert ckpt["last_completed_battle"] == 49
            assert ckpt["win_filter"] is True
            assert ckpt["status"] == "in_progress"

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _load_checkpoint(Path(tmp))
            assert ckpt is None


class TestGenerateDataIntegration:
    """Integration tests for the BC data generation pipeline."""

    def test_single_battle_produces_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=1,
                seed=42,
                output_dir=output_dir,
                flush_interval=1,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            assert len(jsonl_files) == 1
            records = []
            with open(jsonl_files[0]) as f:
                for line in f:
                    records.append(json.loads(line))
            assert len(records) > 0, "Single battle must produce at least one record"

    def test_record_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=1,
                seed=42,
                output_dir=output_dir,
                flush_interval=1,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            with open(jsonl_files[0]) as f:
                record = json.loads(f.readline())

            required = {"battle_id", "turn", "side", "expert",
                        "features", "action_idx", "valid_actions", "won"}
            for key in required:
                assert key in record, f"Missing field: {key}"

    def test_feature_shape_matches_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=1,
                seed=42,
                output_dir=output_dir,
                flush_interval=1,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            with open(jsonl_files[0]) as f:
                record = json.loads(f.readline())
            features = record["features"]
            assert len(features) == FEATURE_DIM, (
                f"Feature dim {len(features)} != {FEATURE_DIM}"
            )

    def test_action_in_valid_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=3,
                seed=42,
                output_dir=output_dir,
                flush_interval=10,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            with open(jsonl_files[0]) as f:
                for line in f:
                    record = json.loads(line)
                    assert record["action_idx"] in record["valid_actions"], (
                        f"Action {record['action_idx']} not in valid_actions "
                        f"for battle {record['battle_id']} turn {record['turn']}"
                    )

    def test_win_filter_only_keeps_winning_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=3,
                seed=42,
                output_dir=output_dir,
                flush_interval=10,
                win_filter=True,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            if not jsonl_files:
                return
            with open(jsonl_files[0]) as f:
                for line in f:
                    record = json.loads(line)
                    assert record["won"] is True, (
                        f"win_filter enabled but got won=False for "
                        f"battle {record['battle_id']}"
                    )

    def test_checkpoint_resume_no_duplicate_battle_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            run_pipeline(
                n_battles=2,
                seed=42,
                output_dir=output_dir,
                flush_interval=1,
                win_filter=False,
                resume=False,
            )

            ckpt = _load_checkpoint(output_dir)
            assert ckpt is not None
            ckpt["status"] = "in_progress"
            ckpt["n_battles"] = 5
            _save_checkpoint(
                output_dir=output_dir,
                timestamp=ckpt["timestamp"],
                seed=ckpt["seed"],
                n_battles=5,
                data_file=ckpt["data_file"],
                last_completed_battle=ckpt["last_completed_battle"],
                win_filter=ckpt["win_filter"],
                status="in_progress",
            )

            run_pipeline(
                n_battles=5,
                seed=42,
                output_dir=output_dir,
                flush_interval=1,
                win_filter=False,
                resume=True,
            )

            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            assert len(jsonl_files) == 1

            record_keys: list[tuple[int, int, int]] = []
            with open(jsonl_files[0]) as f:
                for line in f:
                    record = json.loads(line)
                    record_keys.append(
                        (record["battle_id"], record["turn"], record["side"])
                    )

            assert len(record_keys) == len(set(record_keys)), (
                f"Resume produced duplicate (battle_id, turn, side) tuples: "
                f"{len(record_keys)} records but {len(set(record_keys))} unique"
            )

    def test_multi_battle_action_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=10,
                seed=42,
                output_dir=output_dir,
                flush_interval=20,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            actions: dict[str, int] = {}
            with open(jsonl_files[0]) as f:
                for line in f:
                    record = json.loads(line)
                    aidx = record["action_idx"]
                    actions[str(aidx)] = actions.get(str(aidx), 0) + 1
            unique_actions = len(actions)
            assert unique_actions > 1, (
                f"Only {unique_actions} unique actions across 10 battles; "
                f"expecting diverse action distribution"
            )

    def test_expert_names_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=5,
                seed=42,
                output_dir=output_dir,
                flush_interval=10,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            valid_experts = {"JJJ", "minimon", "caaaden", "dongimon"}
            with open(jsonl_files[0]) as f:
                for line in f:
                    record = json.loads(line)
                    assert record["expert"] in valid_experts, (
                        f"Unknown expert: {record['expert']}"
                    )

    def test_decode_encode_roundtrip_on_recorded_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_pipeline(
                n_battles=3,
                seed=42,
                output_dir=output_dir,
                flush_interval=10,
                win_filter=False,
                resume=False,
            )
            jsonl_files = sorted(output_dir.glob("bc_data_*.jsonl"))
            with open(jsonl_files[0]) as f:
                for line in f:
                    record = json.loads(line)
                    commands = decode_action(record["action_idx"])
                    assert len(commands) == 2
                    for cmd in commands:
                        action_val, target_val = cmd
                        assert action_val in (-1, 0, 1, 2, 3)
                        assert target_val in (0, 1)
