"""MP selection benchmarks + BC data generation.

Runs all steps SEQUENTIALLY to avoid memory overload on EC2.
Logs everything to execute_benchmark_bc.log for morning review.

Usage:
    uv run python scripts/benchmark/execute_benchmark_bc.py
    uv run python scripts/benchmark/execute_benchmark_bc.py --smoke   # quick 1-round smoke test
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = PROJECT_ROOT / "execute_benchmark_bc.log"

BENCHMARK_COMMON_ARGS = [
    "--build-size", "6",
    "--battle-policy", "dongimon",
    "--seed", "42",
    "--n-top", "2",
    "--opponents", "JJJ,caaaden",
]

SELECTION_MODES = ["mp_only", "mp_sim"]


def _log(msg: str) -> None:
    """Print and append a timestamped message to the log file."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_step(step_name: str, cmd: list[str], env: dict[str, str] | None = None) -> bool:
    """Run a subprocess step, streaming output to console and log.

    Args:
        step_name: Human-readable label for this step.
        cmd: Command list to execute.
        env: Environment variables for the subprocess.

    Returns:
        True if the step succeeded (exit code 0), False otherwise.
    """
    _log(f"START: {step_name}")
    _log(f"  cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=21600,
            env=env,
        )
        elapsed = time.perf_counter() - t0

        if proc.stdout:
            for line in proc.stdout.splitlines():
                _log(f"  | {line}")
        if proc.stderr:
            for line in proc.stderr.splitlines():
                _log(f"  ERR| {line}")

        if proc.returncode == 0:
            _log(f"DONE: {step_name} ({elapsed:.1f}s)")
            return True
        else:
            _log(f"FAILED: {step_name} (exit={proc.returncode}, {elapsed:.1f}s)")
            return False

    except subprocess.TimeoutExpired:
        _log(f"TIMEOUT: {step_name} (exceeded 6h limit)")
        return False
    except Exception as exc:
        _log(f"ERROR: {step_name} — {exc}")
        return False


def main() -> None:
    """Run all benchmark and data-generation steps sequentially."""
    parser = argparse.ArgumentParser(
        description="Overnight orchestration: MP benchmarks + BC data gen"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test: 1 round, 5 battles, 10 BC battles",
    )
    args = parser.parse_args()

    if args.smoke:
        n_rounds, n_battles, bc_battles = "1", "5", "10"
        _log("=" * 64)
        _log("SMOKE TEST MODE (minimal rounds/battles)")
        _log("=" * 64)
    else:
        n_rounds, n_battles, bc_battles = "5", "10", "12000"
        _log("=" * 64)
        _log("FULL RUN: MP Selection Benchmarks + BC Data Generation")
        _log("  Optimized: n_top=2, opponents=JJJ+caaaden, 5 rounds, 10 battles")
        _log("=" * 64)

    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + env.get("PATH", "")
    uv = "uv"
    results: dict[str, bool] = {}
    total_t0 = time.perf_counter()

    for mode in SELECTION_MODES:
        step_name = f"Selection benchmark: {mode}"
        cmd = [
            uv, "run", "python", "scripts/benchmark/benchmark_team.py",
            *BENCHMARK_COMMON_ARGS,
            "--n-rounds", n_rounds,
            "--n-battles", n_battles,
            "--selection-mode", mode,
            "--tag", f"overnight_{mode}",
        ]
        results[step_name] = _run_step(step_name, cmd, env)

    step_name = "BC data generation (10K battles, win-filter)"
    cmd = [
        uv, "run", "python", "-m", "src.tree_bc.generate_data",
        "--n-battles", bc_battles,
        "--seed", "42",
        "--output-dir", "data/BC",
        "--win-filter",
    ]
    results[step_name] = _run_step(step_name, cmd, env)

    total_elapsed = time.perf_counter() - total_t0
    _log("")
    _log("=" * 64)
    _log("SUMMARY")
    _log("=" * 64)
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        _log(f"  [{status}] {step}")
    _log(f"  Total elapsed: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    _log("=" * 64)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
