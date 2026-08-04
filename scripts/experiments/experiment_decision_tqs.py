"""Experiment: Fair TQS Decision Quality evaluation.

Compares GA with pure heuristic fitness (control) against GA with hybrid
TQS+heuristic fitness (treatment) in a head-to-head battle evaluation.
Both arms share the same species pool per scenario to eliminate
pool-quality confounds. Battle policy is Greedy for both sides to
neutralize battle-policy bias.

Usage:
    uv run python scripts/experiments/experiment_decision_TQS.py \
        --n-scenarios=100 --n-battles=30 --tqs-weight=0.4 --seed=42
"""

import argparse
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleRuleParam

from scripts.experiments.experiment_utils import (
    _build_team_from_indices,
    _get_pool,
    run_pair_battles,
)
from src.data.features import compute_subteam_features
from src.shared.s3 import sync_from_s3
from src.teambuild.evolution import run_evolution
from src.teambuild.operators import calculate_team_fitness


def _load_tqs_model(model_path: Path) -> dict[str, Any]:
    """Load a pickled TQS model bundle.

    Args:
        model_path: Path to the TQS_*.pkl file.

    Returns:
        Dict with keys: model, scaler, feature_names, config.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"TQS model not found: {model_path}")
    with open(model_path, "rb") as f:
        bundle: dict[str, Any] = pickle.load(f)  # noqa: S301
    return bundle


def _predict_tqs_score(
    team_indices: list[int],
    pool_species: list[Any],
    tqs_bundle: dict[str, Any],
) -> float:
    """Predict TQS quality score for a team.

    Builds feature vector from team members, scales with the TQS
    scaler, and returns the model prediction.

    Args:
        team_indices: List of species indices into pool_species.
        pool_species: Full species pool.
        tqs_bundle: Loaded TQS model bundle.

    Returns:
        Predicted quality score (higher = better).
    """
    members: list[Any] = []
    for idx in team_indices:
        build = _build_team_from_indices([idx], pool_species)
        if build is not None:
            members.extend(build.members)

    if len(members) < 4:
        return 0.0

    feat_dict = compute_subteam_features(members[:6])
    feature_names: list[str] = tqs_bundle["feature_names"]
    x_vec = np.array(
        [[feat_dict.get(n, 0.0) for n in feature_names]], dtype=np.float64
    )
    scaler = tqs_bundle["scaler"]
    model = tqs_bundle["model"]
    x_scaled = scaler.transform(x_vec)
    prediction: float = float(model.predict(x_scaled)[0])
    return prediction


def _run_ga_with_fitness(
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    rng: np.random.Generator,
    tqs_bundle: dict[str, Any] | None,
    tqs_weight: float,
    pop_size: int,
    generations: int,
) -> list[int]:
    """Run GA evolution and return the best team indices.

    When tqs_bundle is None or tqs_weight is 0, uses pure heuristic
    fitness. Otherwise uses hybrid: tqs_weight * TQS + (1-tqs_weight)
    * heuristic.

    Args:
        pool_species: Species pool for evolution.
        viability_scores: Viability scores per species.
        rng: NumPy Generator for reproducibility.
        tqs_bundle: Loaded TQS model bundle, or None for pure heuristic.
        tqs_weight: Weight for TQS component (0.0 = pure heuristic).
        pop_size: GA population size.
        generations: Number of GA generations.

    Returns:
        Best team as list of species indices.
    """
    if tqs_bundle is None or tqs_weight <= 0.0:
        evo_results = run_evolution(
            pool_species=pool_species,
            viability_scores=viability_scores,
            team_size=4,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=0.10,
            elite_fraction=0.10,
            rng=rng,
        )
        return list(evo_results[0]) if evo_results else list(range(6))

    heuristic_scores: dict[tuple[int, ...], float] = {}
    tqs_scores: dict[tuple[int, ...], float] = {}

    def _hybrid_fitness(team_indices: list[int]) -> float:
        key = tuple(sorted(team_indices))
        if key not in heuristic_scores:
            heuristic_scores[key] = calculate_team_fitness(
                team_indices, pool_species, viability_scores
            )
        if key not in tqs_scores:
            tqs_scores[key] = _predict_tqs_score(
                team_indices, pool_species, tqs_bundle
            )
        h_score = heuristic_scores[key]
        t_score = tqs_scores[key]
        return tqs_weight * t_score + (1.0 - tqs_weight) * h_score

    population: list[list[int]] = []
    team_size = 4
    n_pool = len(pool_species)

    init_rng = np.random.default_rng(rng.integers(0, 2**31))
    for _ in range(pop_size):
        team = list(
            init_rng.choice(n_pool, size=team_size, replace=False)
        )
        population.append(team)

    for _gen in range(generations):
        fitnesses = [_hybrid_fitness(t) for t in population]
        ranked = sorted(
            zip(population, fitnesses, strict=False), key=lambda x: -x[1]
        )
        elite_count = max(1, int(pop_size * 0.10))
        next_pop = [team for team, _ in ranked[:elite_count]]

        while len(next_pop) < pop_size:
            candidates = init_rng.choice(
                len(population), size=3, replace=False
            )
            parent_a = max(
                [population[i] for i in candidates],
                key=lambda t: _hybrid_fitness(t),
            )
            candidates_b = init_rng.choice(
                len(population), size=3, replace=False
            )
            parent_b = max(
                [population[i] for i in candidates_b],
                key=lambda t: _hybrid_fitness(t),
            )

            child = list(parent_a)
            crossover_point = int(init_rng.integers(1, team_size))
            for pos in range(crossover_point, team_size):
                if parent_b[pos] not in child:
                    child[pos] = parent_b[pos]
            for pos in range(team_size):
                if init_rng.random() < 0.10:
                    new_gene = int(init_rng.integers(0, n_pool))
                    while new_gene in child:
                        new_gene = int(init_rng.integers(0, n_pool))
                    child[pos] = new_gene
            next_pop.append(child)

        population = next_pop[:pop_size]

    final_fitnesses = [_hybrid_fitness(t) for t in population]
    best_idx = int(np.argmax(final_fitnesses))
    return population[best_idx]


def _find_tqs_model(search_dir: Path) -> Path | None:
    """Find the best available TQS model file.

    Searches for TQS_XGBoost.pkl first (best performer), then
    falls back to any TQS_*.pkl file.

    Args:
        search_dir: Directory to search for model files.

    Returns:
        Path to the model file, or None if not found.
    """
    xgb_path = search_dir / "TQS_XGBoost.pkl"
    if xgb_path.exists():
        return xgb_path
    candidates = sorted(search_dir.glob("TQS_*.pkl"), reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    """Run the fair TQS decision quality experiment."""
    parser = argparse.ArgumentParser(
        description="Fair TQS Decision Quality experiment"
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=100,
        help="Number of independent scenarios to evaluate",
    )
    parser.add_argument(
        "--n-battles", type=int, default=30,
        help="Battles per head-to-head evaluation",
    )
    parser.add_argument(
        "--tqs-weight", type=float, default=0.4,
        help="Weight for TQS in hybrid fitness (0.0 = pure heuristic)",
    )
    parser.add_argument(
        "--tqs-model", type=Path, default=None,
        help="Path to TQS model .pkl (auto-discovers if omitted)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--pop-size", type=int, default=50,
        help="GA population size",
    )
    parser.add_argument(
        "--generations", type=int, default=30,
        help="GA generations per run",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/experiments/decision_tqs"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--s3-bucket", type=str, default="",
        help="S3 bucket to sync experiment data from (skipped if empty)",
    )
    parser.add_argument(
        "--s3-prefix", type=str, default="experiments/",
        help="S3 key prefix to sync into data/experiments (default: experiments/)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.s3_bucket:
        n = sync_from_s3(Path("data/experiments"), args.s3_prefix, args.s3_bucket)
        print(f"Synced {n} files from s3://{args.s3_bucket}/{args.s3_prefix} to data/experiments")

    tqs_model_path = args.tqs_model
    if tqs_model_path is None:
        tqs_model_path = _find_tqs_model(Path("data/experiments/team_scorer"))

    tqs_bundle: dict[str, Any] | None = None
    if tqs_model_path is not None and tqs_model_path.exists():
        tqs_bundle = _load_tqs_model(tqs_model_path)
        print(f"Loaded TQS model from {tqs_model_path}")
    else:
        print("WARNING: No TQS model found. Running pure heuristic vs heuristic.")
        print("  Generate a TQS model first via experiment_team_scorer.py")

    print("=" * 60)
    print("Fair TQS Decision Quality Experiment")
    print(
        f"  scenarios={args.n_scenarios}, battles={args.n_battles}, "
        f"tqs_weight={args.tqs_weight}"
    )
    print(f"  pop_size={args.pop_size}, generations={args.generations}")
    print(f"  seed={args.seed}")
    print(f"  output={output_dir.resolve()}")
    print("=" * 60)

    start_time = time.perf_counter()
    pool_species, viability = _get_pool()

    bp = GreedyBattlePolicy()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    results: list[dict[str, Any]] = []
    control_wins = 0
    treatment_wins = 0
    draws = 0

    for scenario_idx in range(args.n_scenarios):
        scenario_seed = args.seed + scenario_idx * 1000

        control_rng = np.random.default_rng(scenario_seed)
        control_indices = _run_ga_with_fitness(
            pool_species=pool_species,
            viability_scores=viability,
            rng=control_rng,
            tqs_bundle=None,
            tqs_weight=0.0,
            pop_size=args.pop_size,
            generations=args.generations,
        )

        treatment_rng = np.random.default_rng(scenario_seed + 500)
        treatment_indices = _run_ga_with_fitness(
            pool_species=pool_species,
            viability_scores=viability,
            rng=treatment_rng,
            tqs_bundle=tqs_bundle,
            tqs_weight=args.tqs_weight,
            pop_size=args.pop_size,
            generations=args.generations,
        )

        control_team = _build_team_from_indices(control_indices, pool_species)
        treatment_team = _build_team_from_indices(treatment_indices, pool_species)

        if control_team is None or treatment_team is None:
            print(f"  Scenario {scenario_idx + 1}: SKIP (build failure)")
            continue

        pair_seed = scenario_seed + 900
        wins_ctrl, wins_treat, _, _ = run_pair_battles(
            team_a=control_team,
            team_b=treatment_team,
            bp_side_a=bp,
            bp_side_b=bp,
            n_battles=args.n_battles,
            pair_seed=pair_seed,
            params=params,
            sel=sel,
        )

        if wins_treat > wins_ctrl:
            treatment_wins += 1
            outcome = "treatment"
        elif wins_ctrl > wins_treat:
            control_wins += 1
            outcome = "control"
        else:
            draws += 1
            outcome = "draw"

        agreement = set(control_indices) == set(treatment_indices)

        results.append({
            "scenario": scenario_idx,
            "control_wins": wins_ctrl,
            "treatment_wins": wins_treat,
            "outcome": outcome,
            "agreement": agreement,
            "control_indices": control_indices,
            "treatment_indices": treatment_indices,
        })

        if (scenario_idx + 1) % 10 == 0 or scenario_idx == 0:
            total_decided = control_wins + treatment_wins
            treat_wr = (
                treatment_wins / total_decided if total_decided > 0 else 0.0
            )
            print(
                f"  Scenario {scenario_idx + 1}/{args.n_scenarios}: "
                f"treatment_WR={treat_wr:.3f} "
                f"(W={treatment_wins} L={control_wins} D={draws})"
            )

    elapsed = time.perf_counter() - start_time
    total_decided = control_wins + treatment_wins
    treatment_win_rate = (
        treatment_wins / total_decided if total_decided > 0 else 0.0
    )
    agreement_count = sum(1 for r in results if r["agreement"])
    agreement_rate = agreement_count / max(len(results), 1)

    viable = treatment_win_rate >= 0.55

    print("\n" + "=" * 60)
    print("RESULTS")
    print(f"  Scenarios completed: {len(results)}")
    print(f"  Treatment wins: {treatment_wins}")
    print(f"  Control wins:   {control_wins}")
    print(f"  Draws:          {draws}")
    print(f"  Treatment win rate: {treatment_win_rate:.4f}")
    print(f"  Agreement rate:     {agreement_rate:.4f}")
    print(f"  Duration: {elapsed:.0f}s")
    print(f"  VIABLE: {viable} (threshold: >= 0.55)")
    print("=" * 60)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_scenarios": args.n_scenarios,
        "n_battles": args.n_battles,
        "tqs_weight": args.tqs_weight,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "tqs_model_path": str(tqs_model_path) if tqs_model_path else None,
        "treatment_wins": treatment_wins,
        "control_wins": control_wins,
        "draws": draws,
        "treatment_win_rate": treatment_win_rate,
        "agreement_rate": agreement_rate,
        "viable": viable,
        "duration_seconds": round(elapsed, 1),
    }

    summary_path = output_dir / f"dq_tqs_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    details_path = output_dir / f"dq_tqs_details_{timestamp}.jsonl"
    with open(details_path, "w") as f:
        for row in results:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"\n  Summary: {summary_path}")
    print(f"  Details: {details_path}")


if __name__ == "__main__":
    main()
