"""Experiment: Decision Quality — Does the ML model beat simulation?

Validates that MP and TQS models produce better downstream decisions than
the current simulation-based approach. Loads saved .pkl models from
Phase 2 (MP) and Phase 3 (TQS). Does NOT modify any file under src/.

Usage:
    uv run python scripts/experiments/experiment_decision_quality.py \
        --mp-model=data/experiments/matchup_predictor/mp_model_X.pkl \
        --tqs-model=data/experiments/team_scorer/tqs_model_Y.pkl \
        --n-scenarios=100 --n-battles=30 --seed=42
"""

import argparse
import itertools
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.team import Team as VgcTeam
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from scripts.experiments.experiment_utils import run_pair_battles
from src.config.loader import load_battle_weights
from src.data.features import compute_pairwise_features, compute_subteam_features
from src.shared.s3 import sync_from_s3


def _load_model(model_path: Path) -> dict[str, Any]:
    """Load a pickled model bundle.

    Args:
        model_path: Path to the .pkl file.

    Returns:
        Dict with keys: model, scaler, feature_names, config.
    """
    with open(model_path, "rb") as bf:
        return cast(dict[str, Any], pickle.load(bf))


def _dongimon_policy() -> Any:
    """Return Dongimon's battle policy with current weights.

    Returns:
        DongimonBattlePolicy instance.
    """
    return DongimonCompetitor(
        custom_weights=load_battle_weights().model_dump()
    ).battlepolicy


def _extract_features_mp(
    subteam_members: list[Any],
    opp_members: list[Any],
    feature_names: list[str],
) -> np.ndarray[Any, Any]:
    """Extract pairwise features for MP model inference.

    Args:
        subteam_members: List of 4 Pokemon for our subteam.
        opp_members: List of 4 Pokemon for the opponent.
        feature_names: Ordered feature names from the loaded model.

    Returns:
        Feature array of shape (1, n_features) ready for scaler + predict.
    """
    feat_dict = compute_pairwise_features(subteam_members, opp_members)
    return np.array([[feat_dict.get(n, 0.0) for n in feature_names]], dtype=np.float64)


def _extract_features_tqs(
    team_members: list[Any],
    feature_names: list[str],
) -> np.ndarray[Any, Any]:
    """Extract single-team features for TQS model inference.

    Args:
        team_members: List of 6 Pokemon.
        feature_names: Ordered feature names from the loaded model.

    Returns:
        Feature array of shape (1, n_features) ready for scaler + predict.
    """
    feat_dict = compute_subteam_features(team_members)
    return np.array([[feat_dict.get(n, 0.0) for n in feature_names]], dtype=np.float64)


def _run_sub_tournament(
    subteam_members: list[Any],
    opp_members: list[Any],
    battle_policy: Any,
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
    base_seed: int,
) -> float:
    """Simulate C(4,2) vs C(4,2) sub-tournament win rate.

    Evaluates the subteam against the opponent by running battles
    across all active-pairing combinations.

    Args:
        subteam_members: 4 Pokemon for our subteam.
        opp_members: 4 Pokemon for the opponent.
        battle_policy: BattlePolicy for both sides.
        sel: SelectionPolicy for team selection.
        params: BattleRuleParam.
        n_battles: Battles per active-pair combination.
        base_seed: Base RNG seed.

    Returns:
        Win rate (0.0–1.0) of our subteam against the opponent.

    Raises:
        ValueError: If subteam or opponent has fewer than 2 members.
    """

    team_obj = VgcTeam(list(subteam_members))
    opp_obj = VgcTeam(list(opp_members))
    team_view = TeamView(team_obj)
    opp_view = TeamView(opp_obj)

    active_indices = list(itertools.combinations(range(4), 2))

    total_wins = 0
    total_battles = 0
    battle_idx = 0

    for act_a in active_indices:
        sub_a, sub_view_a = subteam(team_obj, team_view, list(act_a))
        for act_b in active_indices:
            sub_b, sub_view_b = subteam(opp_obj, opp_view, list(act_b))
            for _ in range(n_battles):
                battle_teams = get_battle_teams((sub_a, sub_b), 2)
                state = State(battle_teams)
                gen_rng = np.random.default_rng(base_seed + battle_idx)
                rng_tuple = ((gen_rng, gen_rng), (gen_rng, gen_rng))
                engine = BattleEngine(
                    state, params=params,
                    acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
                )
                battle_idx += 1
                while not engine.finished():
                    sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
                    sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
                    cmd0 = battle_policy.decision(sv0, sub_view_b)
                    cmd1 = battle_policy.decision(sv1, sub_view_a)
                    engine.run_turn((cmd0, cmd1))
                if engine.winning_side == 0:
                    total_wins += 1
                total_battles += 1

    return total_wins / max(total_battles, 1)


def _run_mp_test(
    mp_bundle: dict[str, Any],
    dongimon_policy: Any,
    greedy_policy: Any,
    n_scenarios: int,
    n_battles: int,
    sub_tournament_battles: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run MP decision quality test.

    For each scenario, compares simulation-based subteam selection
    against MP model-based selection. When they disagree, runs
    head-to-head battles with Dongimon to determine which is better.

    Args:
        mp_bundle: Loaded MP model dict.
        dongimon_policy: DongimonBattlePolicy instance.
        greedy_policy: GreedyBattlePolicy instance.
        n_scenarios: Number of scenarios to evaluate.
        n_battles: Battles for head-to-head comparison.
        sub_tournament_battles: Battles per sub-tournament pairing.
        seed: Base RNG seed.
        output_dir: Output directory for results.

    Returns:
        Dict with metrics: agreement_rate, mp_wins, sim_wins,
        mp_win_rate_when_disagree, time_sim_total, time_model_total.
    """
    mp_model = mp_bundle["model"]
    mp_scaler = mp_bundle["scaler"]
    mp_feature_names = mp_bundle["feature_names"]

    sel = BasicSelectionPolicy()
    params = BattleRuleParam()
    rng = np.random.default_rng(seed)

    agreements = 0
    disagreements = 0
    mp_wins_when_disagree = 0
    sim_wins_when_disagree = 0
    time_sim_total = 0.0
    time_model_total = 0.0

    for scenario_idx in range(n_scenarios):
        team_a = gen_team(4, 4, rng)
        team_b = gen_team(4, 4, rng)
        opp_members = list(team_b.members[:4])
        scenario_seed = seed + scenario_idx * 10000

        subteam_indices = list(itertools.combinations(range(6), 4))
        sim_scores: list[float] = []
        mp_scores: list[float] = []

        for sub_idx in subteam_indices:
            cand_members = [team_a.members[i] for i in sub_idx]

            t0 = time.perf_counter()
            sim_score = _run_sub_tournament(
                cand_members, opp_members, greedy_policy, sel, params,
                sub_tournament_battles, scenario_seed + sub_idx[0] * 100,
            )
            time_sim_total += time.perf_counter() - t0
            sim_scores.append(sim_score)

            t0 = time.perf_counter()
            x_raw = _extract_features_mp(cand_members, opp_members, mp_feature_names)
            x_scaled = mp_scaler.transform(x_raw)
            mp_score = float(mp_model.predict(x_scaled)[0])
            time_model_total += time.perf_counter() - t0
            mp_scores.append(mp_score)

        best_sim = int(np.argmax(sim_scores))
        best_mp = int(np.argmax(mp_scores))

        if best_sim == best_mp:
            agreements += 1
        else:
            disagreements += 1
            sub_sim_members = [team_a.members[i] for i in subteam_indices[best_sim]]
            sub_mp_members = [team_a.members[i] for i in subteam_indices[best_mp]]

            wins_sim, wins_mp = _head_to_head_comparison(
                sub_sim_members, sub_mp_members, team_b,
                dongimon_policy, greedy_policy,
                n_battles, scenario_seed + 5000,
            )
            if wins_mp > wins_sim:
                mp_wins_when_disagree += 1
            elif wins_sim > wins_mp:
                sim_wins_when_disagree += 1

        if (scenario_idx + 1) % 20 == 0:
            print(
                f"  MP: {scenario_idx + 1}/{n_scenarios} scenarios "
                f"(agree={agreements}, disagree={disagreements}, "
                f"mp_wins={mp_wins_when_disagree}, sim_wins={sim_wins_when_disagree})"
            )

    total = n_scenarios
    agree_rate = agreements / total if total > 0 else 0.0
    mp_win_rate = mp_wins_when_disagree / max(disagreements, 1)

    return {
        "n_scenarios": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": agree_rate,
        "mp_wins_when_disagree": mp_wins_when_disagree,
        "sim_wins_when_disagree": sim_wins_when_disagree,
        "ties_when_disagree": disagreements - mp_wins_when_disagree - sim_wins_when_disagree,
        "mp_win_rate_when_disagree": mp_win_rate,
        "time_sim_total_s": round(time_sim_total, 1),
        "time_model_total_s": round(time_model_total, 3),
    }


def _head_to_head_comparison(
    sub_a_members: list[Any],
    sub_b_members: list[Any],
    full_opp_team: Any,
    side_a_policy: Any,
    side_b_policy: Any,
    n_battles: int,
    seed: int,
) -> tuple[int, int]:
    """Run head-to-head battles between two subteam selections.

    Each selection pilots side A, opponent pilots side B.

    Args:
        sub_a_members: 4 Pokemon for candidate A.
        sub_b_members: 4 Pokemon for candidate B.
        full_opp_team: Full 6-mon Team object for opponent.
        side_a_policy: BattlePolicy for side A.
        side_b_policy: BattlePolicy for side B.
        n_battles: Number of battles to run.
        seed: RNG seed.

    Returns:
        Tuple of (wins_a, wins_b).
    """

    team_a = VgcTeam(sub_a_members)
    team_b = VgcTeam(sub_b_members)
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)
    opp_view = TeamView(full_opp_team)
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    wins_a = 0
    wins_b = 0

    idx_a = sel.decision((team_a, opp_view), 4)
    idx_b = sel.decision((team_b, opp_view), 4)
    opp_idx = sel.decision((full_opp_team, view_a), 4)

    for b_idx in range(n_battles):
        sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
        sub_b, sub_view_b = subteam(team_b, view_b, idx_b)

        sub_opp_a, sub_view_opp_a = subteam(full_opp_team, opp_view, opp_idx)
        sub_opp_b, sub_view_opp_b = subteam(full_opp_team, opp_view, opp_idx)

        battle_teams_a = get_battle_teams((sub_a, sub_opp_a), 2)
        state_a = State(battle_teams_a)
        gen_rng = np.random.default_rng(seed + b_idx * 1000)
        rng_tuple = ((gen_rng, gen_rng), (gen_rng, gen_rng))
        engine_a = BattleEngine(
            state_a, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )
        while not engine_a.finished():
            sv0 = StateView(engine_a.state, 0, (sub_view_a, sub_view_opp_a))
            sv1 = StateView(engine_a.state, 1, (sub_view_opp_a, sub_view_a))
            engine_a.run_turn((
                side_a_policy.decision(sv0, sub_view_opp_a),
                side_b_policy.decision(sv1, sub_view_a),
            ))
        if engine_a.winning_side == 0:
            wins_a += 1

        battle_teams_b = get_battle_teams((sub_b, sub_opp_b), 2)
        state_b = State(battle_teams_b)
        gen_rng2 = np.random.default_rng(seed + b_idx * 1000 + 500)
        rng_tuple2 = ((gen_rng2, gen_rng2), (gen_rng2, gen_rng2))
        engine_b = BattleEngine(
            state_b, params=params,
            acc_rng=rng_tuple2, eff_rng=rng_tuple2, sta_rng=rng_tuple2,
        )
        while not engine_b.finished():
            sv0 = StateView(engine_b.state, 0, (sub_view_b, sub_view_opp_b))
            sv1 = StateView(engine_b.state, 1, (sub_view_opp_b, sub_view_b))
            engine_b.run_turn((
                side_a_policy.decision(sv0, sub_view_opp_b),
                side_b_policy.decision(sv1, sub_view_b),
            ))
        if engine_b.winning_side == 0:
            wins_b += 1

    return wins_a, wins_b


def _run_tqs_test(
    tqs_bundle: dict[str, Any],
    dongimon_policy: Any,
    greedy_policy: Any,
    n_scenarios: int,
    n_battles: int,
    rr_battles: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run TQS decision quality test.

    For each scenario, compares round-robin ranking against TQS
    model ranking. When they disagree, runs head-to-head battles
    with Dongimon to determine which is better.

    Args:
        tqs_bundle: Loaded TQS model dict.
        dongimon_policy: DongimonBattlePolicy instance.
        greedy_policy: GreedyBattlePolicy instance.
        n_scenarios: Number of scenarios.
        n_battles: Battles for head-to-head comparison.
        rr_battles: Battles per round-robin pairing.
        seed: Base RNG seed.
        output_dir: Output directory.

    Returns:
        Dict with decision quality metrics.
    """
    tqs_model = tqs_bundle["model"]
    tqs_scaler = tqs_bundle["scaler"]
    tqs_feature_names = tqs_bundle["feature_names"]

    rng = np.random.default_rng(seed)

    agreements = 0
    disagreements = 0
    tqs_wins_when_disagree = 0
    rr_wins_when_disagree = 0
    time_rr_total = 0.0
    time_tqs_total = 0.0

    for scenario_idx in range(n_scenarios):
        teams = [gen_team(4, 4, rng) for _ in range(10)]
        scenario_seed = seed + scenario_idx * 20000

        t0 = time.perf_counter()
        rr_scores = _round_robin_rank(
            teams, greedy_policy, greedy_policy,
            rr_battles, scenario_seed,
        )
        time_rr_total += time.perf_counter() - t0
        best_rr = int(np.argmax(rr_scores))

        t0 = time.perf_counter()
        tqs_scores: list[float] = []
        for team in teams:
            x_raw = _extract_features_tqs(list(team.members), tqs_feature_names)
            x_scaled = tqs_scaler.transform(x_raw)
            tqs_scores.append(float(tqs_model.predict(x_scaled)[0]))
        time_tqs_total += time.perf_counter() - t0
        best_tqs = int(np.argmax(tqs_scores))

        if best_rr == best_tqs:
            agreements += 1
        else:
            disagreements += 1
            wins_rr, wins_tqs = _head_to_head_tqs(
                teams[best_rr], teams[best_tqs],
                dongimon_policy, n_battles, scenario_seed + 5000,
            )
            if wins_tqs > wins_rr:
                tqs_wins_when_disagree += 1
            elif wins_rr > wins_tqs:
                rr_wins_when_disagree += 1

        if (scenario_idx + 1) % 20 == 0:
            print(
                f"  TQS: {scenario_idx + 1}/{n_scenarios} scenarios "
                f"(agree={agreements}, disagree={disagreements}, "
                f"tqs_wins={tqs_wins_when_disagree}, rr_wins={rr_wins_when_disagree})"
            )

    total = n_scenarios
    agree_rate = agreements / total if total > 0 else 0.0
    tqs_win_rate = tqs_wins_when_disagree / max(disagreements, 1)

    return {
        "n_scenarios": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": agree_rate,
        "tqs_wins_when_disagree": tqs_wins_when_disagree,
        "rr_wins_when_disagree": rr_wins_when_disagree,
        "ties_when_disagree": disagreements - tqs_wins_when_disagree - rr_wins_when_disagree,
        "tqs_win_rate_when_disagree": tqs_win_rate,
        "time_rr_total_s": round(time_rr_total, 1),
        "time_tqs_total_s": round(time_tqs_total, 3),
    }


def _round_robin_rank(
    teams: list[Any],
    bp_side_a: Any,
    bp_side_b: Any,
    n_battles: int,
    seed: int,
) -> list[float]:
    """Run round-robin tournament and return win rate per team.

    Args:
        teams: List of 10 vgc2 Team objects.
        bp_side_a: BattlePolicy for side A.
        bp_side_b: BattlePolicy for side B.
        n_battles: Battles per pairing.
        seed: Base RNG seed.

    Returns:
        List of 10 win rates.
    """
    n = len(teams)
    total_wins = np.zeros(n)
    total_battles_arr = np.zeros(n)
    battle_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            pair_seed = seed + battle_count * 100
            wins_i, wins_j, _, _ = run_pair_battles(
                team_a=teams[i],
                team_b=teams[j],
                bp_side_a=bp_side_a,
                bp_side_b=bp_side_b,
                n_battles=n_battles,
                pair_seed=pair_seed,
            )
            total_wins[i] += wins_i
            total_wins[j] += wins_j
            total_battles_arr[i] += n_battles
            total_battles_arr[j] += n_battles
            battle_count += 1

    wr = np.zeros(n)
    for i in range(n):
        if total_battles_arr[i] > 0:
            wr[i] = total_wins[i] / total_battles_arr[i]
    return wr.tolist()


def _head_to_head_tqs(
    team_a: Any,
    team_b: Any,
    battle_policy: Any,
    n_battles: int,
    seed: int,
) -> tuple[int, int]:
    """Run head-to-head battles between two teams (Dongimon-vs-Dongimon).

    Args:
        team_a: First team.
        team_b: Second team.
        battle_policy: BattlePolicy for both sides.
        n_battles: Number of battles.
        seed: RNG seed.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    return run_pair_battles(
        team_a=team_a,
        team_b=team_b,
        bp_side_a=battle_policy,
        bp_side_b=battle_policy,
        n_battles=n_battles,
        pair_seed=seed,
    )[:2]


def main() -> None:
    """Run the Decision Quality experiment."""
    parser = argparse.ArgumentParser(
        description="Decision Quality: ML model vs simulation-based selection"
    )
    parser.add_argument(
        "--mp-model", type=Path, required=True,
        help="Path to saved MP model (.pkl from Phase 2)",
    )
    parser.add_argument(
        "--tqs-model", type=Path, default=None,
        help="Path to saved TQS model (.pkl from Phase 3)",
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=100,
        help="Number of scenarios to evaluate",
    )
    parser.add_argument(
        "--n-battles", type=int, default=30,
        help="Battles per head-to-head comparison",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/experiments/decision_quality"),
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
    sub_tournament_battles = 3

    if args.s3_bucket:
        n = sync_from_s3(Path("data/experiments"), args.s3_prefix, args.s3_bucket)
        print(f"Synced {n} files from s3://{args.s3_bucket}/{args.s3_prefix} to data/experiments")

    print("=" * 60)
    print("Decision Quality Experiment")
    print(f"  seed={args.seed}, n_scenarios={args.n_scenarios}")
    print(f"  mp_model={args.mp_model}")
    print(f"  tqs_model={args.tqs_model}")
    print(f"  sub_tournament_battles={sub_tournament_battles}")
    print(f"  h2h_battles={args.n_battles}")
    print("=" * 60)

    start = time.perf_counter()

    print("Loading models...")
    mp_bundle = _load_model(args.mp_model)
    print(f"  MP: {type(mp_bundle['model']).__name__}, "
          f"features={len(mp_bundle['feature_names'])}, "
          f"policy={mp_bundle['config'].get('side_a_policy', '?')}")

    if args.tqs_model:
        tqs_bundle = _load_model(args.tqs_model)
        print(f"  TQS: {type(tqs_bundle['model']).__name__}, "
              f"features={len(tqs_bundle['feature_names'])}, "
              f"policy={tqs_bundle['config'].get('policy', '?')}")
    else:
        tqs_bundle = None

    print("Initializing policies...")
    dongimon_policy = _dongimon_policy()
    greedy_policy = GreedyBattlePolicy()

    print(f"\n--- MP Decision Quality ({args.n_scenarios} scenarios) ---")
    mp_results = _run_mp_test(
        mp_bundle, dongimon_policy, greedy_policy,
        args.n_scenarios, args.n_battles, sub_tournament_battles,
        args.seed, output_dir,
    )

    print(
        f"  Agreement rate: {mp_results['agreement_rate']:.2%} "
        f"({mp_results['agreements']}/{mp_results['n_scenarios']})"
    )
    if mp_results["disagreements"] > 0:
        print(
            f"  When disagreeing: MP wins {mp_results['mp_wins_when_disagree']} "
            f"({mp_results['mp_win_rate_when_disagree']:.2%}), "
            f"Sim wins {mp_results['sim_wins_when_disagree']}, "
            f"ties {mp_results['ties_when_disagree']}"
        )
    print(
        f"  Time: sim={mp_results['time_sim_total_s']}s, "
        f"model={mp_results['time_model_total_s']}s"
    )

    tqs_results: dict[str, Any] | None = None
    if tqs_bundle:
        print(f"\n--- TQS Decision Quality ({args.n_scenarios} scenarios) ---")
        tqs_results = _run_tqs_test(
            tqs_bundle, dongimon_policy, greedy_policy,
            args.n_scenarios, args.n_battles, 10,
            args.seed + 100000, output_dir,
        )
        print(
            f"  Agreement rate: {tqs_results['agreement_rate']:.2%} "
            f"({tqs_results['agreements']}/{tqs_results['n_scenarios']})"
        )
        if tqs_results["disagreements"] > 0:
            print(
                f"  When disagreeing: TQS wins {tqs_results['tqs_wins_when_disagree']} "
                f"({tqs_results['tqs_win_rate_when_disagree']:.2%}), "
                f"RR wins {tqs_results['rr_wins_when_disagree']}, "
                f"ties {tqs_results['ties_when_disagree']}"
            )
        print(
            f"  Time: rr={tqs_results['time_rr_total_s']}s, "
            f"model={tqs_results['time_tqs_total_s']}s"
        )

    elapsed = time.perf_counter() - start
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    mp_viable = False
    if mp_results["disagreements"] >= 5 and mp_results["mp_win_rate_when_disagree"] >= 0.55:
        mp_viable = True

    tqs_viable = False
    if tqs_results and tqs_results["disagreements"] >= 5 and tqs_results["tqs_win_rate_when_disagree"] >= 0.55:
        tqs_viable = True

    print("\n--- Verdict ---")
    mp_str = (
        f"MP: {'USEFUL' if mp_viable else 'INCONCLUSIVE'} "
        f"(win_rate={mp_results['mp_win_rate_when_disagree']:.2%}, "
        f"n_disagree={mp_results['disagreements']})"
    )
    print(f"  {mp_str}")
    if tqs_results:
        tqs_str = (
            f"TQS: {'USEFUL' if tqs_viable else 'INCONCLUSIVE'} "
            f"(win_rate={tqs_results['tqs_win_rate_when_disagree']:.2%}, "
            f"n_disagree={tqs_results['disagreements']})"
        )
        print(f"  {tqs_str}")

    dq_mp_path = output_dir / f"dq_mp_{timestamp}.json"
    with open(dq_mp_path, "w") as f:
        json.dump({"timestamp": timestamp, **mp_results}, f, indent=2)

    if tqs_results:
        dq_tqs_path = output_dir / f"dq_tqs_{timestamp}.json"
        with open(dq_tqs_path, "w") as f:
            json.dump({"timestamp": timestamp, **tqs_results}, f, indent=2)

    summary = {
        "timestamp": timestamp,
        "seed": args.seed,
        "n_scenarios": args.n_scenarios,
        "n_battles": args.n_battles,
        "duration_seconds": round(elapsed, 1),
        "mp_viable": mp_viable,
        "tqs_viable": tqs_viable,
        "mp_agreement_rate": mp_results["agreement_rate"],
        "mp_win_rate_when_disagree": mp_results["mp_win_rate_when_disagree"],
        "tqs_agreement_rate": tqs_results["agreement_rate"] if tqs_results else None,
        "tqs_win_rate_when_disagree": tqs_results["tqs_win_rate_when_disagree"] if tqs_results else None,
        "time_sim_mp_s": mp_results["time_sim_total_s"],
        "time_model_mp_s": mp_results["time_model_total_s"],
    }
    dq_summary_path = output_dir / f"dq_summary_{timestamp}.json"
    with open(dq_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  MP results:     {dq_mp_path}")
    if tqs_results:
        print(f"  TQS results:    {dq_tqs_path}")
    print(f"  Summary:        {dq_summary_path}")
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
