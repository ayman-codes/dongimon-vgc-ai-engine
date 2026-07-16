"""Battle royale benchmark for Dongimon against 4 opponents.

Runs 5 matches of 25 battles each against Greedy, JJJ, minimon, and
StocKarpador. Results logged to MLflow and saved to data/results_*.csv.

Two modes:
- Isolated (default): all competitors share the same 6-Pokemon team
- Full (--full): each competitor uses all 3 policies (teambuild, selection, battle)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy, RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.match import Match, subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights

N_MATCHES = 5
N_BATTLES = 25
OPPONENT_NAMES = ["Greedy", "JJJ", "minimon", "StocKarpador"]


class GreedyIsolatedCompetitor(Competitor):
    """Baseline using GreedyBattlePolicy with no selection or teambuild."""

    @property
    def name(self):
        return "Greedy"

    @property
    def battlepolicy(self):
        return GreedyBattlePolicy()

    @property
    def selectionpolicy(self):
        return None

    @property
    def teambuildpolicy(self):
        return None


class GreedyFullCompetitor(Competitor):
    """Baseline using GreedyBattlePolicy with random selection and teambuild."""

    @property
    def name(self):
        return "Greedy"

    @property
    def battlepolicy(self):
        return GreedyBattlePolicy()

    @property
    def selectionpolicy(self):
        return RandomSelectionPolicy()

    @property
    def teambuildpolicy(self):
        return RandomTeamBuildPolicy()


def _import_competitor(module_path: str, class_name: str):
    """Dynamically import a competitor class.

    Args:
        module_path: Dot-separated module path.
        class_name: Name of the competitor class.

    Returns:
        The competitor class.
    """
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _get_isolated_policies():
    """Return dict of opponent name to battle policy for isolated mode.

    Returns:
        Dict mapping opponent name to a BattlePolicy instance.
    """
    policies = {
        "Greedy": GreedyBattlePolicy(),
    }
    for name, mod_path, cls_name in [
        ("JJJ", "competitors.competitor1_jjj", "JJJ_Competitor"),
        ("minimon", "competitors.competitor2_minimon", "minimon"),
        ("StocKarpador", "competitors.competitor3_stockarpador", "StocKarpadorCompetitor"),
    ]:
        comp_cls = _import_competitor(mod_path, cls_name)
        comp = comp_cls()
        policies[name] = comp.battlepolicy
    return policies


def _get_full_competitors():
    """Return dict of opponent name to competitor factory for full mode.

    Returns:
        Dict mapping opponent name to a Competitor class.
    """
    factories = {
        "Greedy": GreedyFullCompetitor,
    }
    for name, mod_path, cls_name in [
        ("JJJ", "competitors.competitor1_jjj", "JJJ_Competitor"),
        ("minimon", "competitors.competitor2_minimon", "minimon"),
        ("StocKarpador", "competitors.competitor3_stockarpador", "StocKarpadorCompetitor"),
    ]:
        factories[name] = _import_competitor(mod_path, cls_name)
    return factories


def _run_isolated_match(
    battle_policy_a, battle_policy_b, base_team, base_view, n_battles, params, battle_seed
):
    """Run N battles between two battle policies using the same team.

    Args:
        battle_policy_a: The first battle policy (Dongimon).
        battle_policy_b: The opponent battle policy.
        base_team: Shared team object.
        base_view: Shared team view.
        n_battles: Number of battles to run.
        params: Battle rule parameters.
        battle_seed: Base RNG seed for battle engine.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    sel = BasicSelectionPolicy()
    wins_a = 0
    wins_b = 0

    for battle_idx in range(n_battles):
        idx_a = sel.decision((base_team, base_view), 4)
        idx_b = sel.decision((base_team, base_view), 4)

        sub_a, sub_view_a = subteam(base_team, base_view, idx_a)
        sub_b, sub_view_b = subteam(base_team, base_view, idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), 2)
        state = State(battle_teams)
        gen = np.random.default_rng(battle_seed + battle_idx)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            cmd0 = battle_policy_a.decision(sv0, sub_view_b)
            cmd1 = battle_policy_b.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
        elif engine.winning_side == 1:
            wins_b += 1

    return wins_a, wins_b


def _run_full_match(opponent_cls, n_battles, params, weights_dict):
    """Run N battles between Dongimon and an opponent using all 3 policies.

    Args:
        opponent_cls: Opponent Competitor class.
        n_battles: Number of battles to run.
        params: Battle rule parameters.
        weights_dict: Custom weights for DongimonBattlePolicy.

    Returns:
        Tuple of (dongimon_wins, opponent_wins).
    """
    dongimon = CompetitorManager(DongimonCompetitor(custom_weights=weights_dict))
    opponent = CompetitorManager(opponent_cls())
    match = Match((dongimon, opponent), n_battles=n_battles, gen=gen_team, params=params)
    match.run()
    wins = match.wins
    return wins[0], wins[1]


def _run_isolated_benchmark(seed, n_matches, n_battles, results_path, weights_dict):
    """Run the isolated battle royale benchmark.

    Args:
        seed: Base RNG seed.
        n_matches: Number of match iterations per opponent.
        n_battles: Number of battles per match.
        results_path: Path to CSV results file.
        weights_dict: Current battle policy weights.

    Returns:
        Dict of per-opponent aggregate (dongimon_wins, opponent_wins).
    """
    params = BattleRuleParam()
    policies = _get_isolated_policies()

    fieldnames = ["match_id", "opponent", "dongimon_wins", "opponent_wins", "total_battles"]
    per_opponent = {opp: [0, 0] for opp in OPPONENT_NAMES}

    with open(results_path, "a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if f_out.tell() == 0:
            writer.writeheader()

        for match_id in range(n_matches):
            match_seed = seed + match_id * 1000
            np.random.seed(match_seed)
            team_rng = np.random.default_rng(match_seed)
            shared_team = gen_team(6, 4, team_rng)
            shared_view = TeamView(shared_team)
            dongimon = DongimonCompetitor(custom_weights=weights_dict)

            for opp_name in OPPONENT_NAMES:
                opp_bp = policies[opp_name]
                dw, ow = _run_isolated_match(
                    dongimon.battlepolicy, opp_bp, shared_team, shared_view, n_battles, params, match_seed
                )
                per_opponent[opp_name][0] += dw
                per_opponent[opp_name][1] += ow

                writer.writerow({
                    "match_id": match_id,
                    "opponent": opp_name,
                    "dongimon_wins": dw,
                    "opponent_wins": ow,
                    "total_battles": dw + ow,
                })
                f_out.flush()

    return {opp: tuple(v) for opp, v in per_opponent.items()}


def _run_full_benchmark(seed, n_matches, n_battles, results_path, weights_dict):
    """Run the full battle royale benchmark with all 3 policies.

    Args:
        seed: Base RNG seed.
        n_matches: Number of match iterations per opponent.
        n_battles: Number of battles per match.
        results_path: Path to CSV results file.
        weights_dict: Current battle policy weights.

    Returns:
        Dict of per-opponent aggregate (dongimon_wins, opponent_wins).
    """
    params = BattleRuleParam()
    competitors_map = _get_full_competitors()

    fieldnames = ["match_id", "opponent", "dongimon_wins", "opponent_wins", "total_battles"]
    per_opponent = {opp: [0, 0] for opp in OPPONENT_NAMES}

    with open(results_path, "a", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if f_out.tell() == 0:
            writer.writeheader()

        for match_id in range(n_matches):
            np.random.seed(seed + match_id * 2000)

            for opp_name in OPPONENT_NAMES:
                opp_cls = competitors_map[opp_name]
                dw, ow = _run_full_match(opp_cls, n_battles, params, weights_dict)
                per_opponent[opp_name][0] += dw
                per_opponent[opp_name][1] += ow

                writer.writerow({
                    "match_id": match_id,
                    "opponent": opp_name,
                    "dongimon_wins": dw,
                    "opponent_wins": ow,
                    "total_battles": dw + ow,
                })
                f_out.flush()

    return {opp: tuple(v) for opp, v in per_opponent.items()}


def _log_to_mlflow(results, weights_dict, mode, seed, n_matches, n_battles, results_path, tag):
    """Log benchmark results to MLflow.

    Args:
        results: Dict of per-opponent (dongimon_wins, opponent_wins).
        weights_dict: Current battle policy weights.
        mode: Benchmark mode ("isolated" or "full").
        seed: RNG seed.
        n_matches: Number of match iterations.
        n_battles: Number of battles per match.
        results_path: Path to CSV results file.
        tag: Optional run tag.
    """
    total_d = sum(v[0] for v in results.values())
    total_o = sum(v[1] for v in results.values())
    grand_total = total_d + total_o

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("dongimon_benchmarks")

    with mlflow.start_run(run_name=f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            "mode": mode,
            "seed": seed,
            "n_matches": n_matches,
            "n_battles_per_match": n_battles,
            "tag": tag or "",
        })
        mlflow.log_params(weights_dict)

        for opp_name, (dw, ow) in results.items():
            total = dw + ow
            wr = dw / max(total, 1)
            mlflow.log_metric(f"win_rate_vs_{opp_name}", wr)

        agg_wr = total_d / max(grand_total, 1)
        mlflow.log_metric("aggregate_win_rate", agg_wr)

        weights_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "config", "battle_weights.yaml"
        )
        mlflow.log_artifact(weights_path)
        mlflow.log_artifact(results_path)


def main():
    parser = argparse.ArgumentParser(
        description="Dongimon battle royale benchmark against 4 opponents."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use all 3 policies (teambuild, selection, battle)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-matches", type=int, default=N_MATCHES, help="Matches per opponent")
    parser.add_argument("--n-battles", type=int, default=N_BATTLES, help="Battles per match")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    args = parser.parse_args()

    mode = "full" if args.full else "isolated"
    weights_dict = load_battle_weights().model_dump()
    np.random.seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"results_{mode}_{timestamp}.csv"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    total_battles = args.n_matches * len(OPPONENT_NAMES) * args.n_battles

    print("=" * 60)
    print(f"Dongimon Battle Royale Benchmark — mode={mode}")
    print(f"  seed={args.seed}, n_matches={args.n_matches}, n_battles={args.n_battles}")
    print("  opponents: Greedy, JJJ, minimon, StocKarpador")
    print(f"  total battles: {total_battles}")
    print("=" * 60)

    if args.full:
        results = _run_full_benchmark(
            args.seed, args.n_matches, args.n_battles, results_path, weights_dict
        )
    else:
        results = _run_isolated_benchmark(
            args.seed, args.n_matches, args.n_battles, results_path, weights_dict
        )

    print(f"\n{'Opponent':<15} {'Dongimon':>10} {'Opponent':>10} {'Win Rate':>10}")
    print("-" * 50)
    for opp_name in OPPONENT_NAMES:
        dw, ow = results[opp_name]
        total = dw + ow
        wr = dw / max(total, 1) * 100
        print(f"{opp_name:<15} {dw:>10} {ow:>10} {wr:>8.1f}%")

    total_d = sum(results[opp][0] for opp in OPPONENT_NAMES)
    total_o = sum(results[opp][1] for opp in OPPONENT_NAMES)
    gt = total_d + total_o
    agg_wr = total_d / max(gt, 1) * 100
    print("-" * 50)
    print(f"{'AGGREGATE':<15} {total_d:>10} {total_o:>10} {agg_wr:>8.1f}%")

    if not args.no_mlflow:
        _log_to_mlflow(
            results, weights_dict, mode, args.seed,
            args.n_matches, args.n_battles, results_path, args.tag,
        )
        print("\nResults logged to MLflow (experiment: dongimon_benchmarks)")

    print(f"\nCSV saved to: {results_path}")


if __name__ == "__main__":
    main()
