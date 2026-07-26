"""Experiment: Greedy vs Dongimon policy correlation test.

Generates team pairs, runs battles with both Greedy and Dongimon as
side A (against Greedy+JJJ opponents), and computes Spearman's rho
between the two sets of outcomes. This determines whether training
data labeled by the fast Greedy policy generalizes to Dongimon.

Usage:
    uv run python scripts/experiments/experiment_policy_correlation.py --n-pairs=500 --n-battles=15
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from scipy.stats import spearmanr
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights


def _import_bp(module_path: str, class_name: str) -> Any:
    """Import a competitor and return its battle policy."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls().battlepolicy


def _dongimon_bp() -> Any:
    """Return Dongimon's battle policy with current weights."""
    return DongimonCompetitor(custom_weights=load_battle_weights().model_dump()).battlepolicy


def _run_with_side_a(
    bp_side_a: Any,
    opp_policies: list[Any],
    teams_a: list[Any],
    teams_b: list[Any],
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    base_seed: int,
    n_battles: int,
    label: str,
) -> list[float]:
    """Run battles and return win rates for side A per pairing.

    Args:
        bp_side_a: Battle policy for side A.
        opp_policies: List of opponent policies for side B.
        teams_a: List of team-A objects.
        teams_b: List of team-B objects.
        sel: Selection policy.
        params: Battle rule parameters.
        base_seed: Base RNG seed.
        n_battles: Battles per pairing.
        label: Human-readable name for progress messages.

    Returns:
        List of win rates (0.0–1.0) for side A, one per pairing.
    """
    win_rates = []
    total = len(teams_a)
    start = time.perf_counter()

    for p_idx, (team_a, team_b) in enumerate(zip(teams_a, teams_b, strict=False)):
        pair_seed = base_seed + p_idx * 100
        view_a = TeamView(team_a)
        view_b = TeamView(team_b)
        wins_a = 0
        policy_rng = np.random.default_rng(pair_seed + 5000)
        for b_idx in range(n_battles):
            battle_seed = pair_seed + b_idx + 2000
            bp_b = opp_policies[int(policy_rng.integers(0, len(opp_policies)))]

            idx_a = sel.decision((team_a, view_b), 4)
            idx_b = sel.decision((team_b, view_a), 4)

            sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
            sub_b, sub_view_b = subteam(team_b, view_b, idx_b)

            battle_teams = get_battle_teams((sub_a, sub_b), 2)
            state = State(battle_teams)
            gen = np.random.default_rng(battle_seed)
            rng_tuple = ((gen, gen), (gen, gen))
            engine = BattleEngine(state, params=params, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple)

            while not engine.finished():
                sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
                sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
                cmd0 = bp_side_a.decision(sv0, sub_view_b)
                cmd1 = bp_b.decision(sv1, sub_view_a)
                engine.run_turn((cmd0, cmd1))

            if engine.winning_side == 0:
                wins_a += 1

        win_rates.append(wins_a / max(n_battles, 1))

        if (p_idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - start
            pct = (p_idx + 1) / total * 100
            print(f"  [{label}] {p_idx + 1}/{total} pairs ({pct:.0f}%), {elapsed:.1f}s")

    return win_rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy correlation test: Greedy vs Dongimon.")
    parser.add_argument("--n-pairs", type=int, default=500, help="Number of team pairings")
    parser.add_argument("--n-battles", type=int, default=15, help="Battles per pairing per policy")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    opp_policies: list[Any] = [
        GreedyBattlePolicy(),
        _import_bp("competitors.competitor1_jjj", "JJJ_Competitor"),
    ]

    greedy_bp = GreedyBattlePolicy()
    dongimon_bp = _dongimon_bp()

    total_policy_battles = args.n_pairs * args.n_battles
    print("=" * 60)
    print("Policy Correlation Test — Greedy vs Dongimon")
    print(f"  seed={args.seed}, n_pairs={args.n_pairs}, n_battles_per_policy={args.n_battles}")
    total_both = total_policy_battles * 2
    print(f"  total battles: {total_policy_battles} Greedy + {total_policy_battles} Dongimon = {total_both}")
    print("  Side B opponents: Greedy, JJJ")
    print("=" * 60)

    start = time.perf_counter()

    teams_a = []
    teams_b = []
    for p_idx in range(args.n_pairs):
        pair_seed = args.seed + p_idx * 100
        gen_a = np.random.default_rng(pair_seed)
        gen_b = np.random.default_rng(pair_seed + 1000)
        teams_a.append(gen_team(6, 4, gen_a))
        teams_b.append(gen_team(6, 4, gen_b))

    greedy_wr = _run_with_side_a(
        greedy_bp, opp_policies, teams_a, teams_b,
        sel, params, args.seed, args.n_battles, "Greedy",
    )

    dongimon_wr = _run_with_side_a(
        dongimon_bp, opp_policies, teams_a, teams_b,
        sel, params, args.seed + 99999, args.n_battles, "Dongimon",
    )

    elapsed = time.perf_counter() - start
    rho, pval = spearmanr(greedy_wr, dongimon_wr)

    print("\n--- Results ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Greedy win rate: mean={np.mean(greedy_wr):.3f}, std={np.std(greedy_wr):.3f}")
    print(f"  Dongimon win rate: mean={np.mean(dongimon_wr):.3f}, std={np.std(dongimon_wr):.3f}")
    print(f"\n  Spearman rho: {rho:.4f} (p={pval:.4f})")

    threshold = 0.70
    if rho > threshold and pval < 0.05:
        print(f"\n  Correlation is strong (rho > {threshold}, p < 0.05).")
        print("  Greedy-labeled data can serve as a proxy for Dongimon outcomes.")
        print("  Full-scale data generation can use Greedy as the primary labeling policy.")
        print("  Recommend: 100K Greedy pairs + 20K Dongimon pairs for validation.")
    else:
        print(f"\n  Correlation is weak (rho <= {threshold} or p >= 0.05).")
        print("  Dongimon data is required in the training set.")
        print("  Recommend: 50-50 split between Greedy + Dongimon in full-scale generation.")


if __name__ == "__main__":
    main()
