"""Optuna weight tuning for Dongimon battle policy using mean win rate.

Evaluates each trial on a FIXED pool of 10 pre-generated teams with
fully seeded battles. Every trial sees identical conditions — the only
variable is the weight vector. Objective is mean win rate across all
teams × opponents × battles (1000 deterministic battles per trial).

Design principles:
    - Fixed team pool eliminates team-luck variance.
    - Seeded RNG (acc, eff, sta) makes every battle reproducible.
    - Mean win rate is bounded [0, 1], path-independent, and directly
      interpretable unlike ELO which depends on matchup ordering.
    - w_survival_impact capped at 0.4 to prevent structural dominance
      of the negative survival penalty over positive damage scoring.

Usage:
    uv run python scripts/tune_weights.py
"""

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import yaml
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.battle.policy import DongimonBattlePolicy

N_TRIALS = 400
N_BATTLES = 25
N_TEAMS = 10
SAVE_INTERVAL = 25
STUDY_NAME = "winrate_fixed_teams_v1"

TEAM_SEEDS: list[int] = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]

_OPPONENTS: dict[str, Callable[[], Any]] | None = None
_TEAM_POOL: list[tuple[Any, TeamView]] | None = None


def _build_opponents() -> dict[str, Callable[[], Any]]:
    """Import opponent battle policies once and cache them.

    Returns:
        Dict mapping opponent name to a zero-argument factory callable
        that returns a fresh BattlePolicy instance.
    """
    global _OPPONENTS
    if _OPPONENTS is not None:
        return _OPPONENTS

    from importlib import import_module

    _OPPONENTS = {}
    _OPPONENTS["Greedy"] = lambda: GreedyBattlePolicy()

    for name, mod_path, cls_name in [
        ("JJJ", "competitors.competitor1_jjj", "JJJ_Competitor"),
        ("minimon", "competitors.competitor2_minimon", "minimon"),
        ("caaaden", "competitors.competitor_caaaden", "CaaadenCompetitor"),
    ]:
        mod = import_module(mod_path)
        cls = getattr(mod, cls_name)

        def _factory(c: Any = cls) -> Any:
            return c().battlepolicy

        _OPPONENTS[name] = _factory

    return _OPPONENTS


def _build_team_pool() -> list[tuple[Any, TeamView]]:
    """Pre-generate the fixed team pool used across all trials.

    Returns:
        List of (team, team_view) tuples, one per TEAM_SEEDS entry.
    """
    global _TEAM_POOL
    if _TEAM_POOL is not None:
        return _TEAM_POOL

    _TEAM_POOL = []
    for seed in TEAM_SEEDS:
        rng = np.random.default_rng(seed)
        team = gen_team(4, 4, rng)
        view = TeamView(team)
        _TEAM_POOL.append((team, view))

    return _TEAM_POOL


def _run_seeded_match(
    bp_a: Any,
    bp_b: Any,
    base_team: Any,
    base_view: TeamView,
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
    match_seed: int,
) -> tuple[int, int]:
    """Run N fully-seeded battles between two battle policies.

    Each battle within the match gets a unique deterministic seed
    derived from match_seed + battle_index.

    Args:
        bp_a: First battle policy instance.
        bp_b: Second battle policy instance.
        base_team: Shared vgc2 Team object.
        base_view: Shared TeamView.
        sel: BasicSelectionPolicy for both sides.
        params: Battle rule parameters.
        n_battles: Number of battles to run.
        match_seed: Base seed for this match (determines all RNG).

    Returns:
        Tuple of (wins_a, wins_b).
    """
    wins_a = 0
    wins_b = 0

    for b_idx in range(n_battles):
        battle_seed = match_seed + b_idx
        gen = np.random.default_rng(battle_seed)

        idx_a = sel.decision((base_team, base_view), 4)
        idx_b = sel.decision((base_team, base_view), 4)

        sub_a, sub_view_a = subteam(base_team, base_view, idx_a)
        sub_b, sub_view_b = subteam(base_team, base_view, idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), 2)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(
            state, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            cmd0 = bp_a.decision(sv0, sub_view_b)
            cmd1 = bp_b.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
        elif engine.winning_side == 1:
            wins_b += 1

    return wins_a, wins_b


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: mean win rate across fixed team pool.

    Evaluates Dongimon's battle policy against all opponents on every
    team in the fixed pool. Returns total_wins / total_decisive_battles.

    Args:
        trial: Optuna trial object providing hyperparameter suggestions.

    Returns:
        Mean win rate in [0.0, 1.0]. Higher = better weights.
    """
    weights = {
        "w_base_score": trial.suggest_float("w_base_score", 0.05, 1.0),
        "w_focus_fire": trial.suggest_float("w_focus_fire", 0.0, 1.0),
        "w_target_priority": trial.suggest_float("w_target_priority", 0.0, 1.0),
        "w_survival_impact": trial.suggest_float("w_survival_impact", 0.0, 0.4),
        "w_off_def_support": trial.suggest_float("w_off_def_support", 0.0, 1.0),
        "w_setup_synergy": trial.suggest_float("w_setup_synergy", 0.0, 1.0),
        "w_env_synergy": trial.suggest_float("w_env_synergy", 0.0, 1.0),
        "w_status_sleep": trial.suggest_float("w_status_sleep", 5.0, 80.0),
        "w_status_burn": trial.suggest_float("w_status_burn", 5.0, 80.0),
        "w_status_para": trial.suggest_float("w_status_para", 5.0, 80.0),
        "w_status_poison": trial.suggest_float("w_status_poison", 5.0, 80.0),
        "w_status_toxic": trial.suggest_float("w_status_toxic", 5.0, 80.0),
    }

    dongimon_bp = DongimonBattlePolicy(custom_weights=weights)
    opponent_factories = _build_opponents()
    team_pool = _build_team_pool()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    total_wins = 0
    total_losses = 0

    for team_idx, (team, view) in enumerate(team_pool):
        opponent_bps = {name: factory() for name, factory in opponent_factories.items()}

        for opp_idx, (_opp_name, opp_bp) in enumerate(opponent_bps.items()):
            match_seed = team_idx * 10000 + opp_idx * 1000 + trial.number * 7
            wins_d, wins_o = _run_seeded_match(
                dongimon_bp, opp_bp,
                team, view, sel, params,
                N_BATTLES, match_seed,
            )
            total_wins += wins_d
            total_losses += wins_o

    total_decisive = total_wins + total_losses
    win_rate = total_wins / total_decisive if total_decisive > 0 else 0.0

    trial.set_user_attr("weights", weights)
    trial.set_user_attr("total_wins", total_wins)
    trial.set_user_attr("total_losses", total_losses)
    trial.set_user_attr("win_rate", win_rate)

    return win_rate


def _save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Optuna callback: save best weights to YAML after every SAVE_INTERVAL trials.

    Args:
        study: The Optuna study instance.
        trial: The current frozen trial (unused -- we read study.best_trial).
    """
    if len(study.trials) % SAVE_INTERVAL != 0:
        return

    best = study.best_trial
    best_yaml = {k: round(v, 4) for k, v in best.params.items()}
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "config",
        "battle_weights.yaml",
    )
    with open(yaml_path, "w") as f:
        yaml.safe_dump(best_yaml, f, default_flow_style=False)

    record = {
        "study": STUDY_NAME,
        "best_trial": best.number,
        "best_win_rate": best.value,
        "best_params": best.params,
        "trials_completed": len(study.trials),
    }
    os.makedirs("mlruns", exist_ok=True)
    fname = f"mlruns/best_winrate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)


def main() -> None:
    """Run the Optuna win-rate weight tuning study."""
    study_path = os.path.join(os.path.dirname(__file__), "..", "optuna_study.db")
    storage_url = f"sqlite:///{os.path.abspath(study_path)}"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
    )

    completed = len(study.trials)
    if completed >= N_TRIALS:
        best = study.best_trial
        print(f"Study already has {completed} trials (>= {N_TRIALS} required).")
        print(f"Best trial #{best.number}: win_rate={best.value:.4f}")
        print(f"Best params: {best.params}")
    else:
        print(f"Study '{STUDY_NAME}' has {completed}/{N_TRIALS} trials.")
        print(f"Running {N_TRIALS - completed} more trials...")
        print(f"  N_TEAMS={N_TEAMS}  N_BATTLES={N_BATTLES}  opponents=4")
        print(f"  battles_per_trial={N_TEAMS * 4 * N_BATTLES}")
        study.optimize(objective, n_trials=N_TRIALS - completed, callbacks=[_save_callback])

    best = study.best_trial
    print(f"\n{'=' * 60}")
    print(f"Best trial #{best.number}: win_rate={best.value:.4f}")
    print("Best weights:")
    for k, v in best.params.items():
        print(f"  {k}: {v:.4f}")

    best_yaml = {k: round(v, 4) for k, v in best.params.items()}
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "config",
        "battle_weights.yaml",
    )
    with open(yaml_path, "w") as f:
        yaml.safe_dump(best_yaml, f, default_flow_style=False)
    print(f"\nSaved best weights to {yaml_path}")

    record = {
        "study": STUDY_NAME,
        "best_trial": best.number,
        "best_win_rate": best.value,
        "best_params": best.params,
        "trials": len(study.trials),
    }
    os.makedirs("mlruns", exist_ok=True)
    fname = f"mlruns/best_winrate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Result saved to {fname}")


if __name__ == "__main__":
    main()
