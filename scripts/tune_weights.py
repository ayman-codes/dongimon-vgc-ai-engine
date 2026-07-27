"""Optuna weight tuning for Dongimon battle policy using ELO rating.

Evaluates each trial by running an isolated ELO tournament where all
competitors share the same procedurally-generated team and use
BasicSelectionPolicy. Only the BattlePolicy differs, isolating weight
effects from teambuild/selection variance.

Each trial runs 10 ELO epochs (pairing → battle → rating update).
The objective is Dongimon's final ELO after all epochs.
"""

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import yaml
from numpy.random import default_rng
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.battle.policy import DongimonBattlePolicy
from src.tuning.elo_rating import run_dongimon_elo_epoch

N_TRIALS = 1500
N_BATTLES = 15
N_EPOCHS = 10
ELO_K = 32.0
INITIAL_ELO = 1200.0
STUDY_NAME = "elo_weight_tuning_v5_fixed_scale"
SAVE_INTERVAL = 30

_DONGIMON = "Dongimon"
_GREEDY = "Greedy"
_OPPONENTS: dict[str, Callable[[], Any]] | None = None


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
    _OPPONENTS[_GREEDY] = lambda: GreedyBattlePolicy()

    for name, mod_path, cls_name in [
        ("JJJ", "competitors.competitor1_jjj", "JJJ_Competitor"),
        ("minimon", "competitors.competitor2_minimon", "minimon"),
        ("caaaden", "competitors.competitor_caaaden", "CaaadenCompetitor")
    ]:
        mod = import_module(mod_path)
        cls = getattr(mod, cls_name)

        def _factory(c: Any = cls) -> Any:
            return c().battlepolicy

        _OPPONENTS[name] = _factory

    return _OPPONENTS


def _run_match(
    bp_a: Any,
    bp_b: Any,
    base_team: Any,
    base_view: Any,
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
) -> tuple[int, int]:
    """Run N battles between two battle policies on the same team.

    Both sides use BasicSelectionPolicy so team selection is
    identical and only the battle policy differs.

    Args:
        bp_a: First battle policy instance.
        bp_b: Second battle policy instance.
        base_team: Shared vgc2 Team object.
        base_view: Shared TeamView.
        sel: BasicSelectionPolicy for both sides.
        params: Battle rule parameters.
        n_battles: Number of battles to run.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    wins_a = 0
    wins_b = 0

    for _ in range(n_battles):
        idx_a = sel.decision((base_team, base_view), 4)
        idx_b = sel.decision((base_team, base_view), 4)

        sub_a, sub_view_a = subteam(base_team, base_view, idx_a)
        sub_b, sub_view_b = subteam(base_team, base_view, idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), 2)
        state = State(battle_teams)
        engine = BattleEngine(state, params=params)

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
    """Optuna objective: return Dongimon's final ELO after an isolated tournament.

    Generates one random team per trial, shared across all policies.
    Runs N_EPOCHS rounds of ELO-paired matches with N_BATTLES per matchup.

    Args:
        trial: Optuna trial object providing hyperparameter suggestions.

    Returns:
        Dongimon's ELO rating after all epochs. Higher = better weights.
    """
    weights = {
        "w_base_score": trial.suggest_float("w_base_score", 0.0, 1.0),
        "w_focus_fire": trial.suggest_float("w_focus_fire", 0.0, 1.0),
        "w_target_priority": trial.suggest_float("w_target_priority", 0.0, 1.0),
        "w_survival_impact": trial.suggest_float("w_survival_impact", 0.0, 1.0),
        "w_off_def_support": trial.suggest_float("w_off_def_support", 0.0, 1.0),
        "w_setup_synergy": trial.suggest_float("w_setup_synergy", 0.0, 1.0),
        "w_env_synergy": trial.suggest_float("w_env_synergy", 0.0, 1.0),
        "w_status_sleep": trial.suggest_float("w_status_sleep", 5.0, 80.0),
        "w_status_burn": trial.suggest_float("w_status_burn", 5.0, 80.0),
        "w_status_para": trial.suggest_float("w_status_para", 5.0, 80.0),
        "w_status_poison": trial.suggest_float("w_status_poison", 5.0, 80.0),
        "w_status_toxic": trial.suggest_float("w_status_toxic", 5.0, 80.0),
    }

    if weights["w_base_score"] < 0.05:
        return float("-inf")

    rng = default_rng(trial.number)
    shared_team = gen_team(6, 4, rng)
    shared_view = TeamView(shared_team)
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    dongimon_bp = DongimonBattlePolicy(custom_weights=weights)
    opponent_factories = _build_opponents()
    opponent_names = list(opponent_factories.keys())
    opponent_bps = {name: factory() for name, factory in opponent_factories.items()}

    elos: dict[str, float] = dict.fromkeys([_DONGIMON] + opponent_names, INITIAL_ELO)

    def battle_runner(p1_name: str, p2_name: str) -> tuple[int, int]:
        bp1 = dongimon_bp if p1_name == _DONGIMON else opponent_bps[p1_name]
        bp2 = dongimon_bp if p2_name == _DONGIMON else opponent_bps[p2_name]
        return _run_match(bp1, bp2, shared_team, shared_view, sel, params, N_BATTLES)

    for _epoch in range(N_EPOCHS):
        elos = run_dongimon_elo_epoch(elos, _DONGIMON, opponent_names, battle_runner, k=ELO_K)

    trial.set_user_attr("weights", weights)
    for name, rating in elos.items():
        trial.set_user_attr(f"elo_{name}", rating)

    return elos.get(_DONGIMON, 0.0)


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
        "best_elo": best.value,
        "best_params": best.params,
        "trials_completed": len(study.trials),
    }
    os.makedirs("mlruns", exist_ok=True)
    fname = f"mlruns/best_elo_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)


def main() -> None:
    """Run the Optuna ELO weight tuning study."""
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
        print(f"Best trial #{best.number}: elo={best.value:.1f}")
        print(f"Best params: {best.params}")
    else:
        print(f"Study '{STUDY_NAME}' has {completed}/{N_TRIALS} trials.")
        print(f"Running {N_TRIALS - completed} more trials...")
        print(f"  N_EPOCHS={N_EPOCHS}  N_BATTLES={N_BATTLES}  ELO_K={ELO_K}")
        study.optimize(objective, n_trials=N_TRIALS - completed, callbacks=[_save_callback])

    best = study.best_trial
    print(f"\n{'=' * 60}")
    print(f"Best trial #{best.number}: elo={best.value:.1f}")
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
        "best_elo": best.value,
        "best_params": best.params,
        "trials": len(study.trials),
    }
    os.makedirs("mlruns", exist_ok=True)
    fname = f"mlruns/best_elo_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Result saved to {fname}")


if __name__ == "__main__":
    main()
