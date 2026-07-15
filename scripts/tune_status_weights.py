"""Optuna tuning for 5 status weights against minimon.

25 trials, 3 battles per trial vs minimon only.
Keeps existing 8 synergy weights fixed.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor

BATTLES = 3
SEED = 42


def objective(trial):
    sw = {
        "w_status_sleep": trial.suggest_float("w_status_sleep", 5, 80),
        "w_status_burn": trial.suggest_float("w_status_burn", 5, 80),
        "w_status_para": trial.suggest_float("w_status_para", 5, 80),
        "w_status_poison": trial.suggest_float("w_status_poison", 5, 80),
        "w_status_toxic": trial.suggest_float("w_status_toxic", 5, 80),
    }

    rng = np.random.default_rng(SEED)
    team = gen_team(6, 4, rng)
    view = TeamView(team)
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()

    from competitors.competitor2_minimon import minimon

    dongimon = DongimonCompetitor(custom_weights=sw)
    opp = minimon()

    wins = 0
    for _ in range(BATTLES):
        idx_a = sel.decision((team, view), 4)
        idx_b = sel.decision((team, view), 4)
        ta, va = subteam(team, view, idx_a)
        tb, vb = subteam(team, view, idx_b)
        bt = get_battle_teams((ta, tb), 2)
        state = State(bt)
        eng = BattleEngine(state, params=params)
        while not eng.finished():
            eng.run_turn((
                dongimon.battlepolicy.decision(StateView(eng.state, 0, (va, vb)), vb),
                opp.battlepolicy.decision(StateView(eng.state, 1, (vb, va)), va),
            ))
        if eng.winning_side == 0:
            wins += 1

    wr = wins / BATTLES
    trial.set_user_attr("wins", wins)
    return wr


def main():
    study_name = f"status_tuning_{time.strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'optuna_study.db')}"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="maximize")
    completed = len(study.trials)
    if completed >= 25:
        print(f"Already have {completed} trials. Best: {study.best_value:.3f}")
    else:
        print(f"Running {25 - completed} more trials (total needed: 25)")
        study.optimize(objective, n_trials=25 - completed)

    best = study.best_trial
    print(f"\nBest trial #{best.number}: win_rate={best.value:.3f}")
    for k, v in best.params.items():
        print(f"  {k}: {v:.1f}")


if __name__ == "__main__":
    main()
