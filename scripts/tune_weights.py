"""Optuna-based weight tuning for the Dongimon battle policy.

Searches over the 8 heuristic weights to maximise win rate against
the top 3 championship competitors (JJJ, minimon, StocKarpador).

Each trial runs 3 battles against each of the 3 opponents (9 battles
per trial). Results are logged to ``mlruns/`` as JSON run files.
The best weight vector is saved to ``src/config/battle_weights.yaml``
after completion.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import yaml
from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from competitor import DongimonCompetitor

N_TRIALS = 50
BATTLES_PER_MATCHUP = 3
OPPONENTS: list[tuple[str, type]] = []


def _init_opponents():
    global OPPONENTS
    if OPPONENTS:
        return
    from scripts.benchmark_competitors import _import_competitor
    OPPONENTS = [
        ("JJJ",          _import_competitor("competitors.competitor1_jjj", "JJJ_Competitor")),
        ("minimon",      _import_competitor("competitors.competitor2_minimon", "minimon")),
        ("StocKarpador", _import_competitor("competitors.competitor3_stockarpador", "StocKarpadorCompetitor")),
    ]


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: return win rate across all opponents.

    Args:
        trial: Optuna trial object.

    Returns:
        Aggregate win rate (0.0–1.0) across all opponents.
    """
    _init_opponents()
    weights = {
        "w_base_score_a":    trial.suggest_float("w_base_score_a",    0.01, 0.50),
        "w_base_score_b":    trial.suggest_float("w_base_score_b",    0.01, 0.50),
        "w_focus_fire":      trial.suggest_float("w_focus_fire",      0.01, 0.50),
        "w_target_priority": trial.suggest_float("w_target_priority", 0.01, 0.50),
        "w_survival_impact": trial.suggest_float("w_survival_impact", 0.01, 0.50),
        "w_off_def_support": trial.suggest_float("w_off_def_support", 0.01, 0.30),
        "w_setup_synergy":   trial.suggest_float("w_setup_synergy",   0.01, 0.30),
        "w_env_synergy":     trial.suggest_float("w_env_synergy",     0.01, 0.30),
    }

    total_wins = 0
    total_battles = 0
    results = {}

    for opp_name, opp_factory in OPPONENTS:
        dongimon = CompetitorManager(DongimonCompetitor(custom_weights=weights))
        opponent = CompetitorManager(opp_factory())
        match = Match((dongimon, opponent), n_battles=BATTLES_PER_MATCHUP, gen=gen_team)
        match.run()

        wins = match.wins
        d_wins = wins[0]
        o_wins = wins[1]
        total_wins += d_wins
        total_battles += d_wins + o_wins
        results[opp_name] = (d_wins, o_wins)

    win_rate = total_wins / max(total_battles, 1)

    trial.set_user_attr("results", results)
    trial.set_user_attr("weights", weights)

    return win_rate


def main():
    _init_opponents()

    study_name = f"dongimon_weight_tuning_{time.strftime('%Y%m%d_%H%M%S')}"
    study_path = os.path.join(os.path.dirname(__file__), "..", "optuna_study.db")
    storage_url = f"sqlite:///{os.path.abspath(study_path)}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
    )

    completed_trials = len(study.trials)
    if completed_trials >= N_TRIALS:
        best = study.best_trial
        print(f"Study already has {completed_trials} trials (≥{N_TRIALS} required).")
        print(f"Best trial #{best.number}: win_rate={best.value:.3f}")
        print(f"Best params: {best.params}")
    else:
        print(f"Study has {completed_trials}/{N_TRIALS} trials. Running {N_TRIALS - completed_trials} more.")
        study.optimize(objective, n_trials=N_TRIALS - completed_trials)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial #{best.number}: win_rate={best.value:.3f}")
    print("Best weights:")
    for k, v in best.params.items():
        print(f"  {k}: {v:.4f}")

    best_yaml = {k: round(v, 4) for k, v in best.params.items()}
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "src", "config", "battle_weights.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(best_yaml, f, default_flow_style=False)
    print(f"\nSaved best weights to {yaml_path}")

    record = {
        "study": study_name,
        "best_trial": best.number,
        "best_win_rate": best.value,
        "best_params": best.params,
        "trials": len(study.trials),
    }
    os.makedirs("mlruns", exist_ok=True)
    fname = f"mlruns/best_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Result saved to {fname}")


if __name__ == "__main__":
    main()
