"""Optuna weight tuning for Dongimon teambuild policy.

Evaluates each trial on FRESH procedurally generated rosters with
fully seeded GA and seeded Greedy battles. Objective = mean win rate
of Dongimon's built teams vs opponent teams across 15 rosters x 4
opponents x 25 battles = 1500 battles per trial.

Design:
    - Fresh rosters per trial prevent overfitting to specific species
      distributions — weights must generalize across diverse pools.
    - Seeded GA (via ga_seed) makes same trial = same team = reproducible.
    - Opponent teams are built once per roster via their own teambuild
      policies (deterministic, near-instant for non-Dongimon competitors).
    - Greedy pilot on both sides isolates teambuild quality from battle skill.
    - Battle royale disabled to isolate GA + archetype weights cleanly.
    - Best weights saved to teambuild_weights.yaml every 20 trials.

Usage:
    uv run python scripts/tune_teambuild.py
"""

import json
import os
import sys
import time
from importlib import import_module
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
from vgc2.battle_engine.modifiers import Nature
from vgc2.battle_engine.pokemon import Pokemon
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.ecosystem import build_team, label_roster, sanitized_team_build_decision
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from src.config.models import TeambuildConfig, TeambuildWeights
from src.teambuild.policy import HesfTeamBuildPolicy

N_TRIALS = 400
N_BATTLES = 25
N_ROSTERS = 15
SAVE_INTERVAL = 20
STUDY_NAME = "teambuild_weights_v1"
MAX_TEAM_SIZE = 4
MAX_MOVES = 4
N_ACTIVE = 2

ARCH_NAMES = ["w_stat", "w_speed", "w_dmg", "w_util", "w_stat_syn", "w_speed_syn"]
GA_NAMES = [
    "ga_viability", "ga_coverage", "ga_defence",
    "ga_stat_diversity", "ga_role_diversity", "ga_coverage_balance",
]

_OPP_SEED_OFFSET: dict[str, int] = {"JJJ": 1, "minimon": 2, "caaaden": 3, "Greedy": 4}
_OPPONENT_FACTORIES: list[tuple[str, Any]] | None = None


def _build_opponents() -> list[tuple[str, Any]]:
    """Import and cache opponent competitors.

    Returns:
        List of (name, competitor_instance) tuples.
    """
    global _OPPONENT_FACTORIES
    if _OPPONENT_FACTORIES is not None:
        return [(n, cls()) for n, cls in _OPPONENT_FACTORIES]

    _OPPONENT_FACTORIES = []
    for name, mod_path, cls_name in [
        ("JJJ", "competitors.competitor1_jjj", "JJJ_Competitor"),
        ("minimon", "competitors.competitor2_minimon", "minimon"),
        ("caaaden", "competitors.competitor_caaaden", "CaaadenCompetitor"),
    ]:
        mod = import_module(mod_path)
        cls = getattr(mod, cls_name)
        _OPPONENT_FACTORIES.append((name, cls))

    return [(n, cls()) for n, cls in _OPPONENT_FACTORIES]


class _GreedyBaseline:
    """Baseline competitor: random team + BasicSelection + Greedy battle."""

    def __init__(self) -> None:
        """Initialize the greedy baseline with a basic selection policy."""
        self.selectionpolicy = BasicSelectionPolicy()
        self.teambuildpolicy = None

    def build_team(self, roster: list[Any], rng: np.random.Generator) -> Team:
        """Build a random team from the roster.

        Args:
            roster: List of PokemonSpecies to select from.
            rng: NumPy Generator for reproducibility.

        Returns:
            A Team with random members, moves, EVs, and nature.
        """
        indices = rng.choice(len(roster), size=min(MAX_TEAM_SIZE, len(roster)), replace=False)
        members = []
        for idx in indices:
            species = roster[int(idx)]
            n_moves = len(species.moves)
            move_idx = list(rng.choice(n_moves, size=min(MAX_MOVES, n_moves), replace=False))
            evs = tuple(int(x) for x in rng.multinomial(510, [1 / 6] * 6))
            members.append(Pokemon(species, move_idx, 100, evs, (31,) * 6, Nature.SERIOUS))
        return Team(members)


def _build_team_for(
    name: str, competitor: Any, roster: list[Any], build_rng: np.random.Generator
) -> Team | None:
    """Build a team using a competitor's teambuild policy.

    Args:
        name: Competitor name for special handling.
        competitor: Competitor instance with teambuildpolicy attribute.
        roster: Full species roster.
        build_rng: NumPy Generator for random team construction.

    Returns:
        Built Team or None on failure.
    """
    if name == "Greedy":
        return _GreedyBaseline().build_team(roster, build_rng)
    try:
        commands = sanitized_team_build_decision(
            competitor.teambuildpolicy, roster, None, MAX_TEAM_SIZE, MAX_MOVES, N_ACTIVE,
        )
        if not commands:
            return None
        return build_team(commands, roster)
    except Exception:
        return None


def _run_seeded_match(
    team_a: Team,
    team_b: Team,
    battle_policy: Any,
    sel: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
    match_seed: int,
) -> tuple[int, int]:
    """Run N seeded Greedy-piloted battles between two teams.

    Args:
        team_a: First team.
        team_b: Second team.
        battle_policy: Shared battle policy instance (Greedy).
        sel: Basic selection policy for subteam selection.
        params: Battle rule parameters.
        n_battles: Number of battles to run.
        match_seed: Base seed for deterministic RNG.

    Returns:
        Tuple of (wins_a, wins_b).
    """
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)

    wins_a = 0
    wins_b = 0
    for b_idx in range(n_battles):
        battle_seed = match_seed + b_idx
        gen = np.random.default_rng(battle_seed)

        idx_a = sel.decision((team_a, view_b), MAX_TEAM_SIZE)
        idx_b = sel.decision((team_b, view_a), MAX_TEAM_SIZE)

        sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
        sub_b, sub_view_b = subteam(team_b, view_b, idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), N_ACTIVE)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(
            state, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            cmd0 = battle_policy.decision(sv0, sub_view_b)
            cmd1 = battle_policy.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
        elif engine.winning_side == 1:
            wins_b += 1
    return wins_a, wins_b


def objective(trial: optuna.Trial) -> float:
    """Evaluate teambuild weights via mean win rate across diverse rosters.

    Args:
        trial: Optuna trial object providing hyperparameter suggestions.

    Returns:
        Mean win rate in [0.0, 1.0]. Higher = better weights.
    """
    arch_raw = {n: trial.suggest_float(n, 0.01, 1.0) for n in ARCH_NAMES}
    arch_total = sum(arch_raw.values())
    arch_weights = {n: arch_raw[n] / arch_total for n in ARCH_NAMES}

    ga_raw = {n: trial.suggest_float(n, 0.01, 1.0) for n in GA_NAMES}
    ga_total = sum(ga_raw.values())
    ga_weights = {n: ga_raw[n] / ga_total for n in GA_NAMES}

    all_weights = {**arch_weights, **ga_weights}
    tb_weights = TeambuildWeights(**all_weights)

    trial.set_user_attr("normalized_weights", all_weights)

    opponents = _build_opponents()
    greedy_baseline = _GreedyBaseline()
    all_opponents: list[tuple[str, Any]] = opponents + [("Greedy", greedy_baseline)]

    greedy_bp = GreedyBattlePolicy()
    sel = BasicSelectionPolicy()
    params = BattleRuleParam()
    tune_cfg = TeambuildConfig(enable_battle_royale=False)

    total_wins = 0
    total_losses = 0

    for roster_idx in range(N_ROSTERS):
        roster_seed = trial.number * 100 + roster_idx
        roster_rng = np.random.default_rng(roster_seed)

        moveset = gen_move_set(200, roster_rng)
        roster = list(gen_pkm_roster(50, moveset, MAX_MOVES, roster_rng))
        label_roster(moveset, roster)

        ga_seed = roster_seed + 1000
        tb_policy = HesfTeamBuildPolicy(config=tune_cfg, custom_weights=tb_weights, ga_seed=ga_seed)
        commands = sanitized_team_build_decision(
            tb_policy, roster, None, MAX_TEAM_SIZE, MAX_MOVES, N_ACTIVE
        )
        if not commands:
            continue
        dongimon_team = build_team(commands, roster)

        for opp_name, opp_competitor in all_opponents:
            opponent_team = _build_team_for(opp_name, opp_competitor, roster, roster_rng)
            if opponent_team is None:
                total_wins += N_BATTLES
                continue

            match_seed = roster_idx * 10000 + _OPP_SEED_OFFSET[opp_name] * 1000 + trial.number * 7
            w_a, w_b = _run_seeded_match(
                dongimon_team, opponent_team,
                greedy_bp, sel, params, N_BATTLES, match_seed,
            )
            total_wins += w_a
            total_losses += w_b

    total_decisive = total_wins + total_losses
    win_rate = total_wins / total_decisive if total_decisive > 0 else 0.0
    return win_rate


def _save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Save best weights to teambuild_weights.yaml every SAVE_INTERVAL trials.

    Args:
        study: The Optuna study instance.
        trial: The current frozen trial.
    """
    if len(study.trials) % SAVE_INTERVAL != 0:
        return
    best = study.best_trial
    best_weights = best.user_attrs["normalized_weights"]
    yaml_path = Path(__file__).parent.parent / "src" / "config" / "teambuild_weights.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: round(v, 4) for k, v in best_weights.items()}, f, default_flow_style=False)

    record = {
        "study": STUDY_NAME,
        "best_trial": best.number,
        "best_win_rate": best.value,
        "best_params": best_weights,
        "trials_completed": len(study.trials),
    }
    os.makedirs("mlruns", exist_ok=True)
    fname = f"mlruns/best_teambuild_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def main() -> None:
    """Run the Optuna teambuild weight tuning study."""
    study_path = Path(__file__).parent.parent / "optuna_study_teambuild.db"
    storage_url = f"sqlite:///{os.path.abspath(study_path)}"
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage_url,
        load_if_exists=True, direction="maximize",
    )
    completed = len(study.trials)
    if completed >= N_TRIALS:
        best = study.best_trial
        print(f"Study already has {completed} trials.")
        print(f"Best trial #{best.number}: win_rate={best.value:.4f}")
    else:
        print(f"Study '{STUDY_NAME}' has {completed}/{N_TRIALS} trials.")
        print(f"Running {N_TRIALS - completed} more...")
        print(f"  N_ROSTERS={N_ROSTERS}  N_BATTLES={N_BATTLES}  opponents=4")
        print(f"  battles_per_trial={N_ROSTERS * 4 * N_BATTLES}")
        study.optimize(objective, n_trials=N_TRIALS - completed, callbacks=[_save_callback])

    best = study.best_trial
    print(f"\nBest trial #{best.number}: win_rate={best.value:.4f}")
    for k, v in best.user_attrs["normalized_weights"].items():
        print(f"  {k}: {v:.4f}")

    yaml_path = Path(__file__).parent.parent / "src" / "config" / "teambuild_weights.yaml"
    best_weights = best.user_attrs["normalized_weights"]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: round(v, 4) for k, v in best_weights.items()}, f, default_flow_style=False)
    print(f"Saved to {yaml_path}")


if __name__ == "__main__":
    main()
