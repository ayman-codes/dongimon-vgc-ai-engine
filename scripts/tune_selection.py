"""Optuna weight tuning for Dongimon selection pair-synergy fast path.

Evaluates each trial on FRESH procedurally generated rosters with seeded
Greedy battles. Objective = mean win rate of Dongimon's teams when the
active pair is chosen by the analytical pair-synergy fast path, across
N_ROSTERS x 4 opponents x N_BATTLES battles per trial.

Design:
    - Team build (default weights) and battle policy (Greedy pilot on both
      sides) are held fixed so the objective isolates SELECTION quality.
    - Dongimon's selection runs the pair-synergy fast path (team is final
      size = pure ordering); opponents use BasicSelectionPolicy.
    - Fresh rosters per trial force weights to generalize across pools.
    - Best weights saved to selection_synergy.yaml every SAVE_INTERVAL.

Usage:
    uv run python scripts/tune_selection.py --trials=1
    uv run python scripts/tune_selection.py            # default 300 trials
"""

import argparse
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

from src.config.models import SelectionSynergyWeights, TeambuildConfig
from src.selection.policy import DongimonSelectionPolicy
from src.teambuild.policy import HesfTeamBuildPolicy

DEFAULT_TRIALS = 300
N_BATTLES = 15
N_ROSTERS = 10
SAVE_INTERVAL = 20
STUDY_NAME = "selection_synergy_v1"
MAX_TEAM_SIZE = 4
MAX_MOVES = 4
N_ACTIVE = 2

SYNERGY_NAMES = ["w_matchup", "w_defense", "w_speed", "w_role", "w_coverage"]

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
    sel_a: Any,
    sel_b: BasicSelectionPolicy,
    params: BattleRuleParam,
    n_battles: int,
    match_seed: int,
) -> tuple[int, int]:
    """Run N seeded Greedy-piloted battles between two teams.

    Args:
        team_a: Dongimon's team.
        team_b: Opponent team.
        battle_policy: Shared battle policy instance (Greedy).
        sel_a: Dongimon selection policy (pair-synergy fast path).
        sel_b: Opponent selection policy (BasicSelectionPolicy).
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

        idx_a = sel_a.decision((team_a, view_b), MAX_TEAM_SIZE)
        idx_b = sel_b.decision((team_b, view_a), MAX_TEAM_SIZE)

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
    """Evaluate selection synergy weights via mean win rate.

    Args:
        trial: Optuna trial object providing hyperparameter suggestions.

    Returns:
        Mean win rate in [0.0, 1.0]. Higher = better weights.
    """
    raw = {n: trial.suggest_float(n, 0.01, 1.0) for n in SYNERGY_NAMES}
    total = sum(raw.values())
    weights = {n: raw[n] / total for n in SYNERGY_NAMES}
    weights["avg_weight"] = 0.6
    weights["worst_weight"] = 0.4

    syn_weights = SelectionSynergyWeights(**weights)
    trial.set_user_attr("normalized_weights", weights)

    opponents = _build_opponents()
    greedy_baseline = _GreedyBaseline()
    all_opponents: list[tuple[str, Any]] = opponents + [("Greedy", greedy_baseline)]

    greedy_bp = GreedyBattlePolicy()
    opp_sel = BasicSelectionPolicy()
    params = BattleRuleParam()
    tb_cfg = TeambuildConfig(enable_battle_royale=False)

    total_wins = 0
    total_losses = 0

    for roster_idx in range(N_ROSTERS):
        roster_seed = trial.number * 100 + roster_idx
        roster_rng = np.random.default_rng(roster_seed)

        moveset = gen_move_set(200, roster_rng)
        roster = list(gen_pkm_roster(50, moveset, MAX_MOVES, roster_rng))
        label_roster(moveset, roster)

        tb_policy = HesfTeamBuildPolicy(config=tb_cfg, ga_seed=roster_seed + 1000)
        commands = sanitized_team_build_decision(
            tb_policy, roster, None, MAX_TEAM_SIZE, MAX_MOVES, N_ACTIVE
        )
        if not commands:
            continue
        dongimon_team = build_team(commands, roster)

        my_sel = DongimonSelectionPolicy(synergy_weights=syn_weights)

        for opp_name, opp_competitor in all_opponents:
            opponent_team = _build_team_for(opp_name, opp_competitor, roster, roster_rng)
            if opponent_team is None:
                total_wins += N_BATTLES
                continue

            match_seed = roster_idx * 10000 + _OPP_SEED_OFFSET[opp_name] * 1000 + trial.number * 7
            w_a, w_b = _run_seeded_match(
                dongimon_team, opponent_team,
                greedy_bp, my_sel, opp_sel, params, N_BATTLES, match_seed,
            )
            total_wins += w_a
            total_losses += w_b

    total_decisive = total_wins + total_losses
    return total_wins / total_decisive if total_decisive > 0 else 0.0


def _save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Save best weights to selection_synergy.yaml every SAVE_INTERVAL trials.

    Args:
        study: The Optuna study instance.
        trial: The current frozen trial.
    """
    if len(study.trials) % SAVE_INTERVAL != 0:
        return
    best = study.best_trial
    best_weights = best.user_attrs["normalized_weights"]
    yaml_path = Path(__file__).parent.parent / "src" / "config" / "selection_synergy.yaml"
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
    fname = f"mlruns/best_selection_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def main() -> None:
    """Run the Optuna selection synergy weight tuning study."""
    parser = argparse.ArgumentParser(description="Tune selection pair-synergy weights with Optuna.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    n_trials = args.trials

    study_path = Path(__file__).parent.parent / "optuna_study_selection.db"
    storage_url = f"sqlite:///{os.path.abspath(study_path)}"
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage_url,
        load_if_exists=True, direction="maximize",
    )
    completed = len(study.trials)
    if completed >= n_trials:
        best = study.best_trial
        print(f"Study already has {completed} trials (>= {n_trials}).")
        print(f"Best trial #{best.number}: win_rate={best.value:.4f}")
    else:
        print(f"Study '{STUDY_NAME}' has {completed}/{n_trials} trials.")
        print(f"Running {n_trials - completed} more...")
        print(f"  N_ROSTERS={N_ROSTERS}  N_BATTLES={N_BATTLES}  opponents=4")
        print(f"  battles_per_trial={N_ROSTERS * 4 * N_BATTLES}")
        study.optimize(objective, n_trials=n_trials - completed, callbacks=[_save_callback])

    best = study.best_trial
    print(f"\nBest trial #{best.number}: win_rate={best.value:.4f}")
    for k, v in best.user_attrs["normalized_weights"].items():
        print(f"  {k}: {v:.4f}")

    yaml_path = Path(__file__).parent.parent / "src" / "config" / "selection_synergy.yaml"
    best_weights = best.user_attrs["normalized_weights"]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: round(v, 4) for k, v in best_weights.items()}, f, default_flow_style=False)
    print(f"Saved to {yaml_path}")


if __name__ == "__main__":
    main()
