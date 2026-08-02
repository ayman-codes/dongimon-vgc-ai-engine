"""Championship-track competition benchmark mirroring the vgc2 engine.

Mirrors ``organization/run_championship_track.py`` from the vgc2 engine
exactly: a shared seeded roster and move set are generated, labelled, and
wrapped in a ``BasicMeta``; the six repo competitors are registered into
``vgc2.competition.ecosystem.Championship`` and compete over ``epochs``
waves. Each wave rebuilds every team, pairs competitors by the configured
pairing strategy, runs ``n_battles`` per matchup, and updates ELO. The
final ranking is printed with the canonical overall score.

Unlike the engine script, competitors run in-process (no proxy servers):
every competitor is registered directly via ``CompetitorManager``.

The field is six competitors so the default ELO pairing never leaves an
agent idle: Dongimon, JJJ, minimon, caaaden, Greedy (vgc2 built-ins), and
peach (submission battle policy).

Usage:
    uv run python scripts/championship_track.py --epochs=100 --n-battles=3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.balance.meta import BasicMeta
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.ecosystem import PAIRING_STRATEGY, RANKING_STRATEGY, Championship, label_roster
from vgc2.competition.score import elo_score, time_score
from vgc2.util.generator import gen_move_set, gen_pkm_roster

N_MOVES = 100
ROSTER_SIZE = 50
MAX_TEAM_SIZE = 4
MAX_PKM_MOVES = 4
N_ACTIVE = 2
ELO_WEIGHT = 0.99
TIME_WEIGHT = 0.01


class GreedyCompetitor(Competitor):  # type: ignore[misc]
    """Competitor using only vgc2 built-in policies.

    Battle, selection, and teambuild are all the engine's stock Greedy /
    Random policies, giving the championship a simple baseline entry.
    """

    def __init__(self, name: str = "Greedy") -> None:
        """Initialize the built-in-policies competitor.

        Args:
            name: Display name used in rankings.
        """
        self.__name = name
        self.__battle_policy = GreedyBattlePolicy()
        self.__selection_policy = RandomSelectionPolicy()
        self.__team_build_policy = RandomTeamBuildPolicy()

    @property
    def name(self) -> str:
        """Return the competitor display name.

        Returns:
            The competitor name.
        """
        return self.__name

    @property
    def battlepolicy(self) -> Any:
        """Return the battle policy instance.

        Returns:
            The vgc2 Greedy battle policy.
        """
        return self.__battle_policy

    @property
    def selectionpolicy(self) -> Any:
        """Return the selection policy instance.

        Returns:
            The vgc2 random selection policy.
        """
        return self.__selection_policy

    @property
    def teambuildpolicy(self) -> Any:
        """Return the teambuild policy instance.

        Returns:
            The vgc2 random teambuild policy.
        """
        return self.__team_build_policy


class PeachCompetitor(Competitor):  # type: ignore[misc]
    """Competitor wrapping the peach submission battle policy.

    Loads the peach competitor from the competition submissions
    directory; selection and teambuild fall back to the vgc2 random
    policies when the submission does not provide them.
    """

    def __init__(self, name: str = "peach") -> None:
        """Initialize the peach competitor.

        Args:
            name: Display name used in rankings.
        """
        from competitors.competitor_peach import PeachCompetitor as _PeachSubmission

        self.__name = name
        inner = _PeachSubmission()
        self.__battle_policy = getattr(inner, "battlepolicy", None) or GreedyBattlePolicy()
        self.__selection_policy = getattr(inner, "selectionpolicy", None) or RandomSelectionPolicy()
        self.__team_build_policy = getattr(inner, "teambuildpolicy", None) or RandomTeamBuildPolicy()

    @property
    def name(self) -> str:
        """Return the competitor display name.

        Returns:
            The competitor name.
        """
        return self.__name

    @property
    def battlepolicy(self) -> Any:
        """Return the battle policy instance.

        Returns:
            The peach submission battle policy.
        """
        return self.__battle_policy

    @property
    def selectionpolicy(self) -> Any:
        """Return the selection policy instance.

        Returns:
            The vgc2 random selection policy or the submission's.
        """
        return self.__selection_policy

    @property
    def teambuildpolicy(self) -> Any:
        """Return the teambuild policy instance.

        Returns:
            The vgc2 random teambuild policy or the submission's.
        """
        return self.__team_build_policy


def _print_roster(move_set: Any, roster: Any) -> None:
    """Print the generated move set and species roster.

    Mirrors the engine's roster printout so every championship run is
    auditable against the canonical script.

    Args:
        move_set: The generated list of moves.
        roster: The generated list of Pokemon species.
    """
    print("## Move Set ##")
    for move in move_set:
        print(move)
    print()
    print("## Roster ##")
    for species in roster:
        print(species)
    print()


def _build_competitors() -> list[Competitor]:
    """Construct the six championship competitors.

    Dongimon uses the production pipeline (GreedyDongi battle policy plus
    the tuned selection and teambuild policies); the external bots are
    imported from ``competitors/``; Greedy is a local wrapper.

    Returns:
        List of the six competitors in registration order.
    """
    from competitor import DongimonCompetitor
    from competitors.competitor1_jjj import JJJ_Competitor
    from competitors.competitor2_minimon import minimon
    from competitors.competitor_caaaden import CaaadenCompetitor

    return [
        DongimonCompetitor(),
        JJJ_Competitor(),
        minimon(),
        CaaadenCompetitor(),
        GreedyCompetitor(),
        PeachCompetitor(),
    ]


def main() -> None:
    """Run the championship track and print the final ranking."""
    parser = argparse.ArgumentParser(description="Championship-track competition benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for roster and move-set generation")
    parser.add_argument("--epochs", type=int, default=100, help="Number of championship waves")
    parser.add_argument("--n-moves", type=int, default=N_MOVES, help="Size of the generated move set")
    parser.add_argument("--roster-size", type=int, default=ROSTER_SIZE, help="Number of species in the roster")
    parser.add_argument("--max-team-size", type=int, default=MAX_TEAM_SIZE, help="Maximum team size")
    parser.add_argument("--n-active", type=int, default=N_ACTIVE, help="Number of active Pokemon per side")
    parser.add_argument("--max-pkm-moves", type=int, default=MAX_PKM_MOVES, help="Moves per Pokemon")
    parser.add_argument("--n-battles", type=int, default=3, help="Battles per matchup per wave")
    parser.add_argument("--pair-strategy", choices=PAIRING_STRATEGY.keys(), default="elo",
                        help="Pairing strategy")
    parser.add_argument("--ranking-strategy", choices=RANKING_STRATEGY.keys(), default="overall",
                        help="Ranking strategy")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    move_set = gen_move_set(args.n_moves, rng)
    roster = gen_pkm_roster(args.roster_size, move_set, args.max_pkm_moves, rng)
    _print_roster(move_set, roster)
    label_roster(move_set, roster)
    meta = BasicMeta(move_set, roster)

    championship = Championship(
        roster,
        meta,
        args.epochs,
        args.n_active,
        args.n_battles,
        args.max_team_size,
        args.max_pkm_moves,
        PAIRING_STRATEGY[args.pair_strategy],
        RANKING_STRATEGY[args.ranking_strategy],
        None,
        ELO_WEIGHT,
        TIME_WEIGHT,
    )
    for competitor in _build_competitors():
        championship.register(CompetitorManager(competitor))

    championship.run()
    ranking = championship.ranking()
    winner = ranking[0]
    print(winner.competitor.name + " wins the championship!")
    overall = (
        championship.elo_weight * elo_score(winner.elo)
        + championship.time_weight * time_score(winner.time / championship.epochs, 0.001, 1)
    )
    print(
        f"ELO {int(winner.elo)} "
        f"Time {winner.time / championship.epochs:.5f} "
        f"Overall {overall:.2f}"
    )


if __name__ == "__main__":
    main()
