"""Competition smoke test — validates the DongimonCompetitor against baselines.

Runs a round-robin match between the Dongimon competitor (using all three
custom policies) and baseline competitors using vgc2's built-in GreedyBattle,
RandomSelection, and RandomTeambuild policies.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team


class BaselineCompetitor(Competitor):  # type: ignore[misc]
    """Competitor using only vgc2 built-in policies."""

    def __init__(self, name: str = "Baseline"):
        self._name = name
        self._bp = GreedyBattlePolicy()
        self._sp = RandomSelectionPolicy()
        self._tp = RandomTeamBuildPolicy()

    @property
    def name(self) -> str:
        return self._name

    @property
    def battlepolicy(self) -> Any:
        return self._bp

    @property
    def selectionpolicy(self) -> Any:
        return self._sp

    @property
    def teambuildpolicy(self) -> Any:
        return self._tp


def main() -> None:
    """Run a smoke test match between Dongimon and the baseline."""
    from competitor import DongimonCompetitor

    dongimon = CompetitorManager(DongimonCompetitor())
    baseline = CompetitorManager(BaselineCompetitor())

    print(f"Running match: {dongimon.competitor.name} vs {baseline.competitor.name}")
    match = Match((dongimon, baseline), n_battles=3, gen=gen_team)
    match.run()

    wins = match.wins
    wins_a, wins_b = wins[0], wins[1]
    print(f"\nResults: {dongimon.competitor.name}: {wins_a} wins, {baseline.competitor.name}: {wins_b} wins")

    if wins_a + wins_b > 0:
        print("Match completed successfully.")
    else:
        print("WARNING: No battles completed!")


if __name__ == "__main__":
    main()
