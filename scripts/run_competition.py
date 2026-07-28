"""Full round-robin competition benchmark with ELO rating.

Runs Dongimon (all three custom policies) against JJJ, minimon, Caaaden,
and a Greedy baseline in a round-robin tournament. Reports per-matchup
win rates and final ELO standings.
"""

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

N_BATTLES = 10
ELO_K = 32.0
INITIAL_ELO = 1200.0


class BaselineCompetitor(Competitor):  # type: ignore[misc]
    """Competitor using only vgc2 built-in policies."""

    def __init__(self, name: str = "Greedy"):
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


def _elo_update(elo_a: float, elo_b: float, score_a: float) -> tuple[float, float]:
    """Compute ELO update for a match result.

    Args:
        elo_a: Current ELO of player A.
        elo_b: Current ELO of player B.
        score_a: Actual score for A (1.0 = win, 0.0 = loss, 0.5 = draw).

    Returns:
        Tuple of (new_elo_a, new_elo_b).
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = elo_a + ELO_K * (score_a - expected_a)
    new_b = elo_b + ELO_K * (score_b - expected_b)
    return new_a, new_b


def main() -> None:
    """Run a full round-robin competition and report ELO standings."""
    from competitor import DongimonCompetitor
    from competitors.competitor1_jjj import JJJ_Competitor
    from competitors.competitor2_minimon import minimon
    from competitors.competitor_caaaden import CaaadenCompetitor

    roster: list[CompetitorManager] = [
        CompetitorManager(DongimonCompetitor()),
        CompetitorManager(JJJ_Competitor()),
        CompetitorManager(minimon()),
        CompetitorManager(CaaadenCompetitor()),
        CompetitorManager(BaselineCompetitor()),
    ]

    names = [cm.competitor.name for cm in roster]
    n = len(roster)
    elos = {name: INITIAL_ELO for name in names}
    results: dict[str, dict[str, str]] = {name: {} for name in names}

    print(f"Round-Robin Competition: {N_BATTLES} battles per matchup")
    print(f"Participants: {', '.join(names)}")
    print("=" * 60)

    total_matchups = n * (n - 1) // 2
    matchup_idx = 0
    start = time.perf_counter()

    for i in range(n):
        for j in range(i + 1, n):
            matchup_idx += 1
            name_a, name_b = names[i], names[j]
            print(f"\n[{matchup_idx}/{total_matchups}] {name_a} vs {name_b} ...", end=" ", flush=True)

            match = Match((roster[i], roster[j]), n_battles=N_BATTLES, gen=gen_team)
            match.run()

            wins_a, wins_b = match.wins[0], match.wins[1]
            total = wins_a + wins_b
            if total == 0:
                print("SKIPPED (no battles completed)")
                continue

            score_a = wins_a / total
            new_elo_a, new_elo_b = _elo_update(elos[name_a], elos[name_b], score_a)
            elos[name_a] = new_elo_a
            elos[name_b] = new_elo_b

            results[name_a][name_b] = f"{wins_a}-{wins_b}"
            results[name_b][name_a] = f"{wins_b}-{wins_a}"

            print(f"{wins_a}-{wins_b} (ELO: {new_elo_a:.0f} / {new_elo_b:.0f})")

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"FINAL ELO STANDINGS ({elapsed:.1f}s total)")
    print("=" * 60)
    sorted_names = sorted(names, key=lambda x: elos[x], reverse=True)
    for rank, name in enumerate(sorted_names, 1):
        print(f"  {rank}. {name:<20} ELO {elos[name]:.1f}")

    print("\nHEAD-TO-HEAD RESULTS")
    print("-" * 60)
    for name in sorted_names:
        record = results[name]
        if record:
            matchups_str = ", ".join(f"vs {opp}: {score}" for opp, score in record.items())
            print(f"  {name}: {matchups_str}")


if __name__ == "__main__":
    main()
