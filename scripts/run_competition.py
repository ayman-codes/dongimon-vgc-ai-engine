"""Championship-track competition benchmark.

Mirrors the vgc-ai championship flow:
  1. Generate a shared species roster + move set (fixed seed).
  2. Each competitor builds a team via their own TeamBuildPolicy.
  3. Match runs selection + battle on pre-built teams (no random gen).
  4. ELO updated per matchup.

Usage:
    uv run python scripts/run_competition.py --epochs=10 --n-battles=3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.balance.meta import BasicMeta
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.ecosystem import build_team, label_roster, sanitized_team_build_decision
from vgc2.competition.match import Match
from vgc2.util.generator import gen_move_set, gen_pkm_roster

N_MOVES = 100
ROSTER_SIZE = 50
MAX_TEAM_SIZE = 4
MAX_PKM_MOVES = 4
N_ACTIVE = 2
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
    """Run a championship-track competition and report ELO standings."""
    parser = argparse.ArgumentParser(description="Championship-track competition benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for roster generation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of championship epochs")
    parser.add_argument("--n-battles", type=int, default=3, help="Battles per matchup")
    args = parser.parse_args()

    import numpy as np

    from competitor import DongimonCompetitor
    from competitors.competitor1_jjj import JJJ_Competitor
    from competitors.competitor2_minimon import minimon
    from competitors.competitor_caaaden import CaaadenCompetitor

    # Generate shared roster (mirrors championship: gen_move_set + gen_pkm_roster)
    rng = np.random.default_rng(args.seed)
    move_set = gen_move_set(N_MOVES, rng)
    roster = gen_pkm_roster(ROSTER_SIZE, move_set, MAX_PKM_MOVES, rng)
    label_roster(move_set, roster)
    meta = BasicMeta(move_set, roster)

    # Register competitors
    cms: list[CompetitorManager] = [
        CompetitorManager(DongimonCompetitor()),
        CompetitorManager(JJJ_Competitor()),
        CompetitorManager(minimon()),
        CompetitorManager(CaaadenCompetitor()),
        CompetitorManager(BaselineCompetitor()),
    ]

    names = [cm.competitor.name for cm in cms]
    n = len(cms)
    elos: dict[str, float] = dict.fromkeys(names, INITIAL_ELO)
    results: dict[str, dict[str, str]] = {name: {} for name in names}

    print("=" * 64)
    print("Championship-Track Competition")
    print(f"  seed={args.seed}, epochs={args.epochs}, n_battles={args.n_battles}")
    print(f"  roster: {ROSTER_SIZE} species, {N_MOVES} moves")
    print(f"  max_team_size={MAX_TEAM_SIZE}, n_active={N_ACTIVE}")
    print(f"  participants: {', '.join(names)}")
    print("=" * 64)

    start = time.perf_counter()

    for epoch in range(args.epochs):
        build_times: dict[str, float] = {}
        for cm in cms:
            t_start = time.perf_counter()
            try:
                cmd = sanitized_team_build_decision(
                    cm.competitor.teambuildpolicy, roster, meta,
                    MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE,
                )
                cm.team = build_team(cmd, roster)
            except Exception as exc:
                print(f"  [WARN] {cm.competitor.name} teambuild failed: {exc}")
                cm.team = None
            build_times[cm.competitor.name] = time.perf_counter() - t_start

        for i in range(n):
            for j in range(i + 1, n):
                name_a, name_b = names[i], names[j]

                if cms[i].team is None or cms[j].team is None:
                    # Award win to the side that built successfully
                    if cms[i].team is not None:
                        elos[name_a] += ELO_K / 2
                    elif cms[j].team is not None:
                        elos[name_b] += ELO_K / 2
                    continue

                # Run both side orders to eliminate engine side-0 bias
                match_fwd = Match(
                    (cms[i], cms[j]),
                    n_active=N_ACTIVE,
                    n_battles=args.n_battles,
                    max_team_size=MAX_TEAM_SIZE,
                    max_pkm_moves=MAX_PKM_MOVES,
                    meta=meta,
                )
                match_fwd.run()

                match_rev = Match(
                    (cms[j], cms[i]),
                    n_active=N_ACTIVE,
                    n_battles=args.n_battles,
                    max_team_size=MAX_TEAM_SIZE,
                    max_pkm_moves=MAX_PKM_MOVES,
                    meta=meta,
                )
                match_rev.run()

                # Combine: fwd has cms[i] as side 0; rev has cms[i] as side 1
                wins_a = match_fwd.wins[0] + match_rev.wins[1]
                wins_b = match_fwd.wins[1] + match_rev.wins[0]
                total = wins_a + wins_b
                if total == 0:
                    continue

                score_a = wins_a / total
                new_a, new_b = _elo_update(elos[name_a], elos[name_b], score_a)
                elos[name_a] = new_a
                elos[name_b] = new_b

                results[name_a][name_b] = f"{wins_a}-{wins_b}"
                results[name_b][name_a] = f"{wins_b}-{wins_a}"

        elapsed = time.perf_counter() - start
        top = sorted(elos.items(), key=lambda x: -x[1])
        top_str = " | ".join(f"{nm}: {r:.0f}" for nm, r in top)
        times_str = " | ".join(f"{nm}: {t:.2f}s" for nm, t in build_times.items())
        print(f"  Epoch {epoch + 1:2d}/{args.epochs}: {top_str}  ({elapsed:.1f}s)")
        print(f"    teambuild: {times_str}")

    # Final standings
    elapsed = time.perf_counter() - start
    sorted_names = sorted(names, key=lambda x: elos[x], reverse=True)

    print("\n" + "=" * 64)
    print(f"FINAL ELO STANDINGS ({elapsed:.1f}s total)")
    print("=" * 64)
    for rank, name in enumerate(sorted_names, 1):
        marker = "  <-- Dongimon" if name == "Dongimon" else ""
        print(f"  {rank}. {name:<20} ELO {elos[name]:.1f}{marker}")

    print("\nHEAD-TO-HEAD (last epoch)")
    print("-" * 64)
    for name in sorted_names:
        record = results[name]
        if record:
            matchups_str = ", ".join(f"vs {opp}: {score}" for opp, score in record.items())
            print(f"  {name}: {matchups_str}")


if __name__ == "__main__":
    main()
