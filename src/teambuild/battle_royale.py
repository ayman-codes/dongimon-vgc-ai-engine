"""Battle royale simulation for the Team Build Policy.

Takes the top K teams from the evolutionary algorithm, hydrates them
into vgc2 Team objects, and runs a round-robin battle tournament using
GreedyBattlePolicy to determine the empirically strongest team.
"""

import time
from typing import Any

from numpy.random import default_rng
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView


def run_battle_royale(
    top_teams: list[list[int]],
    builds_cache: dict[Any, Any],
    pool_species: list[Any],
    roster: list[Any],
    n_battles: int,
    max_time_sec: float,
    params: BattleRuleParam,
) -> list[int]:
    """Run a round-robin battle tournament between the top teams.

    Hydrates each team into hydrated Pokemon objects, creates vgc2
    teams, and runs best-of-N battles between every pair using
    GreedyBattlePolicy. Returns the team with the highest aggregate
    win rate.

    Args:
        top_teams: List of teams (each a list of species indices).
        builds_cache: Dict mapping species -> Pokemon build.
        pool_species: Full species list from roster.
        roster: Full roster for index resolution.
        n_battles: Number of battles per matchup.
        max_time_sec: Maximum wall-clock time for the entire tournament.
        params: Battle rule parameters from the environment.

    Returns:
        The single best team as a list of species indices.
        Falls back to the first team if time runs out.
    """
    start_time = time.perf_counter()
    hydrated = []
    for team in top_teams:
        species_team = [pool_species[i] for i in team]
        team_pkm = []
        for species in species_team:
            build = builds_cache.get(species)
            if build is not None:
                team_pkm.append(build)
        if team_pkm:
            hydrated.append((team, team_pkm))

    if not hydrated:
        return top_teams[0] if top_teams else []

    battle_rng = default_rng()
    policy = GreedyBattlePolicy()
    n_teams = len(hydrated)
    wins = [0] * n_teams
    losses = [0] * n_teams

    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            if time.perf_counter() - start_time > max_time_sec:
                break

            _, team_a_pkm = hydrated[i]
            _, team_b_pkm = hydrated[j]

            team_a = Team(members=team_a_pkm)
            team_b = Team(members=team_b_pkm)

            for _ in range(n_battles):
                battle_teams = get_battle_teams(team_a, team_b, 2, 2)
                state = State(battle_teams)
                rng_tuple = ((battle_rng, battle_rng), (battle_rng, battle_rng))
                engine = BattleEngine(
                    state, acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple
                )

                view_a = TeamView(team_a)
                view_b = TeamView(team_b)

                while not engine.finished():
                    sv0 = StateView(engine.state, 0, (view_a, view_b))
                    sv1 = StateView(engine.state, 1, (view_b, view_a))
                    cmd0 = policy.decision(sv0)
                    cmd1 = policy.decision(sv1)
                    engine.run_turn((cmd0, cmd1))

                if engine.winning_side == 0:
                    wins[i] += 1
                    losses[j] += 1
                elif engine.winning_side == 1:
                    wins[j] += 1
                    losses[i] += 1

    if all(v == 0 for v in wins):
        return top_teams[0]

    def _win_rate(idx: int) -> float:
        w = wins[idx]
        loss = losses[idx]
        return w / max(w + loss, 1)

    best_idx = max(range(n_teams), key=_win_rate)
    return hydrated[best_idx][0]
