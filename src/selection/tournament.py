"""Sub-tournament simulation for the Selection Policy.

Runs pair-vs-pair battle simulations inside the vgc2 BattleEngine
to evaluate which team composition performs best against predicted
opponent builds.
"""

import itertools
from typing import Any

from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.team import BattlingTeam, Team
from vgc2.battle_engine.view import PokemonView, StateView, TeamView


def generate_team_combinations(team: Team, combination_size: int) -> list[tuple[int, ...]]:
    """Generate all unique combinations of Pokemon indices from a team.

    Args:
        team: The source team.
        combination_size: Size of each combination to generate.

    Returns:
        List of index tuples. Empty if combination_size exceeds team size.
    """
    if len(team.members) < combination_size:
        return []
    member_indices = list(range(len(team.members)))
    return list(itertools.combinations(member_indices, combination_size))


def run_sub_tournament(
    my_full_team: Team,
    my_pair_indices: tuple[int, ...],
    opp_view_pair: tuple[PokemonView, ...],
    predicted_builds_dict: dict,
    battle_policy: Any,
    params: BattleRuleParam,
) -> float:
    """Run a sub-tournament for one of our pairs against all predicted opponent build combinations.

    Simulates battles between our pair and each opponent build combination
    using the vgc2 BattleEngine with the provided battle policy.

    Args:
        my_full_team: Our full team.
        my_pair_indices: Indices of our two Pokemon for this matchup.
        opp_view_pair: Tuple of two opponent PokemonView objects.
        predicted_builds_dict: Dict mapping PokemonView to list of predicted Pokemon builds.
        battle_policy: BattlePolicy instance used for both sides.
        params: Battle rule parameters.

    Returns:
        Average win rate for our pair across all matchups (0.0–1.0).
    """
    opp_build_list_a = predicted_builds_dict.get(opp_view_pair[0], [])
    opp_build_list_b = predicted_builds_dict.get(opp_view_pair[1], [])

    if not opp_build_list_a or not opp_build_list_b:
        return 0.0

    sub_wins = 0
    sub_battles = 0

    build_matchups = itertools.product(opp_build_list_a, opp_build_list_b)

    for opp_build_a, opp_build_b in build_matchups:
        my_pair_pkm = [my_full_team.members[i] for i in my_pair_indices]
        my_battling_team = BattlingTeam(active=my_pair_pkm, reserve=[])

        opp_predicted_pair = [opp_build_a, opp_build_b]
        opp_battling_team = BattlingTeam(active=opp_predicted_pair, reserve=[])

        initial_state = State((my_battling_team, opp_battling_team))
        engine = BattleEngine(initial_state)

        dummy_my_view = TeamView(my_full_team)
        dummy_opp_view = TeamView(Team(members=opp_predicted_pair))

        while not engine.finished():
            state_view_p0 = StateView(engine.state, 0, (dummy_my_view, dummy_opp_view))
            state_view_p1 = StateView(engine.state, 1, (dummy_opp_view, dummy_my_view))

            cmd_p0 = battle_policy.decision(state_view_p0)
            cmd_p1 = battle_policy.decision(state_view_p1)

            engine.run_turn((cmd_p0, cmd_p1))

        if engine.winning_side == 0:
            sub_wins += 1
        sub_battles += 1

    return sub_wins / sub_battles if sub_battles > 0 else 0.0
