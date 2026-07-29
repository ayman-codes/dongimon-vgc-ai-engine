"""Sub-tournament simulation for the Selection Policy.

Runs pair-vs-pair battle simulations inside the vgc2 BattleEngine
to evaluate which team composition performs best against predicted
opponent builds.
"""

import itertools
from typing import Any

from numpy.random import default_rng
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.modifiers import Stat
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
    predicted_builds_dict: dict[Any, list[Any]],
    battle_policy: Any,
    params: BattleRuleParam,
) -> float:
    """Run a sub-tournament for one of our pairs against predicted opponent builds.

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

    Raises:
        RuntimeError: If predicted builds are missing, members lack moves, or
            zero battles complete successfully.
    """
    opp_build_list_a = predicted_builds_dict.get(opp_view_pair[0], [])
    opp_build_list_b = predicted_builds_dict.get(opp_view_pair[1], [])

    if not opp_build_list_a:
        raise RuntimeError(
            "run_sub_tournament: empty predicted builds for opponent active slot 0"
        )
    if not opp_build_list_b:
        raise RuntimeError(
            "run_sub_tournament: empty predicted builds for opponent active slot 1"
        )

    sub_wins = 0
    sub_battles = 0

    build_matchups = itertools.product(opp_build_list_a, opp_build_list_b)

    for opp_build_a, opp_build_b in build_matchups:
        if not opp_build_a.moves or not opp_build_b.moves:
            raise RuntimeError(
                "run_sub_tournament: predicted opponent build has empty moveset"
            )

        my_pair_pkm = [my_full_team.members[i] for i in my_pair_indices]
        if not all(p.moves for p in my_pair_pkm):
            raise RuntimeError(
                f"run_sub_tournament: our pair {my_pair_indices} has a member with no moves"
            )

        remaining = [
            (i, my_full_team.members[i])
            for i in range(len(my_full_team.members))
            if i not in my_pair_indices
        ]
        remaining.sort(key=lambda x: sum(x[1].stats[1:6]), reverse=True)
        my_reserve = [pkm for _, pkm in remaining[:2]]
        my_battling_team = BattlingTeam(active=my_pair_pkm, reserve=my_reserve)

        opp_predicted_pair = [opp_build_a, opp_build_b]
        opp_reserve = _pick_opp_reserve(predicted_builds_dict, opp_view_pair)
        opp_battling_team = BattlingTeam(active=opp_predicted_pair, reserve=opp_reserve)

        initial_state = State((my_battling_team, opp_battling_team))
        rng = default_rng(sub_battles)
        rng_tuple = ((rng, rng), (rng, rng))
        engine = BattleEngine(
            initial_state,
            params=params,
            acc_rng=rng_tuple,
            eff_rng=rng_tuple,
            sta_rng=rng_tuple,
        )

        dummy_my_view = TeamView(my_full_team)
        dummy_opp_team = Team(members=opp_predicted_pair + opp_reserve)
        dummy_opp_view = TeamView(dummy_opp_team)

        while not engine.finished():
            state_view_p0 = StateView(engine.state, 0, (dummy_my_view, dummy_opp_view))
            state_view_p1 = StateView(engine.state, 1, (dummy_opp_view, dummy_my_view))

            cmd_p0 = battle_policy.decision(state_view_p0, dummy_opp_view)
            cmd_p1 = battle_policy.decision(state_view_p1, dummy_my_view)

            engine.run_turn((cmd_p0, cmd_p1))

        if engine.winning_side == 0:
            sub_wins += 1
        sub_battles += 1

    if sub_battles == 0:
        raise RuntimeError(
            f"run_sub_tournament: zero battles completed for pair {my_pair_indices}"
        )

    return sub_wins / sub_battles


def _pick_opp_reserve(
    predicted_builds_dict: dict[Any, list[Any]],
    opp_view_pair: tuple[PokemonView, ...],
) -> list[Any]:
    """Select up to 2 reserve Pokemon for the opponent from non-active views.

    Picks the best predicted build from each opponent view not in the
    active pair, ranked by bulk (sum of defensive stats).

    Args:
        predicted_builds_dict: Dict mapping PokemonView to predicted builds.
        opp_view_pair: The two active opponent views (excluded from reserve).

    Returns:
        List of up to 2 Pokemon for the opponent reserve.
    """
    active_set = {id(v) for v in opp_view_pair}
    candidates = []
    for view, builds in predicted_builds_dict.items():
        if id(view) in active_set or not builds:
            continue
        best = builds[0]
        if best.moves:
            candidates.append(best)

    candidates.sort(
        key=lambda p: p.stats[Stat.MAX_HP] + p.stats[Stat.DEFENSE] + p.stats[Stat.SPECIAL_DEFENSE],
        reverse=True,
    )
    return candidates[:2]
