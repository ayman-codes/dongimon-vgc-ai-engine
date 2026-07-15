"""Selection Policy — Team Preview decision-making.

Runs the full pipeline: predict opponent builds from species views,
then evaluate all possible pair combinations via sub-tournament
simulation, then rank and select the best 4-Pokemon roster.
"""

import itertools

from vgc2.agent import SelectionPolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import Team

from src.selection.prediction import predict_opponent_builds
from src.selection.tournament import generate_team_combinations, run_sub_tournament


class DongimonSelectionPolicy(SelectionPolicy):
    """Team Preview selection using pair-vs-pair sub-tournaments.

    For each possible pair of our Pokemon, runs simulated battles
    against each possible opponent pair using predicted opponent
    builds. Returns the top-ranked 4-Pokemon roster.
    """

    def __init__(self):
        super().__init__()
        self._battle_policy = GreedyBattlePolicy()
        self._n_active = 2

    def decision(self, teams: tuple[Team, Team], max_size: int) -> list[int]:
        """Select the best 4-Pokemon roster from our team.

        Args:
            teams: Tuple of (my_team, opponent_team_view).
            max_size: Maximum team size to select (typically 4).

        Returns:
            List of indices into our team members to bring to battle.
        """
        my_full_team, opp_team_view = teams
        n_active = self._n_active

        predicted_builds = {}
        all_opp_views = opp_team_view.members

        for opp_view in all_opp_views:
            builds = predict_opponent_builds(
                pokemon_view=opp_view,
                my_full_team=my_full_team,
                all_opp_views=all_opp_views,
                params=self.params,
            )
            predicted_builds[opp_view] = builds

        my_pairs = generate_team_combinations(my_full_team, n_active)
        opp_pairs = list(itertools.combinations(all_opp_views, n_active))

        if not my_pairs:
            return list(range(min(max_size, len(my_full_team.members))))

        results = dict.fromkeys(my_pairs, 0.0)

        for my_pair in my_pairs:
            total_win_rate = 0.0
            for opp_pair in opp_pairs:
                win_rate = run_sub_tournament(
                    my_full_team=my_full_team,
                    my_pair_indices=my_pair,
                    opp_view_pair=opp_pair,
                    predicted_builds_dict=predicted_builds,
                    battle_policy=self._battle_policy,
                    params=self.params,
                )
                total_win_rate += win_rate

            if opp_pairs:
                results[my_pair] = total_win_rate / len(opp_pairs)

        ranked_pairs = sorted(results.keys(), key=lambda p: results[p], reverse=True)
        num_pairs = max_size // n_active

        if len(ranked_pairs) < num_pairs:
            final_selection = []
            for pair in ranked_pairs:
                final_selection.extend(list(pair))
            remaining = [i for i in range(len(my_full_team.members)) if i not in final_selection]
            final_selection.extend(remaining)
            return final_selection[:max_size]

        final_selection = []
        for i in range(num_pairs):
            pair = ranked_pairs[i]
            final_selection.extend(list(pair))

        return final_selection
