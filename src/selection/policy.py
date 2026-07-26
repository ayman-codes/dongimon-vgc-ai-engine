"""Selection Policy — Team Preview decision-making.

Runs the full pipeline: predict opponent builds from species views,
then score all 4-Pokemon rosters via a JJJ-style damage matrix,
pre-filter to the top candidates, and simulate only those via
pair-vs-pair sub-tournament for the final ranking.
"""

import itertools
from typing import Any

from vgc2.agent import SelectionPolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import Team
from vgc2.battle_engine.modifiers import Stat

from src.selection.prediction import predict_opponent_builds
from src.selection.tournament import generate_team_combinations, run_sub_tournament
from src.shared.types import type_effectiveness, vgc2_type_to_name
from src.teambuild.builds import species_power

N_TOP_CANDIDATES = 5


class DongimonSelectionPolicy(SelectionPolicy):  # type: ignore[misc]
    """Team Preview selection using JJJ-style matrix pre-filter + simulation.

    Generates all C(6,4) = 15 possible 4-Pokemon rosters, scores each
    via a damage-proxy matrix (fast), keeps the top N, evaluates those
    with full pair-vs-pair sub-tournament simulations, and returns
    the empirically strongest roster.
    """

    def __init__(self, n_top_candidates: int = N_TOP_CANDIDATES) -> None:
        super().__init__()
        self._battle_policy = GreedyBattlePolicy()
        self._n_active = 2
        self._n_top = n_top_candidates

    def decision(self, teams: tuple[Team, Team], max_size: int) -> list[int]:
        """Select the best 4-Pokemon roster from our team.

        Args:
            teams: Tuple of (my_team, opponent_team_view).
            max_size: Maximum team size to select (typically 4).

        Returns:
            List of indices into our team members to bring to battle.
        """
        my_full_team, opp_team_view = teams
        opp_views = opp_team_view.members

        predicted_builds = {}
        for opp_view in opp_views:
            builds = predict_opponent_builds(
                pokemon_view=opp_view,
                my_full_team=my_full_team,
                all_opp_views=opp_views,
                params=self.params,
            )
            predicted_builds[opp_view] = builds

        all_rosters = list(generate_team_combinations(my_full_team, max_size))

        if not all_rosters:
            return list(range(min(max_size, len(my_full_team.members))))

        roster_scores = []
        for roster in all_rosters:
            score = self._score_roster_matrix(roster, my_full_team, opp_views)
            roster_scores.append((score, roster))

        roster_scores.sort(key=lambda x: -x[0], reverse=True)
        candidates = roster_scores[:max(self._n_top, 1)]

        my_pairs = generate_team_combinations(my_full_team, self._n_active)
        opp_pairs = list(itertools.combinations(opp_views, self._n_active))

        roster_results: dict[tuple[int, ...], float] = {}
        for _, roster in candidates:
            pair_win_rates = []
            for _, my_pair in enumerate(my_pairs):
                if not set(my_pair).issubset(set(roster)):
                    continue
                total_wr = 0.0
                for opp_pair in opp_pairs:
                    wr = run_sub_tournament(
                        my_full_team=my_full_team,
                        my_pair_indices=my_pair,
                        opp_view_pair=opp_pair,
                        predicted_builds_dict=predicted_builds,
                        battle_policy=self._battle_policy,
                        params=self.params,
                    )
                    total_wr += wr
                avg_wr = total_wr / len(opp_pairs) if opp_pairs else 0.0
                pair_win_rates.append((avg_wr, my_pair))
            if pair_win_rates:
                pair_win_rates.sort(key=lambda x: -x[0])
                roster_results[tuple(roster)] = pair_win_rates[0][0]
            else:
                roster_results[tuple(roster)] = 0.0

        if not roster_results:
            return list(range(min(max_size, len(my_full_team.members))))

        best_roster = max(roster_results, key=lambda r: roster_results[r])
        return list(best_roster)

    def _score_roster_matrix(
        self,
        roster: tuple[int, ...],
        my_full_team: Team,
        opp_views: list[Any],
    ) -> float:
        """Score a 4-Pokemon roster using a JJJ-style damage proxy matrix.

        For each roster member vs each opponent species, estimates a
        damage-proxy (species_power × best type effectiveness) and
        adds a defensive bulk bonus.

        Args:
            roster: Tuple of 4 indices into my_full_team.
            my_full_team: Our full team.
            opp_views: Opponent PokemonView list.

        Returns:
            Float score (higher = better matchup).
        """
        total = 0.0
        for my_idx in roster:
            member = my_full_team.members[my_idx]
            member_spec = member.species if hasattr(member, 'species') else member
            power = species_power(member_spec)

            for opp_view in opp_views:
                opp_species = opp_view.species
                opp_type_names = [vgc2_type_to_name(t.value) for t in opp_species.types]
                best_eff = 1.0
                for move in member_spec.moves:
                    if move.base_power <= 0:
                        continue
                    atk_type = move.pkm_type.value if hasattr(move.pkm_type, 'value') else move.pkm_type
                    atk_name = vgc2_type_to_name(atk_type)
                    eff = type_effectiveness(atk_name, opp_type_names)
                    if eff > best_eff:
                        best_eff = eff
                total += power * best_eff

            total += self._defensive_multiplier(member) * 0.42

        return total

    @staticmethod
    def _defensive_multiplier(pkm: Any) -> float:
        """Compute a normalised bulk estimate for a Pokemon.

        Ratios are relative to typical level-50 maximums (402 HP, 257
        defences) as in JJJ's selection policy.

        Args:
            pkm: A Pokemon member from the team.

        Returns:
            Float bulk multiplier (higher = bulkier).
        """
        if hasattr(pkm, 'stats'):
            hp = pkm.stats[Stat.MAX_HP]
            df = pkm.stats[Stat.DEFENSE]
            spd = pkm.stats[Stat.SPECIAL_DEFENSE]
        elif hasattr(pkm, 'base_stats'):
            hp = pkm.base_stats[Stat.MAX_HP]
            df = pkm.base_stats[Stat.DEFENSE]
            spd = pkm.base_stats[Stat.SPECIAL_DEFENSE]
        else:
            return 1.0
        return float(hp / 402.0) * float(df / 257.0) * float(spd / 257.0)
