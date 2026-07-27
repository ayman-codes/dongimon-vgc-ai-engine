"""Selection Policy — Team Preview decision-making.

Runs the full pipeline: predict opponent builds from species views,
then score all 4-Pokemon rosters via a JJJ-style damage-ratio matrix
with coverage balance, pre-filter to the top candidates, and simulate
only those via pair-vs-pair sub-tournament for the final ranking.
"""

import itertools
from typing import Any

from vgc2.agent import SelectionPolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine import Team
from vgc2.battle_engine.modifiers import Category, Stat

from src.selection.prediction import predict_opponent_builds
from src.selection.tournament import generate_team_combinations, run_sub_tournament
from src.shared.types import type_effectiveness, vgc2_type_to_name

N_TOP_CANDIDATES = 5

_INDIVIDUAL_WEIGHT = 1.07
_BULK_WEIGHT = 0.42
_BALANCE_WEIGHT = 0.30


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
        """Score a 4-Pokemon roster using JJJ-style damage-ratio estimation.

        For each roster member vs each opponent species, estimates the
        maximum damage ratio (fraction of HP dealt) using the simplified
        damage formula with STAB, type effectiveness, accuracy, and
        priority bonus. Adds a coverage balance term penalizing rosters
        that concentrate damage on few opponents.

        Args:
            roster: Tuple of 4 indices into my_full_team.
            my_full_team: Our full team.
            opp_views: Opponent PokemonView list.

        Returns:
            Float score (higher = better matchup).
        """
        n_opp = len(opp_views)
        if n_opp == 0:
            return 0.0

        damage_per_opp = [0.0] * n_opp
        total_individual = 0.0

        for my_idx in roster:
            member = my_full_team.members[my_idx]
            member_spec = member.species if hasattr(member, "species") else member

            individual_score = 0.0
            for opp_idx, opp_view in enumerate(opp_views):
                ratio = self._max_damage_ratio(member_spec, opp_view)
                individual_score += ratio
                damage_per_opp[opp_idx] += ratio

            total_individual += _INDIVIDUAL_WEIGHT * individual_score
            total_individual += _BULK_WEIGHT * self._defensive_multiplier(member)

        max_dmg = max(damage_per_opp) if damage_per_opp else 0.0
        if max_dmg > 0:
            min_dmg = min(damage_per_opp)
            balance = 1.0 - (max_dmg - min_dmg) / max_dmg
        else:
            balance = 0.0

        return total_individual + _BALANCE_WEIGHT * balance

    @staticmethod
    def _max_damage_ratio(my_species: Any, opp_view: Any) -> float:
        """Estimate max damage ratio of one of our species vs an opponent.

        Uses the simplified damage formula:
        ratio = (42 * effective_BP * Atk / Def) / (50 * HP) * STAB * type_eff

        Priority moves get an effective BP bonus (+12 per priority level).

        Args:
            my_species: Our Pokemon species (or Pokemon with .species).
            opp_view: Opponent PokemonView.

        Returns:
            Maximum damage ratio across all moves (fraction of opp HP).
        """
        spec = my_species.species if hasattr(my_species, "species") else my_species
        opp_spec = opp_view.species if hasattr(opp_view, "species") else opp_view

        opp_types = [vgc2_type_to_name(t.value) for t in opp_spec.types]
        opp_hp = opp_spec.base_stats[Stat.MAX_HP]
        opp_def = opp_spec.base_stats[Stat.DEFENSE]
        opp_spd = opp_spec.base_stats[Stat.SPECIAL_DEFENSE]

        if opp_hp <= 0:
            return 0.0

        phys_cats = (Category.PHYSICAL, Category.PHYSICAL.value)
        spec_cats = (Category.SPECIAL, Category.SPECIAL.value)

        best_ratio = 0.0
        for move in spec.moves:
            if move.base_power <= 0:
                continue
            acc = move.accuracy if move.accuracy is not None else 1.0
            stab = 1.5 if move.pkm_type in spec.types else 1.0
            priority = move.priority if hasattr(move, "priority") else 0
            effective_bp = move.base_power + 12 * priority

            atk_name = vgc2_type_to_name(
                move.pkm_type.value if hasattr(move.pkm_type, "value") else move.pkm_type
            )
            eff = type_effectiveness(atk_name, opp_types)

            if move.category in phys_cats:
                atk_stat = spec.base_stats[Stat.ATTACK]
                def_stat = opp_def
            elif move.category in spec_cats:
                atk_stat = spec.base_stats[Stat.SPECIAL_ATTACK]
                def_stat = opp_spd
            else:
                continue

            if def_stat <= 0:
                continue

            ratio = (42.0 * effective_bp * atk_stat / def_stat) / (50.0 * opp_hp)
            ratio *= acc * stab * eff

            if ratio > best_ratio:
                best_ratio = ratio

        return best_ratio

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
