"""Selection Policy — Team Preview decision-making.

Runs the full pipeline: predict opponent builds from species views,
then score all 4-Pokemon rosters via a damage-ratio matrix with coverage
balance, pre-filter to the top candidates, and simulate only those via
pair-vs-pair sub-tournament for the final ranking.

Failures are never swallowed: missing data or zero completed sims raise
RuntimeError so selection bugs surface immediately.
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
    """Team Preview selection using matrix pre-filter + simulation.

    Generates all C(n, max_size) rosters, scores each via a damage-proxy
    matrix, keeps the top N, evaluates pairs with sub-tournament sims, and
    returns best pair first (active) then remaining roster (reserve).

    Pair ranking: win rate, then pair damage proxy, then pair BST.
    """

    def __init__(self, n_top_candidates: int = N_TOP_CANDIDATES) -> None:
        super().__init__()
        self._battle_policy = GreedyBattlePolicy()
        self._n_active = 2
        self._n_top = n_top_candidates

    def decision(self, teams: tuple[Team, Team], max_size: int) -> list[int]:
        """Select the best roster ordered as best active pair then reserves.

        Args:
            teams: Tuple of (my_team, opponent_team_view).
            max_size: Maximum team size to select (typically 4).

        Returns:
            Ordered indices: best pair (active) first, then remaining roster
            members as reserve.

        Raises:
            RuntimeError: On empty team, empty opponent views, failed
                prediction, empty pair scores, or all-zero win rates.
        """
        my_full_team, opp_team_view = teams
        opp_views = list(opp_team_view.members)

        if len(my_full_team.members) == 0:
            raise RuntimeError("selection.decision: own team has zero members")
        if len(opp_views) == 0:
            raise RuntimeError("selection.decision: opponent team view has zero members")

        predicted_builds: dict[Any, list[Any]] = {}
        for opp_view in opp_views:
            builds = predict_opponent_builds(
                pokemon_view=opp_view,
                my_full_team=my_full_team,
                all_opp_views=opp_views,
                params=self.params,
            )
            if not builds:
                raise RuntimeError(
                    "selection.decision: predict_opponent_builds returned empty "
                    f"list for opponent species id={getattr(getattr(opp_view, 'species', None), 'id', '?')}"
                )
            if not any(getattr(b, "moves", None) for b in builds):
                raise RuntimeError(
                    "selection.decision: all predicted builds lack moves for an opponent mon"
                )
            predicted_builds[opp_view] = builds

        all_rosters = list(generate_team_combinations(my_full_team, max_size))
        if not all_rosters:
            raise RuntimeError(
                f"selection.decision: no rosters of size {max_size} from "
                f"team size {len(my_full_team.members)}"
            )

        roster_scores: list[tuple[float, tuple[int, ...]]] = []
        for roster in all_rosters:
            score = self._score_roster_matrix(roster, my_full_team, opp_views)
            roster_scores.append((score, roster))

        roster_scores.sort(key=lambda x: x[0], reverse=True)
        candidates = roster_scores[: max(self._n_top, 1)]

        my_pairs = generate_team_combinations(my_full_team, self._n_active)
        if not my_pairs:
            raise RuntimeError(
                f"selection.decision: no pairs of size {self._n_active} from "
                f"team size {len(my_full_team.members)}"
            )

        opp_pairs = list(itertools.combinations(opp_views, self._n_active))
        if not opp_pairs:
            raise RuntimeError(
                f"selection.decision: no opponent pairs of size {self._n_active} "
                f"from {len(opp_views)} views"
            )

        roster_results: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {}
        for _, roster in candidates:
            pair_records: list[tuple[float, float, float, tuple[int, ...]]] = []
            for my_pair in my_pairs:
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
                avg_wr = total_wr / len(opp_pairs)
                pair_dmg = self._pair_damage_score(my_pair, my_full_team, opp_views)
                pair_bst = self._pair_bst(my_pair, my_full_team)
                pair_records.append((avg_wr, pair_dmg, pair_bst, my_pair))

            if not pair_records:
                raise RuntimeError(
                    f"selection.decision: no eligible pairs inside roster {roster}"
                )

            pair_records.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
            best_wr, _best_dmg, _best_bst, best_pair = pair_records[0]
            roster_results[tuple(roster)] = (best_wr, best_pair)

        if not roster_results:
            raise RuntimeError("selection.decision: roster_results empty after candidate loop")

        all_wrs = [wr for wr, _ in roster_results.values()]
        if all(wr == 0.0 for wr in all_wrs):
            raise RuntimeError(
                "selection.decision: all pair win rates are 0.0 — sub-tournament "
                "produced no wins (check prediction/sim). pair WRs="
                f"{[(r, roster_results[r][0]) for r in roster_results]}"
            )

        best_roster = max(
            roster_results,
            key=lambda r: (
                roster_results[r][0],
                self._pair_damage_score(roster_results[r][1], my_full_team, opp_views),
                self._pair_bst(roster_results[r][1], my_full_team),
            ),
        )
        _best_wr, best_pair = roster_results[best_roster]
        ordered: list[int] = [int(i) for i in best_pair]
        for idx in best_roster:
            if int(idx) not in ordered:
                ordered.append(int(idx))
        if len(ordered) < min(max_size, len(my_full_team.members)):
            raise RuntimeError(
                f"selection.decision: ordered selection length {len(ordered)} "
                f"< expected {min(max_size, len(my_full_team.members))}"
            )
        return ordered[:max_size]

    def _pair_damage_score(
        self,
        pair: tuple[int, ...],
        my_full_team: Team,
        opp_views: list[Any],
    ) -> float:
        """Sum max damage-ratio proxies for each mon in the pair vs all opponents.

        Args:
            pair: Indices of the pair on our team.
            my_full_team: Our full team.
            opp_views: Opponent Pokemon views.

        Returns:
            Aggregate damage proxy (higher = stronger offense vs the field).
        """
        total = 0.0
        for idx in pair:
            member = my_full_team.members[int(idx)]
            member_spec = member.species if hasattr(member, "species") else member
            for opp_view in opp_views:
                total += self._max_damage_ratio(member_spec, opp_view)
        return total

    @staticmethod
    def _pair_bst(pair: tuple[int, ...], my_full_team: Team) -> float:
        """Sum base-stat totals for the pair.

        Args:
            pair: Indices of the pair on our team.
            my_full_team: Our full team.

        Returns:
            Sum of BSTs for the pair members.
        """
        total = 0.0
        for idx in pair:
            member = my_full_team.members[int(idx)]
            base = member.species.base_stats
            total += float(sum(base))
        return total

    def _score_roster_matrix(
        self,
        roster: tuple[int, ...],
        my_full_team: Team,
        opp_views: list[Any],
    ) -> float:
        """Score a roster using damage-ratio estimation.

        Args:
            roster: Tuple of indices into my_full_team.
            my_full_team: Our full team.
            opp_views: Opponent PokemonView list.

        Returns:
            Float score (higher = better matchup).
        """
        n_opp = len(opp_views)
        if n_opp == 0:
            raise RuntimeError("_score_roster_matrix: empty opp_views")

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

        Args:
            pkm: A Pokemon member from the team.

        Returns:
            Float bulk multiplier (higher = bulkier).
        """
        if hasattr(pkm, "stats"):
            hp = pkm.stats[Stat.MAX_HP]
            df = pkm.stats[Stat.DEFENSE]
            spd = pkm.stats[Stat.SPECIAL_DEFENSE]
        elif hasattr(pkm, "base_stats"):
            hp = pkm.base_stats[Stat.MAX_HP]
            df = pkm.base_stats[Stat.DEFENSE]
            spd = pkm.base_stats[Stat.SPECIAL_DEFENSE]
        else:
            return 1.0
        return float(hp / 402.0) * float(df / 257.0) * float(spd / 257.0)
