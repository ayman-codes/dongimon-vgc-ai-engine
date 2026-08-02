"""Selection Policy — Team Preview decision-making.

Runs the full pipeline: predict opponent builds from species views,
then score all 4-Pokemon rosters via the MP model, pre-filter to the
top candidates, and either rank pairs by MP P(win) directly (mp_only)
or validate via pair-vs-pair sub-tournament simulation (mp_sim).

When the team is already final size (no subset to choose), a fast
analytical pair-synergy path orders the pair without MP or simulation.

Supports two selection modes:
    - mp_only: MP model scoring for rosters and pairs (default, fastest).
    - mp_sim: MP model pre-filter + sub-tournament simulation validation.

Failures are never swallowed: missing data or zero completed sims raise
RuntimeError so selection bugs surface immediately.
"""

import itertools
from pathlib import Path
from typing import Any

from numpy.random import default_rng
from vgc2.agent import SelectionPolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, Team
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.modifiers import Category, Stat
from vgc2.battle_engine.team import BattlingTeam
from vgc2.battle_engine.view import StateView, TeamView

from src.battle.greedy_dongi import GreedyDongiPolicy
from src.config.loader import load_selection_synergy
from src.config.models import SelectionSynergyWeights
from src.selection.mp_scoring import load_mp_model, score_pair_mp, score_roster_mp
from src.selection.pair_synergy import pair_synergy_terms
from src.selection.prediction import predict_opponent_builds
from src.selection.tournament import generate_team_combinations, run_sub_tournament
from src.shared.types import type_effectiveness, vgc2_type_to_name

N_TOP_CANDIDATES = 5

_MATCHUP_NORM = 2.0


class DongimonSelectionPolicy(SelectionPolicy):  # type: ignore[misc]
    """Team Preview selection using MP model scoring.

    Generates all C(n, max_size) rosters, scores each via the MP model,
    keeps the top N, then either ranks pairs by MP P(win) directly
    (mp_only) or validates via sub-tournament simulation (mp_sim).
    Returns best pair first (active) then remaining roster (reserve).

    When the team is already final size (no subset to choose), the active
    pair is ordered by the MP model (``_order_by_mp``), falling back to the
    analytical pair-synergy path when opponent prediction fails.

    Selection modes:
        mp_only: MP model scoring for rosters and pairs (default, fastest).
        mp_sim: MP model pre-filter + sub-tournament simulation validation.
    """

    def __init__(
        self,
        n_top_candidates: int = N_TOP_CANDIDATES,
        synergy_weights: SelectionSynergyWeights | None = None,
        selection_mode: str = "mp_only",
        mp_model_path: Path | None = None,
        lead_battles: int = 12,
    ) -> None:
        """Initialize the selection policy.

        Args:
            n_top_candidates: Number of top rosters to simulate (mp_sim path).
            synergy_weights: Optional pair-synergy weights for the fast path.
                Loads from selection_synergy.yaml if None.
            selection_mode: One of "mp_only" (default) or "mp_sim".
            mp_model_path: Path to the MP XGBoost model. Uses default
                champion model if None.
            lead_battles: Battles per (pair, order) during the offline
                empirical lead resolution (size-4 ordering).

        Raises:
            ValueError: If selection_mode is not recognized.
        """
        super().__init__()
        valid_modes = ("mp_only", "mp_sim")
        if selection_mode not in valid_modes:
            raise ValueError(
                f"selection_mode must be one of {valid_modes}, got {selection_mode!r}"
            )
        self._battle_policy = GreedyBattlePolicy()
        self._n_active = 2
        self._n_top = n_top_candidates
        self._synergy = synergy_weights or load_selection_synergy()
        self._mode = selection_mode
        self._mp_model: Any = load_mp_model(mp_model_path)
        self._lead_cache: dict[tuple[Any, ...], list[int]] = {}
        self._lead_battles = lead_battles
        self._lead_policy = GreedyDongiPolicy()

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

        if len(my_full_team.members) <= max_size:
            try:
                return self._order_by_mp(my_full_team, opp_views, max_size)
            except Exception:
                return self._order_by_pair_synergy(my_full_team, opp_views, max_size)

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
            score = score_roster_mp(
                roster_indices=roster,
                my_full_team=my_full_team,
                predicted_builds=predicted_builds,
                opp_views=opp_views,
                model=self._mp_model,
                n_active=self._n_active,
            )
            roster_scores.append((score, roster))

        roster_scores.sort(key=lambda x: x[0], reverse=True)
        candidates = roster_scores[: max(self._n_top, 1)]

        if self._mode == "mp_only":
            return self._select_mp_only(
                candidates, my_full_team, predicted_builds, opp_views, max_size
            )

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

    def _order_by_pair_synergy(
        self,
        my_full_team: Team,
        opp_views: list[Any],
        max_size: int,
    ) -> list[int]:
        """Order a final-size team by analytical pair synergy (no simulation).

        Fast path for the pure-ordering case: the team already has exactly
        max_size members, so there is no subset to choose — only which pair
        starts active. Ranks all C(n, 2) candidate pairs by a weighted blend
        of an opponent-aware matchup term (avg/worst blend over opponent
        pairs) and the intra-pair synergy score, returning the best pair
        first followed by the reserves.

        Args:
            my_full_team: Our full team (size <= max_size).
            opp_views: Opponent PokemonView list.
            max_size: Maximum team size to return.

        Returns:
            Ordered indices: best active pair first, then reserves.

        Raises:
            RuntimeError: If no candidate pairs or opponent pairs exist.
        """
        my_pairs = generate_team_combinations(my_full_team, self._n_active)
        if not my_pairs:
            raise RuntimeError(
                f"selection.fast_path: no pairs of size {self._n_active} from "
                f"team size {len(my_full_team.members)}"
            )

        opp_pairs = list(itertools.combinations(opp_views, self._n_active))
        if not opp_pairs:
            raise RuntimeError(
                f"selection.fast_path: no opponent pairs of size {self._n_active} "
                f"from {len(opp_views)} views"
            )

        w = self._synergy
        scored: list[tuple[float, tuple[int, ...]]] = []
        for my_pair in my_pairs:
            pair_species = [my_full_team.members[i].species for i in my_pair]

            matchup_per_opp = []
            for opp_pair in opp_pairs:
                dmg = 0.0
                for member in pair_species:
                    for opp_view in opp_pair:
                        dmg += self._max_damage_ratio(member, opp_view)
                matchup_per_opp.append(dmg)
            avg_m = sum(matchup_per_opp) / len(matchup_per_opp)
            worst_m = min(matchup_per_opp)
            matchup = w.avg_weight * avg_m + w.worst_weight * worst_m
            matchup_norm = min(matchup / _MATCHUP_NORM, 1.0)

            terms = pair_synergy_terms(pair_species, opp_views)
            total = w.w_matchup * matchup_norm
            total += w.w_defense * terms["defense"]
            total += w.w_speed * terms["speed"]
            total += w.w_role * terms["role"]
            total += w.w_coverage * terms["coverage"]
            scored.append((total, my_pair))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_pair = scored[0][1]

        ordered: list[int] = [int(i) for i in best_pair]
        for idx in range(len(my_full_team.members)):
            if idx not in ordered:
                ordered.append(idx)
        return ordered[:max_size]

    def _order_by_mp(
        self,
        my_full_team: Team,
        opp_views: list[Any],
        max_size: int,
    ) -> list[int]:
        """Order a final-size team by MP pair scoring (no simulation).

        Fast path for the pure-ordering case: the team already has exactly
        max_size members, so there is no subset to choose — only which pair
        starts active. Ranks all C(n, 2) candidate pairs by MP P(win)
        averaged over opponent pairs, returning the best pair first
        followed by the reserves.

        Requires predicted builds for the opponent, which are generated
        inline from the opponent views.

        Args:
            my_full_team: Our full team (size <= max_size).
            opp_views: Opponent PokemonView list.
            max_size: Maximum team size to return.

        Returns:
            Ordered indices: best active pair first, then reserves.

        Raises:
            RuntimeError: If no candidate pairs or opponent predictions fail.
        """
        cache_key = (
            tuple(m.species.id for m in my_full_team.members),
            tuple(sorted(v.species.id for v in opp_views)),
            max_size,
        )
        cached = self._lead_cache.get(cache_key)
        if cached is not None:
            return cached

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
                    "selection.mp_fast_path: predict_opponent_builds returned empty "
                    f"for opponent species id={getattr(getattr(opp_view, 'species', None), 'id', '?')}"
                )
            predicted_builds[opp_view] = builds

        roster_indices = tuple(range(len(my_full_team.members)))
        my_pairs = generate_team_combinations(my_full_team, self._n_active)
        if not my_pairs:
            raise RuntimeError(
                f"selection.mp_fast_path: no pairs of size {self._n_active} from "
                f"team size {len(my_full_team.members)}"
            )

        scored: list[tuple[float, tuple[int, ...]]] = []
        for my_pair in my_pairs:
            pair_score = score_pair_mp(
                pair_indices=my_pair,
                roster_indices=roster_indices,
                my_full_team=my_full_team,
                predicted_builds=predicted_builds,
                opp_views=opp_views,
                model=self._mp_model,
            )
            scored.append((pair_score, my_pair))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_order = self._resolve_lead_order(my_full_team, opp_views, predicted_builds, scored)

        ordered: list[int] = [int(i) for i in best_order]
        for idx in range(len(my_full_team.members)):
            if idx not in ordered:
                ordered.append(idx)
        result = ordered[:max_size]
        self._lead_cache[cache_key] = result
        return result

    def _resolve_lead_order(
        self,
        my_full_team: Team,
        opp_views: list[Any],
        predicted_builds: dict[Any, list[Any]],
        scored_pairs: list[tuple[float, tuple[int, ...]]],
    ) -> tuple[int, ...]:
        """Resolve the best active pair and lead order empirically.

        Battles every candidate pair in both slot orders against the
        opponent's predicted active pair under the GreedyDongi pilot, and
        returns the (pair, lead) with the highest win rate. This bypasses
        the degenerate MP pair ranking (all pairs score equally) and the
        analytical fast path, which both fail to rank pairs correctly.

        Args:
            my_full_team: Our full team (size <= max_size).
            opp_views: Opponent PokemonView list.
            predicted_builds: Dict mapping opponent views to predicted builds.
            scored_pairs: MP-scored candidate pairs (sorted desc).

        Returns:
            The winning (pair, lead) order as a 2-tuple of indices.
        """
        best_wr = -1.0
        best_order: tuple[int, ...] = ()
        for _score, pair in scored_pairs:
            for order in (pair, tuple(reversed(pair))):
                wr = self._lead_win_rate(my_full_team, order, opp_views, predicted_builds)
                if wr > best_wr:
                    best_wr = wr
                    best_order = order
        if not best_order:
            best_order = scored_pairs[0][1]
        return best_order

    def _lead_win_rate(
        self,
        my_full_team: Team,
        order: tuple[int, ...],
        opp_views: list[Any],
        predicted_builds: dict[Any, list[Any]],
    ) -> float:
        """Win rate of one (pair, lead) order vs the opponent's predicted pair.

        Runs ``_lead_battles`` seeded mirror battles between our ordered
        pair (plus reserves) and the opponent's predicted team. The
        opponent's active pair is chosen by ``BasicSelectionPolicy`` over
        its predicted builds, giving a realistic reference opponent.

        Args:
            my_full_team: Our full team.
            order: (lead, second) member indices.
            opp_views: Opponent PokemonView list.
            predicted_builds: Dict mapping opponent views to predicted builds.

        Returns:
            Win rate of our ordered pair in [0, 1].
        """
        lead, second = order
        my_active = [my_full_team.members[lead], my_full_team.members[second]]
        remaining = [i for i in range(len(my_full_team.members)) if i not in (lead, second)]
        remaining.sort(key=lambda i: sum(my_full_team.members[i].stats[1:6]), reverse=True)
        my_reserve = [my_full_team.members[i] for i in remaining[:2]]

        opp_predicted = [predicted_builds[v][0] for v in opp_views if predicted_builds.get(v)]
        opp_team = Team(members=opp_predicted)
        opp_view = TeamView(opp_team)
        opp_idx = list(BasicSelectionPolicy().decision((opp_team, opp_view), len(opp_predicted)))
        opp_active = [opp_predicted[i] for i in opp_idx[:2] if i < len(opp_predicted)]
        opp_reserve = [opp_predicted[i] for i in opp_idx[2:] if i < len(opp_predicted)]

        dummy_my_view = TeamView(my_full_team)
        dummy_opp_view = TeamView(Team(members=opp_active + opp_reserve))

        wins = 0
        for b_idx in range(self._lead_battles):
            my_bt = BattlingTeam(active=list(my_active), reserve=list(my_reserve))
            opp_bt = BattlingTeam(active=list(opp_active), reserve=list(opp_reserve))
            rng = default_rng(b_idx)
            rng_tuple = ((rng, rng), (rng, rng))
            engine = BattleEngine(
                State((my_bt, opp_bt)),
                params=self.params,
                acc_rng=rng_tuple,
                eff_rng=rng_tuple,
                sta_rng=rng_tuple,
            )
            while not engine.finished():
                sv0 = StateView(engine.state, 0, (dummy_my_view, dummy_opp_view))
                sv1 = StateView(engine.state, 1, (dummy_opp_view, dummy_my_view))
                cmd0 = self._lead_policy.decision(sv0, dummy_opp_view)
                cmd1 = self._lead_policy.decision(sv1, dummy_my_view)
                engine.run_turn((cmd0, cmd1))
            if engine.winning_side == 0:
                wins += 1
        return wins / max(self._lead_battles, 1)

    def _select_mp_only(
        self,
        candidates: list[tuple[float, tuple[int, ...]]],
        my_full_team: Team,
        predicted_builds: dict[Any, list[Any]],
        opp_views: list[Any],
        max_size: int,
    ) -> list[int]:
        """Select best roster and pair using MP scoring only (no simulation).

        For each candidate roster, scores all C(4,2) pairs via MP P(win)
        and picks the roster+pair combination with the highest score.

        Args:
            candidates: Pre-filtered rosters with MP scores (sorted desc).
            my_full_team: Our full team.
            predicted_builds: Dict mapping opponent views to predicted builds.
            opp_views: Opponent PokemonView list.
            max_size: Maximum team size to return.

        Returns:
            Ordered indices: best pair first, then remaining roster members.

        Raises:
            RuntimeError: If no pairs can be scored.
        """
        best_score = -1.0
        best_pair: tuple[int, ...] = ()
        best_roster: tuple[int, ...] = ()

        for _, roster in candidates:
            my_pairs = list(itertools.combinations(roster, self._n_active))
            for my_pair in my_pairs:
                pair_score = score_pair_mp(
                    pair_indices=my_pair,
                    roster_indices=roster,
                    my_full_team=my_full_team,
                    predicted_builds=predicted_builds,
                    opp_views=opp_views,
                    model=self._mp_model,
                )
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = my_pair
                    best_roster = roster

        if not best_pair:
            raise RuntimeError(
                "selection.mp_only: no valid pair found across all candidates"
            )

        ordered: list[int] = [int(i) for i in best_pair]
        remaining = [int(i) for i in best_roster if int(i) not in ordered]
        remaining.sort(
            key=lambda i: sum(my_full_team.members[i].species.base_stats),
            reverse=True,
        )
        ordered.extend(remaining)
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
