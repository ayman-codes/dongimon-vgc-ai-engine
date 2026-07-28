"""Move scoring functions for turn-by-turn battle decisions.

Each function evaluates a single action (offensive move, protect, or switch)
and returns a numeric score representing its tactical value.
"""

from typing import Any

from vgc2.battle_engine import BattleRuleParam, calculate_damage
from vgc2.battle_engine.damage_calculator import (
    calculate_burn_damage,
    calculate_poison_damage,
    calculate_sand_damage,
    calculate_stealth_rock_damage,
)
from vgc2.battle_engine.modifiers import Category, Hazard, Stat, Status, Terrain, Type, Weather
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.view import BattlingPokemonView, StateView

from src.config.constants import (
    BASE_STAT_DIVISOR,
    BOOST_VALUE_PER_STAGE,
    COMBINED_KO_BONUS,
    DAMAGE_SCORE_TO_PCT_MULT,
    DEFAULT_ACCURACY,
    DEFAULT_AVG_MOVE_SCORE,
    DEFENSIVE_BOOST_STATS,
    DEFENSIVE_STAT_BOOST_VALUE,
    DOUBLE_RESIST_SCORE,
    DOUBLE_WEAKNESS_SCORE,
    FALLBACK_THREAT_ACC,
    FALLBACK_THREAT_BP,
    FF_COMBINED_BIGGEST_EXTRA,
    FF_HEAVY_DAMAGE_BIGGEST_EXTRA,
    FF_HEAVY_DAMAGE_BOTH,
    FF_SINGLE_ALLY_CONTRIB,
    FF_SINGLE_BIGGEST_EXTRA,
    FF_SINGLE_NO_CONTRIB,
    HAZARD_REDUNDANCY_PENALTY,
    IMMUNE_RESIST_SCORE,
    KO_BONUS,
    LOW_DAMAGE_THRESHOLD,
    OFF_THREAT_DOUBLE_SCORE,
    OFF_THREAT_QUAD_SCORE,
    PP_PENALTY_MULT,
    PP_PENALTY_RATIO,
    PP_PENALTY_THRESHOLD,
    SCREEN_DAMAGE_RATE,
    SCREEN_REDUNDANCY_PENALTY,
    SINGLE_RESIST_SCORE,
    SINGLE_WEAKNESS_SCORE,
    SPEED_BOOST_DIVISOR,
    STAB_MULTIPLIER,
    STAT_STAGE_MAX,
    STAT_STAGE_MIN,
    STRONG_OFFENSIVE_TYPE_THRESHOLD,
    SUBSTANTIAL_DAMAGE_RATIO,
    SWITCH_BASELINE_MULT,
    SWITCH_DEF_PIVOT_THRESHOLD,
    SWITCH_DUAL_BONUS,
    SWITCH_GOOD_BOTH_THRESHOLD,
    SWITCH_OFFENSIVE_MULT,
    SWITCH_SYNERGY_MULT,
    THREAT_PROXY_SCALE,
    THREAT_TOP_MOVES,
    TYPE_EFF_HALF_RESIST,
    TYPE_EFF_IMMUNE,
    TYPE_EFF_MATRIX,
    TYPE_EFF_QUAD_RESIST,
    TYPE_EFF_QUAD_SUPER,
    TYPE_EFF_SUPER,
    TYPE_MATCHUP_SCALE,
)
from src.shared.move_utils import is_status_move


def score_offensive_move(
    attacker: BattlingPokemonView,
    move: Any,
    target_pokemon: BattlingPokemonView,
    state: StateView,
    params: BattleRuleParam,
    status_weights: dict[int, float] | None = None,
) -> tuple[float, bool]:
    """Score a single offensive move against a target.

    Computes base damage value, KO bonus (future damage prevented),
    and accumulated utility from status, stat boosts, field effects,
    screens, hazard removal, and healing.

    Args:
        attacker: The attacking Pokemon view.
        move: The BattlingMove being evaluated.
        target_pokemon: The target Pokemon view.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Tuple of (score, is_ko_potential) where is_ko_potential indicates
        whether this move would KO the target.
    """
    action_score = 0.0
    is_ko_potential = False

    move_const = move.constants

    if move.pp <= 0 or move.disabled:
        return -float("inf"), False

    base_power = float(move_const.base_power)
    is_status_category = is_status_move(move_const.category)

    if not is_status_category and base_power > 0:
        actual_damage_dealt = calculate_damage(
            params=params,
            attacking_side=0,
            move=move_const,
            state=state,
            attacker=attacker,
            defender=target_pokemon,
        )

        move_acc = move_const.accuracy if move_const.accuracy is not None else DEFAULT_ACCURACY

        if target_pokemon.hp > 0:
            damage_for_scoring = min(actual_damage_dealt, target_pokemon.hp)
            percent_hp_dealt = damage_for_scoring / target_pokemon.hp
            action_score += percent_hp_dealt * DAMAGE_SCORE_TO_PCT_MULT * move_acc

        is_ko_potential = actual_damage_dealt >= target_pokemon.hp

        if is_ko_potential:
            action_score += KO_BONUS * move_acc

    utility_score = 0.0

    status_utility = _status_value(attacker, target_pokemon, move_const, state, params, status_weights)
    utility_score += status_utility

    stat_boost_utility = _stat_boost_value(attacker, move_const, state, params)
    utility_score += stat_boost_utility

    is_primary_utility = (
        is_status_category
        or move_const.weather_start != Weather.CLEAR
        or move_const.field_start != Terrain.NONE
        or move_const.toggle_trickroom
        or move_const.hazard != Hazard.NONE
    )

    if is_primary_utility:
        my_active = [p for p in state.sides[0].team.active if p and p.hp > 0]
        my_reserve = [p for p in state.sides[0].team.reserve if p and p.hp > 0]
        opp_actives = state.sides[1].team.active if state.sides[1].team.active else []

        field_utility = _field_setup_move(attacker, move_const, my_active, my_reserve, opp_actives, state, params)
        utility_score += field_utility

    action_score += utility_score

    screen_utility = _screen_value(
        move_const,
        [p for p in state.sides[0].team.active if p and p.hp > 0],
        [p for p in state.sides[1].team.active if p and p.hp > 0],
        state,
        params,
    )
    utility_score += screen_utility

    hazard_removal_names = {"rapid spin", "defog", "mortal spin", "tidy up"}
    if move_const.name.lower() in hazard_removal_names:
        hazard_utility = _hazard_removal_value(state, params)
        utility_score += hazard_utility

    healing_utility = _healing_value(attacker, move_const, state, params)
    utility_score += healing_utility

    if (
        not is_ko_potential
        and move_const.max_pp > 0
        and (move.pp / move_const.max_pp) < PP_PENALTY_RATIO
        and action_score < PP_PENALTY_THRESHOLD
    ):
        action_score *= PP_PENALTY_MULT

    pivot_names = {"u-turn", "volt switch", "flip turn", "parting shot"}
    if move_const.name.lower() in pivot_names:
        reserve_list = [p for p in state.sides[0].team.reserve if p and p.hp > 0]
        if reserve_list:
            best_switch_val = max(
                score_switch_action(attacker, reserve_pkm, [target_pokemon], state, params)
                for reserve_pkm in reserve_list
            )
            action_score += best_switch_val

    return action_score, is_ko_potential


def score_protect_move(
    protector: BattlingPokemonView,
    move: Any,
    state: StateView,
    params: BattleRuleParam,
) -> float:
    """Score using Protect for a Pokemon.

    Values the move by the amount of incoming damage it blocks,
    plus net passive damage (poison, burn, sand) that accrues
    while the protector is immune.

    Args:
        protector: The Pokemon using Protect.
        move: The BattlingMove (must be Protect).
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Score float. Returns -inf if Protect has no PP or is disabled.
    """
    from vgc2.battle_engine.threshold_calculator import protect_modifier

    if move.pp <= 0 or move.disabled:
        return -float("inf")

    estimated_damage_blocked, _ = estimate_incoming_threat(protector, state.sides[1].team.active, state, params)
    protect_score = estimated_damage_blocked

    net_passive = 0.0
    for opp_pkm in state.sides[1].team.active:
        if not opp_pkm or opp_pkm.hp <= 0:
            continue
        if opp_pkm.status in (Status.POISON, Status.TOXIC):
            net_passive += calculate_poison_damage(params, opp_pkm)
        elif opp_pkm.status == Status.BURN:
            net_passive += calculate_burn_damage(params, opp_pkm)
        if state.weather == Weather.SAND:
            net_passive += calculate_sand_damage(params, opp_pkm)

    for ally_pkm in state.sides[0].team.active:
        if not ally_pkm or ally_pkm.hp <= 0 or ally_pkm is protector:
            continue
        if ally_pkm.status in (Status.POISON, Status.TOXIC):
            net_passive -= calculate_poison_damage(params, ally_pkm)
        elif ally_pkm.status == Status.BURN:
            net_passive -= calculate_burn_damage(params, ally_pkm)
        if state.weather == Weather.SAND:
            net_passive -= calculate_sand_damage(params, ally_pkm)

    protect_score += net_passive

    protect_accuracy = move.constants.accuracy if move.constants.accuracy is not None else DEFAULT_ACCURACY
    protect_reliability = protect_accuracy * protect_modifier(params, move.constants, protector)

    return protect_score * protect_reliability  # type: ignore[no-any-return]


def score_switch_action(
    current_pkm: BattlingPokemonView,
    reserve_pkm: BattlingPokemonView,
    opponent_actives: list[Any],
    state: StateView,
    params: BattleRuleParam,
    avg_move_score: float = DEFAULT_AVG_MOVE_SCORE,
) -> float:
    """Score switching the current Pokemon for a reserve.

    Evaluates defensive pivot value (type resistance to opponent STABs),
    offensive threat value (type advantage against opponents), hazard
    damage on entry, and compound bonuses for dual-purpose pivots.

    Args:
        current_pkm: The currently active Pokemon being switched out.
        reserve_pkm: The reserve Pokemon being switched in.
        opponent_actives: List of opponent active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.
        avg_move_score: Average move score for the active Pokemon,
            used to calibrate the switch baseline against current
            move quality.

    Returns:
        Score float. Returns -inf if the reserve has fainted.
    """
    if reserve_pkm.hp <= 0:
        return -float("inf")

    switch_score = avg_move_score * SWITCH_BASELINE_MULT
    num_opponents = len(opponent_actives)
    if num_opponents == 0:
        return switch_score

    total_resistance_score = 0.0
    is_strong_defensive_pivot = False

    for opp_pkm in opponent_actives:
        if not opp_pkm or opp_pkm.hp <= 0:
            continue

        resistance = 0.0
        best_resist = 1.0

        for opp_stab_type in opp_pkm.types:
            eff = _get_type_eff_string(opp_stab_type, reserve_pkm.types)
            best_resist = min(best_resist, eff)

            if eff == TYPE_EFF_IMMUNE:
                resistance += IMMUNE_RESIST_SCORE * TYPE_MATCHUP_SCALE
            elif eff <= TYPE_EFF_QUAD_RESIST:
                resistance += DOUBLE_RESIST_SCORE * TYPE_MATCHUP_SCALE
            elif eff <= TYPE_EFF_HALF_RESIST:
                resistance += SINGLE_RESIST_SCORE * TYPE_MATCHUP_SCALE
            elif eff >= TYPE_EFF_QUAD_SUPER:
                resistance -= DOUBLE_WEAKNESS_SCORE * TYPE_MATCHUP_SCALE
            elif eff >= TYPE_EFF_SUPER:
                resistance -= SINGLE_WEAKNESS_SCORE * TYPE_MATCHUP_SCALE

        total_resistance_score += resistance / num_opponents
        if best_resist <= SWITCH_DEF_PIVOT_THRESHOLD:
            is_strong_defensive_pivot = True

    switch_score += total_resistance_score

    total_offensive_score = 0.0
    is_strong_offensive_threat = False

    if reserve_pkm.constants and reserve_pkm.constants.moves:
        for opp_pkm in opponent_actives:
            if not opp_pkm or opp_pkm.hp <= 0:
                continue

            max_eff = 0.0
            off_contrib = 0

            for incoming_move in reserve_pkm.constants.moves:
                if incoming_move.base_power == 0 and incoming_move.category != Category.OTHER:
                    continue

                eff = _get_type_eff_string(incoming_move.pkm_type, opp_pkm.types)
                max_eff = max(max_eff, eff)

                if eff >= TYPE_EFF_QUAD_SUPER:
                    off_contrib = max(off_contrib, OFF_THREAT_QUAD_SCORE)  # type: ignore[assignment]
                elif eff >= TYPE_EFF_SUPER:
                    off_contrib = max(off_contrib, OFF_THREAT_DOUBLE_SCORE)  # type: ignore[assignment]

            total_offensive_score += off_contrib / num_opponents
            if max_eff >= STRONG_OFFENSIVE_TYPE_THRESHOLD:
                is_strong_offensive_threat = True

    switch_score += total_offensive_score * SWITCH_OFFENSIVE_MULT

    if is_strong_defensive_pivot and is_strong_offensive_threat:
        switch_score += SWITCH_DUAL_BONUS

    if total_resistance_score > SWITCH_GOOD_BOTH_THRESHOLD and total_offensive_score > SWITCH_GOOD_BOTH_THRESHOLD:
        switch_score *= SWITCH_SYNERGY_MULT

    if state.sides[0].conditions.stealth_rock:
        sr_damage = calculate_stealth_rock_damage(params, reserve_pkm)
        switch_score -= sr_damage

    if state.sides[0].conditions.poison_spikes:
        is_immune = any(t in (Type.POISON, Type.STEEL, Type.FLYING) for t in reserve_pkm.types)
        if not is_immune and reserve_pkm.status == Status.NONE:
            poison_dmg = calculate_poison_damage(params, reserve_pkm)
            switch_score -= poison_dmg

    return switch_score


def estimate_incoming_threat(
    my_pokemon: BattlingPokemonView,
    opponent_actives: list[Any],
    state: StateView,
    params: BattleRuleParam,
    locked_moves: dict[int, str] | None = None,
) -> tuple[float, bool]:
    """Estimate maximum likely incoming damage to a Pokemon this turn.

    Evaluates both revealed moves and potential unrevealed moves
    from the opponent's species movepool. Returns the sum of the
    best damage from each opponent, and whether a KO is likely.

    When an opponent slot has a locked move (detected by choice lock
    tracking), only that move is evaluated for that slot.

    Args:
        my_pokemon: The defending Pokemon view.
        opponent_actives: List of opponent active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.
        locked_moves: Dict mapping opponent slot index to their
            Choice-locked move name. Only that move is evaluated.

    Returns:
        Tuple of (max_total_damage, is_likely_ko). max_total_damage is
        the sum of the highest damaging move from each opponent.
    """
    if not my_pokemon or my_pokemon.hp <= 0:
        return 0.0, False

    max_total = 0.0
    any_single_can_ko = False

    for opp_idx, opp_pkm in enumerate(opponent_actives):
        if not opp_pkm or opp_pkm.hp <= 0:
            continue

        lock_name = locked_moves.get(opp_idx) if locked_moves else None
        highest_damage = 0.0

        if lock_name:
            if opp_pkm.battling_moves:
                for opp_move in opp_pkm.battling_moves:
                    if opp_move.constants.name != lock_name:
                        continue
                    if (
                        opp_move.constants.category == Category.OTHER
                        or opp_move.constants.base_power == 0
                        or opp_move.pp <= 0
                        or opp_move.disabled
                    ):
                        continue
                    dmg = calculate_damage(
                        params=params,
                        attacking_side=1,
                        move=opp_move.constants,
                        state=state,
                        attacker=opp_pkm,
                        defender=my_pokemon,
                    )
                    highest_damage = max(highest_damage, dmg)
            if highest_damage == 0.0:
                max_total += 0.0
                continue
        else:
            if opp_pkm.battling_moves:
                for opp_move in opp_pkm.battling_moves:
                    if (
                        opp_move.constants.category == Category.OTHER
                        or opp_move.constants.base_power == 0
                        or opp_move.pp <= 0
                        or opp_move.disabled
                    ):
                        continue

                    dmg = calculate_damage(
                        params=params,
                        attacking_side=1,
                        move=opp_move.constants,
                        state=state,
                        attacker=opp_pkm,
                        defender=my_pokemon,
                    )
                    if dmg > highest_damage:
                        highest_damage = dmg

        if not lock_name and (
            opp_pkm.constants
            and hasattr(opp_pkm.constants, "species")
            and opp_pkm.constants.species
            and hasattr(opp_pkm.constants.species, "moves")
            and opp_pkm.constants.species.moves
        ):
            candidates = []
            for pot_move in opp_pkm.constants.species.moves:
                if pot_move.category == Category.OTHER or pot_move.base_power == 0:
                    continue

                bp = float(pot_move.base_power)
                stab = STAB_MULTIPLIER if pot_move.pkm_type in opp_pkm.types else 1.0
                eff = _get_type_eff_string(pot_move.pkm_type, my_pokemon.types)
                if stab < STAB_MULTIPLIER and eff < TYPE_EFF_SUPER:
                    continue
                proxy = bp * (eff * THREAT_PROXY_SCALE) * stab * pot_move.accuracy
                candidates.append({"move": pot_move, "score": proxy})

            candidates.sort(key=lambda x: x["score"], reverse=True)
            top_moves = [c["move"] for c in candidates[:THREAT_TOP_MOVES]]

            for pot_move in top_moves:
                dmg = calculate_damage(
                    params=params,
                    attacking_side=1,
                    move=pot_move,
                    state=state,
                    attacker=opp_pkm,
                    defender=my_pokemon,
                )
                if dmg > highest_damage:
                    highest_damage = dmg

        max_total += highest_damage
        if highest_damage >= my_pokemon.hp:
            any_single_can_ko = True

    priority_damage = _priority_threat_delta(my_pokemon, opponent_actives, state, params)
    max_total += priority_damage

    is_likely_ko = (max_total >= my_pokemon.hp) or any_single_can_ko
    return max_total, is_likely_ko


def _priority_threat_delta(
    my_pokemon: BattlingPokemonView,
    opponent_actives: list[Any],
    state: StateView,
    params: BattleRuleParam,
) -> float:
    """Additional threat from opponent priority moves that out-speed our own.

    If an opponent has a priority move (priority > 0) and our Pokemon
    does not have any priority move that matches or beats it, the
    opponent's priority move will always go first. Adds that damage
    to the total threat since it cannot be played around by speed.

    Args:
        my_pokemon: The defending Pokemon view.
        opponent_actives: List of opponent active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Additional damage that must be accounted for due to priority.
    """
    our_has_priority = any(
        mv.constants.priority > 0 for mv in (my_pokemon.battling_moves or []) if mv and mv.pp > 0 and not mv.disabled
    )

    max_priority_dmg = 0.0
    for opp_pkm in opponent_actives:
        if not opp_pkm or opp_pkm.hp <= 0:
            continue
        for opp_move in opp_pkm.battling_moves or []:
            if (
                not opp_move
                or opp_move.constants.priority <= 0
                or opp_move.constants.category == Category.OTHER
                or opp_move.constants.base_power == 0
                or opp_move.pp <= 0
                or opp_move.disabled
            ):
                continue
            dmg = calculate_damage(
                params=params,
                attacking_side=1,
                move=opp_move.constants,
                state=state,
                attacker=opp_pkm,
                defender=my_pokemon,
            )
            if dmg > max_priority_dmg:
                max_priority_dmg = dmg

    if max_priority_dmg > 0 and not our_has_priority:
        return max_priority_dmg
    return 0.0


def identify_biggest_threat(
    opponent_actives: list[Any],
    my_team_actives: list[Any],
    state: StateView,
    params: BattleRuleParam,
) -> tuple[Any, ...] | None:
    """Identify the single most threatening opponent active Pokemon.

    Combines offensive potential against our team, presence of
    offensive stat boosts (setup sweeper), and high base
    offensive stats.

    Args:
        opponent_actives: List of opponent active Pokemon views.
        my_team_actives: List of our active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Tuple of (BattlingPokemonView, slot_index) for the biggest threat,
        or None if no threats exist.
    """
    if not opponent_actives or not my_team_actives:
        return None

    best_score = -1.0
    biggest_pkm = None
    biggest_idx = -1

    for opp_idx, opp_pkm in enumerate(opponent_actives):
        if not opp_pkm or opp_pkm.hp <= 0:
            continue

        current_score = 0.0

        max_dmg_to_team = 0.0
        for my_pkm in my_team_actives:
            if my_pkm and my_pkm.hp > 0:
                temp_highest = 0.0
                if opp_pkm.battling_moves:
                    for opp_move in opp_pkm.battling_moves:
                        if (
                            opp_move.constants.category != Category.OTHER
                            and opp_move.pp > 0
                            and not opp_move.disabled
                            and opp_move.constants.pkm_type in opp_pkm.types
                        ):
                            eff = _get_type_eff_string(opp_move.constants.pkm_type, my_pkm.types)
                            acc = (
                                opp_move.constants.accuracy
                                if opp_move.constants.accuracy is not None
                                else DEFAULT_ACCURACY
                            )
                            temp_highest = max(temp_highest, opp_move.constants.base_power * eff * acc)

                if temp_highest < LOW_DAMAGE_THRESHOLD and opp_pkm.constants and opp_pkm.constants.species:
                    for p_type in opp_pkm.types:
                        eff = _get_type_eff_string(p_type, my_pkm.types)
                        temp_highest = max(temp_highest, FALLBACK_THREAT_BP * eff * FALLBACK_THREAT_ACC)

                max_dmg_to_team = max(max_dmg_to_team, temp_highest)

        current_score += max_dmg_to_team

        offensive_boost = 0.0
        if opp_pkm.boosts:
            if opp_pkm.boosts[Stat.ATTACK] > 0:
                offensive_boost += opp_pkm.boosts[Stat.ATTACK]
            if opp_pkm.boosts[Stat.SPECIAL_ATTACK] > 0:
                offensive_boost += opp_pkm.boosts[Stat.SPECIAL_ATTACK]
            if opp_pkm.boosts[Stat.SPEED] > 0:
                offensive_boost += opp_pkm.boosts[Stat.SPEED] / SPEED_BOOST_DIVISOR

        current_score += offensive_boost * BOOST_VALUE_PER_STAGE

        if opp_pkm.constants and opp_pkm.constants.species:
            base_atk = opp_pkm.constants.species.base_stats[Stat.ATTACK]
            base_spa = opp_pkm.constants.species.base_stats[Stat.SPECIAL_ATTACK]
            base_spe = opp_pkm.constants.species.base_stats[Stat.SPEED]
            current_score += (base_atk + base_spa + base_spe) / BASE_STAT_DIVISOR

        if current_score > best_score:
            best_score = current_score
            biggest_pkm = opp_pkm
            biggest_idx = opp_idx

    if biggest_pkm:
        return (biggest_pkm, biggest_idx)
    return None


def calculate_focus_fire_bonus(
    pkm_a_view: BattlingPokemonView,
    move_a_const: Move,
    is_pkm_a_koing: bool,
    pkm_b_view: BattlingPokemonView,
    move_b_const: Move,
    is_pkm_b_koing: bool,
    target_pkm_view: BattlingPokemonView,
    state: StateView,
    params: BattleRuleParam,
    biggest_threat_on_field: Any,
) -> float:
    """Calculate the focus fire bonus for both Pokemon targeting the same opponent.

    Awards bonus proportional to the combined KO potential, individual KO
    potential, or substantial damage contribution against the same target.
    Extra bonus applies when the target is the identified biggest threat.

    Args:
        pkm_a_view: First allied Pokemon view.
        move_a_const: First allied Pokemon's move constants.
        is_pkm_a_koing: Whether move_A independently KOs the target.
        pkm_b_view: Second allied Pokemon view.
        move_b_const: Second allied Pokemon's move constants.
        is_pkm_b_koing: Whether move_B independently KOs the target.
        target_pkm_view: The target opponent Pokemon view.
        state: Current battle state view.
        params: Battle rule parameters.
        biggest_threat_on_field: The biggest threat opponent view (or None).

    Returns:
        Focus fire bonus score.
    """
    if not target_pkm_view or target_pkm_view.hp <= 0:
        return 0.0

    initial_hp = target_pkm_view.hp
    combined_ko = False

    dmg_a = calculate_damage(
        params=params,
        attacking_side=0,
        move=move_a_const,
        state=state,
        attacker=pkm_a_view,
        defender=target_pkm_view,
    )
    dmg_b = calculate_damage(
        params=params,
        attacking_side=0,
        move=move_b_const,
        state=state,
        attacker=pkm_b_view,
        defender=target_pkm_view,
    )
    total_focus_dmg = dmg_a + dmg_b

    if total_focus_dmg >= initial_hp:
        combined_ko = True

    acc_a = move_a_const.accuracy if move_a_const.accuracy is not None else DEFAULT_ACCURACY
    acc_b = move_b_const.accuracy if move_b_const.accuracy is not None else DEFAULT_ACCURACY
    reliability = acc_a * acc_b

    if combined_ko:
        bonus = COMBINED_KO_BONUS * reliability
        if target_pkm_view is biggest_threat_on_field:
            bonus += FF_COMBINED_BIGGEST_EXTRA * reliability
        return bonus

    if is_pkm_a_koing or is_pkm_b_koing:
        temp = 0.0
        if is_pkm_a_koing and dmg_b > 0:
            temp = FF_SINGLE_ALLY_CONTRIB * acc_a
        elif is_pkm_b_koing and dmg_a > 0:
            temp = FF_SINGLE_ALLY_CONTRIB * acc_b
        elif is_pkm_a_koing:
            temp = FF_SINGLE_NO_CONTRIB * acc_a
        elif is_pkm_b_koing:
            temp = FF_SINGLE_NO_CONTRIB * acc_b
        if target_pkm_view is biggest_threat_on_field and temp > 0:
            temp += FF_SINGLE_BIGGEST_EXTRA * reliability
        return temp

    if (
        initial_hp > 0
        and (dmg_a / initial_hp > SUBSTANTIAL_DAMAGE_RATIO)
        and (dmg_b / initial_hp > SUBSTANTIAL_DAMAGE_RATIO)
    ):
        bonus = FF_HEAVY_DAMAGE_BOTH * reliability
        if target_pkm_view is biggest_threat_on_field:
            bonus += FF_HEAVY_DAMAGE_BIGGEST_EXTRA * reliability
        return bonus

    return 0.0


def _get_type_eff_string(move_type: Any, defender_types: list[Any]) -> float:
    """Get type effectiveness via pre-computed matrix lookup.

    Indexes into TYPE_EFF_MATRIX directly using Type enum integer values,
    avoiding string conversion and dict lookups.

    Args:
        move_type: vgc2 Type enum member.
        defender_types: List of vgc2 Type enums from the defender.

    Returns:
        Combined effectiveness multiplier.
    """
    atk_idx = move_type.value if hasattr(move_type, "value") else int(move_type)
    eff = 1.0
    for dt in defender_types:
        def_idx = dt.value if hasattr(dt, "value") else int(dt)
        eff *= TYPE_EFF_MATRIX[atk_idx][def_idx]
    return eff


def _status_value(
    user: BattlingPokemonView,
    target: BattlingPokemonView,
    move: Move,
    state: StateView,
    params: BattleRuleParam,
    status_weights: dict[int, float] | None = None,
) -> float:
    """Calculate the value of inflicting a status condition using configurable weights.

    Each status type has a weight (damage-equivalent points) stored in config.
    The score is ``weight * effect_prob`` where effect_prob is the move's
    probability of applying the condition.

    Args:
        user: The Pokemon using the status move.
        target: The target Pokemon.
        move: The Move constants being evaluated.
        state: Current battle state view (unused in weight-based version).
        params: Battle rule parameters (unused in weight-based version).
        status_weights: Dict mapping Status enum integer values to float weights.

    Returns:
        Float value of the status condition.
    """
    if move.status == Status.NONE or target.status != Status.NONE:
        return 0.0

    if status_weights is None:
        return 0.0

    weight = status_weights.get(move.status.value, 0.0)
    if weight <= 0.0:
        return 0.0

    prob = getattr(move, "status_chance", 1.0)
    if prob <= 0.0:
        prob = 1.0

    return weight * prob


def _stat_boost_value(
    user: BattlingPokemonView,
    move: Move,
    state: StateView,
    params: BattleRuleParam,
) -> float:
    """Calculate the value of a stat-boosting or debuffing move.

    Computes the net damage increase from self-targeting boosts,
    or net damage mitigation from opponent-targeting debuffs.

    Args:
        user: The Pokemon using the move.
        move: The Move constants being evaluated.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Float value of the stat changes.
    """
    from vgc2.battle_engine.pokemon import BattlingPokemon

    if not move.boosts or not any(b != 0 for b in move.boosts):
        return 0.0

    boost_value = 0.0
    opp_actives = [p for p in state.sides[1].team.active if p and p.hp > 0]
    if not opp_actives:
        return 0.0

    stat_map = [
        None,
        Stat.ATTACK,
        Stat.DEFENSE,
        Stat.SPECIAL_ATTACK,
        Stat.SPECIAL_DEFENSE,
        Stat.SPEED,
        Stat.ACCURACY,
        Stat.EVASION,
    ]

    if move.self_boosts:
        total_before = 0.0
        total_after = 0.0

        best_moves = {}
        for opp_idx, opp_pkm in enumerate(opp_actives):
            best_dmg = -1
            best_move = None
            for user_move in user.battling_moves:
                if user_move.constants.category in (Category.PHYSICAL, Category.SPECIAL) and user_move.pp > 0:
                    dmg = calculate_damage(params, 0, user_move.constants, state, user, opp_pkm)
                    if dmg > best_dmg:
                        best_dmg = dmg
                        best_move = user_move.constants
            if best_move:
                best_moves[opp_idx] = {"move": best_move, "damage": best_dmg}

        if not best_moves:
            return 0.0

        for opp_idx, info in best_moves.items():
            dmg_before = info["damage"]
            total_before += dmg_before

            temp_attacker = BattlingPokemon(user.constants)
            temp_attacker.boosts = list(user.boosts)
            for i, change in enumerate(move.boosts):
                if 0 < i < len(temp_attacker.boosts):
                    temp_attacker.boosts[i] = max(STAT_STAGE_MIN, min(STAT_STAGE_MAX, temp_attacker.boosts[i] + change))

            dmg_after = calculate_damage(params, 0, info["move"], state, temp_attacker, opp_actives[opp_idx])
            total_after += dmg_after

        boost_value = total_after - total_before

    else:
        total_mitigated = 0.0
        for opp_pkm in opp_actives:
            dmg_before, _ = estimate_incoming_threat(user, [opp_pkm], state, params)

            temp_defender = BattlingPokemon(opp_pkm.constants)
            temp_defender.boosts = list(opp_pkm.boosts)
            for i, change in enumerate(move.boosts):
                if 0 < i < len(temp_defender.boosts):
                    temp_defender.boosts[i] = max(STAT_STAGE_MIN, min(STAT_STAGE_MAX, temp_defender.boosts[i] + change))

            dmg_after, _ = estimate_incoming_threat(user, [temp_defender], state, params)
            total_mitigated += dmg_before - dmg_after

        boost_value = total_mitigated

    if boost_value == 0:
        for i, change in enumerate(move.boosts):
            if change == 0 or i >= len(stat_map):
                continue
            stat_affected = stat_map[i]
            if stat_affected in DEFENSIVE_BOOST_STATS:
                boost_value += abs(change) * DEFENSIVE_STAT_BOOST_VALUE

    return boost_value


def _screen_value(
    move: Move,
    my_team_actives: list[Any],
    opponent_actives: list[Any],
    state: StateView,
    params: BattleRuleParam,
) -> float:
    """Calculate the value of setting Reflect or Light Screen.

    Sums the mitigated damage (50%) that screens provide against
    relevant incoming attacks for the active team.

    Args:
        move: The Move constants being evaluated.
        my_team_actives: Our active Pokemon views.
        opponent_actives: Opponent active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Float value of the screen. Returns -100 if the screen is already active.
    """
    if not (move.toggle_reflect or move.toggle_lightscreen):
        return 0.0

    is_reflect = move.toggle_reflect
    screen_cat = Category.PHYSICAL if is_reflect else Category.SPECIAL

    if (is_reflect and state.sides[0].conditions.reflect) or (not is_reflect and state.sides[0].conditions.lightscreen):
        return SCREEN_REDUNDANCY_PENALTY

    total_mitigated = 0.0
    for my_pkm in my_team_actives:
        if not my_pkm or my_pkm.hp <= 0:
            continue

        threat_to_pkm = 0.0
        for opp_pkm in opponent_actives:
            if not opp_pkm or opp_pkm.hp <= 0:
                continue

            best_opp_dmg = 0.0
            for opp_move in opp_pkm.constants.species.moves:
                if opp_move.category == screen_cat:
                    dmg = calculate_damage(params, 1, opp_move, state, opp_pkm, my_pkm)
                    if dmg > best_opp_dmg:
                        best_opp_dmg = dmg

            threat_to_pkm += best_opp_dmg

        total_mitigated += threat_to_pkm * SCREEN_DAMAGE_RATE

    return total_mitigated


def _hazard_removal_value(state: StateView, params: BattleRuleParam) -> float:
    """Calculate the value of removing hazards from our side.

    Sums the damage that reserve Pokemon would take from
    Stealth Rock and poison spikes.

    Args:
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Float value of hazard removal.
    """
    my_conds = state.sides[0].conditions
    if not my_conds.stealth_rock and not my_conds.poison_spikes:
        return 0.0

    total_avoided = 0.0
    my_reserve = [p for p in state.sides[0].team.reserve if p and p.hp > 0]

    for pkm in my_reserve:
        if my_conds.stealth_rock:
            total_avoided += calculate_stealth_rock_damage(params, pkm)
        if my_conds.poison_spikes:
            is_immune = any(t in (Type.POISON, Type.STEEL, Type.FLYING) for t in pkm.types)
            if not is_immune and pkm.status == Status.NONE:
                total_avoided += calculate_poison_damage(params, pkm)

    return total_avoided


def _healing_value(
    user: BattlingPokemonView,
    move: Move,
    state: StateView,
    params: BattleRuleParam,
) -> float:
    """Calculate the value of a healing move.

    If healing allows the user to survive an otherwise-lethal attack,
    the value is the damage the user can deal next turn. Otherwise,
    the value is the raw HP restored.

    Args:
        user: The Pokemon using the healing move.
        move: The Move constants being evaluated.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Float value of the healing.
    """
    if move.heal <= 0:
        return 0.0

    hp_restored = user.constants.stats[Stat.MAX_HP] * move.heal
    hp_after = min(user.hp + hp_restored, user.constants.stats[Stat.MAX_HP])

    if user.hp == user.constants.stats[Stat.MAX_HP]:
        return 0.0

    incoming_threat, _ = estimate_incoming_threat(user, state.sides[1].team.active, state, params)

    if user.hp <= incoming_threat and hp_after > incoming_threat:
        best_dmg = 0.0
        opp_actives = [p for p in state.sides[1].team.active if p and p.hp > 0]
        if opp_actives:
            target = opp_actives[0]
            for user_move in user.constants.species.moves:
                if user_move.category in (Category.PHYSICAL, Category.SPECIAL):
                    dmg = calculate_damage(params, 0, user_move, state, user, target)
                    if dmg > best_dmg:
                        best_dmg = dmg
        return best_dmg

    return hp_restored  # type: ignore[no-any-return]


def _field_setup_move(
    attacker: BattlingPokemonView,
    move: Move,
    my_team_actives: list[Any],
    my_team_reserve: list[Any],
    opponent_actives: list[Any],
    current_state: StateView,
    params: BattleRuleParam,
) -> float:
    """Calculate the value of setting a field effect (weather, terrain, hazards, trick room).

    For weather/terrain: computes net damage swing (ally gain minus opponent gain).
    For trick room: computes turn-order reversal value.
    For hazards: computes total expected damage to the opponent's team.

    Args:
        attacker: The Pokemon using the field move.
        move: The Move constants being evaluated.
        my_team_actives: Our active Pokemon views.
        my_team_reserve: Our reserve Pokemon views.
        opponent_actives: Opponent active Pokemon views.
        current_state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Float value of the field setup move.
    """
    from vgc2.battle_engine.game_state import State

    setup_score = 0.0

    new_weather = move.weather_start
    new_terrain = move.field_start

    if (new_weather != Weather.CLEAR and new_weather != current_state.weather) or (
        new_terrain != Terrain.NONE and new_terrain != current_state.field
    ):
        net_swing = 0.0

        temp_state = State((current_state.sides[0].team, current_state.sides[1].team))
        if new_weather != Weather.CLEAR:
            temp_state.weather = new_weather
        if new_terrain != Terrain.NONE:
            temp_state.field = new_terrain

        opp_representative = next((p for p in opponent_actives if p and p.hp > 0), None)
        ally_representative = next((p for p in my_team_actives if p and p.hp > 0), None)

        for pkm in my_team_actives:
            if not pkm or pkm.hp <= 0 or not opp_representative:
                continue
            best_before = -1
            best_after = -1
            for pkm_move in pkm.constants.species.moves:
                if pkm_move.category not in (Category.PHYSICAL, Category.SPECIAL):
                    continue
                dmg_before = calculate_damage(params, 0, pkm_move, current_state, pkm, opp_representative)
                dmg_after = calculate_damage(params, 0, pkm_move, temp_state, pkm, opp_representative)
                if dmg_before > best_before:
                    best_before = dmg_before
                    best_after = dmg_after
            if best_before >= 0:
                net_swing += best_after - best_before

        for pkm in opponent_actives:
            if not pkm or pkm.hp <= 0 or not ally_representative:
                continue
            best_before = -1
            best_after = -1
            for pkm_move in pkm.constants.species.moves:
                if pkm_move.category not in (Category.PHYSICAL, Category.SPECIAL):
                    continue
                dmg_before = calculate_damage(params, 1, pkm_move, current_state, pkm, ally_representative)
                dmg_after = calculate_damage(params, 1, pkm_move, temp_state, pkm, ally_representative)
                if dmg_before > best_before:
                    best_before = dmg_before
                    best_after = dmg_after
            if best_before >= 0:
                net_swing -= best_after - best_before

        setup_score += net_swing

        for pkm in my_team_actives + opponent_actives:
            if not pkm or pkm.hp <= 0:
                continue
            side_mult = 1.0 if pkm in my_team_actives else -1.0
            if new_weather == Weather.SAND:
                setup_score += calculate_sand_damage(params, pkm) * side_mult * -1

    if move.toggle_trickroom:
        if not current_state.trickroom:
            net_value = 0.0
            for my_pkm in my_team_actives:
                for opp_pkm in opponent_actives:
                    if not my_pkm or not opp_pkm:
                        continue
                    if my_pkm.constants.stats[Stat.SPEED] < opp_pkm.constants.stats[Stat.SPEED]:
                        threat_to_opp, _ = estimate_incoming_threat(opp_pkm, [my_pkm], current_state, params)
                        net_value += threat_to_opp
            setup_score += net_value
        else:
            net_value = 0.0
            for my_pkm in my_team_actives:
                for opp_pkm in opponent_actives:
                    if not my_pkm or not opp_pkm:
                        continue
                    if my_pkm.constants.stats[Stat.SPEED] > opp_pkm.constants.stats[Stat.SPEED]:
                        threat_to_opp, _ = estimate_incoming_threat(opp_pkm, [my_pkm], current_state, params)
                        net_value += threat_to_opp
                    elif my_pkm.constants.stats[Stat.SPEED] < opp_pkm.constants.stats[Stat.SPEED]:
                        threat_from_opp, _ = estimate_incoming_threat(my_pkm, [opp_pkm], current_state, params)
                        net_value -= threat_from_opp
            setup_score += net_value

    if move.hazard != Hazard.NONE:
        hazard = move.hazard
        opp_conds = current_state.sides[1].conditions

        if (hazard == Hazard.STEALTH_ROCK and opp_conds.stealth_rock) or (
            hazard == Hazard.TOXIC_SPIKES and opp_conds.poison_spikes
        ):
            setup_score += HAZARD_REDUNDANCY_PENALTY
        else:
            total_hazard_dmg = 0.0
            opp_full_team = opponent_actives + [p for p in current_state.sides[1].team.reserve if p and p.hp > 0]

            for opp_pkm in opp_full_team:
                if hazard == Hazard.STEALTH_ROCK:
                    total_hazard_dmg += calculate_stealth_rock_damage(params, opp_pkm)
                elif hazard == Hazard.TOXIC_SPIKES:
                    is_immune = any(t in (Type.POISON, Type.STEEL, Type.FLYING) for t in opp_pkm.types)
                    if not is_immune and opp_pkm.status == Status.NONE:
                        total_hazard_dmg += calculate_poison_damage(params, opp_pkm)

            setup_score += total_hazard_dmg

    return setup_score
