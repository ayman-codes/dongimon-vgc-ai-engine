"""Joint action pairing logic for double battles.

Evaluates all combinations of actions (moves and switches) for two
active Pokemon and computes a weighted joint score from individual
scores plus cross-slot synergy components.
"""


from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Stat, Terrain, Type, Weather
from vgc2.battle_engine.view import BattlingPokemonView, StateView

from src.battle.move_scoring import (
    calculate_focus_fire_bonus,
    estimate_incoming_threat,
    identify_biggest_threat,
)
from src.config.constants import (
    BOARD_WEIGHT,
    GOOD_OFFENSIVE_FOLLOWUP_THRESHOLD,
    GOOD_PROTECT_THRESHOLD,
    HIGH_VALUE_PROTECT_THRESHOLD,
    LETHAL_SURVIVAL_PENALTY_MULT,
    OFF_DEF_SUPPORT_BONUS,
    PROPORTIONAL_SURVIVAL_PENALTY_MULT,
    SETUP_MOVE_MAX_BP,
    SETUP_MOVE_MIN_SCORE,
    SETUP_SYNERGY_BONUS,
    STRONG_OFFENSIVE_THRESHOLD,
    TARGET_PRIORITY_BASE,
    TARGET_PRIORITY_KO_ALLY_MULT,
    TERRAIN_SYNERGY_BONUS,
    THREAT_KO_ALLY_BASE,
    TRICK_ROOM_SYNERGY_BONUS,
    WEATHER_SYNERGY_BONUS,
)


def evaluate_joint_actions(
    actions_slot_a: list,
    actions_slot_b: list,
    my_pkm_a: BattlingPokemonView,
    my_pkm_b: BattlingPokemonView,
    opp_active_list: list,
    my_active_list: list,
    state: StateView,
    params: BattleRuleParam,
    weights: dict[str, float],
    max_score: float,
    locked_moves: dict[int, str] | None = None,
    lookahead_weight: float = BOARD_WEIGHT,
) -> tuple[list, dict, dict, float]:
    """Evaluate all joint action pairs and select the best.

    For each combination of actions (one per active Pokemon), computes
    individual base scores, survival impact, focus fire, target priority,
    off-def support, setup synergy, and environmental synergy bonuses.
    The best-scoring pair is returned.

    Args:
        actions_slot_a: List of (command, score, is_ko) tuples for Pokemon A.
        actions_slot_b: List of (command, score, is_ko) tuples for Pokemon B.
        my_pkm_a: Pokemon A view.
        my_pkm_b: Pokemon B view.
        opp_active_list: List of opponent active Pokemon views.
        my_active_list: List of our active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.
        weights: Dictionary of synergy weight values.
        max_score: Maximum individual score constant.
        locked_moves: Dict mapping opponent slot to their Choice-locked move name.

    Returns:
        Tuple of (final_commands, log_a, log_b, joint_score).
    """
    default_cmd = (0, 0)
    (default_cmd, -float("inf"), False)

    best_joint = -float("inf")
    chosen_pair = (default_cmd, default_cmd)
    log_a = {"command": None, "score": -float("inf"), "is_ko": False}
    log_b = {"command": None, "score": -float("inf"), "is_ko": False}

    biggest_threat_info = identify_biggest_threat(opp_active_list, my_active_list, state, params)
    biggest_threat_pkm = None
    biggest_threat_slot = -1
    if biggest_threat_info:
        biggest_threat_pkm, biggest_threat_slot = biggest_threat_info

    threat_a_undefended, pkm_a_ko_by_threat = estimate_incoming_threat(
        my_pkm_a, opp_active_list, state, params, locked_moves,
    )
    threat_b_undefended, pkm_b_ko_by_threat = estimate_incoming_threat(
        my_pkm_b, opp_active_list, state, params, locked_moves,
    )

    raw_a: list[float] = []
    raw_b: list[float] = []
    raw_s1: list[float] = []
    raw_s2: list[float] = []
    raw_ff: list[float] = []
    raw_tp: list[float] = []
    raw_od: list[float] = []
    raw_su: list[float] = []
    raw_ev: list[float] = []
    raw_bp: list[float] = []
    pair_info: list[tuple] = []

    for cmd_a_info in actions_slot_a:
        cmd_a, score_a, is_ko_a = cmd_a_info
        is_move_a = cmd_a[0] >= 0
        cmd_a_target = cmd_a[1] if is_move_a else -1
        move_a_const = None
        if is_move_a and cmd_a[0] < len(my_pkm_a.battling_moves):
            move_a_const = my_pkm_a.battling_moves[cmd_a[0]].constants

        for cmd_b_info in actions_slot_b:
            cmd_b, score_b, is_ko_b = cmd_b_info
            is_move_b = cmd_b[0] >= 0
            cmd_b_target = cmd_b[1] if is_move_b else -1
            move_b_const = None
            if is_move_b and cmd_b[0] < len(my_pkm_b.battling_moves):
                move_b_const = my_pkm_b.battling_moves[cmd_b[0]].constants

            sv_a = _survival_impact(
                my_pkm_a, threat_a_undefended, is_move_a, move_a_const,
                is_ko_b, move_b_const, cmd_b_target, biggest_threat_pkm,
                biggest_threat_slot, max_score,
            )
            sv_b = _survival_impact_b(
                my_pkm_b, opp_active_list, is_ko_a, move_a_const, cmd_a_target,
                is_move_b, move_b_const, threat_b_undefended, pkm_b_ko_by_threat,
                biggest_threat_pkm, biggest_threat_slot, max_score, state, params,
                locked_moves,
            )

            ff = _focus_fire_wrapper(
                my_pkm_a, move_a_const, is_ko_a,
                my_pkm_b, move_b_const, is_ko_b,
                cmd_a_target, cmd_b_target, is_move_a, is_move_b,
                opp_active_list, state, params, biggest_threat_pkm,
            )

            tp = _target_priority(
                is_ko_a, is_ko_b, move_a_const, move_b_const,
                cmd_a_target, cmd_b_target, is_move_a, is_move_b,
                biggest_threat_pkm, biggest_threat_slot,
                my_active_list, state, params, locked_moves,
            )

            od = _off_def_support(
                is_move_a, move_a_const, score_a,
                is_move_b, move_b_const, score_b,
                pkm_a_ko_by_threat, pkm_b_ko_by_threat,
            )

            su = _setup_synergy(
                is_move_a, move_a_const, score_a, is_ko_a,
                is_move_b, move_b_const, score_b, is_ko_b,
            )

            ev = _env_synergy(
                is_move_a, move_a_const, is_move_b, move_b_const,
                my_pkm_a, my_pkm_b, state,
            )

            my_alive_after = sum(1 for p in my_active_list if p and p.hp > 0)
            opp_alive_after = sum(1 for p in opp_active_list if p and p.hp > 0)
            if (
                is_ko_a and cmd_a_target >= 0 and cmd_a_target < len(opp_active_list)
                and opp_active_list[cmd_a_target] and opp_active_list[cmd_a_target].hp > 0
            ):
                opp_alive_after -= 1
            if (
                is_ko_b and cmd_b_target >= 0 and cmd_b_target < len(opp_active_list)
                and opp_active_list[cmd_b_target] and opp_active_list[cmd_b_target].hp > 0
            ):
                opp_alive_after -= 1

            bp_score = 2.0 * my_alive_after - 2.5 * opp_alive_after

            raw_a.append(score_a)
            raw_b.append(score_b)
            raw_s1.append(sv_a)
            raw_s2.append(sv_b)
            raw_ff.append(ff)
            raw_tp.append(tp)
            raw_od.append(od)
            raw_su.append(su)
            raw_ev.append(ev)
            raw_bp.append(bp_score)
            pair_info.append((cmd_a, cmd_b, is_ko_a, is_ko_b))

    comp_max = {
        "a": max(raw_a) if raw_a else 1.0,
        "b": max(raw_b) if raw_b else 1.0,
        "s": max(abs(v) for v in raw_s1 + raw_s2) if raw_s1 or raw_s2 else 1.0,
        "ff": max(raw_ff) if raw_ff else 1.0,
        "tp": max(raw_tp) if raw_tp else 1.0,
        "od": max(raw_od) if raw_od else 1.0,
        "su": max(raw_su) if raw_su else 1.0,
        "ev": max(raw_ev) if raw_ev else 1.0,
        "bp": max(abs(v) for v in raw_bp) if raw_bp else 1.0,
    }

    def _div(v: float, m: float) -> float:
        return v / m if m > 0 else 0.0

    for idx, (cmd_a, cmd_b, is_ko_a, is_ko_b) in enumerate(pair_info):
        n_a = _div(raw_a[idx], comp_max["a"])
        n_b = _div(raw_b[idx], comp_max["b"])
        n_s1 = _div(raw_s1[idx], comp_max["s"])
        n_s2 = _div(raw_s2[idx], comp_max["s"])
        n_ff = _div(raw_ff[idx], comp_max["ff"])
        n_tp = _div(raw_tp[idx], comp_max["tp"])
        n_od = _div(raw_od[idx], comp_max["od"])
        n_su = _div(raw_su[idx], comp_max["su"])
        n_ev = _div(raw_ev[idx], comp_max["ev"])
        n_bp = _div(raw_bp[idx], comp_max["bp"])

        joint = (
            n_a * weights.get("w_base_score_a", 0.05)
            + n_b * weights.get("w_base_score_b", 0.15)
            + (n_s1 + n_s2) * weights.get("w_survival_impact", 0.13)
            + n_ff * weights.get("w_focus_fire", 0.27)
            + n_tp * weights.get("w_target_priority", 0.18)
            + n_od * weights.get("w_off_def_support", 0.02)
            + n_su * weights.get("w_setup_synergy", 0.18)
            + n_ev * weights.get("w_env_synergy", 0.02)
            + n_bp * lookahead_weight
        )

        if joint > best_joint:
            best_joint = joint
            chosen_pair = (cmd_a, cmd_b)
            log_a = {"command": cmd_a, "score": raw_a[idx], "is_ko": is_ko_a}
            log_b = {"command": cmd_b, "score": raw_b[idx], "is_ko": is_ko_b}

    return list(chosen_pair), log_a, log_b, best_joint


def _survival_impact(
    pkm: BattlingPokemonView,
    threat_undefended: float,
    is_move: bool,
    move_const: object,
    ally_ko_threat: bool,
    ally_move_const: object,
    ally_target_slot: int,
    biggest_threat_pkm: object,
    biggest_threat_slot: int,
    max_score: float,
) -> float:
    """Compute survival impact score for one Pokemon.

    Positive score means survival is likely. Negative penalty
    applies if the Pokemon would faint and the ally is not
    eliminating the threat.

    Args:
        pkm: The Pokemon being evaluated.
        threat_undefended: Incoming threat if no defensive action.
        is_move: Whether the action is a move.
        move_const: Move constants (None if switch).
        ally_ko_threat: Whether the ally's move KOs the biggest threat.
        ally_move_const: Ally's move constants.
        ally_target_slot: Ally's target slot.
        biggest_threat_pkm: Biggest threat Pokemon view.
        biggest_threat_slot: Biggest threat slot index.
        max_score: Maximum score constant (1000.0).

    Returns:
        Negative float penalty (or zero if safe).
    """
    dmg_taken = threat_undefended
    if is_move and move_const and move_const.protect or not is_move:
        dmg_taken = 0

    if dmg_taken >= pkm.hp and pkm.hp > 0:
        ally_ko_biggest = (
            ally_ko_threat
            and ally_move_const is not None
            and not ally_move_const.protect
            and biggest_threat_pkm is not None
            and ally_target_slot == biggest_threat_slot
        )
        if not ally_ko_biggest:
            return -(max_score * LETHAL_SURVIVAL_PENALTY_MULT)
    elif dmg_taken > 0:
        pkm_max_hp = pkm.constants.stats[Stat.MAX_HP] if pkm.constants else 1.0
        if pkm_max_hp <= 0:
            pkm_max_hp = 1.0
        hp_pct = dmg_taken / pkm_max_hp
        return -(hp_pct * (max_score * PROPORTIONAL_SURVIVAL_PENALTY_MULT))

    return 0.0


def _survival_impact_b(
    pkm_b: BattlingPokemonView,
    opp_active_list: list,
    is_ko_a: bool,
    move_a_const: object,
    cmd_a_target: int,
    is_move_b: bool,
    move_b_const: object,
    threat_b_undefended: float,
    pkm_b_ko_by_threat: bool,
    biggest_threat_pkm: object,
    biggest_threat_slot: int,
    max_score: float,
    state: StateView,
    params: BattleRuleParam,
    locked_moves: dict[int, str] | None = None,
) -> float:
    """Compute survival impact for Pokemon B, adjusted for A's KO.

    Removes opponent Pokemon that A KOs from the threat calculation.

    Args:
        pkm_b: Pokemon B view.
        opp_active_list: Opponent active Pokemon views.
        is_ko_a: Whether Pokemon A's move KOs.
        move_a_const: Pokemon A's move constants.
        cmd_a_target: Pokemon A's target slot.
        is_move_b: Whether Pokemon B's action is a move.
        move_b_const: Pokemon B's move constants.
        threat_b_undefended: Base threat to Pokemon B.
        pkm_b_ko_by_threat: Whether Pokemon B is threatened by KO.
        biggest_threat_pkm: Biggest threat opponent view.
        biggest_threat_slot: Biggest threat slot index.
        max_score: Maximum score constant.
        state: Current battle state view.
        params: Battle rule parameters.

    Returns:
        Negative float penalty (or zero if safe).
    """
    effective_opponents = []
    if move_a_const is not None and is_ko_a and move_a_const.protect is False:
        for i, opp in enumerate(opp_active_list):
            if not (i == cmd_a_target and is_ko_a) and opp and opp.hp > 0:
                effective_opponents.append(opp)
    else:
        effective_opponents = [opp for opp in opp_active_list if opp and opp.hp > 0]

    if not effective_opponents:
        threat_adjusted = 0.0
    else:
        threat_adjusted, _ = estimate_incoming_threat(pkm_b, effective_opponents, state, params, locked_moves)

    dmg_taken = threat_adjusted
    if is_move_b and move_b_const and move_b_const.protect or not is_move_b:
        dmg_taken = 0

    if dmg_taken >= pkm_b.hp and pkm_b.hp > 0:
        a_ko_threat = (
            is_ko_a
            and move_a_const is not None
            and not move_a_const.protect
            and biggest_threat_pkm is not None
            and cmd_a_target == biggest_threat_slot
        )
        if not a_ko_threat:
            return -(max_score * LETHAL_SURVIVAL_PENALTY_MULT)
    elif dmg_taken > 0:
        pkm_max_hp = pkm_b.constants.stats[Stat.MAX_HP] if pkm_b.constants else 1.0
        if pkm_max_hp <= 0:
            pkm_max_hp = 1.0
        hp_pct = dmg_taken / pkm_max_hp
        return -(hp_pct * (max_score * PROPORTIONAL_SURVIVAL_PENALTY_MULT))

    return 0.0


def _focus_fire_wrapper(
    pkm_a: BattlingPokemonView,
    move_a_const: object,
    is_ko_a: bool,
    pkm_b: BattlingPokemonView,
    move_b_const: object,
    is_ko_b: bool,
    cmd_a_target: int,
    cmd_b_target: int,
    is_move_a: bool,
    is_move_b: bool,
    opp_active_list: list,
    state: StateView,
    params: BattleRuleParam,
    biggest_threat_pkm: object,
) -> float:
    """Compute focus fire bonus if both Pokemon target the same opponent.

    Args:
        pkm_a: Pokemon A view.
        move_a_const: Pokemon A's move constants (None if switch).
        is_ko_a: Whether Pokemon A's move individually KOs its target.
        pkm_b: Pokemon B view.
        move_b_const: Pokemon B's move constants (None if switch).
        is_ko_b: Whether Pokemon B's move individually KOs its target.
        cmd_a_target: Target slot for Pokemon A.
        cmd_b_target: Target slot for Pokemon B.
        is_move_a: Whether Pokemon A's action is a move (not switch).
        is_move_b: Whether Pokemon B's action is a move (not switch).
        opp_active_list: Opponent active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.
        biggest_threat_pkm: Biggest threat opponent view.

    Returns:
        Focus fire bonus float, or 0.0 if not focusing same target.
    """
    if not (
        is_move_a and is_move_b and move_a_const and move_b_const
        and not move_a_const.protect and not move_b_const.protect
        and cmd_a_target == cmd_b_target and cmd_a_target != -1
    ):
        return 0.0

    target_idx = cmd_a_target
    if opp_active_list and target_idx < len(opp_active_list) and opp_active_list[target_idx] is not None:
        target = opp_active_list[target_idx]
        if target.hp > 0:
            return calculate_focus_fire_bonus(
                pkm_a, move_a_const, is_ko_a,
                pkm_b, move_b_const, is_ko_b,
                target, state, params, biggest_threat_pkm,
            )

    return 0.0


def _target_priority(
    is_ko_a: bool,
    is_ko_b: bool,
    move_a_const: object,
    move_b_const: object,
    cmd_a_target: int,
    cmd_b_target: int,
    is_move_a: bool,
    is_move_b: bool,
    biggest_threat_pkm: object,
    biggest_threat_slot: int,
    my_active_list: list,
    state: StateView,
    params: BattleRuleParam,
    locked_moves: dict[int, str] | None = None,
) -> float:
    """Compute bonus for KOing the biggest threat.

    Args:
        is_ko_a: Whether Pokemon A's move KOs.
        is_ko_b: Whether Pokemon B's move KOs.
        move_a_const: Pokemon A's move constants.
        move_b_const: Pokemon B's move constants.
        cmd_a_target: Pokemon A's target slot.
        cmd_b_target: Pokemon B's target slot.
        is_move_a: Whether Pokemon A's action is a move.
        is_move_b: Whether Pokemon B's action is a move.
        biggest_threat_pkm: Biggest threat opponent view.
        biggest_threat_slot: Biggest threat slot index.
        my_active_list: Our active Pokemon views.
        state: Current battle state view.
        params: Battle rule parameters.
        locked_moves: Dict mapping opponent slot to their Choice-locked move name.

    Returns:
        Bonus float for KOing the biggest threat.
    """
    if not biggest_threat_pkm or biggest_threat_pkm.hp <= 0:
        return 0.0

    ko_threat_a = (
        is_move_a and move_a_const is not None
        and not move_a_const.protect
        and cmd_a_target == biggest_threat_slot
        and is_ko_a
    )
    ko_threat_b = (
        is_move_b and move_b_const is not None
        and not move_b_const.protect
        and cmd_b_target == biggest_threat_slot
        and is_ko_b
    )

    if not (ko_threat_a or ko_threat_b):
        return 0.0

    bonus = TARGET_PRIORITY_BASE
    for my_pkm in my_active_list:
        if my_pkm and my_pkm.hp > 0:
            _, ko_my = estimate_incoming_threat(my_pkm, [biggest_threat_pkm], state, params, locked_moves)
            if ko_my:
                bonus += THREAT_KO_ALLY_BASE * TARGET_PRIORITY_KO_ALLY_MULT
                break

    return bonus


def _off_def_support(
    is_move_a: bool, move_a_const: object, score_a: float,
    is_move_b: bool, move_b_const: object, score_b: float,
    pkm_a_ko_by_threat: bool,
    pkm_b_ko_by_threat: bool,
) -> float:
    """Compute bonus for one Pokemon attacking while the other protects.

    Args:
        is_move_a: Whether Pokemon A's action is a move.
        move_a_const: Pokemon A's move constants.
        score_a: Pokemon A's individual score.
        is_move_b: Whether Pokemon B's action is a move.
        move_b_const: Pokemon B's move constants.
        score_b: Pokemon B's individual score.
        pkm_a_ko_by_threat: Whether Pokemon A is KO-threatened.
        pkm_b_ko_by_threat: Whether Pokemon B is KO-threatened.

    Returns:
        Bonus float for offensive+defensive pairing.
    """
    if not (move_a_const and move_b_const):
        return 0.0

    bonus = 0.0
    a_strong_off = is_move_a and not move_a_const.protect and score_a > STRONG_OFFENSIVE_THRESHOLD
    b_good_prot = is_move_b and move_b_const.protect and score_b > GOOD_PROTECT_THRESHOLD
    if a_strong_off and b_good_prot and (pkm_b_ko_by_threat or score_b > HIGH_VALUE_PROTECT_THRESHOLD):
        bonus += OFF_DEF_SUPPORT_BONUS

    b_strong_off = is_move_b and not move_b_const.protect and score_b > STRONG_OFFENSIVE_THRESHOLD
    a_good_prot = is_move_a and move_a_const.protect and score_a > GOOD_PROTECT_THRESHOLD
    if b_strong_off and a_good_prot and (pkm_a_ko_by_threat or score_a > HIGH_VALUE_PROTECT_THRESHOLD):
        bonus += OFF_DEF_SUPPORT_BONUS

    return bonus


def _setup_synergy(
    is_move_a: bool, move_a_const: object, score_a: float, is_ko_a: bool,
    is_move_b: bool, move_b_const: object, score_b: float, is_ko_b: bool,
) -> float:
    """Compute bonus for one Pokemon using a setup move while the other attacks or protects.

    Args:
        is_move_a: Whether Pokemon A's action is a move.
        move_a_const: Pokemon A's move constants.
        score_a: Pokemon A's individual score.
        is_ko_a: Whether Pokemon A's move KOs.
        is_move_b: Whether Pokemon B's action is a move.
        move_b_const: Pokemon B's move constants.
        score_b: Pokemon B's individual score.
        is_ko_b: Whether Pokemon B's move KOs.

    Returns:
        Bonus float for setup+follow-up pairing.
    """
    if not (move_a_const and move_b_const):
        return 0.0

    bonus = 0.0
    a_setup = _is_setup_move(is_move_a, move_a_const, score_a, is_ko_a)
    b_setup = _is_setup_move(is_move_b, move_b_const, score_b, is_ko_b)

    if a_setup:
        b_good_off = (
            is_move_b and not move_b_const.protect and not is_ko_b
            and score_b > GOOD_OFFENSIVE_FOLLOWUP_THRESHOLD
        )
        b_good_prot = is_move_b and move_b_const.protect and score_b > GOOD_PROTECT_THRESHOLD
        if b_good_off or b_good_prot:
            bonus += SETUP_SYNERGY_BONUS

    if b_setup:
        a_good_off = (
            is_move_a and not move_a_const.protect and not is_ko_a
            and score_a > GOOD_OFFENSIVE_FOLLOWUP_THRESHOLD
        )
        a_good_prot = is_move_a and move_a_const.protect and score_a > GOOD_PROTECT_THRESHOLD
        if a_good_off or a_good_prot:
            bonus += SETUP_SYNERGY_BONUS

    return bonus


def _is_setup_move(is_move: bool, move_const: object, score: float, is_ko: bool) -> bool:
    if not (is_move and move_const and score > SETUP_MOVE_MIN_SCORE):
        return False

    if move_const.category == Category.OTHER:
        return True
    return bool(
        move_const.base_power < SETUP_MOVE_MAX_BP
        and not is_ko
        and (
            move_const.boosts
            or move_const.weather_start != Weather.CLEAR
            or move_const.field_start != Terrain.NONE
            or move_const.toggle_trickroom
        )
    )


def _env_synergy(
    is_move_a: bool, move_a_const: object,
    is_move_b: bool, move_b_const: object,
    pkm_a_view: BattlingPokemonView,
    pkm_b_view: BattlingPokemonView,
    state: StateView,
) -> float:
    """Compute bonus for one Pokemon setting a field effect that benefits the other.

    Evaluates weather, terrain, and trick room cross-slot synergy.

    Args:
        is_move_a: Whether Pokemon A's action is a move.
        move_a_const: Pokemon A's move constants.
        is_move_b: Whether Pokemon B's action is a move.
        move_b_const: Pokemon B's move constants.
        pkm_a_view: Pokemon A view.
        pkm_b_view: Pokemon B view.
        state: Current battle state view.

    Returns:
        Bonus float for environmental synergy.
    """
    if not (is_move_a and move_a_const and is_move_b and move_b_const):
        return 0.0

    bonus = 0.0

    bonus += _weather_synergy(move_a_const, move_b_const, pkm_a_view, pkm_b_view, state)
    bonus += _weather_synergy(move_b_const, move_a_const, pkm_b_view, pkm_a_view, state)
    bonus += _terrain_synergy(move_a_const, move_b_const, pkm_a_view, pkm_b_view, state)
    bonus += _terrain_synergy(move_b_const, move_a_const, pkm_b_view, pkm_a_view, state)
    bonus += _trick_room_synergy(move_a_const, move_b_const, pkm_a_view, pkm_b_view, state)
    bonus += _trick_room_synergy(move_b_const, move_a_const, pkm_b_view, pkm_a_view, state)

    return bonus


def _weather_synergy(
    setter: object,
    beneficiary: object,
    setter_pkm: object,
    benefit_pkm: object,
    state: StateView,
) -> float:
    """Compute synergy bonus for setting weather that benefits an ally's move type.

    Args:
        setter: Move constants of the Pokemon setting weather.
        beneficiary: Move constants of the Pokemon benefiting.
        setter_pkm: Setter Pokemon view (unused, for API consistency).
        benefit_pkm: Beneficiary Pokemon view (unused).
        state: Current battle state view.

    Returns:
        Synergy bonus float.
    """
    if (
        setter.weather_start != Weather.CLEAR
        and setter.weather_start != state.weather
    ):
        new_weather = setter.weather_start
        if (
            (new_weather == Weather.RAIN and beneficiary.pkm_type == Type.WATER)
            or (new_weather == Weather.SUN and beneficiary.pkm_type == Type.FIRE)
        ):
            return WEATHER_SYNERGY_BONUS
    return 0.0


def _terrain_synergy(
    setter: object,
    beneficiary: object,
    setter_pkm: BattlingPokemonView,
    benefit_pkm: BattlingPokemonView,
    state: StateView,
) -> float:
    """Compute synergy bonus for setting terrain that benefits an ally's attack type.

    Args:
        setter: Move constants setting terrain.
        beneficiary: Move constants of the benefiting Pokemon.
        setter_pkm: Setter Pokemon view.
        benefit_pkm: Beneficiary Pokemon view.
        state: Current battle state view.

    Returns:
        Synergy bonus float.
    """
    if (
        setter.field_start != Terrain.NONE
        and setter.field_start != state.field
    ):
        new_terrain = setter.field_start
        is_grounded = not (
            Type.FLYING in benefit_pkm.types
            or getattr(benefit_pkm.constants.species, "ability", None) == "Levitate"
        )
        if is_grounded and (
            (new_terrain == Terrain.ELECTRIC_TERRAIN and beneficiary.pkm_type == Type.ELECTRIC)
            or (new_terrain == Terrain.GRASSY_TERRAIN and beneficiary.pkm_type == Type.GRASS)
            or (new_terrain == Terrain.PSYCHIC_TERRAIN and beneficiary.pkm_type == Type.PSYCHIC)
        ):
            return TERRAIN_SYNERGY_BONUS
    return 0.0


def _trick_room_synergy(
    setter: object,
    beneficiary: object,
    setter_pkm: BattlingPokemonView,
    benefit_pkm: BattlingPokemonView,
    state: StateView,
) -> float:
    """Compute synergy bonus for setting Trick Room when an ally is slow.

    Args:
        setter: Move constants setting Trick Room.
        beneficiary: Move constants of the benefiting Pokemon.
        setter_pkm: Setter Pokemon view (unused).
        benefit_pkm: Beneficiary Pokemon view.
        state: Current battle state view.

    Returns:
        Synergy bonus float.
    """
    if (
        setter.toggle_trickroom
        and not state.trickroom
        and benefit_pkm.constants
        and benefit_pkm.constants.stats[Stat.SPEED] < 70
        and not beneficiary.priority > 0
    ):
        return TRICK_ROOM_SYNERGY_BONUS
    return 0.0
