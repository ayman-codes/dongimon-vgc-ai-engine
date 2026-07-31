"""Action encoder for BC battle policy.

Maps between joint BattleCommand pairs and discrete action class
indices (0-99) for 100-class classification. Provides valid-action
masking for inference-time filtering of illegal moves.

Encoding scheme:
    Per-Pokemon action (0-9):
        0-7: move actions     (move_idx * 2 + target)
        8-9: switch actions   (8 + reserve_idx)

    Joint action: pkm0_action * 10 + pkm1_action

Action space summary:
    Move actions:  4 moves x 2 targets = 8 per Pokemon
    Switch:         2 reserve slots     = 2 per Pokemon
    Per Pokemon:                        = 10
    Joint (2v2):                         = 100
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vgc2.battle_engine.view import StateView

BattleCommand = tuple[int, int]

JOINT_ACTION_COUNT = 100
PER_POKEMON_ACTIONS = 10
MOVE_ACTIONS = 8
SWITCH_BASE = 8


def encode_action(commands: list[BattleCommand]) -> int:
    """Map a joint command pair to a class index (0-99).

    Per-Pokemon encoding:
        Move: (move_idx, target) -> move_idx * 2 + target  (0-7)
        Switch: (-1, reserve_idx) -> 8 + reserve_idx       (8-9)

    Joint encoding: pkm0_action * 10 + pkm1_action.

    If only one command is provided, the second Pokemon is treated
    as action 0 (dummy, filtered by valid-action masking at inference).

    Args:
        commands: List of 1 or 2 BattleCommand tuples.

    Returns:
        Integer action class index in [0, JOINT_ACTION_COUNT).

    Raises:
        ValueError: If commands is empty.
    """
    if not commands:
        raise ValueError("commands must not be empty")

    actions: list[int] = []
    for cmd in commands:
        action_val, target_val = cmd
        per_pkm = (
            SWITCH_BASE + target_val
            if action_val == -1
            else action_val * 2 + target_val
        )
        actions.append(per_pkm)

    while len(actions) < 2:
        actions.append(0)

    return actions[0] * PER_POKEMON_ACTIONS + actions[1]


def _decode_per_pokemon(action: int) -> BattleCommand:
    """Decode a per-Pokemon action index (0-9) to a BattleCommand.

    Args:
        action: Per-Pokemon action index.

    Returns:
        BattleCommand tuple: (move_idx, target) or (-1, reserve_idx).
    """
    if action >= SWITCH_BASE:
        return (-1, action - SWITCH_BASE)
    move_idx = action // 2
    target = action % 2
    return (move_idx, target)


def decode_action(action_idx: int) -> list[BattleCommand]:
    """Map a class index (0-99) back to a joint command pair.

    Args:
        action_idx: Integer action class index.

    Returns:
        List of 2 BattleCommand tuples.

    Raises:
        ValueError: If action_idx is out of bounds.
    """
    if not (0 <= action_idx < JOINT_ACTION_COUNT):
        raise ValueError(
            f"action_idx {action_idx} is out of bounds [0, {JOINT_ACTION_COUNT})"
        )

    pkm0_action = action_idx // PER_POKEMON_ACTIONS
    pkm1_action = action_idx % PER_POKEMON_ACTIONS
    return [_decode_per_pokemon(pkm0_action), _decode_per_pokemon(pkm1_action)]


def get_valid_actions(state: StateView) -> list[int]:
    """Return indices of all legally executable joint actions.

    Filters out actions involving:
        Fainted Pokemon (own or target).
        Moves with 0 PP.
        Disabled moves.
        Switches when no healthy reserve is available.

    When only one own Pokemon is alive, returns fewer than 100 actions
    (the second slot is treated as forced dummy action 0).

    Args:
        state: A StateView (sides[0] = own team).

    Returns:
        Sorted list of valid joint action indices.
    """
    try:
        own_active = state.sides[0].team.active
        opp_active = state.sides[1].team.active
        own_reserve = state.sides[0].team.reserve
    except (AttributeError, IndexError):
        return []

    valid_per_slot: list[list[int]] = []

    for slot_idx in range(2):
        slot_valid: list[int] = []

        pkm = own_active[slot_idx] if slot_idx < len(own_active) else None
        if pkm is None or (hasattr(pkm, "hp") and pkm.hp <= 0):
            slot_valid.append(0)
            valid_per_slot.append(slot_valid)
            continue

        for move_idx in range(4):
            try:
                move = pkm.battling_moves[move_idx]
            except (AttributeError, IndexError):
                continue

            try:
                if getattr(move, "disabled", False):
                    continue
            except AttributeError:
                pass

            try:
                pp = getattr(move, "pp", 0)
            except AttributeError:
                pp = 0
            if pp <= 0:
                continue

            for target_idx in range(2):
                opp = opp_active[target_idx] if target_idx < len(opp_active) else None
                if opp is None or (hasattr(opp, "hp") and opp.hp <= 0):
                    continue
                slot_valid.append(move_idx * 2 + target_idx)

        for reserve_idx in range(2):
            reserve_pkm = own_reserve[reserve_idx] if reserve_idx < len(own_reserve) else None
            if reserve_pkm is not None and hasattr(reserve_pkm, "hp") and reserve_pkm.hp > 0:
                slot_valid.append(SWITCH_BASE + reserve_idx)

        if not slot_valid:
            slot_valid.append(0)

        valid_per_slot.append(slot_valid)

    joint_valid: list[int] = []
    for a0 in valid_per_slot[0]:
        for a1 in valid_per_slot[1]:
            joint_valid.append(a0 * PER_POKEMON_ACTIONS + a1)

    joint_valid.sort()
    return joint_valid
