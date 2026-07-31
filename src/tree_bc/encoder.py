"""State encoder for BC battle policy.

Converts a vgc2 StateView into a fixed-size numpy feature vector
for downstream classification by an XGBoost action predictor.

Feature layout (301 features, float32):

    Pokemon slots (8 x 23 = 184):
      own_act[0], own_act[1], own_res[0], own_res[1],
      opp_act[0], opp_act[1], opp_res[0], opp_res[1]

      Per-slot features (23):
        0: hp_fraction   [0, 1]     hp / stats[Stat.MAX_HP]
        1: is_alive      {0, 1}     hp > 0
        2-3: type_idx    [0, 1]     int(types[i]) / 18.0 (pad second with 0.0)
        4-10: status_oh  {0, 1}     one-hot over 7 Status values
        11-15: boosts    [-1, 1]    boosts[1:6] / 6.0 (atk,def,spa,spd,spe)
        16-21: base_stat [0, 1]     species.base_stats[i] / 255.0
        22: protect      {0, 1}     pkm.protect

    Moves (8 x 12 = 96):
      own_act[0]_move[0..3], own_act[1]_move[0..3]

      Per-move features (12):
        0: base_power   [0, 1]     base_power / 200.0
        1: accuracy     [0, 1]     accuracy (0.0 if None)
        2: pp_fraction  [0, 1]     pp / max_pp
        3-5: cat_oh     {0, 1}     one-hot over 3 Category values
        6: priority     [-1, 1]    clamp(priority / 5.0)
        7: move_type    [0, 1]     int(pkm_type) / 18.0
        8: is_protect   {0, 1}     move.constants.protect
        9: has_status   {0, 1}     move.constants.status != Status.NONE
        10: has_boost   {0, 1}     any(b != 0 for b in move.constants.boosts)
        11: disabled     {0, 1}     move.disabled

    Global (11):
      weather    one-hot 5   [0:4]
      terrain    one-hot 5   [0:4]
      trickroom  binary 1

    Side conditions (10):
      sides[0]   reflect, lightscreen, tailwind, stealth_rock, poison_spikes (5)
      sides[1]   same (5)

    Total: 184 + 96 + 11 + 10 = 301

Constants
---------
FEATURE_DIM:
    Total feature vector length (301).
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vgc2.battle_engine.view import StateView

FEATURE_DIM = 301

_N_TYPES = 18
_N_STATUS = 7
_N_CATEGORY = 3
_N_WEATHER = 5
_N_TERRAIN = 5

_POKEMON_SLOTS = 8
_POKEMON_FEAT_PER_SLOT = 23
_MOVE_SLOTS = 8
_MOVE_FEAT_PER_SLOT = 12
_GLOBAL_FEAT = 11
_SIDE_COND_FEAT = 10


def _one_hot(value: int, size: int) -> np.ndarray:
    """Create a one-hot encoded float32 vector.

    Args:
        value: Index to set to 1.0.
        size: Total number of slots.

    Returns:
        One-hot vector of shape (size,) with float32 dtype.
    """
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= value < size:
        vec[value] = 1.0
    return vec


def _encode_pokemon(pkm: object) -> np.ndarray:
    """Encode a single Pokemon into 23 features.

    Fainted Pokemon (hp <= 0) and None slots are encoded as all zeros.

    Args:
        pkm: A BattlingPokemonView instance, or None for an empty slot.

    Returns:
        Float32 array of shape (23,).
    """
    feats = np.zeros(_POKEMON_FEAT_PER_SLOT, dtype=np.float32)

    if pkm is None:
        return feats

    try:
        hp = pkm.hp
    except AttributeError:
        return feats

    if hp <= 0:
        return feats

    try:
        max_hp = pkm.constants.stats[0]
    except (AttributeError, IndexError, TypeError):
        max_hp = 1

    feats[0] = float(np.clip(hp / max(max_hp, 1), 0.0, 1.0))
    feats[1] = 1.0

    try:
        types = pkm.types
    except AttributeError:
        types = []

    for t_idx in range(2):
        if t_idx < len(types):
            feats[2 + t_idx] = int(types[t_idx]) / _N_TYPES

    try:
        status_val = int(pkm.status)
    except (AttributeError, TypeError):
        status_val = 0
    feats[4:4 + _N_STATUS] = _one_hot(status_val, _N_STATUS)

    try:
        boosts = pkm.boosts
    except AttributeError:
        boosts = [0] * 8

    for b_idx, b_pos in enumerate(range(1, 6)):
        val = boosts[b_pos] if b_pos < len(boosts) else 0
        feats[11 + b_idx] = float(np.clip(val / 6.0, -1.0, 1.0))

    try:
        base = pkm.constants.species.base_stats
    except AttributeError:
        base = ()

    for s_idx in range(6):
        val = base[s_idx] if s_idx < len(base) else 0
        feats[16 + s_idx] = float(np.clip(val / 255.0, 0.0, 1.0))

    try:
        feats[22] = 1.0 if pkm.protect else 0.0
    except AttributeError:
        feats[22] = 0.0

    return feats


def _encode_move(battling_move: object) -> np.ndarray:
    """Encode a single move into 12 features.

    Detects DUMMY_MOVE via identity check and returns all zeros.
    Handles accuracy=None by encoding as 0.0.

    Args:
        battling_move: A BattlingMove instance, or None for an empty slot.

    Returns:
        Float32 array of shape (12,).
    """
    feats = np.zeros(_MOVE_FEAT_PER_SLOT, dtype=np.float32)

    if battling_move is None:
        return feats

    try:
        from vgc2.battle_engine.view import DUMMY_MOVE
        if battling_move is DUMMY_MOVE:
            return feats
    except ImportError:
        pass

    try:
        mc = battling_move.constants
    except AttributeError:
        return feats

    try:
        bp = mc.base_power
    except AttributeError:
        bp = 0
    feats[0] = float(np.clip(bp / 200.0, 0.0, 1.0))

    try:
        acc = mc.accuracy
        if acc is None:
            acc = 0.0
    except AttributeError:
        acc = 0.0
    feats[1] = float(np.clip(acc, 0.0, 1.0))

    try:
        pp = battling_move.pp
        max_pp = mc.max_pp
    except AttributeError:
        pp = 0
        max_pp = 1
    feats[2] = float(np.clip(pp / max(max_pp, 1), 0.0, 1.0))

    try:
        cat_val = int(mc.category)
    except (AttributeError, TypeError):
        cat_val = 0
    feats[3:3 + _N_CATEGORY] = _one_hot(cat_val, _N_CATEGORY)

    try:
        prio = mc.priority
    except AttributeError:
        prio = 0
    feats[6] = float(np.clip(prio / 5.0, -1.0, 1.0))

    try:
        move_type = int(mc.pkm_type)
    except (AttributeError, TypeError):
        move_type = 0
    feats[7] = move_type / _N_TYPES

    try:
        feats[8] = 1.0 if mc.protect else 0.0
    except AttributeError:
        feats[8] = 0.0

    try:
        status = mc.status
    except AttributeError:
        status = None
    feats[9] = 1.0 if (status is not None and int(status) != 0) else 0.0

    try:
        boosts = mc.boosts
        has_boost = any(b != 0 for b in boosts)
    except (AttributeError, TypeError):
        has_boost = False
    feats[10] = 1.0 if has_boost else 0.0

    try:
        feats[11] = 1.0 if battling_move.disabled else 0.0
    except AttributeError:
        feats[11] = 0.0

    return feats


def _encode_global(state: object) -> np.ndarray:
    """Encode global field state into 11 features.

    Weather one-hot (5), terrain one-hot (5), trickroom binary (1).

    Args:
        state: A StateView instance.

    Returns:
        Float32 array of shape (11,).
    """
    feats = np.zeros(_GLOBAL_FEAT, dtype=np.float32)

    try:
        weather_val = int(state.weather)
    except (AttributeError, TypeError):
        weather_val = 0
    feats[0: _N_WEATHER] = _one_hot(weather_val, _N_WEATHER)

    try:
        terrain_val = int(state.field)
    except (AttributeError, TypeError):
        terrain_val = 0
    feats[_N_WEATHER: _N_WEATHER + _N_TERRAIN] = _one_hot(terrain_val, _N_TERRAIN)

    with suppress(AttributeError):
        feats[_N_WEATHER + _N_TERRAIN] = 1.0 if state.trickroom else 0.0

    return feats


def _encode_side_conditions(side: object) -> np.ndarray:
    """Encode one side's conditions into 5 binary features.

    Order: reflect, lightscreen, tailwind, stealth_rock, poison_spikes.

    Args:
        side: A Side or SideView object with .conditions attribute.

    Returns:
        Float32 array of shape (5,).
    """
    feats = np.zeros(5, dtype=np.float32)

    try:
        cond = side.conditions
    except AttributeError:
        return feats

    for idx, name in enumerate(
        ("reflect", "lightscreen", "tailwind", "stealth_rock", "poison_spikes")
    ):
        try:
            feats[idx] = 1.0 if getattr(cond, name, False) else 0.0
        except AttributeError:
            feats[idx] = 0.0

    return feats


def encode_state(state: StateView) -> np.ndarray:
    """Convert a StateView into a fixed-size float32 feature vector.

    Encodes all Pokemon (own + opponent active and reserve), own moves,
    global field state, and side conditions. Fainted or missing Pokemon
    are zero-padded. Unrevealed opponent moves (DUMMY_MOVE) are zero-padded.

    The output is deterministic: calling encode_state twice on the same
    StateView produces identical arrays.

    Args:
        state: A StateView from a vgc2 battle engine.

    Returns:
        Float32 numpy array of shape (FEATURE_DIM,).

    Raises:
        TypeError: If state is not a StateView.
    """
    feats = np.empty(FEATURE_DIM, dtype=np.float32)
    pos = 0

    for side_idx in range(2):
        try:
            side = state.sides[side_idx]
        except (AttributeError, IndexError):
            side = None

        for slot_type in ("active", "reserve"):
            if side is None:
                team = []
            else:
                try:
                    team = getattr(side.team, slot_type)
                except AttributeError:
                    team = []

            for slot_idx in range(_POKEMON_SLOTS // 4):
                pkm = team[slot_idx] if slot_idx < len(team) else None
                slot_feats = _encode_pokemon(pkm)
                feats[pos: pos + _POKEMON_FEAT_PER_SLOT] = slot_feats
                pos += _POKEMON_FEAT_PER_SLOT

    for active_idx in range(_POKEMON_SLOTS // 4):
        try:
            own_side = state.sides[0]
            active_list = own_side.team.active
        except (AttributeError, IndexError):
            active_list = []

        pkm = active_list[active_idx] if active_idx < len(active_list) else None

        moves: list[object] = []
        if pkm is not None and hasattr(pkm, "hp") and pkm.hp > 0:
            try:
                moves = list(pkm.battling_moves) if hasattr(pkm, "battling_moves") else []
            except (AttributeError, TypeError):
                moves = []

        for move_idx in range(_MOVE_SLOTS // 2):
            move = moves[move_idx] if move_idx < len(moves) else None
            move_feats = _encode_move(move)
            feats[pos: pos + _MOVE_FEAT_PER_SLOT] = move_feats
            pos += _MOVE_FEAT_PER_SLOT

    global_feats = _encode_global(state)
    feats[pos: pos + _GLOBAL_FEAT] = global_feats
    pos += _GLOBAL_FEAT

    for side_idx in range(2):
        try:
            side = state.sides[side_idx]
        except (AttributeError, IndexError):
            side = None

        cond_feats = _encode_side_conditions(side) if side is not None else np.zeros(5, dtype=np.float32)
        feats[pos: pos + 5] = cond_feats
        pos += 5

    return feats
