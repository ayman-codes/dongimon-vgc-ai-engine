"""Tests for BC battle policy state encoder.

Validates encode_state output shape, determinism, value ranges,
edge cases (fainted Pokemon, missing slots), and side symmetry.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from vgc2.battle_engine.modifiers import Category, Status, Terrain, Type, Weather

from src.tree_bc.encoder import FEATURE_DIM, encode_state

_POKEMON_FEAT_PER_SLOT = 23
_MOVE_FEAT_PER_SLOT = 12


def _make_mock_pkm(
    hp=200,
    max_hp=200,
    types=None,
    status=Status.NONE,
    boosts=None,
    base_stats=None,
    protect=False,
    moves=None,
):
    """Create a mock BattlingPokemonView with configurable attributes.

    Args:
        hp: Current HP value.
        max_hp: Maximum HP (stats[0]).
        types: List of Type enum members.
        status: Status enum member.
        boosts: List of 8 ints, indices 1-5 used for stat stages.
        base_stats: Tuple of 6 ints for species base stats.
        protect: Whether Protect is active.
        moves: List of mock BattlingMove objects.

    Returns:
        MagicMock configured as a BattlingPokemonView.
    """
    if types is None:
        types = [Type.NORMAL]
    if boosts is None:
        boosts = [0, 0, 0, 0, 0, 0, 0, 0]
    if base_stats is None:
        base_stats = (100, 100, 100, 100, 100, 100)
    if moves is None:
        moves = [_make_mock_battling_move() for _ in range(4)]

    pkm = MagicMock()
    pkm.hp = hp
    pkm.types = types
    pkm.status = status
    pkm.boosts = boosts
    pkm.protect = protect
    pkm.battling_moves = moves

    constants = MagicMock()
    constants.stats = (max_hp, 100, 80, 100, 80, 100)
    constants.species.base_stats = base_stats
    pkm.constants = constants

    return pkm


def _make_mock_move_const(
    base_power=80,
    accuracy=1.0,
    max_pp=16,
    category=Category.PHYSICAL,
    priority=0,
    pkm_type=Type.NORMAL,
    protect=False,
    status=Status.NONE,
    boosts=None,
):
    """Create a mock Move constants object.

    Args:
        base_power: Base power of the move.
        accuracy: Accuracy (float or None).
        max_pp: Maximum PP.
        category: Category enum member.
        priority: Priority level.
        pkm_type: Type enum member.
        protect: Whether the move is Protect.
        status: Status to inflict.
        boosts: Tuple of 8 ints for stat changes.

    Returns:
        MagicMock configured as Move constants.
    """
    if boosts is None:
        boosts = (0, 0, 0, 0, 0, 0, 0, 0)

    mc = MagicMock()
    mc.base_power = base_power
    mc.accuracy = accuracy
    mc.max_pp = max_pp
    mc.category = category
    mc.priority = priority
    mc.pkm_type = pkm_type
    mc.protect = protect
    mc.status = status
    mc.boosts = boosts
    return mc


def _make_mock_battling_move(
    base_power=80,
    accuracy=1.0,
    max_pp=16,
    category=Category.PHYSICAL,
    priority=0,
    pkm_type=Type.NORMAL,
    protect=False,
    status=Status.NONE,
    boosts=None,
    pp=None,
    disabled=False,
):
    """Create a mock BattlingMove wrapping a Move constants object.

    Args:
        base_power: Base power of the move.
        accuracy: Accuracy (float or None).
        max_pp: Maximum PP.
        category: Category enum member.
        priority: Priority level.
        pkm_type: Type enum member.
        protect: Whether the move is Protect.
        status: Status to inflict.
        boosts: Tuple of 8 ints for stat changes.
        pp: Current PP (defaults to max_pp).
        disabled: Whether the move is disabled.

    Returns:
        MagicMock configured as a BattlingMove.
    """
    move_const = _make_mock_move_const(
        base_power=base_power,
        accuracy=accuracy,
        max_pp=max_pp,
        category=category,
        priority=priority,
        pkm_type=pkm_type,
        protect=protect,
        status=status,
        boosts=boosts,
    )
    mock = MagicMock()
    mock.constants = move_const
    mock.pp = pp if pp is not None else max_pp
    mock.disabled = disabled
    return mock


def _make_mock_state(own_active, opp_active, own_reserve=None, opp_reserve=None):
    """Create a mock StateView with specified teams.

    Must use real lists (not MagicMock lists) so that len() and indexing
    behave correctly.

    Args:
        own_active: List of 2 mock Pokemon for own active slots.
        opp_active: List of 2 mock Pokemon for opponent active slots.
        own_reserve: List of up to 2 mock Pokemon for own reserve.
        opp_reserve: List of up to 2 mock Pokemon for opponent reserve.

    Returns:
        MagicMock configured as a StateView.
    """
    if own_reserve is None:
        own_reserve = []
    if opp_reserve is None:
        opp_reserve = []

    state = MagicMock()
    state.weather = Weather.CLEAR
    state.field = Terrain.NONE
    state.trickroom = False

    side0 = MagicMock()
    side0.team.active = own_active
    side0.team.reserve = own_reserve
    side0.conditions.reflect = False
    side0.conditions.lightscreen = False
    side0.conditions.tailwind = False
    side0.conditions.stealth_rock = False
    side0.conditions.poison_spikes = False

    side1 = MagicMock()
    side1.team.active = opp_active
    side1.team.reserve = opp_reserve
    side1.conditions.reflect = False
    side1.conditions.lightscreen = False
    side1.conditions.tailwind = False
    side1.conditions.stealth_rock = False
    side1.conditions.poison_spikes = False

    state.sides = (side0, side1)
    return state


class TestOutputShape:
    """Validates encode_state output dimensions and type."""

    def test_returns_fixed_length_float32(self):
        own = [_make_mock_pkm(hp=150, types=[Type.FIRE]) for _ in range(2)]
        opp = [_make_mock_pkm(hp=100, types=[Type.WATER]) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (FEATURE_DIM,)
        assert result.ndim == 1

    def test_same_shape_with_different_inputs(self):
        own_a = [_make_mock_pkm(hp=200, types=[Type.GRASS]) for _ in range(2)]
        opp_a = [_make_mock_pkm(hp=1, types=[Type.FIRE]) for _ in range(2)]
        state_a = _make_mock_state(own_a, opp_a)

        own_b = [_make_mock_pkm(hp=50, types=[Type.ELECTRIC, Type.FLYING]) for _ in range(2)]
        opp_b = [_make_mock_pkm(hp=300, types=[Type.DRAGON]) for _ in range(2)]
        state_b = _make_mock_state(own_b, opp_b)

        result_a = encode_state(state_a)
        result_b = encode_state(state_b)

        assert result_a.shape == result_b.shape
        assert result_a.shape == (FEATURE_DIM,)

    def test_shape_with_missing_reserve(self):
        own = [_make_mock_pkm(hp=100) for _ in range(2)]
        opp = [_make_mock_pkm(hp=80) for _ in range(2)]
        state = _make_mock_state(own, opp, own_reserve=[], opp_reserve=[])
        result = encode_state(state)
        assert result.shape == (FEATURE_DIM,)


class TestDeterministic:
    """Validates that encode_state returns identical output for the same input."""

    def test_same_object_produces_same_vector(self):
        own = [_make_mock_pkm(hp=120, types=[Type.ICE]) for _ in range(2)]
        opp = [_make_mock_pkm(hp=90, types=[Type.ROCK]) for _ in range(2)]
        state = _make_mock_state(own, opp)

        result_a = encode_state(state)
        result_b = encode_state(state)

        assert np.array_equal(result_a, result_b)

    def test_separate_calls_with_unchanged_state(self):
        own = [_make_mock_pkm(hp=180, types=[Type.DARK]) for _ in range(2)]
        opp = [_make_mock_pkm(hp=60, types=[Type.FAIRY]) for _ in range(2)]
        state = _make_mock_state(own, opp)

        first = encode_state(state)
        second = encode_state(state)
        third = encode_state(state)

        assert np.array_equal(first, second)
        assert np.array_equal(second, third)

    def test_different_state_produces_different_vector(self):
        own_a = [_make_mock_pkm(hp=200, types=[Type.FIRE]) for _ in range(2)]
        opp_a = [_make_mock_pkm(hp=200, types=[Type.FIRE]) for _ in range(2)]
        state_a = _make_mock_state(own_a, opp_a)

        own_b = [_make_mock_pkm(hp=1, types=[Type.WATER]) for _ in range(2)]
        opp_b = [_make_mock_pkm(hp=1, types=[Type.WATER]) for _ in range(2)]
        state_b = _make_mock_state(own_b, opp_b)

        assert not np.array_equal(encode_state(state_a), encode_state(state_b))


class TestFaintedPokemonZeroed:
    """Validates that Pokemon with hp <= 0 get all-zero feature slots."""

    def test_hp_zero_pokemon_is_zeroed(self):
        alive = _make_mock_pkm(hp=150, types=[Type.FIRE])
        fainted = _make_mock_pkm(hp=0, types=[Type.FIRE])
        own = [alive, fainted]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        slot_start = 1 * _POKEMON_FEAT_PER_SLOT
        slot = result[slot_start: slot_start + _POKEMON_FEAT_PER_SLOT]
        assert np.all(slot == 0.0), f"Fainted slot should be all zeros, got max={slot.max()}"

    def test_hp_negative_pokemon_is_zeroed(self):
        alive = _make_mock_pkm(hp=150)
        fainted = _make_mock_pkm(hp=-5)
        own = [alive, fainted]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        slot_start = 1 * _POKEMON_FEAT_PER_SLOT
        slot = result[slot_start: slot_start + _POKEMON_FEAT_PER_SLOT]
        assert np.all(slot == 0.0)

    def test_none_active_slot_is_zeroed(self):
        own = [_make_mock_pkm(hp=150), None]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        slot_start = 1 * _POKEMON_FEAT_PER_SLOT
        slot = result[slot_start: slot_start + _POKEMON_FEAT_PER_SLOT]
        assert np.all(slot == 0.0)


class TestHpFractionRange:
    """Validates that all HP fraction features stay within [0, 1]."""

    def test_hp_features_in_range(self):
        own = [
            _make_mock_pkm(hp=200, max_hp=200),
            _make_mock_pkm(hp=1, max_hp=200),
        ]
        opp = [
            _make_mock_pkm(hp=100, max_hp=100),
            _make_mock_pkm(hp=0, max_hp=100),
        ]
        reserve_own = [_make_mock_pkm(hp=50, max_hp=200)]
        reserve_opp = [_make_mock_pkm(hp=150, max_hp=200)]
        state = _make_mock_state(own, opp, own_reserve=reserve_own, opp_reserve=reserve_opp)
        result = encode_state(state)

        for slot_idx in range(8):
            hp_feat = result[slot_idx * _POKEMON_FEAT_PER_SLOT]
            assert 0.0 <= hp_feat <= 1.0, (
                f"HP fraction {hp_feat} out of range at slot {slot_idx}"
            )

    def test_hp_exceeding_max_is_clamped(self):
        own = [_make_mock_pkm(hp=500, max_hp=200)] + [_make_mock_pkm(hp=100) for _ in range(1)]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        hp_feat = result[0 * _POKEMON_FEAT_PER_SLOT]
        assert hp_feat <= 1.0, f"HP fraction {hp_feat} exceeds 1.0 for over-max HP"


class TestTypeEncodingValid:
    """Validates that all type index features stay within [0, 1]."""

    def test_type_indices_in_range(self):
        own = [
            _make_mock_pkm(hp=150, types=[Type.FIRE, Type.FLYING]),
            _make_mock_pkm(hp=150, types=[Type.WATER]),
        ]
        opp = [
            _make_mock_pkm(hp=150, types=[Type.DRAGON, Type.GROUND]),
            _make_mock_pkm(hp=150, types=[Type.STEEL]),
        ]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        for slot_idx in range(8):
            t0 = result[slot_idx * _POKEMON_FEAT_PER_SLOT + 2]
            t1 = result[slot_idx * _POKEMON_FEAT_PER_SLOT + 3]
            assert 0.0 <= t0 <= 1.0, f"Type0 {t0} out of range at slot {slot_idx}"
            assert 0.0 <= t1 <= 1.0, f"Type1 {t1} out of range at slot {slot_idx}"

    def test_single_typed_second_is_zero(self):
        own = [_make_mock_pkm(hp=150, types=[Type.FIRE])]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        t0 = result[0 * _POKEMON_FEAT_PER_SLOT + 2]
        t1 = result[0 * _POKEMON_FEAT_PER_SLOT + 3]
        assert t0 == pytest.approx(Type.FIRE.value / 18.0)
        assert t1 == 0.0

    def test_move_type_in_range(self):
        fire_move = _make_mock_battling_move(pkm_type=Type.FIRE)
        water_move = _make_mock_battling_move(pkm_type=Type.WATER)
        own = [
            _make_mock_pkm(hp=150, moves=[fire_move, water_move, fire_move, water_move]),
            _make_mock_pkm(hp=150, moves=[water_move] * 4),
        ]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        pokemon_offset = 8 * _POKEMON_FEAT_PER_SLOT
        for move_idx in range(8):
            move_type = result[pokemon_offset + move_idx * _MOVE_FEAT_PER_SLOT + 7]
            assert 0.0 <= move_type <= 1.0, (
                f"Move type {move_type} out of range at move {move_idx}"
            )


class TestBoostNormalization:
    """Validates that all stat boost features stay within [-1, 1]."""

    def test_boosts_in_normalized_range(self):
        own = [_make_mock_pkm(hp=150, boosts=[0, 6, 0, 0, -6, 6, 0, 0])]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        base = 0 * _POKEMON_FEAT_PER_SLOT + 11
        for b_idx in range(5):
            val = result[base + b_idx]
            assert -1.0 <= val <= 1.0, f"Boost {b_idx} value {val} out of range"

    def test_boosts_clamped_to_range(self):
        own = [_make_mock_pkm(hp=150, boosts=[0, 10, -10, 0, 0, 0, 0, 0])]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        atk = result[0 * _POKEMON_FEAT_PER_SLOT + 11]
        def_boost = result[0 * _POKEMON_FEAT_PER_SLOT + 12]
        assert -1.0 <= atk <= 1.0
        assert -1.0 <= def_boost <= 1.0


class TestNoNanInf:
    """Validates that encode_state never produces NaN or Inf values."""

    def test_no_nan_or_inf_in_output(self):
        own = [_make_mock_pkm(hp=150, types=[Type.GHOST]) for _ in range(2)]
        opp = [_make_mock_pkm(hp=100, types=[Type.PSYCHIC]) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        assert not np.any(np.isnan(result)), "Output contains NaN"
        assert not np.any(np.isinf(result)), "Output contains Inf"

    def test_edge_case_zero_max_hp(self):
        own = [_make_mock_pkm(hp=0, max_hp=0)] + [_make_mock_pkm(hp=100) for _ in range(1)]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        assert not np.any(np.isnan(result)), "Output contains NaN with zero max_hp"
        assert not np.any(np.isinf(result)), "Output contains Inf with zero max_hp"

    def test_extreme_boost_values(self):
        own = [_make_mock_pkm(hp=150, boosts=[0, 100, -100, 50, -50, 0, 0, 0])]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]
        state = _make_mock_state(own, opp)
        result = encode_state(state)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestSymmetricSides:
    """Validates that swapping sides produces mirrored encoding."""

    def test_swapping_sides_swaps_pokemon_feature_blocks(self):
        own_hp = [250, 200]
        opp_hp = [50, 75]
        own = [_make_mock_pkm(hp=own_hp[i], max_hp=300) for i in range(2)]
        opp = [_make_mock_pkm(hp=opp_hp[i], max_hp=300) for i in range(2)]

        s0 = MagicMock()
        s0.team.active = own
        s0.team.reserve = []
        s0.conditions.reflect = True
        s0.conditions.lightscreen = False
        s0.conditions.tailwind = False
        s0.conditions.stealth_rock = True
        s0.conditions.poison_spikes = False

        s1 = MagicMock()
        s1.team.active = opp
        s1.team.reserve = []
        s1.conditions.reflect = False
        s1.conditions.lightscreen = True
        s1.conditions.tailwind = False
        s1.conditions.stealth_rock = False
        s1.conditions.poison_spikes = True

        state_original = MagicMock()
        state_original.sides = (s0, s1)
        state_original.weather = Weather.RAIN
        state_original.field = Terrain.GRASSY_TERRAIN
        state_original.trickroom = True

        state_swapped = MagicMock()
        state_swapped.sides = (s1, s0)
        state_swapped.weather = Weather.RAIN
        state_swapped.field = Terrain.GRASSY_TERRAIN
        state_swapped.trickroom = True

        original = encode_state(state_original)
        swapped = encode_state(state_swapped)

        own_start = 0
        opp_start = 4 * _POKEMON_FEAT_PER_SLOT

        own_block_orig = original[own_start: own_start + 4 * _POKEMON_FEAT_PER_SLOT]
        opp_block_orig = original[opp_start: opp_start + 4 * _POKEMON_FEAT_PER_SLOT]
        own_block_swap = swapped[own_start: own_start + 4 * _POKEMON_FEAT_PER_SLOT]
        opp_block_swap = swapped[opp_start: opp_start + 4 * _POKEMON_FEAT_PER_SLOT]

        assert np.array_equal(own_block_orig, opp_block_swap), (
            "Original own Pokemon should match swapped opponent Pokemon"
        )
        assert np.array_equal(opp_block_orig, own_block_swap), (
            "Original opponent Pokemon should match swapped own Pokemon"
        )

    def test_global_features_unchanged_on_swap(self):
        own = [_make_mock_pkm(hp=100) for _ in range(2)]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]

        s0 = MagicMock()
        s0.team.active = own
        s0.team.reserve = []
        s0.conditions.reflect = False
        s0.conditions.lightscreen = False
        s0.conditions.tailwind = False
        s0.conditions.stealth_rock = False
        s0.conditions.poison_spikes = False

        s1 = MagicMock()
        s1.team.active = opp
        s1.team.reserve = []
        s1.conditions.reflect = False
        s1.conditions.lightscreen = False
        s1.conditions.tailwind = False
        s1.conditions.stealth_rock = False
        s1.conditions.poison_spikes = False

        state_a = MagicMock()
        state_a.sides = (s0, s1)
        state_a.weather = Weather.SAND
        state_a.field = Terrain.ELECTRIC_TERRAIN
        state_a.trickroom = False

        state_b = MagicMock()
        state_b.sides = (s1, s0)
        state_b.weather = Weather.SAND
        state_b.field = Terrain.ELECTRIC_TERRAIN
        state_b.trickroom = False

        global_start = 8 * _POKEMON_FEAT_PER_SLOT + 8 * _MOVE_FEAT_PER_SLOT

        result_a = encode_state(state_a)
        result_b = encode_state(state_b)

        global_a = result_a[global_start: global_start + 11]
        global_b = result_b[global_start: global_start + 11]
        assert np.array_equal(global_a, global_b)

    def test_side_conditions_swap_correctly(self):
        own = [_make_mock_pkm(hp=100) for _ in range(2)]
        opp = [_make_mock_pkm(hp=100) for _ in range(2)]

        s0 = MagicMock()
        s0.team.active = own
        s0.team.reserve = []
        s0.conditions.reflect = True
        s0.conditions.lightscreen = False
        s0.conditions.tailwind = True
        s0.conditions.stealth_rock = False
        s0.conditions.poison_spikes = True

        s1 = MagicMock()
        s1.team.active = opp
        s1.team.reserve = []
        s1.conditions.reflect = False
        s1.conditions.lightscreen = True
        s1.conditions.tailwind = False
        s1.conditions.stealth_rock = True
        s1.conditions.poison_spikes = False

        state_a = MagicMock()
        state_a.sides = (s0, s1)
        state_a.weather = Weather.CLEAR
        state_a.field = Terrain.NONE
        state_a.trickroom = False

        state_b = MagicMock()
        state_b.sides = (s1, s0)
        state_b.weather = Weather.CLEAR
        state_b.field = Terrain.NONE
        state_b.trickroom = False

        cond_start = 8 * _POKEMON_FEAT_PER_SLOT + 8 * _MOVE_FEAT_PER_SLOT + 11

        result_a = encode_state(state_a)
        result_b = encode_state(state_b)

        cond_a_own = result_a[cond_start: cond_start + 5]
        cond_a_opp = result_a[cond_start + 5: cond_start + 10]
        cond_b_own = result_b[cond_start: cond_start + 5]
        cond_b_opp = result_b[cond_start + 5: cond_start + 10]

        assert np.array_equal(cond_a_own, cond_b_opp)
        assert np.array_equal(cond_a_opp, cond_b_own)


class TestSmokeRealState:
    """Smoke test using a real BattleEngine and generated teams."""

    def test_real_battle_state_produces_valid_vector(self):
        from vgc2.battle_engine import BattleEngine, BattleRuleParam
        from vgc2.battle_engine.game_state import State, get_battle_teams
        from vgc2.battle_engine.view import StateView, TeamView
        from vgc2.util.generator import gen_move_set, gen_pkm_roster, gen_team

        move_set = gen_move_set(200)
        gen_pkm_roster(30, move_set)
        rng = np.random.default_rng(42)
        team_a = gen_team(4, 4, rng)
        team_b = gen_team(4, 4, rng)

        params = BattleRuleParam()

        battle_teams = get_battle_teams((team_a, team_b), 2)
        state_obj = State(battle_teams)
        rng_tuple = ((rng, rng), (rng, rng))
        engine = BattleEngine(
            state_obj, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )

        view_a = TeamView(team_a)
        view_b = TeamView(team_b)
        sv = StateView(engine.state, 0, (view_a, view_b))

        result = encode_state(sv)

        assert result.shape == (FEATURE_DIM,)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        print(f"\n  Smoke test passed: shape={result.shape}, "
              f"min={result.min():.4f}, max={result.max():.4f}, "
              f"positives={(result > 0).sum()}/{len(result)}")
