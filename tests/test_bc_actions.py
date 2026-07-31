"""Tests for BC action encoder: encode, decode, and valid-action masking."""

from unittest.mock import MagicMock

import pytest

from src.tree_bc.actions import (
    JOINT_ACTION_COUNT,
    PER_POKEMON_ACTIONS,
    decode_action,
    encode_action,
    get_valid_actions,
)


def _make_mock_pkm(hp=200, moves=None, reserve=None):
    """Create a mock BattlingPokemonView with battling moves.

    Args:
        hp: Current HP.
        moves: List of mock BattlingMove objects (default: 4 generic moves).
        reserve: List of mock reserve Pokemon (for switch tests).

    Returns:
        MagicMock configured as a BattlingPokemonView.
    """
    if moves is None:
        moves = [_make_mock_move() for _ in range(4)]

    pkm = MagicMock()
    pkm.hp = hp
    pkm.battling_moves = moves

    if reserve is not None:
        pkm.reserve = reserve

    return pkm


def _make_mock_move(pp=16, disabled=False):
    """Create a mock BattlingMove.

    Args:
        pp: Current PP.
        disabled: Whether the move is disabled.

    Returns:
        MagicMock configured as a BattlingMove.
    """
    move = MagicMock()
    move.pp = pp
    move.disabled = disabled
    return move


def _make_mock_state(own_active=None, opp_active=None, own_reserve=None):
    """Create a mock StateView with explicit side mocks.

    Builds sides from the inside out to avoid MagicMock auto-generation
    replacing explicitly-set child objects.

    Args:
        own_active: List of 0-2 mock Pokemon for own active slots.
        opp_active: List of 0-2 mock Pokemon for opponent active slots.
        own_reserve: List of 0-2 mock Pokemon for own reserve bench.

    Returns:
        MagicMock configured as a StateView with sides[0] = own, sides[1] = opp.
    """
    if own_active is None:
        own_active = []
    if opp_active is None:
        opp_active = []
    if own_reserve is None:
        own_reserve = []

    side0 = MagicMock()
    side0.team.active = own_active
    side0.team.reserve = own_reserve

    side1 = MagicMock()
    side1.team.active = opp_active

    state = MagicMock()
    state.sides = (side0, side1)
    return state


class TestEncodeAction:
    """Validates encode_action maps commands to correct indices."""

    def test_move_actions_encode_correctly(self):
        commands = [(0, 1), (2, 0)]
        action_idx = encode_action(commands)
        expected = (0 * 2 + 1) * PER_POKEMON_ACTIONS + (2 * 2 + 0)
        assert action_idx == expected
        assert action_idx == 1 * 10 + 4

    def test_switch_actions_encode_correctly(self):
        commands = [(-1, 0), (-1, 1)]
        action_idx = encode_action(commands)
        expected = (8 + 0) * PER_POKEMON_ACTIONS + (8 + 1)
        assert action_idx == expected
        assert action_idx == 8 * 10 + 9

    def test_mixed_move_and_switch(self):
        commands = [(0, 0), (-1, 1)]
        action_idx = encode_action(commands)
        assert action_idx == 0 * 10 + 9

    def test_single_command_pads_with_zero(self):
        action_idx = encode_action([(3, 0)])
        assert action_idx == 6 * 10 + 0

    def test_empty_commands_raises(self):
        with pytest.raises(ValueError):
            encode_action([])

    def test_all_three_switch_targets(self):
        assert encode_action([(-1, 0), (-1, 1)]) == 8 * 10 + 9
        assert encode_action([(-1, 1), (-1, 0)]) == 9 * 10 + 8


class TestDecodeAction:
    """Validates decode_action maps indices back to correct commands."""

    def test_move_action_roundtrip(self):
        commands = [(1, 1), (3, 0)]
        encoded = encode_action(commands)
        decoded = decode_action(encoded)
        assert decoded == commands

    def test_switch_action_roundtrip(self):
        commands = [(-1, 0), (-1, 1)]
        encoded = encode_action(commands)
        decoded = decode_action(encoded)
        assert decoded == commands

    def test_mixed_roundtrip(self):
        commands = [(2, 0), (-1, 1)]
        encoded = encode_action(commands)
        decoded = decode_action(encoded)
        assert decoded == commands

    def test_all_100_actions_are_reachable(self):
        for idx in range(JOINT_ACTION_COUNT):
            commands = decode_action(idx)
            assert 0 <= encode_action(commands) < JOINT_ACTION_COUNT

    def test_all_100_actions_roundtrip(self):
        for idx in range(JOINT_ACTION_COUNT):
            commands = decode_action(idx)
            re_encoded = encode_action(commands)
            assert re_encoded == idx, (
                f"Roundtrip failed at index {idx}: "
                f"decode={commands}, re_encode={re_encoded}"
            )

    def test_decode_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            decode_action(-1)
        with pytest.raises(ValueError):
            decode_action(JOINT_ACTION_COUNT)

    def test_all_decoded_commands_are_in_valid_range(self):
        for idx in range(JOINT_ACTION_COUNT):
            commands = decode_action(idx)
            assert len(commands) == 2
            for cmd in commands:
                action_val, target_val = cmd
                if action_val == -1:
                    assert 0 <= target_val <= 1
                else:
                    assert 0 <= action_val <= 3
                    assert 0 <= target_val <= 1


class TestGetValidActions:
    """Validates get_valid_actions correctly filters illegal actions."""

    def test_all_actions_valid_for_normal_state(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        assert len(valid) > 0
        assert max(valid) < JOINT_ACTION_COUNT
        assert min(valid) >= 0

    def test_cannot_target_fainted_opponent(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=200), _make_mock_pkm(hp=0)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            for cmd in commands:
                action_val, target_val = cmd
                if action_val != -1:
                    assert target_val != 1, (
                        f"Action {action_idx} targets fainted opponent slot 1"
                    )

    def test_cannot_target_both_fainted_opponents(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=0), _make_mock_pkm(hp=0)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            for cmd in commands:
                action_val, target_val = cmd
                if action_val != -1 and target_val not in (0, 1):
                    pytest.fail(f"Action {action_idx} targets invalid slot {target_val}")

    def test_zero_pp_move_excluded(self):
        moves_pkm0 = [_make_mock_move(pp=0), _make_mock_move(pp=16),
                       _make_mock_move(pp=16), _make_mock_move(pp=16)]
        own = [_make_mock_pkm(hp=200, moves=moves_pkm0), _make_mock_pkm(hp=200)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            pkm0_cmd = commands[0]
            if pkm0_cmd[0] != -1:
                assert pkm0_cmd[0] != 0, (
                    f"Action {action_idx} uses move 0 (0 PP) on Pokemon 0"
                )

    def test_disabled_move_excluded(self):
        moves_pkm0 = [_make_mock_move(pp=16, disabled=True), _make_mock_move(pp=16),
                       _make_mock_move(pp=16), _make_mock_move(pp=16)]
        own = [_make_mock_pkm(hp=200, moves=moves_pkm0), _make_mock_pkm(hp=200)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            pkm0_cmd = commands[0]
            if pkm0_cmd[0] != -1:
                assert pkm0_cmd[0] != 0, (
                    f"Action {action_idx} uses disabled move 0 on Pokemon 0"
                )

    def test_single_active_reduces_action_space(self):
        own = [_make_mock_pkm(hp=200), _make_mock_pkm(hp=0)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            assert commands[1] == (0, 0), (
                f"Fainted Pokemon 1 should have dummy command (0,0), got {commands[1]}"
            )

    def test_both_fainted_returns_fallback(self):
        own = [_make_mock_pkm(hp=0), _make_mock_pkm(hp=0)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)

        assert len(valid) >= 1

    def test_switch_actions_included_when_reserve_available(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        own_reserve = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp, own_reserve=own_reserve)
        valid = get_valid_actions(state)

        has_switch = False
        for action_idx in valid:
            commands = decode_action(action_idx)
            for cmd in commands:
                if cmd[0] == -1:
                    has_switch = True
        assert has_switch, "No switch actions found when reserve Pokemon are available"

    def test_switch_excluded_when_reserve_fainted(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        own_reserve = [_make_mock_pkm(hp=0), _make_mock_pkm(hp=0)]
        state = _make_mock_state(own_active=own, opp_active=opp, own_reserve=own_reserve)
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            for cmd in commands:
                assert cmd[0] != -1, (
                    f"Action {action_idx} has switch but all reserves are fainted"
                )

    def test_no_reserve_excludes_switches(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp, own_reserve=[])
        valid = get_valid_actions(state)

        for action_idx in valid:
            commands = decode_action(action_idx)
            for cmd in commands:
                assert cmd[0] != -1, (
                    f"Action {action_idx} has switch but no reserve exists"
                )

    def test_valid_actions_are_sorted(self):
        own = [_make_mock_pkm(hp=200) for _ in range(2)]
        opp = [_make_mock_pkm(hp=200) for _ in range(2)]
        state = _make_mock_state(own_active=own, opp_active=opp)
        valid = get_valid_actions(state)
        assert valid == sorted(valid)


class TestSmokeStateView:
    """Smoke test using a real BattleEngine and generated teams."""

    def test_real_state_produces_valid_actions(self):
        from vgc2.battle_engine import BattleEngine, BattleRuleParam
        from vgc2.battle_engine.game_state import State, get_battle_teams
        from vgc2.battle_engine.view import StateView, TeamView
        from vgc2.util.generator import gen_move_set, gen_pkm_roster, gen_team

        move_set = gen_move_set(200)
        gen_pkm_roster(30, move_set)
        rng = __import__("numpy").random.default_rng(42)
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

        valid = get_valid_actions(sv)

        assert len(valid) > 0
        assert max(valid) < JOINT_ACTION_COUNT

        for action_idx in valid[:5]:
            commands = decode_action(action_idx)
            re_encoded = encode_action(commands)
            assert re_encoded == action_idx

        print(f"\n  Smoke test passed: {len(valid)} valid actions "
              f"out of {JOINT_ACTION_COUNT} total")
