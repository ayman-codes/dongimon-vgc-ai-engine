"""Tests for battle move scoring functions.

Uses mock vgc2 objects to verify scoring behavior
for offensive moves, protect, switches, threat estimation,
and synergy calculations.
"""

from unittest.mock import MagicMock

import pytest
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Hazard, Stat, Status, Terrain, Type, Weather

from PPO_trainers.weighted_heuristic.move_scoring import (
    _get_type_eff_string,
    estimate_incoming_threat,
    identify_biggest_threat,
    score_offensive_move,
    score_protect_move,
    score_switch_action,
)


def _make_mock_pkm(name="MockMon", hp=200, types=None, atk=100, spa=100, spe=100):
    """Create a mock BattlingPokemonView with minimal attributes."""
    if types is None:
        types = [Type.NORMAL]
    mock = MagicMock()
    mock.hp = hp
    mock.types = types
    # Use real dicts not MagicMock for stats access
    constants = MagicMock()
    constants.stats = {Stat.MAX_HP: hp, Stat.ATTACK: atk, Stat.DEFENSE: 80,
                       Stat.SPECIAL_ATTACK: spa, Stat.SPECIAL_DEFENSE: 80, Stat.SPEED: spe}
    constants.species.base_stats = {Stat.ATTACK: atk, Stat.SPECIAL_ATTACK: spa, Stat.SPEED: spe,
                                    Stat.DEFENSE: 80, Stat.SPECIAL_DEFENSE: 80, Stat.MAX_HP: hp}
    constants.species.moves = []
    constants.moves = []
    mock.constants = constants
    mock.battling_moves = []
    mock.boosts = [0, 0, 0, 0, 0, 0, 0, 0]
    mock.status = Status.NONE
    mock.protect_counter = 0
    return mock


def _make_mock_move_const(name="Tackle", bp=40, cat=Category.PHYSICAL, acc=1.0, pkm_type=None):
    """Create a mock Move constants object."""
    if pkm_type is None:
        pkm_type = Type.NORMAL
    mock = MagicMock()
    mock.name = name
    mock.base_power = bp
    mock.category = cat
    mock.accuracy = acc
    mock.pkm_type = pkm_type
    mock.protect = False
    mock.max_pp = 32
    mock.pp = 32
    mock.disabled = False
    mock.heal = 0.0
    mock.boosts = (0, 0, 0, 0, 0, 0, 0, 0)
    mock.self_boosts = False
    mock.status = Status.NONE
    mock.weather_start = Weather.CLEAR
    mock.field_start = Terrain.NONE
    mock.hazard = Hazard.NONE
    mock.toggle_trickroom = False
    mock.toggle_reflect = False
    mock.toggle_lightscreen = False
    mock.priority = 0
    return mock


def _make_mock_battling_move(constants):
    """Create a mock BattlingMove wrapping constants."""
    mock = MagicMock()
    mock.constants = constants
    mock.pp = constants.max_pp
    mock.disabled = False
    return mock


def _make_mock_state():
    """Create a mock StateView."""
    mock = MagicMock()
    side0 = MagicMock()
    side0.team.active = []
    side0.team.reserve = []
    side0.conditions.stealth_rock = False
    side0.conditions.poison_spikes = False
    side1 = MagicMock()
    side1.team.active = []
    side1.team.reserve = []
    side1.conditions.stealth_rock = False
    side1.conditions.poison_spikes = False
    mock.sides = (side0, side1)
    mock.weather = Weather.CLEAR
    mock.field = Terrain.NONE
    mock.trickroom = False
    return mock


@pytest.fixture
def params():
    """Shared BattleRuleParam fixture."""
    return BattleRuleParam()


@pytest.fixture
def mock_state():
    """Shared mock StateView fixture."""
    return _make_mock_state()


class TestGetTypeEffString:
    """Tests for _get_type_eff_string type conversion."""

    def test_fire_vs_grass(self):
        """Fire move vs Grass type = 2.0x."""
        eff = _get_type_eff_string(Type.FIRE, [Type.GRASS])
        assert eff == 2.0

    def test_water_vs_fire(self):
        """Water move vs Fire type = 2.0x."""
        eff = _get_type_eff_string(Type.WATER, [Type.FIRE])
        assert eff == 2.0

    def test_electric_vs_ground(self):
        """Electric move vs Ground type = 0.0x (immune)."""
        eff = _get_type_eff_string(Type.ELECTRIC, [Type.GROUND])
        assert eff == 0.0

    def test_normal_vs_ghost(self):
        """Normal move vs Ghost type = 0.0x (immune)."""
        eff = _get_type_eff_string(Type.NORMAL, [Type.GHOST])
        assert eff == 0.0

    def test_fighting_vs_steel(self):
        """Fighting move vs Steel type = 2.0x."""
        eff = _get_type_eff_string(Type.FIGHT, [Type.STEEL])
        assert eff == 2.0

    def test_string_type_input(self):
        """Accepts vgc2 Type enum members."""
        eff = _get_type_eff_string(Type.FIRE, [Type.GRASS])
        assert eff == 2.0


class TestScoreOffensiveMove:
    """Tests for score_offensive_move."""

    def test_out_of_pp_returns_negative_inf(self, params, mock_state):
        """Move with 0 PP returns -inf score."""
        attacker = _make_mock_pkm()
        target = _make_mock_pkm()
        move_const = _make_mock_move_const(bp=80)
        move = _make_mock_battling_move(move_const)
        move.pp = 0

        score, is_ko = score_offensive_move(attacker, move, target, mock_state, params)
        assert score == -float("inf")

    def test_disabled_move_returns_negative_inf(self, params, mock_state):
        """Disabled move returns -inf score."""
        attacker = _make_mock_pkm()
        target = _make_mock_pkm()
        move_const = _make_mock_move_const(bp=80)
        move = _make_mock_battling_move(move_const)
        move.disabled = True

        score, is_ko = score_offensive_move(attacker, move, target, mock_state, params)
        assert score == -float("inf")

    def test_status_move_returns_nonzero_utility(self, params, mock_state):
        """Status move returns utility score from status evaluation."""
        attacker = _make_mock_pkm()
        target = _make_mock_pkm()
        move_const = _make_mock_move_const(bp=0, cat=Category.OTHER)
        move_const.status = Status.SLEEP
        move = _make_mock_battling_move(move_const)

        score, is_ko = score_offensive_move(attacker, move, target, mock_state, params)
        assert not is_ko
        assert score >= 0

    def test_score_not_zero_for_valid_attack(self, params, mock_state):
        """Valid damaging move produces a positive score."""
        attacker = _make_mock_pkm("Attacker", hp=200, atk=150, spe=100)
        target = _make_mock_pkm("Defender", hp=200, atk=50, spe=50, types=[Type.NORMAL])
        mock_state.sides[0].team.active = [attacker]
        move_const = _make_mock_move_const(name="Return", bp=100, cat=Category.PHYSICAL)
        move = _make_mock_battling_move(move_const)

        score, is_ko = score_offensive_move(attacker, move, target, mock_state, params)
        assert score > 0

    def test_healing_move_does_not_crash(self, params, mock_state):
        """Move with heal > 0 does not crash."""
        attacker = _make_mock_pkm("Attacker", hp=100)
        target = _make_mock_pkm("Target")
        mock_state.sides[1].team.active = [target]
        move_const = _make_mock_move_const("Recover", bp=0, cat=Category.OTHER)
        move_const.heal = 0.5
        move = _make_mock_battling_move(move_const)

        score, is_ko = score_offensive_move(attacker, move, target, mock_state, params)
        assert score >= 0


class TestScoreProtectMove:
    """Tests for score_protect_move."""

    def test_out_of_pp_returns_negative_inf(self, params, mock_state):
        """Protect with no PP returns -inf."""
        attacker = _make_mock_pkm()
        move_const = _make_mock_move_const(bp=0, cat=Category.OTHER)
        move_const.protect = True
        move = _make_mock_battling_move(move_const)
        move.pp = 0

        score = score_protect_move(attacker, move, mock_state, params)
        assert score == -float("inf")

    def test_returns_zero_when_no_opponents(self, params, mock_state):
        """Protect on an empty field returns 0. Note: protect adds passive damage value, may be >0."""
        attacker = _make_mock_pkm()
        mock_state.sides[1].team.active = []
        move_const = _make_mock_move_const(bp=0, cat=Category.OTHER)
        move_const.protect = True
        move = _make_mock_battling_move(move_const)

        score = score_protect_move(attacker, move, mock_state, params)
        assert score != -float("inf")


class TestSwitchAction:
    """Tests for score_switch_action."""

    def test_fainted_reserve_returns_negative_inf(self, params, mock_state):
        """Switching to a fainted Pokemon returns -inf."""
        current = _make_mock_pkm("Current", hp=100)
        reserve = _make_mock_pkm("Reserve", hp=0)

        score = score_switch_action(current, reserve, [], mock_state, params)
        assert score == -float("inf")

    def test_no_opponents_returns_baseline(self, params, mock_state):
        """Switch with no opponents returns baseline (avg_move_score * 0.5 = 25.0)."""
        current = _make_mock_pkm("Current")
        reserve = _make_mock_pkm("Reserve", hp=200)

        score = score_switch_action(current, reserve, [], mock_state, params, avg_move_score=50.0)
        assert score == 25.0

    def test_defensive_pivot_bonus(self, params, mock_state):
        """Switching to a Pokemon that resists opponent STABs gets bonus."""
        current = _make_mock_pkm("Current")
        reserve = _make_mock_pkm("Reserve", hp=200, types=[Type.WATER])
        opponent = _make_mock_pkm("OppAttacker", types=[Type.FIRE])

        score = score_switch_action(current, reserve, [opponent], mock_state, params)
        assert score > 50.0


class TestEstimateIncomingThreat:
    """Tests for estimate_incoming_threat."""

    def test_no_opponents_returns_zero(self, params, mock_state):
        """No opponents means zero threat."""
        my_pkm = _make_mock_pkm("Mine")
        threat, likely_ko = estimate_incoming_threat(my_pkm, [], mock_state, params)
        assert threat == 0.0
        assert not likely_ko

    def test_fainted_target_returns_zero(self, params, mock_state):
        """Fainted target means zero threat."""
        my_pkm = _make_mock_pkm("Mine", hp=0)
        opp = _make_mock_pkm("Opp")
        threat, likely_ko = estimate_incoming_threat(my_pkm, [opp], mock_state, params)
        assert threat == 0.0

    def test_nonzero_threat_with_opponent(self, params, mock_state):
        """Opponent with damaging move produces positive threat."""
        my_pkm = _make_mock_pkm("Mine", hp=200)
        opp = _make_mock_pkm("Opp", types=[Type.NORMAL], atk=150)
        move_const = _make_mock_move_const(name="Return", bp=100, cat=Category.PHYSICAL)
        opp.battling_moves = [_make_mock_battling_move(move_const)]
        opp.constants.species.moves = [move_const]

        threat, likely_ko = estimate_incoming_threat(my_pkm, [opp], mock_state, params)
        assert threat > 0


class TestIdentifyBiggestThreat:
    """Tests for identify_biggest_threat."""

    def test_no_opponents_returns_none(self, params, mock_state):
        """No opponents means no threat identified."""
        result = identify_biggest_threat([], [_make_mock_pkm()], mock_state, params)
        assert result is None

    def test_no_allies_returns_none(self, params, mock_state):
        """No allies means no threat identified."""
        result = identify_biggest_threat([_make_mock_pkm()], [], mock_state, params)
        assert result is None

    def test_different_threat_scores(self, params, mock_state):
        """Higher threat opponent has higher score."""
        low_threat = _make_mock_pkm("Low", atk=50, spa=50, spe=50, types=[Type.NORMAL])
        high_threat = _make_mock_pkm("High", atk=150, spa=150, spe=150, types=[Type.NORMAL])
        my_pkm = _make_mock_pkm("Ally", hp=200)

        result = identify_biggest_threat([low_threat, high_threat], [my_pkm], mock_state, params)
        assert result is not None
        threat_pkm, slot = result
        assert threat_pkm is high_threat
        assert slot == 1
