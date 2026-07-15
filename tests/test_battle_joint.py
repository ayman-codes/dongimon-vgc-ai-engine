"""Tests for joint action pairing logic."""

from unittest.mock import MagicMock

from vgc2.battle_engine.modifiers import Category, Stat, Type, Weather

from src.battle.joint import (
    _env_synergy,
    _is_setup_move,
    _off_def_support,
    _setup_synergy,
    _survival_impact,
    _target_priority,
)


def _make_mock_move_const(protect=False, cat=Category.PHYSICAL, bp=80, pkm_type=None,
                          boosts=None, weather=Weather.CLEAR, tr=False, score=200):
    if pkm_type is None:
        pkm_type = Type.NORMAL
    if boosts is None:
        boosts = (0, 0, 0, 0, 0, 0, 0, 0)
    mock = MagicMock()
    mock.protect = protect
    mock.category = cat
    mock.base_power = bp
    mock.pkm_type = pkm_type
    mock.boosts = boosts
    mock.self_boosts = False
    mock.weather_start = weather
    mock.field_start = 0
    mock.toggle_trickroom = tr
    mock.priority = 0
    return mock


def _make_mock_pkm_view(hp=200, spe=100, types=None):
    if types is None:
        types = [Type.NORMAL]
    mock = MagicMock()
    mock.hp = hp
    mock.types = types
    mock.constants.stats = {Stat.SPEED: spe, Stat.MAX_HP: hp}
    mock.boosts = [0, 0, 0, 0, 0, 0, 0, 0]
    mock.constants.species.ability = None
    return mock


def _make_mock_state():
    mock = MagicMock()
    mock.weather = Weather.CLEAR
    mock.field = 0
    mock.trickroom = False
    return mock


class TestIsSetupMove:
    """Tests for _is_setup_move."""

    def test_status_is_setup(self):
        """A status move with high enough score is setup."""
        move = _make_mock_move_const(cat=Category.OTHER, bp=0)
        assert _is_setup_move(True, move, 200, False)

    def test_low_score_not_setup(self):
        """Low score means not setup even if status."""
        move = _make_mock_move_const(cat=Category.OTHER, bp=0)
        assert not _is_setup_move(True, move, 50, False)

    def test_high_bp_not_setup(self):
        """High BP damaging move is not setup."""
        move = _make_mock_move_const(bp=100)
        assert not _is_setup_move(True, move, 200, False)

    def test_low_bp_with_boosts_is_setup(self):
        """Low BP move with stat boosts is setup."""
        move = _make_mock_move_const(bp=30, boosts=(0, 1, 0, 0, 0, 0, 0, 0))
        assert _is_setup_move(True, move, 200, False)

    def test_trick_room_is_setup(self):
        """Trick Room move counts as setup."""
        move = _make_mock_move_const(bp=0, cat=Category.OTHER, tr=True)
        assert _is_setup_move(True, move, 200, False)

    def test_protect_not_setup(self):
        """Protect is not a setup move."""
        move = _make_mock_move_const(protect=True)
        assert not _is_setup_move(True, move, 200, False)


class TestSurvivalImpact:
    """Tests for _survival_impact."""

    def test_no_threat_no_penalty(self):
        """Zero threat means zero penalty."""
        penalty = _survival_impact(
            _make_mock_pkm_view(hp=200), 0.0, True,
            _make_mock_move_const(), False, None, -1, None, -1, 1000.0,
        )
        assert penalty == 0.0

    def test_switch_eliminates_threat(self):
        """Switching out resets threat to zero."""
        pkm = _make_mock_pkm_view(hp=200)
        penalty = _survival_impact(
            pkm, 150.0, False, None, False, None, -1, None, -1, 1000.0,
        )
        assert penalty == 0.0

    def test_protect_eliminates_threat(self):
        """Using Protect negates incoming threat."""
        pkm = _make_mock_pkm_view(hp=200)
        move = _make_mock_move_const(protect=True)
        penalty = _survival_impact(
            pkm, 150.0, True, move, False, None, -1, None, -1, 1000.0,
        )
        assert penalty == 0.0

    def test_lethal_threat_causes_large_penalty(self):
        """Threat exceeding HP causes large negative penalty."""
        pkm = _make_mock_pkm_view(hp=100)
        penalty = _survival_impact(
            pkm, 200.0, True, _make_mock_move_const(), False, None, -1, None, -1, 1000.0,
        )
        assert penalty < 0

    def test_partial_damage_causes_proportional_penalty(self):
        """Partial HP threat causes scaled penalty."""
        pkm = _make_mock_pkm_view(hp=200)
        penalty = _survival_impact(
            pkm, 50.0, True, _make_mock_move_const(), False, None, -1, None, -1, 1000.0,
        )
        assert penalty < 0


class TestTargetPriority:
    """Tests for _target_priority."""

    def test_no_threat_no_bonus(self):
        """No biggest threat means zero bonus."""
        state = _make_mock_state()
        bonus = _target_priority(
            False, False, None, None, -1, -1, True, True,
            None, -1, [], state, MagicMock(),
        )
        assert bonus == 0.0

    def test_koing_threat_gives_bonus(self):
        """KOing the biggest threat gives positive bonus."""
        state = _make_mock_state()
        threat = _make_mock_pkm_view(hp=100)
        move = _make_mock_move_const(bp=80)
        bonus = _target_priority(
            True, False, move, None, 0, -1, True, False,
            threat, 0, [], state, MagicMock(),
        )
        assert bonus == 450.0


class TestOffDefSupport:
    """Tests for _off_def_support."""

    def test_no_move_no_bonus(self):
        """No move constants means zero bonus."""
        bonus = _off_def_support(True, None, 400, True, None, 400, False, False)
        assert bonus == 0.0

    def test_attack_plus_protect_gives_bonus(self):
        """Strong attack + good protect pairing gets bonus."""
        move_off = _make_mock_move_const(protect=False, bp=100)
        move_def = _make_mock_move_const(protect=True)
        bonus = _off_def_support(True, move_off, 400, True, move_def, 150, False, True)
        assert bonus > 0


class TestSetupSynergy:
    """Tests for _setup_synergy."""

    def test_no_move_no_bonus(self):
        """No move constants means zero bonus."""
        bonus = _setup_synergy(True, None, 200, False, True, None, 200, False)
        assert bonus == 0.0

    def test_setup_plus_attack_gives_bonus(self):
        """Setup move + strong attack follow-up gets bonus."""
        setup_move = _make_mock_move_const(cat=Category.OTHER, bp=0)
        attack_move = _make_mock_move_const(bp=100)
        bonus = _setup_synergy(
            True, setup_move, 200, False,
            True, attack_move, 300, False,
        )
        assert bonus > 0


class TestEnvSynergy:
    """Tests for _env_synergy."""

    def test_no_moves_no_bonus(self):
        """No move constants means zero bonus."""
        state = _make_mock_state()
        pkm_a = _make_mock_pkm_view()
        pkm_b = _make_mock_pkm_view()
        bonus = _env_synergy(True, None, True, None, pkm_a, pkm_b, state)
        assert bonus == 0.0

    def test_rain_sets_water_boost(self):
        """Setting Rain when ally has Water move gives bonus."""
        state = _make_mock_state()
        rain_move = _make_mock_move_const(bp=0, cat=Category.OTHER, weather=Weather.RAIN, pkm_type=Type.WATER)
        water_move = _make_mock_move_const(bp=80, pkm_type=Type.WATER)
        pkm_a = _make_mock_pkm_view()
        pkm_b = _make_mock_pkm_view(types=[Type.WATER])

        bonus = _env_synergy(True, rain_move, True, water_move, pkm_a, pkm_b, state)
        assert bonus > 0
