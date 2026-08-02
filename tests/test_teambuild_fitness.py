"""Tests for teambuild fitness and viability functions."""

from unittest.mock import MagicMock

import pytest
from vgc2.battle_engine.modifiers import Stat, Type
from vgc2.battle_engine.pokemon import PokemonSpecies

from src.teambuild.fitness import calculate_stat_compatibility


def _make_dummy_species(name: str, hp=100, atk=100, df=70, spa=80, spd=70, spe=80,
                        types=None):
    if types is None:
        types = [Type.NORMAL]
    species = MagicMock(spec=PokemonSpecies)
    species.name = name
    species.base_stats = {
        Stat.MAX_HP: hp,
        Stat.ATTACK: atk,
        Stat.DEFENSE: df,
        Stat.SPECIAL_ATTACK: spa,
        Stat.SPECIAL_DEFENSE: spd,
        Stat.SPEED: spe,
    }
    species.types = types
    species.moves = []
    return species


class TestCalculateStatCompatibility:
    """Tests for calculate_stat_compatibility."""

    def test_attack_focused_species_returns_high_atk_score(self):
        """Species with high Atk gets high score for Atk EV investment."""
        species = _make_dummy_species("PhysicalMon", atk=130, spa=50)
        evs = (0, 252, 0, 0, 0, 0)
        score = calculate_stat_compatibility(species, evs)
        assert score > 0

    def test_hp_investment_scored_lower_than_primary_stat(self):
        """HP investment is weighted at 0.25, primary stat at 1.0."""
        species = _make_dummy_species("TestMon", atk=130, spa=50)
        hp_evs = (252, 0, 0, 0, 0, 0)
        atk_evs = (0, 252, 0, 0, 0, 0)
        hp_score = calculate_stat_compatibility(species, hp_evs)
        atk_score = calculate_stat_compatibility(species, atk_evs)
        assert atk_score > hp_score

    def test_second_best_stat_weighted_half(self):
        """Second-best stat EV is weighted at 0.5."""
        species = _make_dummy_species("TestMon", atk=130, spe=100, spa=50)
        evs = (0, 252, 0, 0, 0, 252)
        score = calculate_stat_compatibility(species, evs)
        assert score == pytest.approx(378.0)

    def test_zero_evs_returns_zero(self):
        """No EV investment returns 0."""
        species = _make_dummy_species("TestMon")
        score = calculate_stat_compatibility(species, (0, 0, 0, 0, 0, 0))
        assert score == 0.0

    def test_lower_primary_stat_uses_attack_as_first(self):
        """Species with high SpA uses SpA as primary stat."""
        species = _make_dummy_species("SpecialMon", atk=50, spa=130)
        spa_evs = (0, 0, 0, 252, 0, 0)
        atk_evs = (0, 252, 0, 0, 0, 0)
        spa_score = calculate_stat_compatibility(species, spa_evs)
        atk_score = calculate_stat_compatibility(species, atk_evs)
        assert spa_score > atk_score

    def test_special_defense_as_second_best(self):
        """SpD can be the second-best stat if base SpD is high."""
        species = _make_dummy_species("SpDefMon", atk=130, spd=110, spe=50)
        evs = (0, 252, 0, 0, 252, 0)
        score = calculate_stat_compatibility(species, evs)
        assert score == pytest.approx(378.0)

    def test_deterministic_output(self):
        """Same inputs always produce same score."""
        species = _make_dummy_species("TestMon", atk=100, spa=60)
        evs = (4, 252, 0, 0, 0, 252)
        score1 = calculate_stat_compatibility(species, evs)
        score2 = calculate_stat_compatibility(species, evs)
        assert score1 == score2
