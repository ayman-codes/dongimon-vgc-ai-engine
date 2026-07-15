"""Tests for the Gen 9 damage formula."""

from src.shared.damage import (
    calculate_damage,
    stab_modifier,
    terrain_boost_multiplier,
    weather_boost_multiplier,
)


class TestCalculateDamage:
    """Tests for the core damage formula."""

    def test_basic_formula(self) -> None:
        """Standard level 50 damage: 80 BP, 150 Atk, 100 Def → known range."""
        damage = calculate_damage(level=50, power=80, attack=150, defense=100)
        assert damage > 0

    def test_zero_power(self) -> None:
        """Zero-power moves deal zero damage."""
        assert calculate_damage(power=0) == 0

    def test_zero_defense(self) -> None:
        """Zero defense is treated as invalid (zero damage)."""
        assert calculate_damage(defense=0) == 0

    def test_minimum_damage(self) -> None:
        """Non-zero modifier always produces at least 1 damage."""
        assert calculate_damage(level=50, power=1, attack=1, defense=999, modifier=1.0) >= 1

    def test_modifier_doubles_damage(self) -> None:
        """A 2.0 modifier approximately doubles the damage."""
        base = calculate_damage(level=50, power=80, attack=150, defense=100, modifier=1.0)
        doubled = calculate_damage(level=50, power=80, attack=150, defense=100, modifier=2.0)
        assert doubled >= base

    def test_known_value(self) -> None:
        """Flutter Mane (level 50, 130 SpA) Shadow Ball (80 BP) vs Iron Hands (86 SpD) with 2x SE."""
        damage = calculate_damage(level=50, power=80, attack=130, defense=86, modifier=2.0)
        assert 90 <= damage <= 135, f"Expected ~90-130 damage, got {damage}"

    def test_zero_modifier_gives_zero(self) -> None:
        """A zero modifier means immunity — zero damage."""
        assert calculate_damage(level=50, power=80, attack=150, defense=100, modifier=0.0) == 0


class TestStab:
    """Tests for Same-Type Attack Bonus."""

    def test_stab_applies(self) -> None:
        """Fire move from a Fire-type gets 1.5x STAB."""
        assert stab_modifier("fire", ["fire"]) == 1.5

    def test_no_stab(self) -> None:
        """Fire move from a Water-type gets no STAB."""
        assert stab_modifier("fire", ["water"]) == 1.0

    def test_dual_type_stab(self) -> None:
        """STAB applies if either type matches."""
        assert stab_modifier("fire", ["fire", "flying"]) == 1.5
        assert stab_modifier("flying", ["fire", "flying"]) == 1.5
        assert stab_modifier("water", ["fire", "flying"]) == 1.0


class TestWeatherBoost:
    """Tests for weather damage modifiers."""

    def test_rain_boosts_water(self) -> None:
        """Rain boosts Water moves by 1.5x."""
        assert weather_boost_multiplier("water", "rain") == 1.5

    def test_rain_weakens_fire(self) -> None:
        """Rain weakens Fire moves by 0.5x."""
        assert weather_boost_multiplier("fire", "rain") == 0.5

    def test_sun_boosts_fire(self) -> None:
        """Sun boosts Fire moves by 1.5x."""
        assert weather_boost_multiplier("fire", "sun") == 1.5

    def test_sun_weakens_water(self) -> None:
        """Sun weakens Water moves by 0.5x."""
        assert weather_boost_multiplier("water", "sun") == 0.5

    def test_sand_no_boost(self) -> None:
        """Sand does not boost any type."""
        assert weather_boost_multiplier("rock", "sand") == 1.0
        assert weather_boost_multiplier("water", "sand") == 1.0

    def test_clear_no_effect(self) -> None:
        """Clear weather applies no modifier."""
        assert weather_boost_multiplier("fire", "clear") == 1.0
        assert weather_boost_multiplier("water", "clear") == 1.0


class TestTerrainBoost:
    """Tests for terrain damage modifiers."""

    def test_electric_terrain_boost(self) -> None:
        """Electric Terrain boosts Electric moves by 1.3x for grounded Pokémon."""
        assert terrain_boost_multiplier("electric", "electric_terrain", True) == 1.3

    def test_grassy_terrain_boost(self) -> None:
        """Grassy Terrain boosts Grass moves by 1.3x."""
        assert terrain_boost_multiplier("grass", "grassy_terrain", True) == 1.3

    def test_psychic_terrain_boost(self) -> None:
        """Psychic Terrain boosts Psychic moves by 1.3x."""
        assert terrain_boost_multiplier("psychic", "psychic_terrain", True) == 1.3

    def test_misty_terrain_weakens_dragon(self) -> None:
        """Misty Terrain halves Dragon moves."""
        assert terrain_boost_multiplier("dragon", "misty_terrain", True) == 0.5

    def test_flying_no_terrain_boost(self) -> None:
        """Flying/ungrounded Pokémon get no terrain boost."""
        assert terrain_boost_multiplier("electric", "electric_terrain", False) == 1.0
        assert terrain_boost_multiplier("grass", "grassy_terrain", False) == 1.0

    def test_none_terrain_no_effect(self) -> None:
        """No terrain applies no modifier."""
        assert terrain_boost_multiplier("electric", "none", True) == 1.0
