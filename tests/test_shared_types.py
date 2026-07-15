"""Tests for the string-based type effectiveness chart."""

from src.shared.types import TYPE_CHART, TYPE_NAMES, is_immune, type_effectiveness, vgc2_type_to_name


class TestTypeNames:
    """Tests for type name utilities."""

    def test_has_19_types(self) -> None:
        """Type chart contains exactly 19 types (18 real + no_type filler)."""
        assert len(TYPE_NAMES) == 19
        assert len(TYPE_CHART) == 19

    def test_vgc2_type_to_name_known_indices(self) -> None:
        """vgc2 Type enum indices map to correct lowercase names."""
        assert vgc2_type_to_name(0) == "normal"
        assert vgc2_type_to_name(1) == "fire"
        assert vgc2_type_to_name(9) == "flying"
        assert vgc2_type_to_name(17) == "fairy"

    def test_vgc2_type_to_name_out_of_range(self) -> None:
        """Out-of-range indices return 'typeless'."""
        assert vgc2_type_to_name(99) == "typeless"
        assert vgc2_type_to_name(-1) == "typeless"


class TestTypeEffectiveness:
    """Tests for type_effectiveness() with known matchups."""

    def test_neutral(self) -> None:
        """Normal move vs Normal type = 1.0x."""
        assert type_effectiveness("normal", ["normal"]) == 1.0

    def test_super_effective(self) -> None:
        """Water vs Fire = 2.0x."""
        assert type_effectiveness("water", ["fire"]) == 2.0

    def test_not_very_effective(self) -> None:
        """Fire vs Water = 0.5x."""
        assert type_effectiveness("fire", ["water"]) == 0.5

    def test_immune(self) -> None:
        """Normal vs Ghost = 0.0x (immune)."""
        assert type_effectiveness("normal", ["ghost"]) == 0.0
        assert type_effectiveness("ground", ["flying"]) == 0.0
        assert type_effectiveness("electric", ["ground"]) == 0.0

    def test_quad_resisted(self) -> None:
        """Electric vs Water/Ground = 0.0x (immune due to Ground)."""
        assert type_effectiveness("electric", ["water", "ground"]) == 0.0

        """Grass vs Steel/Fire = 0.25x."""
        assert type_effectiveness("grass", ["steel", "fire"]) == 0.25

    def test_quad_super_effective(self) -> None:
        """Ice vs Dragon/Flying = 4.0x."""
        assert type_effectiveness("ice", ["dragon", "flying"]) == 4.0

    def test_fairy_immune_to_dragon(self) -> None:
        """Fairy is immune to Dragon."""
        assert type_effectiveness("dragon", ["fairy"]) == 0.0

    def test_steel_resists_many(self) -> None:
        """Steel resists Normal, Flying, Psychic, Bug, Rock, Grass, Ice, Dragon, Fairy, Steel."""
        for atk in ["normal", "flying", "psychic", "bug", "rock", "grass", "ice", "steel", "fairy"]:
            assert type_effectiveness(atk, ["steel"]) == 0.5, f"{atk} vs Steel should be 0.5x"

    def test_multiple_defender_types(self) -> None:
        """Combined effectiveness multiplies correctly for dual-type defenders."""
        assert type_effectiveness("fighting", ["normal", "dark"]) == 4.0  # 2.0 * 2.0
        assert type_effectiveness("ground", ["electric", "steel"]) == 4.0  # 2.0 * 2.0
        assert type_effectiveness("fire", ["grass", "ice"]) == 4.0  # 2.0 * 2.0


class TestImmunity:
    """Tests for is_immune() function."""

    def test_ground_immune_to_electric(self) -> None:
        """Ground types are immune to Electric."""
        assert is_immune("electric", ["ground"])

    def test_flying_not_immune_to_electric(self) -> None:
        """Flying is weak to Electric, not immune."""
        assert not is_immune("electric", ["flying"])

    def test_dual_immune(self) -> None:
        """Fighting vs Ghost = immune."""
        assert is_immune("fighting", ["ghost"])


class TestChartIntegrity:
    """Structural integrity tests for the type chart."""

    def test_all_chart_rows_have_19_entries(self) -> None:
        """Every attacking type has effectiveness entries for all 19 defending types."""
        for atk_type in TYPE_NAMES:
            assert len(TYPE_CHART[atk_type]) == 19

    def test_no_missing_defender_keys(self) -> None:
        """Every cell in the chart is populated (no missing keys)."""
        for atk_type in TYPE_NAMES:
            for def_type in TYPE_NAMES:
                assert def_type in TYPE_CHART[atk_type], f"Missing {atk_type} -> {def_type}"
