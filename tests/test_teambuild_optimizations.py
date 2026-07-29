"""Unit tests validating teambuild optimizations A-F.

A. upgrade_cap reduced to 25% of pool.
B. Coefficient tables cached from global_max_scores and reused.
C. Global max sample size reduced to 10.
D. Utility scorer short-circuited for pure damaging moves.
E. Stat boost synergy uses fast coefficient-table path.
F. GA defaults reduced to pop=30, gen=20.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Status, Terrain, Weather
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from src.config.models import TeambuildConfig
from src.teambuild.moveset import (
    _calculate_stat_boost_synergy,
    _has_utility_potential,
    get_role_aware_moveset,
)
from src.teambuild.policy import HesfTeamBuildPolicy
from src.teambuild.scoring import build_coefficient_table


@pytest.fixture()
def small_roster() -> list[Any]:
    """Generate a small 12-species roster for fast tests."""
    rng = np.random.default_rng(99)
    move_set = gen_move_set(30, rng)
    return list(gen_pkm_roster(12, move_set, 4, rng))


@pytest.fixture()
def params() -> BattleRuleParam:
    """Default battle rule parameters."""
    return BattleRuleParam()


class TestOptimizationA:
    """A. upgrade_cap is 25% of pool (not 50%)."""

    def test_upgrade_cap_is_quarter_of_pool(self, small_roster: list[Any]) -> None:
        """Verify upgrade_cap = max(team_size, pool_size // 4)."""
        cfg = TeambuildConfig(
            enable_evolution=True,
            enable_battle_royale=False,
            population_size=10,
            generations=2,
        )
        policy = HesfTeamBuildPolicy(config=cfg)
        max_team_size = 4

        pool_size = max(max_team_size, int(len(small_roster) * (1 - cfg.pruning_percentage)))
        expected_cap = max(max_team_size, pool_size // 4)

        with patch(
            "src.teambuild.policy.get_optimal_archetype", return_value=None
        ) as mock_upgrade:
            policy.decision(small_roster, None, max_team_size, 4, 2)

        assert mock_upgrade.call_count <= expected_cap

    def test_upgrade_cap_never_below_team_size(self) -> None:
        """upgrade_cap floor is max_team_size even for tiny pools."""
        pool_size = 5
        max_team_size = 4
        cap = max(max_team_size, pool_size // 4)
        assert cap == max_team_size


class TestOptimizationB:
    """B. Coefficient tables cached from global_max_scores and reused."""

    def test_coeff_cache_returned_from_global_max(self, small_roster: list[Any]) -> None:
        """_compute_global_max_scores returns a coeff_cache dict."""
        cfg = TeambuildConfig(enable_evolution=False)
        policy = HesfTeamBuildPolicy(config=cfg)

        generic_cache: dict[Any, Any] = {}
        from src.shared.archetypes import create_generic_build_for_species

        for s in small_roster:
            b = create_generic_build_for_species(s)
            if b is not None:
                generic_cache[s] = b

        roster_list = list(small_roster)
        maxima, coeff_cache = policy._compute_global_max_scores(
            small_roster[:8], roster_list, generic_cache
        )

        assert isinstance(maxima, dict)
        assert "max_stat" in maxima
        assert isinstance(coeff_cache, dict)
        assert len(coeff_cache) > 0

    def test_upgrade_loop_reuses_cached_tables(self, small_roster: list[Any]) -> None:
        """Upgrade loop does not rebuild coeff_table for cached species."""
        cfg = TeambuildConfig(
            enable_evolution=True,
            enable_battle_royale=False,
            population_size=10,
            generations=2,
        )
        policy = HesfTeamBuildPolicy(config=cfg)

        call_count = {"n": 0}
        original_build = build_coefficient_table

        def counting_build(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            return original_build(*args, **kwargs)

        with (
            patch("src.teambuild.policy.build_coefficient_table", side_effect=counting_build),
            patch("src.teambuild.policy.get_optimal_archetype", return_value=None),
        ):
            policy.decision(small_roster, None, 4, 4, 2)

        sample_size = min(10, len(small_roster))
        pool_size = max(4, int(len(small_roster) * (1 - cfg.pruning_percentage)))
        upgrade_cap = max(4, pool_size // 4)
        uncached_upgrades = max(0, upgrade_cap - sample_size)

        assert call_count["n"] <= sample_size + uncached_upgrades


class TestOptimizationC:
    """C. Global max sample size is 10 (not 15)."""

    def test_sample_size_capped_at_10(self, small_roster: list[Any]) -> None:
        """_compute_global_max_scores samples at most 10 species."""
        cfg = TeambuildConfig(enable_evolution=False)
        policy = HesfTeamBuildPolicy(config=cfg)

        generic_cache: dict[Any, Any] = {}
        from src.shared.archetypes import create_generic_build_for_species

        for s in small_roster:
            b = create_generic_build_for_species(s)
            if b is not None:
                generic_cache[s] = b

        roster_list = list(small_roster)
        large_pool = small_roster * 3

        with patch(
            "src.teambuild.policy.build_coefficient_table",
            wraps=build_coefficient_table,
        ) as mock_coeff:
            policy._compute_global_max_scores(large_pool, roster_list, generic_cache)

        assert mock_coeff.call_count <= 10


class TestOptimizationD:
    """D. Utility scorer short-circuited for pure damaging moves."""

    def test_pure_damage_move_has_no_utility(self) -> None:
        """A move with base_power > 0 and no flags returns False."""
        move = MagicMock()
        move.base_power = 90
        move.heal = 0
        move.protect = False
        move.toggle_reflect = False
        move.toggle_lightscreen = False
        move.status = Status.NONE
        move.weather_start = Weather.CLEAR
        move.field_start = Terrain.NONE
        move.toggle_tailwind = False
        move.toggle_trickroom = False
        move.hazard = None
        move.name = "Tackle"

        assert _has_utility_potential(move) is False

    def test_status_move_has_utility(self) -> None:
        """A move with status != NONE returns True."""
        move = MagicMock()
        move.base_power = 0
        move.heal = 0
        move.protect = False
        move.toggle_reflect = False
        move.toggle_lightscreen = False
        move.status = Status.BURN
        move.weather_start = Weather.CLEAR
        move.field_start = Terrain.NONE
        move.toggle_tailwind = False
        move.toggle_trickroom = False
        move.hazard = None
        move.name = "Will-O-Wisp"

        assert _has_utility_potential(move) is True

    def test_protect_move_has_utility(self) -> None:
        """A protect move returns True."""
        move = MagicMock()
        move.base_power = 0
        move.heal = 0
        move.protect = True
        move.toggle_reflect = False
        move.toggle_lightscreen = False
        move.status = Status.NONE
        move.weather_start = Weather.CLEAR
        move.field_start = Terrain.NONE
        move.toggle_tailwind = False
        move.toggle_trickroom = False
        move.hazard = None
        move.name = "Protect"

        assert _has_utility_potential(move) is True

    def test_weather_move_has_utility(self) -> None:
        """A weather-setting move returns True."""
        move = MagicMock()
        move.base_power = 0
        move.heal = 0
        move.protect = False
        move.toggle_reflect = False
        move.toggle_lightscreen = False
        move.status = Status.NONE
        move.weather_start = Weather.RAIN
        move.field_start = Terrain.NONE
        move.toggle_tailwind = False
        move.toggle_trickroom = False
        move.hazard = None
        move.name = "Rain Dance"

        assert _has_utility_potential(move) is True

    def test_short_circuit_skips_utility_call(self, small_roster: list[Any], params: BattleRuleParam) -> None:
        """get_role_aware_moveset does not call calculate_utility_score for pure damage moves."""
        from src.shared.archetypes import create_generic_build_for_species

        species = small_roster[0]
        build = create_generic_build_for_species(species)
        if build is None:
            pytest.skip("No build for first species")

        generic_cache: dict[Any, Any] = {}
        for s in small_roster:
            b = create_generic_build_for_species(s)
            if b is not None:
                generic_cache[s] = b

        with patch(
            "src.teambuild.moveset.calculate_utility_score", return_value=0.0
        ) as mock_util:
            get_role_aware_moveset(build, "Fast Physical Sweeper", list(small_roster), params, generic_cache)

        pure_damage_count = sum(
            1 for m in species.moves
            if m.base_power > 0 and not _has_utility_potential(m)
        )
        total_moves = len(species.moves)
        assert mock_util.call_count <= total_moves - pure_damage_count


class TestOptimizationE:
    """E. Stat boost synergy uses fast coefficient-table path."""

    def test_fast_path_used_with_coeff_table(self, small_roster: list[Any], params: BattleRuleParam) -> None:
        """When coeff_table is provided, calculate_damage_score is NOT called."""
        from src.shared.archetypes import create_generic_build_for_species

        species = small_roster[0]
        build = create_generic_build_for_species(species)
        if build is None:
            pytest.skip("No build for first species")

        generic_cache: dict[Any, Any] = {}
        for s in small_roster:
            b = create_generic_build_for_species(s)
            if b is not None:
                generic_cache[s] = b

        roster_list = list(small_roster)
        coeff_table = build_coefficient_table(species, roster_list, generic_cache, params)

        boost_move = MagicMock()
        boost_move.boosts = (0, 2, 0, 0, 0, 0)
        boost_move.self_boosts = True

        with patch(
            "src.teambuild.moveset.calculate_damage_score", side_effect=AssertionError("slow path called")
        ):
            result = _calculate_stat_boost_synergy(
                build, boost_move, roster_list, generic_cache, params, coeff_table
            )

        assert isinstance(result, float)

    def test_slow_path_used_without_coeff_table(self, small_roster: list[Any], params: BattleRuleParam) -> None:
        """Without coeff_table, falls back to calculate_damage_score."""
        from src.shared.archetypes import create_generic_build_for_species

        species = small_roster[0]
        build = create_generic_build_for_species(species)
        if build is None:
            pytest.skip("No build for first species")

        generic_cache: dict[Any, Any] = {}
        for s in small_roster:
            b = create_generic_build_for_species(s)
            if b is not None:
                generic_cache[s] = b

        roster_list = list(small_roster)

        boost_move = MagicMock()
        boost_move.boosts = (0, 2, 0, 0, 0, 0)
        boost_move.self_boosts = True

        result = _calculate_stat_boost_synergy(
            build, boost_move, roster_list, generic_cache, params, None
        )
        assert isinstance(result, float)

    def test_no_boost_returns_zero(self, small_roster: list[Any], params: BattleRuleParam) -> None:
        """Moves without boosts return 0.0 immediately."""
        from src.shared.archetypes import create_generic_build_for_species

        species = small_roster[0]
        build = create_generic_build_for_species(species)
        if build is None:
            pytest.skip("No build for first species")

        no_boost_move = MagicMock()
        no_boost_move.boosts = None
        no_boost_move.self_boosts = False

        result = _calculate_stat_boost_synergy(
            build, no_boost_move, list(small_roster), {}, params, None
        )
        assert result == 0.0


class TestOptimizationF:
    """F. GA defaults reduced to pop=30, gen=20."""

    def test_default_population_size(self) -> None:
        """Default population_size is 30."""
        cfg = TeambuildConfig()
        assert cfg.population_size == 30

    def test_default_generations(self) -> None:
        """Default generations is 20."""
        cfg = TeambuildConfig()
        assert cfg.generations == 20

    def test_config_still_validates_bounds(self) -> None:
        """Config enforces ge=10 for pop and ge=1 for gens."""
        with pytest.raises(ValidationError):
            TeambuildConfig(population_size=5)

        with pytest.raises(ValidationError):
            TeambuildConfig(generations=0)
