"""Tests for genetic operators: init, crossover, mutate, and team fitness."""

import numpy as np
import pytest
from vgc2.battle_engine.modifiers import Category, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies

from src.teambuild.operators import (
    calculate_team_fitness,
    crossover,
    init_population,
    mutate_team,
)


def _make_species(name: str, hp=100, atk=100, df=70, spa=80, spd=70, spe=80, types=None):
    if types is None:
        types = [Type.NORMAL]
    move = Move(
        pkm_type=Type.NORMAL, base_power=80, accuracy=1.0,
        max_pp=15, category=Category.PHYSICAL, name=f"{name}Move",
    )
    return PokemonSpecies(name=name, base_stats=(hp, atk, df, spa, spd, spe), types=types, moves=[move])


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def pool():
    return [
        _make_species(f"S{i}", hp=100 + i * 10, atk=80 + i * 5, spe=70 + i * 2)
        for i in range(20)
    ]


@pytest.fixture
def viability(pool):
    return {pool[i]: float(100 - i * 5) for i in range(len(pool))}


class TestInitPopulation:
    """Tests for init_population."""

    def test_correct_size(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        pop = init_population(pool_s, 6, 10, viab, rng)
        assert len(pop) == 10

    def test_teams_have_6_members(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        pop = init_population(pool_s, 6, 5, viab, rng)
        for team in pop:
            assert len(team) == 6

    def test_no_duplicates_in_team(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        pop = init_population(pool_s, 6, 10, viab, rng)
        for team in pop:
            assert len(set(team)) == 6

    def test_all_indices_in_range(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        pop = init_population(pool_s, 6, 5, viab, rng)
        for team in pop:
            for idx in team:
                assert 0 <= idx < len(pool_s)


class TestCrossover:
    """Tests for crossover."""

    def test_returns_two_children(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        pa, pb = [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]
        ca, cb = crossover(pa, pb, pool_s, viab, rng)
        assert len(ca) == 6
        assert len(cb) == 6

    def test_children_have_unique_species(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        ca, cb = crossover([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], pool_s, viab, rng)
        assert len(set(ca)) == 6
        assert len(set(cb)) == 6

    def test_deterministic_with_same_rng(self):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        pa, pb = [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]
        c1a, c1b = crossover(pa, pb, pool_s, viab, np.random.default_rng(42))
        c2a, c2b = crossover(pa, pb, pool_s, viab, np.random.default_rng(42))
        assert c1a == c2a
        assert c1b == c2b


class TestMutateTeam:
    """Tests for mutate_team."""

    def test_zero_rate_no_change(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        team = [0, 1, 2, 3, 4, 5]
        mutated = mutate_team(team, len(pool_s), 0.0, viab, pool_s, rng)
        assert mutated == team

    def test_no_duplicates_after_mutation(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        team = [0, 1, 2, 3, 4, 5]
        mutated = mutate_team(team, len(pool_s), 0.5, viab, pool_s, rng)
        assert len(set(mutated)) == 6

    def test_all_indices_valid(self, rng):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        team = [0, 1, 2, 3, 4, 5]
        mutated = mutate_team(team, len(pool_s), 0.5, viab, pool_s, rng)
        for idx in mutated:
            assert 0 <= idx < len(pool_s)


class TestCalculateTeamFitness:
    """Tests for calculate_team_fitness."""

    def test_returns_float(self):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        team = [0, 1, 2, 3, 4, 5]
        fitness = calculate_team_fitness(team, pool_s, viab)
        assert isinstance(fitness, float)

    def test_same_team_same_fitness(self):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        team = [0, 1, 2, 3, 4, 5]
        f1 = calculate_team_fitness(team, pool_s, viab)
        f2 = calculate_team_fitness(team, pool_s, viab)
        assert f1 == f2

    def test_higher_viability_team_higher_fitness(self):
        pool_s = _make_species_pool(20)
        viab = {pool_s[i]: float(100 - i * 5) for i in range(len(pool_s))}
        high = [0, 1, 2, 3, 4, 5]
        low = [15, 16, 17, 18, 19, 0]
        f_high = calculate_team_fitness(high, pool_s, viab)
        f_low = calculate_team_fitness(low, pool_s, viab)
        assert f_high >= f_low


def _make_species_pool(n: int) -> list:
    """Create a pool of n species for testing."""
    return [
        _make_species(f"S{i}", hp=100 + i * 10, atk=80 + i * 5, spe=70 + i * 2)
        for i in range(n)
    ]
