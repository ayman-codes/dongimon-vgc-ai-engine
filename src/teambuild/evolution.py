"""Evolutionary algorithm for the Team Build Policy.

Runs a generational loop: evaluate fitness → sort → retain elites →
tournament select → crossover → mutate → repeat. Returns the top K
teams from the final generation.
"""

from typing import Any

from numpy.random import default_rng

from src.teambuild.operators import (
    calculate_team_fitness,
    crossover,
    init_population,
    mutate_team,
    seed_coverage_teams,
)


def run_evolution(
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    team_size: int,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    elite_fraction: float,
    rng: Any,
) -> list[list[int]]:
    """Run the evolutionary algorithm and return the top K teams.

    Args:
        pool_species: Ordered list of species to select from.
        viability_scores: Dict mapping species -> float viability score.
        team_size: Number of species per team (typically 6).
        pop_size: Number of teams in the population.
        generations: Number of generations to evolve.
        mutation_rate: Per-position mutation probability.
        elite_fraction: Fraction of population preserved unchanged.
        rng: NumPy Generator for reproducibility.

    Returns:
        List of top K teams (each team is a list of 6 species indices),
        ranked by fitness descending. K = max(3, ceil(pop_size * elite_fraction)).
    """
    rng_for_seeds = rng if hasattr(rng, 'integers') else default_rng()
    coverage_seeds = seed_coverage_teams(
        pool_species, viability_scores, team_size,
        n_seeds=min(5, max(1, pop_size // 10)), rng=rng_for_seeds,
    )
    n_coverage = len(coverage_seeds)
    n_random = pop_size - n_coverage
    if n_random > 0:
        random_pop = init_population(pool_species, team_size, n_random, viability_scores, rng)
        population = coverage_seeds + random_pop
    else:
        population = coverage_seeds[:pop_size]

    for _gen in range(generations):
        fitnesses = [calculate_team_fitness(team, pool_species, viability_scores) for team in population]

        ranked = sorted(zip(population, fitnesses, strict=False), key=lambda x: -x[1])

        elite_count = max(1, int(pop_size * elite_fraction))
        next_pop = [team for team, _ in ranked[:elite_count]]

        non_elite_count = pop_size - elite_count
        for _ in range(non_elite_count):
            parent_a = _tournament_select(population, fitnesses, 3, rng)
            parent_b = _tournament_select(population, fitnesses, 3, rng)

            child_a, child_b = crossover(
                parent_a,
                parent_b,
                pool_species,
                viability_scores,
                rng,
                team_size=team_size,
            )

            child_a = mutate_team(child_a, len(pool_species), mutation_rate, viability_scores, pool_species, rng)
            child_b = mutate_team(child_b, len(pool_species), mutation_rate, viability_scores, pool_species, rng)

            next_pop.append(child_a)
            if len(next_pop) < pop_size:
                next_pop.append(child_b)

        population = next_pop[:pop_size]

    final_fitnesses = [calculate_team_fitness(team, pool_species, viability_scores) for team in population]

    ranked_final = sorted(zip(population, final_fitnesses, strict=False), key=lambda x: -x[1])
    top_k = max(3, int(pop_size * elite_fraction))
    return [team for team, _ in ranked_final[:top_k]]


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    tournament_size: int,
    rng: Any,
) -> list[int]:
    """Tournament selection: pick k random individuals, return fittest.

    Args:
        population: List of team index lists.
        fitnesses: List of fitness floats aligned with population.
        tournament_size: Number of random candidates to draw.
        rng: NumPy Generator.

    Returns:
        One team from the tournament (the fittest of the sampled group).
    """
    candidates = rng.choice(len(population), min(tournament_size, len(population)), replace=False)
    best = max(candidates, key=lambda i: fitnesses[i])
    return population[best]
