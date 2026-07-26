"""Genetic operators for the Team Build Policy's evolutionary algorithm.

Provides population initialisation, single-point crossover with
deduplication, per-position mutation, and the team-level fitness
function combining viability, type coverage, type defence,
stat diversity, role diversity, and coverage balance.
"""

from typing import Any

import numpy as np
from vgc2.battle_engine.modifiers import Category, Stat, Type

from src.shared.types import type_effectiveness, vgc2_type_to_name
from src.teambuild.builds import species_role

VGC2_TYPE_ORDER = [
    Type.NORMAL,
    Type.FIRE,
    Type.WATER,
    Type.ELECTRIC,
    Type.GRASS,
    Type.ICE,
    Type.FIGHT,
    Type.POISON,
    Type.GROUND,
    Type.FLYING,
    Type.PSYCHIC,
    Type.BUG,
    Type.ROCK,
    Type.GHOST,
    Type.DRAGON,
    Type.DARK,
    Type.STEEL,
    Type.FAIRY,
]

TYPE_NAMES = [vgc2_type_to_name(t.value) for t in VGC2_TYPE_ORDER]

N_TYPES = 18

_COVERAGE_BALANCE_WEIGHT = 0.15
_VIABILITY_WEIGHT = 0.25
_COVERAGE_WEIGHT = 0.20
_DEFENCE_WEIGHT = 0.20
_STAT_DIVERSITY_WEIGHT = 0.10
_ROLE_DIVERSITY_WEIGHT = 0.10


def _type_name(t: Any) -> str:
    """Convert vgc2 Type enum or int to lowercase name.

    Args:
        t: Type enum, int, or already a string.

    Returns:
        Lowercase type name.
    """
    if isinstance(t, str):
        return t
    if isinstance(t, int):
        return vgc2_type_to_name(t)
    return vgc2_type_to_name(t.value)


def init_population(
    pool_species: list[Any],
    team_size: int,
    pop_size: int,
    viability_scores: dict[Any, float],
    rng: Any,
) -> list[list[int]]:
    """Initialise a population of random teams from the species pool.

    Sampling is weighted by each species' viability score so that
    higher-ranked species appear more frequently.

    Args:
        pool_species: Ordered list of species (high viability first).
        team_size: Number of species per team (typically 6).
        pop_size: Number of teams in the population.
        viability_scores: Dict mapping species -> float viability score.
        rng: NumPy Generator for reproducible randomness.

    Returns:
        List of teams, each team is a list of species indices into pool_species.
    """
    n = len(pool_species)
    scores = [max(viability_scores.get(s, 0), 0.001) for s in pool_species]
    probs = [s / sum(scores) for s in scores]

    population = []
    for _ in range(pop_size):
        team = list(rng.choice(n, team_size, replace=False, p=probs))
        population.append(team)

    return population


def seed_coverage_teams(
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    team_size: int,
    n_seeds: int,
    rng: Any,
) -> list[list[int]]:
    """Generate seed teams using JJJ-style type-rank coverage selection.

    Builds a 19xN type-coverage matrix ranked by damage potential, then
    greedily selects teams that minimise coverage range across all 19 types.

    Args:
        pool_species: Species pool to select from.
        viability_scores: Dict mapping species -> viability score.
        team_size: Species per team.
        n_seeds: Number of seed teams to generate.
        rng: NumPy Generator.

    Returns:
        List of seed teams (each a list of species indices).
    """
    n = len(pool_species)

    matrix = _build_type_coverage_matrix(pool_species, viability_scores)

    ranked = _rank_transform(matrix)

    seeds = []
    for seed_offset in range(n_seeds):
        team = _greedy_coverage_select(ranked, n, team_size, rng, seed_offset)
        seeds.append(team)

    return seeds


def _build_type_coverage_matrix(
    pool_species: list[Any],
    viability_scores: dict[Any, float],
) -> np.ndarray:
    """Build a 19xN matrix of type-coverage damage proxies.

    Each cell (type_i, species_j) estimates how well species_j hits type_i,
    combining type effectiveness with raw species power.

    Args:
        pool_species: List of species.
        viability_scores: Species -> power dict.

    Returns:
        NumPy array of shape (19, N).
    """
    n = len(pool_species)
    matrix = np.zeros((N_TYPES + 1, n), dtype=np.float32)

    for j, species in enumerate(pool_species):
        power = max(viability_scores.get(species, 0), 0.001)
        spec_types = [vgc2_type_to_name(t.value) for t in species.types]
        for i, atk_type in enumerate(VGC2_TYPE_ORDER):
            atk_name = vgc2_type_to_name(atk_type.value)
            best_eff = 0.0
            for move in species.moves:
                if move.base_power <= 0:
                    continue
                eff = type_effectiveness(atk_name, spec_types)
                if eff > best_eff:
                    best_eff = eff
            matrix[i, j] = best_eff * power

    return matrix


def _rank_transform(matrix: np.ndarray) -> np.ndarray:
    """Rank-transform each row of the matrix (argsort of argsort).

    Lower rank = better coverage against that type.

    Args:
        matrix: 2D array of shape (N_types, N_species).

    Returns:
        Rank-transformed array of the same shape.
    """
    ranked = np.zeros_like(matrix, dtype=np.int32)
    for i in range(matrix.shape[0]):
        order = np.argsort(matrix[i])
        ranked[i] = np.argsort(order)
    return ranked


def _greedy_coverage_select(
    ranked: np.ndarray,
    n_species: int,
    team_size: int,
    rng: Any,
    seed_offset: int,
) -> list[int]:
    """Greedy coverage selection minimising range across types.

    Args:
        ranked: Rank-transformed type-coverage matrix.
        n_species: Total species count.
        team_size: Desired team size.
        rng: NumPy Generator.
        seed_offset: Offset for deterministic variety.

    Returns:
        List of selected species indices.
    """
    sum_ranks = ranked.sum(axis=0)

    start = int(rng.integers(0, min(3, n_species)))
    if seed_offset > 0:
        start = int(rng.integers(0, min(5 + seed_offset, n_species)))
    selected = [int(np.argmin(sum_ranks))] if seed_offset == 0 else [start]

    coverage = ranked[:, selected[0]].copy().astype(np.float64)

    for _ in range(1, team_size):
        old_range = float(np.max(coverage) - np.min(coverage))
        best_idx = -1
        best_val = -1e9

        for idx in range(n_species):
            if idx in selected:
                continue
            new_cov = coverage + ranked[:, idx].astype(np.float64)
            new_range = float(np.max(new_cov) - np.min(new_cov))
            delta = old_range - new_range
            val = 0.45 * float(sum_ranks[idx]) + 0.45 * delta
            if val > best_val:
                best_val = val
                best_idx = idx

        if best_idx == -1:
            remaining = [i for i in range(n_species) if i not in selected]
            if remaining:
                best_idx = remaining[0]

        selected.append(best_idx)
        coverage += ranked[:, best_idx].astype(np.float64)

    return selected


def crossover(
    parent_a: list[int],
    parent_b: list[int],
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    rng: Any,
    team_size: int = 6,
) -> tuple[list[int], list[int]]:
    """Single-point crossover with deduplication.

    Slices both parents at a random point between 1 and team_size-1, swaps
    tails to produce two children. Duplicate species within a child are resolved
    by replacing with the highest-viability unused species from the pool.

    Args:
        parent_a: First parent team (list of indices).
        parent_b: Second parent team.
        pool_species: Full pool list for deduplication fallback.
        viability_scores: Dict mapping species -> float viability score.
        rng: NumPy Generator.
        team_size: Target team size (default 6).

    Returns:
        Tuple of two child teams, each with team_size unique species indices.
    """
    point = rng.integers(1, min(len(parent_a) - 1, len(parent_b) - 1))

    child1 = parent_a[:point] + parent_b[point:]
    child2 = parent_b[:point] + parent_a[point:]

    child1 = _deduplicate(child1, pool_species, viability_scores, rng, team_size)
    child2 = _deduplicate(child2, pool_species, viability_scores, rng, team_size)

    return child1, child2


def _deduplicate(
    team: list[int],
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    rng: Any,
    team_size: int = 6,
) -> list[int]:
    """Remove duplicate indices from a team, replacing with unused species.

    Args:
        team: Team with possible duplicate indices.
        pool_species: Full species pool.
        viability_scores: Dict mapping species -> viability score.
        rng: NumPy Generator.
        team_size: Desired team size (default 6).

    Returns:
        Team with team_size unique indices.
    """
    seen = set()
    result = []
    for idx in team:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)

    while len(result) < team_size:
        unused = [i for i in range(len(pool_species)) if i not in seen]
        if not unused:
            break
        candidate = _pick_best_unused(unused, viability_scores, pool_species, rng)
        result.append(candidate)
        seen.add(candidate)

    return result


def _pick_best_unused(
    unused: list[int],
    viability_scores: dict[Any, float],
    pool_species: list[Any],
    rng: Any,
) -> int:
    """Pick from unused indices, weighted by viability.

    Args:
        unused: List of unused species indices.
        viability_scores: Dict of scores.
        pool_species: Species list.
        rng: NumPy Generator.

    Returns:
        Chosen index.
    """
    scores = [max(viability_scores.get(pool_species[i], 0), 0.001) for i in unused]
    total = sum(scores)
    probs = [s / total for s in scores]
    return int(rng.choice(unused, 1, p=probs)[0])


def mutate_team(
    team: list[int],
    pool_size: int,
    mutation_rate: float,
    viability_scores: dict[Any, float],
    pool_species: list[Any],
    rng: Any,
) -> list[int]:
    """Per-position mutation with species uniqueness constraint.

    Each position has a ``mutation_rate`` chance of being replaced with
    a random species not already in the team. At least one mutation is
    guaranteed per call.

    Args:
        team: Current team indices.
        pool_size: Number of species in the pool.
        mutation_rate: Probability of mutating each position (0.0–1.0).
        viability_scores: Dict of viability scores.
        pool_species: Species list for lookup.
        rng: NumPy Generator.

    Returns:
        Mutated team with unique indices.
    """
    mutated = list(team)
    any_mutated = False

    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            current_set = set(mutated)
            pool = [j for j in range(pool_size) if j not in current_set]
            if pool:
                weighted = _weighted_pool(pool, viability_scores, pool_species)
                mutated[i] = weighted[int(rng.choice(len(weighted), 1)[0])]
                any_mutated = True

    if not any_mutated and mutation_rate > 0:
        i = int(rng.integers(0, len(mutated)))
        current_set = set(mutated)
        pool = [j for j in range(pool_size) if j not in current_set]
        if pool:
            weighted = _weighted_pool(pool, viability_scores, pool_species)
            mutated[i] = weighted[int(rng.choice(len(weighted), 1)[0])]

    return mutated


def _weighted_pool(
    indices: list[int],
    viability_scores: dict[Any, float],
    pool_species: list[Any],
) -> list[int]:
    """Create a weighted duplicate-reduced pool for mutation selection.

    Lower-viability species are given higher weight during mutation
    (to encourage exploration), inverse of the selection weighting.

    Args:
        indices: Candidate indices.
        viability_scores: Dict of viability scores.
        pool_species: Species list.

    Returns:
        List of indices with lower-viability species appearing multiple times.
    """
    expanded = []
    for i in indices:
        score = max(viability_scores.get(pool_species[i], 0), 0.001)
        weight = max(1, int(1.0 / score))
        expanded.extend([i] * min(weight, 5))
    return expanded if expanded else indices


def calculate_team_fitness(
    team_indices: list[int],
    pool_species: list[Any],
    viability_scores: dict[Any, float],
) -> float:
    """Compute fitness for a team of species.

    Six components weighted into a single score:

    1. **Viability** — sum of individual species power (normalised to 0–1).
    2. **Type coverage** — fraction of 18 types hit super-effectively.
    3. **Type defence** — fraction of weaknesses covered by allies.
    4. **Stat diversity** — bonus for mixing physical/special attackers and speed tiers.
    5. **Role diversity** — bonus for having sweeper + wall + mixed roles.
    6. **Coverage balance** — how evenly the team threatens all 19 types.

    Args:
        team_indices: List of indices into pool_species.
        pool_species: Full species pool.
        viability_scores: Dict mapping species -> float viability score.

    Returns:
        Fitness float (higher = better). Range roughly 0–1.
    """
    members = [pool_species[i] for i in team_indices]

    v = _fitness_viability(members, viability_scores)
    tc = _fitness_type_coverage(members)
    td = _fitness_type_defence(members)
    sd = _fitness_stat_diversity(members)
    rd = _fitness_role_diversity(members)
    cb = _fitness_coverage_balance(members)

    return (
        _VIABILITY_WEIGHT * v
        + _COVERAGE_WEIGHT * tc
        + _DEFENCE_WEIGHT * td
        + _STAT_DIVERSITY_WEIGHT * sd
        + _ROLE_DIVERSITY_WEIGHT * rd
        + _COVERAGE_BALANCE_WEIGHT * cb
    )


def _fitness_viability(members: list[Any], viability_scores: dict[Any, float]) -> float:
    """Normalised species power sum.

    Args:
        members: List of species on the team.
        viability_scores: Dict of viability scores.

    Returns:
        Normalised viability score (0–1).
    """
    max_possible = 6 * max(viability_scores.values()) if viability_scores else 1.0
    total = sum(max(viability_scores.get(s, 0), 0) for s in members)
    return min(total / max_possible, 1.0)


def _fitness_type_coverage(members: list[Any]) -> float:
    """Fraction of 18 types hit super-effectively by the team.

    Args:
        members: List of species.

    Returns:
        Coverage fraction (0–1).
    """
    covered = set()
    for species in members:
        for move in species.moves:
            if move.base_power <= 0:
                continue
            atk_name = _type_name(move.pkm_type)
            for def_name in TYPE_NAMES:
                eff = type_effectiveness(atk_name, [def_name])
                if eff > 1.0:
                    covered.add(def_name)

    return len(covered) / N_TYPES if len(covered) else 0.0


def _fitness_type_defence(members: list[Any]) -> float:
    """Fraction of weaknesses covered by ally resistance or immunity.

    Args:
        members: List of species.

    Returns:
        Defence synergy fraction (0–1).
    """
    member_types = []
    for species in members:
        type_list = [vgc2_type_to_name(t.value) for t in species.types]
        member_types.append(type_list)

    total_weaknesses = 0
    covered_weaknesses = 0

    for i, _species in enumerate(members):
        for atk_name in TYPE_NAMES:
            eff = type_effectiveness(atk_name, member_types[i])
            if eff > 1.0:
                total_weaknesses += 1
                for j, ally_types in enumerate(member_types):
                    if i == j:
                        continue
                    ally_eff = type_effectiveness(atk_name, ally_types)
                    if ally_eff < 1.0:
                        covered_weaknesses += 1
                        break

    return covered_weaknesses / max(total_weaknesses, 1)


def _fitness_stat_diversity(members: list[Any]) -> float:
    """Score for mixing physical/special attackers and fast/slow speed tiers.

    Args:
        members: List of species.

    Returns:
        Diversity score (0–1).
    """
    has_phys = any(
        any(m.category in (Category.PHYSICAL, Category.PHYSICAL.value) and m.base_power > 0 for m in species.moves)
        for species in members
    )
    has_spec = any(
        any(m.category in (Category.SPECIAL, Category.SPECIAL.value) and m.base_power > 0 for m in species.moves)
        for species in members
    )

    speed_values = [s.base_stats[Stat.SPEED] for s in members]
    median_speed = sorted(speed_values)[len(speed_values) // 2] if speed_values else 50
    has_fast = any(sp > median_speed + 10 for sp in speed_values)
    has_slow = any(sp < median_speed - 10 for sp in speed_values)

    score = 0.0
    if has_phys and has_spec:
        score += 0.5
    if has_fast and has_slow:
        score += 0.5

    return score


def _fitness_role_diversity(members: list[Any]) -> float:
    """Role diversity score. Rewards having sweeper + wall + mixed.

    Args:
        members: List of species.

    Returns:
        Role diversity score (0–1). Each missing role subtracts 0.33.
    """
    roles = {species_role(s) for s in members}
    missing = 0
    for required in ("sweeper", "wall", "mixed"):
        if required not in roles:
            missing += 1
    return 1.0 - missing * 0.3


def _fitness_coverage_balance(members: list[Any]) -> float:
    """Coverage balance score — how evenly the team threatens all types.

    Computes the total super-effective coverage count per type across all
    team members. A perfectly balanced team threatens every type roughly
    equally (low range). A team with blind spots has high range.

    Args:
        members: List of species.

    Returns:
        Balance score (0–1). 1.0 = perfect balance, 0.0 = worst.
    """
    coverage_per_type = dict.fromkeys(TYPE_NAMES, 0)
    for species in members:
        for move in species.moves:
            if move.base_power <= 0:
                continue
            atk_name = _type_name(move.pkm_type)
            for def_name in TYPE_NAMES:
                eff = type_effectiveness(atk_name, [def_name])
                if eff > 1.0:
                    coverage_per_type[def_name] = coverage_per_type.get(def_name, 0) + 1

    counts = list(coverage_per_type.values())
    if not counts:
        return 0.0
    min_cov = min(counts)
    max_cov = max(counts)
    if max_cov == min_cov:
        return 1.0
    return 1.0 - (max_cov - min_cov) / max(max_cov, 1)
