"""Genetic operators for the Team Build Policy's evolutionary algorithm.

Provides population initialisation, single-point crossover with
deduplication, per-position mutation, and the team-level fitness
function combining viability, type coverage, type defence,
stat diversity, and role diversity.
"""

from typing import Any

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


def crossover(
    parent_a: list[int],
    parent_b: list[int],
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    rng: Any,
) -> tuple[list[int], list[int]]:
    """Single-point crossover with deduplication.

    Slices both parents at a random point between 1 and 5, swaps tails
    to produce two children. Duplicate species within a child are resolved
    by replacing with the highest-viability unused species from the pool.

    Args:
        parent_a: First parent team (list of indices).
        parent_b: Second parent team.
        pool_species: Full pool list for deduplication fallback.
        viability_scores: Dict mapping species -> float viability score.
        rng: NumPy Generator.

    Returns:
        Tuple of two child teams, each with 6 unique species indices.
    """
    point = rng.integers(1, len(parent_a) - 1)

    child1 = parent_a[:point] + parent_b[point:]
    child2 = parent_b[:point] + parent_a[point:]

    child1 = _deduplicate(child1, pool_species, viability_scores, rng)
    child2 = _deduplicate(child2, pool_species, viability_scores, rng)

    return child1, child2


def _deduplicate(
    team: list[int],
    pool_species: list[Any],
    viability_scores: dict[Any, float],
    rng: Any,
) -> list[int]:
    """Remove duplicate indices from a team, replacing with unused species.

    Args:
        team: Team with possible duplicate indices.
        pool_species: Full species pool.
        viability_scores: Dict mapping species -> viability score.
        rng: NumPy Generator.

    Returns:
        Team with 6 unique indices.
    """
    seen = set()
    result = []
    for idx in team:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)

    while len(result) < 6:
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
    """Compute fitness for a team of 6 species.

    Five components weighted into a single score:

    1. **Viability** — sum of individual species power (normalised to 0–1).
    2. **Type coverage** — fraction of 18 types hit super-effectively.
    3. **Type defence** — fraction of weaknesses covered by allies.
    4. **Stat diversity** — bonus for mixing physical/special attackers and speed tiers.
    5. **Role diversity** — bonus for having sweeper + wall + mixed roles.

    Args:
        team_indices: List of 6 indices into pool_species.
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

    return 0.30 * v + 0.25 * tc + 0.25 * td + 0.10 * sd + 0.10 * rd


def _fitness_viability(members: list[Any], viability_scores: dict[Any, float]) -> float:
    """Sum of species power normalised by max possible.

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
    {vgc2_type_to_name(t.value): t for t in VGC2_TYPE_ORDER}

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

    return len(covered) / 18.0 if len(covered) else 0.0


def _fitness_type_defence(members: list[Any]) -> float:
    """Fraction of team weaknesses that are resisted or immunised by an ally.

    For each member's defensive weaknesses, checks if any other member
    resists or is immune to that type.

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
    """Stat diversity score.

    Checks for a mix of physical and special attackers, and a mix of
    fast and slow speed tiers.

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

    sorted(s.moves[0].category for s in members if s.moves) if all(s.moves for s in members) else []
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
    """Role diversity score.

    Rewards having at least one sweeper, one wall, and one mixed role.

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
