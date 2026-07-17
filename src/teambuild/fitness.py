"""Fitness evaluation and viability ranking for team building.

Computes optimal archetype builds for each species, runs 1v1
viability comparison tournaments, and ranks species by their
average net matchup score.
"""

from typing import Any

from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies

from src.shared.archetypes import create_archetype_builds, create_generic_build_for_species
from src.teambuild.moveset import get_role_aware_moveset


def calculate_stat_compatibility(species: PokemonSpecies, evs: tuple[int, ...]) -> float:
    """Score how well an EV spread complements a species' base stats.

    Rewards investment in the species' best stat, with diminishing
    returns for the second-best stat and HP.

    Args:
        species: The Pokemon species.
        evs: Tuple of EV values (HP, Atk, Def, SpA, SpD, Spe).

    Returns:
        Float compatibility score.
    """
    base_stats = species.base_stats
    indexed = sorted([(base_stats[i], i) for i in range(1, 6)], reverse=True)
    best_idx = indexed[0][1]
    second_idx = indexed[1][1]

    return (evs[best_idx] * 1.0) + (evs[second_idx] * 0.5) + (evs[Stat.MAX_HP] * 0.25)  # type: ignore[no-any-return]


def calculate_1v1_net_score(
    build_a: Pokemon,
    build_b: Pokemon,
    roster: list[Any],
    params: BattleRuleParam,
) -> float:
    """Calculate the net damage potential in a 1v1 matchup.

    Uses the stat-based power proxy (``species_power``) for efficiency.
    Positive means A outclasses B on raw stats.

    Args:
        build_a: First Pokemon build.
        build_b: Second Pokemon build.
        roster: Full roster for context (unused in stat-based scoring).
        params: Battle rule parameters (unused in stat-based scoring).

    Returns:
        Net score. Positive favours build_A.
    """
    from src.teambuild.builds import species_power

    return species_power(build_a.species) - species_power(build_b.species)


def get_optimal_archetype(
    species: PokemonSpecies,
    roster: list[Any],
    global_max_scores: dict[str, float],
    params: BattleRuleParam,
) -> Pokemon | None:
    """Determine the single best competitive build for a species.

    Evaluates all archetype builds via a weighted fitness function
    with six components (stat, speed, damage, utility, stat synergy,
    speed synergy) normalized against global maximums.

    Args:
        species: The Pokemon species to build.
        roster: Full roster for context.
        global_max_scores: Dict of maximum score values for normalization.
        params: Battle rule parameters.

    Returns:
        A single optimized Pokemon object, or a generic build on failure.
    """
    placeholder_moves = species.moves[:4] if species.moves else []
    potential_builds = create_archetype_builds(species, placeholder_moves)

    if not potential_builds:
        return create_generic_build_for_species(species)

    evaluations = []
    for archetype_name, temp_build in potential_builds:
        optimal_moves, all_scores = get_role_aware_moveset(temp_build, archetype_name, roster, params)

        total_damage = sum(all_scores[m]["damage"] for m in optimal_moves)
        total_utility = sum(all_scores[m]["utility"] for m in optimal_moves)
        total_stat_syn = sum(all_scores[m]["stat_syn"] for m in optimal_moves)
        total_speed_syn = sum(all_scores[m]["speed_syn"] for m in optimal_moves)

        evaluations.append(
            {
                "name": archetype_name,
                "build": temp_build,
                "moves": optimal_moves,
                "stat_score": calculate_stat_compatibility(species, temp_build.evs),
                "damage_score": total_damage,
                "utility_score": total_utility,
                "stat_syn_score": total_stat_syn,
                "speed_syn_score": total_speed_syn,
                "speed_stat_score": temp_build.stats[Stat.SPEED],
            }
        )

    if not evaluations:
        return create_generic_build_for_species(species)

    weights = {
        "w_stat": 0.2,
        "w_speed": 0.2,
        "w_dmg": 0.3,
        "w_util": 0.2,
        "w_stat_syn": 0.05,
        "w_speed_syn": 0.05,
    }

    best_info = None
    best_fitness = -float("inf")

    for item in evaluations:
        norm_stat = item["stat_score"] / global_max_scores["max_stat"]
        norm_dmg = item["damage_score"] / global_max_scores["max_dmg"]
        norm_util = item["utility_score"] / global_max_scores["max_util"]
        norm_speed = item["speed_stat_score"] / global_max_scores["max_speed_stat"]
        max_stat_syn = global_max_scores["max_stat_syn"]
        norm_stat_syn = item["stat_syn_score"] / max_stat_syn if max_stat_syn > 0 else 0
        max_speed_syn = global_max_scores["max_speed_syn"]
        norm_speed_syn = item["speed_syn_score"] / max_speed_syn if max_speed_syn > 0 else 0

        fitness = (
            norm_stat * weights["w_stat"]
            + norm_speed * weights["w_speed"]
            + norm_dmg * weights["w_dmg"]
            + norm_util * weights["w_util"]
            + norm_stat_syn * weights["w_stat_syn"]
            + norm_speed_syn * weights["w_speed_syn"]
        )

        if fitness > best_fitness:
            best_fitness = fitness
            best_info = item

    if best_info is None:
        return create_generic_build_for_species(species)

    final_moves = best_info["moves"]
    move_indices = [species.moves.index(m) for m in final_moves if m in species.moves]
    base_build = best_info["build"]

    return Pokemon(
        species,
        move_indices,
        base_build.level,
        base_build.evs,
        base_build.ivs,
        base_build.nature,
    )
