"""Role-aware moveset selection for the Team Build Policy.

Scores all moves in a species' movepool across four independent
dimensions (damage, utility, stat boost synergy, speed control
synergy) and selects the optimal set of 4 for a given archetype.
"""

from typing import Any

from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.modifiers import Category, Stat, Status, Terrain, Weather
from vgc2.battle_engine.pokemon import BattlingPokemon, Pokemon
from vgc2.battle_engine.team import BattlingTeam

from src.shared.archetypes import create_generic_build_for_species
from src.shared.cache import cached_calculate_damage as calculate_damage
from src.teambuild.scoring import (
    calculate_damage_score,
    calculate_damage_score_fast,
    calculate_utility_score,
)


def get_role_aware_moveset(
    attacker_build: Pokemon,
    archetype_name: str,
    roster: list[Any],
    params: BattleRuleParam,
    generic_cache: dict[Any, Any] | None = None,
    coeff_table: dict[Any, list[tuple[float, int]]] | None = None,
) -> tuple[list[Any], dict[Any, Any]]:
    """Select the best 4 moves for a species given a specific role.

    Each move is scored on four independent dimensions to prevent
    scale contamination: damage, utility, stat boost synergy,
    and speed control synergy.

    Args:
        attacker_build: The Pokemon build to select moves for.
        archetype_name: Name of the archetype role (e.g. "Fast Physical Sweeper").
        roster: Full roster of PokemonSpecies for context.
        params: Battle rule parameters.
        generic_cache: Optional precomputed generic builds per species.
        coeff_table: Optional precomputed damage coefficients for fast scoring.

    Returns:
        Tuple of (list of 4 Move objects, dict mapping each move to its scores).
    """
    if not attacker_build.species.moves:
        return [], {}

    fast_damage: dict[Any, float] = {}
    if coeff_table is not None:
        fast_damage = calculate_damage_score_fast(attacker_build, coeff_table)

    move_scores: dict[Any, dict[str, float]] = {
        move: {"damage": 0.0, "utility": 0.0, "stat_syn": 0.0, "speed_syn": 0.0}
        for move in attacker_build.species.moves
    }

    for move in attacker_build.species.moves:
        if coeff_table is not None:
            damage_score = fast_damage.get(move, 0.0)
        else:
            damage_score = calculate_damage_score(attacker_build, move, roster, generic_cache, params)

        if _has_utility_potential(move):
            utility_score = calculate_utility_score(attacker_build, move, roster, params, generic_cache)
        else:
            utility_score = 0.0

        stat_syn = _calculate_stat_boost_synergy(
            attacker_build, move, roster, generic_cache, params, coeff_table
        )
        speed_syn = _calculate_speed_control_synergy(attacker_build, move, roster, generic_cache, params)

        move_scores[move]["damage"] = damage_score
        move_scores[move]["utility"] = utility_score
        move_scores[move]["stat_syn"] = stat_syn
        move_scores[move]["speed_syn"] = speed_syn

    max_damage = max((s.get("damage", 0) for s in move_scores.values()), default=0) or 1.0
    max_utility = max((s.get("utility", 0) for s in move_scores.values()), default=0) or 1.0
    max_stat_syn = max((s.get("stat_syn", 0) for s in move_scores.values()), default=0) or 1.0
    max_speed_syn = max((s.get("speed_syn", 0) for s in move_scores.values()), default=0) or 1.0

    def get_final_score(move: Any) -> float:
        s = move_scores.get(move, {})
        d = s.get("damage", 0) / max_damage
        u = s.get("utility", 0) / max_utility
        st = s.get("stat_syn", 0) / max_stat_syn
        sp = s.get("speed_syn", 0) / max_speed_syn
        return d + u + st + sp

    sorted_moves = sorted(move_scores.keys(), key=get_final_score, reverse=True)
    top_4 = sorted_moves[:4]

    return top_4, move_scores


def _has_utility_potential(move: Any) -> bool:
    """Check whether a move can produce a non-zero utility score.

    Pure damaging moves with no secondary utility flags always score 0
    in calculate_utility_score. Skipping the call avoids expensive
    State/BattlingPokemon object construction for ~60-70% of moves.

    Args:
        move: The Move to check.

    Returns:
        True if the move has any utility-relevant property.
    """
    if move.heal > 0:
        return True
    if move.protect:
        return True
    if move.toggle_reflect or move.toggle_lightscreen:
        return True
    if move.status != Status.NONE:
        return True
    if move.weather_start != Weather.CLEAR or move.field_start != Terrain.NONE:
        return True
    if move.toggle_tailwind or move.toggle_trickroom:
        return True
    if hasattr(move, "hazard") and move.hazard:
        return True
    return move.name.lower() in {"rapid spin", "defog", "mortal spin", "tidy up"}


def _calculate_stat_boost_synergy(
    attacker_build: Pokemon,
    move: Any,
    roster: list[Any],
    optimal_builds_cache: dict[Any, Any] | None,
    params: BattleRuleParam,
    coeff_table: dict[Any, list[tuple[float, int]]] | None = None,
) -> float:
    """Calculate the synergy score for a self-targeting stat-boosting move.

    Measures the net damage increase the boost provides to the user's
    best damaging moves from its entire species movepool. Uses the fast
    coefficient-table path when available to avoid State object creation.

    Args:
        attacker_build: The Pokemon using the move.
        move: The Move to evaluate.
        roster: Full roster for context.
        optimal_builds_cache: Optional cache of optimal builds per species.
        params: Battle rule parameters.
        coeff_table: Optional precomputed damage coefficients for fast scoring.

    Returns:
        Float synergy score.
    """
    if not (move.boosts and move.self_boosts):
        return 0.0

    is_phys_boost = move.boosts[Stat.ATTACK - 1] > 0
    relevant_cat = Category.PHYSICAL if is_phys_boost else Category.SPECIAL

    potential_moves = attacker_build.species.moves
    relevant_damaging = [m for m in potential_moves if m.category == relevant_cat and m.base_power > 0]

    if not relevant_damaging:
        return 0.0

    if coeff_table is not None:
        base_scores = calculate_damage_score_fast(attacker_build, coeff_table)
        boosted_build = apply_temp_boosts(attacker_build, move.boosts)
        boosted_scores = calculate_damage_score_fast(boosted_build, coeff_table)

        top_moves = sorted(relevant_damaging, key=lambda m: base_scores.get(m, 0.0), reverse=True)[:2]
        total_increase = sum(
            boosted_scores.get(m, 0.0) - base_scores.get(m, 0.0) for m in top_moves
        )
    else:
        top_moves = sorted(
            relevant_damaging,
            key=lambda m: calculate_damage_score(attacker_build, m, roster, optimal_builds_cache, params),
            reverse=True,
        )[:2]

        if not top_moves:
            return 0.0

        total_increase = 0.0
        for move_obj in top_moves:
            dmg_before = calculate_damage_score(attacker_build, move_obj, roster, optimal_builds_cache, params)
            boosted_build = apply_temp_boosts(attacker_build, move.boosts)
            dmg_after = calculate_damage_score(boosted_build, move_obj, roster, optimal_builds_cache, params)
            total_increase += dmg_after - dmg_before

    return total_increase / len(top_moves) if top_moves else 0.0


def _calculate_speed_control_synergy(
    attacker_build: Pokemon,
    move: Any,
    roster: list[Any],
    optimal_builds_cache: dict[Any, Any] | None,
    params: BattleRuleParam,
) -> float:
    """Calculate the synergy score for speed-control moves (Tailwind, Trick Room).

    Simulates 1v1 matchups against the roster and quantifies the
    average net damage swing from flipping the turn order.

    Args:
        attacker_build: The Pokemon using the move.
        move: The Move to evaluate.
        roster: Full roster for context.
        optimal_builds_cache: Optional cache of optimal builds.
        params: Battle rule parameters.

    Returns:
        Float synergy score.
    """
    if not (move.toggle_tailwind or move.toggle_trickroom):
        return 0.0

    total_swing = 0.0
    relevant = 0

    my_battle_pkm = BattlingPokemon(attacker_build)
    my_team_shell = BattlingTeam(active=[my_battle_pkm], reserve=[])
    opp_team_shell = BattlingTeam(active=[None], reserve=[])
    state_shell = State((my_team_shell, opp_team_shell))

    for opp_species in roster:
        if opp_species is attacker_build.species:
            continue

        opp_build = None
        if optimal_builds_cache:
            opp_build = optimal_builds_cache.get(opp_species)
        if opp_build is None:
            opp_build = create_generic_build_for_species(opp_species)
            if opp_build is None:
                continue

        opp_battle_pkm = BattlingPokemon(opp_build)
        state_shell.sides[1].team.active[0] = opp_battle_pkm

        my_speed = attacker_build.stats[Stat.SPEED]
        opp_speed = opp_build.stats[Stat.SPEED]

        if not (my_speed < opp_speed):
            continue

        relevant += 1

        my_best = max(
            (m for m in attacker_build.moves if m.base_power > 0),
            key=lambda m: m.base_power,
            default=None,
        )
        opp_best = max(
            (m for m in opp_build.moves if m.base_power > 0),
            key=lambda m: m.base_power,
            default=None,
        )
        if not my_best or not opp_best:
            continue

        my_dmg = calculate_damage(params, 0, my_best, state_shell, my_battle_pkm, opp_battle_pkm)
        opp_dmg = calculate_damage(params, 1, opp_best, state_shell, opp_battle_pkm, my_battle_pkm)

        my_hp_after = attacker_build.stats[Stat.MAX_HP] - opp_dmg
        my_retal = 0 if my_hp_after <= 0 else my_dmg
        net_before = my_retal - opp_dmg

        opp_hp_after = opp_build.stats[Stat.MAX_HP] - my_dmg
        opp_retal = 0 if opp_hp_after <= 0 else opp_dmg
        net_after = my_dmg - opp_retal

        total_swing += net_after - net_before

    return total_swing / relevant if relevant > 0 else 0.0


def apply_temp_boosts(pokemon_build: Pokemon, boost_stages: tuple[int, ...]) -> Pokemon:
    """Create a new Pokemon object with temporary stat boosts applied.

    Copies the original build and recalculates stats to reflect
    the given stat stage changes.

    Args:
        pokemon_build: The original Pokemon object.
        boost_stages: Tuple of stat stage changes.

    Returns:
        A new Pokemon object with updated stats.
    """
    temp = Pokemon(
        species=pokemon_build.species,
        move_indexes=[pokemon_build.species.moves.index(m) for m in pokemon_build.moves],
        level=pokemon_build.level,
        evs=pokemon_build.evs,
        ivs=pokemon_build.ivs,
        nature=pokemon_build.nature,
    )

    new_stats = list(temp.stats)

    for i, change in enumerate(boost_stages[:5]):
        if change != 0:
            stat_idx = i + 1
            boost_multi = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0}
            multiplier = boost_multi.get(change, 1.0)
            new_stats[stat_idx] = int(new_stats[stat_idx] * multiplier)

    temp.stats = tuple(new_stats)
    return temp
