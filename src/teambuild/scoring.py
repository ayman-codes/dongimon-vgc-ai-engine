"""Move scoring for the Team Build Policy.

Evaluates damage output, utility value, and synergy contributions
for moves in the context of the full roster. Used by the archetype
fitness evaluator to select optimal movesets.
"""

from typing import Any

from vgc2.battle_engine import BattleRuleParam, calculate_damage
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.modifiers import Category, Hazard, Stat, Status, Terrain, Type, Weather
from vgc2.battle_engine.pokemon import BattlingPokemon, Pokemon
from vgc2.battle_engine.team import BattlingTeam

from src.shared.archetypes import create_generic_build_for_species
from src.shared.types import type_effectiveness, vgc2_type_to_name


def calculate_damage_score(
    attacker_build: Pokemon,
    move: Any,
    roster: list[Any],
    optimal_builds_cache: dict[Any, Any] | None,
    params: BattleRuleParam,
) -> float:
    """Score a damaging move against the full roster.

    For each species in the roster, computes the damage as a
    percentage of the defender's max HP and averages the result.

    Args:
        attacker_build: The attacker Pokemon build.
        move: The Move to score.
        roster: Full roster of PokemonSpecies.
        optimal_builds_cache: Optional cached optimal builds per species.
        params: Battle rule parameters.

    Returns:
        Average normalized damage score across all roster members.
    """
    if move.base_power == 0:
        return 0.0

    total_normalized = 0.0
    opp_count = 0

    attack_pkm = BattlingPokemon(attacker_build)
    my_team_shell = BattlingTeam(active=[attack_pkm], reserve=[])
    opp_team_shell = BattlingTeam(active=[None], reserve=[])
    state_shell = State((my_team_shell, opp_team_shell))

    for defender_species in roster:
        if defender_species is attacker_build.species:
            continue

        if optimal_builds_cache:
            defender_build = optimal_builds_cache.get(defender_species)
        else:
            defender_build = create_generic_build_for_species(defender_species)

        if defender_build is None:
            continue

        defender_battle_pkm = BattlingPokemon(defender_build)
        state_shell.sides[1].team.active[0] = defender_battle_pkm

        dmg = calculate_damage(
            params=params,
            attacking_side=0,
            move=move,
            state=state_shell,
            attacker=attack_pkm,
            defender=defender_battle_pkm,
        )
        max_hp = defender_build.stats[Stat.MAX_HP]
        total_normalized += (dmg / max_hp) * 100.0 if max_hp > 0 else 0.0
        opp_count += 1

    if opp_count == 0:
        return 0.0
    return total_normalized / opp_count


def calculate_utility_score(
    attacker_build: Pokemon,
    move: Any,
    roster: list[Any],
    params: BattleRuleParam,
) -> float:
    """Score a non-damaging move by its utility value against the full roster.

    Evaluates recovery, screens, hazards, protect, weather/terrain,
    and status moves. Scores are normalized to a "damage equivalent" scale.

    Args:
        attacker_build: The Pokemon using the move.
        move: The Move to score.
        roster: Full roster of PokemonSpecies.
        params: Battle rule parameters.

    Returns:
        Float utility score.
    """
    if not roster:
        return 0.0

    generic_cache: dict[Any, Any] = {}
    for s in roster:
        b = create_generic_build_for_species(s)
        if b is not None:
            generic_cache[s] = b

    hazard_removal_names = {"rapid spin", "defog", "mortal spin", "tidy up"}

    avg_def = sum(b.stats[Stat.DEFENSE] for b in generic_cache.values()) / len(generic_cache) if generic_cache else 1
    avg_spdef = (
        sum(b.stats[Stat.SPECIAL_DEFENSE] for b in generic_cache.values()) / len(generic_cache) if generic_cache else 1
    )

    if move.heal > 0:
        hp_restored = attacker_build.stats[Stat.MAX_HP] * move.heal
        normalized = (hp_restored / attacker_build.stats[Stat.MAX_HP]) * 100.0
        avg_total_def = avg_def + avg_spdef
        pkm_total_def = attacker_build.stats[Stat.DEFENSE] + attacker_build.stats[Stat.SPECIAL_DEFENSE]
        defensive_mult = 1.0 + ((pkm_total_def - avg_total_def) / avg_total_def) if avg_total_def > 0 else 1.0
        return normalized * max(0.5, defensive_mult)  # type: ignore[no-any-return]

    if move.toggle_reflect or move.toggle_lightscreen:
        is_reflect = move.toggle_reflect
        total_avoid = 0.0
        threats = [
            s
            for s in roster
            if (is_reflect and s.base_stats[Stat.ATTACK] > s.base_stats[Stat.SPECIAL_ATTACK])
            or (not is_reflect and s.base_stats[Stat.SPECIAL_ATTACK] > s.base_stats[Stat.ATTACK])
        ]
        if not threats:
            return 0.0

        for team_species in roster:
            team_build = generic_cache.get(team_species)
            if not team_build:
                continue

            avg_avoid = 0.0
            for threat_species in threats:
                threat_build = generic_cache.get(threat_species)
                if not threat_build or threat_build == team_build:
                    continue

                best_move = 0.0
                for threat_move in threat_build.moves:
                    if (is_reflect and threat_move.category == Category.PHYSICAL) or (
                        not is_reflect and threat_move.category == Category.SPECIAL
                    ):
                        dmg = calculate_damage(
                            params=params,
                            attacking_side=0,
                            move=threat_move,
                            state=State(
                                (
                                    BattlingTeam(active=[BattlingPokemon(threat_build)], reserve=[]),
                                    BattlingTeam(active=[BattlingPokemon(team_build)], reserve=[]),
                                )
                            ),
                            attacker=BattlingPokemon(threat_build),
                            defender=BattlingPokemon(team_build),
                        )
                        if dmg > best_move:
                            best_move = dmg

                avg_avoid += best_move * 0.5

            if threats:
                total_avoid += avg_avoid / len(threats)

        avg_dmg = total_avoid / len(roster) if roster else 0.0
        return avg_dmg * 2.5

    if move.name.lower() in hazard_removal_names:
        total_avoid = 0.0
        for species in roster:
            eff = type_effectiveness(
                vgc2_type_to_name(Type.ROCK.value),
                [vgc2_type_to_name(t.value) for t in species.types],
            )
            dmg = (species.base_stats[Stat.MAX_HP] / 8.0) * eff
            total_avoid += dmg
        return total_avoid / len(roster) if roster else 0.0

    if move.hazard == Hazard.STEALTH_ROCK:
        total_dmg = 0.0
        for species in roster:
            eff = type_effectiveness(
                vgc2_type_to_name(Type.ROCK.value),
                [vgc2_type_to_name(t.value) for t in species.types],
            )
            dmg = (species.base_stats[Stat.MAX_HP] / 8.0) * eff
            total_dmg += dmg
        return total_dmg / len(roster) if roster else 0.0

    if move.protect:
        max_threat = 0.0
        for species in roster:
            if species is attacker_build.species:
                continue
            opp_build = generic_cache.get(species)
            if not opp_build:
                continue

            opp_best = max(
                (m for m in opp_build.moves if m.base_power > 0),
                key=lambda m: m.base_power,
                default=None,
            )
            if not opp_best:
                continue

            dmg = calculate_damage(
                params=params,
                attacking_side=1,
                move=opp_best,
                state=State(
                    (
                        BattlingTeam(active=[BattlingPokemon(opp_build)], reserve=[]),
                        BattlingTeam(active=[BattlingPokemon(attacker_build)], reserve=[]),
                    )
                ),
                attacker=BattlingPokemon(opp_build),
                defender=BattlingPokemon(attacker_build),
            )
            if dmg > max_threat:
                max_threat = dmg

        return (max_threat / attacker_build.stats[Stat.MAX_HP]) * 100.0  # type: ignore[no-any-return]

    if move.weather_start != Weather.CLEAR or move.field_start != Terrain.NONE:
        net_swing = 0.0

        def _effect_swing(move_type_boost: Type, multiplier: float, weather_nerf: Type | None = None) -> float:
            swing = 0
            for species in roster:
                build = generic_cache.get(species)
                if not build:
                    continue
                best_move = max(
                    (m for m in build.moves if m.pkm_type == move_type_boost and m.base_power > 0),
                    default=None,
                    key=lambda m: m.base_power,
                )
                if not best_move:
                    continue
                base_dmg = calculate_damage(
                    params=params,
                    attacking_side=0,
                    move=best_move,
                    state=State(
                        (
                            BattlingTeam([BattlingPokemon(build)], reserve=[]),
                            BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]),
                        )
                    ),
                    attacker=BattlingPokemon(build),
                    defender=BattlingPokemon(attacker_build),
                )
                swing += base_dmg * (multiplier - 1)

                if weather_nerf:
                    nerf_move = max(
                        (m for m in build.moves if m.pkm_type == weather_nerf and m.base_power > 0),
                        default=None,
                        key=lambda m: m.base_power,
                    )
                    if nerf_move:
                        nerf_base = calculate_damage(
                            params=params,
                            attacking_side=0,
                            move=nerf_move,
                            state=State(
                                (
                                    BattlingTeam([BattlingPokemon(build)], reserve=[]),
                                    BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]),
                                )
                            ),
                            attacker=BattlingPokemon(build),
                            defender=BattlingPokemon(attacker_build),
                        )
                        swing += nerf_base * (1 - 0.5)
            return swing

        if move.weather_start == Weather.RAIN:
            net_swing = _effect_swing(Type.WATER, 1.5, weather_nerf=Type.FIRE)
        elif move.weather_start == Weather.SUN:
            net_swing = _effect_swing(Type.FIRE, 1.5, weather_nerf=Type.WATER)
        elif move.weather_start == Weather.SAND:
            passive_dmg = 0
            def_boost = 0
            for species in roster:
                build = generic_cache.get(species)
                if not build:
                    continue
                if not any(t in (Type.ROCK, Type.GROUND, Type.STEEL) for t in species.types):
                    passive_dmg += build.stats[Stat.MAX_HP] / 16.0
                if Type.ROCK in species.types:
                    def_boost += build.stats[Stat.SPECIAL_DEFENSE] * 0.33
            net_swing = passive_dmg + def_boost
        elif move.weather_start == Weather.SNOW:
            def_boost = 0
            for species in roster:
                build = generic_cache.get(species)
                if not build:
                    continue
                if Type.ICE in species.types:
                    def_boost += build.stats[Stat.DEFENSE] * 0.33
            net_swing = def_boost
        elif move.field_start == Terrain.ELECTRIC_TERRAIN:
            net_swing = _effect_swing(Type.ELECTRIC, 1.3)
        elif move.field_start == Terrain.GRASSY_TERRAIN:
            net_swing = _effect_swing(Type.GRASS, 1.3)
        elif move.field_start == Terrain.PSYCHIC_TERRAIN:
            net_swing = _effect_swing(Type.PSYCHIC, 1.3)
        elif move.field_start == Terrain.MISTY_TERRAIN:
            net_swing = _effect_swing(Type.DRAGON, 0.5)

        return net_swing / len(roster) if roster else 0.0

    if move.status != Status.NONE:
        base_status = 0.0
        avg_roster_hp = (
            sum(b.stats[Stat.MAX_HP] for b in generic_cache.values()) / len(generic_cache) if generic_cache else 1
        )

        if move.status == Status.BURN:
            passive = avg_roster_hp / 16.0
            phys_threats = [
                b for s, b in generic_cache.items() if s.base_stats[Stat.ATTACK] > s.base_stats[Stat.SPECIAL_ATTACK]
            ]
            mitigated = 0
            if phys_threats:
                strongest = max(phys_threats, key=lambda b: b.stats[Stat.ATTACK])
                best_phys = max(
                    (m for m in strongest.moves if m.category == Category.PHYSICAL),
                    default=None,
                    key=lambda m: m.base_power,
                )
                if best_phys:
                    mitigated = (
                        calculate_damage(
                            params=params,
                            attacking_side=0,
                            move=best_phys,
                            state=State(
                                (
                                    BattlingTeam([BattlingPokemon(strongest)], reserve=[]),
                                    BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]),
                                )
                            ),
                            attacker=BattlingPokemon(strongest),
                            defender=BattlingPokemon(attacker_build),
                        )
                        / 2.0
                    )
            base_status = passive + mitigated

        elif move.status == Status.TOXIC:
            base_status = (avg_roster_hp / 16.0) * (1 + 2 + 3 + 4)

        elif move.status == Status.SLEEP:
            max_dmg_potential = 0.0
            for _, build in generic_cache.items():
                best_move = max(  # type: ignore[assignment]
                    (m for m in build.moves if m.base_power > 0),
                    default=None,
                    key=lambda m: m.base_power,
                )
                if not best_move:
                    continue
                dmg = calculate_damage(
                    params=params,
                    attacking_side=0,
                    move=best_move,
                    state=State(
                        (
                            BattlingTeam(active=[BattlingPokemon(build)], reserve=[]),
                            BattlingTeam(active=[BattlingPokemon(attacker_build)], reserve=[]),
                        )
                    ),
                    attacker=BattlingPokemon(build),
                    defender=BattlingPokemon(attacker_build),
                )
                if dmg > max_dmg_potential:
                    max_dmg_potential = dmg
            base_status = ((max_dmg_potential * 1.5) / avg_roster_hp) * 100

        elif move.status == Status.PARALYZED:
            if not generic_cache:
                base_status = 0.0
            else:
                fastest = max(generic_cache.values(), key=lambda b: b.stats[Stat.SPEED])
                best_move = max(  # type: ignore[assignment]
                    (m for m in fastest.moves if m.base_power > 0),
                    default=None,
                    key=lambda m: m.base_power,
                )
                if best_move:
                    dmg = calculate_damage(
                        params=params,
                        attacking_side=0,
                        move=best_move,
                        state=State(
                            (
                                BattlingTeam([BattlingPokemon(fastest)], reserve=[]),
                                BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]),
                            )
                        ),
                        attacker=BattlingPokemon(fastest),
                        defender=BattlingPokemon(attacker_build),
                    )
                    base_status = ((dmg * 0.25) / avg_roster_hp) * 100

        return base_status

    return 0.0
