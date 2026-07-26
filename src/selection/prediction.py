"""Team preview move prediction and utility scoring.

Used by the Selection Policy to infer opponent movesets and
evaluate their likely impact against our team.
"""

from typing import Any

from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.modifiers import Category, Stat, Status, Terrain, Type, Weather
from vgc2.battle_engine.pokemon import BattlingPokemon, Pokemon, PokemonSpecies
from vgc2.battle_engine.team import BattlingTeam, Team
from vgc2.battle_engine.view import PokemonView

from src.shared.archetypes import create_archetype_builds
from src.shared.cache import cached_calculate_damage as calculate_damage


def predict_moveset(
    species: PokemonSpecies,
    my_full_team: Team,
    all_opp_species_views: list[PokemonView],
    params: BattleRuleParam,
) -> list[Any]:
    """Predict the best 4 moves for a species by scoring its entire movepool.

    Each move is scored by average damage against our team (for damaging
    moves) or utility value (for status moves). The top 4 moves are returned.

    Args:
        species: The Pokemon species whose moves to predict.
        my_full_team: Our team of 6 Pokemon.
        all_opp_species_views: All opponent Pokemon views for context.
        params: Battle rule parameters.

    Returns:
        List of up to 4 Move objects, sorted by score descending.
    """
    if not species.moves:
        return []

    move_scores: dict[Any, float] = dict.fromkeys(species.moves, 0.0)

    archetype_builds = create_archetype_builds(species, species.moves)
    if not archetype_builds:
        return []

    dummy_species = PokemonSpecies(base_stats=(1, 1, 1, 1, 1, 1), types=[], moves=[])
    dummy_pkm = Pokemon(species=dummy_species, move_indexes=[])
    dummy_team = BattlingTeam(active=[dummy_pkm], reserve=[])
    dummy_team.active = []

    my_battling_team = BattlingTeam(active=list(my_full_team.members), reserve=[])
    neutral_state = State((my_battling_team, dummy_team))

    for move in species.moves:
        if move.base_power == 0 and move.category == Category.OTHER:
            move_scores[move] = _calculate_utility_score(
                move,
                species,
                my_full_team,
                all_opp_species_views,
                params,
            )
            continue

        total_damage = 0.0
        for _archetype_name, attacker_build in archetype_builds:
            attack_proto = BattlingPokemon(attacker_build)
            build_total = 0.0

            for my_pkm in my_full_team.members:
                defend_proto = BattlingPokemon(my_pkm)
                dmg = calculate_damage(
                    params=params,
                    attacking_side=1,
                    move=move,
                    state=neutral_state,
                    attacker=attack_proto,
                    defender=defend_proto,
                )
                max_hp = my_pkm.stats[Stat.MAX_HP]
                build_total += (dmg / max_hp) * 100 if max_hp > 0 else 0

            avg_for_build = build_total / len(my_full_team.members) if my_full_team.members else 0
            total_damage += avg_for_build

        move_scores[move] = total_damage / len(archetype_builds) if archetype_builds else 0.0

    sorted_moves = sorted(move_scores.keys(), key=lambda m: move_scores[m], reverse=True)
    return sorted_moves[:4]


def _calculate_utility_score(
    move: Any,
    attacker_species: PokemonSpecies,
    my_full_team: Team,
    all_opp_species_views: list[PokemonView],
    params: BattleRuleParam,
) -> float:
    """Score a non-damaging move by its utility value.

    Evaluates protect (damage blocked), status conditions (burn mitigation,
    toxic damage over time, paralysis/sleep turn denial), weather/terrain
    changes (net damage swing), and entry hazards.

    Args:
        move: The Move to evaluate.
        attacker_species: The species using the move.
        my_full_team: Our full team of 6.
        all_opp_species_views: Opponent Pokemon views.
        params: Battle rule parameters.

    Returns:
        Float utility score.
    """
    if move.base_power > 0:
        return 0.0

    score = 0.0
    my_team_members = my_full_team.members
    all_opp_species = [p.species for p in all_opp_species_views if hasattr(p, "species") and p.species]

    my_dragon = sum(1 for p in my_team_members if Type.DRAGON in p.species.types)
    my_flying = sum(1 for p in my_team_members if Type.FLYING in p.species.types)
    opp_rock = sum(1 for s in all_opp_species if Type.ROCK in s.types)
    opp_ice = sum(1 for s in all_opp_species if Type.ICE in s.types)

    my_grounded = len(my_team_members) - my_flying

    if move.protect:
        max_damage = 0
        defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
        neutral_team = BattlingTeam([defender_proto], [])

        for my_pkm in my_full_team.members:
            for my_move in my_pkm.moves:
                if my_move.base_power > 0:
                    attacker_proto = BattlingPokemon(my_pkm)
                    temp_state = State((BattlingTeam([attacker_proto], []), neutral_team))
                    dmg = calculate_damage(params, 0, my_move, temp_state, attacker_proto, defender_proto)
                    if dmg > max_damage:
                        max_damage = dmg

        attacker_hp = attacker_species.base_stats[Stat.MAX_HP]
        return (max_damage / attacker_hp) * 100 if attacker_hp > 0 else 0

    if move.status == Status.BURN:
        score += (1 / 16) * 100
        my_phys_attackers = [p for p in my_full_team.members if p.stats[Stat.ATTACK] > p.stats[Stat.SPECIAL_ATTACK]]
        if my_phys_attackers:
            strongest = max(my_phys_attackers, key=lambda p: p.stats[Stat.ATTACK])
            best_move = max(
                (m for m in strongest.moves if m.category == Category.PHYSICAL),
                key=lambda m: m.base_power,
                default=None,
            )
            if best_move and best_move.base_power > 0:
                attack_proto = BattlingPokemon(strongest)
                defend_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
                temp_state = State((BattlingTeam([attack_proto], []), BattlingTeam([defend_proto], [])))
                dmg_unburned = calculate_damage(params, 0, best_move, temp_state, attack_proto, defend_proto)
                attack_proto.status = Status.BURN
                dmg_burned = calculate_damage(params, 0, best_move, temp_state, attack_proto, defend_proto)
                prevented = dmg_unburned - dmg_burned
                def_hp = attacker_species.base_stats[Stat.MAX_HP]
                score += (prevented / def_hp) * 100 if def_hp > 0 else 0

    elif move.status == Status.TOXIC:
        score += (10 / 16) * 100

    elif move.status == Status.PARALYZED:
        fastest_pkm = max(my_team_members, key=lambda p: p.stats[Stat.SPEED])
        best_move = max(fastest_pkm.moves, key=lambda m: m.base_power, default=None)
        if best_move and best_move.base_power > 0:
            attack_proto = BattlingPokemon(fastest_pkm)
            defend_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
            temp_state = State((BattlingTeam([attack_proto], []), BattlingTeam([defend_proto], [])))
            dmg_potential = calculate_damage(params, 0, best_move, temp_state, attack_proto, defend_proto)
            denied = dmg_potential * 0.25
            def_hp = attacker_species.base_stats[Stat.MAX_HP]
            score += (denied / def_hp) * 100 if def_hp > 0 else 0

    elif move.status == Status.SLEEP:
        max_dmg = 0
        all_indices = list(range(len(attacker_species.moves)))
        attacker_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=all_indices))
        for my_pkm in my_team_members:
            defend_proto = BattlingPokemon(my_pkm)
            temp_state = State((BattlingTeam([defend_proto], []), BattlingTeam([attacker_proto], [])))
            for opp_move in attacker_species.moves:
                if opp_move.base_power > 0:
                    dmg = calculate_damage(params, 1, opp_move, temp_state, attacker_proto, defend_proto)
                    if dmg > max_dmg:
                        max_dmg = dmg
        avg_hp = sum(p.stats[Stat.MAX_HP] for p in my_team_members) / len(my_team_members) if my_team_members else 1
        score += (max_dmg * 1.5) / avg_hp * 100 if avg_hp > 0 else 0

    def _field_effect_swing(move_type_boost: Type, dmg_mult: float, team: list[Any]) -> float:
        """Calculate net damage gain for a team under a field effect."""
        net = 0
        for pkm in team:
            potential_moves = pkm.moves if hasattr(pkm, "moves") else pkm.species.moves
            best_found = None
            max_power = -1
            for pkm_move in potential_moves:
                if pkm_move.pkm_type == move_type_boost and pkm_move.base_power > max_power:
                    max_power = pkm_move.base_power
                    best_found = pkm_move
            if best_found:
                dummy_species = PokemonSpecies(base_stats=(80, 80, 80, 80, 80, 80), types=[], moves=[])
                generic_defender = BattlingPokemon(Pokemon(species=dummy_species, move_indexes=[]))
                if isinstance(pkm, Pokemon):
                    attacker_proto = BattlingPokemon(pkm)
                else:
                    attacker_proto = BattlingPokemon(Pokemon(species=pkm.species, move_indexes=[]))
                state_no_effect = State((BattlingTeam([generic_defender], []), BattlingTeam([attacker_proto], [])))
                base_dmg = calculate_damage(params, 1, best_found, state_no_effect, attacker_proto, generic_defender)
                net += int(base_dmg * (dmg_mult - 1))
        return net

    if move.weather_start == Weather.RAIN:
        opp_gain = _field_effect_swing(Type.WATER, 1.5, all_opp_species_views)
        opp_nerf = _field_effect_swing(Type.FIRE, 0.5, all_opp_species_views)
        my_gain = _field_effect_swing(Type.WATER, 1.5, my_team_members)
        my_nerf = _field_effect_swing(Type.FIRE, 0.5, my_team_members)
        score += (opp_gain + my_nerf) - (my_gain + opp_nerf)

    elif move.weather_start == Weather.SUN:
        opp_gain = _field_effect_swing(Type.FIRE, 1.5, all_opp_species_views)
        opp_nerf = _field_effect_swing(Type.WATER, 0.5, all_opp_species_views)
        my_gain = _field_effect_swing(Type.FIRE, 1.5, my_team_members)
        my_nerf = _field_effect_swing(Type.WATER, 0.5, my_team_members)
        score += (opp_gain + my_nerf) - (my_gain + opp_nerf)

    elif move.weather_start == Weather.SAND:
        non_immune = sum(
            1 for p in my_team_members if not any(t in (Type.ROCK, Type.GROUND, Type.STEEL) for t in p.species.types)
        )
        if my_team_members:
            score += non_immune * (my_team_members[0].stats[Stat.MAX_HP] / 16)
        score += opp_rock * 20

    elif move.weather_start == Weather.SNOW:
        non_immune = sum(1 for p in my_team_members if Type.ICE not in p.species.types)
        if my_team_members:
            score += non_immune * (my_team_members[0].stats[Stat.MAX_HP] / 16)
        score += opp_ice * 20

    elif move.field_start == Terrain.ELECTRIC_TERRAIN:
        opp_gain = _field_effect_swing(Type.ELECTRIC, 1.3, all_opp_species_views)
        score += opp_gain
        our_sleep_moves = sum(1 for p in my_team_members for m in p.moves if m.status == Status.SLEEP)
        if our_sleep_moves > 0:
            score -= our_sleep_moves * 15
        opp_sleep_moves = sum(1 for s in all_opp_species for m in s.moves if m.status == Status.SLEEP)
        if opp_sleep_moves > 0:
            score += my_grounded * 15

    elif move.field_start == Terrain.GRASSY_TERRAIN:
        standard_unit = my_team_members[0].stats[Stat.MAX_HP] / 16 if my_team_members else 0
        opp_grass_count = sum(1 for s in all_opp_species if Type.GRASS in s.types)
        score += opp_grass_count * (standard_unit * 0.75)
        score += my_grounded * (standard_unit / 2)

    elif move.field_start == Terrain.PSYCHIC_TERRAIN:
        standard_unit = my_team_members[0].stats[Stat.MAX_HP] / 16 if my_team_members else 0
        opp_psychic_count = sum(1 for s in all_opp_species if Type.PSYCHIC in s.types)
        score += opp_psychic_count * (standard_unit * 0.75)
        has_priority = any(any(m.priority > 0 for m in p.moves) for p in my_team_members)
        if has_priority:
            score += my_grounded * 15

    elif move.field_start == Terrain.MISTY_TERRAIN:
        standard_unit = my_team_members[0].stats[Stat.MAX_HP] / 16 if my_team_members else 0
        score += my_dragon * (standard_unit / 2)
        can_status = any(
            any(m.status in {Status.SLEEP, Status.BURN, Status.TOXIC, Status.PARALYZED} for m in p.moves)
            for p in my_team_members
        )
        if can_status:
            score += my_grounded * 15

    return score


def predict_opponent_builds(
    pokemon_view: PokemonView,
    my_full_team: Team,
    all_opp_views: list[PokemonView],
    params: BattleRuleParam,
) -> list[Any]:
    """Predict likely competitive builds for a single opponent Pokemon.

    Runs moveset prediction then archetype build generation.

    Args:
        pokemon_view: The opponent's Pokemon as seen at Team Preview.
        my_full_team: Our full team of 6.
        all_opp_views: All opponent Pokemon views.
        params: Battle rule parameters.

    Returns:
        List of up to 4 Pokemon objects representing likely builds.
    """
    species = pokemon_view.species
    if not species:
        return []

    predicted_moveset = predict_moveset(species, my_full_team, all_opp_views, params)
    archetype_builds = create_archetype_builds(species, predicted_moveset)

    return [build for _, build in archetype_builds]
