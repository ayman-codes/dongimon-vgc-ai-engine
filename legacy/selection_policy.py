# --- Imports ---
import itertools
import typing
from typing import List, Tuple, Optional
import logging
import vgc2.battle_engine
from vgc2.agent import SelectionPolicy, BattlePolicy, SelectionCommand
from vgc2.battle_engine import State, BattleEngine, BattlingTeam, calculate_damage
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.battle_engine.modifiers import *
from vgc2.battle_engine.view import TeamView, StateView, PokemonView
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies, Move, BattlingPokemon
from vgc2.battle_engine.game_state import State


# --- The core logic ---
'''
- The logic here is quite simple, unlike the teambuild which will have to select pokimons from a very large pool of pokimons
- here we will be selecting from a small subset, so, what is the best way to get a very accurate estimation of which 
- of the pokimons in my pool are best, an important consideration is also synergy. The answer is quite simple, evaluate
- evey possible team combiantion against every possible team combination of the opponent, where team size or n_active = 2
- at the end the team from my side that performed the best against every possible combination of the opponent is sorted first
- and the second team and so on. If the policy proves to be too compuationally expensive, then borrowing some of the heursitic 
- functions from MyBattlePolicy, which is full of them, to provide a rough estimation of the best 80% pokimons from my and the
- opponent's side
'''

# For the simulation to work, we need to test our potential pair of pokemons against the opponent's 
# So, there are 3 attributes that we need to estimate for the opponent's team: EV's, Nature and 4 Moves
 


class Selection_Policy(SelectionPolicy):
    """ 
    A policy that dertermines the best team by running
    an internal simulation tournament between all possible
    pairs of Pokemon from each side
    """
    
    def __init__(self):
        super().__init__()
        """
        For this to work 2 conditions are must be absolutely met
        Condition 1: No random battle policy, otherwise the evaluation is pointless
        Condition 2: both my side and the opponent has to use the same battlepolicy
        """
        self.sim_battle_policy = GreedyBattlePolicy()
        
    
        
    def _generate_team_combinations(self, source_team: Team, combination_size: int) -> List[Tuple[int, ...]]:
        """
        Generate all unique combinations of Pokemon Indices from a source team
        
        Param source_team: The team from which to generate combinations
        Param combination_size: The size of each combination to generate
        Returns a list of tuples, each containing indices of the selected Pokemon
        """
        
        if len(source_team.members) < combination_size:
            #raise ValueError("Combination size exceeds the number of available Pokemon in the team.")
            return []
        
        # Get a list of indices for the team members
        member_indices = list(range(len(source_team.members)))
        
        # use Intertools to generate all combinations
        combination_iterator = itertools.combinations(member_indices, combination_size) # combinations(range(4), 3) --> (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        
        #convert the iterator to a list and return it
        return list(combination_iterator)
    
    def _create_archetype_builds(self, species: PokemonSpecies, predicted_moveset: List[Move]) -> List[Pokemon]:
        """
        Creates a list of 2-4 fully-built Pokemon objects representing common
        competitive builds, using a pre-determined moveset.
        """
        if not predicted_moveset:
            return []

        builds = []
        base_stats = species.base_stats
        
        # get IDs of the moves for the Pokemon constructor.
        move_indices = []
        for move in predicted_moveset:
            try:
                # Find the index of our predicted move in the species' master list of moves.
                idx = species.moves.index(move)
                move_indices.append(idx)
            except ValueError:
                # This should not happen if the move came from the species, but it's a safe fallback.
                #logging.warning(f"Could not find predicted move '{move.name}' in species '{species.name}'. Skipping move.")
                continue

        # --- Determine the species' primary offensive leaning ---
        phys_moves = sum(1 for m in predicted_moveset if m.category == Category.PHYSICAL)
        spec_moves = sum(1 for m in predicted_moveset if m.category == Category.SPECIAL)
        
        # A species is physical-leaning if its base Attack is higher,
        # or if stats are equal and it has more physical moves predicted.
        is_physical_leaning = (base_stats[Stat.ATTACK] > base_stats[Stat.SPECIAL_ATTACK]) or \
                              (base_stats[Stat.ATTACK] == base_stats[Stat.SPECIAL_ATTACK] and phys_moves >= spec_moves)
        
        # --- Build 1: Fast Sweeper Archetype ---
        # Maximizes Speed and the primary attacking stat.
        if is_physical_leaning:
            fast_nature = Nature.JOLLY
            fast_evs = (4, 252, 0, 0, 0, 252)  # HP, Atk, Def, SpA, SpD, Spe
        else:
            fast_nature = Nature.TIMID
            fast_evs = (4, 0, 0, 252, 0, 252)

        builds.append(Pokemon(species=species, move_indexes=move_indices, nature=fast_nature, evs=fast_evs))

        # --- Build 2: Bulky Attacker Archetype ---
        # Maximizes HP and the primary attacking stat.
        if is_physical_leaning:
            bulky_nature = Nature.ADAMANT
            bulky_evs = (252, 252, 4, 0, 0, 0)
        else:
            bulky_nature = Nature.MODEST
            bulky_evs = (252, 0, 4, 252, 0, 0)
        
        builds.append(Pokemon(species=species, move_indexes=move_indices, nature=bulky_nature, evs=bulky_evs))

        # --- Build 3: Bulky Defender Archetype ---
        # Maximizes HP and the more prominent defensive stat.
        is_physically_defensive = base_stats[Stat.DEFENSE] >= base_stats[Stat.SPECIAL_DEFENSE]
        
        if is_physically_defensive:
            def_nature = Nature.IMPISH if is_physical_leaning else Nature.BOLD
            def_evs = (252, 0, 252, 0, 4, 0)
        else: # Is specially defensive
            def_nature = Nature.CAREFUL if is_physical_leaning else Nature.CALM
            def_evs = (252, 0, 4, 0, 252, 0)

        builds.append(Pokemon(species=species, move_indexes=move_indices, nature=def_nature, evs=def_evs))
        
        # --- Build 4: Mixed Attacker Archetype (Conditional) ---
        # Only add this build if the species has viable mixed attacking stats.
        atk_stat, spa_stat = base_stats[Stat.ATTACK], base_stats[Stat.SPECIAL_ATTACK]
        if abs(atk_stat - spa_stat) < 15: # Threshold for being a "mixed" attacker
            mixed_nature = Nature.NAUGHTY if is_physical_leaning else Nature.RASH
            mixed_evs = (4, 252, 0, 252, 0, 0) # Simple mixed spread
            
            builds.append(Pokemon(species=species, move_indexes=move_indices, nature=mixed_nature, evs=mixed_evs))

        return builds
    
    def _calculate_utility_score(self, move: Move, attacker_species: PokemonSpecies, my_full_team: Team, all_opp_species_views: List[PokemonView]) -> float:
        """Calculates a 'Damage Equivalence' score for a utility move."""
        # This helper contains the logic for scoring non-damaging moves.
        # A helper function for the _predict_moveset method.
        score = 0.0
        
        my_team_members = my_full_team.members
        
        # Constants: Pkm types
        is_opp_fire = sum(1 for p in all_opp_species_views if Type.FIRE in p.species.types)
        is_me_fire = sum(1 for p in my_team_members if Type.FIRE in p.species.types)
        is_opp_water = sum(1 for p in all_opp_species_views if Type.WATER in p.species.types)
        is_me_water = sum(1 for p in my_team_members if Type.WATER in p.species.types)
        is_opp_grass = sum(1 for p in all_opp_species_views if Type.GRASS in p.species.types)
        is_me_grass = sum(1 for p in my_team_members if Type.GRASS in p.species.types)
        is_opp_electric = sum(1 for p in all_opp_species_views if Type.ELECTRIC in p.species.types)
        is_me_electric = sum(1 for p in my_team_members if Type.ELECTRIC in p.species.types)
        is_opp_psychic = sum(1 for p in all_opp_species_views if Type.PSYCHIC in p.species.types)
        is_me_psychic = sum(1 for p in my_team_members if Type.PSYCHIC in p.species.types)
        is_opp_dragon = sum(1 for p in all_opp_species_views if Type.DRAGON in p.species.types)
        is_me_dragon = sum(1 for p in my_team_members if Type.DRAGON in p.species.types)
        is_opp_rock = sum(1 for p in all_opp_species_views if Type.ROCK in p.species.types)
        is_me_rock = sum(1 for p in my_team_members if Type.ROCK in p.species.types)
        is_opp_steel = sum(1 for p in all_opp_species_views if Type.STEEL in p.species.types)
        is_me_steel = sum(1 for p in my_team_members if Type.STEEL in p.species.types)
        is_opp_ground = sum(1 for p in all_opp_species_views if Type.GROUND in p.species.types)
        is_me_ground = sum(1 for p in my_team_members if Type.GROUND in p.species.types)
        is_opp_ice = sum(1 for p in all_opp_species_views if Type.ICE in p.species.types)
        is_me_ice = sum(1 for p in my_team_members if Type.ICE in p.species.types)

        my_fastest_pkm = max(my_team_members, key=lambda p: p.stats[Stat.SPEED])
        opp_fastest_pkm = max(all_opp_species_views, key=lambda p: p.species.base_stats[Stat.SPEED])

        avg_my_speed = sum(p.stats[Stat.SPEED] for p in my_team_members) / len(my_team_members)
        avg_opp_speed = sum(p.species.base_stats[Stat.SPEED] for p in all_opp_species_views) / len(all_opp_species_views)
        
        # --- 1. Protect ---
        if move.protect:
            max_damage = 0
            # Create a defender prototype 
            defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
            neutral_state_pkm = BattlingTeam([defender_proto],[])

            for my_pkm in my_full_team.members:
                for my_move in my_pkm.moves:
                    if my_move.base_power > 0:
                        attacker_proto = BattlingPokemon(my_pkm)
                        # Create minimal state for the calculation
                        temp_state = State((BattlingTeam([attacker_proto],[]), neutral_state_pkm))
                        
                        damage = calculate_damage(self.sim_battle_policy.params, 0, my_move, temp_state, attacker_proto, defender_proto)
                        if damage > max_damage:
                            max_damage = damage
            
            # The score is the % of the attacker's HP that would be saved.
            attacker_hp = attacker_species.base_stats[Stat.MAX_HP]
            return (max_damage / attacker_hp) * 100 if attacker_hp > 0 else 0

        # --- 2. Status Moves ---
        score = 0.0
        
        # --- Burn and Toxic (Damage over time + Mitigation) ---
        if move.status == Status.BURN:
            # Passive Damage: 1/16th of max HP. set the value as its % damage.
            score += (1/16) * 100 
            
            # Damage Mitigation: How much damage does it prevent from my best physical attacker?
            my_phys_attackers = [p for p in my_full_team.members if p.stats[Stat.ATTACK] > p.stats[Stat.SPECIAL_ATTACK]]
            if my_phys_attackers:
                strongest_phys_attacker = max(my_phys_attackers, key=lambda p: p.stats[Stat.ATTACK])
                best_phys_move = max(strongest_phys_attacker.moves, key=lambda m: m.base_power if m.category == Category.PHYSICAL else -1)

                if best_phys_move.base_power > 0:
                    # Create objects for calculation
                    attacker_proto = BattlingPokemon(strongest_phys_attacker)
                    defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
                    temp_state = State((BattlingTeam([attacker_proto],[]), BattlingTeam([defender_proto],[])))
                    
                    # Calculate damage with and without burn
                    dmg_unburned = calculate_damage(self.sim_battle_policy.params, 0, best_phys_move, temp_state, attacker_proto, defender_proto)
                    
                    # To simulate burn, manually apply the modifier
                    attacker_proto.status = Status.BURN
                    dmg_burned = calculate_damage(self.sim_battle_policy.params, 0, best_phys_move, temp_state, attacker_proto, defender_proto)
                    
                    damage_prevented = dmg_unburned - dmg_burned
                    defender_hp = attacker_species.base_stats[Stat.MAX_HP]
                    score += (damage_prevented / defender_hp) * 100 if defender_hp > 0 else 0

        elif move.status == Status.TOXIC:
            # Approximate damage over 4 turns: (1/16 + 2/16 + 3/16 + 4/16) = 10/16 of max HP
            score += (10/16) * 100

        # --- Paralysis and Sleep (Turn/Damage Denial) ---
        elif move.status == Status.PARALYZED:
            # Value is the average damage lose if my fastest Pokémon gets paralyzed.
            # Paralysis has a 25% chance to prevent a move.
            fastest_pkm = max(my_full_team.members, key=lambda p: p.stats[Stat.SPEED])
            best_move = max(fastest_pkm.moves, key=lambda m: m.base_power)
            
            if best_move.base_power > 0:
                attacker_proto = BattlingPokemon(fastest_pkm)
                defender_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=[]))
                temp_state = State((BattlingTeam([attacker_proto],[]), BattlingTeam([defender_proto],[])))

                damage_potential = calculate_damage(self.sim_battle_policy.params, 0, best_move, temp_state, attacker_proto, defender_proto)
                
                # The "damage equivalent" is the potential damage * the chance of not moving.
                # it's not perfect, but whatever
                damage_denied = damage_potential * 0.25
                defender_hp = attacker_species.base_stats[Stat.MAX_HP]
                score += (damage_denied / defender_hp) * 100 if defender_hp > 0 else 0

        elif move.status == Status.SLEEP:
            # A very powerful effect. Value is denying the opponent turns.
            # base estimation: denying them 1.5 turns of their best possible damage output.
            # Find the max damage the opponent can do to any of my pokemon
            max_damage_potential = 0
            
            # Create a list of all possible move indices for this species
            all_move_indices = list(range(len(attacker_species.moves)))
            attacker_proto = BattlingPokemon(Pokemon(species=attacker_species, move_indexes=all_move_indices))
            
            for my_pkm in my_full_team.members:
                defender_proto = BattlingPokemon(my_pkm)
                temp_state = State((BattlingTeam([defender_proto],[]), BattlingTeam([attacker_proto],[])))
                for opp_move in attacker_species.moves:
                    if opp_move.base_power > 0:
                        damage = calculate_damage(self.sim_battle_policy.params, 1, opp_move, temp_state, attacker_proto, defender_proto)
                        if damage > max_damage_potential:
                            max_damage_potential = damage
            
            # Score is 1.5 turns of denied damage, as a % of my Pokemon's average HP
            avg_my_hp = sum(p.stats[Stat.MAX_HP] for p in my_full_team.members) / len(my_full_team.members)
            score += (max_damage_potential * 1.5) / avg_my_hp * 100 if avg_my_hp > 0 else 0

        # --- 3. Field Effects  ---
         # This helper function calculates the net damage gain for one team when a field effect is active.
        def get_field_effect_damage_swing(move_type_boost: Type, damage_multiplier: float, team_to_evaluate: typing.Union[List[Pokemon], List[PokemonView]]):
            net_damage_gain = 0
            for pkm in team_to_evaluate:
                # Find the best move of the boosted type for this pokemon
                best_move = None
                max_power = -1
                # If pkm is a PokemonView --> access its species
                potential_moves = pkm.moves if hasattr(pkm, 'moves') else pkm.species.moves
                
                # Filter moves to find the best one of the boosted type
                for pkm_move in potential_moves:
                    if pkm_move.pkm_type == move_type_boost and pkm_move.base_power > max_power:
                        max_power = pkm_move.base_power
                        best_move = pkm_move
                
                if best_move:
                    # Calculate damage with and without the boost to find the gain
                    # use a generic defender for this calculation
                    generic_defender_species = PokemonSpecies(base_stats=(80,80,80,80,80,80), types=[], moves=[])
                    generic_defender = BattlingPokemon(Pokemon(species=generic_defender_species, move_indexes=[]))
                    
                    # Create attacker proto, handling both Pokemon and PokemonView
                    attacker_proto = BattlingPokemon(pkm) if isinstance(pkm, Pokemon) else BattlingPokemon(Pokemon(species=pkm.species, move_indexes=[]))
                    
                    # Base damage
                    state_no_effect = State((BattlingTeam([generic_defender],[]), BattlingTeam([attacker_proto],[])))
                    base_damage = calculate_damage(self.sim_battle_policy.params, 1, best_move, state_no_effect, attacker_proto, generic_defender)
                    
                    # Damage with boost
                    boosted_damage = base_damage * damage_multiplier
                    
                    net_damage_gain += (boosted_damage - base_damage)
            return net_damage_gain

        # --- 4. Weather Moves ---
        if move.weather_start == Weather.RAIN:
            # Opponent gains from Water boost, we lose from their gain.
            opp_gain = get_field_effect_damage_swing(Type.WATER, 1.5, all_opp_species_views)
            # We gain from nerfing their Fire moves, opponent loses from our gain.
            my_gain_from_nerf = get_field_effect_damage_swing(Type.FIRE, 0.5, all_opp_species_views)
            score += (opp_gain - my_gain_from_nerf) # score what's good for the opponent

        elif move.weather_start == Weather.SUN:
            opp_gain = get_field_effect_damage_swing(Type.FIRE, 1.5, all_opp_species_views)
            my_gain_from_nerf = get_field_effect_damage_swing(Type.WATER, 0.5, all_opp_species_views)
            score += (opp_gain - my_gain_from_nerf)

        elif move.weather_start == Weather.SAND:
            # Passive damage to my non-immune members + Sp.Def boost for their Rock-types
            non_immune_me = len([p for p in my_full_team.members if not any(t in p.species.types for t in [Type.ROCK, Type.GROUND, Type.STEEL])])
            score += non_immune_me * (my_full_team.members[0].stats[Stat.MAX_HP] / 16) # Approx. HP loss
            
            opp_rock_types = len([p for p in all_opp_species_views if Type.ROCK in p.species.types])
            # A 1.5x Sp.Def boost is like a 33% reduction in special damage taken.
            # estimate as 20 damage-equivalent points per benefiting pokemon.
            score += opp_rock_types * 20

        elif move.weather_start == Weather.SNOW:
            # Passive damage to my non-immune members + Def boost for their Ice-types
            non_immune_me = len([p for p in my_full_team.members if Type.ICE not in p.species.types])
            score += non_immune_me * (my_full_team.members[0].stats[Stat.MAX_HP] / 16)
            
            opp_ice_types = len([p for p in all_opp_species_views if Type.ICE in p.species.types])
            score += opp_ice_types * 20

        # --- Terrain Moves ---
        elif move.field_start == Terrain.ELECTRIC_TERRAIN:
            # 1.3x damage boost for their Electric types
            opp_gain = get_field_effect_damage_swing(Type.ELECTRIC, 1.3, all_opp_species_views)
            score += opp_gain
            # Bonus: Blocks my Spore/Sleep Powder users
            # Damage denial is like a 1.5x damage boost for my team
            i_have_sleep_moves = any(any(m.status == Status.SLEEP for m in p.moves) for p in my_team_members)
            if i_have_sleep_moves:
                avg_hp = (sum(p.stats[Stat.MAX_HP] for p in my_team_members) / 
                          len(my_team_members)) if my_team_members else 1
                best_sleep_bonus = 0.0
                for p in my_team_members:
                    for m in p.moves:
                        if m.status == Status.SLEEP:
                            # Estimate the bonus as the normalized damage denial over 1.5 turns
                            potential = (m.base_power * p.stats[Stat.ATTACK]) / avg_hp * 1.5 * 100
                            if potential > best_sleep_bonus:
                                best_sleep_bonus = potential
                score += best_sleep_bonus
                
        elif move.field_start == Terrain.GRASSY_TERRAIN:
            # Standard Damage Unit: 1/16th of the first team member's max HP.
            standard_damage_unit = my_full_team.members[0].stats[Stat.MAX_HP] / 16
            # Count opponent Grass types and Flying types.
            is_opp_grass = sum(1 for p in all_opp_species_views if Type.GRASS in p.species.types)
            is_opp_flying = sum(1 for p in all_opp_species_views if Type.FLYING in p.species.types)
            # Boost ally Grass moves: add 75% of the standard damage unit for each opposing Grass-type.
            score += is_opp_grass * (standard_damage_unit * 0.75)
            # Passive Healing Bonus: Each grounded (non-Flying) Pokémon yields extra healing.
            score += (6 - is_opp_flying) * (standard_damage_unit / 2)

        elif move.field_start == Terrain.PSYCHIC_TERRAIN:
            standard_damage_unit = my_full_team.members[0].stats[Stat.MAX_HP] / 16
            is_opp_psychic = sum(1 for p in all_opp_species_views if Type.PSYCHIC in p.species.types)
            is_opp_flying = sum(1 for p in all_opp_species_views if Type.FLYING in p.species.types)
            # Boost ally Psychic moves similarly.
            score += is_opp_psychic * (standard_damage_unit * 0.75)
            # Priority Protection: If any of my Pokémon have priority moves, add bonus per protected grounded Pokémon.
            can_i_use_priority = any(any(m.priority > 0 for m in p.moves) for p in my_team_members)
            if can_i_use_priority:
                score += (6 - is_opp_flying) * 15

        elif move.field_start == Terrain.MISTY_TERRAIN:
            standard_damage_unit = my_full_team.members[0].stats[Stat.MAX_HP] / 16
            # Defensive bonus: penalize our Dragon-type attackers.
            is_me_dragon = sum(1 for p in my_team_members if Type.DRAGON in p.species.types)
            score += is_me_dragon * (standard_damage_unit / 2)
            # Status Protection: if any of our Pokémon can inflict status (e.g., Will-O-Wisp, Spore), add bonus.
            can_i_use_status = any(any(m.status in {Status.SLEEP, Status.BURN, Status.TOXIC, Status.PARALYZED} for m in p.moves) for p in my_team_members)
            is_opp_flying = sum(1 for p in all_opp_species_views if Type.FLYING in p.species.types)
            if can_i_use_status:
                score += (6 - is_opp_flying) * 15

        return score

    
    def _predict_moveset(self, attacker_species: PokemonSpecies, my_full_team: Team, all_opp_species_views: List[PokemonView]) -> List[Move]:
        """
        Analyzes a species' movepool and predicts its 4 best moves by scoring
        both damaging and utility moves against our team and considering opponent synergy.
        The initial problem with this heuristic is that it does not consider the opponent's team
        and as such, V2.0 will now consider the opponent's every possible Pokemon team pair
        then instead of doing a simulation, that would be too computationally expensive,
        I will score the damage calculated by the damage calculator, for each move twice,
        once without considering the opponent's partner using utility and the second time
        we use the remaning HP or the extra damage done due to the partner's presence.
        and we add the difference of the score to damage score.
        """
        if not attacker_species.moves:
            return []

        move_scores = {move: 0.0 for move in attacker_species.moves}

        # 1. Generate the archetype builds for the attacker.
        #    We pass a placeholder moveset initially. to solve the stupid chicke and egg problem
        archetype_builds = self._create_archetype_builds(attacker_species, attacker_species.moves)
        if not archetype_builds: return [] # Cannot proceed if no builds are generated

         # --- Create a single, structurally complete, but neutral State object ---
        # This will be reused for all damage calculations.

        # 1. Create a single, placeholder Pokemon to safely initialize a BattlingTeam.
        dummy_species = PokemonSpecies(base_stats=(1, 1, 1, 1, 1, 1), types=[], moves=[])
        dummy_pokemon = Pokemon(species=dummy_species, move_indexes=[])

        # 2. Create a valid BattlingTeam with this dummy, then clear it.
        dummy_team = BattlingTeam(active=[dummy_pokemon], reserve=[])
        dummy_team.active = []

        # 3. Create another valid team for our side (this one isn't empty).
        my_battling_team = BattlingTeam(active=[p for p in my_full_team.members], reserve=[])
        
        # 4. Create the final neutral_state object. It is now structurally complete.
        neutral_state = State((my_battling_team, dummy_team))

        # Pass 2: Iterate through each potential move to score it.
        for move in attacker_species.moves:
            # --- Handle Utility Moves ---
            # Their score is not dependent on the attacker's stats, so we calculate it once.
            if move.base_power == 0 and move.category == Category.OTHER:
                move_scores[move] = self._calculate_utility_score(move, attacker_species, my_full_team, all_opp_species_views)
                continue

            # --- Handle Damaging Moves ---
            total_damage_across_builds = 0.0
            
            # Iterate through each archetype build to get an average score.
            for attacker_build in archetype_builds:
                # This is the core of the "Archetype-Aware" logic.
                # We use the specific stats of each build for the calculation.
                attacker_prototype = BattlingPokemon(attacker_build)
                
                total_damage_for_this_build = 0.0
                for my_pokemon in my_full_team.members:
                    defender_prototype = BattlingPokemon(my_pokemon)
                    
                    # We can reuse the same neutral_state for all calculations.
                    damage = calculate_damage(
                        params=self.sim_battle_policy.params,
                        attacking_side=1, move=move,
                        state=neutral_state,
                        attacker=attacker_prototype,
                        defender=defender_prototype
                    )
                    total_damage_for_this_build += (damage / my_pokemon.stats[Stat.MAX_HP]) * 100 if my_pokemon.stats[Stat.MAX_HP] > 0 else 0
                
                # Average damage for this specific build against our whole team
                avg_damage_for_build = total_damage_for_this_build / len(my_full_team.members) if my_full_team.members else 0
                total_damage_across_builds += avg_damage_for_build

            # The final score for the damaging move is the average across all archetype builds.
            move_scores[move] = total_damage_across_builds / len(archetype_builds) if archetype_builds else 0.0

        # Sort all moves by their final calculated score and return the top 4.
        sorted_moves = sorted(move_scores.keys(), key=lambda m: move_scores[m], reverse=True)
        return sorted_moves[:4]
                        
    def _predict_opponent_builds(self, pokemon_view: PokemonView, my_full_team: Team, all_opp_views: List[PokemonView]) -> List[Pokemon]:
        """
        Orchestrates the prediction process for a single opponent's Pokemon
        :param pokemon_view: The opponent's Pokémon as seen from Team Preview.
        :param my_full_team: Our own full team of 6, used to inform the prediction.
        :return: A list of 2-3 fully-built, predicted Pokemon objects.
        """
        
        # The pokemon_view gives access to the species base stats and full movepool
        species = pokemon_view.species
        if not species:
            return [] # safeguard
        
        # 1. Predict the best 4 movers for this opponent's species to use against our team
        predicted_moveset = self._predict_moveset(species, my_full_team, all_opp_views)

        # 2. Generate Several archetype builds based on the predicted moveset
        archetype_builds = self._create_archetype_builds(species, predicted_moveset)
        
        return archetype_builds               
                        
    def _run_sub_tournament(self, my_full_team: Team, my_pair_indices: Tuple[int, ...],
                             opp_view_pair: Tuple[PokemonView, ...],
                             predicted_builds_dict: dict) -> float:
        """
        Runs a sub-tournament for one of my pairs against all predicted build
        combinations of an opponent's pair.

        :param my_full_team: My full team, as choosen by Teambuild Policy.
        :param my_pair_indices: The indices of my two Pokémon for this matchup.
        :param opp_view_pair: A tuple of the two opponent PokemonView objects.
        :param predicted_builds_dict: The dictionary containing all predicted builds.
        :return: The average win rate for my pair in this sub-tournament.
        """
        
        # Get a list of the predicted builds for the two opponent's Pokémon we are facing
        opp_build_A = predicted_builds_dict.get(opp_view_pair[0], [])
        opp_build_B = predicted_builds_dict.get(opp_view_pair[1], [])
        
        # Limitation: What if I can't simulate against a predictedited build for one of the opponent's Pokémon?
        # For now, set as Zero, TODO: Check for possible solutions
        if not opp_build_A or not opp_build_B:
            #logging.warning(f"Missing predicted builds for opponent's Pokémon: {opp_view_pair}")
            return 0.0
        
        sub_wins = 0
        sub_battles = 0
        
        # Create all combinations of the predicted builds
        build_matchups = itertools.product(opp_build_A, opp_build_B) # product(A, B) returns the same as: ((x,y) for x in A for y in B).
        
        for opp_build_A, opp_build_B in build_matchups:
            # Create a new team for the opponent with the two predicted builds
            opp_predicted_pair = [opp_build_A, opp_build_B]
            
            # Create the battling team (my team)
            my_pair_pokemon = [my_full_team.members[i] for i in my_pair_indices] 
            my_battling_team = BattlingTeam(active=my_pair_pokemon, reserve=[])
            
            # Creaete the opponetn's battling team from the predicted builds 
            opp_battling_team = BattlingTeam(active=opp_predicted_pair, reserve=[])
            
            # Create a battleengine and run
            intital_state = State((my_battling_team, opp_battling_team))
            engine = BattleEngine(intital_state)
            
            # Instance of TeamViews for the GreedyBattlePolicy (doesn't need them but oh well)
            dummy_my_view = TeamView(my_full_team)     
            dummy_opp_view = TeamView(Team(members=opp_predicted_pair))
            
            while not engine.finished():
                state_view_p0 = StateView(engine.state, 0, (dummy_my_view, dummy_opp_view))
                state_view_p1 = StateView(engine.state, 1, (dummy_opp_view, dummy_my_view))
                
                cmd_p0 = self.sim_battle_policy.decision(state_view_p0)
                cmd_p1 = self.sim_battle_policy.decision(state_view_p1)
                
                engine.run_turn((cmd_p0, cmd_p1))
                
            if engine.winning_side == 0:
                sub_wins += 1
            sub_battles += 1
            
        # Return the average win rate for ouur pair against this opponent's pair
        return sub_wins / sub_battles if sub_battles > 0 else 0.0
    
    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        """
        Orchestrates the pair-vs-pair tournament and selects the final team of 4.
        The decision method will loop through the opponent's PokemonView of their choosen team,
        and call the _predict_opponent_builds method to get a list of likely builds for each opponent's Pokemon.
        The result will be stored in a dictionary, to access them during the simulation
        """
        my_full_team, opp_team_view = teams
        #opp_full_team = opp_team_view._team previously used to test the first phase of the logic (simulation)
        n_active = 2

        # --- 1. Prediction Phase ---
        # Key: The opponent's PokemonView
        # Value: A list of predicted builds for that Pokemon
        # Store the predictions: {Pokemon_view: [Build1, Build2, ...]}
        predicted_opponent_builds = {}
        
        # get a list of all opponent's Pokemon species for synergy calculations
        all_opp_species_view = opp_team_view.members
        #logging.info(f'--- Predicting the opponent\'s team builds ---')
        for opp_pkm_view in all_opp_species_view:
            # Predict builds for each opponent's Pokemon
            predicted_builds = self._predict_opponent_builds(
            pokemon_view=opp_pkm_view,
            my_full_team=my_full_team,
            all_opp_views = all_opp_species_view
            )
            predicted_opponent_builds[opp_pkm_view] = predicted_builds
            # NOTE: REMEMBER TO REMOVE LATER BEFORE YOU SUBMIT
            #logging.info(f" -> For {opp_pkm_view.species.name}, predicted {len(predicted_builds)} builds")
        
         
        # --- 2. Simulation Phase ---        
        my_potential_pairs = self._generate_team_combinations(my_full_team, n_active)
        opp_potential_pairs = list(itertools.combinations(all_opp_species_view, n_active))
        
        if not my_potential_pairs:
            #logging.warning("Could not generate potential pairs for my team. Returning default.")
            return SelectionCommand(list(range(min(max_size, len(my_full_team.members)))))
        
        # The opponent's potential pairs will be the combination of their PokemonView objects
        opp_potential_pairs = list(itertools.combinations(all_opp_species_view, n_active))
        
        # The beginning of the tournament
        results = {pair: 0.0 for pair in my_potential_pairs} # store winrates
        
        # Logging
        #logging.info(f"--- Running Main Simulation Tournament ---")
        #logging.info(f" My Pairs: {len(my_potential_pairs)}, opponent's Pairs: {len(opp_potential_pairs)}")
        
        # For each of my Pairs
        for i, my_pair in enumerate(my_potential_pairs):
            total_win_rate_for_my_pair = 0.0
            
            # ...test it against every possible opposing pair.
            for opp_pair_view in opp_potential_pairs:
                # Run the sub-tournament for this specific matchup.
                win_rate = self._run_sub_tournament(
                    my_full_team,
                    my_pair,
                    opp_pair_view,
                    predicted_opponent_builds
                )
                total_win_rate_for_my_pair += win_rate
            
            # The final score for my pairs is its average win rate across all possible opponent pairs.
            if opp_potential_pairs:
                results[my_pair] = total_win_rate_for_my_pair / len(opp_potential_pairs)
            
            #logging.info(f"  > Pair {i+1}/{len(my_potential_pairs)} evaluated. Avg Win Rate: {results[my_pair]:.2%}")
        
        
        # --- 3. Final Ranking ---
        # Yes I will die for you baby, but you won't do the same ~ Bruno Mars
        # Sort my pairs by their final calculated average win rates
        ranked_pairs = sorted(results.keys(), key=lambda p: results[p], reverse=True)
        
        # Understand how many paris we need to select, according to max_size
        num_paris_to_select = max_size // n_active # max_size = 4 and n_active = 2, so we need 2 pairs. works!
        
        if len(ranked_pairs) < num_paris_to_select:
            #logging.error(f"Not enough unique Pairs to form a full team of {max_size}, please don't activate")
            # If true: Fallback to take all available pairs based on their ranking to fill the remaining slots
            final_selection = []
            for pair in ranked_pairs:
                final_selection.extend(list(pair))
            remaining_indices = [i for i in range(len(my_full_team.members)) if i not in final_selection]
            final_selection.extend(remaining_indices)
            return SelectionCommand(final_selection[:max_size]) # Unpack the pair indices
                
        
        # Build the final command list by taking the top N pairs for my ranked list
        # Build the final team selection from the top-ranked pairs
        final_selection = []
        #logging.info(f"--- Final Team Selection ---")
        for i in range(num_paris_to_select):
            pair_indices = ranked_pairs[i]
            final_selection.extend(list(pair_indices))
            
            pair_names = [my_full_team.members[j].species.name for j in pair_indices]
            win_rate = results[pair_indices]
            #if i == 0:
                #logging.info(f"  > Leads (Top Pair): {pair_names} (Avg. Win Rate: {win_rate:.2%})")
            #else:
                #logging.info(f"  > Reserve Pair #{i+1}: {pair_names} (Avg. Win Rate: {win_rate:.2%})")

        return SelectionCommand(final_selection)
 