# Imports
import random

import pandas as pd
from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.balance.meta import Meta, Roster

# VGC2 Framework Imports
from vgc2.battle_engine.damage_calculator import *
from vgc2.battle_engine.game_state import State
from vgc2.battle_engine.modifiers import *
from vgc2.battle_engine.move import *
from vgc2.battle_engine.pokemon import *
from vgc2.battle_engine.team import BattlingTeam
from vgc2.battle_engine.view import *

'''
HESF is a policy that works in 3 stages:
1.2 Stage 1: Heurstic Funnel V1, to build the Pokemon Object from the pokemon Speices and select the max_team_size
1.2 Stage 1.1: Heursitic Funnel V2, to build pokemon objects from the pokemon speices and return n% elite pokemons
2.1 Stage 2: Evolutionary algorithm V1 to determine the best synergy and return the elite pokemons of the max_team_size
2.2 Stage 2: Evolutionary algorithm V2 to determine the best synergy and return the elite pokemons of n% max_team_size
3.1 Stage 3: Simulation V1 to simulate the battle with the elite pokemons and return the best teams of max_team_size
'''

# --- Constants ---
#N_ELITE_POOL_SIZE = 70
DEBUG = False # Set to False to disable detailed console logging
N_MASTER_SQUAD_SIZE = 12
POPULATION_SIZE = 50
NUM_GENERATIONS = 30
ELITE_SQUAD_SELECTION_SIZE = 10
MUTATION_RATE = 0.1

class hesf_TeamBuildPolicy(TeamBuildPolicy):
    """_summary_
    Implements the Heuristic evolutionary simulation funnel (HESF) team building policy.
    - Stage 1: Prunes the Roster to an elite pool based on individual Pokemon attributes.
    - Stage 2: Uses EA to find a synergistic master squad from the elite pool.
    - Stage 3: runs a simulation tournament to select the best final team from the master squad.
    """

    def __init__(self):
        super().__init__()
        self.sim_battle_policy = GreedyBattlePolicy() # will be used in Stage 3 to simulate battles
        self.pruning_percentage = 0.3 # Percentage of the roster to prune
        self.hazard_removal_moves = {"rapid spin", "defog", "mortal spin", "tidy up"}  # Set of known hazard removal moves
        self.normalization_sample_size = 800
        # NOTE: self.selection_helper can be Initiated when needed
        if DEBUG:
            self.debug_data = []
            # Prepare the CSV file with headers
            try:
                with open("hesf_debug_log.csv", "w", newline='') as f:
                    # Use pandas to easily write header
                        pd.DataFrame(columns=[
                        'Species', 'Archetype', 'FinalFitness',
                        'StatScore_Raw', 'SpeedStat_Raw', 'DmgScore_Raw', 'UtilScore_Raw', 'StatSyn_Raw', 'SpeedSyn_Raw',
                        'StatScore_Norm', 'SpeedStat_Norm', 'DmgScore_Norm', 'UtilScore_Norm', 'StatSyn_Norm', 'SpeedSyn_Norm',
                        'StatScore_Weighted', 'SpeedStat_Weighted', 'DmgScore_Weighted', 'UtilScore_Weighted', 'StatSyn_Weighted', 'SpeedSyn_Weighted'
                    ]).to_csv(f, index=False)
            except OSError as e:
                print(f"Error initializing debug log file: {e}")

    def _create_archetype_builds(self, species: PokemonSpecies, predicted_moveset: list[Move]) -> list[tuple[str, Pokemon]]:
        """
        Generates up to 10 battle-ready Pokemon objects, each tagged with its
        archetype name. This change makes the system more robust by avoiding
        reverse-engineering the role from EV spreads.
        """
        if not predicted_moveset: return []

        move_indices = []
        for move in predicted_moveset:
            try:
                idx = species.moves.index(move)
                move_indices.append(idx)
            except ValueError:
                continue
        if not move_indices: return []

        builds: list[tuple[str, Pokemon]] = []
        base_stats = species.base_stats
        DEFAULT_IVS = (31, 31, 31, 31, 31, 31)
        LEVEL = 50
        is_physical_lean = base_stats[Stat.ATTACK] >= base_stats[Stat.SPECIAL_ATTACK]

        # Each archetype is now a tuple of (name, Pokemon_object)
        builds.append(("Fast Physical Sweeper", Pokemon(species, move_indices, LEVEL, (4, 252, 0, 0, 0, 252), DEFAULT_IVS, Nature.JOLLY)))
        builds.append(("Fast Special Sweeper", Pokemon(species, move_indices, LEVEL, (4, 0, 0, 252, 0, 252), DEFAULT_IVS, Nature.TIMID)))
        builds.append(("Bulky Physical Attacker", Pokemon(species, move_indices, LEVEL, (252, 252, 4, 0, 0, 0), DEFAULT_IVS, Nature.ADAMANT)))
        builds.append(("Bulky Special Attacker", Pokemon(species, move_indices, LEVEL, (252, 0, 4, 252, 0, 0), DEFAULT_IVS, Nature.MODEST)))
        builds.append(("Physically Defensive Wall", Pokemon(species, move_indices, LEVEL, (252, 0, 252, 0, 4, 0), DEFAULT_IVS, Nature.IMPISH if is_physical_lean else Nature.BOLD)))
        builds.append(("Specially Defensive Wall", Pokemon(species, move_indices, LEVEL, (252, 0, 4, 0, 252, 0), DEFAULT_IVS, Nature.CAREFUL if is_physical_lean else Nature.CALM)))

        if abs(base_stats[Stat.ATTACK] - base_stats[Stat.SPECIAL_ATTACK]) <= 20:
            builds.append(("Fast Mixed Attacker", Pokemon(species, move_indices, LEVEL, (0, 252, 0, 4, 0, 252), DEFAULT_IVS, Nature.NAIVE)))
            builds.append(("Fast Mixed Attacker", Pokemon(species, move_indices, LEVEL, (0, 4, 0, 252, 0, 252), DEFAULT_IVS, Nature.HASTY)))
            builds.append(("Bulky Mixed Attacker", Pokemon(species, move_indices, LEVEL, (252, 252, 0, 4, 0, 0), DEFAULT_IVS, Nature.NAUGHTY)))
            builds.append(("Bulky Mixed Attacker", Pokemon(species, move_indices, LEVEL, (252, 4, 0, 252, 0, 0), DEFAULT_IVS, Nature.RASH)))

        return builds

    def _get_optimal_archetype(self, species: PokemonSpecies, roster: Roster, global_max_scores: dict[str, float]) -> Pokemon:
        """
        Determines the single best competitive build for a species by applying a
        unified, weighted fitness function to all plausible archetypes. This function
        evaluates builds based on stat efficiency, speed, raw damage output,
        utility, and specific strategic synergies.

        :param species: The PokemonSpecies to build.
        :param roster: The full Roster for evaluation context.
        :return: A single, fully-optimized Pokemon object representing the best build.
        """
        # --- Pre-computation for Normalization ---
        generic_builds = [self._create_generic_build_for_species(s) for s in roster]
        max_speed_in_roster = max(b.stats[Stat.SPEED] for b in generic_builds) if generic_builds else 1.0
        if max_speed_in_roster == 0: max_speed_in_roster = 1.0

        # --- Pass A: Pre-calculate scores for all potential builds ---
        placeholder_moves = species.moves[:4] if species.moves else []
        potential_builds_with_names = self._create_archetype_builds(species, placeholder_moves)

        if not potential_builds_with_names:
            return self._create_generic_build_for_species(species)

        build_evaluations = []
        for archetype_name, temp_build in potential_builds_with_names:
            optimal_moves, all_move_scores = self._get_role_aware_moveset(
                temp_build, archetype_name, roster
            )

            # CORRECTED: Aggregate each disaggregated score component separately.
            total_damage = sum(all_move_scores[m]["damage"] for m in optimal_moves)
            total_utility = sum(all_move_scores[m]["utility"] for m in optimal_moves)
            total_stat_syn = sum(all_move_scores[m]["stat_syn"] for m in optimal_moves)
            total_speed_syn = sum(all_move_scores[m]["speed_syn"] for m in optimal_moves)

            build_evaluations.append({
                "name": archetype_name,
                "build": temp_build,
                "optimal_moves": optimal_moves,
                "stat_score": self._calculate_stat_compatibility(species, temp_build.evs),
                "damage_score": total_damage,
                "utility_score": total_utility,
                "stat_syn_score": total_stat_syn,
                "speed_syn_score": total_speed_syn,
                "speed_stat_score": temp_build.stats[Stat.SPEED]
            })

        if not build_evaluations:
            return self._create_generic_build_for_species(species)

        # Find max values for each component stream for normalization.
        # Local max is a bad idea, for proper normalization I need to use the GLOBAL MAX.
        #max_stat = max(ev["stat_score"] for ev in build_evaluations) or 1.0
        #max_dmg = max(ev["damage_score"] for ev in build_evaluations) or 1.0
        #max_util = max(ev["utility_score"] for ev in build_evaluations) or 1.0
        #max_stat_syn = max(ev["stat_syn_score"] for ev in build_evaluations) or 1.0
        #max_speed_syn = max(ev["speed_syn_score"] for ev in build_evaluations) or 1.0

        best_build_info = None
        max_fitness_score = -float('inf')

        W_STAT = 0.2
        W_SPEED = 0.2
        W_DMG = 0.3
        W_UTIL = 0.2
        W_STAT_SYN = 0.05
        W_SPEED_SYN = 0.05

        for eval_item in build_evaluations:
            # Normalize each component using the pre-calculated global maximums.
            norm_stat = eval_item["stat_score"] / global_max_scores["max_stat"]
            norm_dmg = eval_item["damage_score"] / global_max_scores["max_dmg"]
            norm_util = eval_item["utility_score"] / global_max_scores["max_util"]
            norm_speed_stat = eval_item["speed_stat_score"] / global_max_scores["max_speed_stat"]
            norm_stat_syn = eval_item["stat_syn_score"] / global_max_scores["max_stat_syn"] if global_max_scores["max_stat_syn"] > 0 else 0
            norm_speed_syn = eval_item["speed_syn_score"] / global_max_scores["max_speed_syn"] if global_max_scores["max_speed_syn"] > 0 else 0

            current_fitness_score = (norm_stat * W_STAT) + (norm_speed_stat * W_SPEED) + \
                                    (norm_dmg * W_DMG) + (norm_util * W_UTIL) + \
                                    (norm_stat_syn * W_STAT_SYN) + (norm_speed_syn * W_SPEED_SYN)

            if DEBUG:
                # Append detailed data for every evaluated archetype to the debug list for CSV export.
                self.debug_data.append({
                    'Species': species.name,
                    'Archetype': eval_item["name"],
                    'FinalFitness': current_fitness_score,
                    'StatScore_Raw': eval_item["stat_score"],
                    'SpeedStat_Raw': eval_item["speed_stat_score"],
                    'DmgScore_Raw': eval_item["damage_score"],
                    'UtilScore_Raw': eval_item["utility_score"],
                    'StatSyn_Raw': eval_item["stat_syn_score"],
                    'SpeedSyn_Raw': eval_item["speed_syn_score"],
                    'StatScore_Norm': norm_stat,
                    'SpeedStat_Norm': norm_speed_stat,
                    'DmgScore_Norm': norm_dmg,
                    'UtilScore_Norm': norm_util,
                    'StatSyn_Norm': norm_stat_syn,
                    'SpeedSyn_Norm': norm_speed_syn,
                    'StatScore_Weighted': norm_stat * W_STAT,
                    'SpeedStat_Weighted': norm_speed_stat * W_SPEED,
                    'DmgScore_Weighted': norm_dmg * W_DMG,
                    'UtilScore_Weighted': norm_util * W_UTIL,
                    'StatSyn_Weighted': norm_stat_syn * W_STAT_SYN,
                    'SpeedSyn_Weighted': norm_speed_syn * W_SPEED_SYN
                })

            if current_fitness_score > max_fitness_score:
                max_fitness_score = current_fitness_score
                best_build_info = eval_item

        if best_build_info:
            final_build_moves = best_build_info["optimal_moves"]
            move_indices = [species.moves.index(m) for m in final_build_moves if m in species.moves]
            base_build = best_build_info["build"]

            return Pokemon(species, move_indices, base_build.level, base_build.evs,
                        base_build.ivs, base_build.nature)
        else:
            return self._create_generic_build_for_species(species)

    def _calculate_1v1_net_score(self, build_A: Pokemon, build_B: Pokemon) -> float:
        """
        Calculates the "Net Damage Potential" for a 1v1 matchup between two
        fully-formed Pokemon builds. The score is from the perspective of build_A.

        :param build_A: The first Pokemon object.
        :param build_B: The second Pokemon object.
        :return: A float score representing the net advantage in the matchup.
        """
        # A simplified context for single-opponent damage calculation.
        # pass a list containing only the opponent's species and no cache.
        roster_context_B = [build_B.species]
        roster_context_A = [build_A.species]

        # Score all of A's damaging moves against B and find the best one.
        move_scores_A = {
            m: self._calculate_damage_score(build_A, m, roster_context_B, None)
            for m in build_A.moves if m.base_power > 0
        }
        damage_A_on_B = max(move_scores_A.values()) if move_scores_A else 0.0

        # Score all of B's damaging moves against A and find the best one.
        move_scores_B = {
            m: self._calculate_damage_score(build_B, m, roster_context_A, None)
            for m in build_B.moves if m.base_power > 0
        }
        damage_B_on_A = max(move_scores_B.values()) if move_scores_B else 0.0

        return damage_A_on_B - damage_B_on_A

    def _calculate_stat_compatibility(self, species: PokemonSpecies, evs: Stats) -> float:
        """
        Calculates a score representing how well an EV spread complements a species'
        best non-HP base stats. This replaces the dot-product method to prevent
        score inflation and better reflects competitive building.
        """
        base_stats = species.base_stats

        # Pair the 5 non-HP stats with their indices (1-5) and sort by base stat value
        indexed_stats = sorted([(base_stats[i], i) for i in range(1, 6)], reverse=True)

        # Get the index of the best and second-best stat (e.g., Stat.ATTACK, Stat.SPEED)
        best_stat_index = indexed_stats[0][1]
        second_best_stat_index = indexed_stats[1][1]

        # The score is the EV investment in the best stat, plus half the investment
        # in the second-best stat. This rewards focused builds and produces a score
        # on a much smaller, more comparable scale
        # add a small value for HP investment.
        score = (evs[best_stat_index] * 1.0) + \
                (evs[second_best_stat_index] * 0.5) + \
                (evs[Stat.MAX_HP] * 0.25)

        return score

    def _get_type_effectiveness(self, move_type: Type, defending_types: list[Type]) -> float:
        """
        Calculates the combined type effectiveness multiplier for a move against a target.

        :param move_type: The Type of the incoming attack.
        :param defending_types: A list of the defender's Types.
        :return: The final damage multiplier (e.g., 2.0, 0.5, 4.0).
        """
        # Access the type chart from the battle rule parameters.
        type_chart = self.sim_battle_policy.params.DAMAGE_MULTIPLICATION_ARRAY
        modifier = 1.0

        # Multiply the effectiveness for each of the defender's types.
        for defending_type in defending_types:
            # The .value of the enum corresponds to its index in the chart.
            modifier *= type_chart[move_type.value][defending_type.value]

        return modifier

    def _calculate_damage_score(self, attacker_build: Pokemon, move_to_score: Move,
                                roster: Roster, optimal_builds_cache: dict[PokemonSpecies, Pokemon] | None) -> float:
        """calculate a  score for any single damaging move against the entire field of available Opponent Pokemons

        Args:
            attacker_build (Pokemon): The best archetype build for the attacker
            move_to_score (Move): selected move to score
            roster (Roster): the full roster of available PokemonSpecies
            optimal_builds_cache (Dict[PokemonSpecies, Pokemon]): save optimal builds for effeciency, I am running out of computational resources :'(

        Returns:
            float: score representing the effectiveness of the move 
        """
        # Failsafe check: Effeciency
        if move_to_score.base_power == 0: return 0.0

        # --- Initialize The Constants ---
        total_normalized_damage = 0.0
        opp_count = 0

        # Creating the Attacker Pokemon object
        attack_pkm = BattlingPokemon(attacker_build)

        # Create a shell team and state to populate testing objects
        my_team_shell = BattlingTeam(active=[attack_pkm], reserve=[])

        # The opponent active Pokemon will be slotted here
        opp_team_shell = BattlingTeam(active=[None], reserve=[])
        state_shell = State((my_team_shell, opp_team_shell))

        # --- Iterate through all potential Opponents in the Roster ---
        for defender_spc in roster:
            # Effeciency Check
            if defender_spc is attacker_build.species:
                continue


            # If the cache is available, use the optimal build. Otherwise, create a generic one.
            if optimal_builds_cache:
                defender_build = optimal_builds_cache.get(defender_spc)# Retrieve the pre-calculated best build for the defender from the cache
            else: defender_build = self._create_generic_build_for_species(defender_spc)

            if not defender_build:
                #logging.critical(f"Could not find optimal build for defender species {defender_spc.name}.")
                continue

            defender_battle_pkm = BattlingPokemon(defender_build) # Opponent Pokemon object
            state_shell.sides[1].team.active[0] = defender_battle_pkm  # Set the opponent's active Pokemon

            # --- Damage Calculations and Normalization ---
            dmg = calculate_damage(
                params=self.sim_battle_policy.params,
                attacking_side=0,
                move=move_to_score,
                state=state_shell,
                attacker=attack_pkm,
                defender=defender_battle_pkm
            )

            defender_max_hp = defender_build.stats[Stat.MAX_HP]
            normalized_damage = (dmg / defender_max_hp) * 100.0 if defender_max_hp > 0 else 0.0
            total_normalized_damage += normalized_damage
            opp_count += 1

        if opp_count == 0: return 0.0
        return total_normalized_damage / opp_count

    def _calculate_utility_score(self, attacker_build: Pokemon, move_to_score: Move, roster: Roster) -> float:
        """Calcuates a "Damage Equivalance" score for a non-damaging move (utility moves)
        moves to calculate Healing, setting screens, placting hazards, etc.

        Args:
            attacker_build (Pokemon): The Pokemon build using the move.
            move_to_score (Move): The utility Move to be evaluated
            roster (Roster): The full list of PokemonSpecies for context.
            optimal_builds_cache (Dict[PokemonSpecies, Pokemon]): A dictionary mapping species to their optimal builds.

        Returns:
            float: A float score representing the move's utility.
        """

        if not roster: return 0.0

        # Create a local cache of generic builds for all species in the roster.
        generic_builds_cache = {s: self._create_generic_build_for_species(s) for s in roster}

        # Calculate local averages based on this generic cache.
        avg_roster_hp = sum(b.stats[Stat.MAX_HP] for b in generic_builds_cache.values()) / len(roster)
        avg_def = sum(b.stats[Stat.DEFENSE] for b in generic_builds_cache.values()) / len(roster)
        avg_spdef = sum(b.stats[Stat.SPECIAL_DEFENSE] for b in generic_builds_cache.values()) / len(roster)
        local_avg_defenses = {'def': avg_def, 'spdef': avg_spdef}

        # --- Recovery Moves ---
        if move_to_score.heal > 0:

            hp_restored = attacker_build.stats[Stat.MAX_HP] * move_to_score.heal
            # Calculate the percentage of HP restored
            normalized_hp_restored = (hp_restored / attacker_build.stats[Stat.MAX_HP]) * 100.0

            # Defensive multiplier based on the average defenses of the roster
            avg_total_defenses = local_avg_defenses['def'] + local_avg_defenses['spdef']
            pkm_total_defenses = attacker_build.stats[Stat.DEFENSE] + attacker_build.stats[Stat.SPECIAL_DEFENSE]

            if avg_total_defenses > 0:
                defensive_multiplier = 1.0 + ((pkm_total_defenses - avg_total_defenses) / avg_total_defenses)
            else:
                defensive_multiplier = 1.0

            final_multiplier = max(0.5, defensive_multiplier)
            return normalized_hp_restored * final_multiplier

        # --- Screen Moves ---
        if move_to_score.toggle_reflect or move_to_score.toggle_lightscreen:
            is_reflect = move_to_score.toggle_reflect
            total_dmg_avoid = 0.0

            # Identify relevant threatws from roster based on screen type
            threat_list = [
                s for s in roster if
                (is_reflect and s.base_stats[Stat.ATTACK] > s.base_stats[Stat.SPECIAL_ATTACK]) or
                (not is_reflect and s.base_stats[Stat.SPECIAL_ATTACK] > s.base_stats[Stat.ATTACK])
            ]
            if not threat_list: return 0.0

            # The `Team` to protect is the entire roster, to score general utility
            for teamm8 in roster:
                teamm8_build = generic_builds_cache.get(teamm8) # Retrieve the pre-calculated best build for the teamm8
                if not teamm8_build: continue

                avg_avoid_dmg_teamm8 = 0.0
                for threat_spc in threat_list:
                    # Create the optimal build for teamm8
                    threat_build = generic_builds_cache.get(threat_spc)
                    if not threat_build or threat_build == teamm8_build: continue

                    # Find threat's best relevant move
                    best_move = 0.0
                    for threat_move in threat_build.moves:
                        if (is_reflect and threat_move.category == Category.PHYSICAL) or \
                            (not is_reflect and threat_move.category == Category.SPECIAL):
                            dmg = calculate_damage(
                                params=self.sim_battle_policy.params,
                                attacking_side=0,
                                move=threat_move,
                                state=State((BattlingTeam(active=[BattlingPokemon(threat_build)], reserve=[]),
                                                        BattlingTeam(active=[BattlingPokemon(teamm8_build)], reserve=[]))),
                                attacker=BattlingPokemon(threat_build),
                                defender=BattlingPokemon(teamm8_build)
                            )
                            if dmg > best_move:
                                best_move = dmg

                    avg_avoid_dmg_teamm8 += best_move * (1/2) # Screens mitigate dmg 0.5 in doubles
                if threat_list:
                    total_dmg_avoid += avg_avoid_dmg_teamm8 / len(threat_list)

            # Apply a duration factor to represent the move's multi-turn benefit.
            DURATION_FACTOR = 2.5
            average_dmg_avoided = total_dmg_avoid / len(roster) if roster else 0.0
            return average_dmg_avoided * DURATION_FACTOR

        # --- Hazard Setting M9oves ---
        # Define a list of known Hazard removing moves

        elif move_to_score.name.lower() in self.hazard_removal_moves:
            total_dmg_avoid = 0.0
            # Calculate how much dmg stealth rock  rock does to the roster
            for teamm8 in roster:
                effectiveness = self._get_type_effectiveness(Type.ROCK, teamm8.types) # Get effectiveness against the teamm8
                dmg = (teamm8.base_stats[Stat.MAX_HP] / 8.0) * effectiveness
                total_dmg_avoid += dmg
            # Return the average value of removing hazards.
            return total_dmg_avoid / len(roster) if roster else 0.0

        elif move_to_score.hazard == Hazard.STEALTH_ROCK:
            total_hazard_damage = 0.0
            for opp_species in roster:
                # Use the new helper to get the correct effectiveness.
                effectiveness = self._get_type_effectiveness(Type.ROCK, opp_species.types)
                # Damage is 1/8th of max HP, scaled by effectiveness.
                damage = (opp_species.base_stats[Stat.MAX_HP] / 8.0) * effectiveness
                total_hazard_damage += damage
            # Normalize
            return total_hazard_damage / len(roster) if roster else 0.0


        # --- C. PROTECT MOVE ---
        elif move_to_score.protect:
            max_threat_damage = 0.0
            for opp_species in roster:
                if opp_species is attacker_build.species: continue
                opp_build = generic_builds_cache.get(opp_species)
                if not opp_build: continue

                opp_best_move = max(opp_build.moves, key=lambda m: m.base_power if m.category != Category.OTHER else -1)
                if opp_best_move.base_power == 0: continue

                damage = calculate_damage(self.sim_battle_policy.params, 1, opp_best_move,
                                         State((BattlingTeam(active=[BattlingPokemon(opp_build)], reserve=[]),
                                                BattlingTeam(active=[BattlingPokemon(attacker_build)], reserve=[]))),
                                         BattlingPokemon(opp_build),
                                         BattlingPokemon(attacker_build))
                if damage > max_threat_damage:
                    max_threat_damage = damage

            return (max_threat_damage / attacker_build.stats[Stat.MAX_HP]) * 100.0

        # --- D. WEATHER/TERRAIN MOVES ---
        elif move_to_score.weather_start != Weather.CLEAR or move_to_score.field_start != Terrain.NONE:
            net_damage_swing = 0.0

            # Helper to calculate damage change for a given effect
            def get_effect_swing(move_type_boost, multiplier, weather_nerf=None):
                swing = 0
                for species in roster:
                    build = generic_builds_cache.get(species)
                    if not build: continue

                    best_move = max([m for m in build.moves if m.pkm_type == move_type_boost and m.base_power > 0], default=None, key=lambda m: m.base_power)
                    if not best_move: continue

                    base_damage = calculate_damage(self.sim_battle_policy.params, 0, best_move,
                                                                 State((BattlingTeam([BattlingPokemon(build)], reserve=[]), BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]))),
                                                                 BattlingPokemon(build), BattlingPokemon(attacker_build))
                    swing += base_damage * (multiplier - 1)

                    if weather_nerf:
                        nerf_move = max([m for m in build.moves if m.pkm_type == weather_nerf and m.base_power > 0], default=None, key=lambda m: m.base_power)
                        if nerf_move:
                             nerf_base_dmg =calculate_damage(self.sim_battle_policy.params, 0, nerf_move,
                                                                 State((BattlingTeam([BattlingPokemon(build)], reserve=[]), BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]))),
                                                                 BattlingPokemon(build), BattlingPokemon(attacker_build))
                             swing += nerf_base_dmg * (1 - 0.5) # The damage that is mitigated
                return swing

            if move_to_score.weather_start == Weather.RAIN:
                net_damage_swing = get_effect_swing(Type.WATER, 1.5, weather_nerf=Type.FIRE)
            elif move_to_score.weather_start == Weather.SUN:
                net_damage_swing = get_effect_swing(Type.FIRE, 1.5, weather_nerf=Type.WATER)

            # SandStorm: loops through all species in the roster and create an optimal biuld
            # check for immunity then calculate the score based on HP and Special attack for ROCK, logic explained in Selection Policy
            elif move_to_score.weather_start == Weather.SAND:
                passive_damage_score = 0
                defensive_boost_score = 0
                for species in roster:
                    build = generic_builds_cache.get(species)
                    if not build: continue

                    if not any(t in build.species.types for t in [Type.ROCK, Type.GROUND, Type.STEEL]):
                        passive_damage_score += build.stats[Stat.MAX_HP] / 16.0

                    if Type.ROCK in build.species.types:
                        defensive_boost_score += build.stats[Stat.SPECIAL_DEFENSE] * 0.33

                net_damage_swing = passive_damage_score + defensive_boost_score

            # Snow: is calculated differently as it provides a defensive boost to Ice-types
            # calculation is a percentage of the Ice-type's attack stat added as a score
            elif move_to_score.weather_start == Weather.SNOW:
                defensive_boost_score = 0
                for species in roster:
                    build = generic_builds_cache.get(species)
                    if not build: continue

                    # Snow provides a Defense boost to Ice-types but no passive damage.
                    if Type.ICE in build.species.types:
                        defensive_boost_score += build.stats[Stat.DEFENSE] * 0.33

                net_damage_swing = defensive_boost_score

            elif move_to_score.field_start == Terrain.ELECTRIC_TERRAIN:
                net_damage_swing = get_effect_swing(Type.ELECTRIC, 1.3)
            elif move_to_score.field_start == Terrain.GRASSY_TERRAIN:
                net_damage_swing = get_effect_swing(Type.GRASS, 1.3)
            elif move_to_score.field_start == Terrain.PSYCHIC_TERRAIN:
                net_damage_swing = get_effect_swing(Type.PSYCHIC, 1.3)
            elif move_to_score.field_start == Terrain.MISTY_TERRAIN:
                # Misty Terrain halves the power of Dragon-type moves
                net_damage_swing = get_effect_swing(Type.DRAGON, 0.5)

            # Normalize the net damage swing by the number of roster members
            return net_damage_swing / len(roster) if roster else 0.0

        # --- E. STATUS MOVES ---
        elif move_to_score.status != Status.NONE:
            #is_wall_role = "Defensive" in self._get_archetype_name_from_build(attacker_build)
            #role_multiplier = 1.2 if is_wall_role else 0.8
            base_status_score = 0.0

            avg_roster_hp = sum(b.stats[Stat.MAX_HP] for b in generic_builds_cache.values()) / len(roster)

            if move_to_score.status == Status.BURN:
                passive_damage = avg_roster_hp / 16.0

                phys_threats = [b for s, b in generic_builds_cache.items() if s.base_stats[Stat.ATTACK] > s.base_stats[Stat.SPECIAL_ATTACK]]
                mitigated_damage = 0
                if phys_threats:
                    strongest_phys_threat = max(phys_threats, key=lambda b: b.stats[Stat.ATTACK])
                    best_phys_move = max([m for m in strongest_phys_threat.moves if m.category == Category.PHYSICAL], default=None, key=lambda m:m.base_power)
                    if best_phys_move:
                        # damage with burn is half
                         mitigated_damage = calculate_damage(self.sim_battle_policy.params, 0, best_phys_move,
                                                             State((BattlingTeam([BattlingPokemon(strongest_phys_threat)],reserve=[]),
                                                                    BattlingTeam([BattlingPokemon(attacker_build)],reserve=[]))),
                                                             BattlingPokemon(strongest_phys_threat), BattlingPokemon(attacker_build)) / 2.0

                base_status_score = passive_damage + mitigated_damage

            elif move_to_score.status == Status.TOXIC:
                base_status_score = (avg_roster_hp / 16.0) * (1+2+3+4) # Approx over 4 turns

            elif move_to_score.status == Status.SLEEP:
                # 1) Find the max damage potential among the entire roster
                # 2) Estimate denying 1.5 turns of that damage
                max_damage_potential = 0.0
                for spc, build in generic_builds_cache.items():
                    # Find the strongest damaging move
                    best_move = max(
                        (m for m in build.moves if m.base_power > 0),
                        default=None,
                        key=lambda m: m.base_power
                    )
                    if not best_move:
                        continue
                    dmg = calculate_damage(self.sim_battle_policy.params, 0, best_move,
                                 State((BattlingTeam(active=[BattlingPokemon(build)], reserve=[]),
                                        BattlingTeam(active=[BattlingPokemon(attacker_build)], reserve=[]))),
                                 BattlingPokemon(build), BattlingPokemon(attacker_build))
                    if dmg > max_damage_potential:
                        max_damage_potential = dmg
                # As in selection policy: (max_damage_potential * 1.5)/avg_hp * 100
                base_status_score = ((max_damage_potential * 1.5) / avg_roster_hp) * 100

            elif move_to_score.status == Status.PARALYZED:
                # 1) Identify the fastest build in the roster
                # 2) The value is the damage lost if that fastest Pokémon is paralyzed (25% chance to lose a turn)
                if not generic_builds_cache:
                    base_status_score = 0.0
                else:
                    fastest_build = max(generic_builds_cache.values(), key=lambda b: b.stats[Stat.SPEED])
                    best_move = max(
                        (m for m in fastest_build.moves if m.base_power > 0),
                        default=None,
                        key=lambda m: m.base_power
                    )
                    if best_move:
                        dmg = calculate_damage(self.sim_battle_policy.params, 0, best_move, State((BattlingTeam([BattlingPokemon(fastest_build)], reserve=[]),
                                                                                               BattlingTeam([BattlingPokemon(attacker_build)], reserve=[]))),
                                      BattlingPokemon(fastest_build), BattlingPokemon(attacker_build))
                        # 25% chance to skip this damage
                        base_status_score = ((dmg * 0.25) / avg_roster_hp) * 100
                    else:
                        base_status_score = 0.0

            return base_status_score


        # If the move is not a recognized utility type, return 0.
        return 0.0

    def _calculate_stat_boost_synergy(self, attacker_build: Pokemon, move_to_score: Move, roster: Roster, optimal_builds_cache: dict[PokemonSpecies, Pokemon]) -> float:
        """
        Calculates the synergy score for a stat-boosting move.

        This is achieved by quantifying the net damage increase the boost provides
        to the user's best potential damaging moves from its entire species movepool.
        This fixes a previous flaw where it only checked against placeholder moves.
        """
        # This function is only for self-targeting stat boosts.
        if not (move_to_score.boosts and move_to_score.self_boosts):
            return 0.0

        # Determine if the boost is physical or special.
        is_phys_boost = move_to_score.boosts[Stat.ATTACK - 1] > 0
        relevant_category = Category.PHYSICAL if is_phys_boost else Category.SPECIAL

        # Search the entire species movepool for the best complementary attacks.
        potential_moves = attacker_build.species.moves
        relevant_damaging_moves = [m for m in potential_moves if m.category == relevant_category and m.base_power > 0]

        # Find the top 2 best damaging moves from the pool based on their calculated damage score.
        top_relevant_moves = sorted(
            relevant_damaging_moves,
            key=lambda m: self._calculate_damage_score(attacker_build, m, roster, optimal_builds_cache),
            reverse=True
        )[:2]

        if not top_relevant_moves:
            return 0.0

        total_dmg_increase = 0.0
        for move in top_relevant_moves:
            dmg_before = self._calculate_damage_score(attacker_build, move, roster, optimal_builds_cache)
            boosted_build = self._apply_temp_boosts(attacker_build, move_to_score.boosts)
            dmg_after = self._calculate_damage_score(boosted_build, move, roster, optimal_builds_cache)
            total_dmg_increase += (dmg_after - dmg_before)

        return total_dmg_increase / len(top_relevant_moves) if top_relevant_moves else 0.0

    def _calculate_speed_control_synergy(self, attacker_build: Pokemon, move_to_score: Move, roster: Roster, optimal_builds_cache: dict[PokemonSpecies, Pokemon]) -> float:
        """
        Calculates the synergy score for speed-control moves.

        This is achieved by simulating 1v1 matchups against the roster and
        quantifying the average "Net Damage Swing" gained from flipping the
        turn order. This fixes a flaw where only KO reversals were valued,
        the error was spotted by gathering ddebug data to a CSV file.
        """
        if not (move_to_score.toggle_tailwind or move_to_score.toggle_trickroom):
            return 0.0

        total_swing_score = 0.0
        relevant_matchups = 0

        # Setup the attacker's side for simulation
        my_battle_pkm = BattlingPokemon(attacker_build)
        my_team_shell = BattlingTeam(active=[my_battle_pkm], reserve=[])
        opp_team_shell = BattlingTeam(active=[None], reserve=[])
        state_shell = State((my_team_shell, opp_team_shell))

        for opp_species in roster:
            if opp_species is attacker_build.species: continue

            opp_build = None
            if optimal_builds_cache:
                opp_build = optimal_builds_cache.get(opp_species)
            if not opp_build:
                opp_build = self._create_generic_build_for_species(opp_species)

            opp_battle_pkm = BattlingPokemon(opp_build)
            state_shell.sides[1].team.active[0] = opp_battle_pkm

            my_speed = attacker_build.stats[Stat.SPEED]
            opp_speed = opp_build.stats[Stat.SPEED]

            # Tailwind helps if we are slower. Trick Room helps if we are slower.
            is_speed_control_relevant = my_speed < opp_speed

            if not is_speed_control_relevant:
                continue

            relevant_matchups += 1

            # Find best damaging moves for both combatants
            my_best_move = max((m for m in attacker_build.moves if m.base_power > 0), key=lambda m:m.base_power, default=None)
            opp_best_move = max((m for m in opp_build.moves if m.base_power > 0), key=lambda m:m.base_power, default=None)

            if not my_best_move or not opp_best_move: continue

            my_dmg = calculate_damage(self.sim_battle_policy.params, 0, my_best_move, state_shell, my_battle_pkm, opp_battle_pkm)
            opp_dmg = calculate_damage(self.sim_battle_policy.params, 1, opp_best_move, state_shell, opp_battle_pkm, my_battle_pkm)

            # --- Calculate Net Potential Before and After Speed Control ---

            # Before: Opponent is faster and attacks first.
            my_hp_after_opp_hit = attacker_build.stats[Stat.MAX_HP] - opp_dmg
            my_retaliation_dmg = 0 if my_hp_after_opp_hit <= 0 else my_dmg
            net_potential_before = my_retaliation_dmg - opp_dmg

            # After: With speed control, our Pokemon is now faster and attacks first.
            opp_hp_after_my_hit = opp_build.stats[Stat.MAX_HP] - my_dmg
            opp_retaliation_dmg = 0 if opp_hp_after_my_hit <= 0 else opp_dmg
            net_potential_after = my_dmg - opp_retaliation_dmg

            swing = net_potential_after - net_potential_before
            total_swing_score += swing

        # Return the AVERAGE swing across all relevant matchups.
        return total_swing_score / relevant_matchups if relevant_matchups > 0 else 0.0

    def _apply_temp_boosts(self, pokemonbuild: Pokemon, boost_stages: tuple) -> Pokemon:
        """
        Creates a new temp Pokemon opbjects with its stats recalculated to reflect the temperoray boosts
        
        :param pokemon_build: The original Pokemon object.
        :param boost_stages: A tuple of stat changes (e.g., from a move's .boosts).
        :return: A new Pokemon object with recalculated, boosted stats.
        """

        # Create a new Pokemon Object to avoid modifying the original
        temp_build = Pokemon(
        species=pokemonbuild.species,
        move_indexes=[pokemonbuild.species.moves.index(m) for m in pokemonbuild.moves],
        level=pokemonbuild.level,
        evs=pokemonbuild.evs,
        ivs=pokemonbuild.ivs,
        nature=pokemonbuild.nature
    )
        boost_multi = self.sim_battle_policy.params.BOOST_MULTIPLIER_LOOKUP

        # create a mutable list from the stats tuple
        new_stats = list(temp_build.stats)

        # Apply boosts
        for i, stage_change in enumerate(boost_stages[:5]): # Only consider the 5 combat stats
            if stage_change != 0:
                stat_to_boost = i + 1  # e.g., boost index 0 (Attack) maps to Stat enum 1 (ATTACK)

                current_boost_stage = 0
                new_boost_stage = min(6, max(-6, current_boost_stage + stage_change))

                if new_boost_stage in boost_multi:
                    multiplier = boost_multi.get(stage_change, 1.0)
                    new_stats[stat_to_boost] = int(new_stats[stat_to_boost] * multiplier)

        temp_build.stats = tuple(new_stats) # Convert back to tuple
        return temp_build

    def _get_role_aware_moveset(self, attacker_build: Pokemon, archetype_name: str, roster: Roster) -> tuple[list[Move], dict[Move, dict[str, float]]]:
        """
        Calculates disaggregated scores for all of a species' moves
        and selects the optimal set of 4.

        The internal data structure isolates each score component to prevent
        scale contamination, a key refinement based on analysis of the initial
        implementation's fitness calculation.

        Returns:
            A tuple containing:
            - A list of the top 4 Move objects.
            - A dictionary mapping every move to its component scores.
        """
        if not attacker_build.species.moves:
            return [], {}

        move_scores: dict[Move, dict[str, float]] = {
            move: {"damage": 0.0, "utility": 0.0, "stat_syn": 0.0, "speed_syn": 0.0}
            for move in attacker_build.species.moves
        }

        # This function needs the full cache to correctly score synergy moves,
        # but the cache itself depends on this function. To break the cycle,
        # we pass None and the helpers use generic builds.
        optimal_builds_cache = None

        for move in attacker_build.species.moves:
            # Calculate each component score independently
            damage_score = self._calculate_damage_score(attacker_build, move, roster, optimal_builds_cache)
            utility_score = self._calculate_utility_score(attacker_build, move, roster)
            stat_synergy_score = self._calculate_stat_boost_synergy(attacker_build, move, roster, optimal_builds_cache)
            speed_synergy_score = self._calculate_speed_control_synergy(attacker_build, move, roster, optimal_builds_cache)

            # Populate the disaggregated dictionary
            move_scores[move]["damage"] = damage_score
            move_scores[move]["utility"] = utility_score
            move_scores[move]["stat_syn"] = stat_synergy_score
            move_scores[move]["speed_syn"] = speed_synergy_score

        # --- Final Score Combination for Sorting ---
        def get_final_score(move):
            scores = move_scores.get(move, {})
            # A simple sum is sufficient for sorting to find the best 4 moves for THIS build.
            # The true weighted fitness that compares different builds happens later.
            return sum(scores.values())

        sorted_moves = sorted(move_scores.keys(), key=get_final_score, reverse=True)
        top_4_moves = sorted_moves[:4]

        return top_4_moves, move_scores

    def _create_generic_build_for_species(self, species: PokemonSpecies) -> Pokemon:
        """
        Creates a single, generic 'best attacker' build for a species.
        
        This is used as a temporary opponent build during move evaluation when
        the final optimal build cache is not yet available. It prevents
        circular dependencies.
        """
        # A simple, consistent, generic build
        DEFAULT_EVS: Stats = (85, 85, 85, 85, 85, 85) # Mixed attacker focus
        DEFAULT_IVS: Stats = (31, 31, 31, 31, 31, 31)
        DEFAULT_NATURE = Nature.SERIOUS # Neutral

        # Take the first 4 moves available as a generic moveset
        num_moves = min(4, len(species.moves))
        move_indexes = list(range(num_moves))

        return Pokemon(
            species=species,
            move_indexes=move_indexes,
            level=50,
            evs=DEFAULT_EVS,
            ivs=DEFAULT_IVS,
            nature=DEFAULT_NATURE
        )

    def _get_role_aware_moveset(self, attacker_build: Pokemon, archetype_name: str, roster: Roster) -> tuple[list[Move], dict[Move, float]]:
        """Selects the best 4 moves for a species given a specific role

        Args:
            species (PokemonSpecies): The PokemonSpecies to select moves for.
            archetype_name (str): The competitive archetype name (e.g., "Fast Physical Sweeper").
            roster (Roster): The full roster of available PokemonSpeices

        Returns:
            List[Move]: A list of 4 Move objects
        """

        # FailSafe Check
        if not attacker_build.species.moves:
            #logging.warning(f"Species {attacker_build.species.name} has no moves. Returning empty list.")
            return [], {}

        move_scores: dict[Move, dict[str, float]] = {
            move: {"damage": 0.0, "utility": 0.0, "synergy": 0.0}
            for move in attacker_build.species.moves
        }

        # --- Pass 1: Calculate Preliminary Damage and Utility Scores ---
        for move in attacker_build.species.moves:
            # pass `None` for the cache to use generic opponent builds.
            damage_score = self._calculate_damage_score(attacker_build, move, roster, None)
            utility_score = self._calculate_utility_score(attacker_build, move, roster)
            move_scores[move]["damage"] = damage_score
            move_scores[move]["utility"] = utility_score

        # Create a temporary dict of the combined preliminary scores for synergy calculation
        prelim_combined_scores = {m: v["damage"] + v["utility"] for m, v in move_scores.items()}

        optimal_builds_cache = None

        # --- Pass 2: Calculate Role-Synergy Scores ---
        for move in attacker_build.species.moves:
            stat_synergy_score = self._calculate_stat_boost_synergy(attacker_build, move, roster, optimal_builds_cache)
            speed_synergy_score = self._calculate_speed_control_synergy(attacker_build, move, roster, optimal_builds_cache)

            # Populate the disaggregated dictionary
            move_scores[move]["stat_syn"] = stat_synergy_score
            move_scores[move]["speed_syn"] = speed_synergy_score

        # --- Final Score Combination and Selection ---
        def get_final_score(move):
            scores = move_scores.get(move, {})
            return scores.get("damage", 0) + scores.get("utility", 0) + scores.get("synergy", 0)

        sorted_moves = sorted(move_scores.keys(), key=get_final_score, reverse=True)
        top_4_moves = sorted_moves[:4]

        return top_4_moves, move_scores


    # TODO: implement pruning for the Heursitic method to prune 35%
    # TODO: implement the EA algorithm
    # NOTE: Remember to adjust the Random hill climbing to a simmulated annealing or threshold acceptance
    # TODO: implement the simulation funnel

    def decision(self, roster:Roster, meta: Meta | None, max_team_size: int, max_pkm_moves: int, n_active: int) -> TeamBuildCommand:



        """Determines The best team from a roster using a multi-stage evolutionary funnel approach.

        Args:
            roster (Roster): The full roster of PokemonSpecies available for team building.
            meta (Meta | None): The meta information, which can be used to inform team building decisions.
            max_team_size (int): The maximum number of Pokemon allowed in the final team.
            max_pkm_moves (int): The maximum number of moves each Pokemon can have.
            n_active (int): The number of active Pokemon in a battle.

        Returns:
            TeamBuildCommand: _description_
        """

        # --- NEW: Step 1 - Roster-Wide Pre-calculation via Stochastic Sampling ---
        # To optimize performance, the global maximums for normalization are estimated
        # from a random sample of all possible builds, rather than calculating
        # scores for every single one. without the sampling the complexity was doubled

        # First, generate all potential builds without scoring them.
        all_potential_builds = []
        for temp_species in roster:
            builds_with_names = self._create_archetype_builds(temp_species, temp_species.moves[:4])
            for archetype_name, temp_build in builds_with_names:
                all_potential_builds.append((archetype_name, temp_build))

        # Take a random sample from the full pool of potential builds.
        sample_size = min(self.normalization_sample_size, len(all_potential_builds))
        build_sample = random.sample(all_potential_builds, sample_size)

        # Now, perform the expensive score calculations ONLY on the sample.
        sample_evals = []
        for archetype_name, temp_build in build_sample:
            optimal_moves, all_move_scores = self._get_role_aware_moveset(
                temp_build, archetype_name, roster
            )

            total_damage = sum(all_move_scores[m]["damage"] for m in optimal_moves)
            total_utility = sum(all_move_scores[m]["utility"] for m in optimal_moves)
            total_stat_syn = sum(all_move_scores[m]["stat_syn"] for m in optimal_moves)
            total_speed_syn = sum(all_move_scores[m]["speed_syn"] for m in optimal_moves)

            sample_evals.append({
                "stat_score": self._calculate_stat_compatibility(temp_build.species, temp_build.evs),
                "damage_score": total_damage,
                "utility_score": total_utility,
                "stat_syn_score": total_stat_syn,
                "speed_syn_score": total_speed_syn,
                "speed_stat_score": temp_build.stats[Stat.SPEED]
            })

        # --- NEW: Step 2 - Find Global Maximums from the Sample ---
        global_max_scores = {
            "max_stat": max(ev["stat_score"] for ev in sample_evals) if sample_evals else 1.0,
            "max_dmg": max(ev["damage_score"] for ev in sample_evals) if sample_evals else 1.0,
            "max_util": max(ev["utility_score"] for ev in sample_evals) if sample_evals else 1.0,
            "max_stat_syn": max(ev["stat_syn_score"] for ev in sample_evals) if sample_evals else 1.0,
            "max_speed_syn": max(ev["speed_syn_score"] for ev in sample_evals) if sample_evals else 1.0,
            "max_speed_stat": max(ev["speed_stat_score"] for ev in sample_evals) if sample_evals else 1.0
        }

        num_to_keep = int(len(roster) * (1 - self.pruning_percentage))

        # Reset debug data for this run if DEBUG is on.
        if DEBUG:
            self.debug_data = []

        # --- Pass 1: Generate the Optimal Archetype for every species ---
        optimal_builds_cache: dict[PokemonSpecies, Pokemon] = {}
        for i, species in enumerate(roster):
            # Pass the global_max_scores dictionary to the archetype selection function.
            optimal_builds_cache[species] = self._get_optimal_archetype(species, roster, global_max_scores)

        # --- Pass 2: Calculate Roster Viability Index for Each Species ---
        viability_scores = dict.fromkeys(roster, 0.0)

        species_list = list(roster)
        for i in range(len(species_list)):
            for j in range(i + 1, len(species_list)):
                species_A = species_list[i]
                species_B = species_list[j]

                build_A = optimal_builds_cache.get(species_A)
                build_B = optimal_builds_cache.get(species_B)

                if not build_A or not build_B: continue

                score_A_vs_B = self._calculate_1v1_net_score(build_A, build_B)

                viability_scores[species_A] += score_A_vs_B
                viability_scores[species_B] -= score_A_vs_B

        num_opponents = len(roster) - 1
        if num_opponents > 0:
            for species in viability_scores:
                viability_scores[species] /= num_opponents

        # --- Pass 3: Sort, Select, and Build Final Command ---
        sorted_species = sorted(roster, key=lambda s: viability_scores.get(s, -float('inf')), reverse=True)
        final_team_species = sorted_species[:max_team_size]

        final_command: TeamBuildCommand = []
        for species_to_include in final_team_species:
            roster_index = roster.index(species_to_include)
            optimal_build = optimal_builds_cache[species_to_include]
            move_indices = [optimal_build.species.moves.index(m) for m in optimal_build.moves]
            command_tuple = (roster_index, optimal_build.evs, optimal_build.ivs, optimal_build.nature, move_indices)
            final_command.append(command_tuple)

        # --- New Diagnostic Logging: Save data to CSV ---
        if DEBUG and self.debug_data:
            try:
                # Append the collected data to the file
                pd.DataFrame(self.debug_data).to_csv("hesf_debug_log.csv", mode='a', header=False, index=False)
                print("Diagnostic data for this run has been saved to hesf_debug_log.csv")
            except OSError as e:
                print(f"Error saving debug data to CSV: {e}")

        return final_command
