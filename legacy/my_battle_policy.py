
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import *
from vgc2.battle_engine.constants import *
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.modifiers import Category, Stat, Status, Type
from vgc2.battle_engine.threshold_calculator import protect_modifier
from vgc2.battle_engine.view import BattlingPokemonView, StateView, TeamView

# --- Constants ---
MAX_SCORE = 1000.0 # the maximum score a move can achieve
PRINT_TYPE_MATCHUPS_DEBUG = False  # Set to True to enable type matchup printing for debugging, SET TO FALSE WHEN BENCHMARKING

# --- Weights to find the optimal vote of each function of decision ---
'''W_OFF_DEF_SUPPORT_BONUS = 0.30 
W_BASE_SCORE_A = 0.15 
W_ENV_SYNERGY_BONUS = 0.10 
W_FOCUS_FIRE_BONUS = 0.10 
W_TARGET_PRIORITY_BONUS = 0.10 
W_BASE_SCORE_B = 0.10
W_SURVIVAL_IMPACT = 0.10
W_SETUP_SYNERGY_BONUS = 0.05'''

class MyBattlePolicy(BattlePolicy):

    def __init__(self, detailed_logging: bool = False, custom_weights: dict[str, float] | None = None):
        super().__init__()
        self.battle_params = BattleRuleParam()
        self.detailed_logging = detailed_logging

        # The champion weights
        self.W_OFF_DEF_SUPPORT_BONUS = 0.0200
        self.W_BASE_SCORE_A = 0.0500
        self.W_ENV_SYNERGY_BONUS = 0.0200
        self.W_FOCUS_FIRE_BONUS = 0.2700
        self.W_TARGET_PRIORITY_BONUS = 0.1800
        self.W_BASE_SCORE_B = 0.1500
        self.W_SURVIVAL_IMPACT = 0.1300
        self.W_SETUP_SYNERGY_BONUS = 0.1800

         # If custom_weights are provided, override the defaults
        if custom_weights:
            for key, value in custom_weights.items():
                if hasattr(self, key): # Check if the policy actually has this weight attribute
                    setattr(self, key, value)


    def print_pokemon_type_matchups(self, pokemon_view: BattlingPokemonView):
        """Prints the type matchups for a given Pokémon."""
        if not pokemon_view or not hasattr(pokemon_view, 'types') or not hasattr(pokemon_view, 'constants'):
            print("Debug: Cannot print matchups for invalid pokemon_view object (name or types missing).")
            return

        pokemon_name = "Unknown Pokemon"
        if hasattr(pokemon_view.constants, 'species') and pokemon_view.constants.species is not None:
            pokemon_name = pokemon_view.constants.species.name
        elif hasattr(pokemon_view.constants, 'name'): # Fallback
            pokemon_name = pokemon_view.constants.name

        pokemon_types_str = ", ".join([t.name for t in pokemon_view.types])

        print(f"\n--- Type Matchups for {pokemon_name} (Types: {pokemon_types_str}) ---")
        header = f"{'Attacking Type':<15} | {'Multiplier vs ' + pokemon_name:<30}"
        print(header)
        print("-" * (len(header) + 2))

        chart_rows = len(self.battle_params.DAMAGE_MULTIPLICATION_ARRAY)
        if chart_rows == 0:
            print("Debug: Damage multiplication array is empty.")
            return
        chart_cols = len(self.battle_params.DAMAGE_MULTIPLICATION_ARRAY[0])

        for attacking_type in Type: # Iterate through all defined Type enum members
            if attacking_type.value >= chart_rows :
                # print(f"Debug: Skipping attacking_type {attacking_type.name} (value {attacking_type.value}) - out of chart rows.")
                continue

            effectiveness_multiplier = 1.0
            valid_calculation = True
            for defending_type in pokemon_view.types:
                if attacking_type.value >= chart_rows or \
                defending_type.value >= chart_cols:
                    # print(f"Debug: Type value out of bounds for chart. Attacking: {attacking_type.name}({attacking_type.value}), Defending: {defending_type.name}({defending_type.value})")
                    effectiveness_multiplier = 1.0
                    valid_calculation = False
                    break

                effectiveness_multiplier *= self.battle_params.DAMAGE_MULTIPLICATION_ARRAY[attacking_type.value][defending_type.value]

            desc = ""
            if not valid_calculation and effectiveness_multiplier == 1.0:
                desc = " (Error in calc)"
            elif effectiveness_multiplier == 0: desc = " (Immune)"
            elif 0 < effectiveness_multiplier < 1: desc = " (Resisted)"
            elif effectiveness_multiplier > 1: desc = " (Weak To)"
            else: desc = " (Neutral)"

            print(f"{attacking_type.name:<15} | {effectiveness_multiplier:<4.2f}{desc:<20}")
        print("--- End Matchups ---\n")

    def _get_type_effectiveness(self, move_type: Type, target_types: list[Type]) -> float:
        """Helper to get type effectiveness with boundary checks."""
        effectiveness = 1.0
        chart_rows = len(self.battle_params.DAMAGE_MULTIPLICATION_ARRAY)
        if chart_rows == 0: return 1.0 # Safety for empty chart
        chart_cols = len(self.battle_params.DAMAGE_MULTIPLICATION_ARRAY[0])

        if move_type.value >= chart_rows:
            return 1.0

        for target_type in target_types:
            if target_type.value >= chart_cols:
                continue
            effectiveness *= self.battle_params.DAMAGE_MULTIPLICATION_ARRAY[move_type.value][target_type.value]
        return effectiveness

    def _score_single_offensive_move(self,
                                     attacker: BattlingPokemonView,
                                     move: BattlingMove,
                                     target_pokemon: BattlingPokemonView,
                                     state: StateView) -> tuple[float, bool]: # Returns (score, is_ko)
        """Scores a single offensive move. Does not include complex Protect logic here."""

        action_score = 0.0
        is_ko_potential = False

        if move.pp <= 0 or move.disabled:
            return -float('inf'), False

        base_power = float(move.constants.base_power)
        is_status_category_move = move.constants.category == Category.OTHER

        if base_power == 0 and not is_status_category_move:
             base_power = 10

        # --- 1. Damage Component (for non-status moves) ---
        if not is_status_category_move and base_power > 0:
            actual_damage_dealt = calculate_damage(
                params=self.battle_params,
                attacking_side=0,
                move=move.constants,
                state=state,
                attacker=attacker,
                defender=target_pokemon
            )

            # a) Score based on percentage of target's CURRENT HP.
            # This naturally rewards finishing off weakened targets.
            if target_pokemon.hp > 0:
                damage_for_scoring = min(actual_damage_dealt, target_pokemon.hp)
                percent_hp_dealt = damage_for_scoring / target_pokemon.hp
                action_score += percent_hp_dealt * 100

            # b) Check for KO and add a principled bonus.
            is_ko_potential = (actual_damage_dealt >= target_pokemon.hp)

            if is_ko_potential:
                # The value of a KO is the future damage prevent that Pokemon from dealing.
                #estimate this by calculating the threat it poses to our entire active team.
                ko_bonus = 0
                my_active_team = [p for p in state.sides[0].team.active if p and p.hp > 0]
                for my_pkm in my_active_team:
                    # The threat this target poses to one of my pokemon.
                    threat_to_one_ally, _ = self.estimate_incoming_threat(my_pkm, [target_pokemon], state)
                    ko_bonus += threat_to_one_ally

                action_score += ko_bonus


        # Accuracy (applied after damage/KO, before utility for that move)
        if move.constants.accuracy is not None:
            action_score *= move.constants.accuracy
        else:
            action_score *= 1.0

        # --- 2. Utility Component (Status Effects, Stat Changes) ---
        utility_score = 0.0

        # 2.a Status Infliction (Principled Calculation)
        status_utility = self._calculate_status_value(attacker, target_pokemon, move.constants, state)
        utility_score += status_utility

        # 2.b & 2.c Stat Changes (Principled Calculation)
        stat_boost_utility = self._calculate_stat_boost_value(attacker, move.constants, state)
        utility_score += stat_boost_utility

        # 2.d Environmental Setup Utility (Weather, Terrain, Trick Room)
        is_primary_utility_move = move.constants.category == Category.OTHER or \
                                move.constants.weather_start != Weather.CLEAR or \
                                move.constants.field_start != Terrain.NONE or \
                                move.constants.toggle_trickroom or \
                                move.constants.hazard != Hazard.NONE

        if is_primary_utility_move:
            my_active_list_for_setup = [p for p in state.sides[0].team.active if p and p.hp > 0]
            my_reserve_list_for_setup = [p for p in state.sides[0].team.reserve if p and p.hp > 0]
            opp_actives_list_for_setup = state.sides[1].team.active if state.sides[1].team.active else []

            field_setup_utility = self._score_field_setup_move(
                attacker,
                move.constants,
                my_active_list_for_setup,
                my_reserve_list_for_setup,
                opp_actives_list_for_setup,
                state
            )
            utility_score += field_setup_utility
        action_score += utility_score

        # 2.d Screen Setup Utility
        screen_utility = self._calculate_screen_value(
            move.constants,
            [p for p in state.sides[0].team.active if p and p.hp > 0],
            [p for p in state.sides[1].team.active if p and p.hp > 0],
            state
        )
        utility_score += screen_utility

        # 2.e Hazard Removal Utility
        # Define known hazard removal moves
        hazard_removal_move_names = {"rapid spin", "defog", "mortal spin", "tidy up"}
        if move.constants.name.lower() in hazard_removal_move_names:
            hazard_removal_utility = self._calculate_hazard_removal_value(state)
            utility_score += hazard_removal_utility

        # 2.f Healing Utility
        healing_utility = self._calculate_healing_value(attacker, move.constants, state)
        utility_score += healing_utility

        # PP Conservation (applied last, only if not a KO and not already a high utility move)
        if not is_ko_potential and move.constants.max_pp > 0 and (move.pp / move.constants.max_pp) < 0.3:
            if action_score < 400 : # Threshold to avoid penalizing already good non-KO moves
                action_score *= 0.8

        return max(0, action_score), is_ko_potential

    def _score_protect_move(self,
                            protector: BattlingPokemonView,
                            move: BattlingMove,
                            state: StateView) -> float:
        if move.pp <= 0 or move.disabled:
            return -float('inf')

        # --- 1. Base Value: Damage Mitigation ---
        # The primary value of Protect is the incoming damage it blocks.
        estimated_damage_blocked, _ = self.estimate_incoming_threat(protector, state.sides[1].team.active, state)

        protect_score = estimated_damage_blocked

        # --- 2. Secondary Value: Net Stalling Damage ---
        # Calculate the net effect of passive damage that occurs this turn while protected.
        net_passive_damage_value = 0

        # Add value for passive damage dealt to opponents
        for opp_pkm in state.sides[1].team.active:
            if not opp_pkm or opp_pkm.hp <= 0: continue
            if opp_pkm.status in [Status.POISON, Status.TOXIC]:
                net_passive_damage_value += calculate_poison_damage(self.battle_params, opp_pkm)
            elif opp_pkm.status == Status.BURN:
                net_passive_damage_value += calculate_burn_damage(self.battle_params, opp_pkm)

            if state.weather == Weather.SAND:
                net_passive_damage_value += calculate_sand_damage(self.battle_params, opp_pkm)

        # Subtract cost of passive damage taken by our side (excluding the protector)
        for ally_pkm in state.sides[0].team.active:
            if not ally_pkm or ally_pkm.hp <= 0 or ally_pkm is protector: continue
            if ally_pkm.status in [Status.POISON, Status.TOXIC]:
                net_passive_damage_value -= calculate_poison_damage(self.battle_params, ally_pkm)
            elif ally_pkm.status == Status.BURN:
                net_passive_damage_value -= calculate_burn_damage(self.battle_params, ally_pkm)

            if state.weather == Weather.SAND:
                net_passive_damage_value -= calculate_sand_damage(self.battle_params, ally_pkm)

        protect_score += net_passive_damage_value

        # --- 3. Reliability Scaling ---
        # The final score is scaled by the move's accuracy
        protect_accuracy = move.constants.accuracy if move.constants.accuracy is not None else 1.0
        # The framework's protect_modifier handles diminishing returns on consecutive uses.
        # NOTE: removed an old method where I tried to do it myself, it was not as effective.
        protect_reliability = protect_accuracy * protect_modifier(self.battle_params, move.constants, protector)

        return protect_score * protect_reliability

    def _score_single_switch_action(self,
                                    current_pkm: BattlingPokemonView,
                                    reserve_pkm_to_switch_in: BattlingPokemonView,
                                    opponent_actives: list[BattlingPokemonView],
                                    state: StateView) -> float:

        # --- Start of _score_single_switch_action ---
        if reserve_pkm_to_switch_in.hp <= 0:
            return -float('inf')

        switch_score = 50.0
        num_opponents = len(opponent_actives)
        if num_opponents == 0: return switch_score

        total_resistance_score = 0
        is_strong_defensive_pivot = False

        for opp_pkm in opponent_actives:
            if not opp_pkm or opp_pkm.hp <= 0: continue

            resistance_to_this_opp_numeric = 0
            best_case_resistance_multiplier_vs_opp_stabs = 1.0

            for opp_stab_type in opp_pkm.types:
                current_stab_effectiveness_on_incoming = self._get_type_effectiveness(opp_stab_type, reserve_pkm_to_switch_in.types)

                best_case_resistance_multiplier_vs_opp_stabs = min(best_case_resistance_multiplier_vs_opp_stabs, current_stab_effectiveness_on_incoming)

                if current_stab_effectiveness_on_incoming == 0: resistance_to_this_opp_numeric += (225 * 4)
                elif current_stab_effectiveness_on_incoming <= 0.25: resistance_to_this_opp_numeric += (170 * 4)
                elif current_stab_effectiveness_on_incoming <= 0.5: resistance_to_this_opp_numeric += (80  * 4)
                elif current_stab_effectiveness_on_incoming >= 2.0: resistance_to_this_opp_numeric -= (75 * 4)
                elif current_stab_effectiveness_on_incoming >= 4.0: resistance_to_this_opp_numeric -= (125 * 4)

            total_resistance_score += (resistance_to_this_opp_numeric / num_opponents)
            if best_case_resistance_multiplier_vs_opp_stabs <= 0.25:
                is_strong_defensive_pivot = True

        switch_score += total_resistance_score

        total_offensive_score_by_incoming = 0
        is_strong_offensive_threat = False

        if reserve_pkm_to_switch_in.constants and reserve_pkm_to_switch_in.constants.moves:
            for opp_pkm in opponent_actives:
                if not opp_pkm or opp_pkm.hp <= 0: continue

                max_effectiveness_against_this_opp = 0.0
                offensive_score_contribution_this_opp = 0

                for incoming_pkm_move in reserve_pkm_to_switch_in.constants.moves:
                    if incoming_pkm_move.base_power == 0 and incoming_pkm_move.category != Category.OTHER:
                        continue

                    current_move_effectiveness_on_opp = self._get_type_effectiveness(incoming_pkm_move.pkm_type, opp_pkm.types)
                    max_effectiveness_against_this_opp = max(max_effectiveness_against_this_opp, current_move_effectiveness_on_opp)

                    if current_move_effectiveness_on_opp >= 4.0: offensive_score_contribution_this_opp = max(offensive_score_contribution_this_opp, 125)
                    elif current_move_effectiveness_on_opp >= 2.0: offensive_score_contribution_this_opp = max(offensive_score_contribution_this_opp, 75)

                total_offensive_score_by_incoming += (offensive_score_contribution_this_opp / num_opponents)
                if max_effectiveness_against_this_opp >= 4.0:
                    is_strong_offensive_threat = True

        switch_score += (total_offensive_score_by_incoming * 1.7)

        if is_strong_defensive_pivot and is_strong_offensive_threat:
            switch_score += 150

        if total_resistance_score > 30 and total_offensive_score_by_incoming > 30:
            switch_score *= 1.15 # from logic.md (was 1.2)

        # Principled Hazard Damage Calculation
        if state.sides[0].conditions.stealth_rock:
            # The penalty is the actual damage the Pokemon would take.
            sr_damage = calculate_stealth_rock_damage(self.battle_params, reserve_pkm_to_switch_in)
            switch_score -= sr_damage

        if state.sides[0].conditions.poison_spikes:
            is_immune_to_tspikes = False
            if Type.POISON in reserve_pkm_to_switch_in.types or \
               Type.STEEL in reserve_pkm_to_switch_in.types or \
               Type.FLYING in reserve_pkm_to_switch_in.types:
                is_immune_to_tspikes = True

            # If the pokemon is not immune and not already statused, penalize the switch
            # by the amount of damage it would take on the first turn of poison.
            if not is_immune_to_tspikes and reserve_pkm_to_switch_in.status == Status.NONE:
                poison_damage = calculate_poison_damage(self.battle_params, reserve_pkm_to_switch_in)
                switch_score -= poison_damage

        return switch_score
        # --- End of _score_single_switch_action logic ---


    def estimate_incoming_threat(self,
                                 my_pokemon_target_view: BattlingPokemonView, # Your Pokemon being targeted
                                 opponent_actives: list[BattlingPokemonView], # List of opponent's active Pokemon
                                 state: StateView) -> tuple[float, bool]: # Returns (max_likely_total_damage, is_likely_ko_by_combined_threat)

        # This function will now estimate the maximum likely damage MyPkmTarget might take THIS TURN
        # from the opponent_actives.  consider each opponent individually and then sum
        # if multiple opponents are likely to target my_pokemon_target_view.


        if not my_pokemon_target_view or my_pokemon_target_view.hp <= 0:
            return 0.0, False

        max_total_incoming_damage_calculated = 0.0
        any_single_opponent_can_ko = False

        for opp_pkm_attacker_view in opponent_actives:
            if not opp_pkm_attacker_view or opp_pkm_attacker_view.hp <= 0:
                continue

            highest_damage_from_this_opponent = 0.0

            # --- 1. Evaluate Revealed Moves of this Opponent ---
            if opp_pkm_attacker_view.battling_moves:
                for opp_move_in_battle in opp_pkm_attacker_view.battling_moves:
                    if opp_move_in_battle.constants.category == Category.OTHER or \
                       opp_move_in_battle.constants.base_power == 0 or \
                       opp_move_in_battle.pp <= 0 or opp_move_in_battle.disabled:
                        continue # Skip non-damaging, no power, no PP, or disabled

                    calculated_damage_val = calculate_damage(
                        params=self.battle_params,
                        attacking_side=1,
                        move=opp_move_in_battle.constants,
                        state=state,
                        attacker=opp_pkm_attacker_view,
                        defender=my_pokemon_target_view
                    )
                    if calculated_damage_val > highest_damage_from_this_opponent:
                        highest_damage_from_this_opponent = calculated_damage_val

            # --- 2. Evaluate Potential Moves (If Necessary/To Augment) ---
            # Condition: If no revealed moves were significantly threatening,
            # or to ensure we haven't missed a stronger potential option.

            if opp_pkm_attacker_view.constants and \
               hasattr(opp_pkm_attacker_view.constants, 'species') and \
               opp_pkm_attacker_view.constants.species and \
               hasattr(opp_pkm_attacker_view.constants.species, 'moves') and \
               opp_pkm_attacker_view.constants.species.moves:

                potential_moves_to_evaluate = []
                # Heuristic to select a few "most likely dangerous" potential moves
                # Prioritize high base power STAB and SE moves against my_pokemon_target_view

                # Sort potential moves by a proxy score: BP * STAB_bonus * SE_bonus
                candidate_potential_moves = []
                for pot_move_const in opp_pkm_attacker_view.constants.species.moves:
                    if pot_move_const.category == Category.OTHER or pot_move_const.base_power == 0:
                        continue

                    bp = float(pot_move_const.base_power)
                    stab_bonus = 1.5 if pot_move_const.pkm_type in opp_pkm_attacker_view.types else 1.0
                    eff_bonus = self._get_type_effectiveness(pot_move_const.pkm_type, my_pokemon_target_view.types)
                    # Give higher weight to SE, then STAB, then BP
                    proxy_score = bp * (eff_bonus * 2) * stab_bonus * pot_move_const.accuracy  # eff_bonus squared to heavily favor SE
                    candidate_potential_moves.append({'move': pot_move_const, 'score': proxy_score}) # 56, 57, without 56, with x2

                # Sort by proxy_score and take top N
                candidate_potential_moves.sort(key=lambda x: x['score'], reverse=True)
                potential_moves_to_evaluate = [cand['move'] for cand in candidate_potential_moves[:15]] # Evaluate top 4 potential

                for pot_move_const in potential_moves_to_evaluate:
                    calculated_damage_val = calculate_damage(
                        params=self.battle_params,
                        attacking_side=1,
                        move=pot_move_const,
                        state=state,
                        attacker=opp_pkm_attacker_view,
                        defender=my_pokemon_target_view
                    )
                    if calculated_damage_val > highest_damage_from_this_opponent:
                        highest_damage_from_this_opponent = calculated_damage_val

            # --- Accumulate threat from this opponent ---
            max_total_incoming_damage_calculated += highest_damage_from_this_opponent

            if highest_damage_from_this_opponent >= my_pokemon_target_view.hp:
                any_single_opponent_can_ko = True # Flag if any single opponent poses a KO threat

        # Determine combined KO likelihood
        is_likely_ko_by_combined_threat = (max_total_incoming_damage_calculated >= my_pokemon_target_view.hp) or \
                                          any_single_opponent_can_ko

        return max_total_incoming_damage_calculated, is_likely_ko_by_combined_threat

    def _score_field_setup_move(self,
                                attacker: BattlingPokemonView,
                                move_constant: Move,
                                my_team_actives: list[BattlingPokemonView],
                                my_team_reserve: list[BattlingPokemonView],
                                opponent_actives: list[BattlingPokemonView],
                                current_state: StateView) -> float:
        """
        Calculates the value of setting a field effect based on the net "damage swing"
        or tactical advantage it provides for the current active Pokemon. [CORRECTED VERSION]
        """
        setup_score = 0.0

        # --- A. Weather/Terrain Damage Swing Calculation ---
        new_weather = move_constant.weather_start
        new_terrain = move_constant.field_start

        if (new_weather != Weather.CLEAR and new_weather != current_state.weather) or \
           (new_terrain != Terrain.NONE and new_terrain != current_state.field):

            net_damage_swing = 0
            all_active_pokemon = my_team_actives + opponent_actives

            # Create a temporary future state to calculate the "after" scenario
            temp_state = State((current_state.sides[0].team, current_state.sides[1].team))
            if new_weather != Weather.CLEAR:
                temp_state.weather = new_weather
            if new_terrain != Terrain.NONE:
                temp_state.field = new_terrain

            for pkm in all_active_pokemon:
                if not pkm or pkm.hp <= 0: continue

                best_move_damage_before = -1
                best_move_damage_after = -1

                for pkm_move in pkm.constants.species.moves:
                    if pkm_move.category not in [Category.PHYSICAL, Category.SPECIAL]: continue

                    damage_before = calculate_damage(self.battle_params, 0, pkm_move, current_state, pkm, attacker)
                    damage_after = calculate_damage(self.battle_params, 0, pkm_move, temp_state, pkm, attacker)

                    if damage_before > best_move_damage_before:
                        best_move_damage_before = damage_before
                        best_move_damage_after = damage_after

                swing_for_this_pkm = best_move_damage_after - best_move_damage_before

                side_multiplier = 1.0 if pkm in my_team_actives else -1.0
                net_damage_swing += (swing_for_this_pkm * side_multiplier)

            setup_score += net_damage_swing

            for pkm in all_active_pokemon:
                if not pkm or pkm.hp <= 0: continue
                side_multiplier = 1.0 if pkm in my_team_actives else -1.0
                if new_weather == Weather.SAND:
                    setup_score += (calculate_sand_damage(self.battle_params, pkm) * side_multiplier * -1)

        # --- B. Trick Room Scoring ---
        if move_constant.toggle_trickroom:
            if not current_state.trickroom:
                net_turn_order_value = 0
                for my_pkm in my_team_actives:
                    for opp_pkm in opponent_actives:
                        if not my_pkm or not opp_pkm: continue
                        if my_pkm.constants.stats[Stat.SPEED] < opp_pkm.constants.stats[Stat.SPEED]:
                            threat_to_opp, _ = self.estimate_incoming_threat(opp_pkm, [my_pkm], current_state)
                            net_turn_order_value += threat_to_opp
                setup_score += net_turn_order_value
            else:
                net_turn_order_value = 0
                for my_pkm in my_team_actives:
                    for opp_pkm in opponent_actives:
                        if not my_pkm or not opp_pkm: continue
                        if my_pkm.constants.stats[Stat.SPEED] > opp_pkm.constants.stats[Stat.SPEED]:
                            threat_to_opp, _ = self.estimate_incoming_threat(opp_pkm, [my_pkm], current_state)
                            net_turn_order_value += threat_to_opp
                        elif my_pkm.constants.stats[Stat.SPEED] < opp_pkm.constants.stats[Stat.SPEED]:
                            threat_from_opp, _ = self.estimate_incoming_threat(my_pkm, [opp_pkm], current_state)
                            net_turn_order_value -= threat_from_opp
                setup_score += net_turn_order_value

        # --- C. Hazard Setting Scoring ---
        if move_constant.hazard != Hazard.NONE:
            hazard_to_set = move_constant.hazard
            opp_side_conditions = current_state.sides[1].conditions

            if (hazard_to_set == Hazard.STEALTH_ROCK and opp_side_conditions.stealth_rock) or \
               (hazard_to_set == Hazard.TOXIC_SPIKES and opp_side_conditions.poison_spikes):
                setup_score -= 100 # Penalize redundancy
            else:
                total_potential_hazard_damage = 0
                opponent_full_team = opponent_actives + [p for p in current_state.sides[1].team.reserve if p and p.hp > 0]

                for opp_pkm in opponent_full_team:
                    if hazard_to_set == Hazard.STEALTH_ROCK:
                        total_potential_hazard_damage += calculate_stealth_rock_damage(self.battle_params, opp_pkm)

                    elif hazard_to_set == Hazard.TOXIC_SPIKES:
                        is_immune_to_tspikes = any(t in [Type.POISON, Type.STEEL, Type.FLYING] for t in opp_pkm.types)
                        if not is_immune_to_tspikes and opp_pkm.status == Status.NONE:
                            total_potential_hazard_damage += calculate_poison_damage(self.battle_params, opp_pkm)

                setup_score += total_potential_hazard_damage

        return setup_score

    def _calculate_stat_boost_value(self,
                                user: BattlingPokemonView,
                                move: Move,
                                state: StateView) -> float:
        """
        Calculates the value of a stat-boosting move based on the net change in damage potential.
        This replaces the old magic-number based scoring.
        """
        if not move.boosts or not any(b != 0 for b in move.boosts):
            return 0.0

        boost_value = 0.0
        opp_actives = [p for p in state.sides[1].team.active if p and p.hp > 0]
        if not opp_actives:
            return 0.0 # No opponents to calculate damage against

        stat_map = [None, Stat.ATTACK, Stat.DEFENSE, Stat.SPECIAL_ATTACK, Stat.SPECIAL_DEFENSE, Stat.SPEED, Stat.ACCURACY, Stat.EVASION]

        if move.self_boosts:
            # --- Calculate value of boosting oneself ---
            total_damage_before = 0
            total_damage_after = 0

            # Find the user's best offensive moves against the current opponents
            best_moves_info = {}
            for opp_idx, opp_pkm in enumerate(opp_actives):
                best_damage_to_target = -1
                best_move_for_target = None
                for user_move in user.battling_moves:
                    if user_move.constants.category in [Category.PHYSICAL, Category.SPECIAL] and user_move.pp > 0:
                        dmg = calculate_damage(self.battle_params, 0, user_move.constants, state, user, opp_pkm)
                        if dmg > best_damage_to_target:
                            best_damage_to_target = dmg
                            best_move_for_target = user_move.constants
                if best_move_for_target:
                    best_moves_info[opp_idx] = {'move': best_move_for_target, 'damage': best_damage_to_target}

            if not best_moves_info: return 0.0

            # Calculate the damage increase
            for opp_idx, info in best_moves_info.items():
                damage_before = info['damage']
                total_damage_before += damage_before

                # Create a temporary boosted version of the user FOR a single calculation
                temp_attacker = BattlingPokemon(user.constants) # Create a fresh BattlingPokemon
                temp_attacker.boosts = list(user.boosts) # Copy current boosts
                # Apply the new move's boosts
                for i, stage_change in enumerate(move.boosts):
                    if i > 0 and i < len(temp_attacker.boosts):
                        temp_attacker.boosts[i] = max(-6, min(6, temp_attacker.boosts[i] + stage_change))

                damage_after = calculate_damage(self.battle_params, 0, info['move'], state, temp_attacker, opp_actives[opp_idx])
                total_damage_after += damage_after

            boost_value = total_damage_after - total_damage_before

        else: # Debuffing opponent(s)
            # --- Calculate value of debuffing the opponent ---
            total_mitigated_damage = 0

            for opp_pkm in opp_actives:
                # Estimate the threat from this single opponent BEFORE the debuff
                damage_before, _ = self.estimate_incoming_threat(user, [opp_pkm], state)

                # Create a temporary debuffed version of the opponent
                temp_defender = BattlingPokemon(opp_pkm.constants)
                temp_defender.boosts = list(opp_pkm.boosts)
                for i, stage_change in enumerate(move.boosts):
                        if i > 0 and i < len(temp_defender.boosts):
                            temp_defender.boosts[i] = max(-6, min(6, temp_defender.boosts[i] + stage_change))

                # Estimate threat AFTER the debuff
                damage_after, _ = self.estimate_incoming_threat(user, [temp_defender], state)
                total_mitigated_damage += (damage_before - damage_after)

            boost_value = total_mitigated_damage

        # Give a small intrinsic value to non-damaging boosts like Speed/Defense
        # to ensure they are not scored as 0 if no immediate damage change is found.
        if boost_value == 0:
            for i, stage_change in enumerate(move.boosts):
                if stage_change == 0 or i >= len(stat_map): continue
                stat_affected = stat_map[i]
                if stat_affected in [Stat.DEFENSE, Stat.SPECIAL_DEFENSE, Stat.SPEED]:
                     boost_value += abs(stage_change) * 20 # Small flat bonus to break ties

        return boost_value

    def _calculate_status_value(self,
                            user: BattlingPokemonView,
                            target: BattlingPokemonView,
                            move: Move,
                            state: StateView) -> float:
        """
        Calculates the value of inflicting a status condition based on its actual
        gameplay impact (damage mitigation, passive damage, turn denial).
        """
        if move.status == Status.NONE or target.status != Status.NONE:
            return 0.0

        status_value = 0.0
        target_max_hp = target.constants.stats[Stat.MAX_HP]
        status_to_inflict = move.status

        # --- Check for immunities first ---
        if status_to_inflict in [Status.POISON, Status.TOXIC] and \
        any(t in [Type.POISON, Type.STEEL] for t in target.types): return 0.0
        if status_to_inflict == Status.PARALYZED and Type.ELECTRIC in target.types: return 0.0
        if status_to_inflict == Status.BURN and Type.FIRE in target.types: return 0.0
        if status_to_inflict == Status.FROZEN and Type.ICE in target.types: return 0.0

        # --- Value Calculation ---
        if status_to_inflict == Status.BURN:
            # Value = passive damage + mitigated physical damage
            passive_damage = calculate_burn_damage(self.battle_params, target)

            # Find mitigated damage if target is a physical attacker
            mitigated_damage = 0
            if target.constants.stats[Stat.ATTACK] >= target.constants.stats[Stat.SPECIAL_ATTACK]:
                # Find best physical move threat against the user
                highest_phys_threat = 0
                for opp_move in target.constants.species.moves:
                    if opp_move.category == Category.PHYSICAL:
                        dmg = calculate_damage(self.battle_params, 1, opp_move, state, target, user)
                        if dmg > highest_phys_threat:
                            highest_phys_threat = dmg
                mitigated_damage = highest_phys_threat * 0.5 # Burn halves physical damage
            status_value = passive_damage + mitigated_damage

        elif status_to_inflict == Status.PARALYZED:
            # Value = mitigated damage from 25% full para chance.
            # Use estimate_incoming_threat for a quick evaluation of the target's general threat.
            threat_from_target, _ = self.estimate_incoming_threat(user, [target], state)
            status_value = threat_from_target * self.battle_params.PARALYSIS_THRESHOLD # 0.25

        elif status_to_inflict in [Status.POISON, Status.TOXIC]:
            status_value = calculate_poison_damage(self.battle_params, target)
            if status_to_inflict == Status.TOXIC:
                status_value *= 1.5 # Simple multiplier to represent ramping damage

        elif status_to_inflict == Status.SLEEP:
            # Value = denying ~2 turns of the opponent's damage
            threat_from_target, _ = self.estimate_incoming_threat(user, [target], state)
            status_value = threat_from_target * 2.0 # Approximate 2 turns of denial

        if status_value > 0:
            status_value += 10

        return status_value

    def _calculate_screen_value(self,
                                move: Move,
                                my_team_actives: list[BattlingPokemonView],
                                opponent_actives: list[BattlingPokemonView],
                                state: StateView) -> float:
        """
        Calculates the value of setting a screen (Reflect/Light Screen) based on the
        total relevant incoming damage it mitigates for the active team.
        """
        if not (move.toggle_reflect or move.toggle_lightscreen):
            return 0.0

        is_reflect = move.toggle_reflect
        screen_category = Category.PHYSICAL if is_reflect else Category.SPECIAL

        # Prevent setting a screen that is already active
        if (is_reflect and state.sides[0].conditions.reflect) or \
           (not is_reflect and state.sides[0].conditions.lightscreen):
            return -100 # Penalize redundant screen setting

        total_mitigated_damage = 0

        for my_pkm in my_team_actives:
            if not my_pkm or my_pkm.hp <= 0: continue

            # Estimate incoming threat of the relevant category for this one pokemon
            total_incoming_threat_to_pkm = 0
            for opp_pkm in opponent_actives:
                if not opp_pkm or opp_pkm.hp <= 0: continue

                # Find the opponent's best move of the relevant category against my_pkm
                best_opp_move_dmg = 0
                for opp_move in opp_pkm.constants.species.moves:
                    if opp_move.category == screen_category:
                        damage = calculate_damage(self.battle_params, 1, opp_move, state, opp_pkm, my_pkm)
                        if damage > best_opp_move_dmg:
                            best_opp_move_dmg = damage

                total_incoming_threat_to_pkm += best_opp_move_dmg

            # Screens halve the damage, so the mitigated value is 50% of the total threat
            total_mitigated_damage += total_incoming_threat_to_pkm * 0.5

        return total_mitigated_damage

    def _calculate_hazard_removal_value(self,
                                        state: StateView) -> float:
        """
        Calculates the value of a hazard-removing move by summing the damage
        that our reserve Pokemon would be saved from taking.
        """
        my_side_conditions = state.sides[0].conditions

        # If there are no hazards to clear, the move has no value in this context.
        if not my_side_conditions.stealth_rock and not my_side_conditions.poison_spikes:
            return 0.0

        total_damage_avoided = 0
        my_reserve_pokemon = [p for p in state.sides[0].team.reserve if p and p.hp > 0]

        for pkm in my_reserve_pokemon:
            if my_side_conditions.stealth_rock:
                total_damage_avoided += calculate_stealth_rock_damage(self.battle_params, pkm)

            if my_side_conditions.poison_spikes:
                # Check for immunity before adding poison damage
                is_immune_to_tspikes = any(t in [Type.POISON, Type.STEEL, Type.FLYING] for t in pkm.types)
                if not is_immune_to_tspikes and pkm.status == Status.NONE:
                    total_damage_avoided += calculate_poison_damage(self.battle_params, pkm)

        return total_damage_avoided

    def _calculate_healing_value(self,
                                 user: BattlingPokemonView,
                                 move: Move,
                                 state: StateView) -> float:
        """
        Calculates the value of a healing move based on how it improves the user's
        survivability against incoming threats.
        """
        if move.heal <= 0:
            return 0.0

        # Calculate how much HP is actually restored
        hp_to_be_healed = user.constants.stats[Stat.MAX_HP] * move.heal
        hp_after_heal = min(user.hp + hp_to_be_healed, user.constants.stats[Stat.MAX_HP])

        # If i are already at full HP, the move has no value
        if user.hp == user.constants.stats[Stat.MAX_HP]:
            return 0.0

        # The value of healing is being able to survive an attack i otherwise wouldn't.
        # Estimate the total incoming threat for this turn
        incoming_threat, _ = self.estimate_incoming_threat(user, state.sides[1].team.active, state)

        # If i would have fainted, but now i survive, the value is immense.
        #  value it as the damage i can deal back on the next turn.
        if user.hp <= incoming_threat and hp_after_heal > incoming_threat:
            # Find the user's best move and calculate its damage potential as the reward
            best_damage_output = 0
            opponent_actives = [p for p in state.sides[1].team.active if p and p.hp > 0]
            if opponent_actives:
                target = opponent_actives[0] # Benchmark against the first opponent
                for user_move in user.constants.species.moves:
                     if user_move.category in [Category.PHYSICAL, Category.SPECIAL]:
                        damage = calculate_damage(self.battle_params, 0, user_move, state, user, target)
                        if damage > best_damage_output:
                            best_damage_output = damage
            return best_damage_output

        # If survive anyway, the value is simply the HP gained,
        # representing a buffer for future turns.
        return hp_to_be_healed

    def _identify_biggest_threat_opponent(self, opponent_actives: list[BattlingPokemonView], my_team_actives: list[BattlingPokemonView],
                                          state: StateView) -> tuple[BattlingPokemonView, int] | None:# retunrs (threat, it's slot index)
        """
        Identifies the opponent's active Pokémon that poses the biggest threat.
        Current Heuristics:
        1. Highest potential damage output against your active Pokemon.
        2. Setup sweepers (e.g., has significant stat boosts).
        Returns the threatening BattlingPokemonView and its slot index, or None.
        """
        if not opponent_actives or not my_team_actives:
            return None

        best_threat_score = -1.0
        biggest_threat_pkm: BattlingPokemonView | None = None
        biggest_threat_index: int = -1

        for opp_idx, opp_pkm in enumerate(opponent_actives):
            if not opp_pkm and opp_pkm.hp < 0:
                continue

            current_opp_threat_score = 0.0

            #Factor 1: OPffensive threat to my team
            # Estimate damage this opponent can do to my most valuable Poki
            max_damage_to_myteam = 0.0
            for my_pkm in my_team_actives:
                if my_pkm and my_pkm.hp > 0:
                    # using a simplified version of _estimate_incoming_thread method
                    # for one opponent vs one target

                    temp_highest_stab = 0.0
                    if opp_pkm.battling_moves:
                        for opp_move in opp_pkm.battling_moves:
                            if opp_move.constants.category != Category.OTHER and opp_move.pp > 0 and not opp_move.disabled:
                                if opp_move.constants.pkm_type in opp_pkm.types:
                                    eff = self._get_type_effectiveness(opp_move.constants.pkm_type, my_pkm.types)
                                    acc = opp_move.constants.accuracy if opp_move.constants.accuracy is not None else 1.0

                                    temp_highest_stab = max(temp_highest_stab, opp_move.constants.base_power * eff * acc)
                    # check for powerful moves
                    if temp_highest_stab < 70 and opp_pkm.constants and opp_pkm.constants.species: # If revealed STABs are weak
                        for p_type in opp_pkm.types:
                            eff = self._get_type_effectiveness(p_type, my_pkm.types)
                            temp_highest_stab_ = max(temp_highest_stab, 80 * eff * 0.9) # Assume 80BP, 90% acc STAB

                    max_damage_to_myteam = max(max_damage_to_myteam, temp_highest_stab)

            current_opp_threat_score += max_damage_to_myteam

            #Factor 2: Setup Sweeper (has offensive boosts)
            #check Atk, Spa, Spe boosts
            offensive_boost = 0

            if opp_pkm.boosts:
                if opp_pkm.boosts[Stat.ATTACK] > 0: offensive_boost += opp_pkm.boosts[Stat.ATTACK]
                if opp_pkm.boosts[Stat.SPECIAL_ATTACK] > 0: offensive_boost += opp_pkm.boosts[Stat.SPECIAL_ATTACK]
                if opp_pkm.boosts[Stat.SPEED] > 0: offensive_boost += opp_pkm.boosts[Stat.SPEED] // 2

            current_opp_threat_score += offensive_boost * 50 #adding points for each stage of offensive boost

            # Factor 3: Generally high base stats
            if opp_pkm.constants and opp_pkm.constants.species:
                base_attack = opp_pkm.constants.species.base_stats[Stat.ATTACK]
                base_spattack = opp_pkm.constants.species.base_stats[Stat.SPECIAL_ATTACK]
                base_speed = opp_pkm.constants.species.base_stats[Stat.SPEED]
                current_opp_threat_score += (base_attack + base_spattack + base_speed) / 10

            if current_opp_threat_score > best_threat_score:
                best_threat_score = current_opp_threat_score
                biggest_threat_pkm = opp_pkm
                biggest_threat_index = opp_idx

        if biggest_threat_pkm:
            return biggest_threat_pkm, biggest_threat_index
        return None

    def _calculate_focus_fire_bonus(self,
                                    pkm_A_view: BattlingPokemonView, move_A_const: Move, is_pkm_A_koing_individually: bool,
                                    pkm_B_view: BattlingPokemonView, move_B_const: Move, is_pkm_B_koing_individually: bool,
                                    target_pkm_view: BattlingPokemonView,
                                    state: StateView,
                                    biggest_threat_on_field: BattlingPokemonView | None) -> float:
        # Self explanatory function to calculate the focus fire bonus
        focus_fire_bonus = 0.0
        if not target_pkm_view or target_pkm_view.hp <= 0:
            return 0.0

        initial_target_hp = target_pkm_view.hp
        combined_ko_achieved_by_focus = False

        damage_from_A = calculate_damage(
            params=self.battle_params, attacking_side=0, move=move_A_const,
            state=state, attacker=pkm_A_view, defender=target_pkm_view
        )
        damage_from_B = calculate_damage(
            params=self.battle_params, attacking_side=0, move=move_B_const,
            state=state, attacker=pkm_B_view, defender=target_pkm_view
        )
        total_calculated_focus_damage = damage_from_A + damage_from_B

        if total_calculated_focus_damage >= initial_target_hp:
            combined_ko_achieved_by_focus = True

        acc_A = move_A_const.accuracy if move_A_const.accuracy is not None else 1.0
        acc_B = move_B_const.accuracy if move_B_const.accuracy is not None else 1.0
        reliability_factor = acc_A * acc_B

        if combined_ko_achieved_by_focus:
            focus_fire_bonus = 750.0 * reliability_factor
            if target_pkm_view == biggest_threat_on_field:
                focus_fire_bonus += 200.0 * reliability_factor
        elif (is_pkm_A_koing_individually or is_pkm_B_koing_individually) and not combined_ko_achieved_by_focus:
            temp_bonus = 0.0
            if is_pkm_A_koing_individually and damage_from_B > 0:
                temp_bonus = 120.0 * acc_A
            elif is_pkm_B_koing_individually and damage_from_A > 0:
                temp_bonus = 120.0 * acc_B
            elif is_pkm_A_koing_individually:
                temp_bonus = 100.0 * acc_A
            elif is_pkm_B_koing_individually:
                temp_bonus = 100.0 * acc_B
            if target_pkm_view == biggest_threat_on_field and temp_bonus > 0:
                temp_bonus += 50.0 * reliability_factor
            focus_fire_bonus = temp_bonus
        elif not is_pkm_A_koing_individually and not is_pkm_B_koing_individually and not combined_ko_achieved_by_focus:
            if initial_target_hp > 0:
              if (damage_from_A / initial_target_hp > 0.3) and \
                 (damage_from_B / initial_target_hp > 0.3):
                focus_fire_bonus = 250.0 * reliability_factor
                if target_pkm_view == biggest_threat_on_field:
                    focus_fire_bonus += 100.0 * reliability_factor
        return focus_fire_bonus

    def decision(self,
                 state: StateView,
                 turn_count: int,
                 opponent_team_battle_view: TeamView | None = None
                ) -> list[BattleCommand] | tuple[list[BattleCommand], dict[str, any]]: # Use Dict for specific typing

        my_team_view = state.sides[0].team

        num_my_active = 0
        active_pokemon_slot_indices = [] # Store original slot indices of our active, non-fainted Pokemon
        for i_slot, pkm_in_slot in enumerate(my_team_view.active):
            if pkm_in_slot is not None and pkm_in_slot.hp > 0:
                num_my_active +=1
                active_pokemon_slot_indices.append(i_slot)

        if PRINT_TYPE_MATCHUPS_DEBUG:
            #print(f"\n--- MyBattlePolicy: Turn {turn_count} ---")
            if num_my_active > 0:
                 pkm_slot0_view = my_team_view.active[active_pokemon_slot_indices[0]]
                 if pkm_slot0_view : self.print_pokemon_type_matchups(pkm_slot0_view)
            if num_my_active > 1:
                 pkm_slot1_view = my_team_view.active[active_pokemon_slot_indices[1]]
                 if pkm_slot1_view : self.print_pokemon_type_matchups(pkm_slot1_view)
            # ... print for opponents ...
            if state.sides[1].team.active: # Check if opponent has active Pokémon
                opp_active_pokemons = state.sides[1].team.active
                if len(opp_active_pokemons) > 0 and opp_active_pokemons[0] and opp_active_pokemons[0].hp > 0:
                    self.print_pokemon_type_matchups(opp_active_pokemons[0])
                if len(opp_active_pokemons) > 1 and opp_active_pokemons[1] and opp_active_pokemons[1].hp > 0:
                    self.print_pokemon_type_matchups(opp_active_pokemons[1])


        # --- 1. Generate all possible individual commands for OUR active Pokémon ---
        # List of lists: outer list per active Pokemon, inner list of (BattleCommand, score) tuples
        possible_actions_for_each_slot: list[list[tuple[BattleCommand, float]]] = [[] for _ in range(len(my_team_view.active))]

        for active_pkm_current_idx, original_slot_idx in enumerate(active_pokemon_slot_indices):
            my_pkm_to_act = my_team_view.active[original_slot_idx] # This is the BattlingPokemonView

            # Evaluate Moves
            if my_pkm_to_act.battling_moves:
                for move_idx, move_in_battle in enumerate(my_pkm_to_act.battling_moves):
                    if move_in_battle is None: continue

                    if move_in_battle.constants.protect:
                        score = self._score_protect_move(my_pkm_to_act, move_in_battle, state)
                        possible_actions_for_each_slot[original_slot_idx].append(
                            ( (move_idx, original_slot_idx), score, False ) # Store (command, score, is_ko_flag)
                        )
                    else: # Non-Protect moves (damaging or status targeting others)
                        opp_targets = state.sides[1].team.active
                        if opp_targets:
                            for target_slot_idx, opp_pkm_target in enumerate(opp_targets):
                                if opp_pkm_target is None or opp_pkm_target.hp <= 0: continue
                                individual_move_score, is_ko_by_this_move = self._score_single_offensive_move(
                                    my_pkm_to_act, move_in_battle, opp_pkm_target, state
                                )
                                possible_actions_for_each_slot[original_slot_idx].append(
                                    ( (move_idx, target_slot_idx), individual_move_score, is_ko_by_this_move )
                                )
                        elif move_in_battle.constants.category != Category.OTHER: # Damaging move but no targets
                            possible_actions_for_each_slot[original_slot_idx].append(
                                ( (move_idx, 0), -100.0, False ) # (command, score, is_ko_flag)
                            )
                        elif move_in_battle.constants.category == Category.OTHER: # Non-damaging status move with no direct target (e.g. Tailwind)
                             # Score from _score_single_offensive_move (utility part)
                            individual_move_score, _ = self._score_single_offensive_move(
                                my_pkm_to_act, move_in_battle, my_pkm_to_act, state # Target self for utility calc if no other target
                            )
                            possible_actions_for_each_slot[original_slot_idx].append(
                                ( (move_idx, original_slot_idx), individual_move_score, False) # Target self or 0, not a KO
                            )


            # Evaluate Switches
            if my_team_view.reserve:
                for reserve_list_idx, reserve_pkm_to_switch in enumerate(my_team_view.reserve):
                    if reserve_pkm_to_switch is None or reserve_pkm_to_switch.hp <= 0: continue
                    opp_targets_for_switch_eval = state.sides[1].team.active if state.sides[1].team.active else []
                    score = self._score_single_switch_action(my_pkm_to_act, reserve_pkm_to_switch, opp_targets_for_switch_eval, state)
                    # Switches don't directly cause KOs in this context
                    possible_actions_for_each_slot[original_slot_idx].append(
                        ( (-1, reserve_list_idx), score, False ) # Store (command, score, is_ko_flag)
                    )

            # If no valid actions found for this Pokémon
            if not possible_actions_for_each_slot[original_slot_idx] and my_pkm_to_act.hp > 0:
                 possible_actions_for_each_slot[original_slot_idx].append(
                     ( (0, 0), -float('inf'), False ) # (command, score, is_ko_flag) -> Struggle
                 )

        # --- 2. Select Best Action(s) ---
        final_commands_to_execute: list[BattleCommand] = []
        # For logging
        pkm0_log_details = {"command": None, "score": -float('inf'), "is_ko": False}
        pkm1_log_details = {"command": None, "score": -float('inf'), "is_ko": False}
        joint_score_log: float = -float('inf')

        if num_my_active == 0:
            pass # final_commands_to_execute remains empty

        elif num_my_active == 1:
            single_pkm_original_slot_idx = active_pokemon_slot_indices[0]
            if possible_actions_for_each_slot[single_pkm_original_slot_idx]:
                # Sort actions for this single Pokémon by score
                sorted_actions = sorted(
                    possible_actions_for_each_slot[single_pkm_original_slot_idx],
                    key=lambda x: x[1],  # x[1] is the score
                    reverse=True
                )

                # Unpack all three elements from the chosen action
                chosen_cmd_tuple, chosen_score, chosen_is_ko_flag = sorted_actions[0]
                final_commands_to_execute.append(chosen_cmd_tuple)

                # For logging, use the chosen_is_ko_flag directly
                pkm0_log_details = {"command": chosen_cmd_tuple, "score": chosen_score, "is_ko": chosen_is_ko_flag}
                joint_score_log = chosen_score # For single active, joint score is its own individual score
            else:
                # If no actions, append default and set default log details
                final_commands_to_execute.append((0,0))
                pkm0_log_details = {"command": (0,0), "score": -float('inf'), "is_ko": False}
                joint_score_log = -float('inf')

        elif num_my_active == 2:
            slot_idx_pkm_A, slot_idx_pkm_B = active_pokemon_slot_indices[0], active_pokemon_slot_indices[1]
            my_pkm_A_view = my_team_view.active[slot_idx_pkm_A]
            my_pkm_B_view = my_team_view.active[slot_idx_pkm_B]

            default_cmd = (0,0) # Default action, e.g., first move, first target
            default_score_info = (default_cmd, -float('inf'), False) # (command, score, is_ko_flag) # (command, score, is_ko_flag)

            best_joint_score_found = -float('inf')
            chosen_pair_for_execution: tuple[BattleCommand, BattleCommand] = (default_cmd, default_cmd)


            actions_pkm_A = possible_actions_for_each_slot[slot_idx_pkm_A] if possible_actions_for_each_slot[slot_idx_pkm_A] else [default_score_info]
            actions_pkm_B = possible_actions_for_each_slot[slot_idx_pkm_B] if possible_actions_for_each_slot[slot_idx_pkm_B] else [default_score_info]

            opp_active_list = state.sides[1].team.active if state.sides[1].team.active else []
            my_active_list = [p for p in my_team_view.active if p and p.hp > 0]

            biggest_threat_info = self._identify_biggest_threat_opponent(opp_active_list, my_active_list, state)
            biggest_threat_opp_pkm_view: BattlingPokemonView | None = None
            biggest_threat_opp_slot_idx: int = -1
            if biggest_threat_info:
                biggest_threat_opp_pkm_view, biggest_threat_opp_slot_idx = biggest_threat_info

            # For logging the chosen individual actions' details
            final_pkmA_log_details = {"command": None, "score": -float('inf'), "is_ko": False}
            final_pkmB_log_details = {"command": None, "score": -float('inf'), "is_ko": False}

            # --- Initial threat assessment for PkmA and PkmB (general threat if they don't defend) ---
            # This is the threat they face if they choose an offensive/non-defensive action.
            threat_to_pkm_A_if_undefended, pkm_A_ko_by_undefended_threat = self.estimate_incoming_threat(
                my_pkm_A_view, opp_active_list, state
            )
            threat_to_pkm_B_if_undefended, pkm_B_ko_by_undefended_threat = self.estimate_incoming_threat(
                my_pkm_B_view, opp_active_list, state
            )

            for cmd_A_info_tuple in actions_pkm_A: # Assuming cmd_A_info_tuple = (cmd_A, score_A, is_cmd_A_ko_flag)
                cmd_A, score_A, is_cmd_A_ko_flag = cmd_A_info_tuple # Unpack

                is_cmd_A_move = cmd_A[0] >= 0
                cmd_A_target_slot = cmd_A[1] if is_cmd_A_move else -1
                move_A_constants = None
                if is_cmd_A_move and cmd_A[0] < len(my_pkm_A_view.battling_moves):
                    move_A_constants = my_pkm_A_view.battling_moves[cmd_A[0]].constants

                for cmd_B_info_tuple in actions_pkm_B:
                    cmd_B, score_B, is_cmd_B_ko_flag = cmd_B_info_tuple
                    is_cmd_B_move = cmd_B[0] >= 0
                    cmd_B_target_slot = cmd_B[1] if is_cmd_B_move else -1
                    move_B_constants = None
                    if is_cmd_B_move and cmd_B[0] < len(my_pkm_B_view.battling_moves):
                        move_B_constants = my_pkm_B_view.battling_moves[cmd_B[0]].constants

                    raw_survival_impact_A = 0.0
                    raw_survival_impact_B = 0.0
                    raw_focus_fire_bonus = 0.0
                    raw_target_priority_bonus = 0.0
                    raw_off_def_support_bonus = 0.0
                    raw_setup_synergy_bonus = 0.0
                    raw_env_synergy_bonus = 0.0

                    # --- 1. SURVIVAL LOGIC FOR PKMA (based on cmd_A and initial general threat to PkmA) ---
                    damage_PkmA_actually_takes = threat_to_pkm_A_if_undefended # From initial calculation before cmd_A loop

                    if is_cmd_A_move and move_A_constants and move_A_constants.protect or not is_cmd_A_move:
                        damage_PkmA_actually_takes = 0

                    if damage_PkmA_actually_takes >= my_pkm_A_view.hp and my_pkm_A_view.hp > 0:
                        is_B_koing_biggest_threat = is_cmd_B_ko_flag and move_B_constants and \
                                                   not move_B_constants.protect and \
                                                   biggest_threat_opp_pkm_view and \
                                                   cmd_B_target_slot == biggest_threat_opp_slot_idx
                        if not is_B_koing_biggest_threat:
                            raw_survival_impact_A  -= (MAX_SCORE * 0.75)
                    elif damage_PkmA_actually_takes > 0:
                        pkmA_max_hp = my_pkm_A_view.constants.stats[Stat.MAX_HP] if my_pkm_A_view.constants else 1.0
                        if pkmA_max_hp <= 0: pkmA_max_hp = 1.0
                        percent_hp_lost_A = damage_PkmA_actually_takes / pkmA_max_hp
                        raw_survival_impact_A -= percent_hp_lost_A * (MAX_SCORE * 0.35)

                    # --- 2. SURVIVAL LOGIC FOR PKMB (based on cmd_B and threat *adjusted for cmd_A's outcome*) ---
                    effective_opponents_for_PkmB_eval = []
                    if is_cmd_A_move and move_A_constants and not move_A_constants.protect and is_cmd_A_ko_flag:
                        for i, opp_pkm_view in enumerate(opp_active_list):
                            if not (i == cmd_A_target_slot and is_cmd_A_ko_flag):
                                if opp_pkm_view and opp_pkm_view.hp > 0:
                                    effective_opponents_for_PkmB_eval.append(opp_pkm_view)
                    else:
                        effective_opponents_for_PkmB_eval = [opp for opp in opp_active_list if opp and opp.hp > 0]

                    threat_to_pkm_B_val_adjusted = 0.0
                    pkm_B_threatened_by_ko_adjusted = False

                    if not effective_opponents_for_PkmB_eval:
                        threat_to_pkm_B_val_adjusted = 0.0
                        pkm_B_threatened_by_ko_adjusted = False
                    else:
                        threat_to_pkm_B_val_adjusted, pkm_B_threatened_by_ko_adjusted = self.estimate_incoming_threat(
                            my_pkm_B_view,
                            effective_opponents_for_PkmB_eval,
                            state
                        )

                    damage_PkmB_actually_takes = threat_to_pkm_B_val_adjusted

                    if is_cmd_B_move and move_B_constants and move_B_constants.protect or not is_cmd_B_move:
                        damage_PkmB_actually_takes = 0

                    if damage_PkmB_actually_takes >= my_pkm_B_view.hp and my_pkm_B_view.hp > 0:
                        is_A_koing_biggest_threat = is_cmd_A_ko_flag and move_A_constants and \
                                                   not move_A_constants.protect and \
                                                   biggest_threat_opp_pkm_view and \
                                                   cmd_A_target_slot == biggest_threat_opp_slot_idx
                        if not is_A_koing_biggest_threat:
                            raw_survival_impact_B  -= (MAX_SCORE * 0.75)
                    elif damage_PkmB_actually_takes > 0:
                        pkmB_max_hp = my_pkm_B_view.constants.stats[Stat.MAX_HP] if my_pkm_B_view.constants else 1.0
                        if pkmB_max_hp <=0: pkmB_max_hp = 1.0
                        percent_hp_lost_B = damage_PkmB_actually_takes / pkmB_max_hp
                        raw_survival_impact_B -= percent_hp_lost_B * (MAX_SCORE * 0.35)

                    # --- End of Core Survival Adjustments ---

                    # --- IV.1 Focus Fire (Your existing refined logic from Phase 1.2) ---
                    focus_fire_bonus_val = 0.0
                    if is_cmd_A_move and is_cmd_B_move and \
                       move_A_constants and move_B_constants and \
                       not move_A_constants.protect and not move_B_constants.protect and \
                       cmd_A_target_slot == cmd_B_target_slot and cmd_A_target_slot != -1:

                        target_idx_ff = cmd_A_target_slot
                        if opp_active_list and target_idx_ff < len(opp_active_list) and opp_active_list[target_idx_ff] is not None:
                            target_pkm_view_for_focus = opp_active_list[target_idx_ff]
                            if target_pkm_view_for_focus.hp > 0:
                                raw_focus_fire_bonus = self._calculate_focus_fire_bonus(
                                    my_pkm_A_view, move_A_constants, is_cmd_A_ko_flag,
                                    my_pkm_B_view, move_B_constants, is_cmd_B_ko_flag,
                                    target_pkm_view_for_focus, state, biggest_threat_opp_pkm_view
                                )


                    # --- IV.2 Target Prioritization: KO'ing the biggest threat ---
                    target_priority_bonus = 0.0
                    if biggest_threat_opp_pkm_view and biggest_threat_opp_pkm_view.hp > 0:
                        ko_threat_by_A_this_turn = is_cmd_A_move and move_A_constants and \
                                                   not move_A_constants.protect and \
                                                   cmd_A_target_slot == biggest_threat_opp_slot_idx and \
                                                   is_cmd_A_ko_flag

                        ko_threat_by_B_this_turn = is_cmd_B_move and move_B_constants and \
                                                   not move_B_constants.protect and \
                                                   cmd_B_target_slot == biggest_threat_opp_slot_idx and \
                                                   is_cmd_B_ko_flag

                        if ko_threat_by_A_this_turn or ko_threat_by_B_this_turn:
                            # Re-calculating:
                            temp_target_prio_base = 0.0
                            if ko_threat_by_A_this_turn or ko_threat_by_B_this_turn:
                                temp_target_prio_base = 450.0
                                for my_active_pkm_target_by_threat in my_active_list:
                                    if my_active_pkm_target_by_threat and my_active_pkm_target_by_threat.hp > 0:
                                        _ , ko_my_active_by_biggest_threat = self.estimate_incoming_threat(
                                            my_active_pkm_target_by_threat, [biggest_threat_opp_pkm_view], state
                                        )
                                        if ko_my_active_by_biggest_threat:
                                            temp_target_prio_base += (75.0 * 1.7)
                                            break
                            raw_target_priority_bonus = temp_target_prio_base # Assign final value

                    # --- IV.3 Offensive + Defensive/Support Pairing (PkmA attacks, PkmB Protects) ---
                    if move_A_constants and move_B_constants:
                        temp_off_def_bonus = 0.0 # Accumulate for this section
                        cmd_A_is_strong_offensive = is_cmd_A_move and not move_A_constants.protect and score_A > 300
                        cmd_B_is_good_protect = is_cmd_B_move and move_B_constants.protect and score_B > 100

                        # Use pkm_B_threatened_by_ko_adjusted (defined in survival block for PkmB)
                        if cmd_A_is_strong_offensive and cmd_B_is_good_protect:
                            if pkm_B_threatened_by_ko_adjusted or score_B > 150:
                                temp_off_def_bonus += (125.0 * 2.5)
                        cmd_B_is_strong_offensive = is_cmd_B_move and not move_B_constants.protect and score_B > 300
                        cmd_A_is_good_protect = is_cmd_A_move and move_A_constants.protect and score_A > 100

                        # Use pkm_A_threatened_by_ko (from initial assessment before cmd_A loop)
                        if cmd_B_is_strong_offensive and cmd_A_is_good_protect:
                            if pkm_A_ko_by_undefended_threat or score_A > 150:
                                temp_off_def_bonus += (125.0 * 2.5)
                        raw_off_def_support_bonus = temp_off_def_bonus

                    # --- IV.3 Remainder: One Pkm uses beneficial setup move + Other attacks/Protects ---
                    if move_A_constants and move_B_constants:
                        temp_setup_synergy = 0.0
                        is_cmd_A_good_general_setup = False
                        if is_cmd_A_move and move_A_constants and score_A > 150:
                            if move_A_constants.category == Category.OTHER or \
                               (move_A_constants.base_power < 50 and not is_cmd_A_ko_flag and (
                                move_A_constants.boosts or move_A_constants.weather_start != Weather.CLEAR or \
                                move_A_constants.field_start != Terrain.NONE or move_A_constants.toggle_trickroom)):
                                is_cmd_A_good_general_setup = True
                        is_cmd_B_good_general_setup = False

                        if is_cmd_B_move and move_B_constants and score_B > 150:
                             if move_B_constants.category == Category.OTHER or \
                               (move_B_constants.base_power < 50 and not is_cmd_B_ko_flag and (
                                move_B_constants.boosts or move_B_constants.weather_start != Weather.CLEAR or \
                                move_B_constants.field_start != Terrain.NONE or move_B_constants.toggle_trickroom)):
                                is_cmd_B_good_general_setup = True

                        if is_cmd_A_good_general_setup and move_B_constants:
                            cmd_B_good_follow_up_offense = is_cmd_B_move and not move_B_constants.protect and \
                                                           not is_cmd_B_ko_flag and score_B > 250
                            cmd_B_good_follow_up_protect = is_cmd_B_move and move_B_constants.protect and score_B > 100

                            if cmd_B_good_follow_up_offense or cmd_B_good_follow_up_protect:
                                temp_setup_synergy += (75.0 * 2.5)

                        if is_cmd_B_good_general_setup and move_A_constants:
                            cmd_A_good_follow_up_offense = is_cmd_A_move and not move_A_constants.protect and \
                                                           not is_cmd_A_ko_flag and score_A > 250
                            cmd_A_good_follow_up_protect = is_cmd_A_move and move_A_constants.protect and score_A > 100

                            if cmd_A_good_follow_up_offense or cmd_A_good_follow_up_protect:
                                temp_setup_synergy += (75.0 * 2.5)
                        raw_setup_synergy_bonus = temp_setup_synergy # CHANGED: Assign to component

                    # --- IV.4 Environmental Effects Synergy (One sets field, other exploits THAT field change) ---
                    raw_env_synergy_bonus = 0.0

                    # Check if both cmd_A and cmd_B are valid moves with constants
                    if is_cmd_A_move and move_A_constants and is_cmd_B_move and move_B_constants:

                        # --- Scenario 1: PkmA sets environment, PkmB's move benefits ---
                        temp_bonus_A_sets_B_benefits = 0.0
                        # PkmA sets Weather, PkmB's move benefits
                        if move_A_constants.weather_start != Weather.CLEAR and \
                           move_A_constants.weather_start != state.weather: # PkmA changes the weather
                            newly_set_weather_by_A = move_A_constants.weather_start
                            if (newly_set_weather_by_A == Weather.RAIN and move_B_constants.pkm_type == Type.WATER) or \
                               (newly_set_weather_by_A == Weather.SUN and move_B_constants.pkm_type == Type.FIRE):
                                temp_bonus_A_sets_B_benefits += (75.0 * 1.8)

                        # PkmA sets Terrain, PkmB's move benefits
                        elif move_A_constants.field_start != Terrain.NONE and \
                             move_A_constants.field_start != state.field: # PkmA changes the terrain
                            newly_set_terrain_by_A = move_A_constants.field_start
                            # Check if PkmB is grounded for terrain effect
                            is_pkmB_grounded = not (Type.FLYING in my_pkm_B_view.types or \
                                    getattr(my_pkm_B_view.constants.species, 'ability', None) == "Levitate")
                            if is_pkmB_grounded:
                                if (newly_set_terrain_by_A == Terrain.ELECTRIC_TERRAIN and move_B_constants.pkm_type == Type.ELECTRIC) or \
                                   (newly_set_terrain_by_A == Terrain.GRASSY_TERRAIN and move_B_constants.pkm_type == Type.GRASS) or \
                                   (newly_set_terrain_by_A == Terrain.PSYCHIC_TERRAIN and move_B_constants.pkm_type == Type.PSYCHIC):
                                    temp_bonus_A_sets_B_benefits += (60.0 * 1.8)

                        # PkmA sets Trick Room, PkmB is slow and benefits
                        elif move_A_constants.toggle_trickroom and not state.trickroom: # PkmA sets TR
                            if my_pkm_B_view.constants and my_pkm_B_view.constants.stats[Stat.SPEED] < 70:
                                if not (move_B_constants.priority > 0):
                                    temp_bonus_A_sets_B_benefits += (80.0 * 1.8)
                        raw_env_synergy_bonus += temp_bonus_A_sets_B_benefits

                        # --- Scenario 2: PkmB sets environment, PkmA's move benefits (Symmetric) ---
                        temp_bonus_B_sets_A_benefits = 0.0
                        # PkmB sets Weather, PkmA's move benefits
                        if move_B_constants.weather_start != Weather.CLEAR and \
                           move_B_constants.weather_start != state.weather: # PkmB changes the weather
                            newly_set_weather_by_B = move_B_constants.weather_start
                            if (newly_set_weather_by_B == Weather.RAIN and move_A_constants.pkm_type == Type.WATER) or \
                               (newly_set_weather_by_B == Weather.SUN and move_A_constants.pkm_type == Type.FIRE):
                                temp_bonus_B_sets_A_benefits += (75.0 * 1.8)

                        # PkmB sets Terrain, PkmA's move benefits
                        elif move_B_constants.field_start != Terrain.NONE and \
                             move_B_constants.field_start != state.field: # PkmB changes the terrain
                            newly_set_terrain_by_B = move_B_constants.field_start
                            is_pkmA_grounded = not (Type.FLYING in my_pkm_A_view.types or \
                                    getattr(my_pkm_A_view.constants.species, 'ability', None) == "Levitate")
                            if is_pkmA_grounded:
                                if (newly_set_terrain_by_B == Terrain.ELECTRIC_TERRAIN and move_A_constants.pkm_type == Type.ELECTRIC) or \
                                   (newly_set_terrain_by_B == Terrain.GRASSY_TERRAIN and move_A_constants.pkm_type == Type.GRASS) or \
                                   (newly_set_terrain_by_B == Terrain.PSYCHIC_TERRAIN and move_A_constants.pkm_type == Type.PSYCHIC):
                                    temp_bonus_B_sets_A_benefits += (60.0 * 1.8)

                        # PkmB sets Trick Room, PkmA is slow and benefits
                        elif move_B_constants.toggle_trickroom and not state.trickroom: # PkmB sets TR
                            if my_pkm_A_view.constants and my_pkm_A_view.constants.stats[Stat.SPEED] < 70:
                                if not (move_A_constants.priority > 0):
                                    temp_bonus_B_sets_A_benefits += (80.0 * 1.8)
                        raw_env_synergy_bonus += temp_bonus_B_sets_A_benefits


                    # --- End of Environmental Effects Synergy ---

                    # calculate the weighted sum
                    current_pair_joint_score = (
                        (score_A * self.W_BASE_SCORE_A) + (score_B * self.W_BASE_SCORE_B) +       # Weighted base scores
                        (raw_survival_impact_A + raw_survival_impact_B) * self.W_SURVIVAL_IMPACT + # Weighted survival (negative), NOTE: REMEMBER TO REFINE!
                        raw_focus_fire_bonus * self.W_FOCUS_FIRE_BONUS +
                        raw_target_priority_bonus * self.W_TARGET_PRIORITY_BONUS +
                        raw_off_def_support_bonus * self.W_OFF_DEF_SUPPORT_BONUS +
                        raw_setup_synergy_bonus * self.W_SETUP_SYNERGY_BONUS +
                        raw_env_synergy_bonus * self.W_ENV_SYNERGY_BONUS
                    )

                    # Check if this pair is the best so far
                    if current_pair_joint_score > best_joint_score_found:
                        best_joint_score_found = current_pair_joint_score
                        chosen_pair_for_execution = (cmd_A, cmd_B)
                        # Use is_cmd_A_ko_flag and is_cmd_B_ko_flag for logging
                        final_pkmA_log_details = {"command": cmd_A, "score": score_A, "is_ko": is_cmd_A_ko_flag}
                        final_pkmB_log_details = {"command": cmd_B, "score": score_B, "is_ko": is_cmd_B_ko_flag}

            # After iterating all pairs, final_commands_to_execute and log details are set for the chosen pair
            final_commands_to_execute.extend(list(chosen_pair_for_execution))
            # Update the main log dicts (pkm0_log_details, pkm1_log_details) with the chosen actions' details
            if num_my_active > 0: # Ensure there's a pkm0 to log for
                 pkm0_log_details.update(final_pkmA_log_details)
            if num_my_active > 1: # Ensure there's a pkm1 to log for
                 pkm1_log_details.update(final_pkmB_log_details)
            joint_score_log = best_joint_score_found

        # Ensure final_commands list has the correct length for engine
        while len(final_commands_to_execute) < num_my_active:
            final_commands_to_execute.append((0,0))
        final_commands_to_execute = final_commands_to_execute[:num_my_active]

        # --- Populate and Return Log Data if enabled ---
        if self.detailed_logging:
            log_output_dict: dict[str, any] = {}
            if pkm0_log_details["command"] is not None:
                cmd = pkm0_log_details["command"]
                action_type = "SWITCH" if cmd[0] == -1 else "MOVE"
                log_output_dict["MyPkm0_Action_Type"] = action_type
                log_output_dict["MyPkm0_Command_Tuple"] = str(cmd)
                log_output_dict["MyPkm0_Individual_Heuristic_Score"] = round(pkm0_log_details["score"], 2) if pkm0_log_details["score"] != -float('inf') else "N/A"
                log_output_dict["MyPkm0_Is_KO_Action"] = pkm0_log_details["is_ko"]


            if pkm1_log_details["command"] is not None and num_my_active == 2:
                cmd = pkm1_log_details["command"]
                action_type = "SWITCH" if cmd[0] == -1 else "MOVE"
                log_output_dict["MyPkm1_Action_Type"] = action_type
                log_output_dict["MyPkm1_Command_Tuple"] = str(cmd)
                log_output_dict["MyPkm1_Individual_Heuristic_Score"] = round(pkm1_log_details["score"], 2) if pkm1_log_details["score"] != -float('inf') else "N/A"
                log_output_dict["MyPkm1_Is_KO_Action"] = pkm1_log_details["is_ko"]


            if joint_score_log != -float('inf'):
                 log_output_dict["Joint_Heuristic_Score"] = round(joint_score_log, 2)

            return final_commands_to_execute, log_output_dict
        else:
            return final_commands_to_execute
