"""Thin battle policy orchestrator for the Dongimon agent.

Delegates move scoring, threat estimation, and joint action
evaluation to specialized submodules. Loads heuristic weights
from configuration.
"""

from typing import Any

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleCommand
from vgc2.battle_engine.modifiers import Category
from vgc2.battle_engine.view import StateView, TeamView

from src.battle.joint import evaluate_joint_actions
from src.battle.move_scoring import (
    score_offensive_move,
    score_protect_move,
    score_switch_action,
)
from src.config.loader import load_battle_weights

MAX_SCORE = 1000.0


class DongimonBattlePolicy(BattlePolicy):
    """Battle policy using an 8-component heuristic with joint action pairing.

    Evaluates all possible individual actions (moves, switches) for each
    active Pokemon, then pairs them across both active slots to compute
    cross-slot synergy bonuses (focus fire, survival, target priority,
    off-def support, setup synergy, environmental synergy).

    Weights are loaded from ``src/config/battle_weights.yaml``
    and can be overridden at construction time.
    """

    def __init__(self, detailed_logging: bool = False, custom_weights: dict[str, float] | None = None):
        """Initialize the battle policy.

        Args:
            detailed_logging: If True, returns a log dictionary alongside
                the command list from decision().
            custom_weights: Optional dictionary of weight overrides
                (keys: w_focus_fire, w_survival_impact, etc.).
        """
        super().__init__()
        self.detailed_logging = detailed_logging

        weights = load_battle_weights()
        self._weights = weights.model_dump()

        if custom_weights:
            for key, value in custom_weights.items():
                if hasattr(weights, key):
                    self._weights[key] = value

    def decision(
        self,
        state: StateView,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand] | tuple[list[BattleCommand], dict[str, Any]]:
        """Decide actions for all active Pokemon this turn.

        Generates all possible actions (moves against each target,
        switches to each reserve), scores them individually, then
        pairs them across both active slots to compute joint scores
        with synergy bonuses.

        Args:
            state: Current battle state view from this agent's perspective.
            opp_view: Opponent team view (unused in current implementation).

        Returns:
            If detailed_logging is False: list of BattleCommand tuples.
            If detailed_logging is True: tuple of (commands, log_dict).
        """
        my_team_view = state.sides[0].team

        num_my_active = 0
        active_slots = []
        for i_slot, pkm in enumerate(my_team_view.active):
            if pkm is not None and pkm.hp > 0:
                num_my_active += 1
                active_slots.append(i_slot)

        actions_per_slot: list = [[] for _ in range(len(my_team_view.active))]

        for _, original_slot in enumerate(active_slots):
            my_pkm = my_team_view.active[original_slot]

            if my_pkm.battling_moves:
                for move_idx, move in enumerate(my_pkm.battling_moves):
                    if move is None:
                        continue

                    if move.constants.protect:
                        score = score_protect_move(my_pkm, move, state, self.params)
                        actions_per_slot[original_slot].append(((move_idx, original_slot), score, False))
                    else:
                        opp_targets = state.sides[1].team.active
                        if opp_targets:
                            for target_slot, opp_target in enumerate(opp_targets):
                                if opp_target is None or opp_target.hp <= 0:
                                    continue
                                move_score, is_ko = score_offensive_move(
                                    my_pkm, move, opp_target, state, self.params,
                                )
                                actions_per_slot[original_slot].append(
                                    ((move_idx, target_slot), move_score, is_ko)
                                )
                        elif move.constants.category != Category.OTHER:
                            actions_per_slot[original_slot].append(((move_idx, 0), -100.0, False))
                        elif move.constants.category == Category.OTHER:
                            move_score, _ = score_offensive_move(
                                my_pkm, move, my_pkm, state, self.params,
                            )
                            actions_per_slot[original_slot].append(
                                ((move_idx, original_slot), move_score, False)
                            )

            if my_team_view.reserve:
                move_nums = []
                for act in actions_per_slot[original_slot]:
                    if act[0][0] >= 0:
                        _, sc, _ = act
                        move_nums.append(sc)
                avg_move = sum(move_nums) / max(len(move_nums), 1)
                for reserve_idx, reserve_pkm in enumerate(my_team_view.reserve):
                    if reserve_pkm is None or reserve_pkm.hp <= 0:
                        continue
                    opps = state.sides[1].team.active if state.sides[1].team.active else []
                    score = score_switch_action(my_pkm, reserve_pkm, opps, state, self.params, avg_move)
                    actions_per_slot[original_slot].append(((-1, reserve_idx), score, False))

            if not actions_per_slot[original_slot] and my_pkm.hp > 0:
                actions_per_slot[original_slot].append(((0, 0), -float("inf"), False))

        final_commands: list[BattleCommand] = []
        pkm0_log = {"command": None, "score": -float("inf"), "is_ko": False}
        pkm1_log = {"command": None, "score": -float("inf"), "is_ko": False}
        joint_log = -float("inf")

        if num_my_active == 0:
            pass

        elif num_my_active == 1:
            single_slot = active_slots[0]
            if actions_per_slot[single_slot]:
                sorted_actions = sorted(
                    actions_per_slot[single_slot],
                    key=lambda x: x[1],
                    reverse=True,
                )
                cmd, sc, ko = sorted_actions[0]
                final_commands.append(cmd)
                pkm0_log = {"command": cmd, "score": sc, "is_ko": ko}
                joint_log = sc
            else:
                final_commands.append((0, 0))
                joint_log = -float("inf")

        elif num_my_active == 2:
            slot_a, slot_b = active_slots[0], active_slots[1]
            pkm_a = my_team_view.active[slot_a]
            pkm_b = my_team_view.active[slot_b]

            opps = state.sides[1].team.active if state.sides[1].team.active else []
            my_active = [p for p in my_team_view.active if p and p.hp > 0]

            actions_a = actions_per_slot[slot_a] if actions_per_slot[slot_a] else [((0, 0), -float("inf"), False)]
            actions_b = actions_per_slot[slot_b] if actions_per_slot[slot_b] else [((0, 0), -float("inf"), False)]

            cmds, log_a, log_b, joint = evaluate_joint_actions(
                actions_a, actions_b,
                pkm_a, pkm_b,
                opps, my_active,
                state, self.params,
                self._weights, MAX_SCORE,
            )
            final_commands.extend(cmds)
            pkm0_log = log_a
            pkm1_log = log_b
            joint_log = joint

        while len(final_commands) < num_my_active:
            final_commands.append((0, 0))
        final_commands = final_commands[:num_my_active]

        if self.detailed_logging:
            log: dict[str, Any] = {}
            if pkm0_log["command"] is not None:
                cmd = pkm0_log["command"]
                log["Pkm0_Action_Type"] = "SWITCH" if cmd[0] == -1 else "MOVE"
                log["Pkm0_Score"] = round(pkm0_log["score"], 2) if pkm0_log["score"] != -float("inf") else None
                log["Pkm0_Is_KO"] = pkm0_log["is_ko"]

            if pkm1_log["command"] is not None and num_my_active == 2:
                cmd = pkm1_log["command"]
                log["Pkm1_Action_Type"] = "SWITCH" if cmd[0] == -1 else "MOVE"
                log["Pkm1_Score"] = round(pkm1_log["score"], 2) if pkm1_log["score"] != -float("inf") else None
                log["Pkm1_Is_KO"] = pkm1_log["is_ko"]

            if joint_log != -float("inf"):
                log["Joint_Score"] = round(joint_log, 2)

            return final_commands, log

        return final_commands
