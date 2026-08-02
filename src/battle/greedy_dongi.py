"""Net-damage battle policy with one-turn opponent simulation.

Copies Greedy's exhaustive enumeration pattern and replaces its
offense-only scoring with a net-damage simulation that includes
opponent Greedy response, resolved in the engine's priority+speed
order. Strategic behavior (focus fire, target priority, protect
timing) emerges from accurate simulation — never from weights.

Scoring is lexicographic (tuple comparison):
    (opp_kos, -our_kos, damage_dealt - damage_taken)

No tunable parameters. No normalization. No balancing.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleCommand, calculate_damage
from vgc2.battle_engine.modifiers import Category
from vgc2.battle_engine.priority_calculator import priority_calculator

if TYPE_CHECKING:
    from vgc2.battle_engine.view import BattlingPokemonView, StateView, TeamView


def _precompute_opponent_actions(
    params: object,
    state: StateView,
    attackers: list[BattlingPokemonView],
    defenders: list[BattlingPokemonView],
) -> tuple[list[list[float]], list[float]]:
    """Precompute each opponent's capped damage table and speed key.

    Returns per-opponent best capped damage against each of our Pokemon,
    and the opponent's priority+speed sort key (highest acts first). Both
    are constant across joint actions, so they are computed once per
    decision.

    Args:
        params: Battle rule parameters for damage calculation.
        state: Current battle state view.
        attackers: Our active Pokemon list.
        defenders: Opponent active Pokemon list.

    Returns:
        Tuple of (damage_table, priority_keys) where damage_table[i][j]
        is opponent i's best capped damage against our Pokemon j, and
        priority_keys[i] is opponent i's max priority_calculator key.
    """
    damage_table: list[list[float]] = []
    priority_keys: list[float] = []
    for opp in defenders:
        best_key = 0.0
        for move in opp.battling_moves:
            if move.pp <= 0 or move.disabled:
                continue
            const = move.constants
            if const.category == Category.OTHER or const.base_power <= 0:
                continue
            key = priority_calculator(params, const, opp, state)
            if key > best_key:
                best_key = key
        row: list[float] = []
        for our in attackers:
            best = 0.0
            for move in opp.battling_moves:
                if move.pp <= 0 or move.disabled:
                    continue
                const = move.constants
                if const.category == Category.OTHER or const.base_power <= 0:
                    continue
                raw = calculate_damage(params, 1, const, state, opp, our)
                capped = min(float(raw), float(our.hp))
                if capped > best:
                    best = capped
            row.append(best)
        damage_table.append(row)
        priority_keys.append(best_key)
    return damage_table, priority_keys


def _resolve_turn(
    params: object,
    state: StateView,
    attackers: list[BattlingPokemonView],
    defenders: list[BattlingPokemonView],
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    protecting: list[bool],
    opp_damage: list[list[float]],
    opp_priority: list[float],
) -> tuple[int, int, float, int]:
    """Resolve our joint attack and opponent Greedy response in speed order.

    All attacks (two ours, two opponents') execute in the engine's
    priority+speed order. A Pokemon KO'd before its action does not act,
    so focus fire can deny an opponent's attack when we are faster.
    Sequential HP tracking is applied throughout.

    Args:
        params: Battle rule parameters for damage calculation.
        state: Current battle state view.
        attackers: Our active Pokemon list.
        defenders: Opponent active Pokemon list.
        sources: Move index chosen per attacker slot (-1 = switch).
        targets: Target index chosen per attacker slot.
        protecting: Per-slot flags for our Pokemon using Protect this turn.
        opp_damage: Precomputed per-opponent capped damage per our Pokemon.
        opp_priority: Precomputed per-opponent priority+speed sort keys.

    Returns:
        Tuple of (damage_dealt, opp_kos, damage_taken, our_kos).
    """
    our_hp = [p.hp for p in attackers]
    opp_hp = [d.hp for d in defenders]

    actions: list[tuple[float, int, int, Any, int]] = []
    for i, (source, target) in enumerate(zip(sources, targets, strict=True)):
        if source < 0:
            continue
        if i >= len(attackers) or our_hp[i] <= 0:
            continue
        move = attackers[i].battling_moves[source]
        if move.pp <= 0 or move.disabled:
            continue
        key = priority_calculator(params, move.constants, attackers[i], state)
        actions.append((float(key), 0, i, move.constants, target))

    for opp_idx, opp in enumerate(defenders):
        if opp.hp <= 0:
            continue
        actions.append((opp_priority[opp_idx], 1, opp_idx, None, 0))

    actions.sort(key=lambda a: a[0], reverse=True)

    damage_dealt = 0
    opp_kos = 0
    damage_taken = 0.0
    our_kos = 0

    for _key, side, idx, move_or_none, target in actions:
        if side == 0:
            if our_hp[idx] <= 0 or opp_hp[target] <= 0:
                continue
            raw = calculate_damage(params, 0, move_or_none, state, attackers[idx], defenders[target])
            actual = min(raw, opp_hp[target])
            actual = max(0, actual)
            damage_dealt += actual
            opp_hp[target] -= actual
            if opp_hp[target] <= 0:
                opp_kos += 1
        else:
            if opp_hp[idx] <= 0:
                continue
            best_dmg = 0.0
            best_target = 0
            for t_idx in range(len(our_hp)):
                if our_hp[t_idx] <= 0 or protecting[t_idx]:
                    continue
                dmg = min(opp_damage[idx][t_idx], float(our_hp[t_idx]))
                if dmg > best_dmg:
                    best_dmg = dmg
                    best_target = t_idx
            if best_dmg > 0:
                damage_taken += best_dmg
                our_hp[best_target] -= best_dmg
                if our_hp[best_target] <= 0:
                    our_kos += 1

    return damage_dealt, opp_kos, damage_taken, our_kos


class GreedyDongiPolicy(BattlePolicy):  # type: ignore[misc]
    """Net-damage battle policy with one-turn opponent simulation.

    Enumerates all joint action combinations (moves × targets + switches),
    then scores each by lexicographic tuple:
        (opp_kos, -our_kos, damage_dealt - damage_taken)

    The turn is resolved in the engine's priority+speed order: our joint
    action and the opponent Greedy response execute interleaved, so a
    Pokemon KO'd before its action does not act. Protect moves zero out
    incoming damage for that slot. No weights, no tunable parameters.
    """

    def decision(
        self,
        state: StateView,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        """Select the joint action maximizing net-damage score.

        Action space per slot: (move × target). Each joint action is
        scored by a speed-ordered resolution of our attacks and the
        opponent Greedy response.

        Args:
            state: Current battle state view (side 0 = own team).
            opp_view: Opponent team view (unused, kept for interface).

        Returns:
            List of BattleCommand tuples [(move_idx, target_idx), ...].
        """
        attackers = state.sides[0].team.active
        defenders = state.sides[1].team.active
        n_defenders = len(defenders)

        if not attackers or not defenders:
            return [(0, 0)]

        actions_per_slot: list[list[BattleCommand]] = []
        protect_sets: list[set[int]] = []
        for pkm in attackers:
            slot_actions: list[BattleCommand] = []
            p_set: set[int] = set()
            valid_moves: list[int] = []
            for i, m in enumerate(pkm.battling_moves):
                if m.pp > 0 and not m.disabled:
                    valid_moves.append(i)
                    if m.constants.protect:
                        p_set.add(i)
            if not valid_moves:
                valid_moves = [0]
            for mi in valid_moves:
                for ti in range(n_defenders):
                    slot_actions.append((mi, ti))
            actions_per_slot.append(slot_actions)
            protect_sets.append(p_set)

        opp_damage, opp_priority = _precompute_opponent_actions(
            self.params, state, attackers, defenders,
        )

        best_score: tuple[int, int, float] | None = None
        best_cmds: list[BattleCommand] = [(0, 0)] * len(attackers)

        for joint in product(*actions_per_slot):
            protecting = [False] * len(attackers)
            sources: list[int] = []
            targets: list[int] = []

            for i, cmd in enumerate(joint):
                sources.append(cmd[0])
                targets.append(cmd[1])
                if cmd[0] in protect_sets[i]:
                    protecting[i] = True

            dmg_dealt, opp_kos, dmg_taken, our_kos = _resolve_turn(
                self.params, state, attackers, defenders,
                tuple(sources), tuple(targets), protecting,
                opp_damage, opp_priority,
            )

            score = (opp_kos, -our_kos, dmg_dealt - dmg_taken)

            if best_score is None or score > best_score:
                best_score = score
                best_cmds = list(joint)

        return best_cmds
