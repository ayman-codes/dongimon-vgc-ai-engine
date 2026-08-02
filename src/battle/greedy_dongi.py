"""Net-damage battle policy with one-turn opponent simulation.

Copies Greedy's exhaustive enumeration pattern and replaces its
offense-only scoring with a net-damage simulation that includes
opponent Greedy response. Strategic behavior (focus fire, target
priority, protect timing, switch timing) emerges from accurate
simulation — never from weights.

Scoring is lexicographic (tuple comparison):
    (opp_kos, -our_kos, damage_dealt - damage_taken)

No tunable parameters. No normalization. No balancing.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleCommand, calculate_damage

if TYPE_CHECKING:
    from vgc2.battle_engine.pokemon import BattlingPokemon
    from vgc2.battle_engine.view import BattlingPokemonView, StateView, TeamView


def _simulate_our_attacks(
    params: object,
    state: StateView,
    attackers: list[BattlingPokemonView],
    defenders: list[BattlingPokemonView],
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[int, int, list[int]]:
    """Simulate our joint attack and return (damage_dealt, opp_kos, opp_hp_after).

    Sequential HP tracking: if both Pokemon target the same defender,
    the second hit is computed against reduced HP (focus fire emerges).
    Slots with source < 0 are switches and deal no damage.

    Args:
        params: Battle rule parameters for damage calculation.
        state: Current battle state view.
        attackers: Our active Pokemon list.
        defenders: Opponent active Pokemon list.
        sources: Move index chosen per attacker slot (-1 = switch).
        targets: Target index chosen per attacker slot.

    Returns:
        Tuple of (total_damage_dealt, number_of_opponent_kos,
        remaining HP per defender after our attacks).
    """
    damage = 0
    kos = 0
    hp = [d.hp for d in defenders]

    for i, (source, target) in enumerate(zip(sources, targets, strict=True)):
        if source < 0:
            continue
        attacker = attackers[i]
        move = attacker.battling_moves[source]
        if move.pp <= 0 or move.disabled:
            continue
        defender = defenders[target]
        if defender.hp <= 0:
            continue
        raw = calculate_damage(params, 0, move.constants, state, attacker, defender)
        actual = min(raw, hp[target])
        actual = max(0, actual)
        damage += actual
        hp[target] -= actual
        if hp[target] <= 0:
            kos += 1

    return damage, kos, hp


def _simulate_opponent_response(
    params: object,
    state: StateView,
    our_slots: list[BattlingPokemon | BattlingPokemonView],
    opp_active: list[BattlingPokemonView],
    opp_hp_after: list[int],
    protecting: list[bool] | None = None,
) -> tuple[int, int]:
    """Simulate opponent Greedy response and return (damage_taken, our_kos).

    Assumes opponent plays Greedy: each surviving opponent Pokemon picks
    its highest-damage move against each of our Pokemon. Sequential HP
    tracking applied (opponent focus fire emerges naturally).

    Args:
        params: Battle rule parameters for damage calculation.
        state: Current battle state view.
        our_slots: Our active Pokemon list.
        opp_active: Opponent active Pokemon list.
        opp_hp_after: Remaining HP per opponent after our attacks.
        protecting: Per-slot flags for our Pokemon using Protect this
            turn. Protecting slots take 0 incoming damage.

    Returns:
        Tuple of (total_damage_taken, number_of_our_pokemon_koed).
    """
    if protecting is None:
        protecting = [False] * len(our_slots)

    damage = 0
    kos = 0
    hp = [p.hp for p in our_slots]

    for opp_idx, opp in enumerate(opp_active):
        if opp.hp <= 0 or opp_hp_after[opp_idx] <= 0:
            continue
        best_dmg = 0
        best_target = 0
        for t_idx in range(len(our_slots)):
            if our_slots[t_idx].hp <= 0 or hp[t_idx] <= 0:
                continue
            if protecting[t_idx]:
                continue
            for move in opp.battling_moves:
                if move.pp <= 0 or move.disabled:
                    continue
                raw = calculate_damage(
                    params, 1, move.constants, state, opp, our_slots[t_idx],
                )
                capped = min(raw, hp[t_idx])
                if capped > best_dmg:
                    best_dmg = capped
                    best_target = t_idx
        if best_dmg > 0:
            damage += best_dmg
            hp[best_target] -= best_dmg
            if hp[best_target] <= 0:
                kos += 1

    return damage, kos


class GreedyDongiPolicy(BattlePolicy):  # type: ignore[misc]
    """Net-damage battle policy with one-turn opponent simulation.

    Enumerates all joint action combinations (moves × targets + switches),
    then scores each by lexicographic tuple:
        (opp_kos, -our_kos, damage_dealt - damage_taken)

    The opponent response is simulated assuming Greedy play (pick
    max-damage move). Protect moves zero out incoming damage for
    that slot. Switches replace the defender used in response sim.
    No weights, no tunable parameters.
    """

    def decision(
        self,
        state: StateView,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        """Select the joint action maximizing net-damage score.

        Action space per slot: (move × target) + (switch to reserve).
        Opponent response is cached by (protect_pattern, switch_pattern)
        for efficiency.

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

        # Build per-slot action lists: attacks only.
        # Switching is disabled: the one-turn lookahead cannot model
        # tempo loss or future positioning, causing net regression.
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

        opp_hp_original = [d.hp for d in defenders]

        # Cache opponent response by protect pattern
        response_cache: dict[tuple[bool, ...], tuple[int, int]] = {}

        best_score: tuple[int, int, int] | None = None
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

            cache_key = tuple(protecting)
            if cache_key not in response_cache:
                response_cache[cache_key] = _simulate_opponent_response(
                    self.params, state, list(attackers), defenders,
                    opp_hp_original, protecting,
                )
            dmg_taken, our_kos = response_cache[cache_key]

            dmg_dealt, opp_kos, _ = _simulate_our_attacks(
                self.params, state, attackers, defenders,
                tuple(sources), tuple(targets),
            )

            score = (opp_kos, -our_kos, dmg_dealt - dmg_taken)

            if best_score is None or score > best_score:
                best_score = score
                best_cmds = list(joint)

        return best_cmds
