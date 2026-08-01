"""Net-damage battle policy with one-turn opponent simulation.

Copies Greedy's exhaustive enumeration pattern and replaces its
offense-only scoring with a net-damage simulation that includes
opponent Greedy response. Strategic behavior (focus fire, target
priority) emerges from accurate simulation — never from weights.

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
    from vgc2.battle_engine.view import BattlingPokemonView, StateView, TeamView


def _simulate_our_attacks(
    params: object,
    state: StateView,
    attackers: list[BattlingPokemonView],
    defenders: list[BattlingPokemonView],
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[int, int]:
    """Simulate our joint attack and return (damage_dealt, opp_kos).

    Sequential HP tracking: if both Pokemon target the same defender,
    the second hit is computed against reduced HP (focus fire emerges).

    Args:
        params: Battle rule parameters for damage calculation.
        state: Current battle state view.
        attackers: Our active Pokemon list.
        defenders: Opponent active Pokemon list.
        sources: Move index chosen per attacker slot.
        targets: Target index chosen per attacker slot.

    Returns:
        Tuple of (total_damage_dealt, number_of_opponent_kos).
    """
    damage = 0
    kos = 0
    hp = [d.hp for d in defenders]

    for i, (source, target) in enumerate(zip(sources, targets)):
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

    return damage, kos


def _simulate_opponent_response(
    params: object,
    state: StateView,
    our_active: list[BattlingPokemonView],
    opp_active: list[BattlingPokemonView],
) -> tuple[int, int]:
    """Simulate opponent Greedy response and return (damage_taken, our_kos).

    Assumes opponent plays Greedy: each opponent Pokemon picks its
    highest-damage move against each of our Pokemon. Sequential HP
    tracking applied (opponent focus fire emerges naturally).

    Args:
        params: Battle rule parameters for damage calculation.
        state: Current battle state view.
        our_active: Our active Pokemon list.
        opp_active: Opponent active Pokemon list.

    Returns:
        Tuple of (total_damage_taken, number_of_our_pokemon_koed).
    """
    damage = 0
    kos = 0
    hp = [p.hp for p in our_active]

    for opp in opp_active:
        if opp.hp <= 0:
            continue
        best_dmg = 0
        best_target = 0
        for t_idx, my_pkm in enumerate(our_active):
            if my_pkm.hp <= 0 or hp[t_idx] <= 0:
                continue
            for move in opp.battling_moves:
                if move.pp <= 0 or move.disabled:
                    continue
                dmg = calculate_damage(params, 1, move.constants, state, opp, my_pkm)
                capped = min(dmg, hp[t_idx])
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

    Enumerates all joint move combinations (same as Greedy), then
    scores each by lexicographic tuple:
        (opp_kos, -our_kos, damage_dealt - damage_taken)

    The opponent response is simulated assuming Greedy play (pick
    max-damage move). No weights, no tunable parameters.
    """

    def decision(
        self,
        state: StateView,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        """Select the joint action maximizing net-damage score.

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

        move_ranges: list[list[int]] = []
        for pkm in attackers:
            valid = [
                i for i, m in enumerate(pkm.battling_moves)
                if m.pp > 0 and not m.disabled
            ]
            move_ranges.append(valid if valid else [0])

        target_range = list(range(n_defenders))

        best_score: tuple[int, int, int] | None = None
        best_cmds: list[BattleCommand] = [(0, 0)] * len(attackers)

        for sources in product(*move_ranges):
            for targets in product(target_range, repeat=len(attackers)):
                dmg_dealt, opp_kos = _simulate_our_attacks(
                    self.params, state, attackers, defenders, sources, targets,
                )
                dmg_taken, our_kos = _simulate_opponent_response(
                    self.params, state, attackers, defenders,
                )

                score = (opp_kos, -our_kos, dmg_dealt - dmg_taken)

                if best_score is None or score > best_score:
                    best_score = score
                    best_cmds = list(zip(sources, targets))

        return best_cmds
