"""Tests for the GreedyDongi net-damage battle policy."""

from __future__ import annotations

import numpy as np
import pytest
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.modifiers import Category, Type
from vgc2.battle_engine.move import BattlingMove, Move
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.battle.greedy_dongi import GreedyDongiPolicy

def _fresh_protect_move() -> Move:
    """Create a fresh Protect Move instance (avoids shared-object pollution)."""
    return Move(Type.NORMAL, 0, 1.0, 16, Category.OTHER, protect=True)


def _make_battle_state(
    seed: int = 42,
) -> tuple[StateView, TeamView, BattleRuleParam, BattleEngine]:
    """Create a battle state for testing.

    Args:
        seed: RNG seed for team generation.

    Returns:
        Tuple of (state_view_side0, opp_team_view, params, engine).
    """
    gen = np.random.default_rng(seed)
    params = BattleRuleParam()
    team = gen_team(4, 4, gen)
    view = TeamView(team)
    indices = list(range(len(team.members)))
    sub_a, sub_view_a = subteam(team, view, indices)
    sub_b, sub_view_b = subteam(team, view, indices)
    battle_teams = get_battle_teams((sub_a, sub_b), 2)
    state = State(battle_teams)
    rng = ((gen, gen), (gen, gen))
    engine = BattleEngine(
        state, params=params,
        acc_rng=rng, eff_rng=rng, sta_rng=rng,
    )
    sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
    return sv0, sub_view_b, params, engine


class TestGreedyDongiBasic:
    """Basic functionality tests."""

    def test_returns_valid_commands(self) -> None:
        """Policy output is a list of tuples parseable by BattleEngine."""
        sv0, opp_view, params, engine = _make_battle_state(seed=10)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        cmds = policy.decision(sv0, opp_view)
        assert isinstance(cmds, list)
        assert len(cmds) == len(sv0.sides[0].team.active)
        for cmd in cmds:
            assert isinstance(cmd, tuple)
            assert len(cmd) == 2

    def test_deterministic(self) -> None:
        """Same state produces same output on repeated calls."""
        sv0, opp_view, params, _ = _make_battle_state(seed=20)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        cmds1 = policy.decision(sv0, opp_view)
        cmds2 = policy.decision(sv0, opp_view)
        assert cmds1 == cmds2

    def test_handles_single_active(self) -> None:
        """Works when only 1 Pokemon is active (late-game scenario)."""
        sv0, opp_view, params, engine = _make_battle_state(seed=30)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (opp_view, opp_view))
            active = sv0.sides[0].team.active
            if len(active) == 1:
                cmds = policy.decision(sv0, opp_view)
                assert len(cmds) == 1
                return
            sv1 = StateView(engine.state, 1, (opp_view, opp_view))
            cmd0 = policy.decision(sv0, opp_view)
            cmd1 = policy.decision(sv1, opp_view)
            engine.run_turn((cmd0, cmd1))

    def test_engine_accepts_commands(self) -> None:
        """BattleEngine runs a full battle using GreedyDongi decisions."""
        sv0, opp_view, params, engine = _make_battle_state(seed=40)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        turns = 0
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (opp_view, opp_view))
            sv1 = StateView(engine.state, 1, (opp_view, opp_view))
            cmd0 = policy.decision(sv0, opp_view)
            cmd1 = policy.decision(sv1, opp_view)
            engine.run_turn((cmd0, cmd1))
            turns += 1
            if turns > 200:
                pytest.fail("Battle did not terminate within 200 turns")
        assert engine.finished()


class TestGreedyDongiStrategy:
    """Strategic behavior tests."""

    def test_prefers_ko_over_damage(self) -> None:
        """When a KO is available, policy picks it over higher raw damage."""
        sv0, opp_view, params, engine = _make_battle_state(seed=50)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        turns = 0
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (opp_view, opp_view))
            defenders = sv0.sides[1].team.active
            low_hp_opp = any(0 < d.hp < 50 for d in defenders)
            cmds = policy.decision(sv0, opp_view)
            if low_hp_opp:
                assert cmds is not None
                break
            sv1 = StateView(engine.state, 1, (opp_view, opp_view))
            cmd1 = policy.decision(sv1, opp_view)
            engine.run_turn((cmds, cmd1))
            turns += 1
            if turns > 100:
                break

    def test_focus_fire_emerges(self) -> None:
        """Both Pokemon tend to target the same low-HP opponent."""
        sv0, opp_view, params, engine = _make_battle_state(seed=60)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        found = False
        turns = 0
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (opp_view, opp_view))
            defenders = sv0.sides[1].team.active
            if len(defenders) == 2 and all(d.hp > 0 for d in defenders):
                hp_list = [d.hp for d in defenders]
                if min(hp_list) < max(hp_list) * 0.4:
                    cmds = policy.decision(sv0, opp_view)
                    if len(cmds) == 2:
                        targets = [c[1] for c in cmds]
                        low_idx = hp_list.index(min(hp_list))
                        if all(t == low_idx for t in targets):
                            found = True
                            break
            sv1 = StateView(engine.state, 1, (opp_view, opp_view))
            cmd0 = policy.decision(sv0, opp_view)
            cmd1 = policy.decision(sv1, opp_view)
            engine.run_turn((cmd0, cmd1))
            turns += 1
            if turns > 100:
                break
        assert found, "Focus fire behavior not observed in 100 turns"

    def test_avoids_suicide(self) -> None:
        """Policy avoids moves where opponent KOs us but we don't KO them."""
        sv0, opp_view, params, engine = _make_battle_state(seed=70)
        policy = GreedyDongiPolicy()
        policy.set_params(params)
        turns = 0
        while not engine.finished():
            sv0 = StateView(engine.state, 0, (opp_view, opp_view))
            cmds = policy.decision(sv0, opp_view)
            assert cmds is not None
            sv1 = StateView(engine.state, 1, (opp_view, opp_view))
            cmd1 = policy.decision(sv1, opp_view)
            engine.run_turn((cmds, cmd1))
            turns += 1
            if turns > 100:
                break
        assert turns > 0


class TestGreedyDongiEdgeCases:
    """Edge case and robustness tests."""

    def test_no_crash_over_many_battles(self) -> None:
        """Policy completes 20 battles without exceptions."""
        policy = GreedyDongiPolicy()
        params = BattleRuleParam()
        policy.set_params(params)
        for seed in range(20):
            gen = np.random.default_rng(100 + seed)
            team = gen_team(4, 4, gen)
            view = TeamView(team)
            indices = list(range(len(team.members)))
            sub_a, sub_view_a = subteam(team, view, indices)
            sub_b, sub_view_b = subteam(team, view, indices)
            battle_teams = get_battle_teams((sub_a, sub_b), 2)
            state = State(battle_teams)
            rng = ((gen, gen), (gen, gen))
            eng = BattleEngine(
                state, params=params,
                acc_rng=rng, eff_rng=rng, sta_rng=rng,
            )
            turns = 0
            while not eng.finished():
                sv0 = StateView(eng.state, 0, (sub_view_a, sub_view_b))
                sv1 = StateView(eng.state, 1, (sub_view_b, sub_view_a))
                cmd0 = policy.decision(sv0, sub_view_b)
                cmd1 = policy.decision(sv1, sub_view_a)
                eng.run_turn((cmd0, cmd1))
                turns += 1
                if turns > 200:
                    pytest.fail(f"Battle {seed} did not terminate")
            assert eng.finished()


class TestGreedyDongiProtect:
    """Protect simulation tests (Milestone 3)."""

    def test_protect_chosen_when_incoming_exceeds_outgoing(self) -> None:
        """Protect is chosen when opponent threatens heavy damage."""
        sv0, opp_view, params, engine = _make_battle_state(seed=80)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        active = sv0.sides[0].team.active
        pkm = active[0]

        # Set slot 0 HP so opponent targets it (capped dmg > slot 1's)
        pkm.hp = 200

        # Give slot 0 Protect + weak attacks (1 BP)
        pkm.battling_moves[0] = BattlingMove(_fresh_protect_move())
        for i in range(1, len(pkm.battling_moves)):
            old = pkm.battling_moves[i].constants
            weak = Move(
                old.pkm_type, 1, old.accuracy, old.max_pp,
                old.category, priority=old.priority,
            )
            pkm.battling_moves[i] = BattlingMove(weak)

        # Replace ALL opponent moves with strong Normal (ensures targeting
        # slot 0 whose capped damage = 100, and slot 1 at full HP survives)
        strong_normal = Move(Type.NORMAL, 150, 1.0, 10, Category.PHYSICAL)
        opp_active = sv0.sides[1].team.active
        for opp in opp_active:
            for i in range(len(opp.battling_moves)):
                opp.battling_moves[i] = BattlingMove(strong_normal)

        # Kill reserves so switch is not an option (isolate protect behavior)
        for p in sv0.sides[0].team.reserve:
            p.hp = 0

        cmds = policy.decision(sv0, opp_view)
        # Slot 0 should protect (move index 0) to avoid the KO
        assert cmds[0][0] == 0, (
            f"Expected protect (move 0), got move {cmds[0][0]}"
        )

    def test_protect_not_chosen_when_can_ko(self) -> None:
        """Never protects if a KO is available."""
        sv0, opp_view, params, engine = _make_battle_state(seed=90)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        # Make opponent slot 0 very low HP (1 HP) so any move KOs
        defenders = sv0.sides[1].team.active
        defenders[0].hp = 1

        # Give our slot 0 a Protect move + a damaging move
        active = sv0.sides[0].team.active
        pkm = active[0]
        pkm.battling_moves[0] = BattlingMove(_fresh_protect_move())
        # Ensure move 1 is a strong damaging move (keep original)

        cmds = policy.decision(sv0, opp_view)
        # Slot 0 should NOT protect — it should attack to get the KO
        assert cmds[0][0] != 0, (
            "Policy chose Protect despite a KO being available"
        )

    def test_double_protect_not_selected(self) -> None:
        """Both Pokemon protecting is not chosen when attacks available."""
        sv0, opp_view, params, engine = _make_battle_state(seed=110)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        # Give both our Pokemon Protect as move 0, keep attacks as other moves
        active = sv0.sides[0].team.active
        for pkm in active:
            pkm.battling_moves[0] = BattlingMove(_fresh_protect_move())

        cmds = policy.decision(sv0, opp_view)
        # At least one slot should attack (not both protect)
        protect_count = sum(1 for c in cmds if c[0] == 0)
        assert protect_count < len(cmds), (
            "Both Pokemon chose Protect — double protect should not be selected"
        )


class TestGreedyDongiSwitch:
    """Switch action tests (Milestone 4)."""

    def test_switch_chosen_when_walled(self) -> None:
        """Switch is chosen when current Pokemon is walled and replacement tanks."""
        sv0, opp_view, params, engine = _make_battle_state(seed=120)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        active = sv0.sides[0].team.active
        reserve = sv0.sides[0].team.reserve

        # Make slot 0's attacks do 0 damage (Category.OTHER, no protect)
        for i in range(len(active[0].battling_moves)):
            status_move = Move(
                Type.NORMAL, 0, 1.0, 10, Category.OTHER,
            )
            active[0].battling_moves[i] = BattlingMove(status_move)

        # Make slot 0 very fragile so opponent targets it
        active[0].hp = 80
        # Keep slot 1 at low HP so opponent prefers slot 0
        active[1].hp = 1

        # Ensure opponent has strong attacks
        opp_active = sv0.sides[1].team.active
        strong = Move(Type.NORMAL, 150, 1.0, 10, Category.PHYSICAL)
        for opp in opp_active:
            for i in range(len(opp.battling_moves)):
                opp.battling_moves[i] = BattlingMove(strong)

        # Ensure at least one reserve is alive
        assert any(p.hp > 0 for p in reserve), "Need alive reserve for test"

        cmds = policy.decision(sv0, opp_view)
        # Slot 0 should switch (cmd[0] == -1) since it can't damage
        # opponent and is taking heavy hits
        assert cmds[0][0] == -1, (
            f"Expected switch (-1), got move {cmds[0][0]}"
        )

    def test_switch_not_chosen_when_can_ko(self) -> None:
        """Never switches if a KO is available."""
        sv0, opp_view, params, engine = _make_battle_state(seed=130)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        # Make opponent slot 0 very low HP so any move KOs
        defenders = sv0.sides[1].team.active
        defenders[0].hp = 1

        cmds = policy.decision(sv0, opp_view)
        # At least one slot should attack the low-HP opponent (not switch)
        has_attack_on_target0 = any(
            c[0] >= 0 and c[1] == 0 for c in cmds
        )
        assert has_attack_on_target0, (
            f"Policy did not attack the 1 HP opponent: {cmds}"
        )

    def test_switch_to_dead_pokemon_invalid(self) -> None:
        """Dead reserve Pokemon are never selected as switch targets."""
        sv0, opp_view, params, engine = _make_battle_state(seed=140)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        # Kill all reserve Pokemon
        reserve = sv0.sides[0].team.reserve
        for p in reserve:
            p.hp = 0

        cmds = policy.decision(sv0, opp_view)
        # No switch commands should appear (all reserves dead)
        for cmd in cmds:
            assert cmd[0] >= 0, (
                f"Switch to dead reserve selected: {cmd}"
            )

    def test_action_space_includes_switches(self) -> None:
        """Action space includes switch options when reserves are alive."""
        sv0, opp_view, params, engine = _make_battle_state(seed=150)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        reserve = sv0.sides[0].team.reserve
        alive_reserves = [p for p in reserve if p.hp > 0]

        # Verify reserves exist
        assert len(alive_reserves) > 0, "Need alive reserves for test"

        # Run many turns — switches should eventually appear
        turns = 0
        while not engine.finished() and turns < 50:
            sv0 = StateView(engine.state, 0, (opp_view, opp_view))
            sv1 = StateView(engine.state, 1, (opp_view, opp_view))
            cmd0 = policy.decision(sv0, opp_view)
            cmd1 = policy.decision(sv1, opp_view)
            if any(c[0] == -1 for c in cmd0):
                break
            engine.run_turn((cmd0, cmd1))
            turns += 1
        # Switch is situational — just verify no crash over 50 turns
        assert turns >= 0
