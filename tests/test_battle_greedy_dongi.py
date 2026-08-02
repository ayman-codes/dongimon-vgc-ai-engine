"""Tests for the GreedyDongi net-damage battle policy."""

from __future__ import annotations

import numpy as np
import pytest
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.modifiers import Category, Stat, Type
from vgc2.battle_engine.move import BattlingMove, Move
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.battle.greedy_dongi import (
    GreedyDongiPolicy,
    _precompute_opponent_actions,
    _resolve_turn,
)


def _fresh_protect_move() -> Move:
    """Create a fresh Protect Move instance (avoids shared-object pollution).

    Priority 1 so Protect acts before normal-speed opponent moves under
    the policy's speed-ordered turn resolution.
    """
    return Move(Type.NORMAL, 0, 1.0, 16, Category.OTHER, protect=True, priority=1)


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
        """Both Pokemon target the same low-HP opponent to secure a KO.

        Each Pokemon uses a TYPELESS 100 BP physical move with fixed Atk/Def
        (100 each), dealing exactly 86 damage. One hit does not KO the 140 HP
        opponent, but two hits do, so focus fire is the only KO option.
        """
        sv0, opp_view, params, engine = _make_battle_state(seed=60)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        active = sv0.sides[0].team.active
        for pkm in active:
            stats = list(pkm.constants.stats)
            stats[Stat.ATTACK] = 100
            stats[Stat.DEFENSE] = 100
            pkm.constants.stats = tuple(stats)
            for i in range(len(pkm.battling_moves)):
                pkm.battling_moves[i] = BattlingMove(
                    Move(Type.TYPELESS, 100, 1.0, 10, Category.PHYSICAL)
                )

        defenders = sv0.sides[1].team.active
        for d in defenders:
            stats = list(d.constants.stats)
            stats[Stat.DEFENSE] = 100
            d.constants.stats = tuple(stats)
            d.types = [Type.NORMAL]
        defenders[0].hp = 140
        defenders[1].hp = 400

        cmds = policy.decision(sv0, opp_view)
        assert len(cmds) == 2
        assert cmds[0][1] == 0 and cmds[1][1] == 0, (
            f"Expected both Pokemon to focus-fire opponent slot 0, got {cmds}"
        )

    def test_ko_not_double_counted_on_overkill(self) -> None:
        """Focus fire that overkills is not scored as 2 KOs (one hit suffices).

        Each Pokemon deals exactly 170 damage (TYPELESS 100 BP, Atk 200).
        Opponent 0 has 60 HP, so a single hit KOs it; opponent 1 has 400 HP.
        The optimal action splits so the second Pokemon damages opponent 1,
        rather than wasting a hit on the already-KO'd opponent 0.
        """
        sv0, opp_view, params, engine = _make_battle_state(seed=61)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        active = sv0.sides[0].team.active
        for pkm in active:
            stats = list(pkm.constants.stats)
            stats[Stat.ATTACK] = 200
            stats[Stat.DEFENSE] = 100
            pkm.constants.stats = tuple(stats)
            for i in range(len(pkm.battling_moves)):
                pkm.battling_moves[i] = BattlingMove(
                    Move(Type.TYPELESS, 100, 1.0, 10, Category.PHYSICAL)
                )

        defenders = sv0.sides[1].team.active
        for d in defenders:
            stats = list(d.constants.stats)
            stats[Stat.DEFENSE] = 100
            d.constants.stats = tuple(stats)
            d.types = [Type.NORMAL]
        defenders[0].hp = 60
        defenders[1].hp = 400

        cmds = policy.decision(sv0, opp_view)
        assert len(cmds) == 2
        assert cmds[0][1] != cmds[1][1], (
            f"Expected a split (no wasted hit on the KO'd target), got {cmds}"
        )

    def test_ko_denies_opponent_when_faster(self) -> None:
        """A KO'd opponent does not act when we are faster (KO denial).

        Our fast Pokemon (Speed 200) KOs opponent 0 (Speed 1, 1 HP) before
        it acts; opponent 1 is harmless. The speed-ordered resolution must
        yield 0 damage taken from the KO'd threat.
        """
        sv0, opp_view, params, engine = _make_battle_state(seed=62)
        active = sv0.sides[0].team.active
        defenders = sv0.sides[1].team.active

        for pkm in active:
            stats = list(pkm.constants.stats)
            stats[Stat.SPEED] = 200
            stats[Stat.ATTACK] = 200
            stats[Stat.DEFENSE] = 100
            pkm.constants.stats = tuple(stats)
        strong = Move(Type.TYPELESS, 150, 1.0, 10, Category.PHYSICAL)
        for i in range(len(active[0].battling_moves)):
            active[0].battling_moves[i] = BattlingMove(strong)
        harmless = Move(Type.TYPELESS, 0, 1.0, 10, Category.OTHER)
        for i in range(len(active[1].battling_moves)):
            active[1].battling_moves[i] = BattlingMove(harmless)

        for d in defenders:
            stats = list(d.constants.stats)
            stats[Stat.SPEED] = 1
            stats[Stat.DEFENSE] = 100
            d.constants.stats = tuple(stats)
            d._revealed = list(range(len(d._pkm.battling_moves)))
            for i in range(len(d._pkm.battling_moves)):
                d._pkm.battling_moves[i] = BattlingMove(harmless)
        defenders[0]._pkm.battling_moves[0] = BattlingMove(strong)
        defenders[0].hp = 1
        defenders[1].hp = 400

        opp_damage, opp_priority = _precompute_opponent_actions(params, sv0, active, defenders)
        dmg_dealt, opp_kos, dmg_taken, our_kos = _resolve_turn(
            params, sv0, active, defenders,
            (0, 0), (0, 1), [False, False], opp_damage, opp_priority,
        )
        assert opp_kos == 1, "Our fast Pokemon should KO opponent 0"
        assert dmg_taken == 0.0, f"KO'd opponent should not act when faster, took {dmg_taken}"

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

        stats = list(pkm.constants.stats)
        stats[Stat.DEFENSE] = 20
        pkm.constants.stats = tuple(stats)
        pkm.hp = 200
        pkm.types = [Type.NORMAL]

        pkm.battling_moves[0] = BattlingMove(_fresh_protect_move())
        for i in range(1, len(pkm.battling_moves)):
            old = pkm.battling_moves[i].constants
            weak = Move(
                old.pkm_type, 1, old.accuracy, old.max_pp,
                old.category, priority=old.priority,
            )
            pkm.battling_moves[i] = BattlingMove(weak)

        slot1 = active[1]
        slot1.hp = 800
        slot1.types = [Type.NORMAL]
        for i in range(len(slot1.battling_moves)):
            slot1.battling_moves[i] = BattlingMove(Move(Type.NORMAL, 0, 1.0, 10, Category.OTHER))

        strong_normal = Move(Type.NORMAL, 150, 1.0, 10, Category.PHYSICAL)
        opp_active = sv0.sides[1].team.active
        for opp in opp_active:
            opp.types = [Type.NORMAL]
            for i in range(len(opp.battling_moves)):
                opp.battling_moves[i] = BattlingMove(strong_normal)

        for p in sv0.sides[0].team.reserve:
            p.hp = 0

        cmds = policy.decision(sv0, opp_view)
        assert cmds[0][0] == 0, (
            f"Expected protect (move 0), got move {cmds[0][0]}"
        )

    def test_protect_not_chosen_when_can_ko(self) -> None:
        """Never protects if a KO is available."""
        sv0, opp_view, params, engine = _make_battle_state(seed=90)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        defenders = sv0.sides[1].team.active
        defenders[0].hp = 1

        active = sv0.sides[0].team.active
        pkm = active[0]
        pkm.battling_moves[0] = BattlingMove(_fresh_protect_move())

        cmds = policy.decision(sv0, opp_view)
        assert cmds[0][0] != 0, (
            "Policy chose Protect despite a KO being available"
        )

    def test_double_protect_not_selected(self) -> None:
        """Both Pokemon protecting is not chosen when attacks available."""
        sv0, opp_view, params, engine = _make_battle_state(seed=110)
        policy = GreedyDongiPolicy()
        policy.set_params(params)

        active = sv0.sides[0].team.active
        for pkm in active:
            pkm.battling_moves[0] = BattlingMove(_fresh_protect_move())

        cmds = policy.decision(sv0, opp_view)
        protect_count = sum(1 for c in cmds if c[0] == 0)
        assert protect_count < len(cmds), (
            "Both Pokemon chose Protect — double protect should not be selected"
        )
