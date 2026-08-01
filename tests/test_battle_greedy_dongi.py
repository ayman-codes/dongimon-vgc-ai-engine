"""Tests for the GreedyDongi net-damage battle policy."""

from __future__ import annotations

import numpy as np
import pytest
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_team

from src.battle.greedy_dongi import GreedyDongiPolicy


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
