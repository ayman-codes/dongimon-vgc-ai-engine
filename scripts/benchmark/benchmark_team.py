"""All-vs-all benchmark — teambuild + selection quality (battle policy selectable).

By default both sides use GreedyBattlePolicy so win-rate differences reflect
teambuild + selection only. Pass ``--battle-policy dongimon`` to pilot both
sides with DongimonBattlePolicy (weights from battle_weights.yaml).

Design (statistically valid):
  - Each round generates a shared species roster (fixed seed).
  - Each competitor builds a team from the SAME roster via their own TeamBuildPolicy.
  - Each competitor selects once per matchup via their own SelectionPolicy.
  - Battles are fully seeded (acc_rng, eff_rng, sta_rng).
  - Primary metric: pairwise win rate with bootstrap 95% CI.
  - No ELO path-dependence; win rate is bounded [0, 1].

Usage:
    uv run python scripts/benchmark/benchmark_team.py --seed=42 --n-rounds=10 --n-battles=30
    uv run python scripts/benchmark/benchmark_team.py --battle-policy greedy --save-teams
    uv run python scripts/benchmark/benchmark_team.py --battle-policy dongimon --save-teams
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from numpy.typing import NDArray
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.constants import NATURES
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.modifiers import Category, Hazard, Nature, Stat, Status, Terrain, Weather
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.ecosystem import build_team, label_roster, sanitized_team_build_decision
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from competitor import DongimonCompetitor
from PPO_trainers.weighted_heuristic.policy import DongimonBattlePolicy
from src.config.loader import load_battle_weights

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
ROSTER_SIZE = 50
MOVESET_SIZE = 200
MAX_TEAM_SIZE = 4
MAX_MOVES = 4
N_ACTIVE = 2

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_BOOST_KEYS = ("hp", "atk", "def", "spa", "spd", "spe", "eva", "acc")


def _import_competitor_cls(module_path: str, class_name: str) -> Any:
    """Import a competitor class by module path and class name.

    Args:
        module_path: Dotted module path.
        class_name: Class attribute name on the module.

    Returns:
        The competitor class object.
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class _GreedyBaseline:
    """Baseline: random team from roster + BasicSelection + Greedy battle."""

    name = "Greedy"

    def __init__(self) -> None:
        self.selectionpolicy = BasicSelectionPolicy()
        self.battlepolicy = GreedyBattlePolicy()

    def build_team(self, roster: list[Any], rng: np.random.Generator, build_size: int = MAX_TEAM_SIZE) -> Team:
        """Pick random species and hydrate with random spreads.

        Args:
            roster: Shared species roster for the round.
            rng: Seeded RNG for reproducible builds.
            build_size: Number of team members to build.

        Returns:
            A randomly constructed Team of size up to build_size.
        """
        indices = rng.choice(len(roster), size=min(build_size, len(roster)), replace=False)
        members = []
        for idx in indices:
            species = roster[int(idx)]
            n_moves = len(species.moves)
            move_idx = list(rng.choice(n_moves, size=min(MAX_MOVES, n_moves), replace=False))
            evs = tuple(int(x) for x in rng.multinomial(510, [1 / 6] * 6))
            nature = Nature(int(rng.integers(0, len(Nature))))
            p = Pokemon(
                species=species,
                move_indexes=move_idx,
                level=100,
                ivs=(31,) * 6,
                evs=evs,
                nature=nature,
            )
            members.append(p)
        return Team(members)


_PLAYER_ROSTER: list[tuple[str, Any]] = [
    ("Dongimon", None),
    ("JJJ", _import_competitor_cls("competitors.competitor1_jjj", "JJJ_Competitor")),
    ("minimon", _import_competitor_cls("competitors.competitor2_minimon", "minimon")),
    ("caaaden", _import_competitor_cls("competitors.competitor_caaaden", "CaaadenCompetitor")),
    ("Greedy", None),
]


def _enum_name(value: Any) -> str:
    """Convert an enum or value to a stable string name.

    Args:
        value: Enum member or arbitrary value.

    Returns:
        Enum ``name`` if present, otherwise ``str(value)``.
    """
    return str(getattr(value, "name", value))


def _stats_dict(stats: tuple[int, ...] | list[int]) -> dict[str, int]:
    """Map a 6-stat tuple to a named dictionary.

    Args:
        stats: Stats in HP/ATK/DEF/SPA/SPD/SPE order.

    Returns:
        Named stat mapping with integer values.
    """
    return {k: int(stats[i]) for i, k in enumerate(_STAT_KEYS) if i < len(stats)}


def _boosts_dict(boosts: tuple[int, ...] | list[int]) -> dict[str, int]:
    """Map an 8-stat boost tuple to a named dictionary.

    Args:
        boosts: Boost stages including EVA and ACC slots.

    Returns:
        Named boost mapping with integer values.
    """
    return {k: int(boosts[i]) for i, k in enumerate(_BOOST_KEYS) if i < len(boosts)}


def _nature_modifiers(nature: Nature) -> tuple[str | None, str | None]:
    """Resolve nature plus/minus stat names from vgc2 NATURES.

    Args:
        nature: Nature enum value.

    Returns:
        Tuple of (plus_stat_name, minus_stat_name), either may be None.
    """
    entry = NATURES.get(nature)
    if entry is None:
        return None, None
    plus_idx = int(entry["plus"])
    minus_idx = int(entry["minus"])
    plus_name = _STAT_KEYS[plus_idx] if 0 <= plus_idx < len(_STAT_KEYS) else None
    minus_name = _STAT_KEYS[minus_idx] if 0 <= minus_idx < len(_STAT_KEYS) else None
    return plus_name, minus_name


def _move_to_dict(move: Move) -> dict[str, Any]:
    """Serialize a vgc2 Move with all combat-relevant fields.

    Args:
        move: Move instance from a Pokemon moveset.

    Returns:
        JSON-friendly dict of move identity, combat stats, and utility flags.
    """
    weather = _enum_name(move.weather_start)
    terrain = _enum_name(move.field_start)
    hazard = _enum_name(move.hazard)
    status = _enum_name(move.status)
    return {
        "id": int(move.id),
        "name": str(move.name) if move.name else "",
        "type": _enum_name(move.pkm_type),
        "category": _enum_name(move.category),
        "base_power": int(move.base_power),
        "accuracy": float(move.accuracy),
        "max_pp": int(move.max_pp),
        "priority": int(move.priority),
        "effect_prob": float(move.effect_prob),
        "force_switch": bool(move.force_switch),
        "self_switch": bool(move.self_switch),
        "ignore_evasion": bool(move.ignore_evasion),
        "protect": bool(move.protect),
        "boosts": _boosts_dict(move.boosts),
        "boosts_raw": [int(b) for b in move.boosts],
        "self_boosts": bool(move.self_boosts),
        "heal": float(move.heal),
        "recoil": float(move.recoil),
        "weather": weather if move.weather_start != Weather.CLEAR else None,
        "terrain": terrain if move.field_start != Terrain.NONE else None,
        "trickroom": bool(move.toggle_trickroom),
        "change_type": bool(move.change_type),
        "reflect": bool(move.toggle_reflect),
        "lightscreen": bool(move.toggle_lightscreen),
        "tailwind": bool(move.toggle_tailwind),
        "hazard": hazard if move.hazard != Hazard.NONE else None,
        "status": status if move.status != Status.NONE else None,
        "disable": bool(move.disable),
        "summary": str(move),
    }


def _pokemon_to_dict(pkm: Pokemon, index: int) -> dict[str, Any]:
    """Serialize a built Pokemon with stats, EVs, nature, and full moveset.

    Args:
        pkm: Built Pokemon instance.
        index: Position index within its team.

    Returns:
        JSON-friendly dict with base/final stats and derived review labels.
    """
    base = pkm.species.base_stats
    final = pkm.stats
    moves = list(pkm.moves)
    move_dicts = [_move_to_dict(m) for m in moves]
    n_physical = sum(1 for m in moves if m.category == Category.PHYSICAL)
    n_special = sum(1 for m in moves if m.category == Category.SPECIAL)
    n_other = sum(1 for m in moves if m.category == Category.OTHER)
    atk = int(final[Stat.ATTACK])
    spa = int(final[Stat.SPECIAL_ATTACK])
    if atk > spa * 1.1:
        role_hint = "physical"
    elif spa > atk * 1.1:
        role_hint = "special"
    else:
        role_hint = "mixed"
    nature_plus, nature_minus = _nature_modifiers(pkm.nature)
    move_indexes = list(getattr(pkm, "_move_indexes", []))
    return {
        "index": index,
        "species_id": int(pkm.species.id),
        "species_name": str(pkm.species.name) if pkm.species.name else "",
        "types": [_enum_name(t) for t in pkm.species.types],
        "base_stats": _stats_dict(base),
        "bst": int(sum(base)),
        "final_stats": _stats_dict(final),
        "evs": _stats_dict(pkm.evs),
        "evs_raw": [int(x) for x in pkm.evs],
        "ivs": _stats_dict(pkm.ivs),
        "ivs_raw": [int(x) for x in pkm.ivs],
        "nature": _enum_name(pkm.nature),
        "nature_plus": nature_plus,
        "nature_minus": nature_minus,
        "level": int(pkm.level),
        "move_indexes": [int(i) for i in move_indexes],
        "moves": move_dicts,
        "role_hint": role_hint,
        "max_bp": max((int(m.base_power) for m in moves), default=0),
        "n_physical": n_physical,
        "n_special": n_special,
        "n_other": n_other,
        "has_protect": any(m.protect for m in moves),
        "has_priority": any(int(m.priority) > 0 for m in moves),
        "has_setup": any(any(int(b) != 0 for b in m.boosts) for m in moves),
        "has_status_move": any(m.status != Status.NONE for m in moves),
        "summary": str(pkm),
    }


def _team_to_dict(team: Team) -> dict[str, Any]:
    """Serialize a full Team including aggregate labels.

    Args:
        team: Built team.

    Returns:
        Dict with size, members, average BST, and type set.
    """
    members = [_pokemon_to_dict(pkm, i) for i, pkm in enumerate(team.members)]
    bsts = [int(m["bst"]) for m in members]
    type_set: list[str] = []
    for m in members:
        for t in m["types"]:
            if t not in type_set:
                type_set.append(str(t))
    return {
        "size": len(members),
        "members": members,
        "team_bst_avg": round(float(np.mean(bsts)), 2) if bsts else 0.0,
        "type_coverage": type_set,
    }


def _roster_species_to_dict(species: PokemonSpecies) -> dict[str, Any]:
    """Serialize a roster species for round context.

    Args:
        species: Species entry from the shared roster.

    Returns:
        Compact species summary with base stats and move labels.
    """
    base = species.base_stats
    return {
        "id": int(species.id),
        "name": str(species.name) if species.name else "",
        "types": [_enum_name(t) for t in species.types],
        "base_stats": _stats_dict(base),
        "bst": int(sum(base)),
        "n_moves": len(species.moves),
        "moves": [str(m) for m in species.moves],
    }


def _subteam_from_indices(team: Team, indices: list[int]) -> dict[str, Any]:
    """Serialize the selected subteam members in selection order.

    Args:
        team: Full built team.
        indices: Selected member indices from the selection policy.

    Returns:
        Team-shaped dict over the selected members only.
    """
    selected = Team([team.members[i] for i in indices if 0 <= i < len(team.members)])
    return _team_to_dict(selected)


def _print_team_card(name: str, team: Team, build_time: float) -> None:
    """Print a human-readable team card to the terminal.

    Args:
        name: Competitor name.
        team: Built team to display.
        build_time: Teambuild wall time in seconds.
    """
    print(f"  [{name}] build {build_time:.2f}s")
    for i, pkm in enumerate(team.members):
        types = "/".join(_enum_name(t) for t in pkm.species.types)
        bst = int(sum(pkm.species.base_stats))
        nature = _enum_name(pkm.nature)
        plus, minus = _nature_modifiers(pkm.nature)
        nature_mod = ""
        if plus and minus:
            nature_mod = f" (+{plus.upper()} -{minus.upper()})"
        evs = "/".join(str(int(x)) for x in pkm.evs)
        fs = pkm.stats
        print(
            f"    [{i}] {types}  BST={bst}  Nature={nature}{nature_mod}  "
            f"EVs={evs}"
        )
        print(
            f"        HP{int(fs[0])} Atk{int(fs[1])} Def{int(fs[2])} "
            f"SpA{int(fs[3])} SpD{int(fs[4])} Spe{int(fs[5])}"
        )
        for mi, move in enumerate(pkm.moves):
            print(
                f"        M{mi} {_enum_name(move.category)} {_enum_name(move.pkm_type)} "
                f"BP={int(move.base_power)} ACC={float(move.accuracy):.2f} "
                f"PP={int(move.max_pp)}"
                f"{' PRI=' + str(int(move.priority)) if int(move.priority) != 0 else ''}"
                f"{' Protect' if move.protect else ''}"
                f"{' ' + _enum_name(move.status) if move.status != Status.NONE else ''}"
                f" | {move}"
            )


def _build_team_for(
    name: str,
    competitor: Any,
    roster: list[Any],
    rng: np.random.Generator,
    build_size: int = MAX_TEAM_SIZE,
) -> Team | None:
    """Build a team for a competitor.

    Args:
        name: Competitor display name.
        competitor: Competitor instance exposing teambuildpolicy when applicable.
        roster: Shared species roster.
        rng: Seeded RNG (used by the Greedy baseline).
        build_size: Number of team members to build.

    Returns:
        Built Team, or None if teambuild failed.
    """
    if name == "Greedy":
        baseline = _GreedyBaseline()
        return baseline.build_team(roster, rng, build_size)

    try:
        commands = sanitized_team_build_decision(
            competitor.teambuildpolicy, roster, None, build_size, MAX_MOVES, N_ACTIVE
        )
        if not commands:
            return None
        return build_team(commands, roster)
    except Exception as exc:
        print(f"    [WARN] {name} teambuild failed: {type(exc).__name__}: {exc}")
        return None


def _safe_selection(sel: Any, team: Team, opp_view: TeamView, max_size: int) -> list[int]:
    """Run selection once; failures propagate (no silent fallback).

    Args:
        sel: Selection policy instance.
        team: Own full team.
        opp_view: Opponent team view.
        max_size: Maximum roster size to select.

    Returns:
        List of selected member indices.

    Raises:
        RuntimeError: If selection returns an empty index list.
    """
    idx = list(sel.decision((team, opp_view), max_size))
    if not idx:
        raise RuntimeError(
            f"selection returned empty indices for team size {len(team.members)}"
        )
    return [int(i) for i in idx]


def _try_selection(sel: Any, team: Team, opp_view: TeamView, max_size: int, name: str) -> list[int]:
    """Run selection with silent fallback for non-Dongimon competitors.

    Args:
        sel: Selection policy instance.
        team: Own full team.
        opp_view: Opponent team view.
        max_size: Maximum roster size to select.
        name: Competitor display name for warning messages.

    Returns:
        List of selected member indices. Falls back to [0,1,2,3] on failure.
    """
    try:
        return _safe_selection(sel, team, opp_view, max_size)
    except Exception as exc:
        print(f"    [WARN] {name} selection failed: {type(exc).__name__}: {exc}  → fallback [0,1,2,3]")
        return list(range(min(max_size, len(team.members))))


def _make_battle_policy(battle_policy_name: str) -> Any:
    """Construct the shared battle policy for both sides.

    Args:
        battle_policy_name: ``greedy`` or ``dongimon`` (case-insensitive).

    Returns:
        A BattlePolicy instance used by both sides.

    Raises:
        ValueError: If ``battle_policy_name`` is not recognized.
    """
    key = battle_policy_name.strip().lower()
    if key == "greedy":
        return GreedyBattlePolicy()
    if key == "dongimon":
        weights = load_battle_weights().model_dump()
        return DongimonBattlePolicy(custom_weights=weights)
    raise ValueError(f"Unknown battle policy: {battle_policy_name!r} (use greedy|dongimon)")


def _run_seeded_match(
    team_a: Team,
    team_b: Team,
    sel_a: Any,
    sel_b: Any,
    params: BattleRuleParam,
    n_battles: int,
    match_seed: int,
    battle_policy: Any,
    name_a: str,
    name_b: str,
) -> tuple[int, int, int, list[int], list[int], float, float]:
    """Run N seeded battles; selection is evaluated once per side.

    Both sides use the same ``battle_policy`` instance so battle skill is
    neutralized across competitors (teambuild+selection isolation).
    Dongimon selection fails loud; opponent selection fails silently with
    fallback to [0,1,2,3].

    Args:
        team_a: Full team for side A.
        team_b: Full team for side B.
        sel_a: Selection policy for side A.
        sel_b: Selection policy for side B.
        params: Battle rule parameters.
        n_battles: Number of battles to run.
        match_seed: Base seed for this matchup.
        battle_policy: Shared battle policy for both sides.
        name_a: Display name for side A (fails loud if Dongimon).
        name_b: Display name for side B (fails loud if Dongimon).

    Returns:
        Tuple of (wins_a, wins_b, draws, idx_a, idx_b, sel_time_a, sel_time_b).
    """
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)

    t_sel = time.perf_counter()
    if name_a == "Dongimon":
        idx_a = _safe_selection(sel_a, team_a, view_b, MAX_TEAM_SIZE)
    else:
        idx_a = _try_selection(sel_a, team_a, view_b, MAX_TEAM_SIZE, name_a)
    sel_time_a = time.perf_counter() - t_sel

    t_sel = time.perf_counter()
    if name_b == "Dongimon":
        idx_b = _safe_selection(sel_b, team_b, view_a, MAX_TEAM_SIZE)
    else:
        idx_b = _try_selection(sel_b, team_b, view_a, MAX_TEAM_SIZE, name_b)
    sel_time_b = time.perf_counter() - t_sel

    wins_a = 0
    wins_b = 0
    draws = 0

    for b_idx in range(n_battles):
        battle_seed = match_seed + b_idx
        gen = np.random.default_rng(battle_seed)

        a_is_side0 = (b_idx % 2 == 0)
        if a_is_side0:
            sub_0, sub_view_0 = subteam(team_a, view_a, idx_a)
            sub_1, sub_view_1 = subteam(team_b, view_b, idx_b)
        else:
            sub_0, sub_view_0 = subteam(team_b, view_b, idx_b)
            sub_1, sub_view_1 = subteam(team_a, view_a, idx_a)

        battle_teams = get_battle_teams((sub_0, sub_1), N_ACTIVE)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(
            state,
            params=params,
            acc_rng=rng_tuple,
            eff_rng=rng_tuple,
            sta_rng=rng_tuple,
        )

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_0, sub_view_1))
            sv1 = StateView(engine.state, 1, (sub_view_1, sub_view_0))
            cmd0 = battle_policy.decision(sv0, sub_view_1)
            cmd1 = battle_policy.decision(sv1, sub_view_0)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            if a_is_side0:
                wins_a += 1
            else:
                wins_b += 1
        elif engine.winning_side == 1:
            if a_is_side0:
                wins_b += 1
            else:
                wins_a += 1
        else:
            draws += 1

    return wins_a, wins_b, draws, idx_a, idx_b, sel_time_a, sel_time_b


def _bootstrap_ci(
    wins: NDArray[np.int_],
    losses: NDArray[np.int_],
    n_boot: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
) -> tuple[float, float, float]:
    """Compute win rate and bootstrap confidence interval.

    Args:
        wins: Array of per-round win counts.
        losses: Array of per-round loss counts.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.

    Returns:
        Tuple of (point_estimate, ci_low, ci_high).
    """
    total_w = int(wins.sum())
    total_l = int(losses.sum())
    total = total_w + total_l
    if total == 0:
        return 0.5, 0.0, 1.0

    point = total_w / total
    rng = np.random.default_rng(0)

    n_rounds = len(wins)
    boot_rates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_rounds, size=n_rounds)
        sw = int(wins[idx].sum())
        sl = int(losses[idx].sum())
        st = sw + sl
        boot_rates[b] = sw / st if st > 0 else 0.5

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_rates, 100 * alpha))
    ci_high = float(np.percentile(boot_rates, 100 * (1 - alpha)))
    return point, ci_low, ci_high


def main() -> None:
    """Run the teambuild+selection benchmark and optionally dump team compositions."""
    parser = argparse.ArgumentParser(
        description=(
            "All-vs-all teambuild+selection benchmark. "
            "Battle policy is shared by both sides (greedy or dongimon)."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--n-rounds", type=int, default=10, help="Number of roster rounds")
    parser.add_argument("--n-battles", type=int, default=100, help="Battles per pair per round")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument(
        "--battle-policy",
        type=str,
        default="greedy",
        choices=["greedy", "dongimon", "Greedy", "Dongimon"],
        help="Shared battle policy for both sides (greedy|dongimon)",
    )
    parser.add_argument(
        "--build-size",
        type=int,
        default=MAX_TEAM_SIZE,
        help="Number of team members to build (default: 4). Use 6 to exercise full selection.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="mp_only",
        choices=["mp_only", "mp_sim"],
        help="Dongimon selection mode (mp_only|mp_sim)",
    )
    parser.add_argument(
        "--save-teams",
        action="store_true",
        default=False,
        help="Save full team+selection dumps under data/team_composition/",
    )
    parser.add_argument(
        "--n-top",
        type=int,
        default=None,
        help="Override N_TOP_CANDIDATES for Dongimon selection (default: policy default=5)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="",
        help="Comma-separated opponent filter (e.g. 'JJJ,caaaden'). Default: all.",
    )
    args = parser.parse_args()
    battle_policy_name = args.battle_policy.strip().lower()

    active_roster = _PLAYER_ROSTER
    if args.opponents:
        keep = {s.strip() for s in args.opponents.split(",")}
        keep.add("Dongimon")
        active_roster = [(n, c) for n, c in _PLAYER_ROSTER if n in keep]

    player_names = [p[0] for p in active_roster]
    n_players = len(player_names)
    params = BattleRuleParam()
    shared_battle_policy = _make_battle_policy(battle_policy_name)

    competitors: dict[str, Any] = {}
    weights_dict = load_battle_weights().model_dump()
    for name, cls in active_roster:
        if name == "Dongimon":
            competitors[name] = DongimonCompetitor(
                custom_weights=weights_dict,
                selection_mode=args.selection_mode,
                n_top_candidates=args.n_top,
            )
        elif name == "Greedy":
            competitors[name] = _GreedyBaseline()
        else:
            competitors[name] = cls()

    sel_cache: dict[str, Any] = {}
    for name in player_names:
        if name == "Greedy":
            sel_cache[name] = BasicSelectionPolicy()
        else:
            sel_cache[name] = competitors[name].selectionpolicy

    pair_keys: list[tuple[str, str]] = []
    for i in range(n_players):
        for j in range(i + 1, n_players):
            pair_keys.append((player_names[i], player_names[j]))

    wins_by_pair: dict[tuple[str, str], list[int]] = {k: [] for k in pair_keys}
    losses_by_pair: dict[tuple[str, str], list[int]] = {k: [] for k in pair_keys}
    draws_by_pair: dict[tuple[str, str], list[int]] = {k: [] for k in pair_keys}

    total_matchups = args.n_rounds * len(pair_keys)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = root_dir / "data" / "benchmark_team"
    composition_dir = results_dir / "team_composition"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Dongimon Teambuild+Selection Benchmark (battle = {battle_policy_name})")
    print(f"  seed={args.seed}, n_rounds={args.n_rounds}, n_battles={args.n_battles}")
    print(f"  players: {', '.join(player_names)}")
    print(f"  roster: {ROSTER_SIZE} species, {MOVESET_SIZE} moves")
    print(f"  build_size: {args.build_size}, select_size: {MAX_TEAM_SIZE}")
    print(f"  selection_mode: {args.selection_mode}")
    print(f"  total matchups: {total_matchups}")
    print(f"  battles per pair (total): {args.n_rounds * args.n_battles}")
    print(f"  battle_policy: {battle_policy_name}")
    print(f"  save_teams: {args.save_teams}")
    print("=" * 64)

    t0 = time.perf_counter()
    composition_rounds: list[dict[str, Any]] = []

    for r_idx in range(args.n_rounds):
        round_seed = args.seed + r_idx * 10_000
        roster_rng = np.random.default_rng(round_seed)

        moveset = gen_move_set(MOVESET_SIZE, roster_rng)
        roster = list(gen_pkm_roster(ROSTER_SIZE, moveset, MAX_MOVES, roster_rng))
        label_roster(moveset, roster)

        teams: dict[str, Team | None] = {}
        build_times: dict[str, float] = {}
        for name in player_names:
            build_rng = np.random.default_rng(round_seed + (sum(ord(c) for c in name) % 10_000))
            t_build = time.perf_counter()
            teams[name] = _build_team_for(name, competitors[name], roster, build_rng, args.build_size)
            build_times[name] = time.perf_counter() - t_build

        if args.save_teams:
            for name in player_names:
                t = teams[name]
                if t is not None:
                    _print_team_card(name, t, build_times[name])
                else:
                    print(f"  [{name}] BUILD_FAILED ({build_times[name]:.2f}s)")

        sel_time_sums: dict[str, float] = dict.fromkeys(player_names, 0.0)
        matchup_records: list[dict[str, Any]] = []

        pair_idx = 0
        for i in range(n_players):
            for j in range(i + 1, n_players):
                p1, p2 = player_names[i], player_names[j]
                team_a, team_b = teams[p1], teams[p2]

                if team_a is None or team_b is None:
                    if team_a is not None:
                        wins_by_pair[(p1, p2)].append(args.n_battles)
                        losses_by_pair[(p1, p2)].append(0)
                    elif team_b is not None:
                        wins_by_pair[(p1, p2)].append(0)
                        losses_by_pair[(p1, p2)].append(args.n_battles)
                    else:
                        wins_by_pair[(p1, p2)].append(0)
                        losses_by_pair[(p1, p2)].append(0)
                    draws_by_pair[(p1, p2)].append(0)
                    if args.save_teams:
                        matchup_records.append(
                            {
                                "side_a": p1,
                                "side_b": p2,
                                "selection": None,
                                "battles": {
                                    "n": args.n_battles,
                                    "wins_a": wins_by_pair[(p1, p2)][-1],
                                    "wins_b": losses_by_pair[(p1, p2)][-1],
                                    "draws": 0,
                                    "win_rate_a": None,
                                    "build_failed": True,
                                },
                            }
                        )
                    pair_idx += 1
                    continue

                matchup_seed = round_seed + pair_idx * 1_000 + 1
                w1, w2, dr, idx_a, idx_b, st_a, st_b = _run_seeded_match(
                    team_a,
                    team_b,
                    sel_cache[p1],
                    sel_cache[p2],
                    params,
                    args.n_battles,
                    matchup_seed,
                    shared_battle_policy,
                    p1,
                    p2,
                )

                wins_by_pair[(p1, p2)].append(w1)
                losses_by_pair[(p1, p2)].append(w2)
                draws_by_pair[(p1, p2)].append(dr)
                sel_time_sums[p1] += st_a
                sel_time_sums[p2] += st_b

                decisive = w1 + w2
                wr_a = (w1 / decisive) if decisive > 0 else None
                if args.save_teams:
                    matchup_records.append(
                        {
                            "side_a": p1,
                            "side_b": p2,
                            "selection": {
                                "indices_a": idx_a,
                                "indices_b": idx_b,
                                "time_a_sec": round(st_a, 4),
                                "time_b_sec": round(st_b, 4),
                                "subteam_a": _subteam_from_indices(team_a, idx_a),
                                "subteam_b": _subteam_from_indices(team_b, idx_b),
                            },
                            "battles": {
                                "n": args.n_battles,
                                "wins_a": w1,
                                "wins_b": w2,
                                "draws": dr,
                                "win_rate_a": round(wr_a, 4) if wr_a is not None else None,
                            },
                        }
                    )
                    print(
                        f"    {p1} vs {p2}: {w1}-{w2}-{dr}  "
                        f"sel={idx_a}vs{idx_b}  "
                        f"sel_t={st_a:.2f}s/{st_b:.2f}s"
                    )

                pair_idx += 1

        if args.save_teams:
            teambuild_record: dict[str, Any] = {}
            for name in player_names:
                t = teams[name]
                if t is None:
                    teambuild_record[name] = {
                        "ok": False,
                        "build_time_sec": round(build_times[name], 4),
                        "team": None,
                    }
                else:
                    teambuild_record[name] = {
                        "ok": True,
                        "build_time_sec": round(build_times[name], 4),
                        "team": _team_to_dict(t),
                    }
            composition_rounds.append(
                {
                    "round": r_idx,
                    "roster_seed": round_seed,
                    "roster": [_roster_species_to_dict(sp) for sp in roster],
                    "teambuild": teambuild_record,
                    "matchups": matchup_records,
                }
            )

        elapsed = time.perf_counter() - t0
        times_str = " | ".join(f"{nm}: {t:.2f}s" for nm, t in build_times.items())
        sel_str = " | ".join(f"{nm}: {t:.2f}s" for nm, t in sel_time_sums.items())
        print(f"  Round {r_idx + 1:2d}/{args.n_rounds}  ({elapsed:.1f}s elapsed)")
        print(f"    teambuild: {times_str}")
        print(f"    selection: {sel_str}")

    print("\n" + "=" * 64)
    print("Pairwise Win Rates (row beats column)")
    print("=" * 64)

    results_matrix: dict[str, dict[str, str]] = {n: {} for n in player_names}
    pair_details: list[dict[str, Any]] = []

    for p1, p2 in pair_keys:
        w_arr = np.array(wins_by_pair[(p1, p2)], dtype=np.int_)
        l_arr = np.array(losses_by_pair[(p1, p2)], dtype=np.int_)

        wr, ci_lo, ci_hi = _bootstrap_ci(w_arr, l_arr)
        total_w = int(w_arr.sum())
        total_l = int(l_arr.sum())
        total_d = int(np.array(draws_by_pair[(p1, p2)]).sum())
        n_total = total_w + total_l + total_d

        results_matrix[p1][p2] = f"{wr:.3f} [{ci_lo:.3f},{ci_hi:.3f}]"
        results_matrix[p2][p1] = f"{1 - wr:.3f} [{1 - ci_hi:.3f},{1 - ci_lo:.3f}]"

        pair_details.append(
            {
                "pair": f"{p1} vs {p2}",
                "wins_a": total_w,
                "wins_b": total_l,
                "draws": total_d,
                "total": n_total,
                "win_rate_a": round(wr, 4),
                "ci_low": round(ci_lo, 4),
                "ci_high": round(ci_hi, 4),
            }
        )

        if ci_lo > 0.5:
            sig = f"  ** {p1} significantly better"
        elif ci_hi < 0.5:
            sig = f"  ** {p2} significantly better"
        else:
            sig = "  (not significant)"

        print(
            f"  {p1:>12} vs {p2:<12}: {total_w:4d}-{total_l:4d}-{total_d:3d}  "
            f"WR={wr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]{sig}"
        )

    print("\n" + "=" * 64)
    print("Win Rate Matrix")
    print("=" * 64)
    header = f"{'':>14}" + "".join(f"{n:>14}" for n in player_names)
    print(header)
    for p1 in player_names:
        row = f"{p1:>14}"
        for p2 in player_names:
            if p1 == p2:
                row += f"{'---':>14}"
            else:
                row += f"{results_matrix[p1].get(p2, 'N/A'):>14}"
        print(row)

    agg: dict[str, list[float]] = {n: [] for n in player_names}
    for p1, p2 in pair_keys:
        w_arr = np.array(wins_by_pair[(p1, p2)], dtype=np.int_)
        l_arr = np.array(losses_by_pair[(p1, p2)], dtype=np.int_)
        wr, _, _ = _bootstrap_ci(w_arr, l_arr)
        agg[p1].append(wr)
        agg[p2].append(1.0 - wr)

    mean_wr = {n: float(np.mean(v)) if v else 0.0 for n, v in agg.items()}
    rankings = sorted(mean_wr.items(), key=lambda x: -x[1])

    print("\n" + "=" * 64)
    print("Overall Ranking (mean pairwise win rate)")
    print("=" * 64)
    for rank, (name, wr) in enumerate(rankings, 1):
        marker = "  <-- Dongimon" if name == "Dongimon" else ""
        print(f"  {rank}. {name:<20} {wr:.4f}{marker}")
    print("=" * 64)

    elapsed_sec = round(time.perf_counter() - t0, 1)
    results_path = results_dir / f"elo_team_{battle_policy_name}_{timestamp}.json"
    output: dict[str, Any] = {
        "mode": f"teambuild_selection_{battle_policy_name}_battle",
        "description": (
            f"Battle policy neutralized (both sides = {battle_policy_name}). "
            "Win rates reflect teambuild + selection quality. "
            "Selection runs once per matchup; best pair ordered as active. "
            "Fully seeded, bootstrap 95% CI."
        ),
        "seed": args.seed,
        "n_rounds": args.n_rounds,
        "n_battles_per_pair_per_round": args.n_battles,
        "total_battles_per_pair": args.n_rounds * args.n_battles,
        "battle_policy": battle_policy_name,
        "build_size": args.build_size,
        "select_size": MAX_TEAM_SIZE,
        "selection_mode": args.selection_mode,
        "roster_size": ROSTER_SIZE,
        "moveset_size": MOVESET_SIZE,
        "bootstrap_n": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "tag": args.tag,
        "players": player_names,
        "pair_details": pair_details,
        "mean_win_rates": mean_wr,
        "rankings": [name for name, _ in rankings],
        "elapsed_sec": elapsed_sec,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if args.save_teams and composition_rounds:
        composition_dir.mkdir(parents=True, exist_ok=True)
        tag_suffix = f"_{args.tag}" if args.tag else ""
        teams_path = composition_dir / f"teams_{battle_policy_name}_{timestamp}{tag_suffix}.json"
        composition_payload: dict[str, Any] = {
            "meta": {
                "timestamp": timestamp,
                "seed": args.seed,
                "n_rounds": args.n_rounds,
                "n_battles": args.n_battles,
                "tag": args.tag,
                "players": player_names,
                "max_team_size": MAX_TEAM_SIZE,
                "n_active": N_ACTIVE,
                "max_moves": MAX_MOVES,
                "roster_size": ROSTER_SIZE,
                "moveset_size": MOVESET_SIZE,
                "mode": f"teambuild_selection_{battle_policy_name}_battle",
                "battle_policy": battle_policy_name,
                "description": (
                    f"Own teambuild+selection per competitor; all battles {battle_policy_name}. "
                    "Full vgc2 move/stat labels for manual bottleneck isolation. "
                    "Selection returns best pair as active then reserves."
                ),
                "results_path": str(results_path),
            },
            "rounds": composition_rounds,
            "summary": {
                "mean_win_rates": mean_wr,
                "rankings": [name for name, _ in rankings],
                "pair_details": pair_details,
                "elapsed_sec": elapsed_sec,
            },
        }
        with open(teams_path, "w", encoding="utf-8") as f:
            json.dump(composition_payload, f, indent=2)
        print(f"Teams saved to: {teams_path.resolve()}")


if __name__ == "__main__":
    main()
