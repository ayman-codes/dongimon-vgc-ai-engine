"""All-vs-all benchmark — teambuild + selection quality (battle neutralized).

All competitors use GreedyBattlePolicy to eliminate battle-policy bias.
Win-rate differences reflect ONLY teambuild + selection quality.

Design (statistically valid):
  - Each round generates a shared species roster (fixed seed).
  - Each competitor builds a team from the SAME roster via their own TeamBuildPolicy.
  - Each competitor selects 4 via their own SelectionPolicy.
  - Battles are fully seeded (acc_rng, eff_rng, sta_rng) with Greedy battle.
  - Primary metric: pairwise win rate with bootstrap 95% CI.
  - No ELO path-dependence; win rate is bounded [0, 1].

Usage:
    uv run python scripts/benchmark_team.py --seed=42 --n-rounds=10 --n-battles=100
    uv run python scripts/benchmark_team.py --save-teams   # also save built teams
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from numpy.typing import NDArray
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import BasicSelectionPolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam
from vgc2.battle_engine.game_state import State, get_battle_teams
from vgc2.battle_engine.modifiers import Nature
from vgc2.battle_engine.pokemon import Pokemon
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.ecosystem import build_team, sanitized_team_build_decision
from vgc2.competition.match import subteam
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from competitor import DongimonCompetitor
from src.config.loader import load_battle_weights

N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95
ROSTER_SIZE = 50
MOVESET_SIZE = 200
MAX_TEAM_SIZE = 4
MAX_MOVES = 4
N_ACTIVE = 2


# ─── Competitor factories ─────────────────────────────────────────


def _import_competitor_cls(module_path: str, class_name: str) -> Any:
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


class _GreedyBaseline:
    """Baseline: random team from roster + BasicSelection + Greedy battle."""

    name = "Greedy"

    def __init__(self) -> None:
        self.selectionpolicy = BasicSelectionPolicy()
        self.battlepolicy = GreedyBattlePolicy()

    def build_team(self, roster: list[Any], rng: np.random.Generator) -> Team:
        """Pick 6 random species and hydrate with random spreads."""
        indices = rng.choice(len(roster), size=min(MAX_TEAM_SIZE, len(roster)), replace=False)
        members = []
        for idx in indices:
            species = roster[int(idx)]
            n_moves = len(species.moves)
            move_idx = list(rng.choice(n_moves, size=min(MAX_MOVES, n_moves), replace=False))
            evs = tuple(int(x) for x in rng.multinomial(510, [1 / 6] * 6))
            nature = Nature(int(rng.integers(0, len(Nature))))
            p = Pokemon(
                species=species, move_indexes=move_idx, level=100,
                ivs=(31,) * 6, evs=evs, nature=nature,
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


def _build_team_for(
    name: str,
    competitor: Any,
    roster: list[Any],
    rng: np.random.Generator,
) -> Team | None:
    """Build a team for a competitor. Returns None on failure."""
    if name == "Greedy":
        baseline = _GreedyBaseline()
        return baseline.build_team(roster, rng)

    try:
        commands = sanitized_team_build_decision(
            competitor.teambuildpolicy, roster, None, MAX_TEAM_SIZE, MAX_MOVES, N_ACTIVE
        )
        if not commands:
            return None
        return build_team(commands, roster)
    except Exception as exc:
        print(f"    [WARN] {name} teambuild failed: {type(exc).__name__}: {exc}")
        return None


# ─── Seeded battle loop ──────────────────────────────────────────


def _run_seeded_match(
    team_a: Team,
    team_b: Team,
    sel_a: Any,
    sel_b: Any,
    params: BattleRuleParam,
    n_battles: int,
    match_seed: int,
) -> tuple[int, int, int]:
    """Run N seeded battles between two teams with Greedy battle policy.

    Args:
        team_a: Full team for side A.
        team_b: Full team for side B.
        sel_a: Selection policy for side A.
        sel_b: Selection policy for side B.
        params: Battle rule parameters.
        n_battles: Number of battles to run.
        match_seed: Base seed for this matchup.

    Returns:
        Tuple of (wins_a, wins_b, draws).
    """
    greedy_bp = GreedyBattlePolicy()
    view_a = TeamView(team_a)
    view_b = TeamView(team_b)

    wins_a = 0
    wins_b = 0
    draws = 0

    for b_idx in range(n_battles):
        battle_seed = match_seed + b_idx
        gen = np.random.default_rng(battle_seed)

        idx_a = sel_a.decision((team_a, view_b), 4)
        idx_b = sel_b.decision((team_b, view_a), 4)

        sub_a, sub_view_a = subteam(team_a, view_a, idx_a)
        sub_b, sub_view_b = subteam(team_b, view_b, idx_b)

        battle_teams = get_battle_teams((sub_a, sub_b), N_ACTIVE)
        state = State(battle_teams)
        rng_tuple = ((gen, gen), (gen, gen))
        engine = BattleEngine(
            state, params=params,
            acc_rng=rng_tuple, eff_rng=rng_tuple, sta_rng=rng_tuple,
        )

        while not engine.finished():
            sv0 = StateView(engine.state, 0, (sub_view_a, sub_view_b))
            sv1 = StateView(engine.state, 1, (sub_view_b, sub_view_a))
            cmd0 = greedy_bp.decision(sv0, sub_view_b)
            cmd1 = greedy_bp.decision(sv1, sub_view_a)
            engine.run_turn((cmd0, cmd1))

        if engine.winning_side == 0:
            wins_a += 1
        elif engine.winning_side == 1:
            wins_b += 1
        else:
            draws += 1

    return wins_a, wins_b, draws


# ─── Bootstrap CI ─────────────────────────────────────────────────


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


# ─── Team serialization for --save-teams ──────────────────────────


def _team_to_dict(team: Team) -> list[dict[str, Any]]:
    """Serialize a Team to a JSON-friendly list."""
    members = []
    for pkm in team.members:
        members.append({
            "species": str(pkm.species.name) if hasattr(pkm.species, "name") else str(pkm.species),
            "types": [str(t) for t in pkm.species.types] if hasattr(pkm.species, "types") else [],
            "moves": [str(m) for m in pkm.moves] if hasattr(pkm, "moves") else [],
            "evs": list(pkm.evs) if hasattr(pkm, "evs") else [],
            "nature": str(pkm.nature) if hasattr(pkm, "nature") else "",
        })
    return members


# ─── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-vs-all teambuild+selection benchmark (battle neutralized with Greedy)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--n-rounds", type=int, default=10, help="Number of roster rounds")
    parser.add_argument("--n-battles", type=int, default=100, help="Battles per pair per round")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--save-teams", action="store_true", help="Save built teams per round")
    args = parser.parse_args()

    player_names = [p[0] for p in _PLAYER_ROSTER]
    n_players = len(player_names)
    params = BattleRuleParam()

    # Pre-instantiate competitors (once)
    competitors: dict[str, Any] = {}
    weights_dict = load_battle_weights().model_dump()
    for name, cls in _PLAYER_ROSTER:
        if name == "Dongimon":
            competitors[name] = DongimonCompetitor(custom_weights=weights_dict)
        elif name == "Greedy":
            competitors[name] = _GreedyBaseline()
        else:
            competitors[name] = cls()

    # Selection policies
    sel_cache: dict[str, Any] = {}
    for name in player_names:
        if name == "Greedy":
            sel_cache[name] = BasicSelectionPolicy()
        else:
            sel_cache[name] = competitors[name].selectionpolicy

    # Pairwise accumulators
    pair_keys: list[tuple[str, str]] = []
    for i in range(n_players):
        for j in range(i + 1, n_players):
            pair_keys.append((player_names[i], player_names[j]))

    wins_by_pair: dict[tuple[str, str], list[int]] = {k: [] for k in pair_keys}
    losses_by_pair: dict[tuple[str, str], list[int]] = {k: [] for k in pair_keys}
    draws_by_pair: dict[tuple[str, str], list[int]] = {k: [] for k in pair_keys}

    total_matchups = args.n_rounds * len(pair_keys)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 64)
    print("Dongimon Teambuild+Selection Benchmark (battle = Greedy)")
    print(f"  seed={args.seed}, n_rounds={args.n_rounds}, n_battles={args.n_battles}")
    print(f"  players: {', '.join(player_names)}")
    print(f"  roster: {ROSTER_SIZE} species, {MOVESET_SIZE} moves")
    print(f"  total matchups: {total_matchups}")
    print(f"  battles per pair (total): {args.n_rounds * args.n_battles}")
    print(f"  save_teams: {args.save_teams}")
    print("=" * 64)

    t0 = time.perf_counter()
    saved_teams: list[dict[str, Any]] = []

    for r_idx in range(args.n_rounds):
        round_seed = args.seed + r_idx * 10_000
        roster_rng = np.random.default_rng(round_seed)

        # Generate shared roster for this round
        moveset = gen_move_set(MOVESET_SIZE, roster_rng)
        roster = gen_pkm_roster(ROSTER_SIZE, moveset, MAX_MOVES, roster_rng)

        # Each competitor builds from the SAME roster
        teams: dict[str, Team | None] = {}
        build_times: dict[str, float] = {}
        for name in player_names:
            build_rng = np.random.default_rng(round_seed + hash(name) % 10_000)
            t_build = time.perf_counter()
            teams[name] = _build_team_for(name, competitors[name], roster, build_rng)
            build_times[name] = time.perf_counter() - t_build

        # Save teams if requested
        if args.save_teams:
            round_record: dict[str, Any] = {"round": r_idx, "roster_seed": round_seed, "teams": {}}
            for name in player_names:
                t = teams[name]
                round_record["teams"][name] = _team_to_dict(t) if t else "BUILD_FAILED"
            saved_teams.append(round_record)

        # Skip pairs where a team failed to build
        pair_idx = 0
        for i in range(n_players):
            for j in range(i + 1, n_players):
                p1, p2 = player_names[i], player_names[j]
                team_a, team_b = teams[p1], teams[p2]

                if team_a is None or team_b is None:
                    # Award wins to the side that succeeded
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
                    pair_idx += 1
                    continue

                matchup_seed = round_seed + pair_idx * 1_000 + 1
                w1, w2, dr = _run_seeded_match(
                    team_a, team_b,
                    sel_cache[p1], sel_cache[p2],
                    params, args.n_battles, matchup_seed,
                )

                wins_by_pair[(p1, p2)].append(w1)
                losses_by_pair[(p1, p2)].append(w2)
                draws_by_pair[(p1, p2)].append(dr)
                pair_idx += 1

        elapsed = time.perf_counter() - t0
        times_str = " | ".join(f"{nm}: {t:.2f}s" for nm, t in build_times.items())
        print(f"  Round {r_idx + 1:2d}/{args.n_rounds}  ({elapsed:.1f}s elapsed)")
        print(f"    teambuild: {times_str}")

    # ── Results ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Pairwise Win Rates (row beats column)")
    print("=" * 64)

    results_matrix: dict[str, dict[str, str]] = {n: {} for n in player_names}
    pair_details: list[dict[str, Any]] = []

    for (p1, p2) in pair_keys:
        w_arr = np.array(wins_by_pair[(p1, p2)], dtype=np.int_)
        l_arr = np.array(losses_by_pair[(p1, p2)], dtype=np.int_)

        wr, ci_lo, ci_hi = _bootstrap_ci(w_arr, l_arr)
        total_w = int(w_arr.sum())
        total_l = int(l_arr.sum())
        total_d = int(np.array(draws_by_pair[(p1, p2)]).sum())
        n_total = total_w + total_l + total_d

        results_matrix[p1][p2] = f"{wr:.3f} [{ci_lo:.3f},{ci_hi:.3f}]"
        results_matrix[p2][p1] = f"{1 - wr:.3f} [{1 - ci_hi:.3f},{1 - ci_lo:.3f}]"

        pair_details.append({
            "pair": f"{p1} vs {p2}",
            "wins_a": total_w,
            "wins_b": total_l,
            "draws": total_d,
            "total": n_total,
            "win_rate_a": round(wr, 4),
            "ci_low": round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
        })

        sig = ""
        if ci_lo > 0.5:
            sig = f"  ** {p1} significantly better"
        elif ci_hi < 0.5:
            sig = f"  ** {p2} significantly better"
        else:
            sig = "  (not significant)"

        print(f"  {p1:>12} vs {p2:<12}: {total_w:4d}-{total_l:4d}-{total_d:3d}  "
              f"WR={wr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]{sig}")

    # ── Summary table ────────────────────────────────────────────
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

    # ── Aggregate ranking ────────────────────────────────────────
    agg: dict[str, list[float]] = {n: [] for n in player_names}
    for (p1, p2) in pair_keys:
        w_arr = np.array(wins_by_pair[(p1, p2)], dtype=np.int_)
        l_arr = np.array(losses_by_pair[(p1, p2)], dtype=np.int_)
        wr, _, _ = _bootstrap_ci(w_arr, l_arr)
        agg[p1].append(wr)
        agg[p2].append(1.0 - wr)

    mean_wr = {n: float(np.mean(v)) for n, v in agg.items()}
    rankings = sorted(mean_wr.items(), key=lambda x: -x[1])

    print("\n" + "=" * 64)
    print("Overall Ranking (mean pairwise win rate)")
    print("=" * 64)
    for rank, (name, wr) in enumerate(rankings, 1):
        marker = "  <-- Dongimon" if name == "Dongimon" else ""
        print(f"  {rank}. {name:<20} {wr:.4f}{marker}")
    print("=" * 64)

    # ── Save ─────────────────────────────────────────────────────
    results_path = os.path.join(results_dir, f"elo_team_{timestamp}.json")
    output: dict[str, Any] = {
        "mode": "teambuild_selection_greedy_battle",
        "description": (
            "Battle policy neutralized (all Greedy). "
            "Win rates reflect teambuild + selection quality. "
            "Fully seeded, bootstrap 95% CI."
        ),
        "seed": args.seed,
        "n_rounds": args.n_rounds,
        "n_battles_per_pair_per_round": args.n_battles,
        "total_battles_per_pair": args.n_rounds * args.n_battles,
        "roster_size": ROSTER_SIZE,
        "moveset_size": MOVESET_SIZE,
        "bootstrap_n": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "tag": args.tag,
        "players": player_names,
        "pair_details": pair_details,
        "mean_win_rates": mean_wr,
        "rankings": [name for name, _ in rankings],
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if args.save_teams and saved_teams:
        teams_path = os.path.join(results_dir, f"team_log_{timestamp}.json")
        with open(teams_path, "w", encoding="utf-8") as f:
            json.dump(saved_teams, f, indent=2)
        print(f"Teams saved to: {teams_path}")


if __name__ == "__main__":
    main()
