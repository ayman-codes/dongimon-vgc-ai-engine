"""ELO rating system for multi-agent weight tuning tournaments.

Provides standard ELO probability, rating update, and multi-epoch
tournament logic. Reused from vgc2's competition ELO implementation
but adapted for isolated battle-policy-only evaluation.
"""

import math
from collections.abc import Callable


def elo_probability(rating_a: float, rating_b: float) -> float:
    """Calculate the expected win probability of player A against player B.

    Args:
        rating_a: ELO rating of player A.
        rating_b: ELO rating of player B.

    Returns:
        Probability that player A beats player B (0.0–1.0).
    """
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    winner_is_a: bool,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update ELO ratings for two players after a match.

    Player A's expected probability is computed, then both ratings are
    adjusted by K * (actual – expected).

    Args:
        rating_a: Pre-match ELO rating of player A.
        rating_b: Pre-match ELO rating of player B.
        winner_is_a: True if player A won the match, False otherwise.
        k: K-factor controlling rating volatility (default 32).

    Returns:
        Tuple of (new_rating_a, new_rating_b).
    """
    prob_a = elo_probability(rating_a, rating_b)

    if winner_is_a:
        return rating_a + k * (1.0 - prob_a), rating_b + k * (0.0 - (1.0 - prob_a))

    return rating_a + k * (0.0 - prob_a), rating_b + k * (1.0 - (1.0 - prob_a))


def run_elo_epoch(
    elos: dict[str, float],
    battle_runner: Callable[[str, str], tuple[int, int]],
    k: float = 32.0,
) -> dict[str, float]:
    """Run one epoch of ELO-paired matches.

    Sorts players by ELO descending, pairs them adjacently, runs
    battles for each pair, and updates ratings.

    Args:
        elos: Dict mapping player name to current ELO rating.
        battle_runner: Callable fn(p1_name, p2_name) -> (wins_p1, wins_p2).
        k: K-factor for rating updates.

    Returns:
        Updated elos dict after one round of pairings.
    """
    names = sorted(elos.keys(), key=lambda n: elos[n], reverse=True)
    new_elos = dict(elos)

    for i in range(0, len(names) - 1, 2):
        p1, p2 = names[i], names[i + 1]
        wins_p1, wins_p2 = battle_runner(p1, p2)
        p1_won = wins_p1 > wins_p2
        new_elos[p1], new_elos[p2] = update_elo(
            new_elos[p1],
            new_elos[p2],
            p1_won,
            k,
        )

    return new_elos


def run_dongimon_elo_epoch(
    elos: dict[str, float],
    dongimon_name: str,
    opponent_names: list[str],
    battle_runner: Callable[[str, str], tuple[int, int]],
    k: float = 32.0,
) -> dict[str, float]:
    """Run one epoch where only Dongimon plays against every opponent.

    Opponents never play each other — only Dongimon-vs-opponent matches
    are evaluated, providing clean signal for weight tuning.

    Args:
        elos: Dict mapping player name to current ELO rating.
        dongimon_name: Name of the Dongimon player.
        opponent_names: Names of all opponent players.
        battle_runner: Callable fn(p1_name, p2_name) -> (wins_p1, wins_p2).
        k: K-factor for rating updates.

    Returns:
        Updated elos dict after matches.
    """
    new_elos = dict(elos)
    for opp_name in opponent_names:
        wins_d, wins_o = battle_runner(dongimon_name, opp_name)
        d_won = wins_d > wins_o
        new_elos[dongimon_name], new_elos[opp_name] = update_elo(
            new_elos[dongimon_name],
            new_elos[opp_name],
            d_won,
            k,
        )
    return new_elos
