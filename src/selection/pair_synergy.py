"""Analytical pair-synergy scoring for the Selection Policy fast path.

When the team is already final size, selection is a pure ordering problem:
choose which two Pokemon start active and which two sit in reserve. This
module ranks candidate active pairs WITHOUT battle simulation, using four
teamwork-oriented terms that reward complementarity rather than raw
individual power:

    - offensive coverage vs the opponent's actual types,
    - defensive complementarity (partner covers my weaknesses),
    - speed control (initiative plus a complementary speed spread),
    - role/stat balance (physical + special, diverse roles).

Defensive and role terms reuse the evolutionary team fitness operators,
which are themselves complementarity metrics (weakness coverage, role
mix, evenness) rather than max-is-better aggregates.
"""

from typing import Any

from vgc2.battle_engine.modifiers import Stat

from src.shared.types import type_effectiveness, vgc2_type_to_name
from src.teambuild.operators import _fitness_role_diversity, _fitness_stat_diversity, _fitness_type_defence


def _member_species(member: Any) -> Any:
    """Unwrap a team member to its PokemonSpecies.

    Args:
        member: A Pokemon member (has ``.species``) or a bare species.

    Returns:
        The underlying PokemonSpecies object.
    """
    return member.species if hasattr(member, "species") else member


def _species_type_names(species: Any) -> list[str]:
    """Return lowercase type names for a species.

    Args:
        species: A PokemonSpecies with a ``.types`` iterable of vgc2 Types.

    Returns:
        List of lowercase type name strings.
    """
    return [vgc2_type_to_name(t.value) for t in species.types]


def offensive_coverage_vs(pair_species: list[Any], opp_views: list[Any]) -> float:
    """Score how well the pair's combined moves hit the opponent's types.

    For each opponent, the best super-effective damage ratio across both
    pair members is summed, then normalized by the number of opponents.
    Rewards pairs that jointly threaten the specific opponent field rather
    than generic all-type breadth.

    Args:
        pair_species: The two species forming the candidate active pair.
        opp_views: Opponent PokemonView list.

    Returns:
        Coverage score >= 0. Zero when there are no opponents or no moves.
    """
    if not opp_views:
        return 0.0

    total = 0.0
    for opp_view in opp_views:
        opp_spec = opp_view.species if hasattr(opp_view, "species") else opp_view
        opp_types = _species_type_names(opp_spec)
        best = 0.0
        for spec in pair_species:
            for move in spec.moves:
                if move.base_power <= 0:
                    continue
                acc = move.accuracy if move.accuracy is not None else 1.0
                stab = 1.5 if move.pkm_type in spec.types else 1.0
                atk_name = vgc2_type_to_name(
                    move.pkm_type.value if hasattr(move.pkm_type, "value") else move.pkm_type
                )
                eff = type_effectiveness(atk_name, opp_types)
                ratio = move.base_power * acc * stab * eff
                if ratio > best:
                    best = ratio
        total += best
    return total / len(opp_views)


def speed_control(pair_species: list[Any], opp_views: list[Any]) -> float:
    """Score the pair's speed control relative to the opponent field.

    Combines initiative (fraction of opponents outsped by the pair's
    fastest member) with a complementary speed spread (fast + slow members
    score higher than two same-speed members).

    Args:
        pair_species: The two species forming the candidate active pair.
        opp_views: Opponent PokemonView list.

    Returns:
        Speed control score in [0, 1].
    """
    my_speeds = [float(s.base_stats[Stat.SPEED]) for s in pair_species]
    if not my_speeds:
        return 0.0
    my_max = max(my_speeds)

    opp_speeds: list[float] = []
    for opp_view in opp_views:
        opp_spec = opp_view.species if hasattr(opp_view, "species") else opp_view
        opp_speeds.append(float(opp_spec.base_stats[Stat.SPEED]))

    initiative = sum(1 for o in opp_speeds if my_max > o) / len(opp_speeds) if opp_speeds else 0.5

    spread = (max(my_speeds) - min(my_speeds)) / 130.0
    spread = min(spread, 1.0)
    return 0.7 * initiative + 0.3 * spread


def role_balance(pair_species: list[Any]) -> float:
    """Score the pair's role and stat balance (teamwork, not peak power).

    Blends stat diversity (physical + special attackers, fast + slow tiers)
    with role diversity (sweeper / wall / mixed mix), reusing the
    evolutionary team fitness operators.

    Args:
        pair_species: The two species forming the candidate active pair.

    Returns:
        Role balance score in [0, 1].
    """
    return 0.5 * _fitness_stat_diversity(pair_species) + 0.5 * _fitness_role_diversity(pair_species)


def defensive_synergy(pair_species: list[Any]) -> float:
    """Score the pair's defensive complementarity.

    Fraction of each member's weaknesses that the partner resists or is
    immune to, reusing the evolutionary type-defence fitness operator.

    Args:
        pair_species: The two species forming the candidate active pair.

    Returns:
        Defensive synergy fraction in [0, 1].
    """
    return _fitness_type_defence(pair_species)


def pair_synergy_terms(pair_species: list[Any], opp_views: list[Any]) -> dict[str, float]:
    """Compute the four intra-pair teamwork terms individually.

    Returned separately so the Selection Policy can apply Optuna-tuned
    weights per term. Each term is in [0, 1] (coverage is normalized by
    opponent count and typically lands in a comparable range).

    Args:
        pair_species: The two species forming the candidate active pair.
        opp_views: Opponent PokemonView list.

    Returns:
        Dict with keys ``coverage``, ``defense``, ``speed``, ``role``.
    """
    return {
        "coverage": offensive_coverage_vs(pair_species, opp_views),
        "defense": defensive_synergy(pair_species),
        "speed": speed_control(pair_species, opp_views),
        "role": role_balance(pair_species),
    }


def score_pair_synergy(pair_species: list[Any], opp_views: list[Any]) -> float:
    """Aggregate the four intra-pair teamwork terms into one score.

    Equal-weighted blend, useful as a parameter-free baseline. The
    Selection Policy applies Optuna-tuned per-term weights instead.

    Args:
        pair_species: The two species forming the candidate active pair.
        opp_views: Opponent PokemonView list.

    Returns:
        Combined synergy score (higher = better teamwork).
    """
    terms = pair_synergy_terms(pair_species, opp_views)
    return 0.25 * (terms["coverage"] + terms["defense"] + terms["speed"] + terms["role"])
