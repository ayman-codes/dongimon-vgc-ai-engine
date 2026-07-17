"""Single optimal build creation and species power scoring for the Team Build Policy.

Uses minimon-style role detection (sweeper, tank, mixed) to produce one
optimal EV spread, nature, and moveset per species — no damage calculations,
pure base-stat arithmetic.
"""

from typing import Any

from vgc2.battle_engine.modifiers import Category, Nature, Stat, Status, Terrain, Weather
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies


def create_single_optimal_build(species: PokemonSpecies) -> Pokemon | None:
    """Create one optimal build for a species using minimon-style role detection.

    Determines if the species is a sweeper, tank, or mixed attacker based
    on base stats, assigns role-appropriate EVs and nature, and selects
    the best 4 moves by STAB-weighted base power.

    Args:
        species: The Pokemon species to build.

    Returns:
        A single Pokemon with optimised EVs, IVs, nature, and 4 moves,
        or None if the species has no moves.
    """
    if not species.moves:
        return None

    base = species.base_stats

    role = _detect_role(base)
    evs, nature = _pick_evs_and_nature(role, base)
    move_indices = _pick_moves(species)

    return Pokemon(
        species=species,
        move_indexes=move_indices,
        level=50,
        evs=evs,
        ivs=(31, 31, 31, 31, 31, 31),
        nature=nature,
    )


def species_power(species: PokemonSpecies) -> float:
    """Compute a stat-based raw combat power proxy.

    Uses the product of the best attacking stat and best move base power
    for offensive power, plus a bulk metric. No damage formula calls.

    Args:
        species: The Pokemon species to evaluate.

    Returns:
        Float power score. Higher = stronger in a vacuum.
    """
    base = species.base_stats
    hp = base[Stat.MAX_HP]
    atk = base[Stat.ATTACK]
    spa = base[Stat.SPECIAL_ATTACK]
    df = base[Stat.DEFENSE]
    spd = base[Stat.SPECIAL_DEFENSE]

    phys_cats = (Category.PHYSICAL, Category.PHYSICAL.value)
    spec_cats = (Category.SPECIAL, Category.SPECIAL.value)

    best_phys = max(
        (m.base_power for m in species.moves if m.category in phys_cats and m.base_power > 0),
        default=0,
    )
    best_spec = max(
        (m.base_power for m in species.moves if m.category in spec_cats and m.base_power > 0),
        default=0,
    )

    offensive = max(atk * best_phys, spa * best_spec)
    bulk = hp * (df + spd) * 0.5
    return float(offensive + bulk)


def species_role(species: PokemonSpecies) -> str:
    """Detect the competitive role of a species.

    Returns one of ``"sweeper"``, ``"wall"``, or ``"mixed"``.

    Args:
        species: The Pokemon species.

    Returns:
        Role label string.
    """
    role, _ = _detect_role(species.base_stats)
    return role


def _detect_role(base: tuple[int, ...]) -> tuple[str, str]:
    """Detect role from base stats.

    Args:
        base: Base stats tuple (HP, Atk, Def, SpA, SpD, Spe).

    Returns:
        Tuple of (role_label, subtype) where role is one of
        ``"sweeper"``, ``"wall"``, ``"mixed"``.
    """
    hp, atk, df, spa, spd, spe = base

    is_phys_lean = atk > spa + 10
    is_spec_lean = spa > atk + 10

    if is_phys_lean and atk + spe > hp + df + spd:
        return "sweeper", "physical"
    if is_spec_lean and spa + spe > hp + df + spd:
        return "sweeper", "special"

    if is_phys_lean and hp + df > atk + spa + spe:
        return "wall", "physical_defense"
    if is_spec_lean and hp + spd > atk + spa + spe:
        return "wall", "special_defense"
    if hp + df > atk + spa + spe:
        return "wall", "physical_defense"
    if hp + spd > atk + spa + spe:
        return "wall", "special_defense"

    return "mixed", "allrounder"


def _pick_evs_and_nature(role: tuple[str, str], base: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Pick EV spread and nature for a role.

    Args:
        role: Tuple of (role_label, subtype) from _detect_role.
        base: Base stats tuple.

    Returns:
        Tuple of (evs: tuple[6], nature: Nature enum value).
    """
    _, subtype = role

    if subtype == "physical":
        return (6, 252, 0, 0, 0, 252), Nature.ADAMANT
    if subtype == "special":
        return (6, 0, 0, 252, 0, 252), Nature.TIMID

    if subtype == "physical_defense":
        return (252, 0, 252, 0, 6, 0), Nature.IMPISH
    if subtype == "special_defense":
        return (252, 0, 6, 0, 252, 0), Nature.CALM

    return (252, 128, 0, 128, 0, 2), Nature.HARDY


def _move_utility(move: Any, species: PokemonSpecies) -> float:
    """Score a non-damaging move by its utility value.

    Args:
        move: The Move to evaluate.
        species: The Pokemon species using the move.

    Returns:
        Float utility score.
    """
    if move.heal > 0:
        return 80.0
    if any(b > 0 for b in move.boosts) and move.self_boosts:
        boost_sum = sum(b for b in move.boosts if b > 0)
        return 30.0 * float(boost_sum)
    if move.hazard is not None:
        return 50.0
    if move.protect:
        return 40.0
    if move.status != Status.NONE:
        return 35.0
    if move.toggle_reflect or move.toggle_lightscreen:
        return 60.0
    if move.toggle_tailwind or move.toggle_trickroom:
        return 45.0
    if move.weather_start != Weather.CLEAR or move.field_start != Terrain.NONE:
        return 50.0
    return 0.0


def _pick_moves(species: PokemonSpecies) -> list[int]:
    """Select the best 4 moves for a species.

    Scores all moves by a combination of damage potential (STAB-weighted
    base power) and utility value (setup, recovery, hazards, status, etc.).
    Damaging moves are prioritised for type diversity.

    Args:
        species: The Pokemon species.

    Returns:
        List of up to 4 move indices into species.moves.
    """
    damaging_cats = (Category.PHYSICAL, Category.SPECIAL, Category.PHYSICAL.value, Category.SPECIAL.value)

    scored = []
    for m in species.moves:
        damage_score = 0.0
        if m.base_power > 0 and m.category in damaging_cats:
            stab = 1.5 if m.pkm_type in species.types else 1.0
            damage_score = m.base_power * stab
        utility_score = _move_utility(m, species)
        scored.append((damage_score + utility_score, m))

    scored.sort(key=lambda x: -x[0])

    selected: list[Any] = []
    seen_damage_types = set()
    for _score, move in scored:
        if move.base_power > 0:
            if move.pkm_type in seen_damage_types and any(m.base_power > 0 for m in selected):
                continue
            seen_damage_types.add(move.pkm_type)
        selected.append(move)
        if len(selected) == 4:
            break

    if len(selected) < 4:
        for move in species.moves:
            if move not in selected and len(selected) < 4:
                selected.append(move)

    indices = []
    for move in selected[:4]:
        try:
            idx = species.moves.index(move)
            indices.append(idx)
        except ValueError:
            continue

    if not indices:
        indices = list(range(min(4, len(species.moves))))

    return indices
