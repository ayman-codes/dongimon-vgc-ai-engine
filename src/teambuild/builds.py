"""Single optimal build creation and species power scoring for the Team Build Policy.

Uses physical/special lean detection for EV/nature, damage-first movesets,
and a BST-boosted species power proxy.
"""

from typing import Any

from vgc2.battle_engine.modifiers import Category, Nature, Stat, Status, Terrain, Weather
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies


def create_single_optimal_build(species: PokemonSpecies) -> Pokemon | None:
    """Create one optimal build for a species using offense-first role detection.

    Physical vs special lean uses sum(BP × attack stat). EVs/nature use
    HP-offense spreads (ADAMANT/MODEST). Moves prefer high BP/STAB with at
    most one utility slot.

    Args:
        species: The Pokemon species to build.

    Returns:
        A single Pokemon with optimised EVs, IVs, nature, and 4 moves,
        or None if the species has no moves.
    """
    if not species.moves:
        return None

    base = species.base_stats

    role = _detect_role(base, species)
    evs, nature = _pick_evs_and_nature(role, base)
    move_indices = _pick_moves(species)

    return Pokemon(
        species=species,
        move_indexes=move_indices,
        level=100,
        evs=evs,
        ivs=(31, 31, 31, 31, 31, 31),
        nature=nature,
    )


def species_power(species: PokemonSpecies) -> float:
    """Compute a damage + BST power proxy for a species.

    Offensive term: max over damaging moves of accuracy × BP × attack × STAB.
    Bulk term: HP × (Def + SpD) × 0.35 (down-weighted vs prior bulk-first).
    BST term: sum(base_stats) × 1.5 (firepower bias).

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

    best_offensive = 0.0
    for m in species.moves:
        if m.base_power <= 0:
            continue
        acc = m.accuracy if m.accuracy is not None else 1.0
        stab = 1.5 if m.pkm_type in species.types else 1.0
        if m.category in phys_cats:
            score = acc * m.base_power * atk * stab
        elif m.category in spec_cats:
            score = acc * m.base_power * spa * stab
        else:
            continue
        if score > best_offensive:
            best_offensive = score

    bulk = hp * (df + spd) * 0.35
    bst = float(sum(base)) * 1.5
    return float(best_offensive + bulk + bst)


def species_role(species: PokemonSpecies) -> str:
    """Detect the competitive role of a species.

    Returns one of ``"sweeper"``, ``"wall"``, or ``"mixed"``.

    Args:
        species: The Pokemon species.

    Returns:
        Role label string.
    """
    role, _ = _detect_role(species.base_stats, species)
    return role


def _detect_role(base: tuple[int, ...], species: PokemonSpecies | None = None) -> tuple[str, str]:
    """Detect role from base stats and optional movepool lean.

    When species is provided, physical vs special is decided by
    sum(BP × Atk) vs sum(BP × SpA) over the movepool.

    Args:
        base: Base stats tuple (HP, Atk, Def, SpA, SpD, Spe).
        species: Optional species for movepool-based phys/spec lean.

    Returns:
        Tuple of (role_label, subtype).
    """
    hp, atk, df, spa, spd, spe = base

    is_phys_lean = atk > spa + 10
    is_spec_lean = spa > atk + 10

    if species is not None and species.moves:
        phys_sum = sum(
            m.base_power * atk
            for m in species.moves
            if m.category in (Category.PHYSICAL, Category.PHYSICAL.value) and m.base_power > 0
        )
        spc_sum = sum(
            m.base_power * spa
            for m in species.moves
            if m.category in (Category.SPECIAL, Category.SPECIAL.value) and m.base_power > 0
        )
        if phys_sum > spc_sum:
            is_phys_lean = True
            is_spec_lean = False
        elif spc_sum > phys_sum:
            is_spec_lean = True
            is_phys_lean = False

    if is_phys_lean and atk + spe >= hp + df:
        return "sweeper", "physical"
    if is_spec_lean and spa + spe >= hp + spd:
        return "sweeper", "special"
    if is_phys_lean:
        return "sweeper", "physical"
    if is_spec_lean:
        return "sweeper", "special"

    if hp + df > atk + spa + spe and hp + df > hp + spd:
        return "wall", "physical_defense"
    if hp + spd > atk + spa + spe:
        return "wall", "special_defense"

    return "mixed", "allrounder"


def _pick_evs_and_nature(role: tuple[str, str], base: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Pick EV spread and nature with offense-first spreads.

    Physical: ADAMANT (252 HP / 168 Atk / 84 SpA dump / 6 Spe).
    Special: MODEST (252 HP / 84 Atk dump / 168 SpA / 6 Spe).
    Walls keep bulk spreads; mixed defaults to the stronger offense lean.

    Args:
        role: Tuple of (role_label, subtype) from _detect_role.
        base: Base stats tuple.

    Returns:
        Tuple of (evs: tuple[6], nature: Nature enum value).
    """
    _, subtype = role

    if subtype == "physical":
        return (252, 168, 0, 84, 0, 6), Nature.ADAMANT
    if subtype == "special":
        return (252, 84, 0, 168, 0, 6), Nature.MODEST

    if subtype == "physical_defense":
        return (252, 0, 168, 0, 84, 6), Nature.IMPISH
    if subtype == "special_defense":
        return (252, 0, 84, 0, 168, 6), Nature.CALM

    if base[Stat.ATTACK] >= base[Stat.SPECIAL_ATTACK]:
        return (252, 168, 0, 84, 0, 6), Nature.ADAMANT
    return (252, 84, 0, 168, 0, 6), Nature.MODEST


def _move_utility(move: Any, species: PokemonSpecies) -> float:
    """Score a non-damaging move by its utility value (down-weighted).

    Args:
        move: The Move to evaluate.
        species: The Pokemon species using the move.

    Returns:
        Float utility score.
    """
    if move.heal > 0:
        return 40.0
    if any(b > 0 for b in move.boosts) and move.self_boosts:
        boost_sum = sum(b for b in move.boosts if b > 0)
        return 20.0 * float(boost_sum)
    if move.hazard is not None:
        return 30.0
    if move.protect:
        return 25.0
    if move.status != Status.NONE:
        return 28.0
    if move.toggle_reflect or move.toggle_lightscreen:
        return 35.0
    if move.toggle_tailwind or move.toggle_trickroom:
        return 30.0
    if move.weather_start != Weather.CLEAR or move.field_start != Terrain.NONE:
        return 30.0
    return 0.0


def _pick_moves(species: PokemonSpecies) -> list[int]:
    """Select the best 4 moves with damage-first priority.

    Damaging moves scored by BP × STAB × accuracy (+priority bonus).
    At most one OTHER/utility move is kept so peak firepower is preserved.

    Args:
        species: The Pokemon species.

    Returns:
        List of up to 4 move indices into species.moves.
    """
    damaging_cats = (Category.PHYSICAL, Category.SPECIAL, Category.PHYSICAL.value, Category.SPECIAL.value)

    scored: list[tuple[float, Any, bool]] = []
    for m in species.moves:
        is_damage = m.base_power > 0 and m.category in damaging_cats
        if is_damage:
            stab = 1.5 if m.pkm_type in species.types else 1.0
            acc = m.accuracy if m.accuracy is not None else 1.0
            pri = 150.0 if getattr(m, "priority", 0) and m.priority > 0 else 0.0
            damage_score = m.base_power * 2.0 * stab * acc + pri
            scored.append((damage_score, m, True))
        else:
            scored.append((_move_utility(m, species), m, False))

    scored.sort(key=lambda x: -x[0])

    selected: list[Any] = []
    n_utility = 0
    seen_damage_types: dict[Any, int] = {}
    for _score, move, is_damage in scored:
        if is_damage:
            type_count = seen_damage_types.get(move.pkm_type, 0)
            if type_count >= 2 and any(
                getattr(m, "base_power", 0) > 0 for m in selected
            ):
                continue
            seen_damage_types[move.pkm_type] = type_count + 1
        else:
            if n_utility >= 1:
                continue
            n_utility += 1
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
