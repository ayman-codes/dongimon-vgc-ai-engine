"""Archetype build generation for team preview and team building.

Produces fully-formed Pokemon objects with competitive EV spreads,
natures, and move sets. Used by both Selection and Teambuild policies.
"""

from typing import Any

from vgc2.battle_engine.modifiers import Nature, Stat, Status
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies


def _archetype_moves(name: str, predicted_moveset: list[Any], species: PokemonSpecies) -> list[Any]:
    """Select moves appropriate for a given archetype.

    Walls get healing/status/support moves prioritised.
    Sweepers get damaging + setup moves prioritised.
    Mixed attackers get a balance of both.

    Args:
        name: Archetype name string.
        predicted_moveset: Full predicted moveset for the species.
        species: The Pokemon species.

    Returns:
        List of up to 4 Move objects.
    """
    low = name.lower()
    healing = [m for m in predicted_moveset if m.heal > 0]
    boosting = [m for m in predicted_moveset if any(b > 0 for b in m.boosts) and m.self_boosts]
    status_moves = [m for m in predicted_moveset if m.status != Status.NONE and m.base_power == 0]
    damaging = [m for m in predicted_moveset if m.base_power > 0]
    screens = [m for m in predicted_moveset if m.toggle_reflect or m.toggle_lightscreen]
    hazards = [m for m in predicted_moveset if m.hazard is not None]

    if "wall" in low:
        ordered = healing + screens + hazards + status_moves + boosting + damaging
    elif "sweeper" in low:
        ordered = damaging + boosting + status_moves + healing
    else:
        ordered = damaging + boosting + healing + status_moves + screens + hazards

    seen = set()
    result = []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            result.append(m)
        if len(result) >= 4:
            break
    return result[:4]


def create_archetype_builds(species: PokemonSpecies, predicted_moveset: list[Any]) -> list[tuple[str, Pokemon]]:
    """Generate a list of competitive builds for a species, each tagged with an archetype name.

    Produces up to 10 builds: fast sweeper (physical/special), bulky attacker
    (physical/special), defensive walls (physical/special), and mixed attackers
    if the species has balanced offensive stats. Each archetype gets a tailored
    move selection (walls prefer healing/support, sweepers prefer damaging moves).

    Args:
        species: The Pokemon species to build for.
        predicted_moveset: List of Move objects the Pokemon will use.

    Returns:
        List of (archetype_name, Pokemon) tuples. Empty if no predicted moveset.
    """
    if not predicted_moveset:
        return []

    builds: list[tuple[str, Pokemon]] = []
    base_stats = species.base_stats
    ivs = (31, 31, 31, 31, 31, 31)
    is_phys = base_stats[Stat.ATTACK] >= base_stats[Stat.SPECIAL_ATTACK]
    lv = 100

    def _make(name: str, evs: tuple[int, ...], nature: int) -> None:
        moves = _archetype_moves(name, predicted_moveset, species)
        indices = []
        for move in moves:
            try:
                idx = species.moves.index(move)
                indices.append(idx)
            except ValueError:
                continue
        if not indices:
            indices = list(range(min(4, len(species.moves))))
        builds.append((name, Pokemon(species, indices, lv, evs, ivs, nature)))

    _make("Fast Physical Sweeper", (4, 252, 0, 0, 0, 252), Nature.JOLLY)
    _make("The Dragon Dance Setup Sweeper", (0, 252, 4, 0, 0, 252), Nature.ADAMANT)
    _make("Fast Special Sweeper", (4, 0, 0, 252, 0, 252), Nature.TIMID)
    _make("Speed Physical Offense", (6, 252, 0, 0, 0, 252), Nature.ADAMANT)
    _make("Speed Special Offense", (6, 0, 0, 252, 0, 252), Nature.MODEST)
    _make("Bulky Physical Attacker", (252, 252, 4, 0, 0, 0), Nature.ADAMANT)
    _make("HP Physical Attacker", (252, 168, 0, 84, 0, 6), Nature.ADAMANT)
    _make("Bulky Special Attacker", (252, 0, 4, 252, 0, 0), Nature.MODEST)
    _make("HP Special Attacker", (252, 84, 0, 168, 0, 6), Nature.MODEST)
    _make("Physically Defensive Wall", (252, 0, 252, 0, 4, 0), Nature.IMPISH if is_phys else Nature.BOLD)
    _make("Specially Defensive Wall", (252, 0, 4, 0, 252, 0), Nature.CAREFUL if is_phys else Nature.CALM)

    if abs(base_stats[Stat.ATTACK] - base_stats[Stat.SPECIAL_ATTACK]) <= 20:
        _make("Fast Mixed Attacker", (0, 252, 0, 4, 0, 252), Nature.NAIVE)
        _make("Fast Mixed Attacker", (0, 4, 0, 252, 0, 252), Nature.HASTY)
        _make("Bulky Mixed Attacker", (252, 252, 0, 4, 0, 0), Nature.NAUGHTY)
        _make("Bulky Mixed Attacker", (252, 4, 0, 252, 0, 0), Nature.RASH)

    return builds


def create_generic_build_for_species(species: PokemonSpecies) -> Pokemon | None:
    """Create a single generic 'best attacker' build for a species.

    Used as a fallback opponent build during move evaluation when
    the optimal build cache is not yet available. Neutral nature,
    balanced EVs.

    Args:
        species: The Pokemon species to build.

    Returns:
        A Pokemon instance, or None if the species has no moves.
    """
    if not species.moves:
        return None

    default_evs = (85, 85, 85, 85, 85, 85)
    default_ivs = (31, 31, 31, 31, 31, 31)
    num_moves = min(4, len(species.moves))
    move_indices = list(range(num_moves))

    return Pokemon(
        species=species,
        move_indexes=move_indices,
        level=100,
        evs=default_evs,
        ivs=default_ivs,
        nature=Nature.SERIOUS,
    )
