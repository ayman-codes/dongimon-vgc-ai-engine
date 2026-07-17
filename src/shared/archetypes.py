"""Archetype build generation for team preview and team building.

Produces fully-formed Pokemon objects with competitive EV spreads,
natures, and move sets. Used by both Selection and Teambuild policies.
"""

from typing import Any

from vgc2.battle_engine.modifiers import Nature, Stat
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies


def create_archetype_builds(species: PokemonSpecies, predicted_moveset: list[Any]) -> list[tuple[str, Pokemon]]:
    """Generate a list of competitive builds for a species, each tagged with an archetype name.

    Produces up to 10 builds: fast sweeper (physical/special), bulky attacker
    (physical/special), defensive walls (physical/special), and mixed attackers
    if the species has balanced offensive stats.

    Args:
        species: The Pokemon species to build for.
        predicted_moveset: List of Move objects the Pokemon will use.

    Returns:
        List of (archetype_name, Pokemon) tuples. Empty if no predicted moveset.
    """
    if not predicted_moveset:
        return []

    move_indices = []
    for move in predicted_moveset:
        try:
            idx = species.moves.index(move)
            move_indices.append(idx)
        except ValueError:
            continue
    if not move_indices:
        return []

    builds: list[tuple[str, Pokemon]] = []
    base_stats = species.base_stats
    ivs = (31, 31, 31, 31, 31, 31)
    is_phys = base_stats[Stat.ATTACK] >= base_stats[Stat.SPECIAL_ATTACK]
    lv = 50

    def _make(name: str, evs: tuple[int, ...], nature: int) -> None:
        builds.append((name, Pokemon(species, move_indices, lv, evs, ivs, nature)))

    _make("Fast Physical Sweeper", (4, 252, 0, 0, 0, 252), Nature.JOLLY)
    _make("Fast Special Sweeper", (4, 0, 0, 252, 0, 252), Nature.TIMID)
    _make("Bulky Physical Attacker", (252, 252, 4, 0, 0, 0), Nature.ADAMANT)
    _make("Bulky Special Attacker", (252, 0, 4, 252, 0, 0), Nature.MODEST)
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
        level=50,
        evs=default_evs,
        ivs=default_ivs,
        nature=Nature.SERIOUS,
    )
