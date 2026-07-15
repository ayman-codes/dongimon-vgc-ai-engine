"""String-based type effectiveness chart and lookup functions for Pokémon VGC.

Provides a 19×19 type chart keyed by lowercase type names. All type matching
across the Dongimon project uses this module as the single source of truth,
replacing the enum-based `vgc2.battle_engine.modifiers.Type` lookups.

Attributes:
    TYPE_CHART: Nested dict mapping attacking type names to defending type
        effectiveness multipliers (0.0, 0.5, 1.0, 2.0, 4.0).
    TYPE_NAMES: Ordered list of 19 type names matching the vgc2 Type enum indices.
    VGC2_TYPE_INDEX: Reverse mapping from lowercase type name to vgc2 Type int value.
"""

TYPE_NAMES: list[str] = [
    "normal",       # 0 = NORMAL
    "fire",         # 1 = FIRE
    "water",        # 2 = WATER
    "electric",     # 3 = ELECTRIC
    "grass",        # 4 = GRASS
    "ice",          # 5 = ICE
    "fighting",     # 6 = FIGHT
    "poison",       # 7 = POISON
    "ground",       # 8 = GROUND
    "flying",       # 9 = FLYING
    "psychic",      # 10 = PSYCHIC
    "bug",          # 11 = BUG
    "rock",         # 12 = ROCK
    "ghost",        # 13 = GHOST
    "dragon",       # 14 = DRAGON
    "dark",         # 15 = DARK
    "steel",        # 16 = STEEL
    "fairy",        # 17 = FAIRY
    "typeless",     # 18 = TYPELESS
]

VGC2_ENUM_MAP: dict[str, int] = {name: idx for idx, name in enumerate(TYPE_NAMES)}

VGC2_TYPE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(TYPE_NAMES)}

TYPE_CHART: dict[str, dict[str, float]] = {
    "normal": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 0.5, "bug": 1.0, "ghost": 0.0, "steel": 0.5, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "fighting": {
        "normal": 2.0, "fighting": 1.0, "flying": 0.5, "poison": 0.5, "ground": 1.0,
        "rock": 2.0, "bug": 0.5, "ghost": 0.0, "steel": 2.0, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 0.5, "ice": 2.0,
        "dragon": 1.0, "dark": 2.0, "fairy": 0.5, "typeless": 1.0,
    },
    "flying": {
        "normal": 1.0, "fighting": 2.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 0.5, "bug": 2.0, "ghost": 1.0, "steel": 0.5, "fire": 1.0,
        "water": 1.0, "grass": 2.0, "electric": 0.5, "psychic": 1.0, "ice": 1.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "poison": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 0.5, "ground": 0.5,
        "rock": 0.5, "bug": 1.0, "ghost": 0.5, "steel": 0.0, "fire": 1.0,
        "water": 1.0, "grass": 2.0, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 2.0, "typeless": 1.0,
    },
    "ground": {
        "normal": 1.0, "fighting": 1.0, "flying": 0.0, "poison": 2.0, "ground": 1.0,
        "rock": 2.0, "bug": 0.5, "ghost": 1.0, "steel": 2.0, "fire": 2.0,
        "water": 1.0, "grass": 0.5, "electric": 2.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "rock": {
        "normal": 1.0, "fighting": 0.5, "flying": 2.0, "poison": 1.0, "ground": 0.5,
        "rock": 1.0, "bug": 2.0, "ghost": 1.0, "steel": 0.5, "fire": 2.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 1.0, "ice": 2.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "bug": {
        "normal": 1.0, "fighting": 0.5, "flying": 0.5, "poison": 0.5, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 0.5, "steel": 0.5, "fire": 0.5,
        "water": 1.0, "grass": 2.0, "electric": 1.0, "psychic": 2.0, "ice": 1.0,
        "dragon": 1.0, "dark": 2.0, "fairy": 0.5, "typeless": 1.0,
    },
    "ghost": {
        "normal": 0.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 2.0, "steel": 1.0, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 2.0, "ice": 1.0,
        "dragon": 1.0, "dark": 0.5, "fairy": 1.0, "typeless": 1.0,
    },
    "steel": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 2.0, "bug": 1.0, "ghost": 1.0, "steel": 0.5, "fire": 0.5,
        "water": 0.5, "grass": 1.0, "electric": 0.5, "psychic": 1.0, "ice": 2.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 2.0, "typeless": 1.0,
    },
    "fire": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 0.5, "bug": 2.0, "ghost": 1.0, "steel": 2.0, "fire": 0.5,
        "water": 0.5, "grass": 2.0, "electric": 1.0, "psychic": 1.0, "ice": 2.0,
        "dragon": 0.5, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "water": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 2.0,
        "rock": 2.0, "bug": 1.0, "ghost": 1.0, "steel": 1.0, "fire": 2.0,
        "water": 0.5, "grass": 0.5, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 0.5, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "grass": {
        "normal": 1.0, "fighting": 1.0, "flying": 0.5, "poison": 0.5, "ground": 2.0,
        "rock": 2.0, "bug": 0.5, "ghost": 1.0, "steel": 0.5, "fire": 0.5,
        "water": 2.0, "grass": 0.5, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 0.5, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "electric": {
        "normal": 1.0, "fighting": 1.0, "flying": 2.0, "poison": 1.0, "ground": 0.0,
        "rock": 1.0, "bug": 1.0, "ghost": 1.0, "steel": 1.0, "fire": 1.0,
        "water": 2.0, "grass": 0.5, "electric": 0.5, "psychic": 1.0, "ice": 1.0,
        "dragon": 0.5, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "psychic": {
        "normal": 1.0, "fighting": 2.0, "flying": 1.0, "poison": 2.0, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 1.0, "steel": 0.5, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 0.5, "ice": 1.0,
        "dragon": 1.0, "dark": 0.0, "fairy": 1.0, "typeless": 1.0,
    },
    "ice": {
        "normal": 1.0, "fighting": 1.0, "flying": 2.0, "poison": 1.0, "ground": 2.0,
        "rock": 1.0, "bug": 1.0, "ghost": 1.0, "steel": 0.5, "fire": 0.5,
        "water": 0.5, "grass": 2.0, "electric": 1.0, "psychic": 1.0, "ice": 0.5,
        "dragon": 2.0, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
    "dragon": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 1.0, "steel": 0.5, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 2.0, "dark": 1.0, "fairy": 0.0, "typeless": 1.0,
    },
    "dark": {
        "normal": 1.0, "fighting": 0.5, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 2.0, "steel": 1.0, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 2.0, "ice": 1.0,
        "dragon": 1.0, "dark": 0.5, "fairy": 0.5, "typeless": 1.0,
    },
    "fairy": {
        "normal": 1.0, "fighting": 2.0, "flying": 1.0, "poison": 0.5, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 1.0, "steel": 0.5, "fire": 0.5,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 2.0, "dark": 2.0, "fairy": 1.0, "typeless": 1.0,
    },
    "typeless": {
        "normal": 1.0, "fighting": 1.0, "flying": 1.0, "poison": 1.0, "ground": 1.0,
        "rock": 1.0, "bug": 1.0, "ghost": 1.0, "steel": 1.0, "fire": 1.0,
        "water": 1.0, "grass": 1.0, "electric": 1.0, "psychic": 1.0, "ice": 1.0,
        "dragon": 1.0, "dark": 1.0, "fairy": 1.0, "typeless": 1.0,
    },
}


def vgc2_type_to_name(vgc2_type_value: int) -> str:
    """Convert a vgc2 Type enum integer value to its lowercase string name.

    Args:
        vgc2_type_value: Integer index from vgc2.battle_engine.modifiers.Type.

    Returns:
        Lowercase type name string.
    """
    if 0 <= vgc2_type_value < len(TYPE_NAMES):
        return TYPE_NAMES[vgc2_type_value]
    return "typeless"


def type_effectiveness(move_type: str, defender_types: list[str]) -> float:
    """Calculate the combined type effectiveness multiplier for a move.

    Multiplies effectiveness against each of the defender's types,
    producing values like 0.0 (immune), 0.25 (quad-resisted),
    1.0 (neutral), 2.0 (super-effective), or 4.0 (quad-weak).

    Args:
        move_type: Lowercase attacking type name (e.g. "fire").
        defender_types: List of lowercase defending type names.

    Returns:
        Combined effectiveness multiplier as a float.
    """
    effectiveness = 1.0
    chart_row = TYPE_CHART.get(move_type, {})
    for def_type in defender_types:
        effectiveness *= chart_row.get(def_type, 1.0)
    return effectiveness


def is_immune(move_type: str, defender_types: list[str]) -> bool:
    """Check if a defender is immune to a given attacking type.

    Args:
        move_type: Lowercase attacking type name.
        defender_types: List of lowercase defending type names.

    Returns:
        True if any defending type grants immunity (0.0x) to the move type.
    """
    return type_effectiveness(move_type, defender_types) == 0.0
