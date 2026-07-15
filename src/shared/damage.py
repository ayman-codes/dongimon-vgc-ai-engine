"""Gen 9 Pokémon damage formula and related combat math.

Provides standalone damage calculation functions used by all three policy
engines (Battle, Selection, Teambuild). All functions are pure — they accept
numeric inputs and return numeric outputs with no vgc2 dependency.

Core formula: ((2 * Level / 5 + 2) * Power * A / D / 50 + 2) * Modifier
"""


def calculate_damage(
    level: int = 50,
    power: int = 80,
    attack: int = 100,
    defense: int = 100,
    modifier: float = 1.0,
) -> int:
    """Calculate base damage before the final modifier is applied.

    Implements the standard Gen 9 damage formula:
        damage = floor(floor((2 * L / 5 + 2) * Power * A / D) / 50 + 2) * Modifier

    Args:
        level: Attacker level (default 50 for VGC).
        power: Move base power.
        attack: Attacker's effective Attack or Special Attack stat.
        defense: Defender's effective Defense or Special Defense stat.
        modifier: Combined damage modifier (type effectiveness, STAB, weather,
            terrain, burn, screens, crit, random factor).

    Returns:
        Final damage as an integer (minimum 1 if modifier > 0).
    """
    if power <= 0 or defense <= 0:
        return 0
    base = int((2 * level / 5 + 2) * power * attack / defense)
    damage = int(base / 50 + 2)
    result = max(1, int(damage * modifier)) if modifier > 0 else 0
    return result


def stab_modifier(move_type: str, attacker_types: list[str]) -> float:
    """Compute the Same-Type Attack Bonus multiplier.

    Args:
        move_type: Lowercase attacking type name.
        attacker_types: List of lowercase type names of the attacker.

    Returns:
        1.5 if the move type matches any of the attacker's types, else 1.0.
    """
    return 1.5 if move_type in attacker_types else 1.0


def type_modifier(move_type: str, defender_types: list[str]) -> float:
    """Alias for type_effectiveness from the types module.

    Args:
        move_type: Lowercase attacking type name.
        defender_types: List of lowercase defending type names.

    Returns:
        Combined type effectiveness multiplier.
    """
    from src.shared.types import type_effectiveness

    return type_effectiveness(move_type, defender_types)


def weather_boost_multiplier(move_type: str, active_weather: str) -> float:
    """Compute the weather damage modifier for a move.

    Args:
        move_type: Lowercase attacking type name.
        active_weather: Lowercase weather name ("clear", "rain", "sun", "sand", "snow").

    Returns:
        1.5 if weather boosts the move, 0.5 if it weakens, else 1.0.
    """
    if active_weather == "rain":
        if move_type == "water":
            return 1.5
        if move_type == "fire":
            return 0.5
    elif active_weather == "sun":
        if move_type == "fire":
            return 1.5
        if move_type == "water":
            return 0.5
    return 1.0


def terrain_boost_multiplier(move_type: str, active_terrain: str, is_grounded: bool) -> float:
    """Compute the terrain damage modifier for a move.

    Args:
        move_type: Lowercase attacking type name.
        active_terrain: Lowercase terrain name ("none", "electric_terrain",
            "grassy_terrain", "psychic_terrain", "misty_terrain").
        is_grounded: Whether the attacker is touching the ground.

    Returns:
        Terrain boost multiplier (1.3 for matching terrain, else 1.0).
    """
    if not is_grounded:
        return 1.0
    terrain_map = {
        "electric_terrain": "electric",
        "grassy_terrain": "grass",
        "psychic_terrain": "psychic",
        "misty_terrain": "dragon",
    }
    boosted_type = terrain_map.get(active_terrain)
    if boosted_type is None:
        return 1.0
    if active_terrain == "misty_terrain":
        return 0.5
    return 1.3 if move_type == boosted_type else 1.0
