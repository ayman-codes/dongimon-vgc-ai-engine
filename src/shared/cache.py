"""LRU-cached wrapper around vgc2's calculate_damage for memoizing repeated evaluations.

Prediction, scoring, and moveset modules evaluate the same (species, move, defender)
triples across iterative loops. This module wraps the engine's damage formula with
an LRU eviction cache keyed on the deterministic attributes that affect damage output.
"""

from collections import OrderedDict
from typing import Any

from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine import calculate_damage as _vgc2_damage
from vgc2.battle_engine.modifiers import Stat

_CACHE_MAXSIZE = 200000
_cache: OrderedDict[tuple[Any, ...], int] = OrderedDict()
_hits: int = 0
_misses: int = 0


def _safe_spec_name(pkm: Any) -> Any:
    """Extract species name from a Pokemon or BattlingPokemon object.

    Args:
        pkm: A Pokemon, BattlingPokemon, or BattlingPokemonView.

    Returns:
        Species name string, or empty string if not resolvable.
    """
    for attr in ("species", "constants"):
        obj = getattr(pkm, attr, None)
        if obj and hasattr(obj, "species") and obj.species and hasattr(obj.species, "name"):
            return obj.species.name
    obj = getattr(pkm, "species", None)
    if obj and hasattr(obj, "name"):
        return obj.name
    return ""


def _safe_stat(pkm: Any, stat: Stat) -> Any:
    """Extract a specific stat value from a Pokemon or BattlingPokemon.

    Args:
        pkm: A Pokemon, BattlingPokemon, or BattlingPokemonView.
        stat: The Stat constant (integer index) to retrieve.

    Returns:
        Integer stat value, or 0 if not resolvable.
    """
    for attr in ("constants",):
        obj = getattr(pkm, attr, None)
        if obj and hasattr(obj, "stats"):
            stats_obj = obj.stats
            if isinstance(stats_obj, dict):
                return int(stats_obj.get(stat, 0))
            if isinstance(stats_obj, (tuple, list)) and isinstance(stat, int) and stat < len(stats_obj):
                return int(stats_obj[stat])
    if hasattr(pkm, "stats"):
        stats_obj = pkm.stats
        if isinstance(stats_obj, dict):
            return int(stats_obj.get(stat, 0))
        if isinstance(stats_obj, (tuple, list)) and isinstance(stat, int) and stat < len(stats_obj):
            return int(stats_obj[stat])
    return 0


def _safe_move_name(move: Any) -> Any:
    """Extract move name string.

    Args:
        move: A Move or Move constants object.

    Returns:
        Move name string, or repr fallback.
    """
    if hasattr(move, "name"):
        return move.name
    return str(move)


def _make_key(attacker: Any, defender: Any, move: Any, state: Any, attacking_side: int) -> tuple[Any, ...]:
    """Build a hashable key from the arguments to calculate_damage.

    Args:
        attacker: BattlingPokemon attacking.
        defender: BattlingPokemon defending.
        move: Move object being used.
        state: Current battle State.
        attacking_side: 0 or 1.

    Returns:
        Tuple of primitives suitable for dictionary keying.
    """
    return (
        _safe_spec_name(attacker),
        _safe_stat(attacker, Stat.ATTACK),
        _safe_stat(attacker, Stat.SPECIAL_ATTACK),
        _safe_spec_name(defender),
        _safe_stat(defender, Stat.DEFENSE),
        _safe_stat(defender, Stat.SPECIAL_DEFENSE),
        _safe_move_name(move),
        int(state.weather) if hasattr(state, "weather") else 0,
        int(state.field) if hasattr(state, "field") else 0,
        attacking_side,
    )


def cached_calculate_damage(
    params: BattleRuleParam,
    attacking_side: int,
    move: Any,
    state: Any,
    attacker: Any,
    defender: Any,
) -> Any:
    """Memoized wrapper around vgc2.battle_engine.calculate_damage.

    Caches up to 200,000 entries with LRU eviction. Call with the exact
    same signature as the original vgc2 function.

    Args:
        params: Battle rule parameters.
        attacking_side: 0 or 1.
        move: Move constants or Move object.
        state: Current State object.
        attacker: BattlingPokemon attacking.
        defender: BattlingPokemon defending.

    Returns:
        Integer damage value.
    """
    global _hits, _misses
    key = _make_key(attacker, defender, move, state, attacking_side)
    if key in _cache:
        _hits += 1
        _cache.move_to_end(key)
        return _cache[key]
    _misses += 1
    result = _vgc2_damage(
        params=params,
        attacking_side=attacking_side,
        move=move,
        state=state,
        attacker=attacker,
        defender=defender,
    )
    if len(_cache) >= _CACHE_MAXSIZE:
        _cache.popitem(last=False)
    _cache[key] = result
    return result


def cache_stats() -> dict[str, int]:
    """Return current cache hit/miss/size statistics.

    Returns:
        Dict with keys hits, misses, size.
    """
    return {"hits": _hits, "misses": _misses, "size": len(_cache)}
