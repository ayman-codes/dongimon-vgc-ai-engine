"""Feature computation for Matchup Predictor training and inference.

Provides per-subteam feature extraction (53 features) and pairwise
delta features (56 features) used by both the data generation pipeline
and the future MP-based selection pre-filter.

Feature schema (per subteam, 53 total):
    7 stats x 4 aggregations (avg, max, min, std) = 28
    Speed brackets (4): spd_b0_50, spd_b51_80, spd_b81_110, spd_b111_p
    Type coverage SE vector (18): type_cov_{name}_se
    Weakness count (1): weakness_count
    Move BP (2): avg_move_bp, max_move_bp

Pairwise features (56 total):
    53 deltas (A - B) + type_advantage_net + type_adv_a + type_adv_b
"""

from typing import Any

import numpy as np

from src.shared.types import type_effectiveness, vgc2_type_to_name

N_TYPES = 18

TYPE_NAMES = [
    "normal", "fire", "water", "electric", "grass",
    "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon",
    "dark", "steel", "fairy",
]

BST_FEATURE_NAMES = ["bst_avg_diff", "bst_max_diff", "bst_min_diff", "bst_std_diff"]


def compute_subteam_features(members: list[Any]) -> dict[str, float]:
    """Compute 53 features on the 4 Pokemon that fought.

    Features (in insertion order):
        7 stats x 4 aggregations - avg, max, min, std (28):
            bst_avg, bst_max, bst_min, bst_std,
            hp_avg, hp_max, hp_min, hp_std,
            atk_avg, ..., spe_std
        Speed brackets (4): spd_b0_50, spd_b51_80, spd_b81_110, spd_b111_p
        Type coverage SE vector (18): type_cov_{name}_se per type
        Weakness count (1): weakness_count
        Move BP (2): avg_move_bp, max_move_bp

    BST-only ablation indices: [0, 1, 2, 3] corresponding to
    bst_avg, bst_max, bst_min, bst_std.

    Args:
        members: List of 4 Pokemon objects.

    Returns:
        Dict of feature_name -> float value.
    """
    feat: dict[str, float] = {}

    bst_list: list[int] = []
    hp_list: list[int] = []
    atk_list: list[int] = []
    def_list: list[int] = []
    spa_list: list[int] = []
    spd_list: list[int] = []
    spe_list: list[int] = []

    for pkm in members:
        if hasattr(pkm, "species"):
            base = pkm.species.base_stats
        elif hasattr(pkm, "constants"):
            base = pkm.constants.base
        else:
            base = (0,) * 8

        bst = int(sum(base))
        bst_list.append(bst)
        hp_list.append(int(base[0]))
        atk_list.append(int(base[1]))
        def_list.append(int(base[2]))
        spa_list.append(int(base[3]))
        spd_list.append(int(base[4]))
        spe_list.append(int(base[5]))

    for label, lst in [
        ("bst", bst_list), ("hp", hp_list), ("atk", atk_list),
        ("def", def_list), ("spa", spa_list), ("spd", spd_list),
        ("spe", spe_list),
    ]:
        vals = lst if lst else [0]
        feat[f"{label}_avg"] = float(np.mean(vals))
        feat[f"{label}_max"] = float(np.max(vals))
        feat[f"{label}_min"] = float(np.min(vals))
        feat[f"{label}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    speed_brackets = {"spd_b0_50": 0, "spd_b51_80": 0, "spd_b81_110": 0, "spd_b111_p": 0}
    for s in spe_list:
        if s <= 50:
            speed_brackets["spd_b0_50"] += 1
        elif s <= 80:
            speed_brackets["spd_b51_80"] += 1
        elif s <= 110:
            speed_brackets["spd_b81_110"] += 1
        else:
            speed_brackets["spd_b111_p"] += 1
    for bracket, count in speed_brackets.items():
        feat[bracket] = float(count)

    type_se_coverage = [0] * N_TYPES
    for pkm in members:
        covered_by_pkm: set[int] = set()
        moves = pkm.moves if hasattr(pkm, "moves") else []
        for move in moves:
            if not hasattr(move, "base_power") or move.base_power <= 0:
                continue
            atk_type = int(move.pkm_type) if hasattr(move.pkm_type, "value") else int(move.pkm_type)
            if atk_type < 0 or atk_type >= N_TYPES:
                continue
            atk_name = vgc2_type_to_name(atk_type)
            for def_type in range(N_TYPES):
                def_name = vgc2_type_to_name(def_type)
                eff = type_effectiveness(atk_name, [def_name])
                if eff > 1.0:
                    covered_by_pkm.add(def_type)
        for def_type in covered_by_pkm:
            type_se_coverage[def_type] += 1

    for t_idx, count in enumerate(type_se_coverage):
        feat[f"type_cov_{TYPE_NAMES[t_idx]}_se"] = float(count)

    total_weaknesses = 0.0
    member_type_lists: list[list[str]] = []
    for pkm in members:
        spec_obj = pkm.species if hasattr(pkm, "species") else pkm
        tl = [vgc2_type_to_name(int(t)) for t in spec_obj.types]
        member_type_lists.append(tl)

    for tl in member_type_lists:
        for atk_name in TYPE_NAMES:
            eff = type_effectiveness(atk_name, tl)
            if eff > 1.0:
                total_weaknesses += 1.0
    feat["weakness_count"] = total_weaknesses

    move_bp_sum = 0
    move_count = 0
    max_bp = 0
    for pkm in members:
        moves = pkm.moves if hasattr(pkm, "moves") else []
        for move in moves:
            if hasattr(move, "base_power") and move.base_power > 0:
                bp = move.base_power
                move_bp_sum += bp
                move_count += 1
                if bp > max_bp:
                    max_bp = bp
    feat["avg_move_bp"] = float(move_bp_sum / max(move_count, 1)) if move_count > 0 else 0.0
    feat["max_move_bp"] = float(max_bp)

    return feat


def count_se_moves_against(subteam_a: list[Any], subteam_b: list[Any]) -> float:
    """Count super-effective offensive types A has against B's defensive types.

    Higher = A has better type matchup against B.

    Args:
        subteam_a: List of 4 Pokemon (attackers).
        subteam_b: List of 4 Pokemon (defenders).

    Returns:
        Number of SE hits A has against B's collected types.
    """
    defender_types: list[list[str]] = []
    for pkm in subteam_b:
        spec_obj = pkm.species if hasattr(pkm, "species") else pkm
        tl = [vgc2_type_to_name(int(t)) for t in spec_obj.types]
        defender_types.append(tl)

    count = 0.0
    seen: set[tuple[int, int]] = set()
    for pkm in subteam_a:
        moves = pkm.moves if hasattr(pkm, "moves") else []
        for move in moves:
            if not hasattr(move, "base_power") or move.base_power <= 0:
                continue
            atk_type = int(move.pkm_type) if hasattr(move.pkm_type, "value") else int(move.pkm_type)
            if atk_type < 0 or atk_type >= N_TYPES:
                continue
            atk_name = vgc2_type_to_name(atk_type)
            for d_idx, dtl in enumerate(defender_types):
                if (atk_type, d_idx) in seen:
                    continue
                eff = type_effectiveness(atk_name, dtl)
                if eff > 1.0:
                    count += 1.0
                    seen.add((atk_type, d_idx))
    return count


def compute_pairwise_features(
    subteam_a: list[Any],
    subteam_b: list[Any],
) -> dict[str, float]:
    """Compute pairwise delta features between two subteams.

    Computes per-team features and delta (A - B) for each, plus
    type advantage metrics. Total: 56 features.

    Args:
        subteam_a: List of 4 Pokemon (side A).
        subteam_b: List of 4 Pokemon (side B).

    Returns:
        Dict of pairwise feature_name -> float value.
    """
    feat_a = compute_subteam_features(subteam_a)
    feat_b = compute_subteam_features(subteam_b)

    deltas: dict[str, float] = {}
    for key in feat_a:
        val_a = feat_a.get(key, 0.0)
        val_b = feat_b.get(key, 0.0)
        deltas[f"{key}_diff"] = val_a - val_b

    type_adv_a = count_se_moves_against(subteam_a, subteam_b)
    type_adv_b = count_se_moves_against(subteam_b, subteam_a)
    deltas["type_advantage_net"] = type_adv_a - type_adv_b
    deltas["type_adv_a"] = type_adv_a
    deltas["type_adv_b"] = type_adv_b

    return deltas
