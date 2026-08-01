"""XGBoost behavior-cloning battle policy wrapper.

Extracted from ``scripts/benchmark/benchmark_battle.py`` into a reusable
``BattlePolicy`` implementation used as a benchmark sparring partner.
"""

from pathlib import Path

import joblib
import numpy as np
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleCommand
from vgc2.battle_engine.view import StateView, TeamView

from src.tree_bc.actions import decode_action, get_valid_actions
from src.tree_bc.encoder import encode_state


class TreeBCBattlePolicy(BattlePolicy):  # type: ignore[misc]  # vgc2.agent is untyped (Any)
    """XGBoost behavior-cloning policy with valid-action masking.

    Loads the trained XGBoost bundle (model + label inverse map) and
    predicts the joint action index (0-99) with probability mass masked
    to the valid action set for the current state.
    """

    def __init__(self, model_path: Path | str):
        """Initialize the policy.

        Args:
            model_path: Path to the joblib bundle containing the
                ``model`` and ``inverse_map`` keys.
        """
        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._inverse_map = bundle["inverse_map"]

    def decision(self, state: StateView, opp_view: TeamView | None) -> list[BattleCommand]:
        """Predict joint action from encoded state with masking.

        Args:
            state: Current battle state view (side 0 = own team).
            opp_view: Opponent team view (unused, kept for interface).

        Returns:
            List of BattleCommand tuples, or two default commands when no
            valid action exists.
        """
        valid = get_valid_actions(state)
        if not valid:
            return [(0, 0), (0, 0)]

        obs = encode_state(state).reshape(1, -1)
        raw_proba = self._model.predict_proba(obs)[0]
        n_classes = len(self._inverse_map)
        full_proba = np.zeros(100, dtype=np.float64)
        for new_idx in range(min(n_classes, len(raw_proba))):
            full_proba[self._inverse_map[new_idx]] = raw_proba[new_idx]

        mask = np.zeros(100, dtype=bool)
        mask[valid] = True
        full_proba[~mask] = 0.0

        return decode_action(int(full_proba.argmax()))
