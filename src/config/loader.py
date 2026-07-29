"""Configuration loader for YAML-based policy weights."""

from pathlib import Path

import yaml

from src.config.models import (
    BattleWeights,
    SelectionConfig,
    SelectionSynergyWeights,
    TeambuildConfig,
    TeambuildWeights,
)


def load_battle_weights(path: Path | None = None) -> BattleWeights:
    """Load battle policy weights from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to `battle_weights.yaml`
            in the config directory.

    Returns:
        Validated BattleWeights instance.
    """
    if path is None:
        path = Path(__file__).parent / "battle_weights.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return BattleWeights(**data)


def selection_config() -> SelectionConfig:
    """Return the default SelectionConfig.

    Returns:
        SelectionConfig with default values.
    """
    return SelectionConfig()


def teambuild_config() -> TeambuildConfig:
    """Return the default TeambuildConfig.

    Returns:
        TeambuildConfig with default values.
    """
    return TeambuildConfig()


def load_selection_synergy(path: Path | None = None) -> SelectionSynergyWeights:
    """Load selection pair-synergy weights from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to ``selection_synergy.yaml``
            in the config directory.

    Returns:
        Validated SelectionSynergyWeights instance.
    """
    if path is None:
        path = Path(__file__).parent / "selection_synergy.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SelectionSynergyWeights(**data)


def load_teambuild_weights(path: Path | None = None) -> TeambuildWeights:
    """Load teambuild weights from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to ``teambuild_weights.yaml``
            in the config directory.

    Returns:
        Validated TeambuildWeights instance.
    """
    if path is None:
        path = Path(__file__).parent / "teambuild_weights.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TeambuildWeights(**data)
