"""Shared test fixtures for the Dongimon test suite."""

import pytest


@pytest.fixture
def sample_types() -> dict[str, list[str]]:
    """Return sample type configurations for testing."""
    return {
        "fire": ["fire"],
        "water": ["water"],
        "grass": ["grass"],
        "electric": ["electric"],
        "normal": ["normal"],
        "fire_flying": ["fire", "flying"],
        "water_ground": ["water", "ground"],
        "steel_fairy": ["steel", "fairy"],
        "dragon_flying": ["dragon", "flying"],
        "ghost_dark": ["ghost", "dark"],
    }
