"""Tests for selection policy prediction and sub-tournament modules."""

from unittest.mock import MagicMock

from vgc2.battle_engine.modifiers import Stat, Type
from vgc2.battle_engine.pokemon import PokemonSpecies
from vgc2.battle_engine.team import Team

from src.selection.tournament import generate_team_combinations


def _make_dummy_species(name: str, types=None, atk=80, spa=80):
    if types is None:
        types = [Type.NORMAL]
    return PokemonSpecies(
        name=name,
        base_stats=(100, atk, 70, spa, 70, 80),
        types=types,
        moves=[],
    )


def _make_dummy_team(*species_names):
    """Create a Team with minimal Pokemon members from species names."""
    team = MagicMock(spec=Team)
    members = []
    for name in species_names:
        species = _make_dummy_species(name)
        pkm = MagicMock()
        pkm.species = species
        pkm.stats = {Stat.MAX_HP: 100, Stat.ATTACK: 80, Stat.SPECIAL_ATTACK: 80, Stat.SPEED: 80}
        pkm.moves = []
        members.append(pkm)
    team.members = members
    return team


class TestGenerateTeamCombinations:
    """Tests for generate_team_combinations."""

    def test_returns_all_combinations(self):
        """C(4,2) = 6 combinations from a 4-member team."""
        team = _make_dummy_team("A", "B", "C", "D")
        combos = generate_team_combinations(team, 2)
        assert len(combos) == 6

    def test_returns_tuples_of_indices(self):
        """Each combination is a tuple of int indices."""
        team = _make_dummy_team("A", "B", "C")
        combos = generate_team_combinations(team, 2)
        for combo in combos:
            assert isinstance(combo, tuple)
            for idx in combo:
                assert isinstance(idx, int)

    def test_empty_when_team_too_small(self):
        """Combination larger than team size returns empty."""
        team = _make_dummy_team("A", "B")
        combos = generate_team_combinations(team, 3)
        assert combos == []

    def test_single_member_combination(self):
        """C(3,1) = 3 combinations."""
        team = _make_dummy_team("A", "B", "C")
        combos = generate_team_combinations(team, 1)
        assert len(combos) == 3
        assert all(len(c) == 1 for c in combos)

    def test_six_member_team_C6_2(self):
        """C(6,2) = 15 combinations."""
        team = _make_dummy_team("A", "B", "C", "D", "E", "F")
        combos = generate_team_combinations(team, 2)
        assert len(combos) == 15
        assert all(len(c) == 2 for c in combos)
        # All combinations should be unique
        assert len(set(combos)) == 15

    def test_no_duplicates(self):
        """No duplicate indices within a combination."""
        team = _make_dummy_team("A", "B", "C")
        combos = generate_team_combinations(team, 2)
        for combo in combos:
            assert len(set(combo)) == len(combo)
