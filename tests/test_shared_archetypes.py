"""Tests for shared archetype build generation."""

from vgc2.battle_engine.modifiers import Nature, Stat, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies

from src.shared.archetypes import create_archetype_builds, create_generic_build_for_species


def _make_dummy_species(name: str, atk: int = 80, spa: int = 80, spe: int = 80,
                        types: list | None = None, moves: list | None = None) -> PokemonSpecies:
    """Create a minimal PokemonSpecies for testing."""
    if types is None:
        types = [Type.NORMAL]
    if moves is None:
        moves = []
    sp = PokemonSpecies(
        name=name,
        base_stats=(100, atk, 70, spa, 70, spe),
        types=types,
        moves=moves,
    )
    return sp


def _make_dummy_move(name: str = "Tackle", bp: int = 40, cat: int = 0) -> Move:
    """Create a minimal Move for testing."""
    return Move(
        name=name,
        pkm_type=Type.NORMAL,
        base_power=bp,
        accuracy=1.0,
        max_pp=35,
        category=cat,
    )


class TestCreateArchetypeBuilds:
    """Tests for create_archetype_builds."""

    def test_empty_moveset_returns_empty(self):
        """No predicted moves means no builds."""
        species = _make_dummy_species("TestMon")
        builds = create_archetype_builds(species, [])
        assert len(builds) == 0

    def test_empty_moveset_from_empty_moves(self):
        """Species with no moves and empty predicted set returns empty."""
        species = _make_dummy_species("Empty")
        builds = create_archetype_builds(species, [])
        assert len(builds) == 0

    def test_returns_list_of_tuples(self):
        """Returns list of (str, Pokemon) tuples."""
        species = _make_dummy_species("TestMon", moves=[_make_dummy_move()])
        moves = [_make_dummy_move()]
        builds = create_archetype_builds(species, moves)
        for item in builds:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], Pokemon)

    def test_physical_leaning_species_has_physical_builds(self):
        """Species with higher Atk than SpA gets Physical archetypes."""
        tackle = _make_dummy_move("Tackle", 40, 0)
        species = _make_dummy_species("PhysicalMon", atk=120, spa=60, moves=[tackle])
        moves = [tackle]
        builds = create_archetype_builds(species, moves)
        names = [name for name, _ in builds]
        assert "Fast Physical Sweeper" in names
        assert "Fast Special Sweeper" in names

    def test_special_leaning_species_has_special_builds(self):
        """Species with higher SpA than Atk gets Special archetypes."""
        sb = _make_dummy_move("Shadow Ball", 80, 1)
        species = _make_dummy_species("SpecialMon", atk=60, spa=120, moves=[sb])
        moves = [sb]
        builds = create_archetype_builds(species, moves)
        archetype_names = [name for name, _ in builds]
        assert "Fast Physical Sweeper" in archetype_names
        assert "Fast Special Sweeper" in archetype_names

    def test_base_number_of_builds(self):
        """Standard species gets 6 builds (no mixed attacker bonus)."""
        tackle = _make_dummy_move()
        species = _make_dummy_species("StandardMon", atk=100, spa=50, moves=[tackle])
        moves = [tackle]
        builds = create_archetype_builds(species, moves)
        assert len(builds) == 6

    def test_mixed_attacker_gets_extra_builds(self):
        """Balanced Atk/SpA species gets 10 builds (4 extra mixed)."""
        tackle = _make_dummy_move()
        species = _make_dummy_species("MixedMon", atk=80, spa=80, moves=[tackle])
        moves = [tackle]
        builds = create_archetype_builds(species, moves)
        assert len(builds) == 10

    def test_fast_sweeper_has_252_speed_evs(self):
        """Fast Sweeper builds have 252 Speed EVs."""
        species = _make_dummy_species("FastMon", atk=100, spa=60)
        moves = [_make_dummy_move("Tackle", 40, 0)]
        builds = create_archetype_builds(species, moves)
        for name, pkm in builds:
            if "Fast" in name:
                assert pkm.evs[Stat.SPEED] == 252, f"{name} should have 252 Spe EVs"

    def test_bulky_attacker_has_252_hp_evs(self):
        """Bulky builds have 252 HP EVs."""
        species = _make_dummy_species("BulkyMon", atk=100, spa=60)
        moves = [_make_dummy_move("Tackle", 40, 0)]
        builds = create_archetype_builds(species, moves)
        for name, pkm in builds:
            if "Bulky" in name:
                assert pkm.evs[Stat.MAX_HP] == 252, f"{name} should have 252 HP EVs"

    def test_defensive_wall_has_252_hp_and_defense(self):
        """Defensive walls have 252 HP and a defensive stat."""
        species = _make_dummy_species("WallMon", atk=60, spa=60)
        moves = [_make_dummy_move("Tackle", 40, 0)]
        builds = create_archetype_builds(species, moves)
        for name, pkm in builds:
            if "Defensive" in name:
                assert pkm.evs[Stat.MAX_HP] == 252
                assert pkm.evs[Stat.DEFENSE] == 252 or pkm.evs[Stat.SPECIAL_DEFENSE] == 252


class TestCreateGenericBuild:
    """Tests for create_generic_build_for_species."""

    def test_returns_pokemon(self):
        """Returns a fully-formed Pokemon object."""
        species = _make_dummy_species("TestMon")
        moves = _make_dummy_move()
        species.moves = [moves]
        build = create_generic_build_for_species(species)
        assert build is not None
        assert isinstance(build, Pokemon)

    def test_returns_none_for_no_moves(self):
        """Species with no moves returns None."""
        species = _make_dummy_species("Empty")
        build = create_generic_build_for_species(species)
        assert build is None

    def test_has_four_moves(self):
        """Generic build has exactly 4 moves."""
        species = _make_dummy_species("TestMon")
        species.moves = [_make_dummy_move(f"Move{i}", 40, 0) for i in range(6)]
        build = create_generic_build_for_species(species)
        assert build is not None
        assert len(build.moves) == 4

    def test_has_neutral_nature(self):
        """Generic build uses SERIOUS (neutral) nature."""
        species = _make_dummy_species("TestMon")
        species.moves = [_make_dummy_move()]
        build = create_generic_build_for_species(species)
        assert build is not None
        assert build.nature == Nature.SERIOUS

    def test_has_balanced_evs(self):
        """Generic build has 85 EVs in every stat."""
        species = _make_dummy_species("TestMon")
        species.moves = [_make_dummy_move()]
        build = create_generic_build_for_species(species)
        assert build is not None
        assert all(ev == 85 for ev in build.evs)
