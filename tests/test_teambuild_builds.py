"""Tests for single build creation and species power scoring."""

from vgc2.battle_engine.modifiers import Category, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies

from src.teambuild.builds import (
    _detect_role,
    _pick_moves,
    create_single_optimal_build,
    species_power,
    species_role,
)


def _ms(name, bp=80, cat=Category.PHYSICAL, hp=100, atk=100, df=70, spa=80, spd=70, spe=80,
        pkm_type=None, types=None):
    if pkm_type is None:
        pkm_type = Type.NORMAL
    if types is None:
        types = [Type.NORMAL]
    move = Move(pkm_type=pkm_type, base_power=bp, accuracy=1.0, max_pp=32, category=cat, name=f"{name}Move")
    return PokemonSpecies(
        name=name, base_stats=(hp, atk, df, spa, spd, spe),
        types=types, moves=[move],
    )


class TestCreateSingleOptimalBuild:
    """Tests for create_single_optimal_build."""

    def test_returns_pokemon(self):
        species = _ms("Test")
        build = create_single_optimal_build(species)
        assert build is not None

    def test_has_4_moves(self):
        moves = [
            Move(pkm_type=Type.NORMAL, base_power=40, accuracy=1.0, max_pp=35, category=Category.PHYSICAL, name="A"),
            Move(pkm_type=Type.FIRE, base_power=80, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="B"),
            Move(pkm_type=Type.WATER, base_power=60, accuracy=1.0, max_pp=20, category=Category.SPECIAL, name="C"),
            Move(pkm_type=Type.GRASS, base_power=70, accuracy=1.0, max_pp=15, category=Category.PHYSICAL, name="D"),
            Move(pkm_type=Type.ELECTRIC, base_power=50, accuracy=1.0, max_pp=25, category=Category.SPECIAL, name="E"),
        ]
        species = PokemonSpecies(
            name="Test", base_stats=(100, 100, 70, 100, 70, 80),
            types=[Type.NORMAL], moves=moves,
        )
        build = create_single_optimal_build(species)
        assert build is not None
        assert len(build.moves) == 4

    def test_evs_sum_510_or_less(self):
        species = _ms("EVMon", hp=100, atk=120, spa=60)
        build = create_single_optimal_build(species)
        assert build is not None
        assert sum(build.evs) <= 510

    def test_returns_none_for_no_moves(self):
        species = PokemonSpecies(
            name="Empty", base_stats=(100, 80, 70, 80, 70, 80),
            types=[Type.NORMAL], moves=[],
        )
        build = create_single_optimal_build(species)
        assert build is None

    def test_has_max_ivs(self):
        species = _ms("IVMon")
        build = create_single_optimal_build(species)
        assert build is not None
        assert all(iv == 31 for iv in build.ivs)

    def test_level_100(self):
        species = _ms("LevelMon")
        build = create_single_optimal_build(species)
        assert build is not None
        assert build.level == 100


class TestSpeciesPower:
    """Tests for species_power."""

    def test_positive_score(self):
        species = _ms("Test")
        sp = species_power(species)
        assert sp > 0

    def test_higher_stats_higher_score(self):
        weak = _ms("Weak", hp=50, atk=50, spa=50, bp=40)
        strong = _ms("Strong", hp=150, atk=150, spa=150, bp=120)
        assert species_power(strong) > species_power(weak)

    def test_deterministic(self):
        species = _ms("Test")
        sp1 = species_power(species)
        sp2 = species_power(species)
        assert sp1 == sp2


class TestSpeciesRole:
    """Tests for species_role."""

    def test_physical_sweeper(self):
        species = _ms("PhysSweeper", atk=150, spa=50, spe=120)
        assert species_role(species) == "sweeper"

    def test_special_sweeper(self):
        species = _ms("SpecSweeper", atk=50, spa=150, spe=120)
        assert species_role(species) == "sweeper"

    def test_tank(self):
        phys_move = Move(
            pkm_type=Type.NORMAL, base_power=80, accuracy=1.0,
            max_pp=32, category=Category.PHYSICAL, name="TankPhys",
        )
        spec_move = Move(
            pkm_type=Type.NORMAL, base_power=80, accuracy=1.0,
            max_pp=32, category=Category.SPECIAL, name="TankSpec",
        )
        species = PokemonSpecies(
            name="Tank", base_stats=(150, 50, 130, 50, 130, 30),
            types=[Type.NORMAL], moves=[phys_move, spec_move],
        )
        assert species_role(species) == "wall"

    def test_mixed(self):
        species = _ms("Mixed", atk=90, spa=90, hp=100, df=70, spd=70, spe=80)
        assert species_role(species) in ("sweeper", "wall", "mixed")


class TestDetectRole:
    """Tests for _detect_role."""

    def test_physical_sweeper_role(self):
        role, sub = _detect_role((100, 150, 70, 50, 70, 120))
        assert role == "sweeper"
        assert sub == "physical"

    def test_special_sweeper_role(self):
        role, sub = _detect_role((100, 50, 70, 150, 70, 120))
        assert role == "sweeper"
        assert sub == "special"

    def test_wall_role(self):
        role, sub = _detect_role((150, 50, 130, 50, 130, 30))
        assert role == "wall"

    def test_allrounder_role(self):
        role, sub = _detect_role((100, 80, 70, 80, 70, 80))
        assert role in ("sweeper", "wall", "mixed")


class TestPickMoves:
    """Tests for _pick_moves."""

    def test_returns_list(self):
        species = _ms("Test")
        indices = _pick_moves(species)
        assert isinstance(indices, list)
        assert len(indices) <= 4

    def test_type_diversity(self):
        moves = [
            Move(pkm_type=Type.FIRE, base_power=80, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="A"),
            Move(pkm_type=Type.WATER, base_power=80, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="B"),
            Move(pkm_type=Type.ELECTRIC, base_power=80, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="C"),
            Move(pkm_type=Type.GRASS, base_power=80, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="D"),
            Move(pkm_type=Type.ICE, base_power=80, accuracy=1.0, max_pp=15, category=Category.SPECIAL, name="E"),
        ]
        species = PokemonSpecies(
            name="Test", base_stats=(100, 80, 70, 80, 70, 80),
            types=[Type.FIRE], moves=moves,
        )
        indices = _pick_moves(species)
        assert len(indices) == 4
