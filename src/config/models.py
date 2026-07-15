"""Pydantic configuration models for the Dongimon engine.

All tunable hyperparameters flow through these models. Weights are loaded
from YAML files and validated at startup.
"""

from pydantic import BaseModel, Field


class BattleWeights(BaseModel):
    """Tunable weights for the 8-component heuristic battle policy.

    Each weight controls the contribution of its corresponding synergy
    or priority component to the final joint-action score. All values
    must be positive.

    Attributes:
        w_off_def_support: Offensive + defensive support pairing bonus weight.
        w_base_score_a: Base individual move score for Pokémon A.
        w_env_synergy: Environmental effects synergy bonus weight.
        w_focus_fire: Focus fire (dual-targeting same opponent) bonus weight.
        w_target_priority: Priority for targeting the biggest threat.
        w_base_score_b: Base individual move score for Pokémon B.
        w_survival_impact: Survival risk penalty weight (negative contribution).
        w_setup_synergy: Setup move + follow-up pairing synergy bonus weight.
    """

    w_off_def_support: float = Field(default=0.0200, gt=0.0, description="Off/Def support pairing bonus")
    w_base_score_a: float = Field(default=0.0500, gt=0.0, description="Base score for Pokémon A")
    w_env_synergy: float = Field(default=0.0200, gt=0.0, description="Environmental synergy bonus")
    w_focus_fire: float = Field(default=0.2700, gt=0.0, description="Focus fire bonus")
    w_target_priority: float = Field(default=0.1800, gt=0.0, description="Target priority bonus")
    w_base_score_b: float = Field(default=0.1500, gt=0.0, description="Base score for Pokémon B")
    w_survival_impact: float = Field(default=0.1300, gt=0.0, description="Survival impact penalty")
    w_setup_synergy: float = Field(default=0.1800, gt=0.0, description="Setup synergy bonus")


class SelectionConfig(BaseModel):
    """Configuration for the Selection Policy simulation tournament.

    Attributes:
        simulation_battle_policy_class: Name of the battle policy class used
            for internal simulations (must be a key in BATTLE_POLICY_REGISTRY).
        n_active: Number of active Pokémon per side (default 2 for doubles).
        max_team_size: Maximum team size to select (default 4 for VGC).
    """

    simulation_battle_policy_class: str = Field(default="greedy", description="Policy for internal sim battles")
    n_active: int = Field(default=2, ge=1, le=2, description="Active Pokémon per side")
    max_team_size: int = Field(default=4, ge=1, le=6, description="Maximum team size to select")


class TeambuildConfig(BaseModel):
    """Configuration for the HESF Team Build Policy.

    Controls the three-stage pipeline: heuristic funnel → evolutionary
    algorithm → battle royale simulation.

    Attributes:
        pruning_percentage: Fraction of roster to eliminate in Stage 1 (0.0–1.0).
        normalization_sample_size: Number of builds to sample for global max estimation.
        debug: Whether to write detailed CSV debug logs during team building.
        enable_evolution: If False, skip Stages 2 and 3, return top-N from Stage 1.
        enable_battle_royale: If False, return the fitness-ranked top team from Stage 2.
        population_size: Number of teams in the evolutionary population.
        generations: Number of generations to evolve.
        mutation_rate: Per-position mutation probability (0.0–1.0).
        elite_fraction: Fraction of population preserved unchanged each generation.
        battle_royale_battles: Number of vgc2 battles per matchup in Stage 3.
        battle_royale_timeout_sec: Max wall-clock time for Stage 3 simulation.
    """

    pruning_percentage: float = Field(default=0.3, ge=0.0, le=1.0, description="Roster pruning fraction")
    normalization_sample_size: int = Field(default=800, ge=1, description="Sample size for global max estimation")
    debug: bool = Field(default=False, description="Enable CSV debug logging")
    enable_evolution: bool = Field(default=True, description="Run evolutionary algorithm")
    enable_battle_royale: bool = Field(default=True, description="Run simulation tournament")
    population_size: int = Field(default=50, ge=10, le=200, description="Population size")
    generations: int = Field(default=30, ge=1, le=100, description="Number of generations")
    mutation_rate: float = Field(default=0.10, ge=0.0, le=1.0, description="Per-position mutation rate")
    elite_fraction: float = Field(default=0.10, ge=0.0, le=1.0, description="Elite preservation fraction")
    battle_royale_battles: int = Field(default=5, ge=1, le=50, description="Battles per matchup")
    battle_royale_timeout_sec: float = Field(default=30.0, ge=1.0, le=120.0, description="Simulation timeout")
