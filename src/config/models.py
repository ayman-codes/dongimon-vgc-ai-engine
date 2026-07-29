"""Pydantic configuration models for the Dongimon engine.

All tunable hyperparameters flow through these models. Weights are loaded
from YAML files and validated at startup.
"""

from pydantic import BaseModel, Field


class BattleWeights(BaseModel):
    """Tunable weights for the 7-component heuristic battle policy.

    Each weight controls the contribution of its corresponding synergy
    or priority component to the final joint-action score. All values
    are in [0, 1] after fixed-scale normalization.

    Attributes:
        w_base_score: Combined individual move score weight for both Pokemon.
        w_env_synergy: Environmental effects synergy bonus weight.
        w_focus_fire: Focus fire (dual-targeting same opponent) bonus weight.
        w_target_priority: Priority for targeting the biggest threat.
        w_survival_impact: Survival risk penalty weight (negative contribution).
        w_off_def_support: Offensive + defensive support pairing bonus weight.
        w_setup_synergy: Setup move + follow-up pairing synergy bonus weight.
    """

    w_base_score: float = Field(default=0.25, ge=0.0, le=1.0, description="Combined base score for both Pokemon")
    w_env_synergy: float = Field(default=0.02, ge=0.0, le=1.0, description="Environmental synergy bonus")
    w_focus_fire: float = Field(default=0.27, ge=0.0, le=1.0, description="Focus fire bonus")
    w_target_priority: float = Field(default=0.18, ge=0.0, le=1.0, description="Target priority bonus")
    w_survival_impact: float = Field(default=0.13, ge=0.0, le=1.0, description="Survival impact penalty")
    w_off_def_support: float = Field(default=0.02, ge=0.0, le=1.0, description="Off/Def support pairing bonus")
    w_setup_synergy: float = Field(default=0.18, ge=0.0, le=1.0, description="Setup synergy bonus")
    w_status_burn: float = Field(default=30.0, ge=0.0, description="Burn status value weight")
    w_status_sleep: float = Field(default=40.0, ge=0.0, description="Sleep status value weight")
    w_status_para: float = Field(default=20.0, ge=0.0, description="Paralysis status value weight")
    w_status_poison: float = Field(default=15.0, ge=0.0, description="Poison status value weight")
    w_status_toxic: float = Field(default=25.0, ge=0.0, description="Toxic status value weight")


class SelectionConfig(BaseModel):
    """Configuration for the Selection Policy.

    Attributes:
        selection_mode: Selection algorithm — 'hybrid' (matrix + simulation),
            'matrix' (damage matrix only), or 'simulate' (full simulation).
        n_top_candidates: Number of top-ranked rosters to simulate (hybrid mode).
        n_active: Number of active Pokémon per side (default 2 for doubles).
        max_team_size: Maximum team size to select (default 4 for VGC).
    """

    selection_mode: str = Field(default="hybrid", description="Selection mode: hybrid, matrix, or simulate")
    n_top_candidates: int = Field(default=5, ge=1, le=15, description="Top rosters to simulate in hybrid mode")
    n_active: int = Field(default=2, ge=1, le=2, description="Active Pokémon per side")
    max_team_size: int = Field(default=4, ge=1, le=6, description="Maximum team size to select")


class SelectionSynergyWeights(BaseModel):
    """Tunable weights for the analytical pair-synergy selection fast path.

    Used when the team is already final size (pure ordering): the policy
    skips the simulation pipeline and ranks the C(n, 2) candidate active
    pairs by a weighted blend of an opponent-aware matchup term and four
    intra-pair teamwork terms. All weights are non-negative; only their
    relative magnitudes matter (Optuna normalizes them to sum to 1).

    Attributes:
        w_matchup: Opponent-aware damage matchup weight (avg/worst blend).
        w_defense: Defensive complementarity weight (weakness coverage).
        w_speed: Speed-control weight (initiative + speed spread).
        w_role: Role/stat balance weight (phys+special, role mix).
        w_coverage: Offensive type-coverage weight vs the opponent's types.
        avg_weight: Fixed blend weight on the average opponent pair.
        worst_weight: Fixed blend weight on the strongest opponent pair.
    """

    w_matchup: float = Field(default=0.40, ge=0.0, le=1.0, description="Opponent-aware matchup weight")
    w_defense: float = Field(default=0.20, ge=0.0, le=1.0, description="Defensive complementarity weight")
    w_speed: float = Field(default=0.15, ge=0.0, le=1.0, description="Speed control weight")
    w_role: float = Field(default=0.10, ge=0.0, le=1.0, description="Role/stat balance weight")
    w_coverage: float = Field(default=0.15, ge=0.0, le=1.0, description="Offensive coverage weight")
    avg_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Blend weight on average opponent pair")
    worst_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Blend weight on strongest opponent pair")

    def synergy_dict(self) -> dict[str, float]:
        """Return the five tunable term weights as a dict.

        Returns:
            Dict mapping term names to their values.
        """
        return {
            "w_matchup": self.w_matchup,
            "w_defense": self.w_defense,
            "w_speed": self.w_speed,
            "w_role": self.w_role,
            "w_coverage": self.w_coverage,
        }


class TeambuildConfig(BaseModel):
    """Configuration for the HESF Team Build Policy.

    Controls the three-stage pipeline: heuristic funnel → evolutionary
    algorithm → battle royale simulation.

    Attributes:
        hp_filter_min: Minimum base HP to keep a species (0 = disabled).
        fitness_mode: Fitness evaluation — 'heuristic' or 'model' (future use).
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

    hp_filter_min: int = Field(default=120, ge=0, le=255, description="Minimum base HP to keep species (0=disabled)")
    fitness_mode: str = Field(default="heuristic", description="Fitness mode: heuristic or model")
    pruning_percentage: float = Field(default=0.2, ge=0.0, le=1.0, description="Roster pruning fraction")
    normalization_sample_size: int = Field(default=800, ge=1, description="Sample size for global max estimation")
    debug: bool = Field(default=False, description="Enable CSV debug logging")
    enable_evolution: bool = Field(default=True, description="Run evolutionary algorithm")
    enable_battle_royale: bool = Field(default=True, description="Run simulation tournament")
    population_size: int = Field(default=30, ge=10, le=200, description="Population size")
    generations: int = Field(default=20, ge=1, le=100, description="Number of generations")
    mutation_rate: float = Field(default=0.10, ge=0.0, le=1.0, description="Per-position mutation rate")
    elite_fraction: float = Field(default=0.10, ge=0.0, le=1.0, description="Elite preservation fraction")
    battle_royale_battles: int = Field(default=5, ge=1, le=50, description="Battles per matchup")
    battle_royale_timeout_sec: float = Field(default=30.0, ge=1.0, le=120.0, description="Simulation timeout")


class TeambuildWeights(BaseModel):
    """Tunable weights for the HESF teambuild pipeline.

    Group A — Archetype upgrade weights control how get_optimal_archetype
    chooses between Fast/Bulky/Wall archetype variants per species.
    Higher values favour the corresponding fitness component.

    Group B — GA fitness weights control how calculate_team_fitness
    ranks teams during evolutionary selection. Higher values favour
    teams strong in the corresponding dimension.
    """

    w_stat: float = Field(default=0.12, ge=0.0, le=1.0)
    w_speed: float = Field(default=0.30, ge=0.0, le=1.0)
    w_dmg: float = Field(default=0.40, ge=0.0, le=1.0)
    w_util: float = Field(default=0.10, ge=0.0, le=1.0)
    w_stat_syn: float = Field(default=0.04, ge=0.0, le=1.0)
    w_speed_syn: float = Field(default=0.04, ge=0.0, le=1.0)

    ga_viability: float = Field(default=0.35, ge=0.0, le=1.0)
    ga_coverage: float = Field(default=0.18, ge=0.0, le=1.0)
    ga_defence: float = Field(default=0.13, ge=0.0, le=1.0)
    ga_stat_diversity: float = Field(default=0.09, ge=0.0, le=1.0)
    ga_role_diversity: float = Field(default=0.08, ge=0.0, le=1.0)
    ga_coverage_balance: float = Field(default=0.17, ge=0.0, le=1.0)

    def archetype_dict(self) -> dict[str, float]:
        """Return Group A weights as a dict for get_optimal_archetype.

        Returns:
            Dict mapping weight names to their values.
        """
        return {
            "w_stat": self.w_stat,
            "w_speed": self.w_speed,
            "w_dmg": self.w_dmg,
            "w_util": self.w_util,
            "w_stat_syn": self.w_stat_syn,
            "w_speed_syn": self.w_speed_syn,
        }

    def ga_dict(self) -> dict[str, float]:
        """Return Group B weights as a dict for calculate_team_fitness.

        Returns:
            Dict mapping weight names to their values.
        """
        return {
            "ga_viability": self.ga_viability,
            "ga_coverage": self.ga_coverage,
            "ga_defence": self.ga_defence,
            "ga_stat_diversity": self.ga_stat_diversity,
            "ga_role_diversity": self.ga_role_diversity,
            "ga_coverage_balance": self.ga_coverage_balance,
        }
