"""HESF Team Build Policy — Heuristic Evolutionary Simulation Funnel.

Stage 1: Create a single optimal build per species using role detection
(minimon-style), rank by stat-based power proxy (StocKarpador-style),
then upgrade pool builds via multi-archetype evaluation.
Stage 2: Evolutionary algorithm for team synergy (type coverage, defence,
stat diversity, role diversity).
Stage 3: Battle royale simulation tournament for final validation.
"""

from typing import Any

from numpy.random import default_rng
from vgc2.agent import TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster
from vgc2.battle_engine.modifiers import Stat

from src.config.loader import teambuild_config
from src.config.models import TeambuildConfig
from src.shared.archetypes import create_archetype_builds, create_generic_build_for_species
from src.teambuild.battle_royale import run_battle_royale
from src.teambuild.builds import create_single_optimal_build, species_power
from src.teambuild.evolution import run_evolution
from src.teambuild.fitness import calculate_stat_compatibility, get_optimal_archetype
from src.teambuild.moveset import get_role_aware_moveset
from src.teambuild.scoring import build_coefficient_table


class HesfTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Three-stage team building pipeline.

    Stage 1 builds one optimal Pokemon per species (minimon-style role
    detection) and ranks species by stat-based power. Stage 2 evolves
    teams of 6 using synergy-aware fitness. Stage 3 validates the top
    teams via vgc2 BattleEngine simulation.
    """

    def __init__(self, config: TeambuildConfig | None = None):
        super().__init__()
        self._config = config or teambuild_config()

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> list[Any]:
        """Build a team from the roster.

        Args:
            roster: List of available PokemonSpecies.
            meta: Optional metagame usage data (unused).
            max_team_size: Maximum Pokemon per team.
            max_pkm_moves: Maximum moves per Pokemon.
            n_active: Number of active Pokemon in battle (unused).

        Returns:
            TeamBuildCommand: (roster_id, evs, ivs, nature, move_indices) per Pokemon.
        """
        rng = default_rng()
        cfg = self._config

        builds_cache = {}
        for species in roster:
            build = create_single_optimal_build(species)
            if build is not None:
                builds_cache[species] = build

        viability = {s: species_power(s) for s in builds_cache}

        sorted_species = sorted(viability, key=lambda s: viability[s], reverse=True)

        hp_threshold = cfg.hp_filter_min
        if hp_threshold > 0:
            hp_eligible = [s for s in sorted_species if s.base_stats[0] >= hp_threshold]
            if len(hp_eligible) >= max_team_size:
                sorted_species = hp_eligible

        if not cfg.enable_evolution:
            top_species = sorted_species[:max_team_size]
            return self._build_commands(top_species, roster, builds_cache, max_pkm_moves)

        pool_size = max(max_team_size, int(len(sorted_species) * (1 - cfg.pruning_percentage)))
        pool_species = sorted_species[:pool_size]

        generic_cache: dict[Any, Any] = {}
        for s in roster:
            b = create_generic_build_for_species(s)
            if b is not None:
                generic_cache[s] = b

        roster_list = list(roster)
        global_max_scores, coeff_cache = self._compute_global_max_scores(
            pool_species, roster_list, generic_cache
        )

        upgrade_cap = max(max_team_size, len(pool_species) // 4)
        for species in pool_species[:upgrade_cap]:
            try:
                coeff_table = coeff_cache.get(species)
                if coeff_table is None:
                    coeff_table = build_coefficient_table(
                        species, roster_list, generic_cache, self.params
                    )
                upgraded = get_optimal_archetype(
                    species, roster_list, global_max_scores, self.params, generic_cache, coeff_table
                )
                if upgraded is not None:
                    builds_cache[species] = upgraded
            except (RuntimeError, ValueError, IndexError, KeyError):
                pass

        pool_viability = {s: viability[s] for s in pool_species}

        try:
            top_teams = run_evolution(
                pool_species=pool_species,
                viability_scores=pool_viability,
                team_size=max_team_size,
                pop_size=cfg.population_size,
                generations=cfg.generations,
                mutation_rate=cfg.mutation_rate,
                elite_fraction=cfg.elite_fraction,
                rng=rng,
            )
        except (RuntimeError, ValueError, IndexError):
            safe_size = min(max_team_size, len(pool_species))
            top_teams = [list(range(safe_size))]

        if cfg.enable_battle_royale:
            try:
                best_team_idx = run_battle_royale(
                    top_teams=top_teams,
                    builds_cache=builds_cache,
                    pool_species=pool_species,
                    roster=roster,
                    n_battles=cfg.battle_royale_battles,
                    max_time_sec=cfg.battle_royale_timeout_sec,
                    params=self.params,
                )
            except (RuntimeError, ValueError, IndexError):
                best_team_idx = top_teams[0]
        else:
            best_team_idx = top_teams[0]

        selected = [pool_species[i] for i in best_team_idx]
        return self._build_commands(selected, roster, builds_cache, max_pkm_moves)

    def _build_commands(
        self,
        selected: list[Any],
        roster: Roster,
        builds_cache: dict[Any, Any],
        max_pkm_moves: int,
    ) -> list[Any]:
        """Convert selected species to a TeamBuildCommand.

        Args:
            selected: Ordered list of selected species.
            roster: Full roster for index resolution.
            builds_cache: Build cache mapping species -> Pokemon.
            max_pkm_moves: Maximum moves per Pokemon.

        Returns:
            TeamBuildCommand list.
        """
        commands = []
        for species in selected:
            roster_index = roster.index(species)
            build = builds_cache.get(species)
            if build is None:
                continue
            move_indices = [build.species.moves.index(m) for m in build.moves if m in build.species.moves][
                :max_pkm_moves
            ]
            cmd = (roster_index, build.evs, build.ivs, build.nature, move_indices)
            commands.append(cmd)
        return commands

    def _compute_global_max_scores(
        self,
        pool_species: list[Any],
        roster_list: list[Any],
        generic_cache: dict[Any, Any],
    ) -> tuple[dict[str, float], dict[Any, Any]]:
        """Compute normalization maxima by sampling top pool species.

        Evaluates archetype builds for a bounded sample of species to
        determine the maximum achievable scores for each fitness component.
        These maxima are used to normalize scores in get_optimal_archetype.
        Also caches the coefficient tables built during sampling for reuse
        in the upgrade loop, avoiding redundant O(n) damage precomputation.

        Args:
            pool_species: Species pool sorted by viability (best first).
            roster_list: Full roster as a list for damage calculation context.
            generic_cache: Precomputed generic builds per species.

        Returns:
            Tuple of (maxima dict, coeff_table cache dict).
        """
        sample_size = min(10, len(pool_species))
        sample = pool_species[:sample_size]

        max_stat = 1.0
        max_dmg = 1.0
        max_util = 1.0
        max_stat_syn = 1.0
        max_speed_syn = 1.0
        max_speed_stat = 1.0
        coeff_cache: dict[Any, Any] = {}

        for species in sample:
            placeholder_moves = species.moves[:4] if species.moves else []
            archetype_builds = create_archetype_builds(species, placeholder_moves)
            coeff_table = build_coefficient_table(species, roster_list, generic_cache, self.params)
            coeff_cache[species] = coeff_table
            for _name, temp_build in archetype_builds:
                stat_score = calculate_stat_compatibility(species, temp_build.evs)
                if stat_score > max_stat:
                    max_stat = stat_score
                speed_stat = temp_build.stats[Stat.SPEED]
                if speed_stat > max_speed_stat:
                    max_speed_stat = float(speed_stat)

                try:
                    optimal_moves, all_scores = get_role_aware_moveset(
                        temp_build, _name, roster_list, self.params, generic_cache, coeff_table
                    )
                    total_dmg = sum(all_scores[m]["damage"] for m in optimal_moves)
                    total_util = sum(all_scores[m]["utility"] for m in optimal_moves)
                    total_stat_syn = sum(all_scores[m]["stat_syn"] for m in optimal_moves)
                    total_speed_syn = sum(all_scores[m]["speed_syn"] for m in optimal_moves)

                    if total_dmg > max_dmg:
                        max_dmg = total_dmg
                    if total_util > max_util:
                        max_util = total_util
                    if total_stat_syn > max_stat_syn:
                        max_stat_syn = total_stat_syn
                    if total_speed_syn > max_speed_syn:
                        max_speed_syn = total_speed_syn
                except (RuntimeError, ValueError, IndexError, KeyError):
                    pass

        return (
            {
                "max_stat": max_stat,
                "max_dmg": max_dmg,
                "max_util": max_util,
                "max_stat_syn": max_stat_syn,
                "max_speed_syn": max_speed_syn,
                "max_speed_stat": max_speed_stat,
            },
            coeff_cache,
        )
