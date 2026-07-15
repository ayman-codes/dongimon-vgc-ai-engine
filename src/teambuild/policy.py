"""HESF Team Build Policy — Heuristic Evolutionary Simulation Funnel.

Stage 1: Create a single optimal build per species using role detection
(minimon-style), rank by stat-based power proxy (StocKarpador-style).
Stage 2: Evolutionary algorithm for team synergy (type coverage, defence,
stat diversity, role diversity).
Stage 3: Battle royale simulation tournament for final validation.
"""

from numpy.random import default_rng
from vgc2.agent import TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster

from src.config.loader import teambuild_config
from src.config.models import TeambuildConfig
from src.teambuild.battle_royale import run_battle_royale
from src.teambuild.builds import create_single_optimal_build, species_power
from src.teambuild.evolution import run_evolution


class HesfTeamBuildPolicy(TeamBuildPolicy):
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
    ) -> list:
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

        # Stage 1: builds and viability
        builds_cache = {}
        for species in roster:
            build = create_single_optimal_build(species)
            if build is not None:
                builds_cache[species] = build

        viability = {s: species_power(s) for s in builds_cache}

        sorted_species = sorted(viability, key=viability.get, reverse=True)

        if not cfg.enable_evolution:
            top_species = sorted_species[:max_team_size]
            return self._build_commands(top_species, roster, builds_cache, max_pkm_moves)

        pool_size = max(max_team_size, int(len(sorted_species) * (1 - cfg.pruning_percentage)))
        pool_species = sorted_species[:pool_size]

        pool_index_map = {s: i for i, s in enumerate(pool_species)}
        pool_viability = {pool_index_map[s]: viability[s] for s in pool_species}

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
        except Exception:
            top_teams = [list(range(max_team_size))]

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
            except Exception:
                best_team_idx = top_teams[0]
        else:
            best_team_idx = top_teams[0]

        selected = [pool_species[i] for i in best_team_idx]
        return self._build_commands(selected, roster, builds_cache, max_pkm_moves)

    def _build_commands(
        self,
        selected: list,
        roster: Roster,
        builds_cache: dict,
        max_pkm_moves: int,
    ) -> list:
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
            move_indices = [
                build.species.moves.index(m) for m in build.moves
                if m in build.species.moves
            ][:max_pkm_moves]
            cmd = (roster_index, build.evs, build.ivs, build.nature, move_indices)
            commands.append(cmd)
        return commands
