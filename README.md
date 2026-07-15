# Dongimon — RL & Game Theory Engine for Pokémon VGC

A multi-policy AI engine for competitive Pokémon doubles (VGC),
featuring heuristic battle decisions, game-theoretic team preview
selection, and roster-viability-based team building.

## Quick Start

```bash
# Install dependencies
pip install -e "C:\Users\[username]\pokemon-vgc-engine"
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run a competition smoke test 
uv run python scripts/run_competition.py
```

## Package Structure

```
src/
├── shared/              # Type chart, damage formula, move utils, archetypes
├── battle/              # Battle policy (move scoring, joint actions, threat)
├── selection/           # Team preview selection (prediction, sub-tournaments)
├── teambuild/           # Team building (archetype fitness, viability ranking)
├── config/              # Pydantic config models, YAML weights
└── rl/                  # RL training pipeline (placeholder)
```

## Policies

| Policy | Class | Module |
|--------|-------|--------|
| Battle | `DongimonBattlePolicy` | `src.battle.policy` |
| Selection | `DongimonSelectionPolicy` | `src.selection.policy` |
| Team Build | `HesfTeamBuildPolicy` | `src.teambuild.policy` |

## Extending

1. Subclass one of the ABCs in `vgc2.agent` (BattlePolicy, SelectionPolicy, TeamBuildPolicy).
2. Wire it into `competitor.py` as a property.
3. Test with `uv run python scripts/run_competition.py`.

## License

Custom VGC-AI Tournament Exclusion License — personal and educational
use permitted; commercial and tournament use prohibited.
