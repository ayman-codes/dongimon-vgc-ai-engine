# Dongimon -- Multi-Policy AI Engine for Pokemon VGC

<p align="center">
  <strong>Heuristic Battle AI, Game-Theoretic Team Preview, and Evolutionary Team Building for Competitive Doubles</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python 3.11+"></a>
  <a href=".github/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/dongimon/dongimon/ci.yml?branch=main&logo=github" alt="CI"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff" alt="Ruff"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/typed-mypy-039dfc" alt="mypy"></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow" alt="MLflow"></a>
</p>

---

## Overview

Dongimon is a **multi-policy AI engine** for competitive Pokemon VGC (double battles). It models Pokemon VGC as a turn-based strategy game with partial observability and implements three independent policies:

- **Battle Policy** -- Joint-action heuristic scoring with cross-slot synergy evaluation for turn-by-turn decisions
- **Selection Policy** -- Game-theoretic team preview using predicted opponent builds and sub-tournament payoff matrices
- **Team Build Policy** -- Three-stage evolutionary pipeline (heuristic funnel, genetic algorithm, battle royale simulation)

Built on top of the `vgc2` simulation engine (v2.1.3)

---

## Core Skills

<table>
<tr>
<th width="220">Domain</th>
<th>Skills</th>
</tr>
<tr>
<td><strong>Heuristic Decision Making</strong></td>
<td>Joint-Action Synergy Scoring, Move Value Decomposition (Damage + KO + Status + Utility), Threat Estimation with Fog-of-War Inference, Choice Item Lock Detection, Board Position Evaluation (Lookahead), Protect / Switch Scoring, Priority Move Threat Delta</td>
</tr>
<tr>
<td><strong>Game Theory & Selection</strong></td>
<td>Team Preview Payoff Matrices, Sub-Tournament Simulation, Opponent Moveset Prediction, Archetype Build Generation, Combinatorial Pair Evaluation, Nash-Inspired Greedy Selection</td>
</tr>
<tr>
<td><strong>Evolutionary Algorithms</strong></td>
<td>Population Initialisation (Weighted Random Sampling), 5-Component Team Fitness (Viability + Type Coverage + Type Defence + Stat Diversity + Role Diversity), Tournament Selection, Single-Point Crossover with Deduplication, Per-Position Mutation, Elite Preservation, Battle Royale Tournament Validation</td>
</tr>
<tr>
<td><strong>Hyperparameter Optimisation</strong></td>
<td>Optuna Bayesian Optimisation (14-Weight Tuning), Sensitivity Analysis, Multi-Competitor Objective Maximisation, Study Persistence (SQLite), Resume Support</td>
</tr>
<tr>
<td><strong>Software Engineering</strong></td>
<td>Modular Policy Architecture (Single-Responsibility Modules), Pydantic Configuration Validation, mypy Strict Type Checking, Ruff Linting, pytest Unit Tests, YAML Weights Configuration, Clean Separation of Config/Logic/Evaluation</td>
</tr>
<tr>
<td><strong>MLOps & Benchmarking</strong></td>
<td>MLflow Experiment Tracking, Isolated vs. Full-Competitor Benchmark Modes, CSV Result Logging, Seed-Based Reproducibility, Deterministic Battle Engine (ZERO_RNG), JSON Run Artifacts</td>
</tr>
<tr>
<td><strong>Reinforcement Learning (Planned)</strong></td>
<td>Behavior Cloning (BC), Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), Policy Space Response Oracles (PSRO), Self-Play, RL from Demonstrations (RLfD), Transfer Learning, vgc2 BattleEnv (Gymnasium)</td>
</tr>
</table>

---

## Architecture

```
                    DongimonCompetitor
                           |
         +-----------------+-----------------+
         |                 |                 |
  DongimonBattlePolicy  SelectionPolicy  HesfTeamBuildPolicy
         |                 |                 |
    +----+----+           +-+---------+    +--+--+----------+
    |    |    |           |           |    |     |          |
  Move  Joint  Threat   Predict  Tournament  Builds  Evolution  BattleRoyale
  Score  Pair  Detect    Moveset  Simulation  Power   Operators
                                          +------+------+
                                          |      |      |
                                       Fitness  Scoring  Moveset
```

### Policy Breakdown

**Battle Policy** (`src/battle/`) -- Each turn, generates all legal actions (moves &times; targets + switches to reserves), scores them individually via `move_scoring.py`, then pairs actions across both active slots and evaluates **9 weighted synergy components**: base individual scores, survival impact, focus fire, target priority, off-def support, setup synergy, environmental synergy, and board position lookahead. Weights are loaded from a tuned `battle_weights.yaml` configuration.

**Selection Policy** (`src/selection/`) -- At Team Preview, predicts opponent movesets using archetype-based damage/utility scoring, then evaluates all C(6,2) &times; C(6,2) pair matchups via sub-tournament simulation (vgc2 BattleEngine with GreedyBattlePolicy). Returns the top-ranked 4-Pokemon roster.

**Team Build Policy** (`src/teambuild/`) -- Three-stage pipeline:
1. **Heuristic Funnel** -- Creates optimal single builds per species (minimon-style role detection + STAB-weighted move selection), ranks by stat-based power proxy
2. **Evolutionary Algorithm** -- Population of 50 teams over 30 generations with 5-component fitness (viability 30%, type coverage 25%, type defence 25%, stat diversity 10%, role diversity 10%), tournament selection, single-point crossover, per-position mutation
3. **Battle Royale** -- Round-robin vgc2 simulation between top K teams using GreedyBattlePolicy to empirically validate the best roster

---

## Quick Start

```bash
# Install dependencies
uv sync --dev

# Run test suite
uv run pytest tests/ -v

# Smoke test (3 battles vs Greedy baseline)
uv run python scripts/run_competition.py

# Isolated benchmark (battle policy only, 5 matches x 25 battles)
uv run python scripts/benchmark.py --n-matches=5 --n-battles=25

# Full benchmark (all 3 policies, 5 matches x 25 battles)
uv run python scripts/benchmark.py --full --n-matches=5 --n-battles=25

# Optuna weight tuning (resumes from existing study)
uv run python scripts/tune_weights.py
```

---

## Package Structure

```
dongimon/
├── competitor.py                  # Root entry point: DongimonCompetitor
├── pyproject.toml                  # Project config (Python 3.11+, deps, tooling)
│
├── src/
│   ├── shared/                     # Shared utilities (all policies)
│   │   ├── types.py                #  19x19 string-based type chart + lookups
│   │   ├── damage.py               #  Gen 9 damage formula, STAB/weather/terrain mods
│   │   ├── move_utils.py           #  Accuracy factor, PP penalty, category checks
│   │   └── archetypes.py           #  Generate competitive builds per species
│   │
│   ├── config/                     # Configuration & constants
│   │   ├── models.py               #  Pydantic models: BattleWeights, SelectionConfig, TeambuildConfig
│   │   ├── loader.py               #  YAML weight loader, config factories
│   │   ├── constants.py            #  ~120 named constants (KO bonuses, thresholds, etc.)
│   │   └── battle_weights.yaml     #  Tuned 14-weight vector from Optuna
│   │
│   ├── battle/                     # Turn-by-turn battle policy
│   │   ├── policy.py               #  DongimonBattlePolicy (orchestrator)
│   │   ├── move_scoring.py         #  Offensive move / protect / switch scoring, threat estimation
│   │   └── joint.py               #  Joint action pairing with 9 synergy components
│   │
│   ├── selection/                  # Team preview selection policy
│   │   ├── policy.py               #  DongimonSelectionPolicy (pair-vs-pair tournament)
│   │   ├── prediction.py           #  Opponent moveset prediction via damage/utility scoring
│   │   └── tournament.py           #  Sub-tournament simulation with vgc2 BattleEngine
│   │
│   ├── teambuild/                  # HESF team building policy
│   │   ├── policy.py               #  HesfTeamBuildPolicy (3-stage pipeline orchestrator)
│   │   ├── builds.py               #  Single optimal build, species_power, role detection
│   │   ├── fitness.py              #  Archtype fitness evaluation, stat compatibility
│   │   ├── scoring.py              #  Damage / utility move scoring against full roster
│   │   ├── moveset.py              #  Role-aware 4-move selection (4-dimension scoring)
│   │   ├── evolution.py            #  GA loop: init -> fitness -> sort -> elitism -> crossover -> mutate
│   │   ├── operators.py            #  GA operators: init_pop, crossover, mutate, team fitness
│   │   └── battle_royale.py        #  Round-robin simulation tournament validation
│   │
│   ├── rl/                         # Reinforcement learning (placeholder)
│   │   └── __init__.py
│   │
│   └── tracking/                   # Benchmark result logging
│       └── benchmark_tracker.py    #  BenchmarkTracker (context manager -> JSON)
│
├── scripts/
│   ├── run_competition.py          # Smoke test 
│   ├── benchmark.py                # Battle royale benchmark (isolated + full modes)
│   ├── benchmark_competitors.py    # Full competitor benchmark 
│   ├── benchmark_isolated.py       # Battle-policy-only benchmark (same team)
│   ├── tune_weights.py             # Optuna 14-weight tuning pipeline
│   ├── compare_legacy.py           # Legacy vs extracted policy comparison
│   └── start_mlflow_server.py      # Launch MLflow tracking server
│
├── tests/                          # pytest test suite (11 files)
│   ├── test_battle_move_scoring.py
│   ├── test_battle_joint.py
│   ├── test_selection_tournament.py
│   ├── test_teambuild_*.py
│   ├── test_shared_*.py
│   └── conftest.py
│
├── competitors/                    # 12 championship competitor wrappers
├── context/                        # Documentation & API reference
├── data/                           # Benchmark result CSVs
├── legacy/                         # Archived monolithic pre-modularization policies
└── mlruns/                         # MLflow run artifacts & best-weight JSONs
```

---

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Battle Policy | Operational | 9-component heuristic with tuned weights, Choice Lock detection, board position lookahead |
| Selection Policy | Operational | Sub-tournament based, uses GreedyBattlePolicy for internal sims |
| Team Build Policy | Operational | 3-stage HESF pipeline with GA + battle royale |
| Optuna Tuning | Complete | 14 weights tuned against JJJ/minimon/StocKarpador |
| Benchmark Suite | Complete | Isolated + full modes, CSV logging, MLflow tracking |
| RL Training | Planned | BC pretraining -> DQN/PPO -> PSRO self-play -> Transfer learning |

---

## Extending

1. Subclass one of the ABCs in `vgc2.agent` (BattlePolicy, SelectionPolicy, TeamBuildPolicy)
2. Wire your policy into `competitor.py` as a property
3. Test with `uv run python scripts/run_competition.py`
4. Benchmark with `uv run python scripts/benchmark.py`

---

<p align="center">
  <sub>Built with Python 3.11+ &middot; vgc2 v2.1.3 &middot; MLflow &middot; Optuna &middot; Pydantic &middot; YAML</sub>
</p>
