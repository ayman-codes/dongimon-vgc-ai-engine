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
<td>Optuna Bayesian Optimisation (Battle 14-Weight, Teambuild 12-Weight, Selection 5-Weight), Sensitivity Analysis, Multi-Competitor Objective Maximisation, Study Persistence (SQLite), Resume Support, EC2 Parallel Studies</td>
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

**Battle Policy** -- The Optuna-tuned weighted heuristic (`DongimonBattlePolicy`) now lives under `PPO_trainers/weighted_heuristic/` (moved from `src/battle/`), kept as a benchmark opponent and PPO sparring partner. `src/battle/` is reserved for the upcoming Greedy_Dongi net-damage policy (see `context/greedy_dongi_plan.md`). Each turn it generates all legal actions (moves &times; targets + switches to reserves), scores them individually, then pairs actions across both active slots and evaluates **9 weighted synergy components**: base individual scores, survival impact, focus fire, target priority, off-def support, setup synergy, environmental synergy, and board position lookahead. Weights are loaded from a tuned `battle_weights.yaml` configuration.

**Selection Policy** (`src/selection/`) -- At Team Preview, two paths: (1) **Fast path** (team ≤ 4): analytical pair-synergy ranking of all C(n,2) active pairs using 4 teamwork terms (offensive coverage, defensive complementarity, speed control, role balance) blended with an opponent-aware matchup term — no simulation needed. (2) **Full path** (team > 4): predicts opponent movesets, pre-filters rosters via damage matrix, then runs sub-tournament simulations. Weights tuned via Optuna and stored in `selection_synergy.yaml`.

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

# Smoke test (championship-track)
uv run python scripts/championship_track.py

# Battle-policy benchmark (BP only, no selection)
uv run python scripts/benchmark/benchmark_battle.py --seed=42 --n-rounds=5 --n-battles=20

# Teambuild + selection benchmark
uv run python scripts/benchmark/benchmark_team.py --seed=42 --n-rounds=10 --n-battles=30

# Optuna weight tuning (resumes from existing study)
uv run python scripts/tune_weights.py             # Battle policy (14 weights)
uv run python scripts/tune_teambuild.py           # Teambuild (12 weights, default 400 trials)
uv run python scripts/tune_selection.py           # Selection synergy (5 weights, default 300 trials)
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
│   │   ├── models.py               #  Pydantic models: BattleWeights, SelectionConfig, TeambuildConfig, SelectionSynergyWeights
│   │   ├── loader.py               #  YAML weight loader, config factories
│   │   ├── constants.py            #  ~120 named constants (KO bonuses, thresholds, etc.)
│   │   ├── battle_weights.yaml     #  Tuned 14-weight vector from Optuna
│   │   ├── teambuild_weights.yaml  #  Tuned 12-weight teambuild vector (Optuna trial #192)
│   │   └── selection_synergy.yaml  #  Tuned 5-weight selection fast path (Optuna trial #211)
│   │
│   ├── battle/                     # Turn-by-turn battle policy (greedy_dongi, M2)
│   │   └── __init__.py
│   │
│   ├── selection/                  # Team preview selection policy
│   │   ├── policy.py               #  DongimonSelectionPolicy (fast path + full pipeline)
│   │   ├── pair_synergy.py         #  Analytical pair-synergy scorer (4 teamwork terms)
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
├── PPO_trainers/                   # Retired / sparring-partner battle policies
│   ├── weighted_heuristic/         #  Old DongimonBattlePolicy (moved from src/battle/)
│   │   ├── policy.py
│   │   ├── move_scoring.py
│   │   └── joint.py
│   └── tree_bc_policy/             #  TreeBC XGBoost BC inference wrapper
│       └── policy.py
│
├── scripts/
│   ├── benchmark/                  # Benchmark scripts
│   │   ├── benchmark_battle.py     #  Pure battle-policy ELO benchmark
│   │   ├── benchmark_team.py       #  Teambuild + selection benchmark
│   │   ├── benchmark_selection_bp.py      #  Selection + battle-policy ELO benchmark
│   │   ├── benchmark_greedy_vs_dongimon.py  #  Dongimon vs Greedy head-to-head
│   │   ├── compare_legacy.py       #  Legacy vs extracted policy comparison
│   │   └── execute_benchmark_bc.py #  Overnight orchestration (benchmarks + BC data)
│   ├── championship_track.py       # Championship-track benchmark (mirrors vgc2 engine)
│   ├── tune_weights.py             # Optuna 14-weight BP tuning pipeline
│   ├── tune_teambuild.py           # Optuna 12-weight teambuild tuning (400 trials)
│   ├── tune_selection.py           # Optuna 5-weight selection synergy tuning (300 trials)
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
| Selection Policy | Operational | Analytical pair-synergy fast path (≤4) + sub-tournament full path (>4), Optuna-tuned |
| Team Build Policy | Operational | 3-stage HESF pipeline with GA + battle royale, Optuna-tuned weights |
| Optuna Tuning (BP) | Complete | 14 battle weights tuned against JJJ/minimon/StocKarpador |
| Optuna Tuning (TB) | Complete | 12 teambuild weights, best trial #192 WR=0.8060 (400 trials) |
| Optuna Tuning (Sel) | Complete | 5 selection synergy weights, best trial #211 WR=0.8083 (300 trials) |
| Benchmark Suite | Complete | Isolated + full modes, CSV logging, MLflow tracking |
| RL Training | Planned | BC pretraining -> DQN/PPO -> PSRO self-play -> Transfer learning |

---

## Extending

1. Subclass one of the ABCs in `vgc2.agent` (BattlePolicy, SelectionPolicy, TeamBuildPolicy)
2. Wire policy into `competitor.py` as a property
3. Test with `uv run python scripts/championship_track.py`
4. Benchmark with `uv run python scripts/benchmark.py`

---

## Cloud Computing

This project required a wicked amount of computing to generate data for the Matchup Predictor; an XGBoost model to predict which pair would be the winners, the matchup predictor also used optuna trials to find the optimal hyperparameters. It beat even the simulation at a fraction of the speed.

**Matchup Predictor (MP)** -- `src/data/generate.py` ran 24,000 pairings × 50 battles (1.2M battles total), each logged as 56 pairwise feature deltas (type advantages, stat diffs, coverage). `src/data/train.py` then tuned XGBoost, LightGBM, and Random Forest with **150 Optuna trials per model**, nested 5-fold CV, and a 70/15/15 stratified split. XGBoost won (test AUROC **0.79**), and its learned win-probability inference now replaces the damage-ratio matrix pre-filter in the Selection Policy (`src/selection/mp_scoring.py`) -- it evaluates every opponent pair combination and averages P(win), which is orders of magnitude faster than sub-tournament battle simulation while capturing the same matchup dynamics. A single Optuna run tuned the whole thing.

**Team Build Quality Scorer (TQS)** -- unfortunately this model proved to be unapplicable. It was an XGBoost regression model meant to augment the GA fitness function in teambuild, and in isolation it looked great (R² = 0.51 on quarantine data). But the decision-quality experiment proved it did NOT produce better teams than the heuristic (33.3% win rate standalone), and the extra ML complexity was never justified without proven downstream benefit. Despite that, I gained useful information to build the new Battle policy, Behavior cloning policy, PPO and others.

**Greedy_Dongi Battle Policy** -- the lessons from the TQS experiments fed directly into the new battle policy. `src/battle/greedy_dongi.py` copies Greedy's exhaustive enumeration and replaces its offense-only scoring with a **net-damage simulation** that includes opponent response resolved in the engine's priority+speed order. Scoring is lexicographic -- `(opponent KOs, -our KOs, damage dealt - damage taken)` -- with no tunable parameters, no normalization, no balancing. Strategic behavior (focus fire, target priority, protect timing) emerges from accurate simulation, never from weights.

**Behavior Cloning Policy** -- a supervised learner trained on expert demonstrations (state, joint action, outcome). The first TreeBC agent used a 301-feature encoder (`src/tree_bc/encoder.py`) over a 100-class joint action space, but mixed-expert labels averaged contradictions and it benchmarked dead last -- that failure defined the rules for everything after: single-expert labels, win rate as the only metric, and evaluate early.

**PPO** -- the planned reinforcement-learning replacement. Neural BC on **GreedyDongi-only** data (120K win-filtered records) warm-starts a cleanRL-style PPO with logit masking (60-90% of the 100 actions are invalid per state), training against fixed opponents first then ELO-gated self-play. The tiny MLP is CPU-bound in the battle sim, so no GPU spend -- just pure core count.

**All models and data were trained and generated on EC2 c7i.xlarge instance** (Sapphire Rapids, credits-funded), with Optuna study DBs downloaded back to the repo. The 14-weight battle tuning converged at 253 trials; teambuild ran 400 trials and selection 300. PPO long runs are planned for c7i.2xlarge spot.

## Cloud Storage

All data is synced and saved on AWS S3 and all code can call the data from the bucket.

---

<p align="center">
  <sub>Built with Python 3.11+ &middot; vgc2 v2.1.3 &middot; MLflow &middot; Optuna &middot; Pydantic &middot; YAML</sub>
</p>
