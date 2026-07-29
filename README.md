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

**Battle Policy** (`src/battle/`) -- Each turn, generates all legal actions (moves &times; targets + switches to reserves), scores them individually via `move_scoring.py`, then pairs actions across both active slots and evaluates **9 weighted synergy components**: base individual scores, survival impact, focus fire, target priority, off-def support, setup synergy, environmental synergy, and board position lookahead. Weights are loaded from a tuned `battle_weights.yaml` configuration.

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

# Smoke test (3 battles vs Greedy baseline)
uv run python scripts/run_competition.py

# Isolated benchmark (battle policy only, 5 matches x 25 battles)
uv run python scripts/benchmark.py --n-matches=5 --n-battles=25

# Full benchmark (all 3 policies, 5 matches x 25 battles)
uv run python scripts/benchmark.py --full --n-matches=5 --n-battles=25

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
│   ├── battle/                     # Turn-by-turn battle policy
│   │   ├── policy.py               #  DongimonBattlePolicy (orchestrator)
│   │   ├── move_scoring.py         #  Offensive move / protect / switch scoring, threat estimation
│   │   └── joint.py               #  Joint action pairing with 9 synergy components
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
├── scripts/
│   ├── run_competition.py          # Smoke test 
│   ├── benchmark.py                # Battle royale benchmark (isolated + full modes)
│   ├── benchmark_competitors.py    # Full competitor benchmark 
│   ├── benchmark_isolated.py       # Battle-policy-only benchmark (same team)
│   ├── tune_weights.py             # Optuna 14-weight BP tuning pipeline
│   ├── tune_teambuild.py           # Optuna 12-weight teambuild tuning (400 trials)
│   ├── tune_selection.py           # Optuna 5-weight selection synergy tuning (300 trials)
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
| Selection Policy | Operational | Analytical pair-synergy fast path (≤4) + sub-tournament full path (>4), Optuna-tuned |
| Team Build Policy | Operational | 3-stage HESF pipeline with GA + battle royale, Optuna-tuned weights |
| Optuna Tuning (BP) | In Progress | 14 battle weights tuned against JJJ/minimon/StocKarpador |
| Optuna Tuning (TB) | Complete | 12 teambuild weights, best trial #192 WR=0.8060 (400 trials) |
| Optuna Tuning (Sel) | Complete | 5 selection synergy weights, best trial #211 WR=0.8083 (300 trials) |
| Benchmark Suite | Complete | Isolated + full modes, CSV logging, MLflow tracking |
| RL Training | Planned | BC pretraining -> DQN/PPO -> PSRO self-play -> Transfer learning |

---

## Extending

1. Subclass one of the ABCs in `vgc2.agent` (BattlePolicy, SelectionPolicy, TeamBuildPolicy)
2. Wire your policy into `competitor.py` as a property
3. Test with `uv run python scripts/run_competition.py`
4. Benchmark with `uv run python scripts/benchmark.py`

---

## Performance History

### v4 (pre-July 29): Teambuild ranked 4th--5th despite sophisticated pipeline

Dongimon's three-stage HESF teambuild produced teams that consistently underperformed against simpler competitors under both Greedy and Dongimon battle pilots.

| Benchmark | Dongimon Rank | Dongimon WR | vs JJJ | vs caaaden |
|-----------|--------------|-------------|--------|------------|
| Greedy BP (teambuild+selection only) | 4th | 0.423 | 0.453 | 0.277 |
| Dongimon BP (full pipeline) | 5th | 0.253 | 0.090 | 0.177 |

**Diagnosis:** The archetype upgrade function (`get_optimal_archetype`) systematically favoured bulky-offense builds over speed-offense builds through three compounding biases:

1. **Level mismatch (critical):** `build_coefficient_table` calculated damage using level-50 defenders against level-100 attackers, inflating damage estimates by ~2×. When every archetype appeared to OHKO every opponent, the `w_dmg` fitness component dominated and speed differentiation was washed out.

2. **Weight imbalance:** The bulky camp (`w_dmg=0.36` + `w_util=0.16` = 0.52) outweighed the speed camp (`w_speed=0.24` + `w_stat=0.16` = 0.40) by +30%. Jolly/Timid fast sweepers could never outscore Adamant/Modest bulky attackers in the archetype fitness evaluation.

3. **GA fitness misalignment:** The genetic algorithm weighted defensive synergy and type coverage breadth (0.38 combined) more heavily than raw species viability (0.30). JJJ and caaaden succeed by prioritising individual species power (BST, offensive firepower) over team-level synergy.

### v5 (July 29): Weight and coefficient fixes

**Fixes applied** (6 files, ~25 lines changed):

| File | Change | Reason |
|------|--------|--------|
| `src/teambuild/scoring.py` | `level_factor` 50→100 in `build_coefficient_table` | Match attacker level (lv100) for accurate damage scaling |
| `src/shared/archetypes.py` | `level` 50→100 in `create_generic_build_for_species` | Defender stats match attacker level |
| `src/teambuild/fitness.py` | `w_speed` 0.24→0.30, `w_util` 0.16→0.10, `w_dmg` 0.36→0.40, `w_stat` 0.16→0.12 | Shift archetype preference from bulky-utility (0.52) toward speed-damage |
| `src/teambuild/operators.py` | `_VIABILITY_WEIGHT` 0.30→0.35, `_COVERAGE_WEIGHT` 0.22→0.18, `_DEFENCE_WEIGHT` 0.16→0.13 | Align GA with JJJ/caaaden (more BST, less synergy) |
| `src/selection/prediction.py` | Filter builds with empty movesets in `predict_opponent_builds` | Prevent crashes when opponent species has non-standard movepools |
| `scripts/benchmark_team.py` | `_try_selection` wrapper for opponents (silent fallback); `_safe_selection` for Dongimon (loud failure) | Isolate Dongimon failures; opponent failures never terminate benchmark |

### v5 Results (10 rounds × 30 battles, same seed)

| Benchmark | Dongimon Rank | Dongimon WR | vs JJJ | vs caaaden | Change |
|-----------|--------------|-------------|--------|------------|--------|
| Greedy BP | **1st** (was 4th) | **0.666** (was 0.423) | 0.693 | 0.493 | **+57%** |
| Dongimon BP | **2nd** (was 5th) | **0.618** (was 0.253) | 0.513 | 0.363 | **+144%** |

**Head-to-head improvements:**

- vs JJJ (Dongimon BP): 0.090 → **0.513** (no longer a significant loss)
- vs caaaden (Greedy BP): 0.277 → **0.493** (near-even split)
- vs minimon (Dongimon BP): 0.250 → **0.910** (significant win)

**Teambuild team composition shifted:** Fast/Jolly/Naive/Hasty natures with 252 Spe EVs now appear regularly alongside the existing bulky ADAMANT/MODEST builds, giving the GA a genuine choice between speed and bulk.

### v6 (July 29): Optuna-tuned Teambuild + Selection weights

**Teambuild tuning** (`scripts/tune_teambuild.py`, 400 trials on EC2 c7i.xlarge):
- Objective: mean win rate of Dongimon teams (Greedy pilot) vs 4 opponents × 10 fresh rosters × 15 battles per trial
- Best trial #192: **WR = 0.8060**
- Key insight: GA heavily favours `ga_viability` (0.446) and `ga_stat_diversity` (0.418) over type coverage/defence (≤0.04 each); archetype weights favour `w_stat_syn` (0.334) and `w_speed_syn` (0.224) — synergy-aware stat/speed upgrades dominate raw damage

**Selection tuning** (`scripts/tune_selection.py`, 300 trials):
- Objective: mean win rate with analytical pair-synergy fast path (Greedy pilot) vs 4 opponents × 10 fresh rosters × 15 battles per trial
- Best trial #211: **WR = 0.8083**
- Key insight: `w_speed` (0.364) and `w_matchup` (0.346) dominate; speed control and raw matchup damage matter far more than defensive complementarity (0.073) or coverage breadth (0.076)

**Tuned weight files:**
- `src/config/teambuild_weights.yaml` — 12 weights (6 archetype + 6 GA fitness)
- `src/config/selection_synergy.yaml` — 5 term weights + fixed avg/worst blend (0.6/0.4)

---

<p align="center">
  <sub>Built with Python 3.11+ &middot; vgc2 v2.1.3 &middot; MLflow &middot; Optuna &middot; Pydantic &middot; YAML</sub>
</p>
