# VGC 2025 Championship Competitors — Benchmark Reference

Source: `edition/vgc2025/` in the vgc2 engine repository.

12 competitors competed in a 25-wave round-robin championship with a
51-species procedurally-generated roster and 101-move move pool.

---

## Top 3 (Study Priority)

### 1. JJJ (ELO 1606.59) — `competitor1_jjj.py`
**Champion.** By JunSung Kim (DONGMIN KIM).

- **Battle:** Greedy focus-fire tactics using a hardcoded type chart
  (`DAMAGE_MULTIPLICATION_ARRAY` copied from vgc2). Computes defensive
  multipliers from base stats, picks moves maximizing type advantage.
- **Selection:** Custom selection based on defensive matchups against
  the opponent's team composition.
- **Teambuild:** Max firepower — prioritises high base stat Pokémon
  with complementary types.
- **Key insight:** Defensive multiplier computed from ratio of HP/Def
  to maximum values (402 HP, 257 Def). Simple but effective.

### 2. minimon (ELO 1215.19) — `competitor2_minimon.py`
**2nd place.** By Leon Brunke.

- **Battle:** Greedy expected-damage maximization with status weighting.
  `expected_damage()` calculates damage × accuracy + status bonus
  (20–60 depending on status type). Uses `argmax` for move selection.
- **Selection:** `DiverseTypeSelectionPolicy` — picks Pokémon that
  cover as many different types as possible in the team preview.
- **Teambuild:** `StrongestTeamBuildPolicy` — picks the species with
  the highest sum of base stats.
- **Key insight:** Status bonus weighting (Burn=20, Paralysis=30,
  Sleep=60) provides a simple heuristic for prioritising utility moves.

### 3. StocKarpador (ELO 1211.77) — `competitor3_stockarpador.py`
**3rd place.** By Fidelio Luc Reichard and Malte Rost.

- **Battle:** `MonteCarloBattlePolicy` — tree search using `copy_state()`
  and `forward()` from `vgc2.util.forward`. Evaluates state by a weighted
  heuristic: `3 * my_hp - 5 * opp_hp + 2 * my_alive - 2.5 * opp_alive`.
  Uses `ZERO_RNG` for deterministic rollouts. Runs concurrent futures
  for parallelism.
- **Selection:** Heuristic — picks active Pokémon based on type
  advantage against opponent leads.
- **Teambuild:** Heuristic — favours high-stat species with balanced
  type coverage.
- **Key insight:** Single state evaluation function combining HP ratio,
  alive count, and type advantage. Monte Carlo lookahead with
  deterministic RNG for reproducibility.

---

## Other Competitors

### Botzilla (ELO 1211.41) — `competitor_botzilla.py`
- **Battle:** `QTableBattlePolicy` — loads a pre-trained Q-table
  (`q_table.csv`, 6,094 rows) and uses `numpy.argmax` over action
  values for a given encoded state.
- **Selection:** `BalancedStatSelectionPolicy` — stat-based ranking.
- **Teambuild:** `EducatedTeamBuildPolicy` — type-coverage-aware roster.
- **Key insight:** The only submission using tabular RL (Q-learning).
  State encoding via `vgc2.util.encoding.EncodeContext`.

### IceMonte (ELO 1187.90) — `competitor_icemonte.py`
- Custom battle policy with a 3-phase switch logic:
  1. If any own Pokémon is weak to opponent → switch
  2. If HP < 30% → switch to preserve it
  3. Otherwise: greedy damage
- Custom selection and team build policies.

### Peach (ELO 1179.64) — `competitor_peach.py`
- By Lilly Gerlach and Anna-Lena Penk.
- Custom battle, selection, and team build policies.
- Battle policy uses greedy damage with status move awareness.

### Smart Jirachi (ELO 1159.90) — `competitor_jirachi.py`
- **Battle:** Always Smart Beam Search — beam search over possible
  actions with limited budget (70ms per turn).
- **Selection:** Max Firepower — picks the strongest 4 Pokémon by
  combined base stats.
- **Teambuild:** Max Firepower — picks species with highest
  total base stats.

### Caaaden (ELO 1162.53) — `competitor_caaaden.py`
- Custom battle policy with Korean comments.
- Finds best move by iterating over all moves and targets, picking
  the combination with highest calculated damage.
- Fallback switch logic when a Pokémon faints.

### Laze (ELO 1095.19) — `competitor_laze.py`
- Custom battle, selection, and team build policies.

### EvoTrainer — `competitor_evo.py`
- Evolutionary strategy battle policy with evolved hyperparameters
  stored in `genes.npy`. No team build policy (uses random).
- Designed for battle track, not championship. No ELO in championship.

### Yamabuki — `competitor_yamabuki.py`
- **Battle Track winner**, not a championship competitor.
- Monte Carlo Tree Search with a trained LogisticRegression
  win-rate predictor (164 features extracted from state).
- Uses `monte_carlo_multi_process_battle_policy.py` with configurable
  `decision_time`, `rollout_turns`, and `c_puct`.
- Selection and team build are random.

---

## Missing Competitors (No Source Available)

| Name | ELO | Notes |
|------|-----|-------|
| Wolfe | 1230.74 | 2nd place overall. No submission folder. |
| Example 11 | 1063.97 | Built-in template ExampleCompetitor. |
| higmon | 1075.16 | No submission folder. |

---

## Running a Benchmark Match

```python
from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team
from competitors.competitor1_jjj import JJJ_Competitor
from competitor import DongimonCompetitor

dongimon = CompetitorManager(DongimonCompetitor("Dongimon"))
jjj = CompetitorManager(JJJ_Competitor())
match = Match((dongimon, jjj), n_battles=3, gen=gen_team)
match.run()
print(f"Dongimon wins: {match.wins[0]}, JJJ wins: {match.wins[1]}")
```
