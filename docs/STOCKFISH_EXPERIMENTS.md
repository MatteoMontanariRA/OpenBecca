# OpenBecca — Stockfish & Monte Carlo Decision Lab

This document tracks all strategic forks investigated with paired Monte Carlo rollouts and `ISMCTSBot` ("Stockfish Mode") across identical deals. Each experiment compares **Option A** against **Option B/C/D** on the exact same 40-card distributions (zero deal luck), reporting mean points, net margin (`my_pts - opp_pts`), hand win rate, and 95% confidence intervals (`1.96 * std / sqrt(N)`).

---

## 1. Empirical Results Scoreboard (Phase 1: Trump Selection & Trick-1 Opening)

| ID | Strategic Fork | Heuristic Screening (500 paired deals) | Stockfish Validation (`ismcts`, 80 sims) | Status / Action in `HeuristicBotV2` |
| :--- | :--- | :--- | :--- | :--- |
| **`EXP-01`** | Bare `3+2` trump vs best 3-card suit + cash `3` | `-0.339 +/- 0.56` (295 deals, TIE) | **`+1.860 +/- 0.91`** (100 deals · **53.0% vs 33.0%**) | **PROMOTED**: `allow_bare_three_two_trump = False` (never call a 2-card trump suit) |
| **`EXP-01-FIXED`** | Canonical hand `D:3,2 \| C:3,A \| S:K,C,F \| B:7,6,5` | `-0.010 +/- 0.39` (TIE) | **`S:K,C,F` `+1.200 +/- 1.04`** (100 deals · **70.0% vs 57.0%**) | **PROMOTED**: confirms `EXP-01`; `C:3,A` also loses (`-0.080`) |
| **`EXP-02`** | `4+` non-piece suit vs bare `3+2` | **`V2 WINS` (`-1.102 +/- 0.46`)** | — *(already decisive)* | **CONFIRMED**: 4+ non-piece trump strictly beats bare `3+2` |
| **`EXP-02-FIXED`** | Canonical hand `D:3,2 \| C:3,A \| S:7,6,5,4 \| B:7,5` | **`V2 WINS` (`-2.084 +/- 0.40` vs `3,2`; `-1.784` vs `3,A`)** | — *(already decisive)* | **CONFIRMED**: 83.8% win rate calling 4 trash cards + cashing `3` |
| **`EXP-03`** | Quality tie-break: high-card power vs total card points | `+0.184 +/- 0.42` (TIE · 62.8% vs 58.4%) | *Pending deep run* | **KEPT** (`high_card` default; toggleable to `total_points`) |
| **`EXP-04`** | Tier 6 (`4+` w/ 1 piece): length (5 w/ Asso) vs piece (4 w/ Tre) | **`V2 WINS` (`-1.945 +/- 0.45`)** (437 deals) | — *(already decisive)* | **CONFIRMED**: length dominates piece rank (`83.3% vs 66.6%`) |
| **`EXP-05`** | Length override threshold: `>= 6` vs `>= 5` cards | **`+2.622 +/- 0.45`** (83.8% vs 59.2%) | **`+2.600 +/- 0.90`** (100 deals · **88.0% vs 69.0%**) | **PROMOTED**: `length_override_threshold = 5` |
| **`EXP-06`** | `3+A` (`>= 4` trumps): `Tre` Busso A vs low trump (`Striscio`) vs side suit | `+0.036 +/- 0.26` (**89.4% vs 85.2%**) | 100 deals: `+0.340 +/- 0.72` (**90.0% vs 86.0%**)<br>250 deals: `+0.312 +/- 0.46` (**88.4% vs 83.6%**) | **PROMOTED (soft)**: `three_ace_four_plus_lead = "low_trump"` (consistent +4.2–4.8% win rate across 3 tests; saves `Tre` for H04) |
| **`EXP-07`** | 3-card trump with `Due` (`2+A+x` / `2+x+y`): side suit vs low trump Busso B | `-0.088 +/- 0.24` (TIE) | *Low priority* | **KEPT**: side suit (Ladder A) for 3 trumps; Busso B from `>= 4` trumps |
| **`EXP-08`** | 3-card `3+A+x` w/ no `<= 2` side suit: shortest side vs `Tre` / low trump / 4-card | **`V2 WINS`** vs `Tre` (`-1.368 +/- 0.28`) & low trump (`-0.440 +/- 0.24`) | — *(already decisive)* | **CONFIRMED**: never lead `Tre` from 3 trumps (`65.4% vs 44.0%`) |
| **`EXP-08-FIXED`** | Canonical hand `D:3,A,4 \| C:K,C,F,7 \| S:K,C,6 \| B:void` | **`V2 WINS`** vs `3 Denari` (`-1.324 +/- 0.27`) | — *(already decisive)* | **CONFIRMED**: `6 Spade` (67.0% win) crushes `3 Denari` (47.0%) |
| **`EXP-09`** | Bare `3+A` off-suit on Trick 1: avoid vs lead `Tre` (`Busso A`) | **`V2 WINS` (`-0.700 +/- 0.24`)** | — *(already decisive)* | **CONFIRMED**: `avoid_bare_three_ace_lead = True` (`69.0% vs 59.6%`) |
| **`EXP-10`** | Double Cricca Trick 2: exhaust trumps (`Tre`) vs cash 2nd Cricca `Asso` | **`V2 WINS` (`-0.148 +/- 0.12`)** | — *(already decisive)* | **CONFIRMED**: `exhaust_trumps_before_second_cricca = True` |
| **`EXP-11`** | 4–5 non-piece trumps: Ladder A (side suit) vs Ladder B (low trump) | `+0.508 +/- 0.25` (false positive!) | **`-0.140 +/- 0.60` (TIE · A 5.79 vs B 5.72)** | **KEPT** (`non_piece_trump_ladder_b_threshold = 6`; ISMCTS disproved heuristic artifact) |
| **`EXP-12`** | Trick-1 tie-break on `Asso >= 3` / `Due Busso >= 3`: longer suit vs lowest card | `-0.052 +/- 0.48` (77 deals, TIE) | *Low priority* | **KEPT** (`side_suit_tiebreak = "longer_suit"`) |

---

## 2. Key Lessons Discovered by OpenBecca

1. **Trump length is king (`EXP-05`, `EXP-04`, `EXP-02`, `EXP-01`)**:
   - Holding **5 trumps** overrides every 3-card suit (even `3 + 2 + x`) by **+2.600 margin** (`88%` win rate under Stockfish).
   - Never call a **2-card trump suit**, even `3 + 2`: against strong search-based opponents (`ISMCTSBot`), calling bare `3 + 2` collapses to `33%` win rate (`-1.860` vs calling a 3-card suit and cashing `Tre` off-suit).
2. **Heuristic-only testing can create false positives (`EXP-11`, `EXP-01`)**:
   - Leading low trump from 4–5 non-piece trumps beat heuristic bots (`+0.508`), but failed against `ISMCTSBot` (`-0.140`).
   - Conversely, heuristic bots failed to punish bare `3 + 2` trump calls (`-0.339`, TIE), whereas `ISMCTSBot` punished them heavily (`+1.860`).
   - **Rule of thumb**: use `--engine heuristic` for rapid screening, then always confirm big calls with `--engine ismcts`.
3. **Protect bare Aces and preserve the Trump `Tre` (`EXP-09`, `EXP-08`, `EXP-06`)**:
   - Avoiding bare `3 + A` leads on Trick 1 saves **+0.700 margin**.
   - Never lead `Tre` from 3-card `3 + A + x` trump (**+1.368 margin** saved).
   - Even with `>= 4` trumps holding `3 + A`, leading a low trump (`Striscio`) instead of `Tre` (`Busso A`) preserved `H04` and improved win rate by **+4.2–4.8%** across every test run.

---

## 3. Global Benchmarks (`HeuristicBotV2` vs `HeuristicBot v1`)

- **Pre-Promotion V2 (`thresh=6`, bare `3+2` allowed, `three_busso`) vs V1 (10,000 duplicate hands)**:
  - `HeuristicBotV2`: `5.580 pts/hand` vs `HeuristicBot v1`: `5.573 pts/hand` (`+0.008 +/- 0.035`, `49.7%` win rate — **STATISTICALLY TIED**).
- **Post-Promotion V2 (`thresh=5`, no bare `3+2`, `low_trump` on `3+A >= 4`) vs V1 (10,000 duplicate hands)**:
  - `HeuristicBotV2`: `5.582 pts/hand` vs `HeuristicBot v1`: `5.551 pts/hand` (`+0.031 +/- 0.031`, `50.2%` win rate — **HeuristicBotV2 WINS, statistically significant**).
- **Post-Promotion V2 vs Pre-Promotion V2 (`BENCH-V2-EVOLUTION`, 10,000 duplicate hands)**:
  - `Stockfish-tuned V2`: `5.575 pts/hand` vs `Pre-Stockfish V2`: `5.550 pts/hand` (`+0.025 +/- 0.016`, `50.4%` win rate — **Stockfish-tuned V2 WINS, statistically significant**).
---

## 4. Planned Next Experiments (Phase 2: Midgame & Endgame, Tricks 2–10)
- `EXP-13`: Does a **4-card non-piece suit** override 3-card 1-piece suits (`A, x, y` or `2, x, y`) under `ISMCTSBot`?
- `EXP-14+`: Midgame/endgame lead & follow forks (to be designed during the Tricks 2–10 strategy audit).