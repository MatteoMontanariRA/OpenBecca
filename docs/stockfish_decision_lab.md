# OpenBecca — Stockfish Decision Lab (Monte Carlo Experiment Registry)

This document catalogs all **12 strategic trade-offs** (`EXP-01` through `EXP-12`) plus the global benchmark (`BENCH-GLOBAL`) from the frozen Ravenna Beccacino Opening Specification (`HeuristicBotV2`), designed to be evaluated empirically via **paired counterfactual Monte Carlo simulations** ("Stockfish-style" decision testing).

All experiments strictly preserve the locked OpenBecca v1 signaling rules (`Busso` Case A, `Busso` Case B, `Volo`, `Striscio`, `None`) with full compatibility with `ISMCTSBot` and `NeuralISMCTSBot`.

---

## 1. Evaluation Protocol (Paired Counterfactual Monte Carlo)

To eliminate dealing variance and isolate pure strategic Expected Value (EV), every experiment follows the same reproducible protocol implemented in `src/openbecca/analysis/decision_lab.py`:

1. **Targeted Sampling (Trigger)**: Generate random 40-card deals (using rejection sampling or mathematically unbiased constructive sampling for rare triggers such as double *Cricca*) and retain $N$ deals where Player 0 (the trump caller and Trick 1 leader) satisfies the exact trigger condition.
2. **Paired Counterfactual Branches**: On each frozen 40-card deal, play out the complete 10-trick hand multiple times from the exact same cards, forcing **Choice A** (the `HeuristicBotV2` baseline) versus **Choice B** (and **Choice C / D** where applicable) at the decision point. All subsequent plays are completed by the same engine (`HeuristicBotV2` for fast high-$N$ screening, or `ISMCTSBot` for deep search-based verdicts) with identical RNG seeds.
3. **Paired Statistical Significance**: Because choices are evaluated on identical deals, compute the paired margin difference $\Delta_i = \text{margin}_X(i) - \text{margin}_A(i)$ per deal, along with the mean $\bar{\Delta}$, the $95\%$ confidence interval ($\bar{\Delta} \pm 1.96 \cdot \text{SE}$), and hand win rate.
4. **Structured JSONL Logging (`logs/experiments/<exp_id>.jsonl`)**: Store every simulated branch with `deal_id`, `player0_hand`, `choice`, `trump_suit`, `forced_card`, `points_team0`, `points_team1`, `margin`, `delta_vs_baseline`, and `won_hand` so results can be inspected or sliced by hand features later.

---

## 2. Experiment Registry

### EXP-01 — Bare `Tre + Due` Emergency (Only 3-Card Non-Piece Suits Available)
* **Trigger**: No `Cricca`, no $\ge 6$-card suit, no $\ge 3$-card suit with a Cricca piece, no $\ge 4$-card non-piece suit, while holding bare `Tre + Due` (2 cards).
  * *Canonical hand (`EXP-01-FIXED`)*: `Denari: Tre, Due` | `Coppe: Tre, Asso` | `Spade: Re, Cavallo, Fante` | `Bastoni: Sette, Sei, Cinque`.
* **Choices**:
  * **Choice A (Baseline V2)**: Call bare `Tre + Due` (`Denari`) and lead via Ladder A (`Cinque di Bastoni`).
  * **Choice B**: Call the best 3-card non-piece suit (`Spade`) and cash `Tre` of `Denari` on Trick 1.
  * **Choice C** *(when also holding bare `Tre + Asso`)*: Call bare `Tre + Asso` (`Coppe`) and cash `Tre` of `Denari` on Trick 1.

### EXP-02 — 4-Card Trash Suit vs Bare `Tre + Due` Emergency
* **Trigger**: No `Cricca`, no $\ge 6$-card suit, no $\ge 3$-card suit with a Cricca piece, holding both a $\ge 4$-card non-piece suit and a bare `Tre + Due` suit.
  * *Canonical hand (`EXP-02-FIXED`)*: `Denari: Tre, Due` | `Coppe: Tre, Asso` | `Spade: Re, Cavallo, Fante, Sette` | `Bastoni: Sei, Cinque`.
* **Choices**:
  * **Choice A (Baseline V2)**: Call the $\ge 4$-card non-piece suit and lead `Tre` from bare `Tre + Due`.
  * **Choice B**: Call bare `Tre + Due` and lead via Ladder A.

### EXP-03 — Suit Quality Tie-Break Metric (High Card vs Total Points)
* **Trigger**: Inside the winning trump-selection tier, two candidate suits tie in card count, and lexicographical high-card comparison (`power` tuple) selects a different suit than total point value (`sum(points_thirds)`).
  * Covers both figure ties (`Re, Quattro` vs `Cavallo, Fante`) and single-piece 4-card ties (`Tre, Sette, Sei, Cinque` vs `Asso, Re, Cavallo, Fante`).
* **Choices**:
  * **Choice A (Baseline V2)**: Highest-card comparison (poker-style `power` tuple).
  * **Choice B**: Highest total points (`sum(points_thirds)`).

### EXP-04 — Tier 6 Trade-Off: 5-Card Weak Piece vs 4-Card Strong Piece
* **Trigger**: Winning trump tier is Tier 6 ($\ge 4$ cards with $\ge 1$ Cricca piece), holding a 5-card suit with a weaker piece (`Asso` or `Due`) against a 4-card suit with a stronger piece (`Tre` or `Due`).
* **Choices**:
  * **Choice A (Baseline V2)**: Prefer **length** (5-card suit).
  * **Choice B**: Prefer **piece strength** (4-card suit with stronger Cricca piece).

### EXP-05 — Length Override Threshold (5 Cards vs 6+ Cards)
* **Trigger**: No `Cricca` and no $\ge 6$-card suit; holding a 5-card suit where a threshold of 5 overrides the 3-card piece hierarchy (e.g., `Tre, Due, Quattro` or `Tre, Sei, Quattro` vs a 5-card suit).
* **Choices**:
  * **Choice A (Baseline V2, threshold = 6)**: Follow the piece hierarchy (call the stronger 3-card suit).
  * **Choice B (threshold = 5)**: Override and call the 5-card suit.

### EXP-06 — Trick 1 with `Tre + Asso` and $\ge 4$ Trumps
* **Trigger**: Caller names a non-`Cricca` trump suit with $\ge 4$ trumps containing `Tre + Asso` (without `Due`).
* **Choices**:
  * **Choice A (Baseline V2)**: Lead `Tre` of trump (`Busso` Case A, intentional exception to `H04`).
  * **Choice B**: Lead the lowest trump (`Striscio`, saving `Tre` for Trick 10).
  * **Choice C**: Do not lead trump; open from a side suit using Ladder A.

### EXP-07 — Trick 1 with `Due` in Exactly 3 Trumps (`Due + Asso + x` or `Due + x + y`)
* **Trigger**: Caller names a non-`Cricca` trump suit with exactly 3 trumps containing `Due` (without `Tre`).
* **Choices**:
  * **Choice A (Baseline V2)**: Lead from a side suit using Ladder A (reserving trump `Busso` Case B for $\ge 4$ trumps).
  * **Choice B**: Lead the lowest trump card (`Busso` Case B).

### EXP-08 — Trick 1 with `Tre + Asso + x` (3 Trumps) and No Short Side Suit ($\le 2$ Cards)
* **Trigger**: Caller holds exactly 3 trumps (`Tre, Asso, x`) and the remaining 7 cards have a `4-3-0` or `7-0-0` distribution with no side `Tre`, no side `Due` `Busso`, no `Volo`, and no side `Asso` ($\ge 3$ cards).
  * *Canonical hand (`EXP-08-FIXED`)*: `Denari (trump): Tre, Asso, Quattro` | `Coppe: Re, Cavallo, Fante, Sette` | `Spade: Re, Cavallo, Sei` | `Bastoni: void`.
* **Choices**:
  * **Choice A (Baseline V2)**: Lead the lowest card of the shortest non-trump suit (`Sei di Spade`, silence).
  * **Choice B**: Lead `Tre` of trump (`Busso` Case A).
  * **Choice C**: Lead the lowest trump (`Quattro di Denari`, silence).
  * **Choice D**: Lead the lowest card of the 4-card side suit (`Sette di Coppe`, `Striscio`).

### EXP-09 — Bare `Tre + Asso` (2 Cards) Non-Trump Lead Avoidance on Trick 1
* **Trigger**: Caller opens via Ladder A on Trick 1, holds bare `Tre + Asso` (2 cards) in a non-trump suit, and holds no higher-priority side `Tre` (`Tre + Due` bare or `Tre + Asso + x`).
* **Choices**:
  * **Choice A (Baseline V2)**: Skip bare `Tre + Asso` and continue down Ladder A (protecting the `Asso`).
  * **Choice B (`H01` + `Busso` Case A)**: Lead `Tre` from bare `Tre + Asso` immediately.

### EXP-10 — Holding Two Criccas: Trick 2 Sequencing After Trump `Asso`
* **Trigger**: Caller holds trump `Cricca` plus at least one non-trump `Cricca`, wins Trick 1 with trump `Asso`, and at least one trump remains in other players' hands on Trick 2.
* **Choices**:
  * **Choice A (Baseline V2)**: Lead `Tre` of trump on Trick 2 (finish exhausting enemy trumps first).
  * **Choice B**: Immediately cash `Asso` of the shortest non-trump `Cricca` on Trick 2.

### EXP-11 — 4–5 Trumps Without `Asso`/`Due`: Ladder A vs Lowest Trump
* **Trigger**: Caller holds 4 or 5 trumps without `Asso` or `Due` (`Tre` alone or 0 pieces) and has no side `Tre`, no side `Due` `Busso`, and no `Volo` on Trick 1 (where Ladder A and Ladder B diverge).
* **Choices**:
  * **Choice A (Baseline V2 — Ladder A)**: Lead side `Asso` ($\ge 3$ cards), 2-card setup suit, or shortest non-trump suit.
  * **Choice B (Ladder B)**: Lead the lowest trump card immediately.

### EXP-12 — Trick 1 Tie-Break on Side `Due` (`Busso`) or Side `Asso` ($\ge 3$ Cards)
* **Trigger**: Caller opens via Ladder A, reaches Step 2 (multiple side `Due` `Busso` suits) or Step 4 (multiple side `Asso` $\ge 3$ suits), and "longest suit" picks a different suit than "lowest card to lead".
* **Choices**:
  * **Choice A (Baseline V2)**: Pick the suit with **more cards**.
  * **Choice B**: Pick the suit with the **lowest card** to play.

### BENCH-GLOBAL — Global Head-to-Head Validation
* **Objective**: Benchmark `HeuristicBotV2` against `HeuristicBot` v1 over **10,000 duplicate hands** (swapping Team 0 and Team 1 on identical deals to cancel dealing luck) and verify overall point margin and hand win rate.

---

## 3. CLI Usage

```bash
# 1. Fast global benchmark: HeuristicBotV2 vs HeuristicBot v1 (10,000 duplicate hands)
python -m openbecca.analysis.decision_lab --exp BENCH-GLOBAL --deals 5000

# 2. Run all 12 Stockfish experiments (fast screening with HeuristicBotV2 completion)
python -m openbecca.analysis.decision_lab --exp all --deals 400 --engine heuristic

# 3. Deep Stockfish-grade test on a specific decision using ISMCTSBot
python -m openbecca.analysis.decision_lab --exp EXP-01-FIXED --deals 150 --engine ismcts --simulations 80
python -m openbecca.analysis.decision_lab --exp EXP-06 --deals 150 --engine ismcts --simulations 80
```