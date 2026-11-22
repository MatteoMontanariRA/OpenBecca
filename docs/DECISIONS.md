### Empirical Ablation Results (10,000 duplicate hands per hypothesis)
- **Baseline**: `HeuristicBot` beats `RandomBot` **7.481 vs 3.664 pts/hand** (+3.817 edge, 80.5% win rate, 6,571 hands/s on i7-1260P).
- **Confirmed Winners**: `H01` (+0.374), `H02` (+0.207), `H08` (+0.160), `H04` (+0.065), `H12` (+0.065), `H10` (+0.039).
- **Disproved Hypothesis**: `H06` (**-0.165 pts/hand**, 47.8% win rate). Jumping up with `Tre` in 2nd seat against an opponent's Case-B `Busso` (and/or returning the suit) flushes out our own `Tre` on a low card and promotes the caller's `Due`. Disabled by default in `HeuristicConfig`.
- **Phase 2 Search Milestone**: `ISMCTSBot` (120 sims/move, belief-filtered SO-ISMCTS) beats `HeuristicBot` **6.060 vs 5.060 pts/hand** (+1.000 net edge, 57.0% win rate across 100 duplicate hands). Position analysis runs 400 simulations in ~0.42s on Intel i7-1260P.
- **Phase 3 RL & AlphaZero Milestone**:
- Trained `PPOBot` via 1h CPU League Self-Play (~250k hands), reaching `-0.340` pts/hand (`47.5%` win rate) vs `HeuristicBot` at ~83 hands/s.
- Built `NeuralISMCTSBot` (PUCT + belief filtering + `prior_temp=1.5`) and `expert_iter.py` (multi-core Expert Iteration distilling 1,200 `ISMCTSBot` hands / 34,024 decisions into `az_gen1_sharp.pt` with `--pi-sharpen 4.0 --value-coef 0.05`).
- Promoted `rollout_weight=1.0` (neural policy priors + full determinized rollouts) to default: **`AlphaZero` (`az_gen1_sharp.pt`) beats `ISMCTSBot` `5.655` vs `5.435` pts/hand (`+0.220` net edge, `51.5%` win rate across 200 duplicate hands at 120 sims/move)**.
- Integrated `AlphaZero` and `PPOBot` (with automatic checkpoint discovery and NN Prior % display) into `ui/app.py` and `benchmark.py`.
### Phase 2 Decision Lab — HeuristicBotV2 Stockfish Audit (Trump & Opening)
- Built `HeuristicBotV2` (formal 12-tier trump selection, Trick-1 opening plans, exact 10-trump counter, and suit card-counting) and `decision_lab.py` for paired Monte Carlo / `ISMCTSBot` A/B experiments (`EXP-01`..`EXP-12`).
- **Stockfish-Validated Promotions**:
  - `EXP-05`: Lowered length override threshold from `6` to `5` cards (**+2.600 +/- 0.90** margin, `88.0% vs 69.0%` win rate under `ISMCTSBot`).
  - `EXP-01` & `EXP-01-FIXED`: Eliminated calling bare 2-card `3+2` as trump over any 3-card suit (**+1.860 +/- 0.91** general, **+1.200 +/- 1.04** canonical hand under `ISMCTSBot`).
  - `EXP-06`: Switched `>= 4` trumps holding `3+A` Trick-1 lead to lowest trump (`Striscio`, saving `Tre` for `H04`), gaining **+4.2–4.8% win rate** across 3 independent tests (`88.4% vs 83.6%` over 250 `ISMCTSBot` deals).
- **Confirmed Human Rules**: `EXP-02`, `EXP-04`, `EXP-08`, `EXP-09` (`-0.700` penalty for bare `3+A` lead), `EXP-10`, and `EXP-11` (`ISMCTSBot` disproved the heuristic false positive on 4–5 non-piece trumps).