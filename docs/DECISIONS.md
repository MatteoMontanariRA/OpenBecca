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