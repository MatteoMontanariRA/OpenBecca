# OpenBecca — Project Roadmap

## Phase 1: Core Engine & Heuristic Baseline (Completed)
- [x] Repository setup, architecture, and decision logs
- [x] `cards.py`: 40-card deck, exact strength & integer-thirds scoring, Italian written names
- [x] `signals.py`: Strict v1 automatic signaling (`Busso > Volo > Striscio`)
- [x] `rules.py`: Legal move validation, trick resolution, Cricca & Call-out mechanics
- [x] `state.py` & `engine.py`: Imperfect-information `Observation`, full hand/match loop, JSON match logs
- [x] `RandomBot` & `HeuristicBot` (encoding Hypotheses H01–H12)
- [x] Parallel benchmark runner (`benchmark.py`) utilizing multi-core CPU

## Phase 2: Logic Search Engine ("Stockfish Mode") & Web UI (Nearly Complete)
- [x] `ISMCTSBot`: Information Set Monte Carlo Tree Search with belief filtering (respecting voids, `Busso`, `Volo`, `Striscio`, and revealed `Cricca`)
- [x] `evaluator.py`: Move ranking with expected points (EV) and win probability per legal card
- [x] Custom position/hand editor: input any table state and query the AI for best play
- [x] Local Streamlit Web UI (`ui/app.py`): Play 1v3, Watch 4 AI, Analyze, Custom Setup
- [ ] Ablation report: evaluate Hypotheses H01–H12 against `ISMCTSBot`

## Phase 3: Advanced Gameplay & Self-Learning AI (In Progress)
- [x] **Self-Play Reinforcement Learning (PyTorch Masked PPO)**: 250-dim imperfect-information encoder, masked Actor-Critic network, League self-play trainer (`--resume`, `--free-busso`), and `PPOBot`
- [x] **AlphaZero-Style Neural IS-MCTS & Expert Iteration**: Implemented PUCT `NeuralISMCTSBot`, multi-core `expert_iter.py` distillation (`az_gen1_sharp.pt` beats `ISMCTSBot` by `+0.220` pts/hand at 120 sims), and integrated `AlphaZero` + `PPOBot` into `benchmark.py` and `ui/app.py`
- [ ] **Next-Gen RL Performance**: Delayed internal node expansion for 2–3× faster `NeuralISMCTSBot`, Perfect-Information Critic for strong pure-neural (`rollout_weight=0.0`) search, and Generation 2 Expert Iteration
- [ ] **Relax `Busso` constraint across all AI**: Allow `ISMCTSBot` and UI bots (not just `--free-busso` in PPO) to decide autonomously when obeying partner's `Busso` is optimal
- [ ] **Optional / Tactical Signaling & Bluffing**: Allow choosing silence over `Busso`, configurable *Striscio* thresholds, and optional bluff rules
- [ ] **41-Point Macro Match Optimization**: Train/adapt AI to shift strategy based on global match score (e.g., defensive play at 39–38 vs aggressive Cricca/Call-out decisions)
- [ ] **Explainable Move Commentary**: Natural-language breakdown of *why* the AI prefers Move A over Move B (e.g., "protects Asso di Coppe", "avoids enemy void in Spade")