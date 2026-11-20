# OpenBecca — Project Roadmap

## Phase 1: Core Engine & Heuristic Baseline (Current)
- [x] Repository setup, architecture, and decision logs
- [x] `cards.py`: 40-card deck, exact strength & integer-thirds scoring, Italian written names
- [x] `signals.py`: Strict v1 automatic signaling (`Busso > Volo > Striscio`)
- [x] `rules.py`: Legal move validation, trick resolution, Cricca & Call-out mechanics
- [x] `state.py` & `engine.py`: Imperfect-information `Observation`, full hand/match loop, JSON match logs
- [x] `RandomBot` & `HeuristicBot` (encoding Hypotheses H01–H12)
- [x] Parallel benchmark runner (`benchmark.py`) utilizing multi-core CPU

## Phase 2: Logic Search Engine ("Stockfish Mode") & Web UI
- [x] `ISMCTSBot`: Information Set Monte Carlo Tree Search with belief filtering (respecting voids, `Busso`, `Volo`, `Striscio`, and revealed `Cricca`)
- [x] `evaluator.py`: Move ranking with expected points (EV) and win probability per legal card
- [x] Custom position/hand editor: input any table state and query the AI for best play
- [x] Local Streamlit Web UI (`ui/app.py`): Play 1v3, Watch 4 AI, Analyze, Custom Setup
- [ ] Ablation report: evaluate Hypotheses H01–H12 against `ISMCTSBot`

## Phase 3: Advanced Gameplay & Strategic Depth (Future Upgrades)
- [ ] **Relax `Busso` constraint for AI**: Allow AI agents to decide autonomously whether obeying partner's `Busso` is optimal (e.g., ignoring it when holding only low cards or facing a certain enemy trump).
- [ ] **Optional / Tactical Signaling & Bluffing**: Allow choosing silence over `Busso`, configurable *Striscio* thresholds, and optional bluff rules.
- [ ] **41-Point Macro Match Optimization**: Train/adapt AI to shift strategy based on global match score (e.g., defensive play at 39–38 vs aggressive Cricca/Call-out decisions).
- [ ] **Pure Reinforcement Learning Environment**: Expose a `Gymnasium`-compatible environment and train self-play neural networks (PyTorch PPO / Deep CFR / AlphaZero-style IS-MCTS + Value/Policy Net).
- [ ] **Explainable Move Commentary**: Natural-language breakdown of *why* the AI prefers Move A over Move B (e.g., "protects Asso di Coppe", "avoids enemy void in Spade").