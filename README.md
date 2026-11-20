# OpenBecca

**OpenBecca** is an open-source Python engine, AI research sandbox, and move-analysis tool for **Beccacino** (the Ravenna variant of *Marafone*, a traditional 4-player trick-taking card game from Romagna, Italy).

The long-term goal of OpenBecca is to build a "Stockfish-like" training partner and analyzer for Beccacino: play against AI, watch 4-AI simulations, set up custom table positions to inspect expected points per move, and empirically test human strategic heuristics against search-based and self-learning AI agents.

## Key Features (Planned & In Progress)
- **Strict Rules Engine**: Exact 40-card deck, integer-thirds scoring (zero floating-point rounding bugs), *Cricca* (in-hand and out-of-hand), *Chiamarsi fuori* (call-out), and strict v1 signaling (*Busso*, *Volo*, *Striscio*).
- **Imperfect-Information Architecture**: Agents only observe legal table information—never hidden hands.
- **Incremental AI Stack**:
  1. Random legal baseline (`RandomBot`)
  2. Human-strategy benchmark (`HeuristicBot`, encoding 12 testable Ravenna strategies)
  3. Imperfect-information search (`ISMCTSBot` with expected value per move)
  4. Self-learning RL agents (future roadmap)
- **Local Web UI**: Interactive Streamlit interface to play, simulate, and analyze custom scenarios.

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -e ".[dev]"
pytest