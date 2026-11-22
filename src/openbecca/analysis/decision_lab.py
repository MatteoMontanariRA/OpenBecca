"""OpenBecca Decision Lab: Stockfish & Monte Carlo paired A/B testing for strategic forks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openbecca.agents.base import Agent
from openbecca.agents.heuristic_bot import HeuristicBot
from openbecca.agents.heuristic_bot_v2 import (
    PRE_STOCKFISH_V2_CONFIG,
    HeuristicBotV2,
    HeuristicV2Config,
)
from openbecca.agents.ismcts_bot import ISMCTSBot
from openbecca.cards import ALL_CARDS, Card, Deck, Rank, Suit
from openbecca.engine import GameEngine
from openbecca.rules import has_cricca, team_of
from openbecca.signals import Signal
from openbecca.state import Observation


# ------------------------------------------------------------------
# Forced-Decision Wrapper Agent (locks specific decision, delegates rest)
# ------------------------------------------------------------------
class ForcedDecisionAgent(Agent):
    """Wraps a base agent while overriding trump, Trick-1 lead, and/or Trick-2 lead."""

    def __init__(
        self,
        base_agent: Agent,
        *,
        forced_trump: Suit | None = None,
        forced_trick1_card: Card | None = None,
        forced_trick2_card: Card | None = None,
    ) -> None:
        super().__init__(
            name=f"Forced({base_agent.name})",
            enforce_busso_constraints=base_agent.enforce_busso_constraints,
        )
        self.base_agent = base_agent
        self.forced_trump = forced_trump
        self.forced_trick1_card = forced_trick1_card
        self.forced_trick2_card = forced_trick2_card

    def choose_trump(self, obs: Observation) -> Suit:
        if self.forced_trump is not None:
            return self.forced_trump
        return self.base_agent.choose_trump(obs)

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return self.base_agent.decide_declare_cricca(obs)

    def decide_callout(self, obs: Observation) -> bool:
        return self.base_agent.decide_callout(obs)

    def choose_card(self, obs: Observation) -> Card:
        if (
            self.forced_trick1_card is not None
            and not obs.current_trick
            and obs.current_trick_number == 1
            and self.forced_trick1_card in obs.legal_cards
        ):
            return self.forced_trick1_card
        if (
            self.forced_trick2_card is not None
            and not obs.current_trick
            and obs.current_trick_number == 2
            and self.forced_trick2_card in obs.legal_cards
        ):
            return self.forced_trick2_card
        return self.base_agent.choose_card(obs)


# ------------------------------------------------------------------
# Experiment Option & Result Structures
# ------------------------------------------------------------------
@dataclass(frozen=True)
class OptionSpec:
    """Single branch inside a paired experiment."""

    code: str
    label: str
    forced_trump: Suit | None = None
    forced_trick1_card: Card | None = None
    forced_trick2_card: Card | None = None


@dataclass(frozen=True)
class ExperimentOptionSummary:
    code: str
    label: str
    avg_points: float
    avg_margin: float
    win_rate: float
    delta_vs_a: float
    ci95_vs_a: float
    verdict: str


@dataclass(frozen=True)
class ExperimentReport:
    exp_id: str
    engine_name: str
    simulations: int
    num_deals: int
    elapsed_seconds: float
    options: tuple[ExperimentOptionSummary, ...]
    log_path: Path | None = None

    def format_table(self) -> str:
        sim_txt = f" ({self.simulations} sims)" if self.engine_name == "ismcts" else ""
        lines = [
            f"\n=== {self.exp_id} | Engine: {self.engine_name}{sim_txt} | "
            f"{self.num_deals} paired deals in {self.elapsed_seconds:.2f}s ===",
            "-" * 112,
            f"{'Opt':<3} {'Choice':<46} {'AvgPts':>7} {'Margin':>8} {'Win%':>7} "
            f"{'Delta vs A':>10} {'95% CI':>10}  {'Verdict'}",
            "-" * 112,
        ]
        for idx, opt in enumerate(self.options):
            if idx == 0:
                delta_str = "BASELINE"
                ci_str = "---"
            else:
                delta_str = f"{opt.delta_vs_a:+10.3f}"
                ci_str = f"+/- {opt.ci95_vs_a:5.2f}"
            lines.append(
                f"{opt.code:<3} {opt.label[:46]:<46} {opt.avg_points:7.2f} "
                f"{opt.avg_margin:+8.2f} {opt.win_rate * 100:6.1f}% "
                f"{delta_str:>10} {ci_str:>10}  {opt.verdict}"
            )
        lines.append("-" * 112)
        if self.log_path is not None:
            lines.append(f"Saved detailed JSONL log -> {self.log_path}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Helper builders for canonical fixed hands & deal filters
# ------------------------------------------------------------------
def _make_dummy_obs(hand: list[Card], trump_suit: Suit | None = None) -> Observation:
    counts = {s: sum(1 for c in hand if c.suit == s) for s in Suit}
    return Observation(
        player_id=0,
        team_id=0,
        hand=tuple(hand),
        initial_suit_counts=counts,
        hand_number=1,
        trump_caller=0,
        trump_suit=trump_suit,
        cricca_declared_by=(
            0 if (trump_suit is not None and has_cricca(hand, trump_suit)) else None
        ),
        completed_tricks=(),
        current_trick=(),
        current_trick_leader=0,
        current_trick_signal=Signal.NONE,
        played_cards=frozenset(),
        match_scores=(0, 0),
        hand_thirds=(0, 0),
        target_score=41,
        legal_cards=tuple(hand),
    )


def _deal_with_fixed_p0(p0_hand: list[Card], rng: random.Random) -> list[list[Card]]:
    remaining = [c for c in ALL_CARDS if c not in p0_hand]
    rng.shuffle(remaining)
    return [
        sorted(p0_hand, key=lambda c: c.sort_key()),
        sorted(remaining[0:10], key=lambda c: c.sort_key()),
        sorted(remaining[10:20], key=lambda c: c.sort_key()),
        sorted(remaining[20:30], key=lambda c: c.sort_key()),
    ]


# Canonical fixed hands from strategy audit
FIXED_HAND_EXP01 = [
    Card(Suit.COINS, Rank.THREE),
    Card(Suit.COINS, Rank.TWO),
    Card(Suit.CUPS, Rank.THREE),
    Card(Suit.CUPS, Rank.ACE),
    Card(Suit.SWORDS, Rank.KING),
    Card(Suit.SWORDS, Rank.KNIGHT),
    Card(Suit.SWORDS, Rank.KNAVE),
    Card(Suit.CLUBS, Rank.SEVEN),
    Card(Suit.CLUBS, Rank.SIX),
    Card(Suit.CLUBS, Rank.FIVE),
]

FIXED_HAND_EXP02 = [
    Card(Suit.COINS, Rank.THREE),
    Card(Suit.COINS, Rank.TWO),
    Card(Suit.CUPS, Rank.THREE),
    Card(Suit.CUPS, Rank.ACE),
    Card(Suit.SWORDS, Rank.SEVEN),
    Card(Suit.SWORDS, Rank.SIX),
    Card(Suit.SWORDS, Rank.FIVE),
    Card(Suit.SWORDS, Rank.FOUR),
    Card(Suit.CLUBS, Rank.SEVEN),
    Card(Suit.CLUBS, Rank.FIVE),
]

FIXED_HAND_EXP08 = [
    Card(Suit.COINS, Rank.THREE),
    Card(Suit.COINS, Rank.ACE),
    Card(Suit.COINS, Rank.FOUR),
    Card(Suit.CUPS, Rank.KING),
    Card(Suit.CUPS, Rank.KNIGHT),
    Card(Suit.CUPS, Rank.KNAVE),
    Card(Suit.CUPS, Rank.SEVEN),
    Card(Suit.SWORDS, Rank.KING),
    Card(Suit.SWORDS, Rank.KNIGHT),
    Card(Suit.SWORDS, Rank.SIX),
]


def _extract_options_for_hand(exp_id: str, hand: list[Card]) -> list[OptionSpec] | None:
    """Return paired OptionSpecs if `hand` triggers `exp_id`, else None."""
    obs_no_trump = _make_dummy_obs(hand, trump_suit=None)

    if exp_id == "EXP-01-FIXED":
        return [
            OptionSpec(
                "A",
                "Call Spade (K,C,F) + cash 3 Denari (V2)",
                forced_trump=Suit.SWORDS,
                forced_trick1_card=Card(Suit.COINS, Rank.THREE),
            ),
            OptionSpec(
                "B",
                "Call Denari (3,2) + lead 5 Bastoni",
                forced_trump=Suit.COINS,
                forced_trick1_card=Card(Suit.CLUBS, Rank.FIVE),
            ),
            OptionSpec(
                "C",
                "Call Coppe (3,A) + cash 3 Denari",
                forced_trump=Suit.CUPS,
                forced_trick1_card=Card(Suit.COINS, Rank.THREE),
            ),
        ]

    if exp_id == "EXP-02-FIXED":
        return [
            OptionSpec(
                "A",
                "Call Spade (4 trash) + cash 3 Denari (V2)",
                forced_trump=Suit.SWORDS,
                forced_trick1_card=Card(Suit.COINS, Rank.THREE),
            ),
            OptionSpec(
                "B",
                "Call Denari (3,2) + lead 5 Bastoni",
                forced_trump=Suit.COINS,
                forced_trick1_card=Card(Suit.CLUBS, Rank.FIVE),
            ),
            OptionSpec(
                "C",
                "Call Coppe (3,A) + cash 3 Denari",
                forced_trump=Suit.CUPS,
                forced_trick1_card=Card(Suit.COINS, Rank.THREE),
            ),
        ]

    if exp_id == "EXP-08-FIXED":
        return [
            OptionSpec(
                "A",
                "Lead 6 Spade (shortest side, V2)",
                forced_trump=Suit.COINS,
                forced_trick1_card=Card(Suit.SWORDS, Rank.SIX),
            ),
            OptionSpec(
                "B",
                "Lead 3 Denari (Trump Busso A)",
                forced_trump=Suit.COINS,
                forced_trick1_card=Card(Suit.COINS, Rank.THREE),
            ),
            OptionSpec(
                "C",
                "Lead 4 Denari (low trump)",
                forced_trump=Suit.COINS,
                forced_trick1_card=Card(Suit.COINS, Rank.FOUR),
            ),
            OptionSpec(
                "D",
                "Lead 7 Coppe (4-card Striscio)",
                forced_trump=Suit.COINS,
                forced_trick1_card=Card(Suit.CUPS, Rank.SEVEN),
            ),
        ]

    if exp_id == "EXP-01":
        bot_a = HeuristicBotV2(config=HeuristicV2Config(allow_bare_three_two_trump=False))
        bot_b = HeuristicBotV2(config=HeuristicV2Config(allow_bare_three_two_trump=True))
        t_a = bot_a.choose_trump(obs_no_trump)
        t_b = bot_b.choose_trump(obs_no_trump)
        if t_a == t_b:
            return None
        c_a = bot_a.choose_card(_make_dummy_obs(hand, trump_suit=t_a))
        c_b = bot_b.choose_card(_make_dummy_obs(hand, trump_suit=t_b))
        return [
            OptionSpec("A", "Call best 3-card suit + cash bare 3 (V2)", t_a, c_a),
            OptionSpec("B", "Call bare 3+2 suit instead", t_b, c_b),
        ]

    if exp_id == "EXP-02":
        bot_a = HeuristicBotV2(
            config=HeuristicV2Config(
                allow_bare_three_two_trump=True,
                four_non_piece_beats_bare_three_two=True,
            )
        )
        bot_b = HeuristicBotV2(
            config=HeuristicV2Config(
                allow_bare_three_two_trump=True,
                four_non_piece_beats_bare_three_two=False,
            )
        )
        t_a = bot_a.choose_trump(obs_no_trump)
        t_b = bot_b.choose_trump(obs_no_trump)
        if t_a == t_b:
            return None
        c_a = bot_a.choose_card(_make_dummy_obs(hand, trump_suit=t_a))
        c_b = bot_b.choose_card(_make_dummy_obs(hand, trump_suit=t_b))
        return [
            OptionSpec("A", "Call 4+ non-piece suit (V2)", t_a, c_a),
            OptionSpec("B", "Call bare 3+2 suit instead", t_b, c_b),
        ]

    if exp_id == "EXP-03":
        bot_a = HeuristicBotV2(config=HeuristicV2Config(quality_metric="high_card"))
        bot_b = HeuristicBotV2(config=HeuristicV2Config(quality_metric="total_points"))
        t_a = bot_a.choose_trump(obs_no_trump)
        t_b = bot_b.choose_trump(obs_no_trump)
        if t_a == t_b:
            return None
        c_a = bot_a.choose_card(_make_dummy_obs(hand, trump_suit=t_a))
        c_b = bot_b.choose_card(_make_dummy_obs(hand, trump_suit=t_b))
        return [
            OptionSpec("A", "High-card quality (V2)", t_a, c_a),
            OptionSpec("B", "Total-points quality", t_b, c_b),
        ]

    if exp_id == "EXP-04":
        # Need length_override_threshold=6 to isolate Tier 6 (5 cards w/ piece vs 4 w/ piece)
        bot_a = HeuristicBotV2(
            config=HeuristicV2Config(
                length_override_threshold=6, prefer_length_at_tier6=True
            )
        )
        bot_b = HeuristicBotV2(
            config=HeuristicV2Config(
                length_override_threshold=6, prefer_length_at_tier6=False
            )
        )
        t_a = bot_a.choose_trump(obs_no_trump)
        t_b = bot_b.choose_trump(obs_no_trump)
        if t_a == t_b:
            return None
        c_a = bot_a.choose_card(_make_dummy_obs(hand, trump_suit=t_a))
        c_b = bot_b.choose_card(_make_dummy_obs(hand, trump_suit=t_b))
        return [
            OptionSpec("A", "Longer suit (5+ cards, V2)", t_a, c_a),
            OptionSpec("B", "Stronger piece (4 cards)", t_b, c_b),
        ]

    if exp_id == "EXP-05":
        bot_a = HeuristicBotV2(config=HeuristicV2Config(length_override_threshold=5))
        bot_b = HeuristicBotV2(config=HeuristicV2Config(length_override_threshold=6))
        t_a = bot_a.choose_trump(obs_no_trump)
        t_b = bot_b.choose_trump(obs_no_trump)
        if t_a == t_b:
            return None
        c_a = bot_a.choose_card(_make_dummy_obs(hand, trump_suit=t_a))
        c_b = bot_b.choose_card(_make_dummy_obs(hand, trump_suit=t_b))
        return [
            OptionSpec("A", "Override with 5-card suit (thresh=5, V2)", t_a, c_a),
            OptionSpec("B", "Keep piece hierarchy (thresh=6)", t_b, c_b),
        ]

    if exp_id == "EXP-06":
        bot_low = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_four_plus_lead="low_trump")
        )
        t = bot_low.choose_trump(obs_no_trump)
        t_cards = [c for c in hand if c.suit == t]
        tranks = {c.rank for c in t_cards}
        if not (len(t_cards) >= 4 and Rank.THREE in tranks and Rank.ACE in tranks and Rank.TWO not in tranks):
            return None
        bot_busso = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_four_plus_lead="three_busso")
        )
        bot_side = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_four_plus_lead="side_suit")
        )
        obs_t = _make_dummy_obs(hand, trump_suit=t)
        c_a = bot_low.choose_card(obs_t)
        c_b = bot_busso.choose_card(obs_t)
        c_c = bot_side.choose_card(obs_t)
        return [
            OptionSpec("A", "Lead lowest trump (Striscio, save Tre, V2)", t, c_a),
            OptionSpec("B", "Lead Tre trump (Busso Case A)", t, c_b),
            OptionSpec("C", "Lead side suit (Ladder A)", t, c_c),
        ]

    if exp_id == "EXP-07":
        bot_a = HeuristicBotV2(config=HeuristicV2Config(lead_trump_on_three_card_due=False))
        bot_b = HeuristicBotV2(config=HeuristicV2Config(lead_trump_on_three_card_due=True))
        t = bot_a.choose_trump(obs_no_trump)
        t_cards = [c for c in hand if c.suit == t]
        tranks = {c.rank for c in t_cards}
        if not (len(t_cards) == 3 and Rank.TWO in tranks and Rank.THREE not in tranks):
            return None
        obs_t = _make_dummy_obs(hand, trump_suit=t)
        c_a = bot_a.choose_card(obs_t)
        c_b = bot_b.choose_card(obs_t)
        if c_a == c_b:
            return None
        return [
            OptionSpec("A", "Lead side suit (Ladder A, V2)", t, c_a),
            OptionSpec("B", "Lead lowest trump (Busso Case B)", t, c_b),
        ]

    if exp_id == "EXP-08":
        bot_a = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_three_card_fallback="shortest_side")
        )
        t = bot_a.choose_trump(obs_no_trump)
        t_cards = [c for c in hand if c.suit == t]
        tranks = {c.rank for c in t_cards}
        if not (len(t_cards) == 3 and Rank.THREE in tranks and Rank.ACE in tranks and Rank.TWO not in tranks):
            return None
        obs_t = _make_dummy_obs(hand, trump_suit=t)
        bot_b = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_three_card_fallback="trump_three")
        )
        bot_c = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_three_card_fallback="low_trump")
        )
        bot_d = HeuristicBotV2(
            config=HeuristicV2Config(three_ace_three_card_fallback="longest_side")
        )
        c_a = bot_a.choose_card(obs_t)
        c_b = bot_b.choose_card(obs_t)
        c_c = bot_c.choose_card(obs_t)
        c_d = bot_d.choose_card(obs_t)
        # Only trigger when Steps 1-5 didn't fire (so fallback options actually diverge)
        if c_a == c_b:
            return None
        return [
            OptionSpec("A", "Lead shortest non-trump suit (3-card, V2)", t, c_a),
            OptionSpec("B", "Lead Tre of trump (Busso Case A)", t, c_b),
            OptionSpec("C", "Lead lowest trump (silence)", t, c_c),
            OptionSpec("D", "Lead 4-card non-trump suit (Striscio)", t, c_d),
        ]

    if exp_id == "EXP-09":
        bot_a = HeuristicBotV2(config=HeuristicV2Config(avoid_bare_three_ace_lead=True))
        bot_b = HeuristicBotV2(config=HeuristicV2Config(avoid_bare_three_ace_lead=False))
        t = bot_a.choose_trump(obs_no_trump)
        obs_t = _make_dummy_obs(hand, trump_suit=t)
        c_a = bot_a.choose_card(obs_t)
        c_b = bot_b.choose_card(obs_t)
        if c_a == c_b:
            return None
        return [
            OptionSpec("A", "Avoid bare 3+A lead (V2)", t, c_a),
            OptionSpec("B", "Lead Tre of bare 3+A (Busso A)", t, c_b),
        ]

    if exp_id == "EXP-10":
        cricca_suits = [s for s in Suit if has_cricca(hand, s)]
        if len(cricca_suits) < 2:
            return None
        bot_a = HeuristicBotV2()
        t = bot_a.choose_trump(obs_no_trump)
        other_criccas = [s for s in cricca_suits if s != t]
        second_suit = min(
            other_criccas, key=lambda s: (sum(1 for c in hand if c.suit == s), s.order)
        )
        return [
            OptionSpec(
                "A",
                "Trick 2: draw Tre of trump (exhaust trumps, V2)",
                forced_trump=t,
                forced_trick1_card=Card(t, Rank.ACE),
                forced_trick2_card=Card(t, Rank.THREE),
            ),
            OptionSpec(
                "B",
                "Trick 2: cash Asso of 2nd Cricca immediately",
                forced_trump=t,
                forced_trick1_card=Card(t, Rank.ACE),
                forced_trick2_card=Card(second_suit, Rank.ACE),
            ),
        ]

    if exp_id == "EXP-11":
        bot_a = HeuristicBotV2(
            config=HeuristicV2Config(non_piece_trump_ladder_b_threshold=6)
        )
        bot_b = HeuristicBotV2(
            config=HeuristicV2Config(non_piece_trump_ladder_b_threshold=4)
        )
        t = bot_a.choose_trump(obs_no_trump)
        t_cards = [c for c in hand if c.suit == t]
        if not (4 <= len(t_cards) <= 5 and not any(c.rank.is_cricca_piece for c in t_cards)):
            return None
        obs_t = _make_dummy_obs(hand, trump_suit=t)
        c_a = bot_a.choose_card(obs_t)
        c_b = bot_b.choose_card(obs_t)
        if c_a == c_b:
            return None
        return [
            OptionSpec("A", "Ladder A (side suit, V2)", t, c_a),
            OptionSpec("B", "Ladder B (lead lowest trump)", t, c_b),
        ]

    if exp_id == "EXP-12":
        bot_a = HeuristicBotV2(config=HeuristicV2Config(side_suit_tiebreak="longer_suit"))
        bot_b = HeuristicBotV2(config=HeuristicV2Config(side_suit_tiebreak="lowest_card"))
        t = bot_a.choose_trump(obs_no_trump)
        obs_t = _make_dummy_obs(hand, trump_suit=t)
        c_a = bot_a.choose_card(obs_t)
        c_b = bot_b.choose_card(obs_t)
        if c_a == c_b:
            return None
        return [
            OptionSpec("A", "Longer side suit (V2)", t, c_a),
            OptionSpec("B", "Lowest card side suit", t, c_b),
        ]

    raise ValueError(f"Unknown experiment ID: {exp_id}")


ALL_EXPERIMENTS: tuple[str, ...] = (
    "EXP-01",
    "EXP-01-FIXED",
    "EXP-02",
    "EXP-02-FIXED",
    "EXP-03",
    "EXP-04",
    "EXP-05",
    "EXP-06",
    "EXP-07",
    "EXP-08",
    "EXP-08-FIXED",
    "EXP-09",
    "EXP-10",
    "EXP-11",
    "EXP-12",
)


# ------------------------------------------------------------------
# Parallel Workers
# ------------------------------------------------------------------
def _make_play_agent(engine_name: str, seat: int, seed: int, simulations: int) -> Agent:
    if engine_name == "ismcts":
        return ISMCTSBot(
            name=f"IS-{seat}",
            simulations=simulations,
            trump_simulations=max(12, simulations // 4),
            seed=seed,
        )
    return HeuristicBotV2(name=f"H2-{seat}")


def _eval_paired_deal_worker(
    args: tuple[
        int,
        list[list[Card]],
        list[OptionSpec],
        str,
        int,
        int,
    ],
) -> dict[str, Any]:
    deal_idx, dealt_hands, options, engine_name, simulations, seed = args
    outcomes: list[dict[str, Any]] = []

    for opt_idx, opt in enumerate(options):
        s = seed + deal_idx * 1009 + opt_idx * 37
        base_agents = [
            _make_play_agent(engine_name, p, s + p, simulations) for p in range(4)
        ]
        base_agents[0] = ForcedDecisionAgent(
            base_agents[0],
            forced_trump=opt.forced_trump,
            forced_trick1_card=opt.forced_trick1_card,
            forced_trick2_card=opt.forced_trick2_card,
        )
        res = GameEngine(base_agents).play_hand(
            trump_caller=0, allow_callout=False, dealt_hands=dealt_hands
        )
        my_pts = res.hand_points[0]
        opp_pts = res.hand_points[1]
        outcomes.append(
            {
                "code": opt.code,
                "label": opt.label,
                "trump": res.trump_suit.italian_name,
                "trick1": (
                    res.tricks[0].plays[0].card.italian_name if res.tricks else ""
                ),
                "my_pts": my_pts,
                "opp_pts": opp_pts,
                "margin": my_pts - opp_pts,
                "won": 1.0 if my_pts > opp_pts else (0.5 if my_pts == opp_pts else 0.0),
            }
        )

    return {
        "deal_idx": deal_idx,
        "p0_hand": [c.italian_name for c in dealt_hands[0]],
        "outcomes": outcomes,
    }


def _sample_matching_deals(
    exp_id: str, target_deals: int, seed: int, max_scan: int = 300_000
) -> list[tuple[list[list[Card]], list[OptionSpec]]]:
    rng = random.Random(seed)
    fixed_map = {
        "EXP-01-FIXED": FIXED_HAND_EXP01,
        "EXP-02-FIXED": FIXED_HAND_EXP02,
        "EXP-08-FIXED": FIXED_HAND_EXP08,
    }
    if exp_id in fixed_map:
        p0 = fixed_map[exp_id]
        opts = _extract_options_for_hand(exp_id, p0)
        assert opts is not None
        return [(_deal_with_fixed_p0(p0, rng), opts) for _ in range(target_deals)]

    deck = Deck(seed=seed)
    collected: list[tuple[list[list[Card]], list[OptionSpec]]] = []
    for _ in range(max_scan):
        dealt = deck.deal()
        opts = _extract_options_for_hand(exp_id, dealt[0])
        if opts is not None:
            collected.append((dealt, opts))
            if len(collected) >= target_deals:
                break
    return collected


def run_experiment(
    exp_id: str,
    *,
    num_deals: int = 500,
    engine_name: str = "heuristic",
    simulations: int = 80,
    workers: int | None = None,
    seed: int = 42,
    save_log: bool = True,
) -> ExperimentReport:
    """Run a single paired A/B experiment and optionally save a JSONL log."""
    start = time.perf_counter()
    samples = _sample_matching_deals(exp_id, target_deals=num_deals, seed=seed)
    if not samples:
        raise RuntimeError(f"Could not sample any matching hands for {exp_id}.")

    num_workers = workers or max(1, min(os.cpu_count() or 1, 8))
    tasks = [
        (idx, dealt, opts, engine_name, simulations, seed)
        for idx, (dealt, opts) in enumerate(samples)
    ]

    if num_workers == 1 or len(tasks) < 4:
        records = [_eval_paired_deal_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            records = list(pool.map(_eval_paired_deal_worker, tasks))

    n = len(records)
    num_opts = len(records[0]["outcomes"])
    option_summaries: list[ExperimentOptionSummary] = []

    margins_a = [r["outcomes"][0]["margin"] for r in records]
    for opt_idx in range(num_opts):
        code = records[0]["outcomes"][opt_idx]["code"]
        label = records[0]["outcomes"][opt_idx]["label"]
        pts = [r["outcomes"][opt_idx]["my_pts"] for r in records]
        margins = [r["outcomes"][opt_idx]["margin"] for r in records]
        wins = [r["outcomes"][opt_idx]["won"] for r in records]

        avg_pts = sum(pts) / n
        avg_margin = sum(margins) / n
        win_rate = sum(wins) / n

        if opt_idx == 0:
            delta = 0.0
            ci95 = 0.0
            verdict = "BASELINE (V2)"
        else:
            diffs = [margins[i] - margins_a[i] for i in range(n)]
            delta = sum(diffs) / n
            if n > 1:
                var = sum((d - delta) ** 2 for d in diffs) / (n - 1)
                ci95 = 1.96 * math.sqrt(var / n)
            else:
                ci95 = 0.0
            if delta - ci95 > 0.0:
                verdict = "BETTER THAN V2 (sig)"
            elif delta + ci95 < 0.0:
                verdict = "V2 WINS (sig)"
            else:
                verdict = "TIE / INCONCLUSIVE"

        option_summaries.append(
            ExperimentOptionSummary(
                code=code,
                label=label,
                avg_points=avg_pts,
                avg_margin=avg_margin,
                win_rate=win_rate,
                delta_vs_a=delta,
                ci95_vs_a=ci95,
                verdict=verdict,
            )
        )

    log_path: Path | None = None
    if save_log:
        log_dir = Path("logs") / "experiments"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{exp_id.lower()}_{engine_name}.jsonl"
        with log_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start
    return ExperimentReport(
        exp_id=exp_id,
        engine_name=engine_name,
        simulations=simulations,
        num_deals=n,
        elapsed_seconds=elapsed,
        options=tuple(option_summaries),
        log_path=log_path,
    )


# ------------------------------------------------------------------
# Global Duplicate Benchmarks (V2 vs V1, and Tuned V2 vs Pre-Stockfish V2)
# ------------------------------------------------------------------
def _bench_pair_chunk(
    args: tuple[str, str, int, int, int]
) -> tuple[list[float], list[float], list[float]]:
    team_a_kind, team_b_kind, count, base_seed, chunk_id = args
    deck = Deck(seed=base_seed + chunk_id * 100_003)

    def make_bot(kind: str, seat: int) -> Agent:
        if kind == "v2":
            return HeuristicBotV2(name=f"V2-{seat}")
        if kind == "v2_pre":
            return HeuristicBotV2(name=f"V2Pre-{seat}", config=PRE_STOCKFISH_V2_CONFIG)
        if kind == "v1":
            return HeuristicBot(name=f"V1-{seat}")
        raise ValueError(f"Unknown bench bot kind: {kind}")

    a_pts_list: list[float] = []
    b_pts_list: list[float] = []
    a_wins_list: list[float] = []

    for _ in range(count):
        dealt = deck.deal()

        # Pass 1: A on seats 0,2 vs B on seats 1,3
        agents_1 = [
            make_bot(team_a_kind, 0),
            make_bot(team_b_kind, 1),
            make_bot(team_a_kind, 2),
            make_bot(team_b_kind, 3),
        ]
        res1 = GameEngine(agents_1).play_hand(allow_callout=False, dealt_hands=dealt)

        # Pass 2: B on seats 0,2 vs A on seats 1,3 (same deal)
        agents_2 = [
            make_bot(team_b_kind, 0),
            make_bot(team_a_kind, 1),
            make_bot(team_b_kind, 2),
            make_bot(team_a_kind, 3),
        ]
        res2 = GameEngine(agents_2).play_hand(allow_callout=False, dealt_hands=dealt)

        # Pair-averaged per deal to compute valid paired 95% CI
        a_deal = 0.5 * (res1.hand_points[0] + res2.hand_points[1])
        b_deal = 0.5 * (res1.hand_points[1] + res2.hand_points[0])
        w1 = 1.0 if res1.hand_points[0] > res1.hand_points[1] else (0.5 if res1.hand_points[0] == res1.hand_points[1] else 0.0)
        w2 = 1.0 if res2.hand_points[1] > res2.hand_points[0] else (0.5 if res2.hand_points[1] == res2.hand_points[0] else 0.0)

        a_pts_list.append(a_deal)
        b_pts_list.append(b_deal)
        a_wins_list.append(0.5 * (w1 + w2))

    return a_pts_list, b_pts_list, a_wins_list


def run_global_benchmark(
    *,
    team_a_kind: str = "v2",
    team_b_kind: str = "v1",
    team_a_label: str = "HeuristicBotV2 (Stockfish-tuned)",
    team_b_label: str = "HeuristicBot v1",
    title: str = "BENCH-GLOBAL",
    num_deals: int = 5000,
    workers: int | None = None,
    seed: int = 42,
) -> str:
    """Run a duplicate-deal parallel head-to-head benchmark with 95% CI."""
    start = time.perf_counter()
    num_workers = workers or max(1, min(os.cpu_count() or 1, 8))
    per_w, rem = divmod(num_deals, num_workers)
    tasks = [
        (team_a_kind, team_b_kind, per_w + (1 if i < rem else 0), seed, i)
        for i in range(num_workers)
        if (per_w + (1 if i < rem else 0)) > 0
    ]

    if num_workers == 1 or len(tasks) == 1:
        chunks = [_bench_pair_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            chunks = list(pool.map(_bench_pair_chunk, tasks))

    a_pts = [x for c in chunks for x in c[0]]
    b_pts = [x for c in chunks for x in c[1]]
    a_wins = [x for c in chunks for x in c[2]]

    n = len(a_pts)
    avg_a = sum(a_pts) / n
    avg_b = sum(b_pts) / n
    diffs = [a_pts[i] - b_pts[i] for i in range(n)]
    edge = sum(diffs) / n
    var = sum((d - edge) ** 2 for d in diffs) / max(1, n - 1)
    ci95 = 1.96 * math.sqrt(var / n)
    win_rate = sum(a_wins) / n
    elapsed = time.perf_counter() - start

    if edge - ci95 > 0.0:
        verdict = f"{team_a_label} WINS (statistically significant)"
    elif edge + ci95 < 0.0:
        verdict = f"{team_b_label} WINS (statistically significant)"
    else:
        verdict = "STATISTICALLY TIED"

    return (
        f"\n=== {title}: {team_a_label} vs {team_b_label} "
        f"({n * 2} duplicate hands in {elapsed:.2f}s) ===\n"
        f"{team_a_label:<36}: {avg_a:.3f} pts/hand\n"
        f"{team_b_label:<36}: {avg_b:.3f} pts/hand\n"
        f"{'Net edge (A - B)':<36}: {edge:+.3f} +/- {ci95:.3f} pts/hand (95% CI)\n"
        f"{'Team A hand win rate':<36}: {win_rate * 100:.1f}%\n"
        f"{'Verdict':<36}: {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenBecca Stockfish Decision Lab.")
    parser.add_argument(
        "--exp",
        type=str,
        default="BENCH-GLOBAL",
        help="Experiment ID (EXP-01..EXP-12, *-FIXED, all, BENCH-GLOBAL, BENCH-V2-EVOLUTION).",
    )
    parser.add_argument("--deals", type=int, default=500, help="Paired deals per experiment.")
    parser.add_argument(
        "--engine",
        choices=["heuristic", "ismcts"],
        default="heuristic",
        help="Rollout engine for paired experiments.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=80,
        help="ISMCTS simulations per move when --engine ismcts.",
    )
    parser.add_argument("--workers", type=int, default=None, help="CPU worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    exp_arg = args.exp.upper()
    if exp_arg == "BENCH-GLOBAL":
        print(
            run_global_benchmark(
                team_a_kind="v2",
                team_b_kind="v1",
                team_a_label="HeuristicBotV2 (Stockfish-tuned)",
                team_b_label="HeuristicBot v1",
                title="BENCH-GLOBAL",
                num_deals=args.deals,
                workers=args.workers,
                seed=args.seed,
            )
        )
        return

    if exp_arg == "BENCH-V2-EVOLUTION":
        print(
            run_global_benchmark(
                team_a_kind="v2",
                team_b_kind="v2_pre",
                team_a_label="HeuristicBotV2 (Stockfish-tuned)",
                team_b_label="HeuristicBotV2 (Pre-Stockfish)",
                title="BENCH-V2-EVOLUTION",
                num_deals=args.deals,
                workers=args.workers,
                seed=args.seed,
            )
        )
        return

    exp_list = list(ALL_EXPERIMENTS) if exp_arg == "ALL" else [exp_arg]
    for exp_id in exp_list:
        report = run_experiment(
            exp_id,
            num_deals=args.deals,
            engine_name=args.engine,
            simulations=args.simulations,
            workers=args.workers,
            seed=args.seed,
        )
        print(report.format_table())


if __name__ == "__main__":
    main()