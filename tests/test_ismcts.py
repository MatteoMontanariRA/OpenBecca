"""Unit tests for belief sampling, SO-ISMCTS search, and Stockfish evaluator."""

import random

import pytest

from openbecca.agents import HeuristicBot, ISMCTSBot
from openbecca.agents.ismcts_bot import infer_belief_constraints, sample_consistent_hands
from openbecca.analysis.evaluator import build_custom_observation, evaluate_position
from openbecca.cards import ALL_CARDS, Card, Deck, Rank, Suit
from openbecca.engine import GameEngine
from openbecca.rules import TrickPlay
from openbecca.signals import Signal
from openbecca.state import TrickRecord


def test_belief_sampler_respects_voids_busso_and_cricca() -> None:
    # Build a valid Trick 2 scenario where:
    # - Player 1 declared Cricca in Coins (already played Asso di Denari on Trick 1)
    # - Player 2 led Trick 1 with Tre di Coppe + Busso (Case A -> must hold Asso di Coppe, not Due)
    # - Player 3 played off-suit on Trick 1 -> void in Cups
    trick1 = TrickRecord(
        trick_number=1,
        leader=2,
        signal=Signal.BUSSO,
        plays=(
            TrickPlay(2, Card(Suit.CUPS, Rank.THREE)),
            TrickPlay(3, Card(Suit.SWORDS, Rank.FOUR)),  # Void in Cups
            TrickPlay(0, Card(Suit.CUPS, Rank.TWO)),
            TrickPlay(1, Card(Suit.COINS, Rank.ACE)),  # Cut with Coins Asso -> void in Cups
        ),
        winner=1,
        points_thirds=5,
    )

    my_hand = [
        Card(Suit.SWORDS, Rank.THREE),
        Card(Suit.SWORDS, Rank.TWO),
        Card(Suit.SWORDS, Rank.ACE),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.CLUBS, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.FIVE),
        Card(Suit.COINS, Rank.FIVE),
        Card(Suit.CUPS, Rank.FOUR),
    ]

    obs = build_custom_observation(
        player_hand=my_hand,
        trump_suit=Suit.COINS,
        player_id=0,
        trump_caller=1,
        completed_tricks=[trick1],
        cricca_declared_by=1,
        current_trick_leader=1,
    )

    constraints = infer_belief_constraints(obs)
    rng = random.Random(42)

    for _ in range(25):
        hands = sample_consistent_hands(obs, constraints, rng)
        assert len(hands[0]) == 9
        assert all(len(hands[p]) == 9 for p in (1, 2, 3))
        # Player 1 must hold remaining Coins Cricca pieces (Due & Tre) and be void in Cups
        assert Card(Suit.COINS, Rank.TWO) in hands[1]
        assert Card(Suit.COINS, Rank.THREE) in hands[1]
        assert all(c.suit != Suit.CUPS for c in hands[1])
        # Player 2 must hold Asso di Coppe and NOT hold Due di Coppe
        assert Card(Suit.CUPS, Rank.ACE) in hands[2]
        assert Card(Suit.CUPS, Rank.TWO) not in hands[2]
        # Player 3 must be void in Cups
        assert all(c.suit != Suit.CUPS for c in hands[3])


def test_ismcts_full_hand_legal_play() -> None:
    agents = [
        ISMCTSBot("IS-0", simulations=40, trump_simulations=8, seed=1),
        HeuristicBot("H-1"),
        ISMCTSBot("IS-2", simulations=40, trump_simulations=8, seed=2),
        HeuristicBot("H-3"),
    ]
    engine = GameEngine(agents, seed=7)
    res = engine.play_hand(allow_callout=False)
    assert len(res.tricks) == 10
    expected_total = 14 if res.cricca_declared_by is not None else 11
    assert sum(res.hand_points) == expected_total


def test_stockfish_evaluator_ranks_moves_and_formats_table() -> None:
    hands = Deck(seed=99).deal()
    obs = build_custom_observation(
        player_hand=hands[0],
        trump_suit=Suit.COINS,
        player_id=0,
        trump_caller=0,
    )
    analysis = evaluate_position(obs, simulations=80, min_visits_per_move=4, seed=42)

    assert len(analysis.evaluations) == len(obs.legal_cards)
    assert analysis.evaluations[0].ev_loss_vs_best == 0.0
    # Verify descending order of expected_margin
    margins = [e.expected_margin for e in analysis.evaluations]
    assert margins == sorted(margins, reverse=True)

    table = analysis.format_table()
    assert "OpenBecca Position Analysis" in table
    assert "EV Pts" in table


def test_custom_observation_rejects_inconsistent_card_counts() -> None:
    hands = Deck(seed=1).deal()
    with pytest.raises(ValueError, match="exactly 4 cards must be played"):
        build_custom_observation(
            player_hand=hands[0][:9],  # 9 cards in hand implies 4 cards played
            trump_suit=Suit.COINS,
            extra_played_cards=[],  # Missing played cards!
        )