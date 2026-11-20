"""Unit tests for strict v1 lead signaling rules."""

import pytest

from openbecca.cards import Card, Rank, Suit
from openbecca.signals import Signal, determine_lead_signal


def test_busso_case_a_three_and_ace() -> None:
    hand = [
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.ACE),
        Card(Suit.COINS, Rank.FIVE),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.CUPS, Rank.THREE),
        hand_before_play=hand,
        initial_suit_count=2,
        played_cards_history=[],
    )
    assert signal == Signal.BUSSO


def test_busso_case_a_suppressed_if_two_already_played() -> None:
    hand = [
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.ACE),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.CUPS, Rank.THREE),
        hand_before_play=hand,
        initial_suit_count=2,
        played_cards_history=[Card(Suit.CUPS, Rank.TWO)],
    )
    assert signal == Signal.NONE


def test_busso_case_b_two_with_at_least_three_cards() -> None:
    hand = [
        Card(Suit.SWORDS, Rank.TWO),
        Card(Suit.SWORDS, Rank.KING),
        Card(Suit.SWORDS, Rank.FOUR),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.SWORDS, Rank.FOUR),
        hand_before_play=hand,
        initial_suit_count=3,
        played_cards_history=[],
    )
    assert signal == Signal.BUSSO


def test_busso_case_b_blocked_with_only_two_cards() -> None:
    # Strategy H03/extra #3: do not call Busso with Due + only 1 other card.
    hand = [
        Card(Suit.SWORDS, Rank.TWO),
        Card(Suit.SWORDS, Rank.FOUR),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.SWORDS, Rank.FOUR),
        hand_before_play=hand,
        initial_suit_count=2,
        played_cards_history=[],
    )
    assert signal == Signal.NONE


def test_busso_case_b_blocked_if_three_already_played() -> None:
    hand = [
        Card(Suit.SWORDS, Rank.TWO),
        Card(Suit.SWORDS, Rank.SEVEN),
        Card(Suit.SWORDS, Rank.FOUR),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.SWORDS, Rank.FOUR),
        hand_before_play=hand,
        initial_suit_count=3,
        played_cards_history=[Card(Suit.SWORDS, Rank.THREE)],
    )
    assert signal == Signal.NONE


def test_priority_busso_over_striscio() -> None:
    hand = [
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CLUBS, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.FOUR),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.CLUBS, Rank.THREE),
        hand_before_play=hand,
        initial_suit_count=4,
        played_cards_history=[],
    )
    assert signal == Signal.BUSSO


def test_priority_volo_over_striscio() -> None:
    # Started with 4 cards of Coins, now playing the last remaining one -> Volo wins.
    hand = [
        Card(Suit.COINS, Rank.FIVE),
        Card(Suit.CUPS, Rank.ACE),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.COINS, Rank.FIVE),
        hand_before_play=hand,
        initial_suit_count=4,
        played_cards_history=[],
    )
    assert signal == Signal.VOLO


def test_striscio_when_started_with_four_and_cards_remain() -> None:
    hand = [
        Card(Suit.COINS, Rank.SEVEN),
        Card(Suit.COINS, Rank.FIVE),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.COINS, Rank.FIVE),
        hand_before_play=hand,
        initial_suit_count=4,
        played_cards_history=[],
    )
    assert signal == Signal.STRISCIO


def test_silence_when_no_condition_met() -> None:
    hand = [
        Card(Suit.COINS, Rank.KING),
        Card(Suit.COINS, Rank.FIVE),
    ]
    signal = determine_lead_signal(
        played_card=Card(Suit.COINS, Rank.FIVE),
        hand_before_play=hand,
        initial_suit_count=3,
        played_cards_history=[],
    )
    assert signal == Signal.NONE


def test_raises_if_card_not_in_hand() -> None:
    with pytest.raises(ValueError):
        determine_lead_signal(
            played_card=Card(Suit.COINS, Rank.ACE),
            hand_before_play=[Card(Suit.COINS, Rank.FOUR)],
            initial_suit_count=1,
            played_cards_history=[],
        )