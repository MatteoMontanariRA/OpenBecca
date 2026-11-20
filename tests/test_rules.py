"""Unit tests for trick resolution, legal moves, Cricca, and Call-out rules."""

from openbecca.cards import Card, Rank, Suit
from openbecca.rules import (
    TrickPlay,
    determine_trick_winner,
    evaluate_callout,
    get_legal_cards,
    has_cricca,
    trick_points_thirds,
)
from openbecca.signals import Signal


def test_trick_winner_without_trump() -> None:
    plays = [
        TrickPlay(0, Card(Suit.CUPS, Rank.KING)),
        TrickPlay(1, Card(Suit.CUPS, Rank.THREE)),
        TrickPlay(2, Card(Suit.CUPS, Rank.ACE)),
        TrickPlay(3, Card(Suit.SWORDS, Rank.THREE)),  # Off-suit, not trump
    ]
    assert determine_trick_winner(plays, trump_suit=Suit.COINS) == 1


def test_trick_winner_with_trump_cut() -> None:
    plays = [
        TrickPlay(0, Card(Suit.CUPS, Rank.THREE)),
        TrickPlay(1, Card(Suit.COINS, Rank.FOUR)),  # Lowest trump beats non-trump Tre
        TrickPlay(2, Card(Suit.CUPS, Rank.ACE)),
        TrickPlay(3, Card(Suit.COINS, Rank.KNAVE)),  # Higher trump wins
    ]
    assert determine_trick_winner(plays, trump_suit=Suit.COINS) == 3


def test_trick_points_and_last_trick_bonus() -> None:
    plays = [
        TrickPlay(0, Card(Suit.CUPS, Rank.ACE)),  # 3 thirds
        TrickPlay(1, Card(Suit.CUPS, Rank.THREE)),  # 1 third
        TrickPlay(2, Card(Suit.CUPS, Rank.KING)),  # 1 third
        TrickPlay(3, Card(Suit.CUPS, Rank.FOUR)),  # 0 thirds
    ]
    assert trick_points_thirds(plays, is_last_trick=False) == 5
    assert trick_points_thirds(plays, is_last_trick=True) == 8


def test_follow_suit_rule() -> None:
    hand = [
        Card(Suit.CUPS, Rank.FIVE),
        Card(Suit.CUPS, Rank.ACE),
        Card(Suit.COINS, Rank.THREE),
    ]
    current_trick = [TrickPlay(0, Card(Suit.CUPS, Rank.SEVEN))]
    legal = get_legal_cards(hand, current_trick, trump_suit=Suit.COINS)
    assert set(legal) == {Card(Suit.CUPS, Rank.FIVE), Card(Suit.CUPS, Rank.ACE)}


def test_void_can_play_any_card() -> None:
    hand = [Card(Suit.SWORDS, Rank.FIVE), Card(Suit.COINS, Rank.THREE)]
    current_trick = [TrickPlay(0, Card(Suit.CUPS, Rank.SEVEN))]
    legal = get_legal_cards(hand, current_trick, trump_suit=Suit.COINS)
    assert set(legal) == set(hand)


def test_busso_response_forced_for_bots_but_free_for_humans() -> None:
    hand = [
        Card(Suit.SWORDS, Rank.TWO),
        Card(Suit.SWORDS, Rank.FIVE),
        Card(Suit.COINS, Rank.KING),
    ]
    current_trick = [
        TrickPlay(0, Card(Suit.SWORDS, Rank.THREE)),
        TrickPlay(1, Card(Suit.SWORDS, Rank.FOUR)),
    ]

    # Bot partner (player 2) must play highest card of Swords (Due)
    bot_legal = get_legal_cards(
        hand,
        current_trick,
        trump_suit=Suit.COINS,
        enforce_busso_constraints=True,
        current_trick_signal=Signal.BUSSO,
        current_trick_leader=0,
        player_id=2,
    )
    assert bot_legal == [Card(Suit.SWORDS, Rank.TWO)]

    # Human partner (enforce_busso_constraints=False) remains free to play any Sword
    human_legal = get_legal_cards(
        hand,
        current_trick,
        trump_suit=Suit.COINS,
        enforce_busso_constraints=False,
        current_trick_signal=Signal.BUSSO,
        current_trick_leader=0,
        player_id=2,
    )
    assert set(human_legal) == {Card(Suit.SWORDS, Rank.TWO), Card(Suit.SWORDS, Rank.FIVE)}


def test_busso_return_lead_forced_for_bots() -> None:
    hand = [
        Card(Suit.SWORDS, Rank.FIVE),
        Card(Suit.COINS, Rank.THREE),
    ]
    legal = get_legal_cards(
        hand,
        current_trick=[],
        trump_suit=Suit.COINS,
        enforce_busso_constraints=True,
        busso_return_suit=Suit.SWORDS,
    )
    assert legal == [Card(Suit.SWORDS, Rank.FIVE)]


def test_in_hand_cricca_lead_restriction() -> None:
    hand = [
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.FOUR),
        Card(Suit.CUPS, Rank.KING),
    ]
    assert has_cricca(hand, Suit.COINS)
    legal = get_legal_cards(
        hand,
        current_trick=[],
        trump_suit=Suit.COINS,
        is_first_trick_cricca_lead=True,
    )
    assert set(legal) == {
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.THREE),
    }


def test_callout_success_and_41_0_forfeit() -> None:
    # 39 + floor(9/3) = 42 >= 41 -> Team 0 wins
    ok = evaluate_callout(calling_team=0, match_scores=(39, 30), hand_thirds=(9, 0))
    assert ok.succeeded is True
    assert ok.winning_team == 0
    assert ok.final_match_scores == (42, 30)

    # 38 + floor(8/3) = 40 < 41 -> Team 0 loses entire match 0-41
    fail = evaluate_callout(calling_team=0, match_scores=(38, 30), hand_thirds=(8, 0))
    assert fail.succeeded is False
    assert fail.winning_team == 1
    assert fail.final_match_scores == (0, 41)