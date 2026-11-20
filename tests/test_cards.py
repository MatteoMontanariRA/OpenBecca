"""Unit tests for cards, deck, Italian names, and exact integer-thirds scoring."""

from openbecca.cards import ALL_CARDS, FOUR_OF_COINS, Card, Deck, Rank, Suit


def test_deck_has_40_unique_cards() -> None:
    assert len(ALL_CARDS) == 40
    assert len(set(ALL_CARDS)) == 40


def test_total_card_points_in_thirds() -> None:
    # 4 Aces (12 thirds) + 4 Threes + 4 Twos + 12 Figures (20 thirds) = 32 thirds
    total_thirds = sum(card.points_thirds for card in ALL_CARDS)
    assert total_thirds == 32


def test_rank_power_hierarchy() -> None:
    expected_order = [
        Rank.FOUR,
        Rank.FIVE,
        Rank.SIX,
        Rank.SEVEN,
        Rank.KNAVE,
        Rank.KNIGHT,
        Rank.KING,
        Rank.ACE,
        Rank.TWO,
        Rank.THREE,
    ]
    powers = [r.power for r in expected_order]
    assert powers == sorted(powers)
    assert len(set(powers)) == 10


def test_italian_written_names() -> None:
    assert FOUR_OF_COINS.italian_name == "Quattro di Denari"
    assert Card(Suit.CUPS, Rank.THREE).italian_name == "Tre di Coppe"
    assert Card(Suit.SWORDS, Rank.KNAVE).italian_name == "Fante di Spade"
    assert Card(Suit.CLUBS, Rank.ACE).italian_name == "Asso di Bastoni"


def test_deterministic_deal() -> None:
    deck_a = Deck(seed=42)
    deck_b = Deck(seed=42)
    hands_a = deck_a.deal()
    hands_b = deck_b.deal()

    assert hands_a == hands_b
    assert len(hands_a) == 4
    assert all(len(hand) == 10 for hand in hands_a)
    dealt_cards = [c for hand in hands_a for c in hand]
    assert len(set(dealt_cards)) == 40