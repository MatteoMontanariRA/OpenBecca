"""Core card, suit, rank, and deck models for Beccacino (40-card Italian deck)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """The four traditional Italian/Spanish suits."""

    COINS = ("Denari", 0)
    CUPS = ("Coppe", 1)
    SWORDS = ("Spade", 2)
    CLUBS = ("Bastoni", 3)

    def __init__(self, italian_name: str, order: int) -> None:
        self.italian_name = italian_name
        self.order = order

    def __str__(self) -> str:
        return self.italian_name


class Rank(Enum):
    """Card ranks ordered from weakest (4) to strongest (3) in trick-taking power.

    Attributes:
        italian_name: Traditional Italian written name (e.g., 'Tre', 'Fante').
        power: Trick-taking hierarchy from 1 (Quattro) to 10 (Tre).
        points_thirds: Card value expressed in exact thirds of a point.
    """

    FOUR = ("Quattro", 1, 0)
    FIVE = ("Cinque", 2, 0)
    SIX = ("Sei", 3, 0)
    SEVEN = ("Sette", 4, 0)
    KNAVE = ("Fante", 5, 1)
    KNIGHT = ("Cavallo", 6, 1)
    KING = ("Re", 7, 1)
    ACE = ("Asso", 8, 3)
    TWO = ("Due", 9, 1)
    THREE = ("Tre", 10, 1)

    def __init__(self, italian_name: str, power: int, points_thirds: int) -> None:
        self.italian_name = italian_name
        self.power = power
        self.points_thirds = points_thirds

    @property
    def is_figure(self) -> bool:
        """Return True if the rank is a face card (Fante, Cavallo, Re)."""
        return self in (Rank.KNAVE, Rank.KNIGHT, Rank.KING)

    @property
    def is_cricca_piece(self) -> bool:
        """Return True if the rank belongs to a Cricca (Asso, Due, Tre)."""
        return self in (Rank.ACE, Rank.TWO, Rank.THREE)

    def __str__(self) -> str:
        return self.italian_name


@dataclass(frozen=True, order=False)
class Card:
    """Immutable representation of a single Beccacino playing card."""

    suit: Suit
    rank: Rank

    @property
    def power(self) -> int:
        """Return trick-taking power within the card's suit."""
        return self.rank.power

    @property
    def points_thirds(self) -> int:
        """Return point value in exact thirds of a point."""
        return self.rank.points_thirds

    @property
    def italian_name(self) -> str:
        """Return the full Italian card name in letters (e.g., 'Tre di Denari')."""
        return f"{self.rank.italian_name} di {self.suit.italian_name}"

    def sort_key(self) -> tuple[int, int]:
        """Sort key grouping cards by suit and descending trick-taking power."""
        return (self.suit.order, -self.rank.power)

    def __str__(self) -> str:
        return self.italian_name


# Special card that determines the first trump caller on Hand 1.
FOUR_OF_COINS = Card(Suit.COINS, Rank.FOUR)

# Canonical 40-card deck tuple.
ALL_CARDS: tuple[Card, ...] = tuple(
    Card(suit, rank) for suit in Suit for rank in Rank
)


class Deck:
    """Standard 40-card deck with deterministic shuffling and dealing."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._cards: list[Card] = list(ALL_CARDS)

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        self._cards = list(ALL_CARDS)
        self._rng.shuffle(self._cards)

    def deal(self) -> list[list[Card]]:
        """Shuffle and deal 4 hands of 10 sorted cards each."""
        self.shuffle()
        hands: list[list[Card]] = []
        for player_idx in range(4):
            start = player_idx * 10
            hand = sorted(self._cards[start : start + 10], key=lambda c: c.sort_key())
            hands.append(hand)
        return hands