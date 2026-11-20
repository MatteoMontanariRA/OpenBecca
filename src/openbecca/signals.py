"""Strict v1 signaling logic for the trick leader (no bluffing)."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence

from openbecca.cards import Card, Rank


class Signal(Enum):
    """Allowed verbal/gesture signals for the trick leader."""

    NONE = ""
    BUSSO = "Busso"
    VOLO = "Volo"
    STRISCIO = "Striscio"

    def __str__(self) -> str:
        return self.value


def determine_lead_signal(
    played_card: Card,
    hand_before_play: Sequence[Card],
    initial_suit_count: int,
    played_cards_history: Iterable[Card],
) -> Signal:
    """Compute the strict v1 signal announced when leading a trick.

    Priority order: Busso > Volo > Striscio > None.

    Rules implemented:
    - Busso (Case A): Leader holds both Tre and Asso of the led suit (without holding
      Due, and Due has not been played yet) and leads Tre.
    - Busso (Case B): Leader holds Due (not Tre), holds at least 3 cards of that suit
      before playing, Tre has not been played yet, and leads a card lower than Asso.
    - Volo: Leader has 0 cards of the led suit remaining after playing this card.
    - Striscio: Leader started the hand with > 3 cards (>= 4) of the led suit and
      still holds at least 1 card of that suit after playing.
    - None: Silence when none of the above applies.
    """
    if played_card not in hand_before_play:
        raise ValueError(f"Played card {played_card} is not in leader's hand.")

    suit = played_card.suit
    suit_cards_before = [c for c in hand_before_play if c.suit == suit]
    suit_ranks_before = {c.rank for c in suit_cards_before}
    history_suit_ranks = {c.rank for c in played_cards_history if c.suit == suit}

    # 1. Check Busso (highest priority)
    is_busso_case_a = (
        played_card.rank == Rank.THREE
        and Rank.ACE in suit_ranks_before
        and Rank.TWO not in suit_ranks_before
        and Rank.TWO not in history_suit_ranks
    )

    is_busso_case_b = (
        Rank.TWO in suit_ranks_before
        and Rank.THREE not in suit_ranks_before
        and Rank.THREE not in history_suit_ranks
        and len(suit_cards_before) >= 3
        and played_card.rank.power < Rank.ACE.power
    )

    if is_busso_case_a or is_busso_case_b:
        return Signal.BUSSO

    # 2. Check Volo (no cards of the led suit remain in hand)
    remaining_of_suit = len(suit_cards_before) - 1
    if remaining_of_suit == 0:
        return Signal.VOLO

    # 3. Check Striscio (started the hand with > 3 cards of this suit)
    if initial_suit_count > 3:
        return Signal.STRISCIO

    # 4. Silence
    return Signal.NONE