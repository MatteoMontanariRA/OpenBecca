"""Pure, stateless rules functions for Beccacino (Ravenna variant)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from openbecca.cards import FOUR_OF_COINS, Card, Rank, Suit
from openbecca.signals import Signal

LAST_TRICK_BONUS_THIRDS = 3  # +1 point for winning Trick 10
CRICCA_BONUS_THIRDS = 9  # +3 points for declaring trump Cricca
DEFAULT_TARGET_SCORE = 41


@dataclass(frozen=True)
class TrickPlay:
    """A single card played by a player inside a trick."""

    player: int
    card: Card


def team_of(player_id: int) -> int:
    """Return team index (0 for players 0 & 2, 1 for players 1 & 3)."""
    if player_id not in (0, 1, 2, 3):
        raise ValueError(f"Invalid player_id: {player_id}. Must be in 0..3.")
    return player_id % 2


def partner_of(player_id: int) -> int:
    """Return the teammate's player index sitting across the table."""
    if player_id not in (0, 1, 2, 3):
        raise ValueError(f"Invalid player_id: {player_id}. Must be in 0..3.")
    return (player_id + 2) % 4


def next_player(player_id: int) -> int:
    """Return the next player in counter-clockwise order."""
    return (player_id + 1) % 4


def find_four_of_coins_holder(hands: Sequence[Sequence[Card]]) -> int:
    """Return the player index (0..3) holding Quattro di Denari."""
    for player_id, hand in enumerate(hands):
        if FOUR_OF_COINS in hand:
            return player_id
    raise ValueError("Quattro di Denari was not found in any player's hand.")


def has_cricca(hand: Iterable[Card], suit: Suit) -> bool:
    """Return True if the hand contains Asso, Due, and Tre of the given suit."""
    ranks = {card.rank for card in hand if card.suit == suit}
    return {Rank.ACE, Rank.TWO, Rank.THREE}.issubset(ranks)


def thirds_to_points(thirds: int) -> int:
    """Convert exact thirds of a point to whole integer points (floor division)."""
    return thirds // 3


def determine_trick_winner(plays: Sequence[TrickPlay], trump_suit: Suit) -> int:
    """Return the player_id who wins a completed trick.

    Rules:
    - If at least one trump card was played, the highest-power trump wins.
    - Otherwise, the highest-power card matching the led suit wins.
    """
    if not plays:
        raise ValueError("Cannot determine winner of an empty trick.")

    led_suit = plays[0].card.suit
    trump_plays = [p for p in plays if p.card.suit == trump_suit]

    if trump_plays:
        winning_play = max(trump_plays, key=lambda p: p.card.power)
    else:
        led_plays = [p for p in plays if p.card.suit == led_suit]
        winning_play = max(led_plays, key=lambda p: p.card.power)

    return winning_play.player


def trick_points_thirds(plays: Sequence[TrickPlay], is_last_trick: bool = False) -> int:
    """Return total points captured in a trick, expressed in exact thirds."""
    card_thirds = sum(p.card.points_thirds for p in plays)
    bonus = LAST_TRICK_BONUS_THIRDS if is_last_trick else 0
    return card_thirds + bonus


def get_legal_cards(
    hand: Sequence[Card],
    current_trick: Sequence[TrickPlay],
    trump_suit: Suit,
    *,
    is_first_trick_cricca_lead: bool = False,
    enforce_busso_constraints: bool = False,
    current_trick_signal: Signal = Signal.NONE,
    current_trick_leader: int | None = None,
    player_id: int | None = None,
    busso_return_suit: Suit | None = None,
) -> list[Card]:
    """Return the list of legal cards a player can play in the current situation.

    Standard rules:
    - If leading Trick 1 after declaring an in-hand Cricca, must lead a trump
      Cricca piece (Asso, Due, or Tre of trump).
    - If following a trick, must follow the led suit when able.

    Optional v1 AI Busso constraints (when enforce_busso_constraints=True):
    - When partner led the current trick with Busso and player holds the led suit,
      player must play their highest-power card of that suit.
    - When player won partner's Busso trick and is leading the next trick, player
      must lead back the same suit if still held.
    """
    if not hand:
        return []

    # Case 1: Leading a trick
    if not current_trick:
        if is_first_trick_cricca_lead:
            cricca_cards = [
                c for c in hand if c.suit == trump_suit and c.rank.is_cricca_piece
            ]
            if not cricca_cards:
                raise ValueError("Player declared in-hand Cricca but holds no Cricca card.")
            return cricca_cards

        if enforce_busso_constraints and busso_return_suit is not None:
            return_suit_cards = [c for c in hand if c.suit == busso_return_suit]
            if return_suit_cards:
                return return_suit_cards

        return list(hand)

    # Case 2: Following a trick
    led_suit = current_trick[0].card.suit
    matching_suit = [c for c in hand if c.suit == led_suit]
    if not matching_suit:
        return list(hand)

    if (
        enforce_busso_constraints
        and current_trick_signal == Signal.BUSSO
        and current_trick_leader is not None
        and player_id is not None
        and partner_of(player_id) == current_trick_leader
    ):
        highest_card = max(matching_suit, key=lambda c: c.power)
        return [highest_card]

    return matching_suit


@dataclass(frozen=True)
class CalloutOutcome:
    """Result of a team calling out ('Chiamarsi fuori')."""

    calling_team: int
    succeeded: bool
    winning_team: int
    final_match_scores: tuple[int, int]
    verified_total: int


def evaluate_callout(
    calling_team: int,
    match_scores: tuple[int, int],
    hand_thirds: tuple[int, int],
    target_score: int = DEFAULT_TARGET_SCORE,
) -> CalloutOutcome:
    """Resolve a call-out attempt.

    - If match_scores[team] + floor(hand_thirds[team] / 3) >= target_score:
      calling team wins immediately.
    - Otherwise: calling team loses the entire match immediately (target_score - 0).
    """
    opponent_team = 1 - calling_team
    verified_total = match_scores[calling_team] + thirds_to_points(hand_thirds[calling_team])

    if verified_total >= target_score:
        updated = [match_scores[0], match_scores[1]]
        updated[calling_team] = verified_total
        return CalloutOutcome(
            calling_team=calling_team,
            succeeded=True,
            winning_team=calling_team,
            final_match_scores=(updated[0], updated[1]),
            verified_total=verified_total,
        )

    forfeit_scores = [0, 0]
    forfeit_scores[opponent_team] = target_score
    return CalloutOutcome(
        calling_team=calling_team,
        succeeded=False,
        winning_team=opponent_team,
        final_match_scores=(forfeit_scores[0], forfeit_scores[1]),
        verified_total=verified_total,
    )