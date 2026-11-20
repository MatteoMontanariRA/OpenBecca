"""Game state, imperfect-information observations, and JSON-serializable records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openbecca.cards import Card, Suit
from openbecca.rules import CalloutOutcome, TrickPlay, team_of, thirds_to_points
from openbecca.signals import Signal


@dataclass(frozen=True)
class TrickRecord:
    """Complete public record of a finished trick."""

    trick_number: int  # 1..10
    leader: int
    signal: Signal
    plays: tuple[TrickPlay, ...]
    winner: int
    points_thirds: int

    @property
    def led_suit(self) -> Suit:
        return self.plays[0].card.suit

    def to_dict(self) -> dict[str, Any]:
        return {
            "trick_number": self.trick_number,
            "leader": self.leader,
            "signal": self.signal.value,
            "plays": [
                {"player": p.player, "card": p.card.italian_name} for p in self.plays
            ],
            "winner": self.winner,
            "points_thirds": self.points_thirds,
        }


@dataclass(frozen=True)
class Observation:
    """Imperfect-information view exposed to a single player.

    Guarantees zero data leakage: opponents' and partner's hidden cards are
    never included in this object.
    """

    player_id: int
    team_id: int
    hand: tuple[Card, ...]
    initial_suit_counts: dict[Suit, int]
    hand_number: int
    trump_caller: int
    trump_suit: Suit | None
    cricca_declared_by: int | None
    completed_tricks: tuple[TrickRecord, ...]
    current_trick: tuple[TrickPlay, ...]
    current_trick_leader: int
    current_trick_signal: Signal
    played_cards: frozenset[Card]
    match_scores: tuple[int, int]
    hand_thirds: tuple[int, int]
    target_score: int
    pending_busso_return_suit: Suit | None = None
    legal_cards: tuple[Card, ...] = ()

    @property
    def current_trick_number(self) -> int:
        return len(self.completed_tricks) + 1


@dataclass
class HandResult:
    """Summary of a completed hand (either all 10 tricks or stopped by call-out)."""

    hand_number: int
    trump_caller: int
    trump_suit: Suit
    cricca_declared_by: int | None
    tricks: list[TrickRecord]
    hand_thirds: tuple[int, int]
    hand_points: tuple[int, int]
    callout: CalloutOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_number": self.hand_number,
            "trump_caller": self.trump_caller,
            "trump_suit": self.trump_suit.italian_name,
            "cricca_declared_by": self.cricca_declared_by,
            "hand_thirds": list(self.hand_thirds),
            "hand_points": list(self.hand_points),
            "callout": (
                {
                    "calling_team": self.callout.calling_team,
                    "succeeded": self.callout.succeeded,
                    "winning_team": self.callout.winning_team,
                    "verified_total": self.callout.verified_total,
                    "final_match_scores": list(self.callout.final_match_scores),
                }
                if self.callout is not None
                else None
            ),
            "tricks": [t.to_dict() for t in self.tricks],
        }


@dataclass
class MatchResult:
    """Summary and replay log of a full match up to target_score."""

    target_score: int
    winner_team: int
    final_scores: tuple[int, int]
    hands: list[HandResult] = field(default_factory=list)
    ended_by_callout: bool = False
    sudden_death_played: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_score": self.target_score,
            "winner_team": self.winner_team,
            "final_scores": list(self.final_scores),
            "ended_by_callout": self.ended_by_callout,
            "sudden_death_played": self.sudden_death_played,
            "hands_played": len(self.hands),
            "hands": [h.to_dict() for h in self.hands],
        }

    def save_json(self, path: str | Path) -> Path:
        """Export the complete match replay to a UTF-8 JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path


@dataclass
class HandState:
    """Internal full-information state maintained exclusively by the GameEngine."""

    hand_number: int
    trump_caller: int
    hands: list[list[Card]]
    initial_suit_counts: list[dict[Suit, int]]
    match_scores: tuple[int, int]
    target_score: int
    trump_suit: Suit | None = None
    cricca_declared_by: int | None = None
    completed_tricks: list[TrickRecord] = field(default_factory=list)
    current_trick: list[TrickPlay] = field(default_factory=list)
    current_trick_leader: int = 0
    current_trick_signal: Signal = Signal.NONE
    played_cards: set[Card] = field(default_factory=set)
    hand_thirds: list[int] = field(default_factory=lambda: [0, 0])
    pending_busso_return_suit: Suit | None = None

    def __post_init__(self) -> None:
        self.current_trick_leader = self.trump_caller

    def build_observation(
        self,
        player_id: int,
        legal_cards: tuple[Card, ...] = (),
    ) -> Observation:
        """Create a safe imperfect-information Observation for a single player."""
        busso_suit = (
            self.pending_busso_return_suit
            if not self.current_trick and player_id == self.current_trick_leader
            else None
        )
        return Observation(
            player_id=player_id,
            team_id=team_of(player_id),
            hand=tuple(self.hands[player_id]),
            initial_suit_counts=dict(self.initial_suit_counts[player_id]),
            hand_number=self.hand_number,
            trump_caller=self.trump_caller,
            trump_suit=self.trump_suit,
            cricca_declared_by=self.cricca_declared_by,
            completed_tricks=tuple(self.completed_tricks),
            current_trick=tuple(self.current_trick),
            current_trick_leader=self.current_trick_leader,
            current_trick_signal=self.current_trick_signal,
            played_cards=frozenset(self.played_cards),
            match_scores=self.match_scores,
            hand_thirds=(self.hand_thirds[0], self.hand_thirds[1]),
            target_score=self.target_score,
            pending_busso_return_suit=busso_suit,
            legal_cards=legal_cards,
        )

    def current_hand_points(self) -> tuple[int, int]:
        return (
            thirds_to_points(self.hand_thirds[0]),
            thirds_to_points(self.hand_thirds[1]),
        )