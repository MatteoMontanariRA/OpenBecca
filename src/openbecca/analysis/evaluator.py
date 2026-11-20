"""Stockfish-style position evaluator and manual scenario builder."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

from openbecca.agents.ismcts_bot import ISMCTSBot, MoveStats
from openbecca.cards import ALL_CARDS, Card, Suit
from openbecca.rules import TrickPlay, get_legal_cards, team_of
from openbecca.signals import Signal
from openbecca.state import Observation, TrickRecord


@dataclass(frozen=True)
class MoveEvaluation:
    """Detailed Stockfish-style evaluation for a single legal candidate card."""

    card: Card
    expected_points: float
    expected_opp_points: float
    expected_margin: float
    ev_loss_vs_best: float
    win_probability: float
    visits: int


@dataclass(frozen=True)
class PositionAnalysis:
    """Ranked evaluation of all legal moves in a given Beccacino position."""

    player_id: int
    trump_suit: Suit
    trick_number: int
    total_simulations: int
    elapsed_seconds: float
    evaluations: tuple[MoveEvaluation, ...]

    @property
    def best_move(self) -> MoveEvaluation:
        return self.evaluations[0]

    def format_table(self) -> str:
        """Return a clean Stockfish-style text table of ranked legal moves."""
        lines = [
            f"OpenBecca Position Analysis | Player {self.player_id} | "
            f"Trump: {self.trump_suit.italian_name} | Trick {self.trick_number}/10 | "
            f"{self.total_simulations} sims in {self.elapsed_seconds:.2f}s",
            "-" * 82,
            f"{'#':>2}  {'Card':22s} {'EV Pts':>7} {'Margin':>8} {'Loss':>7} {'Win%':>7} {'Sims':>6}",
            "-" * 82,
        ]
        for idx, ev in enumerate(self.evaluations, start=1):
            lines.append(
                f"{idx:>2}  {ev.card.italian_name:22s} "
                f"{ev.expected_points:7.2f} "
                f"{ev.expected_margin:+8.2f} "
                f"{ev.ev_loss_vs_best:+7.2f} "
                f"{ev.win_probability * 100:6.1f}% "
                f"{ev.visits:6d}"
            )
        lines.append("-" * 82)
        return "\n".join(lines)


def evaluate_position(
    obs: Observation,
    *,
    simulations: int = 400,
    min_visits_per_move: int = 12,
    seed: int | None = None,
) -> PositionAnalysis:
    """Evaluate every legal card in obs and rank them by expected margin."""
    if obs.trump_suit is None:
        raise ValueError("Cannot evaluate card moves before trump_suit is chosen.")
    if not obs.legal_cards:
        raise ValueError("Observation has no legal_cards to evaluate.")

    start = time.perf_counter()
    bot = ISMCTSBot(simulations=simulations, seed=seed)

    # Cap warmup so small simulation budgets still give UCB1 room to focus.
    capped_min_visits = min(
        min_visits_per_move, max(1, simulations // (len(obs.legal_cards) * 2))
    )
    raw_stats: list[MoveStats] = bot.analyze_moves(
        obs,
        simulations=simulations,
        min_visits_per_move=capped_min_visits,
    )
    elapsed = time.perf_counter() - start

    best_margin = raw_stats[0].expected_margin
    evaluations = tuple(
        MoveEvaluation(
            card=s.card,
            expected_points=s.expected_points,
            expected_opp_points=s.expected_opp_points,
            expected_margin=s.expected_margin,
            ev_loss_vs_best=s.expected_margin - best_margin,
            win_probability=s.win_probability,
            visits=s.visits,
        )
        for s in raw_stats
    )

    cards_on_table = len(obs.current_trick)
    cards_in_hand = len(obs.hand)
    me_played = any(p.player == obs.player_id for p in obs.current_trick)
    tricks_left = cards_in_hand + (1 if me_played else 0)
    trick_num = 11 - tricks_left

    return PositionAnalysis(
        player_id=obs.player_id,
        trump_suit=obs.trump_suit,
        trick_number=trick_num,
        total_simulations=simulations,
        elapsed_seconds=elapsed,
        evaluations=evaluations,
    )


def build_custom_observation(
    *,
    player_hand: Sequence[Card],
    trump_suit: Suit,
    player_id: int = 0,
    trump_caller: int = 0,
    current_trick: Sequence[TrickPlay] = (),
    current_trick_leader: int | None = None,
    current_trick_signal: Signal = Signal.NONE,
    completed_tricks: Sequence[TrickRecord] = (),
    extra_played_cards: Iterable[Card] = (),
    cricca_declared_by: int | None = None,
    match_scores: tuple[int, int] = (0, 0),
    hand_thirds: tuple[int, int] = (0, 0),
    target_score: int = 41,
    initial_suit_counts: dict[Suit, int] | None = None,
    pending_busso_return_suit: Suit | None = None,
    enforce_busso_constraints: bool = False,
) -> Observation:
    """Build a validated Observation from a manually configured table scenario.

    Supports both full trick histories and lightweight custom setups where the
    user simply lists cards already played in `extra_played_cards`.
    """
    hand_list = list(player_hand)
    if not hand_list:
        raise ValueError("player_hand cannot be empty.")
    if len(set(hand_list)) != len(hand_list):
        raise ValueError("player_hand contains duplicate cards.")

    played_set: set[Card] = set(extra_played_cards)
    for trick in completed_tricks:
        for play in trick.plays:
            played_set.add(play.card)
    for play in current_trick:
        played_set.add(play.card)

    overlap = played_set & set(hand_list)
    if overlap:
        raise ValueError(
            f"Cards cannot be both in hand and already played: {[str(c) for c in overlap]}"
        )

    leader = (
        current_trick[0].player
        if current_trick
        else (current_trick_leader if current_trick_leader is not None else player_id)
    )

    # Validate total card count consistency (completed tricks + table + 4 hands = 40)
    tricks_completed_from_hand = 10 - len(hand_list)
    expected_played_count = tricks_completed_from_hand * 4 + len(current_trick)
    if len(played_set) != expected_played_count:
        raise ValueError(
            f"With {len(hand_list)} cards in hand and {len(current_trick)} cards on the table, "
            f"exactly {expected_played_count} cards must be played (got {len(played_set)})."
        )

    # Compute initial suit counts if not supplied manually
    if initial_suit_counts is None:
        counts = {s: sum(1 for c in hand_list if c.suit == s) for s in Suit}
        for trick in completed_tricks:
            for play in trick.plays:
                if play.player == player_id:
                    counts[play.card.suit] += 1
        init_counts = counts
    else:
        init_counts = dict(initial_suit_counts)

    is_first_cricca_lead = (
        not current_trick
        and len(hand_list) == 10
        and cricca_declared_by == player_id
        and player_id == trump_caller
    )
    legal = get_legal_cards(
        hand=hand_list,
        current_trick=current_trick,
        trump_suit=trump_suit,
        is_first_trick_cricca_lead=is_first_cricca_lead,
        enforce_busso_constraints=enforce_busso_constraints,
        current_trick_signal=current_trick_signal,
        current_trick_leader=leader,
        player_id=player_id,
        busso_return_suit=pending_busso_return_suit if not current_trick else None,
    )

    return Observation(
        player_id=player_id,
        team_id=team_of(player_id),
        hand=tuple(hand_list),
        initial_suit_counts=init_counts,
        hand_number=1,
        trump_caller=trump_caller,
        trump_suit=trump_suit,
        cricca_declared_by=cricca_declared_by,
        completed_tricks=tuple(completed_tricks),
        current_trick=tuple(current_trick),
        current_trick_leader=leader,
        current_trick_signal=current_trick_signal,
        played_cards=frozenset(played_set),
        match_scores=match_scores,
        hand_thirds=hand_thirds,
        target_score=target_score,
        pending_busso_return_suit=pending_busso_return_suit,
        legal_cards=tuple(legal),
    )