"""Core hand and match orchestration engine for OpenBecca."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from openbecca.agents.base import Agent
from openbecca.cards import Card, Deck, Suit
from openbecca.rules import (
    CRICCA_BONUS_THIRDS,
    DEFAULT_TARGET_SCORE,
    CalloutOutcome,
    TrickPlay,
    determine_trick_winner,
    evaluate_callout,
    find_four_of_coins_holder,
    get_legal_cards,
    has_cricca,
    next_player,
    partner_of,
    team_of,
    trick_points_thirds,
)
from openbecca.signals import Signal, determine_lead_signal
from openbecca.state import HandResult, HandState, MatchResult, TrickRecord


def _count_suits(hand: Sequence[Card]) -> dict[Suit, int]:
    counts = Counter(card.suit for card in hand)
    return {suit: counts.get(suit, 0) for suit in Suit}


class GameEngine:
    """Runs Beccacino hands and full matches with strict rule enforcement."""

    def __init__(
        self,
        agents: Sequence[Agent],
        target_score: int = DEFAULT_TARGET_SCORE,
        seed: int | None = None,
    ) -> None:
        if len(agents) != 4:
            raise ValueError(f"Beccacino requires exactly 4 agents, got {len(agents)}.")
        self.agents = list(agents)
        self.target_score = target_score
        self._deck = Deck(seed=seed)

    def play_hand(
        self,
        *,
        hand_number: int = 1,
        trump_caller: int | None = None,
        match_scores: tuple[int, int] = (0, 0),
        allow_callout: bool = True,
        dealt_hands: list[list[Card]] | None = None,
    ) -> HandResult:
        """Play a single 10-trick hand (or stop early on a valid/invalid call-out)."""
        hands = [list(h) for h in dealt_hands] if dealt_hands is not None else self._deck.deal()

        if trump_caller is None:
            trump_caller = find_four_of_coins_holder(hands)

        initial_counts = [_count_suits(h) for h in hands]
        state = HandState(
            hand_number=hand_number,
            trump_caller=trump_caller,
            hands=hands,
            initial_suit_counts=initial_counts,
            match_scores=match_scores,
            target_score=self.target_score,
        )

        # 1. Trump selection phase
        caller_obs = state.build_observation(trump_caller)
        chosen_trump = self.agents[trump_caller].choose_trump(caller_obs)
        if not isinstance(chosen_trump, Suit):
            raise ValueError(f"Agent {trump_caller} returned invalid trump suit: {chosen_trump}")
        state.trump_suit = chosen_trump

        # 2. Cricca check phase (in-hand for caller, out-of-hand for all other players)
        self._resolve_cricca_phase(state)

        # 3. Early call-out window right after trump & Cricca declaration
        if allow_callout:
            callout = self._check_callout_window(state, start_player=trump_caller)
            if callout is not None:
                return self._finalize_hand(state, callout=callout)

        # 4. Play 10 tricks
        for trick_idx in range(1, 11):
            leader = state.current_trick_leader
            state.current_trick = []
            state.current_trick_signal = Signal.NONE

            for offset in range(4):
                player_id = (leader + offset) % 4
                agent = self.agents[player_id]
                is_first_trick_cricca_lead = (
                    trick_idx == 1
                    and offset == 0
                    and state.cricca_declared_by == player_id
                    and player_id == state.trump_caller
                )

                legal = get_legal_cards(
                    hand=state.hands[player_id],
                    current_trick=state.current_trick,
                    trump_suit=state.trump_suit,
                    is_first_trick_cricca_lead=is_first_trick_cricca_lead,
                    enforce_busso_constraints=agent.enforce_busso_constraints,
                    current_trick_signal=state.current_trick_signal,
                    current_trick_leader=leader,
                    player_id=player_id,
                    busso_return_suit=(
                        state.pending_busso_return_suit if offset == 0 else None
                    ),
                )

                obs = state.build_observation(player_id, legal_cards=tuple(legal))
                chosen_card = agent.choose_card(obs)

                if chosen_card not in legal:
                    raise ValueError(
                        f"Illegal move by player {player_id} ({agent.name}): "
                        f"{chosen_card} not in {[str(c) for c in legal]}"
                    )

                # Compute automatic strict v1 signal for the trick leader
                if offset == 0:
                    state.current_trick_signal = determine_lead_signal(
                        played_card=chosen_card,
                        hand_before_play=state.hands[player_id],
                        initial_suit_count=state.initial_suit_counts[player_id][chosen_card.suit],
                        played_cards_history=state.played_cards,
                    )

                # Commit play
                state.hands[player_id].remove(chosen_card)
                state.current_trick.append(TrickPlay(player=player_id, card=chosen_card))
                state.played_cards.add(chosen_card)

            # Resolve trick winner & points
            winner = determine_trick_winner(state.current_trick, state.trump_suit)
            pts_thirds = trick_points_thirds(
                state.current_trick, is_last_trick=(trick_idx == 10)
            )
            state.hand_thirds[team_of(winner)] += pts_thirds

            led_suit = state.current_trick[0].card.suit
            record = TrickRecord(
                trick_number=trick_idx,
                leader=leader,
                signal=state.current_trick_signal,
                plays=tuple(state.current_trick),
                winner=winner,
                points_thirds=pts_thirds,
            )
            state.completed_tricks.append(record)

            # Update Busso return constraint for next trick
            if state.current_trick_signal == Signal.BUSSO and winner == partner_of(leader):
                state.pending_busso_return_suit = led_suit
            else:
                state.pending_busso_return_suit = None

            state.current_trick = []
            state.current_trick_signal = Signal.NONE
            state.current_trick_leader = winner

            # Call-out window at the end of completed tricks 1..9
            if allow_callout and trick_idx < 10:
                callout = self._check_callout_window(state, start_player=winner)
                if callout is not None:
                    return self._finalize_hand(state, callout=callout)

        return self._finalize_hand(state, callout=None)

    def play_match(self) -> MatchResult:
        """Play hands until a team reaches target_score (with sudden-death tie-break)."""
        match_scores = [0, 0]
        hands_log: list[HandResult] = []
        hand_number = 0
        next_caller: int | None = None
        in_sudden_death = False
        sudden_death_played = False
        winner_team: int | None = None
        ended_by_callout = False

        while winner_team is None:
            hand_number += 1
            hand_result = self.play_hand(
                hand_number=hand_number,
                trump_caller=next_caller,
                match_scores=(match_scores[0], match_scores[1]),
                allow_callout=not in_sudden_death,
            )
            hands_log.append(hand_result)
            next_caller = next_player(hand_result.trump_caller)

            if hand_result.callout is not None:
                ended_by_callout = True
                winner_team = hand_result.callout.winning_team
                match_scores = list(hand_result.callout.final_match_scores)
                break

            match_scores[0] += hand_result.hand_points[0]
            match_scores[1] += hand_result.hand_points[1]

            if in_sudden_death:
                if hand_result.hand_points[0] != hand_result.hand_points[1]:
                    winner_team = (
                        0 if hand_result.hand_points[0] > hand_result.hand_points[1] else 1
                    )
                continue

            if match_scores[0] >= self.target_score or match_scores[1] >= self.target_score:
                if match_scores[0] > match_scores[1]:
                    winner_team = 0
                elif match_scores[1] > match_scores[0]:
                    winner_team = 1
                else:
                    # Exact tie at >= target_score -> play a single sudden-death hand
                    in_sudden_death = True
                    sudden_death_played = True

        return MatchResult(
            target_score=self.target_score,
            winner_team=winner_team,
            final_scores=(match_scores[0], match_scores[1]),
            hands=hands_log,
            ended_by_callout=ended_by_callout,
            sudden_death_played=sudden_death_played,
        )

    def _resolve_cricca_phase(self, state: HandState) -> None:
        assert state.trump_suit is not None
        for offset in range(4):
            player_id = (state.trump_caller + offset) % 4
            if has_cricca(state.hands[player_id], state.trump_suit):
                obs = state.build_observation(player_id)
                if self.agents[player_id].decide_declare_cricca(obs):
                    state.cricca_declared_by = player_id
                    state.hand_thirds[team_of(player_id)] += CRICCA_BONUS_THIRDS
                # At most one player can hold trump Asso + Due + Tre.
                break

    def _check_callout_window(
        self, state: HandState, start_player: int
    ) -> CalloutOutcome | None:
        for offset in range(4):
            player_id = (start_player + offset) % 4
            obs = state.build_observation(player_id)
            if self.agents[player_id].decide_callout(obs):
                return evaluate_callout(
                    calling_team=team_of(player_id),
                    match_scores=state.match_scores,
                    hand_thirds=(state.hand_thirds[0], state.hand_thirds[1]),
                    target_score=state.target_score,
                )
        return None

    def _finalize_hand(
        self, state: HandState, callout: CalloutOutcome | None
    ) -> HandResult:
        assert state.trump_suit is not None
        return HandResult(
            hand_number=state.hand_number,
            trump_caller=state.trump_caller,
            trump_suit=state.trump_suit,
            cricca_declared_by=state.cricca_declared_by,
            tricks=list(state.completed_tricks),
            hand_thirds=(state.hand_thirds[0], state.hand_thirds[1]),
            hand_points=state.current_hand_points(),
            callout=callout,
        )