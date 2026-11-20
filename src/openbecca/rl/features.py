"""Imperfect-information observation encoder and action indexing for RL agents."""

from __future__ import annotations

import numpy as np

from openbecca.cards import ALL_CARDS, Card, Suit
from openbecca.rules import determine_trick_winner
from openbecca.signals import Signal
from openbecca.state import Observation

NUM_ACTIONS = 40
OBS_DIM = 250

# Canonical index 0..39: suit.order * 10 + (rank.power - 1)
INDEX_TO_CARD: tuple[Card, ...] = tuple(
    sorted(ALL_CARDS, key=lambda c: (c.suit.order, c.rank.power))
)

_SIGNAL_TO_IDX: dict[Signal, int] = {
    Signal.NONE: 0,
    Signal.BUSSO: 1,
    Signal.VOLO: 2,
    Signal.STRISCIO: 3,
}


def card_to_index(card: Card) -> int:
    """Map a Card bijectively to an action/feature index in [0, 39]."""
    return card.suit.order * 10 + (card.rank.power - 1)


def index_to_card(index: int) -> Card:
    """Map an index in [0, 39] back to its Card."""
    return INDEX_TO_CARD[index]


def legal_action_mask(obs: Observation) -> np.ndarray:
    """Return a boolean mask of shape (40,) with True on legal actions."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for card in obs.legal_cards:
        mask[card_to_index(card)] = True
    return mask


def _infer_public_voids(obs: Observation) -> dict[int, set[Suit]]:
    """Infer guaranteed voids strictly from completed tricks, current trick, and Volo."""
    voids: dict[int, set[Suit]] = {p: set() for p in range(4)}

    for trick in obs.completed_tricks:
        led_suit = trick.led_suit
        if trick.signal == Signal.VOLO:
            voids[trick.leader].add(led_suit)
        for play in trick.plays[1:]:
            if play.card.suit != led_suit:
                voids[play.player].add(led_suit)

    if obs.current_trick:
        led_suit = obs.current_trick[0].card.suit
        if obs.current_trick_signal == Signal.VOLO:
            voids[obs.current_trick_leader].add(led_suit)
        for play in obs.current_trick[1:]:
            if play.card.suit != led_suit:
                voids[play.player].add(led_suit)

    return voids


def encode_observation(obs: Observation) -> np.ndarray:
    """Encode an imperfect-information Observation into a 250-dim float32 vector.

    All seat-dependent features are rotated relative to obs.player_id so the
    network is seat-invariant (rel 0 = me, 1 = right opp, 2 = partner, 3 = left opp).
    """
    vec = np.zeros(OBS_DIM, dtype=np.float32)
    me = obs.player_id
    offset = 0

    # 1. My current hand (40)
    for card in obs.hand:
        vec[offset + card_to_index(card)] = 1.0
    offset += 40

    # 2. All cards played so far (40)
    for card in obs.played_cards:
        vec[offset + card_to_index(card)] = 1.0
    offset += 40

    # 3. Cards currently on the table by relative seat 1, 2, 3 (120)
    for play in obs.current_trick:
        rel = (play.player - me) % 4
        if rel in (1, 2, 3):
            vec[offset + (rel - 1) * 40 + card_to_index(play.card)] = 1.0
    offset += 120

    # 4. Trump suit one-hot (4)
    if obs.trump_suit is not None:
        vec[offset + obs.trump_suit.order] = 1.0
    offset += 4

    # 5. Led suit one-hot + 'leading' flag (5)
    if obs.current_trick:
        vec[offset + obs.current_trick[0].card.suit.order] = 1.0
    else:
        vec[offset + 4] = 1.0
    offset += 5

    # 6. Current trick leader signal one-hot (4)
    vec[offset + _SIGNAL_TO_IDX[obs.current_trick_signal]] = 1.0
    offset += 4

    # 7. Publicly inferred voids for relative seats 1, 2, 3 across 4 suits (12)
    voids = _infer_public_voids(obs)
    for rel in (1, 2, 3):
        pid = (me + rel) % 4
        for suit in voids[pid]:
            vec[offset + (rel - 1) * 4 + suit.order] = 1.0
    offset += 12

    # 8. Relative seat of current trick leader (4)
    vec[offset + ((obs.current_trick_leader - me) % 4)] = 1.0
    offset += 4

    # 9. Relative seat of current trick temporary winner + 'empty table' flag (5)
    if obs.current_trick and obs.trump_suit is not None:
        curr_winner = determine_trick_winner(obs.current_trick, obs.trump_suit)
        vec[offset + ((curr_winner - me) % 4)] = 1.0
    else:
        vec[offset + 4] = 1.0
    offset += 5

    # 10. Relative seat of Cricca declarer + 'none' flag (5)
    if obs.cricca_declared_by is not None:
        vec[offset + ((obs.cricca_declared_by - me) % 4)] = 1.0
    else:
        vec[offset + 4] = 1.0
    offset += 5

    # 11. Pending Busso return suit + 'none' flag (5)
    if obs.pending_busso_return_suit is not None:
        vec[offset + obs.pending_busso_return_suit.order] = 1.0
    else:
        vec[offset + 4] = 1.0
    offset += 5

    # 12. Normalized scalar context features (6)
    my_thirds = obs.hand_thirds[obs.team_id]
    opp_thirds = obs.hand_thirds[1 - obs.team_id]
    vec[offset + 0] = obs.current_trick_number / 10.0
    vec[offset + 1] = len(obs.current_trick) / 3.0
    vec[offset + 2] = my_thirds / 35.0
    vec[offset + 3] = opp_thirds / 35.0
    vec[offset + 4] = (my_thirds - opp_thirds) / 35.0
    vec[offset + 5] = 1.0 if obs.trump_caller == me else 0.0
    offset += 6

    assert offset == OBS_DIM
    return vec