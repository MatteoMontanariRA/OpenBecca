"""Abstract base class shared by human players and all AI bots."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openbecca.cards import Card, Suit
from openbecca.state import Observation


class Agent(ABC):
    """Common decision interface for any Beccacino player (human or AI).

    Attributes:
        name: Human-readable display name.
        enforce_busso_constraints: If True (v1 AI default), the engine restricts
            legal moves to obey partner's Busso signals. False for humans and
            future unconstrained RL agents.
    """

    def __init__(self, name: str, enforce_busso_constraints: bool = True) -> None:
        self.name = name
        self.enforce_busso_constraints = enforce_busso_constraints

    @abstractmethod
    def choose_trump(self, obs: Observation) -> Suit:
        """Select the trump suit when acting as the hand's trump caller."""

    @abstractmethod
    def decide_declare_cricca(self, obs: Observation) -> bool:
        """Decide whether to declare a held trump Cricca (+3 points)."""

    @abstractmethod
    def decide_callout(self, obs: Observation) -> bool:
        """Decide whether to call out ('Chiamarsi fuori') at a valid window."""

    @abstractmethod
    def choose_card(self, obs: Observation) -> Card:
        """Choose one card from obs.legal_cards to play on the current trick."""