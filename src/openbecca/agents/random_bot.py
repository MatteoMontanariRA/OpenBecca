"""Baseline random legal agent used for smoke testing and benchmarking."""

from __future__ import annotations

import random

from openbecca.agents.base import Agent
from openbecca.cards import Card, Suit
from openbecca.rules import thirds_to_points
from openbecca.state import Observation


class RandomBot(Agent):
    """Plays uniformly at random among legal moves.

    Always declares Cricca when held, and calls out only when verified points
    already reach the match target score.
    """

    def __init__(
        self,
        name: str = "RandomBot",
        seed: int | None = None,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        self._rng = random.Random(seed)

    def choose_trump(self, obs: Observation) -> Suit:
        return self._rng.choice(list(Suit))

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return True

    def decide_callout(self, obs: Observation) -> bool:
        team = obs.team_id
        current_total = obs.match_scores[team] + thirds_to_points(obs.hand_thirds[team])
        return current_total >= obs.target_score

    def choose_card(self, obs: Observation) -> Card:
        if not obs.legal_cards:
            raise ValueError("RandomBot received empty legal_cards.")
        return self._rng.choice(obs.legal_cards)