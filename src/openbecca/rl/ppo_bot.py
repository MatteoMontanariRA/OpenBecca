"""Agent wrapper allowing a trained PPO ActorCriticNet to play in OpenBecca."""

from __future__ import annotations

from pathlib import Path

import torch

from openbecca.agents.base import Agent
from openbecca.agents.heuristic_bot import HeuristicBot
from openbecca.cards import Card, Suit
from openbecca.rl.features import (
    card_to_index,
    encode_observation,
    index_to_card,
    legal_action_mask,
)
from openbecca.rl.network import ActorCriticNet
from openbecca.rules import thirds_to_points
from openbecca.state import Observation

# Reuse H12 smart trump selection while RL-v1 focuses on card play.
_TRUMP_HELPER = HeuristicBot(name="TrumpHelper")


class PPOBot(Agent):
    """Neural-network agent backed by a trained ActorCriticNet checkpoint."""

    def __init__(
        self,
        name: str = "PPOBot",
        *,
        model: ActorCriticNet | None = None,
        checkpoint_path: str | Path | None = None,
        deterministic: bool = True,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        self.deterministic = deterministic

        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = self.load_model(checkpoint_path)
        else:
            self.model = ActorCriticNet()

        self.model.eval()

    @staticmethod
    def load_model(checkpoint_path: str | Path) -> ActorCriticNet:
        """Load an ActorCriticNet from a saved training checkpoint."""
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        model = ActorCriticNet(hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def choose_trump(self, obs: Observation) -> Suit:
        return _TRUMP_HELPER.choose_trump(obs)

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return True

    def decide_callout(self, obs: Observation) -> bool:
        team = obs.team_id
        total = obs.match_scores[team] + thirds_to_points(obs.hand_thirds[team])
        return total >= obs.target_score

    def choose_card(self, obs: Observation) -> Card:
        if not obs.legal_cards:
            raise ValueError("PPOBot received empty legal_cards.")
        if len(obs.legal_cards) == 1:
            return obs.legal_cards[0]

        obs_np = encode_observation(obs)
        mask_np = legal_action_mask(obs)
        action_idx, _, _ = self.model.act_single(
            obs_np, mask_np, deterministic=self.deterministic
        )
        return index_to_card(action_idx)

    @torch.no_grad()
    def predict_move_probabilities(
        self, obs: Observation
    ) -> tuple[dict[Card, float], float]:
        """Return (probability per legal card, state value estimate in [-1, 1])."""
        obs_t = torch.from_numpy(encode_observation(obs)).unsqueeze(0)
        mask_t = torch.from_numpy(legal_action_mask(obs)).unsqueeze(0)
        masked_logits, value = self.model.forward(obs_t, mask_t)
        probs = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()

        card_probs = {
            card: float(probs[card_to_index(card)]) for card in obs.legal_cards
        }
        return card_probs, float(value.item())