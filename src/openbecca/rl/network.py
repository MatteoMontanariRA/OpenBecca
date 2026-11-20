"""Compact CPU-friendly Actor-Critic neural network with legal action masking."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from openbecca.rl.features import NUM_ACTIONS, OBS_DIM


def _orthogonal_init(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class ActorCriticNet(nn.Module):
    """Shared-trunk MLP outputting masked card-play policy logits and state value."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 256,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        fc1 = nn.Linear(obs_dim, hidden_dim)
        fc2 = nn.Linear(hidden_dim, hidden_dim)
        _orthogonal_init(fc1, gain=np.sqrt(2.0))
        _orthogonal_init(fc2, gain=np.sqrt(2.0))

        self.trunk = nn.Sequential(
            fc1,
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            fc2,
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        _orthogonal_init(self.policy_head, gain=0.01)
        _orthogonal_init(self.value_head, gain=1.0)

    def forward(
        self, obs: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (masked_logits [B, 40], value [B])."""
        features = self.trunk(obs)
        raw_logits = self.policy_head(features)
        masked_logits = raw_logits.masked_fill(~legal_mask, -1e9)
        value = self.value_head(features).squeeze(-1)
        return masked_logits, value

    @torch.no_grad()
    def act_single(
        self,
        obs_np: np.ndarray,
        mask_np: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Select a single legal action from numpy observation & mask arrays."""
        obs_t = torch.from_numpy(obs_np).unsqueeze(0)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        masked_logits, value = self.forward(obs_t, mask_t)
        dist = Categorical(logits=masked_logits)

        if deterministic:
            action_t = torch.argmax(masked_logits, dim=-1)
        else:
            action_t = dist.sample()

        log_prob = dist.log_prob(action_t)
        return int(action_t.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log_probs, entropies, and values for a minibatch of transitions."""
        masked_logits, values = self.forward(obs, legal_mask)
        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values