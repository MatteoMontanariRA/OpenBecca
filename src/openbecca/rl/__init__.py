"""Reinforcement Learning (Masked PPO Self-Play) package for OpenBecca."""

from openbecca.rl.features import (
    NUM_ACTIONS,
    OBS_DIM,
    card_to_index,
    encode_observation,
    index_to_card,
    legal_action_mask,
)
from openbecca.rl.network import ActorCriticNet
from openbecca.rl.ppo_bot import PPOBot
from openbecca.rl.trainer import PPOConfig, train_ppo

__all__ = [
    "NUM_ACTIONS",
    "OBS_DIM",
    "card_to_index",
    "index_to_card",
    "legal_action_mask",
    "encode_observation",
    "ActorCriticNet",
    "PPOBot",
    "PPOConfig",
    "train_ppo",
]