"""Unit tests for RL feature encoding, masked Actor-Critic, PPOBot, and training."""

from pathlib import Path

import numpy as np

from openbecca.agents import HeuristicBot
from openbecca.analysis.evaluator import build_custom_observation
from openbecca.cards import ALL_CARDS, Deck, Suit
from openbecca.engine import GameEngine
from openbecca.rl import (
    NUM_ACTIONS,
    OBS_DIM,
    PPOBot,
    PPOConfig,
    card_to_index,
    encode_observation,
    index_to_card,
    legal_action_mask,
    train_ppo,
)


def test_card_index_bijection() -> None:
    indices = [card_to_index(c) for c in ALL_CARDS]
    assert sorted(indices) == list(range(NUM_ACTIONS))
    for card in ALL_CARDS:
        assert index_to_card(card_to_index(card)) == card


def test_observation_encoder_shape_and_mask() -> None:
    hands = Deck(seed=42).deal()
    obs = build_custom_observation(
        player_hand=hands[0], trump_suit=Suit.COINS, player_id=0, trump_caller=0
    )
    vec = encode_observation(obs)
    mask = legal_action_mask(obs)

    assert vec.shape == (OBS_DIM,)
    assert vec.dtype == np.float32
    assert mask.shape == (NUM_ACTIONS,)
    assert int(mask.sum()) == len(obs.legal_cards)


def test_ppo_bot_plays_legal_hand_and_saves_loads_checkpoint(tmp_path: Path) -> None:
    cfg = PPOConfig(
        updates=2,
        hands_per_update=4,
        hidden_dim=64,
        minibatch_size=32,
        ppo_epochs=2,
        eval_every=2,
        eval_deals=4,
        checkpoint_dir=str(tmp_path),
        seed=7,
    )
    latest_ckpt = train_ppo(cfg, verbose=False)
    assert latest_ckpt.exists()
    assert (tmp_path / "ppo_best.pt").exists()

    # Load saved checkpoint into PPOBot and play a complete hand vs HeuristicBot
    agents = [
        PPOBot("PPO-0", checkpoint_path=latest_ckpt, deterministic=True),
        HeuristicBot("H-1"),
        PPOBot("PPO-2", checkpoint_path=latest_ckpt, deterministic=True),
        HeuristicBot("H-3"),
    ]
    res = GameEngine(agents, seed=11).play_hand(allow_callout=False)
    assert len(res.tricks) == 10
    expected_total = 14 if res.cricca_declared_by is not None else 11
    assert sum(res.hand_points) == expected_total