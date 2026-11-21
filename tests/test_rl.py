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
    NeuralISMCTSBot,
    PPOBot,
    PPOConfig,
    ExpertIterConfig,
    card_to_index,
    encode_observation,
    index_to_card,
    legal_action_mask,
    train_ppo,
    run_expert_iteration,
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

def test_neural_ismcts_bot_pure_and_hybrid_legal_play() -> None:
    for weight in (0.0, 0.5):
        agents = [
            NeuralISMCTSBot(
                "AZ-0", simulations=20, trump_simulations=6, rollout_weight=weight, seed=1
            ),
            HeuristicBot("H-1"),
            NeuralISMCTSBot(
                "AZ-2", simulations=20, trump_simulations=6, rollout_weight=weight, seed=2
            ),
            HeuristicBot("H-3"),
        ]
        res = GameEngine(agents, seed=19).play_hand(allow_callout=False)
        assert len(res.tricks) == 10
        expected_total = 14 if res.cricca_declared_by is not None else 11
        assert sum(res.hand_points) == expected_total

def test_expert_iteration_generates_distills_and_loads(tmp_path: Path) -> None:
    out_ckpt = tmp_path / "az_test.pt"
    out_data = tmp_path / "data.npz"
    cfg = ExpertIterConfig(
        teacher="ismcts",
        output_checkpoint=str(out_ckpt),
        dataset_path=str(out_data),
        num_hands=3,
        simulations=16,
        hidden_dim=64,
        epochs=2,
        batch_size=16,
        eval_deals=0,
        workers=1,
        seed=13,
    )
    saved = run_expert_iteration(cfg, verbose=False)
    assert saved.exists()
    assert out_data.exists()

    bot = PPOBot("ExIt-0", checkpoint_path=saved, deterministic=True)
    res = GameEngine(
        [bot, HeuristicBot("H-1"), PPOBot("ExIt-2", model=bot.model), HeuristicBot("H-3")],
        seed=5,
    ).play_hand(allow_callout=False)
    assert len(res.tricks) == 10

def test_fast_numpy_net_matches_pytorch_forward() -> None:
    import torch
    from openbecca.rl.network import ActorCriticNet
    from openbecca.rl.neural_mcts import _FastNumpyNet

    torch.manual_seed(99)
    model = ActorCriticNet(hidden_dim=128)
    model.eval()
    fast_net = _FastNumpyNet(model)

    hands = Deck(seed=7).deal()
    obs = build_custom_observation(
        player_hand=hands[0], trump_suit=Suit.SWORDS, player_id=0, trump_caller=0
    )
    obs_np = encode_observation(obs)
    mask_np = legal_action_mask(obs)
    prior_temp = 1.5

    fast_probs, fast_val = fast_net.predict(obs_np, mask_np, prior_temp)
    with torch.no_grad():
        logits, val = model.forward(
            torch.from_numpy(obs_np).unsqueeze(0),
            torch.from_numpy(mask_np).unsqueeze(0),
        )
        torch_probs = torch.softmax(logits / prior_temp, dim=-1).squeeze(0).numpy()
        torch_val = float(val.item())

    assert np.allclose(fast_probs, torch_probs, atol=1e-5)
    assert abs(fast_val - torch_val) < 1e-5