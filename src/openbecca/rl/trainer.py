"""Self-Play + League Masked PPO trainer for OpenBecca (CPU-optimized)."""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openbecca.agents.heuristic_bot import HeuristicBot
from openbecca.agents.random_bot import RandomBot
from openbecca.cards import Card, Deck, Suit
from openbecca.engine import GameEngine
from openbecca.rl.features import (
    card_to_index,
    encode_observation,
    index_to_card,
    legal_action_mask,
)
from openbecca.rl.network import ActorCriticNet
from openbecca.rl.ppo_bot import PPOBot
from openbecca.rules import (
    CRICCA_BONUS_THIRDS,
    TrickPlay,
    determine_trick_winner,
    find_four_of_coins_holder,
    get_legal_cards,
    has_cricca,
    partner_of,
    team_of,
    trick_points_thirds,
)
from openbecca.signals import Signal, determine_lead_signal
from openbecca.state import HandState, TrickRecord

_HEURISTIC_OPPONENT = HeuristicBot(name="LeagueHeuristic")


@dataclass
class PPOConfig:
    """Hyperparameters tuned for 4-player 10-trick Beccacino on CPU."""

    updates: int = 500
    hands_per_update: int = 128
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.015
    ppo_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    self_play_prob: float = 0.55
    league_prob: float = 0.25
    max_snapshots: int = 10
    snapshot_every: int = 20
    eval_every: int = 10
    eval_deals: int = 80
    enforce_busso_constraints: bool = True
    seed: int = 42
    checkpoint_dir: str = "checkpoints"


@dataclass
class _StepRecord:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0


def _collect_hand_trajectories(
    current_net: ActorCriticNet,
    opponent_mode: str,
    snapshot_net: ActorCriticNet | None,
    deck: Deck,
    *,
    enforce_busso: bool,
) -> list[list[_StepRecord]]:
    """Simulate one full 10-trick hand and collect trajectories for learning seats."""
    hands = deck.deal()
    caller = find_four_of_coins_holder(hands)
    init_counts = [
        {s: Counter(c.suit for c in h).get(s, 0) for s in Suit} for h in hands
    ]

    state = HandState(
        hand_number=1,
        trump_caller=caller,
        hands=hands,
        initial_suit_counts=init_counts,
        match_scores=(0, 0),
        target_score=41,
    )

    # Trump selection (H12 helper)
    state.trump_suit = _HEURISTIC_OPPONENT.choose_trump(state.build_observation(caller))

    # Cricca phase (track fixed Cricca thirds separately to avoid reward noise)
    cricca_thirds = [0, 0]
    for offset in range(4):
        pid = (caller + offset) % 4
        if has_cricca(state.hands[pid], state.trump_suit):
            state.cricca_declared_by = pid
            t = team_of(pid)
            state.hand_thirds[t] += CRICCA_BONUS_THIRDS
            cricca_thirds[t] += CRICCA_BONUS_THIRDS
            break

    # In pure self-play all 4 players train; otherwise Team 0 (seats 0 & 2) trains.
    learning_seats = {0, 1, 2, 3} if opponent_mode == "self" else {0, 2}
    trajectories: dict[int, list[_StepRecord]] = {p: [] for p in learning_seats}

    for trick_idx in range(1, 11):
        leader = state.current_trick_leader
        state.current_trick = []
        state.current_trick_signal = Signal.NONE

        for offset in range(4):
            pid = (leader + offset) % 4
            is_first_cricca_lead = (
                trick_idx == 1
                and offset == 0
                and state.cricca_declared_by == pid
                and pid == state.trump_caller
            )
            legal = get_legal_cards(
                hand=state.hands[pid],
                current_trick=state.current_trick,
                trump_suit=state.trump_suit,
                is_first_trick_cricca_lead=is_first_cricca_lead,
                enforce_busso_constraints=enforce_busso,
                current_trick_signal=state.current_trick_signal,
                current_trick_leader=leader,
                player_id=pid,
                busso_return_suit=state.pending_busso_return_suit if offset == 0 else None,
            )
            obs = state.build_observation(pid, legal_cards=tuple(legal))

            if pid in learning_seats:
                obs_np = encode_observation(obs)
                mask_np = legal_action_mask(obs)
                action_idx, log_prob, value = current_net.act_single(
                    obs_np, mask_np, deterministic=False
                )
                chosen_card = index_to_card(action_idx)
                trajectories[pid].append(
                    _StepRecord(
                        obs=obs_np,
                        mask=mask_np,
                        action=action_idx,
                        log_prob=log_prob,
                        value=value,
                    )
                )
            elif opponent_mode == "snapshot" and snapshot_net is not None:
                obs_np = encode_observation(obs)
                mask_np = legal_action_mask(obs)
                action_idx, _, _ = snapshot_net.act_single(
                    obs_np, mask_np, deterministic=False
                )
                chosen_card = index_to_card(action_idx)
            else:
                chosen_card = _HEURISTIC_OPPONENT.choose_card(obs)

            if offset == 0:
                state.current_trick_signal = determine_lead_signal(
                    played_card=chosen_card,
                    hand_before_play=state.hands[pid],
                    initial_suit_count=state.initial_suit_counts[pid][chosen_card.suit],
                    played_cards_history=state.played_cards,
                )

            state.hands[pid].remove(chosen_card)
            state.current_trick.append(TrickPlay(player=pid, card=chosen_card))
            state.played_cards.add(chosen_card)

        winner = determine_trick_winner(state.current_trick, state.trump_suit)
        pts_thirds = trick_points_thirds(
            state.current_trick, is_last_trick=(trick_idx == 10)
        )
        winner_team = team_of(winner)
        state.hand_thirds[winner_team] += pts_thirds

        # Dense per-trick reward in thirds normalized to [-1, +1] scale
        for pid in learning_seats:
            signed_thirds = pts_thirds if team_of(pid) == winner_team else -pts_thirds
            trajectories[pid][-1].reward = signed_thirds / 33.0

        led_suit = state.current_trick[0].card.suit
        state.completed_tricks.append(
            TrickRecord(
                trick_number=trick_idx,
                leader=leader,
                signal=state.current_trick_signal,
                plays=tuple(state.current_trick),
                winner=winner,
                points_thirds=pts_thirds,
            )
        )
        if state.current_trick_signal == Signal.BUSSO and winner == partner_of(leader):
            state.pending_busso_return_suit = led_suit
        else:
            state.pending_busso_return_suit = None

        state.current_trick = []
        state.current_trick_signal = Signal.NONE
        state.current_trick_leader = winner

    # Exact terminal rounding adjustment so total reward matches integer floor scoring
    t0_thirds = state.hand_thirds[0] - cricca_thirds[0]
    t1_thirds = state.hand_thirds[1] - cricca_thirds[1]
    t0_int_margin = (t0_thirds // 3) - (t1_thirds // 3)
    t0_raw_sum = (t0_thirds - t1_thirds) / 33.0
    t0_adj = (t0_int_margin / 11.0) - t0_raw_sum

    for pid in learning_seats:
        adj = t0_adj if team_of(pid) == 0 else -t0_adj
        trajectories[pid][-1].reward += adj

    return list(trajectories.values())


def _compute_gae_batch(
    trajectories: list[list[_StepRecord]],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten player trajectories into tensors with GAE advantages and returns."""
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    act_list: list[int] = []
    logp_list: list[float] = []
    adv_list: list[float] = []
    ret_list: list[float] = []

    for traj in trajectories:
        n = len(traj)
        advantages = [0.0] * n
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            delta = traj[t].reward + gamma * next_value - traj[t].value
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            next_value = traj[t].value

        for t in range(n):
            obs_list.append(traj[t].obs)
            mask_list.append(traj[t].mask)
            act_list.append(traj[t].action)
            logp_list.append(traj[t].log_prob)
            adv_list.append(advantages[t])
            ret_list.append(advantages[t] + traj[t].value)

    obs_t = torch.from_numpy(np.asarray(obs_list, dtype=np.float32))
    mask_t = torch.from_numpy(np.asarray(mask_list, dtype=np.bool_))
    act_t = torch.tensor(act_list, dtype=torch.long)
    logp_t = torch.tensor(logp_list, dtype=torch.float32)
    adv_t = torch.tensor(adv_list, dtype=torch.float32)
    ret_t = torch.tensor(ret_list, dtype=torch.float32)

    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    return obs_t, mask_t, act_t, logp_t, adv_t, ret_t


def evaluate_against_opponent(
    model: ActorCriticNet,
    opponent_kind: str,
    *,
    num_deals: int = 80,
    seed: int = 999,
    enforce_busso: bool = True,
) -> tuple[float, float]:
    """Evaluate PPOBot (deterministic) on duplicate hands vs RandomBot or HeuristicBot.

    Returns (net_point_edge_per_hand, hand_win_rate).
    """
    model.eval()
    deck = Deck(seed=seed)
    ppo_pts = 0
    opp_pts = 0
    ppo_wins = 0
    total_hands = 0

    for idx in range(num_deals):
        dealt = deck.deal()

        def _opp(seat: int, s: int):
            if opponent_kind == "random":
                return RandomBot(f"R-{seat}", seed=s, enforce_busso_constraints=enforce_busso)
            return HeuristicBot(f"H-{seat}", enforce_busso_constraints=enforce_busso)

        # Duplicate Run 1: PPO in Team 0 (seats 0 & 2)
        agents_1 = [
            PPOBot("PPO-0", model=model, deterministic=True, enforce_busso_constraints=enforce_busso),
            _opp(1, seed + idx * 10 + 1),
            PPOBot("PPO-2", model=model, deterministic=True, enforce_busso_constraints=enforce_busso),
            _opp(3, seed + idx * 10 + 3),
        ]
        r1 = GameEngine(agents_1).play_hand(allow_callout=False, dealt_hands=dealt)
        ppo_pts += r1.hand_points[0]
        opp_pts += r1.hand_points[1]
        if r1.hand_points[0] > r1.hand_points[1]:
            ppo_wins += 1
        total_hands += 1

        # Duplicate Run 2: PPO in Team 1 (seats 1 & 3)
        agents_2 = [
            _opp(0, seed + idx * 10 + 4),
            PPOBot("PPO-1", model=model, deterministic=True, enforce_busso_constraints=enforce_busso),
            _opp(2, seed + idx * 10 + 6),
            PPOBot("PPO-3", model=model, deterministic=True, enforce_busso_constraints=enforce_busso),
        ]
        r2 = GameEngine(agents_2).play_hand(allow_callout=False, dealt_hands=dealt)
        ppo_pts += r2.hand_points[1]
        opp_pts += r2.hand_points[0]
        if r2.hand_points[1] > r2.hand_points[0]:
            ppo_wins += 1
        total_hands += 1

    edge = (ppo_pts - opp_pts) / total_hands
    win_rate = ppo_wins / total_hands
    return edge, win_rate


def _save_checkpoint(
    path: Path,
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    total_hands: int,
    best_heuristic_edge: float,
    hidden_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "total_hands": total_hands,
            "best_heuristic_edge": best_heuristic_edge,
            "hidden_dim": hidden_dim,
        },
        path,
    )


def train_ppo(
    cfg: PPOConfig,
    *,
    resume_path: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Run the self-play Masked PPO training loop and return path to latest checkpoint."""
    # Single-thread rollouts avoid CPU thread-sync overhead on small MLPs.
    torch.set_num_threads(2)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = ActorCriticNet(hidden_dim=cfg.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    start_update = 0
    total_hands = 0
    best_heuristic_edge = -999.0

    if resume_path is not None:
        ckpt = torch.load(Path(resume_path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt.get("update", 0))
        total_hands = int(ckpt.get("total_hands", 0))
        best_heuristic_edge = float(ckpt.get("best_heuristic_edge", -999.0))
        if verbose:
            print(
                f"Resumed from {resume_path} (update={start_update}, "
                f"hands={total_hands}, best_vs_heuristic={best_heuristic_edge:+.3f})"
            )

    ckpt_dir = Path(cfg.checkpoint_dir)
    latest_path = ckpt_dir / "ppo_latest.pt"
    best_path = ckpt_dir / "ppo_best.pt"

    rng = random.Random(cfg.seed + start_update * 97)
    deck = Deck(seed=cfg.seed + start_update * 1009)
    snapshots: list[dict[str, torch.Tensor]] = []
    snapshot_net = ActorCriticNet(hidden_dim=cfg.hidden_dim)
    snapshot_net.eval()

    t0 = time.perf_counter()
    if verbose:
        print(
            f"Starting OpenBecca Masked PPO training | updates={cfg.updates} | "
            f"hands/update={cfg.hands_per_update} | hidden={cfg.hidden_dim} | "
            f"busso_constraint={cfg.enforce_busso_constraints}"
        )

    current_update = start_update
    try:
        for step in range(1, cfg.updates + 1):
            current_update = start_update + step
            model.eval()
            all_trajs: list[list[_StepRecord]] = []

            # 1. Rollout phase: play hands in Self-Play / League / Heuristic mixture
            for _ in range(cfg.hands_per_update):
                u = rng.random()
                if u < cfg.self_play_prob:
                    mode = "self"
                    snap = None
                elif u < cfg.self_play_prob + cfg.league_prob and snapshots:
                    mode = "snapshot"
                    snapshot_net.load_state_dict(rng.choice(snapshots))
                    snap = snapshot_net
                else:
                    mode = "heuristic"
                    snap = None

                trajs = _collect_hand_trajectories(
                    model,
                    mode,
                    snap,
                    deck,
                    enforce_busso=cfg.enforce_busso_constraints,
                )
                all_trajs.extend(trajs)
                total_hands += 1

            obs_t, mask_t, act_t, old_logp_t, adv_t, ret_t = _compute_gae_batch(
                all_trajs, cfg.gamma, cfg.gae_lambda
            )

            # 2. PPO optimization phase
            model.train()
            n_samples = obs_t.shape[0]
            indices = np.arange(n_samples)
            last_policy_loss = 0.0
            last_value_loss = 0.0
            last_entropy = 0.0

            for _ in range(cfg.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_samples, cfg.minibatch_size):
                    mb = indices[start : start + cfg.minibatch_size]
                    new_logp, entropy, new_values = model.evaluate_actions(
                        obs_t[mb], mask_t[mb], act_t[mb]
                    )

                    ratio = torch.exp(new_logp - old_logp_t[mb])
                    surr1 = ratio * adv_t[mb]
                    surr2 = (
                        torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                        * adv_t[mb]
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (ret_t[mb] - new_values).pow(2).mean()
                    entropy_mean = entropy.mean()

                    loss = (
                        policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy_mean
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    last_policy_loss = float(policy_loss.item())
                    last_value_loss = float(value_loss.item())
                    last_entropy = float(entropy_mean.item())

            # 3. Periodically add frozen snapshot to opponent league
            if current_update % cfg.snapshot_every == 0:
                snapshots.append(copy.deepcopy(model.state_dict()))
                if len(snapshots) > cfg.max_snapshots:
                    snapshots.pop(0)

            # 4. Periodic evaluation & checkpointing
            if step == 1 or step % cfg.eval_every == 0 or step == cfg.updates:
                elapsed = time.perf_counter() - t0
                h_edge, h_win = evaluate_against_opponent(
                    model,
                    "heuristic",
                    num_deals=cfg.eval_deals,
                    seed=2026,
                    enforce_busso=cfg.enforce_busso_constraints,
                )
                r_edge, r_win = evaluate_against_opponent(
                    model,
                    "random",
                    num_deals=max(30, cfg.eval_deals // 2),
                    seed=3026,
                    enforce_busso=cfg.enforce_busso_constraints,
                )

                saved_tag = ""
                if h_edge > best_heuristic_edge:
                    best_heuristic_edge = h_edge
                    _save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        update=current_update,
                        total_hands=total_hands,
                        best_heuristic_edge=best_heuristic_edge,
                        hidden_dim=cfg.hidden_dim,
                    )
                    saved_tag = " [NEW BEST]"

                _save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    update=current_update,
                    total_hands=total_hands,
                    best_heuristic_edge=best_heuristic_edge,
                    hidden_dim=cfg.hidden_dim,
                )

                if verbose:
                    print(
                        f"Upd {current_update:4d} | Hands {total_hands:7d} | "
                        f"vs Rand: {r_edge:+.2f} ({r_win * 100:4.1f}%) | "
                        f"vs Heur: {h_edge:+.2f} ({h_win * 100:4.1f}%) | "
                        f"VLoss {last_value_loss:.3f} | Ent {last_entropy:.2f} | "
                        f"{elapsed:5.1f}s{saved_tag}"
                    )

    except KeyboardInterrupt:
        if verbose:
            print("\nTraining interrupted by user — saving latest checkpoint safely...")
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            update=current_update,
            total_hands=total_hands,
            best_heuristic_edge=best_heuristic_edge,
            hidden_dim=cfg.hidden_dim,
        )
        if verbose:
            print(f"Saved {latest_path}. Resume anytime with --resume {latest_path}")

    return latest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenBecca Masked PPO agent.")
    parser.add_argument("--updates", type=int, default=300, help="Number of PPO updates.")
    parser.add_argument("--hands-per-update", type=int, default=128, help="Hands collected per update.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden layer width.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N updates.")
    parser.add_argument("--eval-deals", type=int, default=80, help="Unique deals (x2 duplicate) per eval.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint output folder.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--free-busso",
        action="store_true",
        help="Disable strict v1 Busso constraint so RL learns freely when to obey Busso.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    cfg = PPOConfig(
        updates=args.updates,
        hands_per_update=args.hands_per_update,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        eval_every=args.eval_every,
        eval_deals=args.eval_deals,
        checkpoint_dir=args.checkpoint_dir,
        enforce_busso_constraints=not args.free_busso,
        seed=args.seed,
    )
    train_ppo(cfg, resume_path=args.resume, verbose=True)


if __name__ == "__main__":
    main()