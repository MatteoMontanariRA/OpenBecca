"""AlphaZero-style Expert Iteration (ExIt): distill MCTS search into ActorCriticNet.

Workflow per generation:
1. Parallel multi-core self-play using a strong search teacher (ISMCTSBot for Gen 1,
   or NeuralISMCTSBot loaded from the previous generation for Gen 2+).
2. Collect (observation, legal_mask, mcts_visit_policy, exact_future_value) tuples.
3. Train ActorCriticNet via supervised Policy Cross-Entropy + Value MSE loss.
4. Evaluate the distilled network directly against HeuristicBot and save checkpoint.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openbecca.agents.heuristic_bot import HeuristicBot
from openbecca.agents.ismcts_bot import ISMCTSBot, MoveStats
from openbecca.cards import Deck, Suit
from openbecca.rl.features import (
    NUM_ACTIONS,
    OBS_DIM,
    card_to_index,
    encode_observation,
    legal_action_mask,
)
from openbecca.rl.network import ActorCriticNet
from openbecca.rl.neural_mcts import NeuralISMCTSBot
from openbecca.rl.ppo_bot import PPOBot
from openbecca.rl.trainer import evaluate_against_opponent
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

_TRUMP_HELPER = HeuristicBot(name="ExItTrumpHelper")


@dataclass
class ExpertIterConfig:
    """Configuration for one Expert Iteration generation."""

    teacher: str = "ismcts"  # "ismcts" (Gen 1) or "alphazero" (Gen 2+)
    teacher_checkpoint: str | None = None
    init_checkpoint: str | None = None
    output_checkpoint: str = "checkpoints/az_gen1.pt"
    dataset_path: str | None = "checkpoints/datasets/gen1_data.npz"
    load_dataset: str | None = None
    num_hands: int = 1000
    simulations: int = 120
    rollout_weight: float = 0.5
    policy_temp: float = 0.85
    hidden_dim: int = 256
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_coef: float = 1.0
    val_fraction: float = 0.1
    eval_deals: int = 100
    workers: int | None = None
    enforce_busso_constraints: bool = True
    seed: int = 42
    pi_sharpen: float = 1.0


def _stats_to_policy_target(stats: list[MoveStats], policy_temp: float) -> np.ndarray:
    """Convert root MCTS visit counts into a normalized 40-action probability target."""
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    inv_t = 1.0 / max(policy_temp, 1e-3)
    weights = np.array([max(0, s.visits) ** inv_t for s in stats], dtype=np.float64)
    total = float(weights.sum())

    if total <= 0.0:
        uniform = 1.0 / len(stats)
        for s in stats:
            pi[card_to_index(s.card)] = uniform
        return pi

    for s, w in zip(stats, weights):
        pi[card_to_index(s.card)] = float(w / total)
    return pi


def _generate_chunk(
    args: tuple[
        str,
        str | None,
        int,
        int,
        float,
        float,
        bool,
        int,
        int,
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker process generating Expert Iteration training examples from full hands."""
    (
        teacher_type,
        teacher_ckpt,
        num_hands,
        simulations,
        rollout_weight,
        policy_temp,
        enforce_busso,
        base_seed,
        chunk_id,
    ) = args

    torch.set_num_threads(1)
    shared_model = None
    if teacher_type == "alphazero":
        if teacher_ckpt is not None:
            shared_model = PPOBot.load_model(teacher_ckpt)
        else:
            shared_model = ActorCriticNet()
            shared_model.eval()

    deck = Deck(seed=base_seed + chunk_id * 100_003)

    obs_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []
    pi_buf: list[np.ndarray] = []
    val_buf: list[float] = []

    for hand_idx in range(num_hands):
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
        state.trump_suit = _TRUMP_HELPER.choose_trump(state.build_observation(caller))

        # Track fixed Cricca bonus separately so value head predicts play-earned margin
        cricca_thirds = [0, 0]
        for offset in range(4):
            pid = (caller + offset) % 4
            if has_cricca(state.hands[pid], state.trump_suit):
                state.cricca_declared_by = pid
                t = team_of(pid)
                state.hand_thirds[t] += CRICCA_BONUS_THIRDS
                cricca_thirds[t] += CRICCA_BONUS_THIRDS
                break

        s_seed = base_seed + chunk_id * 10_000 + hand_idx * 17
        teachers: list[ISMCTSBot | NeuralISMCTSBot] = []
        for seat in range(4):
            if teacher_type == "alphazero":
                teachers.append(
                    NeuralISMCTSBot(
                        name=f"TeacherAZ-{seat}",
                        model=shared_model,
                        simulations=simulations,
                        rollout_weight=rollout_weight,
                        seed=s_seed + seat,
                        enforce_busso_constraints=enforce_busso,
                    )
                )
            else:
                teachers.append(
                    ISMCTSBot(
                        name=f"TeacherIS-{seat}",
                        simulations=simulations,
                        seed=s_seed + seat,
                        enforce_busso_constraints=enforce_busso,
                    )
                )

        # Store per-hand records until final score is known: (obs, mask, pi, team, curr_no_cricca_diff)
        hand_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, float]] = []

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
                    busso_return_suit=(
                        state.pending_busso_return_suit if offset == 0 else None
                    ),
                )

                if len(legal) == 1:
                    chosen_card = legal[0]
                else:
                    obs = state.build_observation(pid, legal_cards=tuple(legal))
                    stats = teachers[pid].analyze_moves(obs, simulations=simulations)
                    pi_target = _stats_to_policy_target(stats, policy_temp)

                    t = team_of(pid)
                    my_nc = state.hand_thirds[t] - cricca_thirds[t]
                    opp_nc = state.hand_thirds[1 - t] - cricca_thirds[1 - t]
                    curr_nc_diff_pts = (my_nc - opp_nc) / 3.0

                    hand_records.append(
                        (
                            encode_observation(obs),
                            legal_action_mask(obs),
                            pi_target,
                            t,
                            curr_nc_diff_pts,
                        )
                    )

                    if teacher_type == "alphazero":
                        chosen_card = max(
                            stats, key=lambda s: (s.visits, s.expected_margin)
                        ).card
                    else:
                        chosen_card = max(
                            stats, key=lambda s: (s.expected_margin, s.visits)
                        ).card

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
            state.hand_thirds[team_of(winner)] += pts_thirds

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

        # Compute exact remaining non-Cricca margin normalized by /11.0 (matches NeuralISMCTSBot)
        t0_nc_pts = (state.hand_thirds[0] - cricca_thirds[0]) // 3
        t1_nc_pts = (state.hand_thirds[1] - cricca_thirds[1]) // 3
        final_t0_margin = float(t0_nc_pts - t1_nc_pts)

        for obs_v, mask_v, pi_v, t, curr_nc_diff_pts in hand_records:
            final_my_margin = final_t0_margin if t == 0 else -final_t0_margin
            future_margin_norm = (final_my_margin - curr_nc_diff_pts) / 11.0
            obs_buf.append(obs_v)
            mask_buf.append(mask_v)
            pi_buf.append(pi_v)
            val_buf.append(float(np.clip(future_margin_norm, -1.2, 1.2)))

    if not obs_buf:
        return (
            np.zeros((0, OBS_DIM), dtype=np.float32),
            np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
            np.zeros((0, NUM_ACTIONS), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(mask_buf, dtype=np.bool_),
        np.asarray(pi_buf, dtype=np.float32),
        np.asarray(val_buf, dtype=np.float32),
    )


def generate_expert_dataset(
    cfg: ExpertIterConfig, *, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate an Expert Iteration dataset in parallel across CPU cores."""
    t0 = time.perf_counter()
    num_workers = cfg.workers or max(1, min(os.cpu_count() or 1, 8))

    hands_per_worker, remainder = divmod(cfg.num_hands, num_workers)
    tasks = []
    for idx in range(num_workers):
        count = hands_per_worker + (1 if idx < remainder else 0)
        if count > 0:
            tasks.append(
                (
                    cfg.teacher,
                    cfg.teacher_checkpoint,
                    count,
                    cfg.simulations,
                    cfg.rollout_weight,
                    cfg.policy_temp,
                    cfg.enforce_busso_constraints,
                    cfg.seed,
                    idx,
                )
            )

    if verbose:
        print(
            f"[ExIt] Generating {cfg.num_hands} hands with teacher='{cfg.teacher}' "
            f"({cfg.simulations} sims/move, {len(tasks)} CPU workers)..."
        )

    if len(tasks) == 1:
        chunks = [_generate_chunk(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            chunks = list(pool.map(_generate_chunk, tasks))

    obs_arr = np.concatenate([c[0] for c in chunks], axis=0)
    mask_arr = np.concatenate([c[1] for c in chunks], axis=0)
    pi_arr = np.concatenate([c[2] for c in chunks], axis=0)
    val_arr = np.concatenate([c[3] for c in chunks], axis=0)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(
            f"[ExIt] Collected {len(obs_arr):,} non-trivial decisions in {elapsed:.1f}s "
            f"({cfg.num_hands / max(elapsed, 1e-6):.2f} hands/s)"
        )

    if cfg.dataset_path:
        out_path = Path(cfg.dataset_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path, obs=obs_arr, mask=mask_arr, pi=pi_arr, val=val_arr
        )
        if verbose:
            print(f"[ExIt] Saved dataset to {out_path}")

    return obs_arr, mask_arr, pi_arr, val_arr


def distill_dataset(
    obs_arr: np.ndarray,
    mask_arr: np.ndarray,
    pi_arr: np.ndarray,
    val_arr: np.ndarray,
    cfg: ExpertIterConfig,
    *,
    verbose: bool = True,
) -> Path:
    """Train ActorCriticNet via masked Cross-Entropy + Value MSE on expert data."""
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if cfg.init_checkpoint is not None:
        model = PPOBot.load_model(cfg.init_checkpoint)
        hidden_dim = model.hidden_dim
        if verbose:
            print(f"[ExIt] Warm-starting weights from {cfg.init_checkpoint}")
    else:
        hidden_dim = cfg.hidden_dim
        model = ActorCriticNet(hidden_dim=hidden_dim)

    if cfg.pi_sharpen != 1.0:
        pi_pow = np.where(mask_arr, np.power(np.maximum(pi_arr, 0.0), cfg.pi_sharpen), 0.0)
        row_sums = pi_pow.sum(axis=-1, keepdims=True)
        pi_arr = np.where(row_sums > 0.0, pi_pow / np.maximum(row_sums, 1e-12), pi_arr).astype(np.float32)

    n_total = len(obs_arr)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * cfg.val_fraction)) if n_total >= 20 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val > 0 else perm

    obs_train = torch.from_numpy(obs_arr[train_idx])
    mask_train = torch.from_numpy(mask_arr[train_idx])
    pi_train = torch.from_numpy(pi_arr[train_idx])
    val_train = torch.from_numpy(val_arr[train_idx])

    has_val = n_val > 0
    if has_val:
        obs_val = torch.from_numpy(obs_arr[val_idx])
        mask_val = torch.from_numpy(mask_arr[val_idx])
        pi_val = torch.from_numpy(pi_arr[val_idx])
        val_val = torch.from_numpy(val_arr[val_idx])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs)
    )

    best_val_loss = float("inf")
    best_state = None

    if verbose:
        print(
            f"[ExIt] Distilling {len(train_idx):,} train / {n_val:,} val samples "
            f"for {cfg.epochs} epochs (lr={cfg.lr}, batch={cfg.batch_size})..."
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(train_idx))
        tr_ploss_sum = 0.0
        tr_vloss_sum = 0.0
        tr_batches = 0

        for start in range(0, len(train_idx), cfg.batch_size):
            mb = order[start : start + cfg.batch_size]
            logits, values = model.forward(obs_train[mb], mask_train[mb])
            log_probs = torch.log_softmax(logits, dim=-1)
            safe_terms = torch.where(
                mask_train[mb], pi_train[mb] * log_probs, torch.zeros_like(log_probs)
            )
            policy_loss = -safe_terms.sum(dim=-1).mean()
            value_loss = F.mse_loss(values, val_train[mb])
            loss = policy_loss + cfg.value_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_ploss_sum += float(policy_loss.item())
            tr_vloss_sum += float(value_loss.item())
            tr_batches += 1

        scheduler.step()
        tr_p = tr_ploss_sum / max(1, tr_batches)
        tr_v = tr_vloss_sum / max(1, tr_batches)

        if has_val:
            model.eval()
            with torch.no_grad():
                v_logits, v_vals = model.forward(obs_val, mask_val)
                v_logp = torch.log_softmax(v_logits, dim=-1)
                v_safe = torch.where(
                    mask_val, pi_val * v_logp, torch.zeros_like(v_logp)
                )
                val_p = float((-v_safe.sum(dim=-1).mean()).item())
                val_v = float(F.mse_loss(v_vals, val_val).item())
                val_loss = val_p + cfg.value_coef * val_v
                top1_acc = float(
                    (v_logits.argmax(dim=-1) == pi_val.argmax(dim=-1))
                    .float()
                    .mean()
                    .item()
                )

            tag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                tag = " *"

            if verbose and (epoch == 1 or epoch % 3 == 0 or epoch == cfg.epochs):
                print(
                    f"  Epoch {epoch:2d}/{cfg.epochs} | "
                    f"Train P={tr_p:.3f} V={tr_v:.3f} | "
                    f"Val P={val_p:.3f} V={val_v:.3f} | "
                    f"Top-1 Match: {top1_acc * 100:5.1f}%{tag}"
                )
        else:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    out_path = Path(cfg.output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": 0,
            "total_hands": cfg.num_hands,
            "best_heuristic_edge": -999.0,
            "hidden_dim": hidden_dim,
            "source": f"expert_iteration:{cfg.teacher}",
        },
        out_path,
    )

    if verbose:
        print(f"[ExIt] Saved distilled checkpoint to {out_path}")
        if cfg.eval_deals > 0:
            h_edge, h_win = evaluate_against_opponent(
                model,
                "heuristic",
                num_deals=cfg.eval_deals,
                seed=2026,
                enforce_busso=cfg.enforce_busso_constraints,
            )
            print(
                f"[ExIt] Pure Neural (PPOBot) vs HeuristicBot "
                f"({cfg.eval_deals * 2} duplicate hands): "
                f"{h_edge:+.3f} pts/hand ({h_win * 100:.1f}% win rate)"
            )

    return out_path


def run_expert_iteration(cfg: ExpertIterConfig, *, verbose: bool = True) -> Path:
    """Execute full Expert Iteration: generate (or load) data -> distill -> evaluate."""
    if cfg.load_dataset is not None:
        data = np.load(Path(cfg.load_dataset))
        obs_arr = data["obs"]
        mask_arr = data["mask"]
        pi_arr = data["pi"]
        val_arr = data["val"]
        if verbose:
            print(f"[ExIt] Loaded {len(obs_arr):,} samples from {cfg.load_dataset}")
    else:
        obs_arr, mask_arr, pi_arr, val_arr = generate_expert_dataset(cfg, verbose=verbose)

    return distill_dataset(obs_arr, mask_arr, pi_arr, val_arr, cfg, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenBecca Expert Iteration (AlphaZero distillation)."
    )
    parser.add_argument(
        "--teacher",
        choices=["ismcts", "alphazero"],
        default="ismcts",
        help="Search teacher generating target policies (ismcts=Gen1, alphazero=Gen2+).",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default=None,
        help="Checkpoint used by teacher when --teacher=alphazero.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to warm-start neural weights (e.g. checkpoints/ppo_best.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/az_gen1.pt",
        help="Output checkpoint path for the distilled model.",
    )
    parser.add_argument(
        "--save-dataset",
        type=str,
        default="checkpoints/datasets/gen1_data.npz",
        help="Path to save generated .npz dataset.",
    )
    parser.add_argument(
        "--load-dataset",
        type=str,
        default=None,
        help="Skip generation and train directly from an existing .npz dataset.",
    )
    parser.add_argument("--hands", type=int, default=1000, help="Number of teacher hands.")
    parser.add_argument("--simulations", type=int, default=120, help="MCTS sims per move.")
    parser.add_argument("--rollout-weight", type=float, default=0.5, help="AlphaZero teacher blend.")
    parser.add_argument("--policy-temp", type=float, default=0.85, help="Visit sharpening temperature.")
    parser.add_argument("--epochs", type=int, default=15, help="Supervised training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--eval-deals", type=int, default=100, help="Eval deals (x2 duplicate) vs Heuristic.")
    parser.add_argument("--workers", type=int, default=None, help="CPU worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--value-coef", type=float, default=1.0, help="Value MSE loss weight.")
    parser.add_argument("--pi-sharpen", type=float, default=1.0, help="Exponent to sharpen loaded policy targets.")
    args = parser.parse_args()

    cfg = ExpertIterConfig(
        teacher=args.teacher,
        teacher_checkpoint=args.teacher_checkpoint,
        init_checkpoint=args.init_checkpoint,
        output_checkpoint=args.output,
        dataset_path=args.save_dataset,
        load_dataset=args.load_dataset,
        num_hands=args.hands,
        simulations=args.simulations,
        rollout_weight=args.rollout_weight,
        policy_temp=args.policy_temp,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_deals=args.eval_deals,
        workers=args.workers,
        seed=args.seed,
    )
    run_expert_iteration(cfg, verbose=True)


if __name__ == "__main__":
    main()