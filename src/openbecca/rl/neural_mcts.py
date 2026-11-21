"""AlphaZero-style Neural IS-MCTS (PUCT + Policy Prior + Value Leaf Evaluation).

CPU-optimized with:
- Zero-overhead NumPy single-sample inference (_FastNumpyNet).
- Delayed internal node expansion (expand_after_visits) when rollout_weight=1.0.
- Automatic bypass of neural inference on forced 1-card moves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openbecca.agents.base import Agent
from openbecca.agents.ismcts_bot import (
    MoveStats,
    _SimState,
    infer_belief_constraints,
    sample_consistent_hands,
)
from openbecca.cards import ALL_CARDS, Card, Suit
from openbecca.rl.features import (
    card_to_index,
    encode_observation,
    legal_action_mask,
)
from openbecca.rl.network import ActorCriticNet
from openbecca.rl.ppo_bot import PPOBot
from openbecca.rules import (
    CRICCA_BONUS_THIRDS,
    has_cricca,
    team_of,
    thirds_to_points,
)
from openbecca.signals import Signal
from openbecca.state import Observation


class _FastNumpyNet:
    """Zero-overhead NumPy forward pass matching ActorCriticNet for batch=1 CPU inference."""

    def __init__(self, model: ActorCriticNet) -> None:
        fc1 = model.trunk[0]
        ln1 = model.trunk[1]
        fc2 = model.trunk[3]
        ln2 = model.trunk[4]

        self.w1 = fc1.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        self.b1 = fc1.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        self.ln1_w = ln1.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        self.ln1_b = ln1.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        self.ln1_eps = float(ln1.eps)

        self.w2 = fc2.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        self.b2 = fc2.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        self.ln2_w = ln2.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        self.ln2_b = ln2.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        self.ln2_eps = float(ln2.eps)

        self.wp = model.policy_head.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        self.bp = model.policy_head.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        self.wv = (
            model.value_head.weight.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=True)
        )
        self.bv = float(model.value_head.bias.detach().cpu().item())

    @staticmethod
    def _layer_norm_silu(
        x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float
    ) -> np.ndarray:
        mean = float(x.mean())
        diff = x - mean
        inv_std = 1.0 / math.sqrt(float(np.dot(diff, diff)) / x.size + eps)
        normed = diff * inv_std * weight + bias
        clipped = np.clip(normed, -30.0, 30.0)
        return normed / (1.0 + np.exp(-clipped))

    def predict(
        self, obs_np: np.ndarray, mask_np: np.ndarray, prior_temp: float
    ) -> tuple[np.ndarray, float]:
        """Return (masked softmax probs [40], scalar value) without PyTorch dispatch."""
        h = self.w1 @ obs_np + self.b1
        h = self._layer_norm_silu(h, self.ln1_w, self.ln1_b, self.ln1_eps)
        h = self.w2 @ h + self.b2
        h = self._layer_norm_silu(h, self.ln2_w, self.ln2_b, self.ln2_eps)

        logits = self.wp @ h + self.bp
        value = float(np.dot(self.wv, h) + self.bv)

        valid_logits = logits[mask_np] / prior_temp
        valid_logits = valid_logits - float(valid_logits.max())
        valid_exps = np.exp(valid_logits)

        probs = np.zeros(logits.shape, dtype=np.float32)
        probs[mask_np] = (valid_exps / float(valid_exps.sum())).astype(np.float32)
        return probs, value


@dataclass
class _NeuralISNode:
    """Tree node storing PUCT prior P(s, a) alongside visit & value statistics."""

    move: Card | None = None
    parent: _NeuralISNode | None = None
    player_just_moved: int | None = None
    prior: float = 1.0
    children: dict[Card, _NeuralISNode] = field(default_factory=dict)
    is_expanded: bool = False
    visits: int = 0
    availability: int = 0
    total_reward: float = 0.0
    total_team_points: float = 0.0
    total_opp_points: float = 0.0
    total_wins: float = 0.0

    def puct_score(self, c_puct: float, fpu_value: float = 0.5) -> float:
        """AlphaZero PUCT score adapted for Single-Observer Information Sets."""
        q_val = (self.total_reward / self.visits) if self.visits > 0 else fpu_value
        u_val = (
            c_puct
            * self.prior
            * math.sqrt(max(1, self.availability))
            / (1 + self.visits)
        )
        return q_val + u_val


class NeuralISMCTSBot(Agent):
    """AlphaZero-style IS-MCTS guided by a trained PPO ActorCriticNet checkpoint.

    Combines:
    - Belief-filtered determinization (respecting voids, Busso, Volo, Striscio, Cricca).
    - Policy-guided PUCT selection using temperature-scaled neural priors.
    - Configurable leaf evaluation:
        * rollout_weight = 1.0 -> policy-guided PUCT + delayed expansion + rollout (default)
        * rollout_weight = 0.5 -> hybrid blend (neural value + heuristic rollout)
        * rollout_weight = 0.0 -> pure neural value head evaluation
    """

    def __init__(
        self,
        name: str = "AlphaZeroISMCTS",
        *,
        model: ActorCriticNet | None = None,
        checkpoint_path: str | Path | None = None,
        simulations: int = 120,
        trump_simulations: int = 24,
        c_puct: float = 1.25,
        prior_temp: float = 1.5,
        rollout_weight: float = 1.0,
        expand_after_visits: int = 3,
        seed: int | None = None,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        if not 0.0 <= rollout_weight <= 1.0:
            raise ValueError("rollout_weight must be in [0.0, 1.0].")

        self.simulations = max(1, simulations)
        self.trump_simulations = max(4, trump_simulations)
        self.c_puct = c_puct
        self.prior_temp = max(0.1, prior_temp)
        self.rollout_weight = rollout_weight
        self.expand_after_visits = max(1, expand_after_visits)
        self._rng = random.Random(seed)

        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = PPOBot.load_model(checkpoint_path)
        else:
            self.model = ActorCriticNet()

        self.model.eval()
        self._fast_net = _FastNumpyNet(self.model)

    def _evaluate_net(self, obs: Observation) -> tuple[np.ndarray, float]:
        """Run fast NumPy forward pass returning (action_probs [40], future_margin_norm)."""
        return self._fast_net.predict(
            encode_observation(obs), legal_action_mask(obs), self.prior_temp
        )

    def _estimate_leaf_points(
        self, sim: _SimState, v_future_norm: float
    ) -> tuple[float, float]:
        """Blend neural value estimate and optional heuristic rollout into (team0, team1) pts."""
        if self.rollout_weight >= 1.0:
            pts = sim.rollout_to_end()
            return float(pts[0]), float(pts[1])

        total_pts = 14.0 if sim.cricca_declared_by is not None else 11.0
        pid = sim.current_player
        t = team_of(pid)

        curr_margin_pts = (sim.hand_thirds[t] - sim.hand_thirds[1 - t]) / 3.0
        pred_margin = curr_margin_pts + v_future_norm * 11.0
        pred_margin = max(-total_pts, min(total_pts, pred_margin))

        nn_my_pts = 0.5 * (total_pts + pred_margin)
        nn_opp_pts = 0.5 * (total_pts - pred_margin)
        nn_pts = (nn_my_pts, nn_opp_pts) if t == 0 else (nn_opp_pts, nn_my_pts)

        if self.rollout_weight <= 0.0:
            return nn_pts

        roll_pts = sim.rollout_to_end()
        w = self.rollout_weight
        return (
            (1.0 - w) * nn_pts[0] + w * float(roll_pts[0]),
            (1.0 - w) * nn_pts[1] + w * float(roll_pts[1]),
        )

    def choose_trump(self, obs: Observation) -> Suit:
        """Select trump using paired Monte Carlo rollouts across all 4 suits."""
        me = obs.player_id
        my_team = obs.team_id
        unknown = [c for c in ALL_CARDS if c not in obs.hand]

        worlds: list[list[list[Card]]] = []
        for _ in range(self.trump_simulations):
            pool = list(unknown)
            self._rng.shuffle(pool)
            hands: list[list[Card]] = [[] for _ in range(4)]
            hands[me] = list(obs.hand)
            cursor = 0
            for p in range(4):
                if p != me:
                    hands[p] = pool[cursor : cursor + 10]
                    cursor += 10
            worlds.append(hands)

        best_suit = Suit.COINS
        best_margin = -1e9

        for suit in Suit:
            total_margin = 0.0
            for hands in worlds:
                sim_hands = [list(h) for h in hands]
                init_counts = [
                    {s: sum(1 for c in h if c.suit == s) for s in Suit}
                    for h in sim_hands
                ]
                hand_thirds = [0, 0]
                cricca_by: int | None = None
                for offset in range(4):
                    pid = (me + offset) % 4
                    if has_cricca(sim_hands[pid], suit):
                        cricca_by = pid
                        hand_thirds[team_of(pid)] += CRICCA_BONUS_THIRDS
                        break

                sim = _SimState(
                    hands=sim_hands,
                    initial_suit_counts=init_counts,
                    trump_caller=me,
                    trump_suit=suit,
                    cricca_declared_by=cricca_by,
                    completed_tricks=[],
                    current_trick=[],
                    current_trick_leader=me,
                    current_trick_signal=Signal.NONE,
                    played_cards=set(),
                    hand_thirds=hand_thirds,
                    pending_busso_return_suit=None,
                    match_scores=obs.match_scores,
                    target_score=obs.target_score,
                    hand_number=obs.hand_number,
                )
                pts = sim.rollout_to_end()
                total_margin += pts[my_team] - pts[1 - my_team]

            avg_margin = total_margin / len(worlds)
            if avg_margin > best_margin:
                best_margin = avg_margin
                best_suit = suit

        return best_suit

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return True

    def decide_callout(self, obs: Observation) -> bool:
        team = obs.team_id
        total = obs.match_scores[team] + thirds_to_points(obs.hand_thirds[team])
        return total >= obs.target_score

    def choose_card(self, obs: Observation) -> Card:
        if not obs.legal_cards:
            raise ValueError("NeuralISMCTSBot received empty legal_cards.")
        if len(obs.legal_cards) == 1:
            return obs.legal_cards[0]

        stats = self.analyze_moves(obs, simulations=self.simulations)
        best = max(stats, key=lambda s: (s.visits, s.expected_margin))
        return best.card

    def analyze_moves(
        self,
        obs: Observation,
        *,
        simulations: int | None = None,
        min_visits_per_move: int = 0,
    ) -> list[MoveStats]:
        """Run PUCT Neural IS-MCTS and return ranked MoveStats for each legal card."""
        if not obs.legal_cards:
            raise ValueError("Cannot analyze moves when obs.legal_cards is empty.")

        total_sims = simulations if simulations is not None else self.simulations
        root_legal = list(obs.legal_cards)
        constraints = infer_belief_constraints(obs)

        root = _NeuralISNode(is_expanded=True)
        if len(root_legal) == 1:
            only = root_legal[0]
            root.children[only] = _NeuralISNode(
                move=only, parent=root, player_just_moved=obs.player_id, prior=1.0
            )
        else:
            root_probs, _ = self._evaluate_net(obs)
            for card in root_legal:
                root.children[card] = _NeuralISNode(
                    move=card,
                    parent=root,
                    player_just_moved=obs.player_id,
                    prior=float(root_probs[card_to_index(card)]),
                )

        warmup_total = (
            min(total_sims, len(root_legal) * min_visits_per_move)
            if min_visits_per_move > 0
            else 0
        )

        for sim_idx in range(total_sims):
            sampled_hands = sample_consistent_hands(obs, constraints, self._rng)
            sim = _SimState.from_observation(obs, sampled_hands, constraints)

            # 1. Root move selection (PUCT guided by neural prior)
            for c in root_legal:
                root.children[c].availability += 1

            if sim_idx < warmup_total:
                root_move = root_legal[sim_idx % len(root_legal)]
            else:
                root_move = max(
                    root_legal,
                    key=lambda c: root.children[c].puct_score(self.c_puct),
                )
            sim.apply_move(root_move)
            node = root.children[root_move]

            # 2. Tree traversal (Selection) until reaching an unexpanded leaf or terminal state
            while not sim.is_terminal and node.is_expanded:
                pid = sim.current_player
                legal = sim.legal_moves(enforce_busso=True)

                if len(legal) == 1:
                    m = legal[0]
                    child = node.children.get(m)
                    if child is None:
                        child = _NeuralISNode(
                            move=m, parent=node, player_just_moved=pid, prior=1.0
                        )
                        node.children[m] = child
                    child.availability += 1
                    sim.apply_move(m)
                    node = child
                    continue

                missing = [m for m in legal if m not in node.children]
                if missing:
                    fallback_prior = 1.0 / len(legal)
                    for m in missing:
                        node.children[m] = _NeuralISNode(
                            move=m,
                            parent=node,
                            player_just_moved=pid,
                            prior=fallback_prior,
                        )

                for m in legal:
                    node.children[m].availability += 1

                chosen = max(
                    legal,
                    key=lambda m: node.children[m].puct_score(self.c_puct),
                )
                sim.apply_move(chosen)
                node = node.children[chosen]

            # 3. Expansion & Evaluation at leaf
            if sim.is_terminal:
                final_points = (
                    float(thirds_to_points(sim.hand_thirds[0])),
                    float(thirds_to_points(sim.hand_thirds[1])),
                )
            else:
                pid = sim.current_player
                legal = sim.legal_moves(enforce_busso=True)

                if len(legal) == 1:
                    m = legal[0]
                    node.children[m] = _NeuralISNode(
                        move=m, parent=node, player_just_moved=pid, prior=1.0
                    )
                    node.is_expanded = True
                    if self.rollout_weight >= 1.0:
                        roll = sim.rollout_to_end()
                        final_points = (float(roll[0]), float(roll[1]))
                    else:
                        _, v_future = self._evaluate_net(sim.to_observation(pid, legal))
                        final_points = self._estimate_leaf_points(sim, v_future)
                elif (
                    self.rollout_weight >= 1.0
                    and (node.visits + 1) < self.expand_after_visits
                ):
                    roll = sim.rollout_to_end()
                    final_points = (float(roll[0]), float(roll[1]))
                else:
                    sim_obs = sim.to_observation(pid, legal)
                    probs, v_future = self._evaluate_net(sim_obs)
                    for m in legal:
                        node.children[m] = _NeuralISNode(
                            move=m,
                            parent=node,
                            player_just_moved=pid,
                            prior=float(probs[card_to_index(m)]),
                        )
                    node.is_expanded = True
                    final_points = self._estimate_leaf_points(sim, v_future)

            # 4. Backpropagation (perspective-aware per team)
            curr: _NeuralISNode | None = node
            while curr is not None:
                curr.visits += 1
                if curr.player_just_moved is not None:
                    t = team_of(curr.player_just_moved)
                    my_pts = final_points[t]
                    opp_pts = final_points[1 - t]
                    margin = my_pts - opp_pts
                    normalized = max(0.0, min(1.0, (margin + 14.0) / 28.0))
                    curr.total_reward += normalized
                    curr.total_team_points += my_pts
                    curr.total_opp_points += opp_pts
                    if my_pts > opp_pts + 1e-6:
                        curr.total_wins += 1.0
                    elif abs(my_pts - opp_pts) <= 1e-6:
                        curr.total_wins += 0.5
                curr = curr.parent

        results: list[MoveStats] = []
        for card in root_legal:
            child = root.children[card]
            if child.visits > 0:
                ev_pts = child.total_team_points / child.visits
                ev_opp = child.total_opp_points / child.visits
                ev_margin = ev_pts - ev_opp
                win_prob = child.total_wins / child.visits
            else:
                ev_pts = 0.0
                ev_opp = 0.0
                ev_margin = -99.0
                win_prob = 0.0

            results.append(
                MoveStats(
                    card=card,
                    visits=child.visits,
                    expected_points=ev_pts,
                    expected_opp_points=ev_opp,
                    expected_margin=ev_margin,
                    win_probability=win_prob,
                )
            )

        results.sort(key=lambda s: (s.visits > 0, s.expected_margin, s.visits), reverse=True)
        return results