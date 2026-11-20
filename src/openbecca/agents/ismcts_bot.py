"""Information-Set Monte Carlo Tree Search (SO-ISMCTS) agent with belief filtering."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from openbecca.agents.base import Agent
from openbecca.agents.heuristic_bot import HeuristicBot
from openbecca.cards import ALL_CARDS, Card, Rank, Suit
from openbecca.rules import (
    CRICCA_BONUS_THIRDS,
    TrickPlay,
    determine_trick_winner,
    get_legal_cards,
    has_cricca,
    partner_of,
    team_of,
    thirds_to_points,
    trick_points_thirds,
)
from openbecca.signals import Signal, determine_lead_signal
from openbecca.state import Observation, TrickRecord

# Stateless fast rollout policy shared across simulations.
_ROLLOUT_POLICY = HeuristicBot(name="RolloutHeuristic")


# ----------------------------------------------------------------------
# Belief inference & consistent hand sampling (strictly imperfect info)
# ----------------------------------------------------------------------
@dataclass
class BeliefConstraints:
    """Publicly inferred constraints on hidden hands of the other 3 players."""

    target_sizes: dict[int, int]
    voids: dict[int, set[Suit]]
    must_hold: dict[int, set[Card]]
    forbidden_cards: dict[int, set[Card]]
    min_suit_counts: dict[int, dict[Suit, int]]
    max_suit_counts: dict[int, dict[Suit, int]]
    played_by_player_suit: dict[int, dict[Suit, int]]


def infer_belief_constraints(obs: Observation) -> BeliefConstraints:
    """Extract all mathematically guaranteed constraints from public table history."""
    me = obs.player_id
    others = [p for p in range(4) if p != me]

    current_trick_players = {play.player for play in obs.current_trick}
    # Every player held the same number of cards at the start of the current trick.
    cards_at_trick_start = len(obs.hand) + (1 if me in current_trick_players else 0)
    target_sizes = {
        p: cards_at_trick_start - (1 if p in current_trick_players else 0)
        for p in others
    }

    voids: dict[int, set[Suit]] = {p: set() for p in others}
    must_hold: dict[int, set[Card]] = {p: set() for p in others}
    forbidden_cards: dict[int, set[Card]] = {p: set() for p in others}
    min_suit_counts: dict[int, dict[Suit, int]] = {
        p: {s: 0 for s in Suit} for p in others
    }
    max_suit_counts: dict[int, dict[Suit, int]] = {
        p: {s: 10 for s in Suit} for p in others
    }
    played_by_player_suit: dict[int, dict[Suit, int]] = {
        p: {s: 0 for s in Suit} for p in range(4)
    }

    unknown_cards = set(ALL_CARDS) - obs.played_cards - set(obs.hand)

    # Count all cards played so far per player and suit.
    all_tricks_plays: list[tuple[int, Signal, tuple[TrickPlay, ...]]] = [
        (t.leader, t.signal, t.plays) for t in obs.completed_tricks
    ]
    if obs.current_trick:
        all_tricks_plays.append(
            (obs.current_trick_leader, obs.current_trick_signal, obs.current_trick)
        )

    for _, _, plays in all_tricks_plays:
        for play in plays:
            played_by_player_suit[play.player][play.card.suit] += 1

    # 1. Declared Cricca guarantees any unplayed trump Asso/Due/Tre is in declarer's hand.
    if (
        obs.trump_suit is not None
        and obs.cricca_declared_by is not None
        and obs.cricca_declared_by != me
    ):
        declarer = obs.cricca_declared_by
        for rank in (Rank.ACE, Rank.TWO, Rank.THREE):
            card = Card(obs.trump_suit, rank)
            if card in unknown_cards:
                must_hold[declarer].add(card)

    # 2. Infer voids and strict v1 signal implications trick by trick.
    for trick_idx, (leader, signal, plays) in enumerate(all_tricks_plays):
        if not plays:
            continue
        led_suit = plays[0].card.suit

        # Following off-suit proves a void in led_suit.
        for play in plays[1:]:
            if play.player != me and play.card.suit != led_suit:
                voids[play.player].add(led_suit)

        if leader == me:
            continue

        lead_card = plays[0].card

        if signal == Signal.VOLO:
            voids[leader].add(led_suit)

        elif signal == Signal.STRISCIO:
            # Leader started with >= 4 cards of led_suit.
            played_total = played_by_player_suit[leader][led_suit]
            min_rem = max(0, 4 - played_total)
            min_suit_counts[leader][led_suit] = max(
                min_suit_counts[leader][led_suit], min_rem
            )

        elif signal == Signal.NONE:
            # Silence in strict v1 guarantees initial count <= 3 for led_suit.
            played_total = played_by_player_suit[leader][led_suit]
            max_rem = max(0, 3 - played_total)
            max_suit_counts[leader][led_suit] = min(
                max_suit_counts[leader][led_suit], max_rem
            )

        elif signal == Signal.BUSSO:
            if lead_card.rank == Rank.THREE:
                # Case A: held Asso (and not Due) when leading Tre.
                ace = Card(led_suit, Rank.ACE)
                two = Card(led_suit, Rank.TWO)
                if ace in unknown_cards:
                    must_hold[leader].add(ace)
                forbidden_cards[leader].add(two)
            elif lead_card.rank.power < Rank.ACE.power:
                # Case B: held Due (not Tre) and >= 3 cards of led_suit before playing.
                two = Card(led_suit, Rank.TWO)
                three = Card(led_suit, Rank.THREE)
                if two in unknown_cards:
                    must_hold[leader].add(two)
                forbidden_cards[leader].add(three)

                played_from_here = 0
                for later_idx in range(trick_idx, len(all_tricks_plays)):
                    for later_play in all_tricks_plays[later_idx][2]:
                        if later_play.player == leader and later_play.card.suit == led_suit:
                            played_from_here += 1
                min_rem = max(0, 3 - played_from_here)
                min_suit_counts[leader][led_suit] = max(
                    min_suit_counts[leader][led_suit], min_rem
                )

    # Reconcile voids with min/max suit bounds.
    for p in others:
        for s in voids[p]:
            min_suit_counts[p][s] = 0
            max_suit_counts[p][s] = 0

    return BeliefConstraints(
        target_sizes=target_sizes,
        voids=voids,
        must_hold=must_hold,
        forbidden_cards=forbidden_cards,
        min_suit_counts=min_suit_counts,
        max_suit_counts=max_suit_counts,
        played_by_player_suit=played_by_player_suit,
    )


def sample_consistent_hands(
    obs: Observation,
    constraints: BeliefConstraints,
    rng: random.Random,
) -> list[list[Card]]:
    """Sample a full 4-player hand assignment consistent with public beliefs.

    Falls back gracefully (soft constraints -> voids only -> uniform) so custom
    or noisy scenarios never crash search.
    """
    me = obs.player_id
    others = [p for p in range(4) if p != me]
    unknown_cards = [c for c in ALL_CARDS if c not in obs.played_cards and c not in obs.hand]

    expected_unknown = sum(constraints.target_sizes[p] for p in others)
    if len(unknown_cards) != expected_unknown:
        raise ValueError(
            f"Inconsistent observation: {len(unknown_cards)} unknown cards vs "
            f"{expected_unknown} expected slots across other players."
        )

    # Tier 1: full constraints (voids + must_hold + forbidden + min/max suit counts)
    for _ in range(40):
        deal = _try_constrained_sample(
            obs, others, unknown_cards, constraints, rng, use_count_bounds=True
        )
        if deal is not None:
            return deal

    # Tier 2: hard constraints only (voids + must_hold + forbidden)
    for _ in range(30):
        deal = _try_constrained_sample(
            obs, others, unknown_cards, constraints, rng, use_count_bounds=False
        )
        if deal is not None:
            return deal

    # Tier 3: uniform random fallback
    pool = list(unknown_cards)
    rng.shuffle(pool)
    hands: list[list[Card]] = [[] for _ in range(4)]
    hands[me] = list(obs.hand)
    cursor = 0
    for p in others:
        size = constraints.target_sizes[p]
        hands[p] = pool[cursor : cursor + size]
        cursor += size
    return hands


def _try_constrained_sample(
    obs: Observation,
    others: list[int],
    unknown_cards: list[Card],
    constraints: BeliefConstraints,
    rng: random.Random,
    *,
    use_count_bounds: bool,
) -> list[list[Card]] | None:
    me = obs.player_id
    hands: list[list[Card]] = [[] for _ in range(4)]
    hands[me] = list(obs.hand)
    cap = dict(constraints.target_sizes)
    suit_counts = {p: Counter() for p in others}
    assigned: set[Card] = set()

    # Step 1: place mandatory cards (Cricca pieces, Busso Asso/Due)
    for p in others:
        for card in constraints.must_hold[p]:
            if card in assigned or cap[p] <= 0 or card.suit in constraints.voids[p]:
                return None
            hands[p].append(card)
            assigned.add(card)
            cap[p] -= 1
            suit_counts[p][card.suit] += 1

    remaining = [c for c in unknown_cards if c not in assigned]

    # Step 2: satisfy minimum suit count requirements (Striscio, Busso Case B)
    if use_count_bounds:
        reqs: list[tuple[int, Suit, int]] = []
        for p in others:
            for suit in Suit:
                need = constraints.min_suit_counts[p][suit] - suit_counts[p][suit]
                if need > 0:
                    reqs.append((p, suit, need))
        rng.shuffle(reqs)

        for p, suit, need in reqs:
            if cap[p] < need:
                return None
            candidates = [
                c
                for c in remaining
                if c.suit == suit and c not in constraints.forbidden_cards[p]
            ]
            if len(candidates) < need:
                return None
            chosen = rng.sample(candidates, need)
            for card in chosen:
                hands[p].append(card)
                remaining.remove(card)
                cap[p] -= 1
                suit_counts[p][suit] += 1

    # Step 3: assign remaining cards in Most-Restricted-First (MRV) order
    def _eligible_players(card: Card) -> list[int]:
        elig: list[int] = []
        for p in others:
            if cap[p] <= 0:
                continue
            if card.suit in constraints.voids[p]:
                continue
            if card in constraints.forbidden_cards[p]:
                continue
            if (
                use_count_bounds
                and suit_counts[p][card.suit] >= constraints.max_suit_counts[p][card.suit]
            ):
                continue
            elig.append(p)
        return elig

    # Sort by fewest eligible players first (with random tie-breaking)
    tie_breaks = {c: rng.random() for c in remaining}
    remaining.sort(key=lambda c: (len(_eligible_players(c)), tie_breaks[c]))

    for card in remaining:
        elig = _eligible_players(card)
        if not elig:
            return None
        # Weight selection by remaining empty slots to preserve uniform-like deal odds.
        weights = [cap[p] for p in elig]
        chosen_p = rng.choices(elig, weights=weights, k=1)[0]
        hands[chosen_p].append(card)
        cap[chosen_p] -= 1
        suit_counts[chosen_p][card.suit] += 1

    return hands


# ----------------------------------------------------------------------
# Lightweight fast simulation state for IS-MCTS tree & rollouts
# ----------------------------------------------------------------------
class _SimState:
    """Mutable full-information state cloned from a determinized world."""

    __slots__ = (
        "hands",
        "initial_suit_counts",
        "trump_caller",
        "trump_suit",
        "cricca_declared_by",
        "completed_tricks",
        "current_trick",
        "current_trick_leader",
        "current_trick_signal",
        "played_cards",
        "hand_thirds",
        "pending_busso_return_suit",
        "match_scores",
        "target_score",
        "hand_number",
    )

    def __init__(
        self,
        hands: list[list[Card]],
        initial_suit_counts: list[dict[Suit, int]],
        trump_caller: int,
        trump_suit: Suit,
        cricca_declared_by: int | None,
        completed_tricks: list[TrickRecord],
        current_trick: list[TrickPlay],
        current_trick_leader: int,
        current_trick_signal: Signal,
        played_cards: set[Card],
        hand_thirds: list[int],
        pending_busso_return_suit: Suit | None,
        match_scores: tuple[int, int],
        target_score: int,
        hand_number: int,
    ) -> None:
        self.hands = hands
        self.initial_suit_counts = initial_suit_counts
        self.trump_caller = trump_caller
        self.trump_suit = trump_suit
        self.cricca_declared_by = cricca_declared_by
        self.completed_tricks = completed_tricks
        self.current_trick = current_trick
        self.current_trick_leader = current_trick_leader
        self.current_trick_signal = current_trick_signal
        self.played_cards = played_cards
        self.hand_thirds = hand_thirds
        self.pending_busso_return_suit = pending_busso_return_suit
        self.match_scores = match_scores
        self.target_score = target_score
        self.hand_number = hand_number

    @classmethod
    def from_observation(
        cls,
        obs: Observation,
        sampled_hands: list[list[Card]],
        constraints: BeliefConstraints,
    ) -> _SimState:
        assert obs.trump_suit is not None
        initial_counts: list[dict[Suit, int]] = []
        for p in range(4):
            if p == obs.player_id:
                initial_counts.append(dict(obs.initial_suit_counts))
            else:
                rem = Counter(c.suit for c in sampled_hands[p])
                played = constraints.played_by_player_suit[p]
                initial_counts.append({s: rem.get(s, 0) + played[s] for s in Suit})

        return cls(
            hands=[list(h) for h in sampled_hands],
            initial_suit_counts=initial_counts,
            trump_caller=obs.trump_caller,
            trump_suit=obs.trump_suit,
            cricca_declared_by=obs.cricca_declared_by,
            completed_tricks=list(obs.completed_tricks),
            current_trick=list(obs.current_trick),
            current_trick_leader=obs.current_trick_leader,
            current_trick_signal=obs.current_trick_signal,
            played_cards=set(obs.played_cards),
            hand_thirds=[obs.hand_thirds[0], obs.hand_thirds[1]],
            pending_busso_return_suit=obs.pending_busso_return_suit,
            match_scores=obs.match_scores,
            target_score=obs.target_score,
            hand_number=obs.hand_number,
        )

    @property
    def is_terminal(self) -> bool:
        return not any(self.hands)

    @property
    def current_player(self) -> int:
        return (self.current_trick_leader + len(self.current_trick)) % 4

    @property
    def trick_number(self) -> int:
        # Works both for normal games and custom setups where completed_tricks is empty.
        cards_remaining_total = sum(len(h) for h in self.hands) + len(self.current_trick)
        tricks_remaining = cards_remaining_total // 4
        return 11 - tricks_remaining

    def legal_moves(self, *, enforce_busso: bool = True) -> list[Card]:
        pid = self.current_player
        is_lead = len(self.current_trick) == 0
        is_cricca_lead = (
            is_lead
            and self.trick_number == 1
            and self.cricca_declared_by == pid
            and pid == self.trump_caller
        )
        return get_legal_cards(
            hand=self.hands[pid],
            current_trick=self.current_trick,
            trump_suit=self.trump_suit,
            is_first_trick_cricca_lead=is_cricca_lead,
            enforce_busso_constraints=enforce_busso,
            current_trick_signal=self.current_trick_signal,
            current_trick_leader=self.current_trick_leader,
            player_id=pid,
            busso_return_suit=self.pending_busso_return_suit if is_lead else None,
        )

    def apply_move(self, card: Card) -> None:
        pid = self.current_player
        if not self.current_trick:
            self.current_trick_signal = determine_lead_signal(
                played_card=card,
                hand_before_play=self.hands[pid],
                initial_suit_count=self.initial_suit_counts[pid][card.suit],
                played_cards_history=self.played_cards,
            )

        self.hands[pid].remove(card)
        self.current_trick.append(TrickPlay(player=pid, card=card))
        self.played_cards.add(card)

        if len(self.current_trick) == 4:
            is_last = not any(self.hands)
            winner = determine_trick_winner(self.current_trick, self.trump_suit)
            pts_thirds = trick_points_thirds(self.current_trick, is_last_trick=is_last)
            self.hand_thirds[team_of(winner)] += pts_thirds

            led_suit = self.current_trick[0].card.suit
            record = TrickRecord(
                trick_number=len(self.completed_tricks) + 1,
                leader=self.current_trick_leader,
                signal=self.current_trick_signal,
                plays=tuple(self.current_trick),
                winner=winner,
                points_thirds=pts_thirds,
            )
            self.completed_tricks.append(record)

            if self.current_trick_signal == Signal.BUSSO and winner == partner_of(
                self.current_trick_leader
            ):
                self.pending_busso_return_suit = led_suit
            else:
                self.pending_busso_return_suit = None

            self.current_trick = []
            self.current_trick_signal = Signal.NONE
            self.current_trick_leader = winner

    def to_observation(self, player_id: int, legal: list[Card]) -> Observation:
        busso_suit = (
            self.pending_busso_return_suit
            if not self.current_trick and player_id == self.current_trick_leader
            else None
        )
        return Observation(
            player_id=player_id,
            team_id=team_of(player_id),
            hand=tuple(self.hands[player_id]),
            initial_suit_counts=self.initial_suit_counts[player_id],
            hand_number=self.hand_number,
            trump_caller=self.trump_caller,
            trump_suit=self.trump_suit,
            cricca_declared_by=self.cricca_declared_by,
            completed_tricks=tuple(self.completed_tricks),
            current_trick=tuple(self.current_trick),
            current_trick_leader=self.current_trick_leader,
            current_trick_signal=self.current_trick_signal,
            played_cards=frozenset(self.played_cards),
            match_scores=self.match_scores,
            hand_thirds=(self.hand_thirds[0], self.hand_thirds[1]),
            target_score=self.target_score,
            pending_busso_return_suit=busso_suit,
            legal_cards=tuple(legal),
        )

    def rollout_to_end(self) -> tuple[int, int]:
        """Finish the hand rapidly using the heuristic rollout policy."""
        while not self.is_terminal:
            pid = self.current_player
            legal = self.legal_moves(enforce_busso=True)
            if len(legal) == 1:
                move = legal[0]
            else:
                obs = self.to_observation(pid, legal)
                move = _ROLLOUT_POLICY.choose_card(obs)
            self.apply_move(move)

        return (
            thirds_to_points(self.hand_thirds[0]),
            thirds_to_points(self.hand_thirds[1]),
        )


# ----------------------------------------------------------------------
# SO-ISMCTS Tree Node & Agent
# ----------------------------------------------------------------------
@dataclass
class _ISNode:
    """Single node in the Single-Observer Information Set MCTS tree."""

    move: Card | None = None
    parent: _ISNode | None = None
    player_just_moved: int | None = None
    children: dict[Card, _ISNode] = field(default_factory=dict)
    visits: int = 0
    availability: int = 0
    total_reward: float = 0.0
    total_team_points: float = 0.0
    total_opp_points: float = 0.0
    total_wins: float = 0.0

    def ucb_score(self, exploration: float) -> float:
        mean_val = self.total_reward / self.visits
        return mean_val + exploration * math.sqrt(
            math.log(max(1, self.availability)) / self.visits
        )


@dataclass(frozen=True)
class MoveStats:
    """Raw search statistics for one legal card at the root."""

    card: Card
    visits: int
    expected_points: float
    expected_opp_points: float
    expected_margin: float
    win_probability: float


class ISMCTSBot(Agent):
    """Search-based AI using Single-Observer IS-MCTS with belief filtering.

    Predicts future moves via tree search + heuristic rollouts across sampled
    worlds consistent with voids, Busso, Volo, Striscio, and declared Cricca.
    """

    def __init__(
        self,
        name: str = "ISMCTSBot",
        *,
        simulations: int = 200,
        trump_simulations: int = 32,
        exploration: float = 0.7,
        seed: int | None = None,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        self.simulations = max(1, simulations)
        self.trump_simulations = max(4, trump_simulations)
        self.exploration = exploration
        self._rng = random.Random(seed)

    def choose_trump(self, obs: Observation) -> Suit:
        """Select trump via paired Monte Carlo rollouts across all 4 suits."""
        me = obs.player_id
        my_team = obs.team_id
        unknown = [c for c in ALL_CARDS if c not in obs.hand]

        # Sample paired worlds so all 4 suits are compared on identical deals.
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
            raise ValueError("ISMCTSBot received empty legal_cards.")
        if len(obs.legal_cards) == 1:
            return obs.legal_cards[0]

        stats = self.analyze_moves(obs, simulations=self.simulations)
        # Prefer highest expected margin, breaking ties by visits.
        best = max(stats, key=lambda s: (s.expected_margin, s.visits))
        return best.card

    def analyze_moves(
        self,
        obs: Observation,
        *,
        simulations: int | None = None,
        min_visits_per_move: int = 0,
    ) -> list[MoveStats]:
        """Run SO-ISMCTS from obs and return ranked MoveStats for every legal card."""
        if not obs.legal_cards:
            raise ValueError("Cannot analyze moves when obs.legal_cards is empty.")

        total_sims = simulations if simulations is not None else self.simulations
        root_legal = list(obs.legal_cards)
        constraints = infer_belief_constraints(obs)
        root = _ISNode()

        # Pre-create root children for every legal move.
        for card in root_legal:
            root.children[card] = _ISNode(
                move=card, parent=root, player_just_moved=obs.player_id
            )

        # Ensure minimum warm-start visits per root card (useful for Stockfish UI tables).
        warmup_total = min(total_sims, len(root_legal) * max(1, min_visits_per_move))

        for sim_idx in range(total_sims):
            sampled_hands = sample_consistent_hands(obs, constraints, self._rng)
            sim = _SimState.from_observation(obs, sampled_hands, constraints)

            # 1. Root move selection (round-robin warmup, then UCB1)
            if sim_idx < warmup_total:
                root_move = root_legal[sim_idx % len(root_legal)]
                for c in root_legal:
                    root.children[c].availability += 1
                node = root.children[root_move]
            else:
                for c in root_legal:
                    root.children[c].availability += 1
                untried_root = [c for c in root_legal if root.children[c].visits == 0]
                if untried_root:
                    root_move = self._rng.choice(untried_root)
                else:
                    root_move = max(
                        root_legal,
                        key=lambda c: root.children[c].ucb_score(self.exploration),
                    )
                node = root.children[root_move]

            sim.apply_move(root_move)

            # 2. Selection & Expansion deeper in the SO-ISMCTS tree
            while not sim.is_terminal:
                legal = sim.legal_moves(enforce_busso=True)
                untried = [m for m in legal if m not in node.children]

                if untried:
                    chosen = self._rng.choice(untried)
                    mover = sim.current_player
                    for m in legal:
                        if m in node.children:
                            node.children[m].availability += 1
                    child = _ISNode(move=chosen, parent=node, player_just_moved=mover)
                    child.availability = 1
                    node.children[chosen] = child
                    sim.apply_move(chosen)
                    node = child
                    break

                # All legal moves at this state exist in tree -> UCB1 selection
                for m in legal:
                    node.children[m].availability += 1
                chosen = max(
                    legal, key=lambda m: node.children[m].ucb_score(self.exploration)
                )
                sim.apply_move(chosen)
                node = node.children[chosen]

            # 3. Heuristic Rollout to Trick 10
            final_points = sim.rollout_to_end()

            # 4. Backpropagation (perspective-aware per team)
            curr: _ISNode | None = node
            while curr is not None:
                curr.visits += 1
                if curr.player_just_moved is not None:
                    t = team_of(curr.player_just_moved)
                    my_pts = final_points[t]
                    opp_pts = final_points[1 - t]
                    margin = my_pts - opp_pts
                    # Normalize margin from [-14, +14] to [0, 1] for UCB1 stability.
                    normalized = (margin + 14.0) / 28.0
                    curr.total_reward += normalized
                    curr.total_team_points += my_pts
                    curr.total_opp_points += opp_pts
                    if my_pts > opp_pts:
                        curr.total_wins += 1.0
                    elif my_pts == opp_pts:
                        curr.total_wins += 0.5
                curr = curr.parent

        results: list[MoveStats] = []
        for card in root_legal:
            child = root.children[card]
            v = max(1, child.visits)
            ev_pts = child.total_team_points / v
            ev_opp = child.total_opp_points / v
            results.append(
                MoveStats(
                    card=card,
                    visits=child.visits,
                    expected_points=ev_pts,
                    expected_opp_points=ev_opp,
                    expected_margin=ev_pts - ev_opp,
                    win_probability=child.total_wins / v,
                )
            )

        results.sort(key=lambda s: (s.expected_margin, s.visits), reverse=True)
        return results