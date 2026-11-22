"""HeuristicBotV2: formal 12-tier trump selection & Trick-1 opening specification.

Defaults incorporate empirical Stockfish (ISMCTSBot) discoveries from Decision Lab:
- EXP-05 (+2.600 margin): length override threshold lowered from 6 to 5 cards.
- EXP-01 (+1.860 margin): never call a bare 2-card 3+2 suit over any 3+ card suit.
- EXP-06 (+4.8% win rate across 3 tests): lead lowest trump (Striscio, saving Tre)
  on >=4 trumps holding 3+A instead of burning Tre on Trick 1.
- EXP-02/04/08/09/10/11: confirmed human instincts retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openbecca.agents.base import Agent
from openbecca.agents.heuristic_bot import HeuristicBot, HeuristicConfig
from openbecca.cards import Card, Rank, Suit
from openbecca.rules import has_cricca, partner_of, team_of, thirds_to_points
from openbecca.signals import Signal
from openbecca.state import Observation


@dataclass(frozen=True)
class HeuristicV2Config:
    """Configurable switches for HeuristicBotV2 and Decision Lab A/B experiments."""

    # EXP-05 (PROMOTED): >=5 cards overrides piece hierarchy (legacy: 6)
    length_override_threshold: int = 5

    # EXP-01 (PROMOTED): never call bare 3+2 over 3-card suits (legacy: True)
    allow_bare_three_two_trump: bool = False

    # EXP-02 (CONFIRMED): 4+ non-piece suit beats bare 3+2 (relevant if bare 3+2 enabled)
    four_non_piece_beats_bare_three_two: bool = True

    # EXP-03: "high_card" (lexicographic card power) vs "total_points" (sum of points_thirds)
    quality_metric: str = "high_card"

    # EXP-04 (CONFIRMED): at Tier 6 (4+ cards w/ >=1 piece), length beats piece rank
    prefer_length_at_tier6: bool = True

    # EXP-06 (PROMOTED): >=4 trumps with 3+A lead choice:
    # "low_trump" (Striscio, preserves H04), "three_busso" (Tre Busso A), "side_suit" (Ladder A)
    three_ace_four_plus_lead: str = "low_trump"

    # EXP-07: on 3-card trump with Due (2+A+x or 2+x+y), lead side suit (False) vs low trump Busso B (True)
    lead_trump_on_three_card_due: bool = False

    # EXP-08 (CONFIRMED): fallback for 3-card 3+A+x when no <=2-card side suit exists
    # "shortest_side", "trump_three", "low_trump", "longest_side"
    three_ace_three_card_fallback: str = "shortest_side"

    # EXP-09 (CONFIRMED): avoid leading Tre from bare 3+A off-suit on Trick 1
    avoid_bare_three_ace_lead: bool = True

    # EXP-10 (CONFIRMED): exhaust enemy trumps before cashing a 2nd Cricca on Trick 2
    exhaust_trumps_before_second_cricca: bool = True

    # EXP-11 (CONFIRMED via ISMCTS): Ladder B threshold for non-piece trumps (6 vs 4)
    non_piece_trump_ladder_b_threshold: int = 6

    # EXP-12: tie-break on Trick 1 for Due Busso (>=3) and Asso (>=3) side suits
    # "longer_suit" vs "lowest_card"
    side_suit_tiebreak: str = "longer_suit"

    # Exact 10-trump counting across all trump-drawing continuations
    use_exact_trump_count: bool = True

    # Track played cards per suit when selecting among safe non-trump Threes (Tricks 2+)
    use_suit_card_counting: bool = True


# Original human baseline before Stockfish promotions (useful for evolution benchmarks)
PRE_STOCKFISH_V2_CONFIG = HeuristicV2Config(
    length_override_threshold=6,
    allow_bare_three_two_trump=True,
    three_ace_four_plus_lead="three_busso",
)


def _suit_cards(cards: Sequence[Card], suit: Suit) -> list[Card]:
    return [c for c in cards if c.suit == suit]


def _lowest_card(cards: Sequence[Card]) -> Card:
    return min(cards, key=lambda c: (c.power, c.suit.order))


def _infer_known_voids(obs: Observation) -> dict[int, set[Suit]]:
    """Infer strictly proven suit voids from completed tricks and current trick."""
    voids: dict[int, set[Suit]] = {p: set() for p in range(4)}

    for trick in obs.completed_tricks:
        led = trick.led_suit
        if trick.signal == Signal.VOLO:
            voids[trick.leader].add(led)
        for play in trick.plays[1:]:
            if play.card.suit != led:
                voids[play.player].add(led)

    if obs.current_trick:
        led = obs.current_trick[0].card.suit
        if obs.current_trick_signal == Signal.VOLO:
            voids[obs.current_trick_leader].add(led)
        for play in obs.current_trick[1:]:
            if play.card.suit != led:
                voids[play.player].add(led)

    return voids


class HeuristicBotV2(Agent):
    """Formal Ravenna strategy agent combining V2 trump/opening logic with V1 midgame."""

    def __init__(
        self,
        name: str = "HeuristicBotV2",
        config: HeuristicV2Config | None = None,
        v1_config: HeuristicConfig | None = None,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        self.config = config or HeuristicV2Config()
        self._fallback_bot = HeuristicBot(
            name=f"{name}-v1",
            config=v1_config or HeuristicConfig(),
            enforce_busso_constraints=enforce_busso_constraints,
        )

    # ------------------------------------------------------------------
    # Quality & Trump Selection (12-tier cascade)
    # ------------------------------------------------------------------
    def _quality_key(self, cards: Sequence[Card]) -> tuple[object, ...]:
        powers_desc = tuple(sorted((c.power for c in cards), reverse=True))
        if self.config.quality_metric == "total_points":
            return (sum(c.points_thirds for c in cards), powers_desc)
        return powers_desc

    def _pick_best_suit(
        self,
        candidates: list[Suit],
        by_suit: dict[Suit, list[Card]],
        *,
        tier6_mode: bool = False,
    ) -> Suit:
        if tier6_mode and not self.config.prefer_length_at_tier6:
            def key_fn(s: Suit) -> tuple[object, ...]:
                cards = by_suit[s]
                max_piece = max(
                    (c.power for c in cards if c.rank.is_cricca_piece), default=0
                )
                return (max_piece, len(cards), self._quality_key(cards), -s.order)

            return max(candidates, key=key_fn)

        return max(
            candidates,
            key=lambda s: (len(by_suit[s]), self._quality_key(by_suit[s]), -s.order),
        )

    def choose_trump(self, obs: Observation) -> Suit:
        by_suit: dict[Suit, list[Card]] = {s: _suit_cards(obs.hand, s) for s in Suit}
        ranks_by_suit: dict[Suit, set[Rank]] = {
            s: {c.rank for c in cards} for s, cards in by_suit.items()
        }

        # Tier 1: Cricca (3 + 2 + A)
        t1 = [s for s in Suit if has_cricca(by_suit[s], s)]
        if t1:
            return self._pick_best_suit(t1, by_suit)

        # Tier 2: Long-suit override (>= 5 cards by default; >= 6 in legacy V2)
        t2 = [s for s in Suit if len(by_suit[s]) >= self.config.length_override_threshold]
        if t2:
            return self._pick_best_suit(t2, by_suit)

        # Tier 3: 3 + 2 with >= 3 cards
        t3 = [
            s
            for s in Suit
            if len(by_suit[s]) >= 3
            and Rank.THREE in ranks_by_suit[s]
            and Rank.TWO in ranks_by_suit[s]
        ]
        if t3:
            return self._pick_best_suit(t3, by_suit)

        # Tier 4: 3 + A with >= 3 cards
        t4 = [
            s
            for s in Suit
            if len(by_suit[s]) >= 3
            and Rank.THREE in ranks_by_suit[s]
            and Rank.ACE in ranks_by_suit[s]
        ]
        if t4:
            return self._pick_best_suit(t4, by_suit)

        # Tier 5: 2 + A with >= 3 cards
        t5 = [
            s
            for s in Suit
            if len(by_suit[s]) >= 3
            and Rank.TWO in ranks_by_suit[s]
            and Rank.ACE in ranks_by_suit[s]
        ]
        if t5:
            return self._pick_best_suit(t5, by_suit)

        # Tier 6: >= 4 cards with at least 1 Cricca piece
        t6 = [
            s
            for s in Suit
            if len(by_suit[s]) >= 4
            and any(r.is_cricca_piece for r in ranks_by_suit[s])
        ]
        if t6:
            return self._pick_best_suit(t6, by_suit, tier6_mode=True)

        # Tier 7: 3 + 2 extra cards (3 cards, solo Three)
        t7 = [s for s in Suit if len(by_suit[s]) >= 3 and Rank.THREE in ranks_by_suit[s]]
        if t7:
            return self._pick_best_suit(t7, by_suit)

        # Tier 8: 2 + 2 extra cards (3 cards, solo Two)
        t8 = [s for s in Suit if len(by_suit[s]) >= 3 and Rank.TWO in ranks_by_suit[s]]
        if t8:
            return self._pick_best_suit(t8, by_suit)

        # Tier 9: A + 2 extra cards (3 cards, solo Ace)
        t9 = [s for s in Suit if len(by_suit[s]) >= 3 and Rank.ACE in ranks_by_suit[s]]
        if t9:
            return self._pick_best_suit(t9, by_suit)

        # Tiers 10-11: >= 4 non-piece cards vs optional bare 3 + 2
        t_four_non_piece = [s for s in Suit if len(by_suit[s]) >= 4]
        t_bare_three_two = [
            s
            for s in Suit
            if len(by_suit[s]) == 2
            and Rank.THREE in ranks_by_suit[s]
            and Rank.TWO in ranks_by_suit[s]
        ]

        if self.config.four_non_piece_beats_bare_three_two:
            if t_four_non_piece:
                return self._pick_best_suit(t_four_non_piece, by_suit)
            if self.config.allow_bare_three_two_trump and t_bare_three_two:
                return self._pick_best_suit(t_bare_three_two, by_suit)
        else:
            if self.config.allow_bare_three_two_trump and t_bare_three_two:
                return self._pick_best_suit(t_bare_three_two, by_suit)
            if t_four_non_piece:
                return self._pick_best_suit(t_four_non_piece, by_suit)

        # Tier 12: >= 3 non-piece cards (mathematically guaranteed for any 10-card hand)
        t12 = [s for s in Suit if len(by_suit[s]) >= 3]
        if t12:
            return self._pick_best_suit(t12, by_suit)

        # Safety net for non-standard partial hands in unit tests
        return self._pick_best_suit(list(Suit), by_suit)

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return True

    def decide_callout(self, obs: Observation) -> bool:
        team = obs.team_id
        total = obs.match_scores[team] + thirds_to_points(obs.hand_thirds[team])
        return total >= obs.target_score

    # ------------------------------------------------------------------
    # Card Selection (Opening Plan + Exact Trump Counting + V1 Midgame)
    # ------------------------------------------------------------------
    def choose_card(self, obs: Observation) -> Card:
        if not obs.legal_cards:
            raise ValueError("HeuristicBotV2 received empty legal_cards.")
        if len(obs.legal_cards) == 1:
            return obs.legal_cards[0]

        legal_set = set(obs.legal_cards)

        # Leading a trick
        if not obs.current_trick:
            if obs.current_trick_number == 1 and obs.player_id == obs.trump_caller:
                card = self._choose_caller_trick1_lead(obs)
                if card in legal_set:
                    return card
            else:
                card = self._choose_followup_lead(obs)
                if card is not None and card in legal_set:
                    return card

        return self._fallback_bot.choose_card(obs)

    # ------------------------------------------------------------------
    # Caller's Trick-1 Opening Plan
    # ------------------------------------------------------------------
    def _choose_caller_trick1_lead(self, obs: Observation) -> Card:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        trump_cards = _suit_cards(obs.hand, trump)
        tn = len(trump_cards)
        tranks = {c.rank for c in trump_cards}

        has_3 = Rank.THREE in tranks
        has_2 = Rank.TWO in tranks
        has_a = Rank.ACE in tranks
        piece_count = int(has_3) + int(has_2) + int(has_a)

        # 1. Trump Cricca -> always lead Asso of trump
        if piece_count == 3:
            return Card(trump, Rank.ACE)

        # 2. 3 + 2 with >= 3 trumps -> lead Tre of trump
        if has_3 and has_2 and tn >= 3:
            return Card(trump, Rank.THREE)

        # 3. 3 + A with >= 4 trumps -> configurable (EXP-06: default low_trump / Striscio)
        if has_3 and has_a and not has_2 and tn >= 4:
            mode = self.config.three_ace_four_plus_lead
            if mode == "low_trump":
                return _lowest_card(trump_cards)
            if mode == "three_busso":
                return Card(trump, Rank.THREE)
            if mode == "side_suit":
                return self._choose_side_lead_ladder_a(obs)
            return _lowest_card(trump_cards)

        # 4. 2 + A or solo 2 with >= 4 trumps -> lowest trump (Busso Case B)
        if has_2 and not has_3 and tn >= 4:
            return _lowest_card(trump_cards)

        # 5. Solo Asso with >= 4 trumps -> lowest trump (protects Asso, H10)
        if has_a and not has_3 and not has_2 and tn >= 4:
            return _lowest_card(trump_cards)

        # 6. 2 + A or solo 2 with 3 trumps (EXP-07)
        if has_2 and not has_3 and tn == 3:
            if self.config.lead_trump_on_three_card_due:
                return _lowest_card(trump_cards)
            return self._choose_side_lead_ladder_a(obs)

        # 7. 3 + A with 3 trumps (EXP-08)
        if has_3 and has_a and not has_2 and tn == 3:
            return self._choose_side_lead_ladder_a(obs, is_three_ace_three_card=True)

        # 8. >= 6 trumps without Asso & without Due (solo 3 or 0 pieces) -> Ladder B
        #    Also supports EXP-11 threshold (4 or 6) for 0-piece trumps.
        if (tn >= 6 and not has_a and not has_2) or (
            piece_count == 0 and tn >= self.config.non_piece_trump_ladder_b_threshold
        ):
            return self._choose_side_lead_ladder_b(obs)

        # 9. All other cases (solo 3 w/ 3-5, solo A w/ 3, 0 pieces w/ 3-5, bare 3+2) -> Ladder A
        return self._choose_side_lead_ladder_a(obs)

    def _side_lead_steps_1_to_3(self, obs: Observation) -> Card | None:
        """Shared Steps 1-3 for Ladder A and Ladder B on Trick 1."""
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        side_suits = [s for s in Suit if s != trump]
        by_suit = {s: _suit_cards(obs.hand, s) for s in side_suits}

        # Step 0 (safety for forced-trump experiments): non-trump Cricca -> cash Asso of shortest
        non_trump_criccas = [s for s in side_suits if has_cricca(by_suit[s], s)]
        if non_trump_criccas:
            best_s = min(non_trump_criccas, key=lambda s: (len(by_suit[s]), s.order))
            return Card(best_s, Rank.ACE)

        # Step 1: Side suit with Tre (excluding bare 3+A if avoid_bare_three_ace_lead=True)
        three_candidates: list[tuple[tuple[int, int, int], Suit]] = []
        for s in side_suits:
            cards = by_suit[s]
            ranks = {c.rank for c in cards}
            if Rank.THREE not in ranks:
                continue
            is_bare_3a = len(cards) == 2 and ranks == {Rank.THREE, Rank.ACE}
            if is_bare_3a and self.config.avoid_bare_three_ace_lead:
                continue

            if len(cards) == 2 and ranks == {Rank.THREE, Rank.TWO}:
                prio = 0  # Bare 3+2 cashout
            elif len(cards) >= 3 and Rank.ACE in ranks and Rank.TWO not in ranks:
                prio = 1  # Busso Case A (3+A+x)
            elif len(cards) == 1:
                prio = 2  # Singleton Tre (Volo)
            else:
                prio = 3  # Other Tre
            three_candidates.append(((prio, len(cards), s.order), s))

        if three_candidates:
            _, best_s = min(three_candidates, key=lambda item: item[0])
            return Card(best_s, Rank.THREE)

        # Step 2: Side suit with Due for Busso Case B (>= 3 cards, no Tre, lowest < Asso)
        busso_b_candidates: list[tuple[tuple[int, int, int], Card]] = []
        for s in side_suits:
            cards = by_suit[s]
            ranks = {c.rank for c in cards}
            if len(cards) >= 3 and Rank.TWO in ranks and Rank.THREE not in ranks:
                low = _lowest_card(cards)
                if low.power < Rank.ACE.power:
                    if self.config.side_suit_tiebreak == "lowest_card":
                        key = (low.power, -len(cards), s.order)
                    else:
                        key = (-len(cards), low.power, s.order)
                    busso_b_candidates.append((key, low))

        if busso_b_candidates:
            _, best_card = min(busso_b_candidates, key=lambda item: item[0])
            return best_card

        # Step 3: Volo (singleton <= Re; excludes bare Asso and bare Due)
        volo_candidates: list[Card] = []
        for s in side_suits:
            cards = by_suit[s]
            if len(cards) == 1 and cards[0].power <= Rank.KING.power:
                volo_candidates.append(cards[0])

        if volo_candidates:
            return min(volo_candidates, key=lambda c: (c.power, c.suit.order))

        return None

    def _choose_side_lead_ladder_a(
        self, obs: Observation, *, is_three_ace_three_card: bool = False
    ) -> Card:
        assert obs.trump_suit is not None
        trump = obs.trump_suit

        # Steps 1-3
        step13 = self._side_lead_steps_1_to_3(obs)
        if step13 is not None:
            return step13

        side_suits = [s for s in Suit if s != trump]
        by_suit = {s: _suit_cards(obs.hand, s) for s in side_suits}

        # Step 4: Side suit with Asso (>= 3 cards) -> lowest card (even Cavallo from A,K,C)
        ace_candidates: list[tuple[tuple[int, int, int], Card]] = []
        for s in side_suits:
            cards = by_suit[s]
            ranks = {c.rank for c in cards}
            if len(cards) >= 3 and Rank.ACE in ranks:
                low = _lowest_card(cards)
                if self.config.side_suit_tiebreak == "lowest_card":
                    key = (low.power, -len(cards), s.order)
                else:
                    key = (-len(cards), low.power, s.order)
                ace_candidates.append((key, low))

        if ace_candidates:
            _, best_card = min(ace_candidates, key=lambda item: item[0])
            return best_card

        # Step 5: 2-card side suit without ANY Cricca pieces (prepares Volo safely)
        two_card_safe: list[Card] = []
        for s in side_suits:
            cards = by_suit[s]
            if len(cards) == 2 and not any(c.rank.is_cricca_piece for c in cards):
                two_card_safe.append(_lowest_card(cards))

        if two_card_safe:
            return min(two_card_safe, key=lambda c: (c.power, c.suit.order))

        # Step 6: No <=2-card safe side suit available (e.g., 4-3-0 or 7-0-0 distribution)
        trump_cards = _suit_cards(obs.hand, trump)
        long_side_suits = [s for s in side_suits if len(by_suit[s]) >= 3]

        if is_three_ace_three_card:
            fb = self.config.three_ace_three_card_fallback
            if fb == "trump_three" and Card(trump, Rank.THREE) in obs.hand:
                return Card(trump, Rank.THREE)
            if fb == "low_trump" and trump_cards:
                return _lowest_card(trump_cards)
            if fb == "longest_side" and long_side_suits:
                best_s = max(
                    long_side_suits,
                    key=lambda s: (len(by_suit[s]), -_lowest_card(by_suit[s]).power, -s.order),
                )
                return _lowest_card(by_suit[best_s])

        if long_side_suits:
            best_s = min(
                long_side_suits,
                key=lambda s: (len(by_suit[s]), _lowest_card(by_suit[s]).power, s.order),
            )
            return _lowest_card(by_suit[best_s])

        # Step 7: Deadlock fallback (all side suits empty or forbidden 2-card piece pairs)
        if trump_cards:
            return _lowest_card(trump_cards)

        return _lowest_card(obs.legal_cards)

    def _choose_side_lead_ladder_b(self, obs: Observation) -> Card:
        """Ladder B (>=6 non-A/2 trumps, or configurable 4+ non-piece): Steps 1-3 then low trump."""
        assert obs.trump_suit is not None
        step13 = self._side_lead_steps_1_to_3(obs)
        if step13 is not None:
            return step13

        trump_cards = _suit_cards(obs.hand, obs.trump_suit)
        if trump_cards:
            return _lowest_card(trump_cards)
        return _lowest_card(obs.legal_cards)

    # ------------------------------------------------------------------
    # Tricks 2-10 Continuation Plans (Exact 10-Trump Count & Card Counting)
    # ------------------------------------------------------------------
    def _others_hold_trumps(self, obs: Observation) -> bool:
        assert obs.trump_suit is not None
        if not self.config.use_exact_trump_count:
            return True
        trump = obs.trump_suit
        played_trumps = sum(1 for c in obs.played_cards if c.suit == trump)
        my_trumps = sum(1 for c in obs.hand if c.suit == trump)
        return (played_trumps + my_trumps) < 10

    def _choose_followup_lead(self, obs: Observation) -> Card | None:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        me = obs.player_id
        my_team = obs.team_id
        opponents = [p for p in range(4) if team_of(p) != my_team]
        voids = _infer_known_voids(obs)
        others_have_trumps = self._others_hold_trumps(obs)

        trump_cards = _suit_cards(obs.hand, trump)
        tranks = {c.rank for c in trump_cards}

        # 1. Caller's Trump Cricca continuation (exhaust enemy trumps with Tre then Due)
        if me == obs.trump_caller and obs.cricca_declared_by == me:
            # EXP-10 alternative: cash 2nd Cricca immediately on Trick 2 before exhausting trumps
            if (
                not self.config.exhaust_trumps_before_second_cricca
                and obs.current_trick_number == 2
            ):
                cricca_ace = self._pick_non_trump_cricca_ace(obs, voids, opponents)
                if cricca_ace is not None:
                    return cricca_ace

            if others_have_trumps:
                if Rank.THREE in tranks:
                    return Card(trump, Rank.THREE)
                if Rank.TWO in tranks and Card(trump, Rank.THREE) in obs.played_cards:
                    return Card(trump, Rank.TWO)

            # Once enemy trumps are exhausted (or no Tre/Due left), cash any 2nd/3rd Cricca!
            cricca_ace = self._pick_non_trump_cricca_ace(obs, voids, opponents)
            if cricca_ace is not None:
                return cricca_ace

        # 2. Caller's 3+2 trump opener continuation -> draw Due on Trick 2 if trumps remain
        if (
            me == obs.trump_caller
            and obs.current_trick_number == 2
            and obs.completed_tricks
            and obs.completed_tricks[0].plays[0].card == Card(trump, Rank.THREE)
            and obs.initial_suit_counts.get(trump, 0) >= 3
            and Rank.TWO in tranks
            and others_have_trumps
        ):
            return Card(trump, Rank.TWO)

        # 3. Caller's 4+ solo-Asso trump opener continuation -> keep drawing if trumps remain
        if (
            me == obs.trump_caller
            and obs.current_trick_number == 2
            and obs.initial_suit_counts.get(trump, 0) >= 4
            and Rank.ACE in tranks
            and Rank.THREE not in tranks
            and Rank.TWO not in tranks
            and len(trump_cards) >= 2
            and others_have_trumps
            and obs.completed_tricks
            and obs.completed_tricks[0].led_suit == trump
        ):
            low_t = _lowest_card(trump_cards)
            if low_t.power < Rank.ACE.power:
                return low_t

        # 4. Bare 3+2 side-suit cashout continuation (Trick 1 cashed Tre -> cash Due if safe per H08)
        if (
            obs.current_trick_number == 2
            and obs.completed_tricks
            and obs.completed_tricks[0].leader == me
            and obs.completed_tricks[0].plays[0].card.rank == Rank.THREE
        ):
            prev_suit = obs.completed_tricks[0].led_suit
            if prev_suit != trump:
                rem = _suit_cards(obs.hand, prev_suit)
                if (
                    len(rem) == 1
                    and rem[0].rank == Rank.TWO
                    and not self._opponent_can_cut(prev_suit, voids, opponents, others_have_trumps)
                ):
                    return rem[0]

        # 5. Any non-trump Cricca cashout (shortest suit first) if safe
        cricca_ace = self._pick_non_trump_cricca_ace(obs, voids, opponents)
        if cricca_ace is not None:
            return cricca_ace

        # 6. Suit card-counting upgrade for multiple safe non-trump Threes on Tricks 2-9
        if self.config.use_suit_card_counting and obs.current_trick_number < 10:
            best_three = self._pick_safest_non_trump_three(
                obs, voids, opponents, others_have_trumps
            )
            if best_three is not None:
                return best_three

        return None

    def _opponent_can_cut(
        self,
        suit: Suit,
        voids: dict[int, set[Suit]],
        opponents: list[int],
        others_have_trumps: bool,
    ) -> bool:
        if not others_have_trumps:
            return False
        return any(suit in voids[opp] for opp in opponents)

    def _pick_non_trump_cricca_ace(
        self,
        obs: Observation,
        voids: dict[int, set[Suit]],
        opponents: list[int],
    ) -> Card | None:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        others_have_trumps = self._others_hold_trumps(obs)
        candidates: list[tuple[int, int, Suit]] = []
        for s in Suit:
            if s == trump:
                continue
            cards = _suit_cards(obs.hand, s)
            if has_cricca(cards, s) and not self._opponent_can_cut(
                s, voids, opponents, others_have_trumps
            ):
                candidates.append((len(cards), s.order, s))
        if not candidates:
            return None
        _, _, best_s = min(candidates)
        return Card(best_s, Rank.ACE)

    def _pick_safest_non_trump_three(
        self,
        obs: Observation,
        voids: dict[int, set[Suit]],
        opponents: list[int],
        others_have_trumps: bool,
    ) -> Card | None:
        """Choose the non-trump Tre whose suit has fewest cards already played."""
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        partner = partner_of(obs.player_id)
        candidates: list[tuple[tuple[int, int, int, int], Card]] = []

        for s in Suit:
            if s == trump:
                continue
            cards = _suit_cards(obs.hand, s)
            ranks = {c.rank for c in cards}
            if Rank.THREE not in ranks:
                continue
            if self._opponent_can_cut(s, voids, opponents, others_have_trumps):
                continue

            # Softly deprioritize bare 3+A if Due is still unplayed
            is_bare_3a_risk = (
                len(cards) == 2
                and ranks == {Rank.THREE, Rank.ACE}
                and Card(s, Rank.TWO) not in obs.played_cards
                and self.config.avoid_bare_three_ace_lead
            )
            played_in_suit = sum(1 for c in obs.played_cards if c.suit == s)
            partner_void_bonus = 0 if (s in voids[partner] and others_have_trumps) else 1

            # Fewer played cards = lower probability an opponent is void
            key = (
                1 if is_bare_3a_risk else 0,
                played_in_suit,
                partner_void_bonus,
                s.order,
            )
            candidates.append((key, Card(s, Rank.THREE)))

        # Only override V1 when there's at least one non-risky safe Tre (or multiple Threes)
        non_risky = [c for c in candidates if c[0][0] == 0]
        if non_risky:
            _, best_card = min(non_risky, key=lambda item: item[0])
            return best_card
        return None