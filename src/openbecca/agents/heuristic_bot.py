"""Rule-based Ravenna strategy agent encoding testable hypotheses H01-H12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openbecca.agents.base import Agent
from openbecca.cards import ALL_CARDS, Card, Rank, Suit
from openbecca.rules import (
    TrickPlay,
    determine_trick_winner,
    has_cricca,
    partner_of,
    team_of,
    thirds_to_points,
)
from openbecca.signals import Signal, determine_lead_signal
from openbecca.state import Observation


@dataclass(frozen=True)
class HeuristicConfig:
    """Toggle switches for ablation studies on Hypotheses H01-H12."""

    h01_early_non_trump_three: bool = True
    h02_feed_partner_three: bool = True
    h03_trump_with_figure: bool = True
    h04_save_trump_three_for_last: bool = True
    h05_defensive_two_against_ace: bool = True
    h06_counter_busso_with_three: bool = False
    h07_non_trump_cricca_ace_lead: bool = True
    h08_void_aware_lead_priority: bool = True
    h09_safe_trump_ace: bool = True
    h10_trump_drawing_opener: bool = True
    h11_fallback_ace_suit_lead: bool = True
    h12_smart_trump_selection: bool = True


class HeuristicBot(Agent):
    """Plays Beccacino using the 12 Ravenna heuristics documented in DECISIONS.md.

    Strictly respects imperfect information: voids and remaining boss cards are
    inferred solely from completed tricks, the current trick, and public signals.
    """

    def __init__(
        self,
        name: str = "HeuristicBot",
        config: HeuristicConfig | None = None,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        self.config = config or HeuristicConfig()

    def choose_trump(self, obs: Observation) -> Suit:
        """Select trump using H12 (Cricca -> 3+2+extra -> 4+ cards -> strong 3-card)."""
        by_suit: dict[Suit, list[Card]] = {
            suit: [c for c in obs.hand if c.suit == suit] for suit in Suit
        }

        if not self.config.h12_smart_trump_selection:
            return max(Suit, key=lambda s: len(by_suit[s]))

        # 1. Always call a suit with Cricca (Asso + Due + Tre)
        cricca_suits = [s for s in Suit if has_cricca(by_suit[s], s)]
        if cricca_suits:
            return max(cricca_suits, key=lambda s: len(by_suit[s]))

        # Identify suits where we hold ONLY Tre + Due (keep non-trump to cash early)
        pure_three_two_suits: set[Suit] = set()
        strong_three_two_suits: list[Suit] = []
        for suit, cards in by_suit.items():
            ranks = {c.rank for c in cards}
            if Rank.THREE in ranks and Rank.TWO in ranks:
                if len(cards) >= 3:
                    strong_three_two_suits.append(suit)
                else:
                    pure_three_two_suits.add(suit)

        # 2. Tre + Due + at least one other card of that suit
        if strong_three_two_suits:
            return max(
                strong_three_two_suits,
                key=lambda s: (len(by_suit[s]), sum(c.power for c in by_suit[s])),
            )

        # Candidate suits excluding pure 2-card (Tre + Due) holdings when alternatives exist
        candidates = [s for s in Suit if s not in pure_three_two_suits] or list(Suit)

        # 3. Four or more cards in a suit
        long_suits = [s for s in candidates if len(by_suit[s]) >= 4]
        if long_suits:
            return max(
                long_suits,
                key=lambda s: (
                    len(by_suit[s]),
                    sum(1 for c in by_suit[s] if c.rank in (Rank.THREE, Rank.TWO, Rank.ACE)),
                    sum(c.power for c in by_suit[s]),
                ),
            )

        # 4. Strong 3-card suit (e.g., Tre + Re + Fante) or best overall score
        def _suit_strength(suit: Suit) -> float:
            cards = by_suit[suit]
            ranks = {c.rank for c in cards}
            score = len(cards) * 2.5
            if Rank.THREE in ranks:
                score += 4.0
            if Rank.TWO in ranks:
                score += 3.0
            if Rank.ACE in ranks:
                score += 2.5
            score += sum(1.0 for c in cards if c.rank.is_figure)
            return score

        return max(candidates, key=_suit_strength)

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return True

    def decide_callout(self, obs: Observation) -> bool:
        team = obs.team_id
        total = obs.match_scores[team] + thirds_to_points(obs.hand_thirds[team])
        return total >= obs.target_score

    def choose_card(self, obs: Observation) -> Card:
        if not obs.legal_cards:
            raise ValueError("HeuristicBot received empty legal_cards.")
        if len(obs.legal_cards) == 1:
            return obs.legal_cards[0]

        assert obs.trump_suit is not None
        known_voids = self._infer_known_voids(obs)

        if not obs.current_trick:
            return self._choose_lead_card(obs, known_voids)
        return self._choose_follow_card(obs, known_voids)

    # ------------------------------------------------------------------
    # Public-information inference (strictly imperfect information)
    # ------------------------------------------------------------------
    def _infer_known_voids(self, obs: Observation) -> dict[int, set[Suit]]:
        voids: dict[int, set[Suit]] = {p: set() for p in range(4)}

        for trick in obs.completed_tricks:
            led_suit = trick.led_suit
            if trick.signal == Signal.VOLO:
                voids[trick.leader].add(led_suit)
            for play in trick.plays[1:]:
                if play.card.suit != led_suit:
                    voids[play.player].add(led_suit)

        if obs.current_trick:
            led_suit = obs.current_trick[0].card.suit
            if obs.current_trick_signal == Signal.VOLO:
                voids[obs.current_trick_leader].add(led_suit)
            for play in obs.current_trick[1:]:
                if play.card.suit != led_suit:
                    voids[play.player].add(led_suit)

        return voids

    def _is_suit_safe_to_lead(
        self, suit: Suit, obs: Observation, known_voids: dict[int, set[Suit]]
    ) -> bool:
        """H08 void check: avoid leading suits where an enemy is void unless partner
        acts after the void opponent (and the last opponent is not also void)."""
        if suit == obs.trump_suit:
            return True

        p = obs.player_id
        opp_before_partner = (p + 1) % 4
        partner = (p + 2) % 4
        opp_after_partner = (p + 3) % 4

        opp1_void = suit in known_voids[opp_before_partner]
        partner_void = suit in known_voids[partner]
        opp2_void = suit in known_voids[opp_after_partner]

        if not opp1_void and not opp2_void:
            return True
        if partner_void and not opp2_void:
            return True
        # Opp1 is void, Opp2 is not void -> partner gets last word on trumping
        if opp1_void and not opp2_void:
            return True
        return False

    def _opponents_acting_after_me(self, obs: Observation) -> list[int]:
        already_played = len(obs.current_trick)
        remaining: list[int] = []
        for offset in range(already_played + 1, 4):
            pid = (obs.current_trick_leader + offset) % 4
            if team_of(pid) != obs.team_id:
                remaining.append(pid)
        return remaining

    def _higher_trump_possibly_held_by_later_opponent(self, obs: Observation) -> bool:
        assert obs.trump_suit is not None
        later_opps = self._opponents_acting_after_me(obs)
        if not later_opps:
            return False

        unseen = set(ALL_CARDS) - obs.played_cards - set(obs.hand)
        higher_trumps = {
            Card(obs.trump_suit, Rank.THREE),
            Card(obs.trump_suit, Rank.TWO),
        }
        return bool(unseen & higher_trumps)

    # ------------------------------------------------------------------
    # Leading a trick (first card on the table)
    # ------------------------------------------------------------------
    def _choose_lead_card(
        self, obs: Observation, known_voids: dict[int, set[Suit]]
    ) -> Card:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        legal = list(obs.legal_cards)
        played_suits = {t.led_suit for t in obs.completed_tricks}
        trump_three = Card(trump, Rank.THREE)
        trump_two = Card(trump, Rank.TWO)
        trump_ace = Card(trump, Rank.ACE)

        # H06: Counter-Busso follow-up — if we just won opponent's Busso with Tre, lead back
        #if self.config.h06_counter_busso_with_three and obs.completed_tricks:
        #    last = obs.completed_tricks[-1]
        #    if (
        #        last.signal == Signal.BUSSO
        ##        and team_of(last.leader) != obs.team_id
        #        and last.winner == obs.player_id
        #    ):
        #        my_play = next(p for p in last.plays if p.player == obs.player_id)
        #        if my_play.card.suit == last.led_suit and my_play.card.rank == Rank.THREE:
        #            same_suit = [c for c in legal if c.suit == last.led_suit]
        #            if same_suit:
        #                return min(same_suit, key=lambda c: (c.points_thirds, c.power))

        # H10: Trump-drawing opener when calling trump with strong trump holdings
        if self.config.h10_trump_drawing_opener and obs.trump_caller == obs.player_id:
            my_trumps = [c for c in legal if c.suit == trump]
            my_trump_ranks = {c.rank for c in my_trumps}

            # Case 10a: Called with Tre + Due + >=1 other trump -> lead Tre then Due early
            if obs.current_trick_number == 1 and len(my_trumps) >= 3:
                if trump_three in legal and trump_two in legal:
                    return trump_three
            if (
                obs.current_trick_number == 2
                and trump_three in obs.played_cards
                and trump_two in legal
                and len(my_trumps) >= 2
            ):
                return trump_two

            # Case 10b: Started with >= 4 trumps including Asso -> draw enemy trumps
            if (
                obs.initial_suit_counts[trump] >= 4
                and trump_ace in obs.hand
                and obs.current_trick_number <= 3
                and my_trumps
            ):
                if trump_ace in legal and not self._higher_trump_possibly_held_by_later_opponent(obs):
                    return trump_ace
                non_three_trumps = [c for c in my_trumps if c != trump_three]
                if non_three_trumps:
                    return min(non_three_trumps, key=lambda c: (c.points_thirds, c.power))

        # H04: Keep trump Tre out of lead candidates before Trick 10 when alternatives exist
        pool = legal
        if (
            self.config.h04_save_trump_three_for_last
            and obs.current_trick_number < 10
            and trump_three in pool
            and len(pool) > 1
        ):
            pool = [c for c in pool if c != trump_three]

        # H07: Non-trump Cricca -> lead Asso on first appearance of that suit
        if self.config.h07_non_trump_cricca_ace_lead:
            for suit in Suit:
                if suit == trump or suit in played_suits:
                    continue
                if has_cricca(obs.hand, suit):
                    ace = Card(suit, Rank.ACE)
                    if ace in pool and (
                        not self.config.h08_void_aware_lead_priority
                        or self._is_suit_safe_to_lead(suit, obs, known_voids)
                    ):
                        return ace

        # Helper to prefer safe suits (H08)
        def _filter_safe(cards: list[Card]) -> list[Card]:
            if not self.config.h08_void_aware_lead_priority:
                return cards
            return [c for c in cards if self._is_suit_safe_to_lead(c.suit, obs, known_voids)]

        # H01 & H08: Lead non-trump Tre (especially on unseen/safe suits)
        if self.config.h01_early_non_trump_three:
            non_trump_threes = [
                c for c in pool if c.suit != trump and c.rank == Rank.THREE
            ]
            safe_threes = _filter_safe(non_trump_threes)
            if safe_threes:
                # Prefer a Tre that also triggers Busso (Case A), else unseen suit
                busso_threes = [c for c in safe_threes if self._would_signal_busso(c, obs)]
                if busso_threes:
                    return busso_threes[0]
                unseen_threes = [c for c in safe_threes if c.suit not in played_suits]
                return unseen_threes[0] if unseen_threes else safe_threes[0]

        # H08: Lead boss Due (when Tre of that suit is already played)
        if self.config.h08_void_aware_lead_priority:
            boss_twos = [
                c
                for c in pool
                if c.suit != trump
                and c.rank == Rank.TWO
                and Card(c.suit, Rank.THREE) in obs.played_cards
            ]
            safe_twos = _filter_safe(boss_twos)
            if safe_twos:
                return safe_twos[0]

        # H08: Valid Busso lead (prefer non-trump Busso on safe suits)
        if self.config.h08_void_aware_lead_priority:
            busso_leads = [c for c in pool if self._would_signal_busso(c, obs)]
            safe_busso = _filter_safe([c for c in busso_leads if c.suit != trump])
            if safe_busso:
                return min(safe_busso, key=lambda c: (c.points_thirds, c.power))
            if busso_leads:
                return min(busso_leads, key=lambda c: (c.points_thirds, c.power))

        # H08: Lead a suit where partner is void (and last opponent isn't) so partner can cut
        if self.config.h08_void_aware_lead_priority:
            partner = partner_of(obs.player_id)
            opp_after_partner = (obs.player_id + 3) % 4
            partner_cut_cards = [
                c
                for c in pool
                if c.suit != trump
                and c.suit in known_voids[partner]
                and c.suit not in known_voids[opp_after_partner]
                and c.rank != Rank.ACE
            ]
            if partner_cut_cards:
                return min(partner_cut_cards, key=lambda c: (c.points_thirds, c.power))

        # H11: Fallback — lead a low card from a suit where we hold the Asso
        if self.config.h11_fallback_ace_suit_lead:
            ace_suits = {
                c.suit for c in obs.hand if c.rank == Rank.ACE and c.suit != trump
            }
            ace_suit_lows = [
                c
                for c in pool
                if c.suit in ace_suits and c.rank != Rank.ACE
            ]
            safe_ace_lows = _filter_safe(ace_suit_lows)
            if safe_ace_lows:
                return min(safe_ace_lows, key=lambda c: (c.points_thirds, c.power))
            if ace_suit_lows:
                return min(ace_suit_lows, key=lambda c: (c.points_thirds, c.power))

        # General safe low non-trump discard lead
        non_trump_pool = [c for c in pool if c.suit != trump]
        safe_non_trump = _filter_safe(non_trump_pool)
        if safe_non_trump:
            return min(safe_non_trump, key=lambda c: (c.points_thirds, c.power))
        if non_trump_pool:
            return min(non_trump_pool, key=lambda c: (c.points_thirds, c.power))

        return min(pool, key=lambda c: (c.points_thirds, c.power))

    # ------------------------------------------------------------------
    # Following a trick (2nd, 3rd, or 4th card on the table)
    # ------------------------------------------------------------------
    def _choose_follow_card(
        self, obs: Observation, known_voids: dict[int, set[Suit]]
    ) -> Card:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        legal = list(obs.legal_cards)
        led_suit = obs.current_trick[0].card.suit
        following_suit = legal[0].suit == led_suit and all(c.suit == led_suit for c in legal)

        current_winner = determine_trick_winner(obs.current_trick, trump)
        partner_winning = team_of(current_winner) == obs.team_id
        later_opps = self._opponents_acting_after_me(obs)

        if following_suit:
            return self._follow_matching_suit(
                obs, legal, led_suit, partner_winning, later_opps, known_voids
            )
        return self._play_when_void(obs, legal, partner_winning, later_opps)

    def _follow_matching_suit(
        self,
        obs: Observation,
        legal: list[Card],
        led_suit: Suit,
        partner_winning: bool,
        later_opps: list[int],
        known_voids: dict[int, set[Suit]],
    ) -> Card:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        trump_three = Card(trump, Rank.THREE)
        played_suits = {t.led_suit for t in obs.completed_tricks}

        # H06: Against opponent's Busso, take immediately with Tre if held
        if (
            self.config.h06_counter_busso_with_three
            and obs.current_trick_signal == Signal.BUSSO
            and team_of(obs.current_trick_leader) != obs.team_id
        ):
            three = Card(led_suit, Rank.THREE)
            if three in legal and self._would_win_current_trick(obs, three):
                return three

        # H02: Partner led non-trump Tre and is still winning -> feed Asso or figure
        if (
            self.config.h02_feed_partner_three
            and led_suit != trump
            and partner_winning
            and obs.current_trick[0].player == partner_of(obs.player_id)
            and obs.current_trick[0].card.rank == Rank.THREE
        ):
            enemy_might_cut = any(led_suit in known_voids[opp] for opp in later_opps)
            if not enemy_might_cut:
                ace = Card(led_suit, Rank.ACE)
                if ace in legal:
                    return ace
                figures = [c for c in legal if c.rank.is_figure]
                if figures:
                    return min(figures, key=lambda c: c.power)

        # If partner is already winning and no opponent acts after us, feed maximum points safely
        if partner_winning and not later_opps:
            return self._best_safe_feed(legal, obs)

        # H09: Following trump — play trump Asso if safe and we can win or partner wins
        if self.config.h09_safe_trump_ace and led_suit == trump:
            trump_ace = Card(trump, Rank.ACE)
            if trump_ace in legal:
                if (
                    self._would_win_current_trick(obs, trump_ace) or partner_winning
                ) and not self._higher_trump_possibly_held_by_later_opponent(obs):
                    return trump_ace

        # H05: Defensive Due on first appearance of a non-trump suit with low cards on table
        if (
            self.config.h05_defensive_two_against_ace
            and led_suit != trump
            and led_suit not in played_suits
            and later_opps
            and not any(p.card.suit == trump for p in obs.current_trick)
        ):
            two = Card(led_suit, Rank.TWO)
            ace = Card(led_suit, Rank.ACE)
            if (
                two in legal
                and ace not in obs.hand
                and ace not in obs.played_cards
                and all(p.card.power < Rank.ACE.power for p in obs.current_trick)
            ):
                return two

        # Standard tactical win vs duck decision
        winning_cards = [c for c in legal if self._would_win_current_trick(obs, c)]
        if (
            self.config.h04_save_trump_three_for_last
            and obs.current_trick_number < 10
            and trump_three in winning_cards
            and len(winning_cards) > 1
        ):
            winning_cards = [c for c in winning_cards if c != trump_three]

        trick_has_value = (
            sum(p.card.points_thirds for p in obs.current_trick) >= 2
            or obs.current_trick_number == 10
        )

        if not partner_winning and winning_cards:
            # If last to act (no later opponents), win as cheaply as possible (or with Asso!)
            if not later_opps:
                ace = Card(led_suit, Rank.ACE)
                if ace in winning_cards:
                    return ace
                return min(winning_cards, key=lambda c: c.power)
            # If opponents act after us, take valuable tricks or take with high card
            if trick_has_value:
                return max(winning_cards, key=lambda c: c.power)
            return min(winning_cards, key=lambda c: c.power)

        # Cannot beat or partner is already winning safely -> play lowest-value card
        return self._least_valuable_card(legal, obs)

    def _play_when_void(
        self,
        obs: Observation,
        legal: list[Card],
        partner_winning: bool,
        later_opps: list[int],
    ) -> Card:
        assert obs.trump_suit is not None
        trump = obs.trump_suit
        trumps = [c for c in legal if c.suit == trump]
        non_trumps = [c for c in legal if c.suit != trump]
        trump_three = Card(trump, Rank.THREE)
        trump_ace = Card(trump, Rank.ACE)

        # If partner is winning and no opponent acts after us -> feed points (Asso/figure)
        if partner_winning and not later_opps:
            if non_trumps:
                return self._best_safe_feed(non_trumps, obs)
            return self._least_valuable_card(legal, obs)

        # If partner is winning with a strong card (e.g. non-trump Tre or trump), avoid overtrumping
        if partner_winning and non_trumps:
            partner_card = next(
                (p.card for p in obs.current_trick if p.player == partner_of(obs.player_id)),
                None,
            )
            if partner_card is not None and (
                partner_card.suit == trump or partner_card.rank == Rank.THREE
            ):
                # Feed a figure or safe discard rather than cutting partner
                return self._least_valuable_card(non_trumps, obs)

        # Try to cut with a winning trump
        winning_trumps = [c for c in trumps if self._would_win_current_trick(obs, c)]
        if winning_trumps:
            # H04: Protect trump Tre before Trick 10 when other winning trumps exist
            if (
                self.config.h04_save_trump_three_for_last
                and obs.current_trick_number < 10
                and trump_three in winning_trumps
                and len(winning_trumps) > 1
            ):
                winning_trumps = [c for c in winning_trumps if c != trump_three]

            # H09: Cut with trump Asso when safe from higher enemy trumps
            if (
                self.config.h09_safe_trump_ace
                and trump_ace in winning_trumps
                and not self._higher_trump_possibly_held_by_later_opponent(obs)
            ):
                return trump_ace

            # H03: Prefer trumping with a figure (Fante, Cavallo, Re) to win +1/3 pt
            if self.config.h03_trump_with_figure:
                trump_figures = [c for c in winning_trumps if c.rank.is_figure]
                if trump_figures:
                    return min(trump_figures, key=lambda c: c.power)

            # Avoid spending trump Tre early on a zero-point trick if we can discard trash
            trick_pts = sum(p.card.points_thirds for p in obs.current_trick)
            if (
                self.config.h04_save_trump_three_for_last
                and winning_trumps == [trump_three]
                and obs.current_trick_number < 10
                and trick_pts < 3
                and non_trumps
            ):
                return self._least_valuable_card(non_trumps, obs)

            return min(winning_trumps, key=lambda c: c.power)

        # Cannot cut profitably -> discard least valuable non-trump (or lowest trump)
        if non_trumps:
            return self._least_valuable_card(non_trumps, obs)
        return self._least_valuable_card(legal, obs)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _would_signal_busso(self, card: Card, obs: Observation) -> bool:
        sig = determine_lead_signal(
            played_card=card,
            hand_before_play=obs.hand,
            initial_suit_count=obs.initial_suit_counts[card.suit],
            played_cards_history=obs.played_cards,
        )
        return sig == Signal.BUSSO

    def _would_win_current_trick(self, obs: Observation, card: Card) -> bool:
        assert obs.trump_suit is not None
        simulated = list(obs.current_trick) + [TrickPlay(obs.player_id, card)]
        return determine_trick_winner(simulated, obs.trump_suit) == obs.player_id

    def _best_safe_feed(self, cards: Iterable[Card], obs: Observation) -> Card:
        """Feed highest points to a winning partner without throwing away trump Tre/Due."""
        assert obs.trump_suit is not None
        pool = list(cards)
        non_boss = [
            c
            for c in pool
            if not (c.suit == obs.trump_suit and c.rank in (Rank.THREE, Rank.TWO))
        ]
        target = non_boss or pool
        return max(target, key=lambda c: (c.points_thirds, -c.power))

    def _least_valuable_card(self, cards: Iterable[Card], obs: Observation) -> Card:
        """Discard the lowest-value, lowest-power card (preserving trump Tre when possible)."""
        assert obs.trump_suit is not None
        pool = list(cards)
        trump_three = Card(obs.trump_suit, Rank.THREE)
        if (
            self.config.h04_save_trump_three_for_last
            and obs.current_trick_number < 10
            and trump_three in pool
            and len(pool) > 1
        ):
            pool = [c for c in pool if c != trump_three]
        return min(
            pool,
            key=lambda c: (
                1 if c.suit == obs.trump_suit else 0,
                c.points_thirds,
                c.power,
            ),
        )