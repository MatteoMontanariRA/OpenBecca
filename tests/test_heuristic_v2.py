"""Unit and smoke tests for Stockfish-tuned HeuristicBotV2 and Decision Lab."""

from openbecca.agents.heuristic_bot_v2 import (
    PRE_STOCKFISH_V2_CONFIG,
    HeuristicBotV2,
    HeuristicV2Config,
)
from openbecca.analysis.decision_lab import (
    FIXED_HAND_EXP01,
    FIXED_HAND_EXP02,
    FIXED_HAND_EXP08,
    run_experiment,
)
from openbecca.cards import Card, Deck, Rank, Suit
from openbecca.engine import GameEngine
from openbecca.rules import TrickPlay
from openbecca.signals import Signal
from openbecca.state import Observation, TrickRecord


def _make_obs(
    hand: list[Card],
    *,
    player_id: int = 0,
    trump_caller: int = 0,
    trump_suit: Suit | None = Suit.COINS,
    cricca_declared_by: int | None = None,
    completed_tricks: tuple[TrickRecord, ...] = (),
    played_cards: frozenset[Card] = frozenset(),
    initial_suit_counts: dict[Suit, int] | None = None,
) -> Observation:
    counts = (
        initial_suit_counts
        if initial_suit_counts is not None
        else {s: sum(1 for c in hand if c.suit == s) for s in Suit}
    )
    return Observation(
        player_id=player_id,
        team_id=player_id % 2,
        hand=tuple(hand),
        initial_suit_counts=counts,
        hand_number=1,
        trump_caller=trump_caller,
        trump_suit=trump_suit,
        cricca_declared_by=cricca_declared_by,
        completed_tricks=completed_tricks,
        current_trick=(),
        current_trick_leader=player_id,
        current_trick_signal=Signal.NONE,
        played_cards=played_cards,
        match_scores=(0, 0),
        hand_thirds=(0, 0),
        target_score=41,
        legal_cards=tuple(hand),
    )


def test_trump_tier1_cricca_tiebreak_length_quality_suit_order() -> None:
    bot = HeuristicBotV2()
    # Two Criccas of length 3: Denari vs Coppe -> Denari wins by suit order
    hand = [
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.TWO),
        Card(Suit.CUPS, Rank.ACE),
        Card(Suit.SWORDS, Rank.SEVEN),
        Card(Suit.SWORDS, Rank.SIX),
        Card(Suit.CLUBS, Rank.FIVE),
        Card(Suit.CLUBS, Rank.FOUR),
    ]
    assert bot.choose_trump(_make_obs(hand, trump_suit=None)) == Suit.COINS


def test_exp05_promoted_five_card_override_beats_short_three_two() -> None:
    # 5 cards in Cups vs 3+2+4 in Coins -> Stockfish-promoted V2 calls Cups (thresh=5)
    hand = [
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.FOUR),
        Card(Suit.CUPS, Rank.KING),
        Card(Suit.CUPS, Rank.KNIGHT),
        Card(Suit.CUPS, Rank.KNAVE),
        Card(Suit.CUPS, Rank.SEVEN),
        Card(Suit.CUPS, Rank.SIX),
        Card(Suit.SWORDS, Rank.FIVE),
        Card(Suit.CLUBS, Rank.FOUR),
    ]
    assert HeuristicBotV2().choose_trump(_make_obs(hand, trump_suit=None)) == Suit.CUPS
    # Legacy pre-Stockfish config (thresh=6) still calls Coins
    legacy_bot = HeuristicBotV2(config=PRE_STOCKFISH_V2_CONFIG)
    assert legacy_bot.choose_trump(_make_obs(hand, trump_suit=None)) == Suit.COINS


def test_exp01_promoted_never_calls_bare_three_two_over_three_card_suit() -> None:
    # Canonical EXP-01 hand: D:3,2 | C:3,A | S:K,C,F | B:7,6,5
    bot = HeuristicBotV2()
    obs = _make_obs(FIXED_HAND_EXP01, trump_suit=None)
    assert bot.choose_trump(obs) == Suit.SWORDS

    # And on Trick 1 after calling Spade, it cashes Tre di Denari from bare 3+2
    obs_t1 = _make_obs(FIXED_HAND_EXP01, trump_suit=Suit.SWORDS)
    assert bot.choose_card(obs_t1) == Card(Suit.COINS, Rank.THREE)

    # Legacy pre-Stockfish config still calls Denari
    legacy_bot = HeuristicBotV2(config=PRE_STOCKFISH_V2_CONFIG)
    assert legacy_bot.choose_trump(obs) == Suit.COINS


def test_exp02_four_trash_beats_bare_three_two() -> None:
    bot = HeuristicBotV2()
    assert bot.choose_trump(_make_obs(FIXED_HAND_EXP02, trump_suit=None)) == Suit.SWORDS


def test_exp06_promoted_low_trump_lead_on_four_plus_three_ace() -> None:
    hand = [
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.SEVEN),
        Card(Suit.COINS, Rank.FOUR),
        Card(Suit.CUPS, Rank.KING),
        Card(Suit.CUPS, Rank.FIVE),
        Card(Suit.SWORDS, Rank.KNIGHT),
        Card(Suit.SWORDS, Rank.SIX),
        Card(Suit.CLUBS, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.FIVE),
    ]
    obs = _make_obs(hand, trump_suit=Suit.COINS)
    # Promoted default leads lowest trump (Quattro di Denari, Striscio, saves Tre)
    assert HeuristicBotV2().choose_card(obs) == Card(Suit.COINS, Rank.FOUR)
    # Explicit legacy option leads Tre di Denari (Busso Case A)
    busso_bot = HeuristicBotV2(config=HeuristicV2Config(three_ace_four_plus_lead="three_busso"))
    assert busso_bot.choose_card(obs) == Card(Suit.COINS, Rank.THREE)


def test_exp08_three_card_three_ace_leads_shortest_side_suit() -> None:
    bot = HeuristicBotV2()
    obs = _make_obs(FIXED_HAND_EXP08, trump_suit=Suit.COINS)
    assert bot.choose_card(obs) == Card(Suit.SWORDS, Rank.SIX)


def test_exp09_avoids_bare_three_ace_offsuit_lead() -> None:
    hand = [
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.FOUR),
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.ACE),
        Card(Suit.SWORDS, Rank.SEVEN),
        Card(Suit.SWORDS, Rank.FIVE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.CLUBS, Rank.SIX),
        Card(Suit.CLUBS, Rank.FOUR),
    ]
    obs = _make_obs(hand, trump_suit=Suit.COINS)
    # Must NOT lead Tre di Coppe (bare 3+A); leads Cinque di Spade (2-card non-piece)
    assert HeuristicBotV2().choose_card(obs) == Card(Suit.SWORDS, Rank.FIVE)


def test_exact_10_trump_count_stops_drawing_on_cricca() -> None:
    # Caller holds 6 remaining trumps (including Tre & Due) and 4 trumps were played -> 10 total!
    # Must stop drawing trumps and cash 2nd Cricca in Cups (Asso di Coppe).
    hand = [
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.KING),
        Card(Suit.COINS, Rank.KNIGHT),
        Card(Suit.COINS, Rank.KNAVE),
        Card(Suit.COINS, Rank.SEVEN),
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.TWO),
        Card(Suit.CUPS, Rank.ACE),
    ]
    trick1_cards = [
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.SIX),
        Card(Suit.COINS, Rank.FIVE),
        Card(Suit.COINS, Rank.FOUR),
    ]
    trick1 = TrickRecord(
        trick_number=1,
        leader=0,
        signal=Signal.STRISCIO,
        plays=tuple(TrickPlay(i, trick1_cards[i]) for i in range(4)),
        winner=0,
        points_thirds=3,
    )
    obs = _make_obs(
        hand,
        trump_suit=Suit.COINS,
        cricca_declared_by=0,
        completed_tricks=(trick1,),
        played_cards=frozenset(trick1_cards),
    )
    assert HeuristicBotV2().choose_card(obs) == Card(Suit.CUPS, Rank.ACE)


def test_heuristic_v2_plays_full_legal_hands() -> None:
    agents = [HeuristicBotV2(f"V2-{i}") for i in range(4)]
    engine = GameEngine(agents, seed=123)
    for _ in range(20):
        res = engine.play_hand(allow_callout=False)
        assert len(res.tricks) == 10
        assert sum(res.hand_points) in (11, 14)


def test_decision_lab_smoke() -> None:
    report = run_experiment(
        "EXP-01-FIXED",
        num_deals=6,
        engine_name="heuristic",
        workers=1,
        seed=7,
        save_log=False,
    )
    assert report.num_deals == 6
    assert len(report.options) == 3