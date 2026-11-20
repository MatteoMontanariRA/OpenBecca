"""Unit and performance tests for HeuristicBot (Hypotheses H01-H12)."""

from openbecca.agents import HeuristicBot, HeuristicConfig
from openbecca.analysis.benchmark import run_hand_benchmark
from openbecca.cards import Card, Rank, Suit
from openbecca.rules import TrickPlay
from openbecca.signals import Signal
from openbecca.state import Observation


def _make_obs(
    hand: list[Card],
    *,
    player_id: int = 0,
    trump_caller: int = 0,
    trump_suit: Suit | None = Suit.COINS,
    current_trick: tuple[TrickPlay, ...] = (),
    current_trick_leader: int = 0,
    current_trick_signal: Signal = Signal.NONE,
    played_cards: frozenset[Card] = frozenset(),
    legal_cards: tuple[Card, ...] | None = None,
) -> Observation:
    counts = {s: sum(1 for c in hand if c.suit == s) for s in Suit}
    return Observation(
        player_id=player_id,
        team_id=player_id % 2,
        hand=tuple(hand),
        initial_suit_counts=counts,
        hand_number=1,
        trump_caller=trump_caller,
        trump_suit=trump_suit,
        cricca_declared_by=None,
        completed_tricks=(),
        current_trick=current_trick,
        current_trick_leader=current_trick_leader,
        current_trick_signal=current_trick_signal,
        played_cards=played_cards,
        match_scores=(0, 0),
        hand_thirds=(0, 0),
        target_score=41,
        legal_cards=legal_cards if legal_cards is not None else tuple(hand),
    )


def test_h12_trump_selection_prefers_cricca_and_avoids_pure_three_two() -> None:
    bot = HeuristicBot()

    # Pure Tre + Due in Cups, 4 cards in Swords -> must call Swords, NOT Cups
    hand = [
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.TWO),
        Card(Suit.SWORDS, Rank.KING),
        Card(Suit.SWORDS, Rank.KNIGHT),
        Card(Suit.SWORDS, Rank.SEVEN),
        Card(Suit.SWORDS, Rank.FIVE),
        Card(Suit.COINS, Rank.FOUR),
        Card(Suit.COINS, Rank.SIX),
        Card(Suit.CLUBS, Rank.FIVE),
        Card(Suit.CLUBS, Rank.FOUR),
    ]
    assert bot.choose_trump(_make_obs(hand, trump_suit=None)) == Suit.SWORDS

    # Cricca in Bastoni -> always call Bastoni
    cricca_hand = hand[:-3] + [
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
    ]
    assert bot.choose_trump(_make_obs(cricca_hand, trump_suit=None)) == Suit.CLUBS


def test_h02_feeds_ace_on_partners_non_trump_three() -> None:
    bot = HeuristicBot()
    hand = [Card(Suit.CUPS, Rank.ACE), Card(Suit.CUPS, Rank.FOUR)]
    obs = _make_obs(
        hand,
        player_id=2,
        trump_suit=Suit.COINS,
        current_trick=(
            TrickPlay(0, Card(Suit.CUPS, Rank.THREE)),
            TrickPlay(1, Card(Suit.CUPS, Rank.FIVE)),
        ),
        current_trick_leader=0,
    )
    assert bot.choose_card(obs) == Card(Suit.CUPS, Rank.ACE)


def test_h03_trumps_with_figure_when_void() -> None:
    bot = HeuristicBot()
    hand = [
        Card(Suit.COINS, Rank.FOUR),
        Card(Suit.COINS, Rank.KNAVE),
        Card(Suit.CLUBS, Rank.SEVEN),
    ]
    obs = _make_obs(
        hand,
        player_id=1,
        trump_suit=Suit.COINS,
        current_trick=(TrickPlay(0, Card(Suit.CUPS, Rank.THREE)),),
        current_trick_leader=0,
    )
    assert bot.choose_card(obs) == Card(Suit.COINS, Rank.KNAVE)


def test_h05_plays_defensive_two_to_block_ace() -> None:
    bot = HeuristicBot()
    hand = [Card(Suit.SWORDS, Rank.TWO), Card(Suit.SWORDS, Rank.FOUR)]
    obs = _make_obs(
        hand,
        player_id=2,
        trump_suit=Suit.COINS,
        current_trick=(
            TrickPlay(0, Card(Suit.SWORDS, Rank.FIVE)),
            TrickPlay(1, Card(Suit.SWORDS, Rank.SIX)),
        ),
        current_trick_leader=0,
    )
    assert bot.choose_card(obs) == Card(Suit.SWORDS, Rank.TWO)

def test_h06_counters_opponent_busso_with_three() -> None:
    bot = HeuristicBot(config=HeuristicConfig(h06_counter_busso_with_three=True))
    hand = [Card(Suit.SWORDS, Rank.THREE), Card(Suit.SWORDS, Rank.FIVE)]
    obs = _make_obs(
        hand,
        player_id=1,
        trump_suit=Suit.COINS,
        current_trick=(TrickPlay(0, Card(Suit.SWORDS, Rank.FOUR)),),
        current_trick_leader=0,
        current_trick_signal=Signal.BUSSO,
    )
    assert bot.choose_card(obs) == Card(Suit.SWORDS, Rank.THREE)


def test_h07_leads_ace_with_non_trump_cricca() -> None:
    bot = HeuristicBot()
    hand = [
        Card(Suit.CUPS, Rank.THREE),
        Card(Suit.CUPS, Rank.TWO),
        Card(Suit.CUPS, Rank.ACE),
        Card(Suit.SWORDS, Rank.FIVE),
    ]
    obs = _make_obs(hand, player_id=0, trump_caller=1, trump_suit=Suit.COINS)
    assert bot.choose_card(obs) == Card(Suit.CUPS, Rank.ACE)


def test_heuristic_decisively_beats_random_bot() -> None:
    summary = run_hand_benchmark(
        "heuristic", "random", num_deals=60, duplicate=True, workers=1, seed=123
    )
    assert summary.point_diff > 1.0
    assert summary.team_a_hand_win_rate > 0.65