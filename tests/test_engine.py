"""Integration tests for full hands, matches, Cricca, Call-out, and JSON logs."""

from pathlib import Path

from openbecca.agents import Agent, RandomBot
from openbecca.cards import ALL_CARDS, Card, Rank, Suit
from openbecca.engine import GameEngine
from openbecca.state import Observation


class ScriptedAgent(Agent):
    """Test helper agent with configurable trump, Cricca, and Call-out behavior."""

    def __init__(
        self,
        name: str,
        preferred_trump: Suit = Suit.COINS,
        declare_cricca: bool = True,
        always_callout: bool = False,
        enforce_busso_constraints: bool = True,
    ) -> None:
        super().__init__(name=name, enforce_busso_constraints=enforce_busso_constraints)
        self.preferred_trump = preferred_trump
        self.declare_cricca = declare_cricca
        self.always_callout = always_callout

    def choose_trump(self, obs: Observation) -> Suit:
        return self.preferred_trump

    def decide_declare_cricca(self, obs: Observation) -> bool:
        return self.declare_cricca

    def decide_callout(self, obs: Observation) -> bool:
        return self.always_callout

    def choose_card(self, obs: Observation) -> Card:
        return obs.legal_cards[0]


def test_single_hand_scores_always_sum_to_11_or_14() -> None:
    bots = [RandomBot(name=f"Bot{i}", seed=100 + i) for i in range(4)]
    engine = GameEngine(bots, seed=42)

    for _ in range(30):
        res = engine.play_hand(allow_callout=False)
        assert len(res.tricks) == 10
        expected_total = 14 if res.cricca_declared_by is not None else 11
        assert sum(res.hand_points) == expected_total


def test_immediate_callout_after_cricca_at_39_points() -> None:
    # Build deterministic hands where player 0 has Coins Cricca + Quattro di Denari
    p0_cricca = [
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.THREE),
        Card(Suit.COINS, Rank.FOUR),
    ]
    remaining = [c for c in ALL_CARDS if c not in p0_cricca]
    hands = [
        p0_cricca + remaining[:6],
        remaining[6:16],
        remaining[16:26],
        remaining[26:36],
    ]

    agents = [
        ScriptedAgent("P0", preferred_trump=Suit.COINS, declare_cricca=True, always_callout=True),
        ScriptedAgent("P1"),
        ScriptedAgent("P2"),
        ScriptedAgent("P3"),
    ]
    engine = GameEngine(agents, target_score=41)
    res = engine.play_hand(match_scores=(39, 35), allow_callout=True, dealt_hands=hands)

    assert res.cricca_declared_by == 0
    assert len(res.tricks) == 0  # Hand stopped immediately before Trick 1
    assert res.callout is not None
    assert res.callout.succeeded is True
    assert res.callout.winning_team == 0
    assert res.callout.final_match_scores == (42, 35)


def test_out_of_hand_cricca_awards_3_points_to_holder_team() -> None:
    # Player 0 has Quattro di Denari and calls Coins; Player 1 holds Coins Cricca!
    p0_fixed = [Card(Suit.COINS, Rank.FOUR)]
    p1_cricca = [
        Card(Suit.COINS, Rank.ACE),
        Card(Suit.COINS, Rank.TWO),
        Card(Suit.COINS, Rank.THREE),
    ]
    remaining = [c for c in ALL_CARDS if c not in p0_fixed and c not in p1_cricca]
    hands = [
        p0_fixed + remaining[:9],
        p1_cricca + remaining[9:16],
        remaining[16:26],
        remaining[26:36],
    ]

    agents = [ScriptedAgent(f"P{i}", preferred_trump=Suit.COINS) for i in range(4)]
    engine = GameEngine(agents)
    res = engine.play_hand(allow_callout=False, dealt_hands=hands)

    assert res.trump_caller == 0
    assert res.cricca_declared_by == 1
    assert sum(res.hand_points) == 14


def test_wrong_callout_forfeits_match_41_to_0() -> None:
    agents = [
        ScriptedAgent("P0", always_callout=True),  # Calls out with 0 points!
        ScriptedAgent("P1"),
        ScriptedAgent("P2"),
        ScriptedAgent("P3"),
    ]
    engine = GameEngine(agents, target_score=41, seed=7)
    match = engine.play_match()

    assert match.ended_by_callout is True
    assert match.winner_team == 1
    assert match.final_scores == (0, 41)


def test_full_match_and_json_export(tmp_path: Path) -> None:
    bots = [RandomBot(name=f"Bot{i}", seed=200 + i) for i in range(4)]
    engine = GameEngine(bots, target_score=41, seed=99)
    match = engine.play_match()

    assert match.winner_team in (0, 1)
    assert max(match.final_scores) >= 41
    assert len(match.hands) >= 4

    out_file = match.save_json(tmp_path / "replay.json")
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "winner_team" in content