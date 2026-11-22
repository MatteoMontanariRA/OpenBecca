"""OpenBecca local web interface built with Streamlit.

Provides three modes:
1. Play (1 Human vs 3 AI) on a Romagnole card table with instant clickable cards,
   step-by-step bot plays (configurable ms delay), and AlphaZero/Stockfish hints.
2. Watch 4 AI (spectator replay with all 4 Romagnole hands revealed and
   step-by-step animated play at configurable ms delay).
3. Stockfish Studio (manual position setup & expected-value move ranking).
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from openbecca.agents import Agent, HeuristicBot, ISMCTSBot, RandomBot
from openbecca.agents.ismcts_bot import MoveStats
from openbecca.analysis.evaluator import (
    build_custom_observation,
    evaluate_position,
)
from openbecca.cards import ALL_CARDS, Card, Deck, Rank, Suit
from openbecca.engine import GameEngine
from openbecca.rules import (
    CRICCA_BONUS_THIRDS,
    DEFAULT_TARGET_SCORE,
    TrickPlay,
    determine_trick_winner,
    evaluate_callout,
    find_four_of_coins_holder,
    get_legal_cards,
    has_cricca,
    next_player,
    partner_of,
    team_of,
    thirds_to_points,
    trick_points_thirds,
)
from openbecca.signals import Signal, determine_lead_signal
from openbecca.state import HandResult, HandState, MatchResult, Observation, TrickRecord

SUIT_ICONS: dict[Suit, str] = {
    Suit.COINS: "🪙",
    Suit.CUPS: "🏆",
    Suit.SWORDS: "⚔️",
    Suit.CLUBS: "🪵",
}

SUIT_TERMS: dict[Suit, str] = {
    Suit.COINS: "denara",
    Suit.CUPS: "coppe",
    Suit.SWORDS: "spade",
    Suit.CLUBS: "bastoni",
}

RANK_EN: dict[Rank, str] = {
    Rank.ACE: "Ace",
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.KNAVE: "Jack",
    Rank.KNIGHT: "Knight",
    Rank.KING: "King",
}

CHECKPOINT_DIR = Path("checkpoints")
PREFERRED_CHECKPOINTS = ("az_gen1_sharp.pt", "az_gen1.pt", "ppo_best.pt", "ppo_latest.pt")

TABLE_COMPONENT_DIR = Path(__file__).resolve().parent / "assets"
CARDS_DIR = TABLE_COMPONENT_DIR / "cards"
_becca_table = components.declare_component("becca_table", path=str(TABLE_COMPONENT_DIR))


def card_id(card: Card) -> str:
    return f"{card.suit.name}_{card.rank.name}"


def card_from_id(cid: str) -> Card:
    suit_name, rank_name = cid.split("_", 1)
    return Card(Suit[suit_name], Rank[rank_name])


def sort_hand_romagnole(cards: list[Card]) -> list[Card]:
    """Order denara, coppe, spade, bastoni using OpenBecca's canonical sort_key."""
    return sorted(cards, key=lambda c: c.sort_key())


def card_display_name(card: Card) -> str:
    """Format card name in English with Romagnolo suit (e.g. '7 of denara')."""
    return f"{RANK_EN[card.rank]} of {SUIT_TERMS[card.suit]}"


def card_badge(card: Card) -> str:
    return f"{SUIT_ICONS[card.suit]} {card_display_name(card)}"


def suit_badge(suit: Suit) -> str:
    return f"{SUIT_ICONS[suit]} {SUIT_TERMS[suit]}"


def signal_label(sig: Signal) -> str:
    if sig == Signal.NONE:
        return "—"
    return sig.value.lower()


def count_downloaded_cards() -> int:
    if not CARDS_DIR.exists():
        return 0
    return sum(1 for c in ALL_CARDS if (CARDS_DIR / f"{card_id(c)}.png").exists())


def discover_checkpoints() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    found = [p for p in CHECKPOINT_DIR.glob("*.pt") if p.is_file()]
    pref_rank = {name: i for i, name in enumerate(PREFERRED_CHECKPOINTS)}
    found.sort(key=lambda p: (pref_rank.get(p.name, 99), p.name))
    return [str(p) for p in found]


@st.cache_resource
def load_cached_neural_model(checkpoint_path: str) -> Any:
    import torch
    from openbecca.rl.ppo_bot import PPOBot

    torch.set_num_threads(2)
    return PPOBot.load_model(checkpoint_path)


def available_bot_types(has_checkpoint: bool) -> list[str]:
    if has_checkpoint:
        return ["AlphaZero", "ISMCTSBot", "PPOBot", "HeuristicBot", "RandomBot"]
    return ["ISMCTSBot", "HeuristicBot", "RandomBot"]


def create_bot(
    bot_type: str,
    seat: int,
    simulations: int,
    seed: int | None = None,
    *,
    checkpoint_path: str | None = None,
    rollout_weight: float = 1.0,
) -> Agent:
    if bot_type == "RandomBot":
        return RandomBot(name=f"Random-{seat}", seed=seed)
    if bot_type == "HeuristicBot":
        return HeuristicBot(name=f"Heuristic-{seat}")
    if bot_type == "ISMCTSBot":
        return ISMCTSBot(
            name=f"ISMCTS-{seat}",
            simulations=simulations,
            trump_simulations=max(16, simulations // 4),
            seed=seed,
        )
    if bot_type == "PPOBot":
        from openbecca.rl.ppo_bot import PPOBot

        model = load_cached_neural_model(checkpoint_path) if checkpoint_path else None
        return PPOBot(name=f"PPO-{seat}", model=model, deterministic=True)
    if bot_type == "AlphaZero":
        from openbecca.rl.neural_mcts import NeuralISMCTSBot

        model = load_cached_neural_model(checkpoint_path) if checkpoint_path else None
        return NeuralISMCTSBot(
            name=f"AlphaZero-{seat}",
            model=model,
            simulations=simulations,
            trump_simulations=max(16, simulations // 4),
            rollout_weight=rollout_weight,
            seed=seed,
        )
    raise ValueError(f"Unknown bot type: {bot_type}")


def render_table_component(
    *,
    hands: dict[int, list[Card]],
    hand_visible: dict[int, bool],
    seat_names: dict[int, str],
    legal_cards: list[Card],
    interactive: bool,
    active_player: int | None,
    trick_plays: list[TrickPlay] | tuple[TrickPlay, ...],
    trick_leader: int | None,
    trick_signal: Signal,
    trick_winner: int | None,
    trump_suit: Suit | None,
    trick_label: str,
    center_banner: str = "",
    hub_actions: list[dict[str, Any]] | None = None,
    turn_token: str = "",
    key: str = "becca_table_main",
) -> dict[str, Any] | None:
    """Serialize game state and invoke the visual Romagnole card table component."""
    hands_payload: dict[str, list[dict[str, str]]] = {}
    counts_payload: dict[str, int] = {}
    for pid in range(4):
        ordered = sort_hand_romagnole(hands[pid])
        counts_payload[str(pid)] = len(ordered)
        if hand_visible[pid]:
            hands_payload[str(pid)] = [
                {"id": card_id(c), "name": card_display_name(c)} for c in ordered
            ]
        else:
            hands_payload[str(pid)] = []

    plays_payload = [
        {"player": pl.player, "id": card_id(pl.card), "name": card_display_name(pl.card)}
        for pl in trick_plays
    ]
    trump_label = f"Briscola: {suit_badge(trump_suit)}" if trump_suit else "Choosing briscola..."

    return _becca_table(
        hands=hands_payload,
        card_counts=counts_payload,
        hand_visible={str(k): v for k, v in hand_visible.items()},
        seat_names={str(k): v for k, v in seat_names.items()},
        legal_ids=[card_id(c) for c in legal_cards],
        interactive=interactive,
        active_player=active_player,
        trick_plays=plays_payload,
        trick_leader=trick_leader,
        trick_signal=trick_signal.value.lower() if trick_signal != Signal.NONE else "",
        trick_winner=trick_winner,
        trump_label=trump_label,
        trick_label=trick_label,
        center_banner=center_banner,
        hub_actions=hub_actions or [],
        turn_token=turn_token,
        key=key,
        default=None,
    )


def run_position_analysis(
    obs: Observation,
    *,
    engine: str,
    simulations: int,
    checkpoint_path: str | None,
    rollout_weight: float,
) -> dict[str, Any]:
    if engine == "AlphaZero" and checkpoint_path is not None:
        from openbecca.rl.neural_mcts import NeuralISMCTSBot
        from openbecca.rl.ppo_bot import PPOBot

        model = load_cached_neural_model(checkpoint_path)
        t0 = time.perf_counter()
        nn_priors, _ = PPOBot(model=model).predict_move_probabilities(obs)
        bot = NeuralISMCTSBot(
            model=model,
            simulations=simulations,
            rollout_weight=rollout_weight,
        )
        n_legal = max(1, len(obs.legal_cards))
        min_v = max(1, min(4, simulations // (n_legal * 3)))
        stats: list[MoveStats] = bot.analyze_moves(
            obs, simulations=simulations, min_visits_per_move=min_v
        )
        elapsed = time.perf_counter() - t0
        best_margin = stats[0].expected_margin

        rows = [
            {
                "Rank": idx,
                "Card": card_badge(s.card),
                "EV Pts": round(s.expected_points, 2),
                "Opp EV": round(s.expected_opp_points, 2),
                "Margin": round(s.expected_margin, 2),
                "Loss vs Best": round(s.expected_margin - best_margin, 2),
                "Win %": f"{s.win_probability * 100:.1f}%",
                "NN Prior %": f"{nn_priors.get(s.card, 0.0) * 100:.1f}%",
                "Sims": s.visits,
            }
            for idx, s in enumerate(stats, start=1)
        ]
        return {
            "engine_label": f"AlphaZero ({Path(checkpoint_path).name}, w={rollout_weight:.1f})",
            "total_simulations": simulations,
            "elapsed_seconds": elapsed,
            "trump_suit": obs.trump_suit,
            "trick_number": obs.current_trick_number,
            "best_card": stats[0].card,
            "best_ev": stats[0].expected_points,
            "best_margin": stats[0].expected_margin,
            "best_win": stats[0].win_probability,
            "rows": rows,
            "chart_data": {card_display_name(s.card): round(s.expected_margin, 2) for s in stats},
        }

    analysis = evaluate_position(obs, simulations=simulations)
    best = analysis.best_move
    rows = [
        {
            "Rank": idx,
            "Card": card_badge(ev.card),
            "EV Pts": round(ev.expected_points, 2),
            "Opp EV": round(ev.expected_opp_points, 2),
            "Margin": round(ev.expected_margin, 2),
            "Loss vs Best": round(ev.ev_loss_vs_best, 2),
            "Win %": f"{ev.win_probability * 100:.1f}%",
            "Sims": ev.visits,
        }
        for idx, ev in enumerate(analysis.evaluations, start=1)
    ]
    return {
        "engine_label": "ISMCTSBot (Classic)",
        "total_simulations": analysis.total_simulations,
        "elapsed_seconds": analysis.elapsed_seconds,
        "trump_suit": analysis.trump_suit,
        "trick_number": analysis.trick_number,
        "best_card": best.card,
        "best_ev": best.expected_points,
        "best_margin": best.expected_margin,
        "best_win": best.win_probability,
        "rows": rows,
        "chart_data": {
            card_display_name(ev.card): round(ev.expected_margin, 2)
            for ev in analysis.evaluations
        },
    }


def render_analysis_panel(result: dict[str, Any]) -> None:
    st.caption(
        f"Engine: **{result['engine_label']}** | "
        f"{result['total_simulations']} simulations in {result['elapsed_seconds']:.2f}s | "
        f"Briscola: {suit_badge(result['trump_suit'])} | Trick {result['trick_number']}/10"
    )
    st.success(
        f"Best move: **{card_badge(result['best_card'])}** "
        f"(Expected: **{result['best_ev']:.2f} pts**, "
        f"Margin: **{result['best_margin']:+.2f}**, "
        f"Win%: **{result['best_win'] * 100:.1f}%**)"
    )
    st.dataframe(result["rows"], use_container_width=True, hide_index=True)
    st.bar_chart(result["chart_data"], height=220)


# ======================================================================
# MODE 1: PLAY (1 HUMAN + 3 AI)
# ======================================================================
def _init_play_match(
    partner_type: str,
    opp_type: str,
    ai_sims: int,
    target_score: int,
    seed: int | None,
    checkpoint_path: str | None,
    rollout_weight: float,
) -> None:
    rng = random.Random(seed)
    agents: dict[int, Agent] = {
        1: create_bot(
            opp_type,
            1,
            ai_sims,
            seed=rng.randint(1, 100_000),
            checkpoint_path=checkpoint_path,
            rollout_weight=rollout_weight,
        ),
        2: create_bot(
            partner_type,
            2,
            ai_sims,
            seed=rng.randint(1, 100_000),
            checkpoint_path=checkpoint_path,
            rollout_weight=rollout_weight,
        ),
        3: create_bot(
            opp_type,
            3,
            ai_sims,
            seed=rng.randint(1, 100_000),
            checkpoint_path=checkpoint_path,
            rollout_weight=rollout_weight,
        ),
    }
    st.session_state.play = {
        "agents": agents,
        "match_cfg": {
            "partner_type": partner_type,
            "opp_type": opp_type,
            "ai_sims": ai_sims,
            "target_score": target_score,
            "checkpoint_path": checkpoint_path,
            "rollout_weight": rollout_weight,
        },
        "seat_names": {
            0: "You (P0 · Team 0)",
            1: f"P1 · {opp_type} (T1)",
            2: f"P2 · {partner_type} (T0)",
            3: f"P3 · {opp_type} (T1)",
        },
        "deck": Deck(seed=seed),
        "target_score": target_score,
        "match_scores": [0, 0],
        "hand_number": 0,
        "next_caller": None,
        "in_sudden_death": False,
        "sudden_death_played": False,
        "hands_log": [],
        "match_winner": None,
        "ended_by_callout": False,
        "state": None,
        "phase": "INIT",
        "last_trick": None,
        "cricca_prompt_player": None,
        "status_banner": "",
        "hint": None,
        "last_event_seq": None,
    }
    _start_new_hand()


def _start_new_hand() -> None:
    p = st.session_state.play
    p["hand_number"] += 1
    p["hint"] = None
    p["last_trick"] = None
    p["status_banner"] = ""

    hands = p["deck"].deal()
    caller = (
        p["next_caller"]
        if p["next_caller"] is not None
        else find_four_of_coins_holder(hands)
    )
    init_counts = [
        {s: Counter(c.suit for c in h).get(s, 0) for s in Suit} for h in hands
    ]
    state = HandState(
        hand_number=p["hand_number"],
        trump_caller=caller,
        hands=hands,
        initial_suit_counts=init_counts,
        match_scores=(p["match_scores"][0], p["match_scores"][1]),
        target_score=p["target_score"],
    )
    state.current_trick_leader = caller
    p["state"] = state

    if caller == 0:
        p["phase"] = "HUMAN_TRUMP"
        p["status_banner"] = (
            "You hold 4 of denara! Choose the briscola suit."
            if p["hand_number"] == 1
            else "It is your turn to choose the briscola suit."
        )
    else:
        obs = state.build_observation(caller)
        chosen = p["agents"][caller].choose_trump(obs)
        state.trump_suit = chosen
        p["status_banner"] = f"Player {caller} chose briscola: {suit_badge(chosen)}."
        _run_cricca_phase()


def _on_choose_trump(suit: Suit) -> None:
    p = st.session_state.play
    if p["phase"] != "HUMAN_TRUMP":
        return
    state: HandState = p["state"]
    state.trump_suit = suit
    p["status_banner"] = f"You chose briscola: {suit_badge(suit)}."
    _run_cricca_phase()


def _on_decide_cricca(declare: bool) -> None:
    p = st.session_state.play
    if p["phase"] != "HUMAN_CRICCA":
        return
    state: HandState = p["state"]
    if declare:
        state.cricca_declared_by = 0
        state.hand_thirds[0] += CRICCA_BONUS_THIRDS
        p["status_banner"] = "You declared cricca (+3 pts)!"
    else:
        p["status_banner"] = "You chose not to declare cricca."
    _after_cricca_phase()


def _on_next_trick() -> None:
    p = st.session_state.play
    if p["phase"] != "TRICK_COMPLETE":
        return
    _continue_after_trick_pause()


def _on_deal_next_hand() -> None:
    p = st.session_state.play
    if p["phase"] != "HAND_OVER":
        return
    _start_new_hand()


def _on_callout_human() -> None:
    _resolve_callout(calling_team=0)


def _on_restart_match() -> None:
    cfg = st.session_state.play["match_cfg"]
    _init_play_match(
        cfg["partner_type"],
        cfg["opp_type"],
        cfg["ai_sims"],
        cfg["target_score"],
        None,
        cfg["checkpoint_path"],
        cfg["rollout_weight"],
    )


def _run_cricca_phase() -> None:
    p = st.session_state.play
    state: HandState = p["state"]
    assert state.trump_suit is not None

    for offset in range(4):
        pid = (state.trump_caller + offset) % 4
        if has_cricca(state.hands[pid], state.trump_suit):
            if pid == 0:
                p["phase"] = "HUMAN_CRICCA"
                p["cricca_prompt_player"] = 0
                return
            obs = state.build_observation(pid)
            if p["agents"][pid].decide_declare_cricca(obs):
                state.cricca_declared_by = pid
                state.hand_thirds[team_of(pid)] += CRICCA_BONUS_THIRDS
                label = "in-hand" if pid == state.trump_caller else "out-of-hand"
                p["status_banner"] += f" Player {pid} declared {label} cricca (+3 pts)!"
            break

    _after_cricca_phase()


def _after_cricca_phase() -> None:
    p = st.session_state.play
    state: HandState = p["state"]

    if not p["in_sudden_death"]:
        for offset in range(4):
            pid = (state.trump_caller + offset) % 4
            if pid == 0:
                continue
            obs = state.build_observation(pid)
            if p["agents"][pid].decide_callout(obs):
                _resolve_callout(team_of(pid))
                return

    p["phase"] = "PLAY_TRICK"


def _resolve_callout(calling_team: int) -> None:
    p = st.session_state.play
    state: HandState = p["state"]
    outcome = evaluate_callout(
        calling_team=calling_team,
        match_scores=(p["match_scores"][0], p["match_scores"][1]),
        hand_thirds=(state.hand_thirds[0], state.hand_thirds[1]),
        target_score=p["target_score"],
    )
    assert state.trump_suit is not None
    hand_res = HandResult(
        hand_number=state.hand_number,
        trump_caller=state.trump_caller,
        trump_suit=state.trump_suit,
        cricca_declared_by=state.cricca_declared_by,
        tricks=list(state.completed_tricks),
        hand_thirds=(state.hand_thirds[0], state.hand_thirds[1]),
        hand_points=state.current_hand_points(),
        callout=outcome,
    )
    p["hands_log"].append(hand_res)
    p["match_scores"] = list(outcome.final_match_scores)
    p["match_winner"] = outcome.winning_team
    p["ended_by_callout"] = True
    p["phase"] = "MATCH_OVER"

    who = "Your team (Team 0)" if calling_team == 0 else "Opponents (Team 1)"
    if outcome.succeeded:
        p["status_banner"] = (
            f"{who} called out with {outcome.verified_total} pts — VALID WIN!"
        )
    else:
        p["status_banner"] = (
            f"{who} called out with only {outcome.verified_total} pts (< {p['target_score']}) "
            f"— FORFEIT ({outcome.final_match_scores[0]}–{outcome.final_match_scores[1]})!"
        )


def _current_player(state: HandState) -> int:
    return (state.current_trick_leader + len(state.current_trick)) % 4


def _legal_for_player(state: HandState, pid: int, *, enforce_busso: bool) -> list[Card]:
    assert state.trump_suit is not None
    is_lead = len(state.current_trick) == 0
    is_first_cricca_lead = (
        is_lead
        and len(state.completed_tricks) == 0
        and state.cricca_declared_by == pid
        and pid == state.trump_caller
    )
    return get_legal_cards(
        hand=state.hands[pid],
        current_trick=state.current_trick,
        trump_suit=state.trump_suit,
        is_first_trick_cricca_lead=is_first_cricca_lead,
        enforce_busso_constraints=enforce_busso,
        current_trick_signal=state.current_trick_signal,
        current_trick_leader=state.current_trick_leader,
        player_id=pid,
        busso_return_suit=state.pending_busso_return_suit if is_lead else None,
    )


def _commit_card_play(pid: int, card: Card) -> None:
    p = st.session_state.play
    state: HandState = p["state"]
    assert state.trump_suit is not None

    if len(state.current_trick) == 0:
        state.current_trick_signal = determine_lead_signal(
            played_card=card,
            hand_before_play=state.hands[pid],
            initial_suit_count=state.initial_suit_counts[pid][card.suit],
            played_cards_history=state.played_cards,
        )

    state.hands[pid].remove(card)
    state.current_trick.append(TrickPlay(player=pid, card=card))
    state.played_cards.add(card)

    if len(state.current_trick) == 4:
        trick_idx = len(state.completed_tricks) + 1
        winner = determine_trick_winner(state.current_trick, state.trump_suit)
        pts_thirds = trick_points_thirds(
            state.current_trick, is_last_trick=(trick_idx == 10)
        )
        state.hand_thirds[team_of(winner)] += pts_thirds

        led_suit = state.current_trick[0].card.suit
        record = TrickRecord(
            trick_number=trick_idx,
            leader=state.current_trick_leader,
            signal=state.current_trick_signal,
            plays=tuple(state.current_trick),
            winner=winner,
            points_thirds=pts_thirds,
        )
        state.completed_tricks.append(record)
        p["last_trick"] = record

        if state.current_trick_signal == Signal.BUSSO and winner == partner_of(
            state.current_trick_leader
        ):
            state.pending_busso_return_suit = led_suit
        else:
            state.pending_busso_return_suit = None

        state.current_trick = []
        state.current_trick_signal = Signal.NONE
        state.current_trick_leader = winner
        p["hint"] = None

        if trick_idx == 10:
            _finalize_completed_hand()
        else:
            p["phase"] = "TRICK_COMPLETE"


def _step_one_bot(bot_delay_ms: int) -> None:
    """Play strictly ONE bot card with user-configured ms pacing, then trigger rerun."""
    p = st.session_state.play
    state: HandState = p["state"]
    pid = _current_player(state)
    if pid == 0:
        return

    t0 = time.perf_counter()
    agent = p["agents"][pid]
    legal = _legal_for_player(state, pid, enforce_busso=agent.enforce_busso_constraints)
    obs = state.build_observation(pid, legal_cards=tuple(legal))
    chosen = agent.choose_card(obs)
    compute_s = time.perf_counter() - t0

    remaining_s = max(0.05, (bot_delay_ms / 1000.0) - compute_s)
    time.sleep(remaining_s)

    _commit_card_play(pid, chosen)
    st.rerun()


def _continue_after_trick_pause() -> None:
    p = st.session_state.play
    state: HandState = p["state"]

    if not p["in_sudden_death"]:
        for offset in range(4):
            pid = (state.current_trick_leader + offset) % 4
            if pid == 0:
                continue
            obs = state.build_observation(pid)
            if p["agents"][pid].decide_callout(obs):
                _resolve_callout(team_of(pid))
                return

    p["phase"] = "PLAY_TRICK"


def _finalize_completed_hand() -> None:
    p = st.session_state.play
    state: HandState = p["state"]
    assert state.trump_suit is not None

    hand_pts = state.current_hand_points()
    hand_res = HandResult(
        hand_number=state.hand_number,
        trump_caller=state.trump_caller,
        trump_suit=state.trump_suit,
        cricca_declared_by=state.cricca_declared_by,
        tricks=list(state.completed_tricks),
        hand_thirds=(state.hand_thirds[0], state.hand_thirds[1]),
        hand_points=hand_pts,
        callout=None,
    )
    p["hands_log"].append(hand_res)
    p["next_caller"] = next_player(state.trump_caller)
    p["match_scores"][0] += hand_pts[0]
    p["match_scores"][1] += hand_pts[1]

    if p["in_sudden_death"]:
        if hand_pts[0] != hand_pts[1]:
            p["match_winner"] = 0 if hand_pts[0] > hand_pts[1] else 1
            p["phase"] = "MATCH_OVER"
            return
        p["phase"] = "HAND_OVER"
        return

    s0, s1 = p["match_scores"]
    if s0 >= p["target_score"] or s1 >= p["target_score"]:
        if s0 > s1:
            p["match_winner"] = 0
            p["phase"] = "MATCH_OVER"
        elif s1 > s0:
            p["match_winner"] = 1
            p["phase"] = "MATCH_OVER"
        else:
            p["in_sudden_death"] = True
            p["sudden_death_played"] = True
            p["status_banner"] = "Exact tie at target score! Playing a sudden-death hand."
            p["phase"] = "HAND_OVER"
    else:
        p["phase"] = "HAND_OVER"


def _consume_table_event(event: Any, turn_token: str) -> dict[str, Any] | None:
    """Validate and deduplicate a click event coming from the card table iframe."""
    p = st.session_state.play
    if (
        isinstance(event, dict)
        and event.get("seq") is not None
        and event.get("seq") != p["last_event_seq"]
        and event.get("turn_token") == turn_token
    ):
        p["last_event_seq"] = event.get("seq")
        return event
    return None


def render_play_mode(
    checkpoint_path: str | None, rollout_weight: float, bot_delay_ms: int
) -> None:
    st.header("Play Beccacino (1 Human + 3 AI)")
    bot_options = available_bot_types(checkpoint_path is not None)

    with st.sidebar:
        st.subheader("Match Settings")
        partner_type = st.selectbox("Partner AI (Player 2)", bot_options, index=0)
        opp_type = st.selectbox("Opponents AI (Players 1 & 3)", bot_options, index=0)
        ai_sims = st.slider(
            "MCTS simulations/move", min_value=40, max_value=400, value=120, step=20
        )
        hint_engine_opts = (
            ["AlphaZero", "ISMCTSBot"] if checkpoint_path is not None else ["ISMCTSBot"]
        )
        hint_engine = st.selectbox("Hint Analyzer Engine", hint_engine_opts, index=0)
        hint_sims = st.slider(
            "Hint simulations", min_value=100, max_value=1000, value=300, step=50
        )
        target_score = st.number_input(
            "Target Score", min_value=11, max_value=61, value=DEFAULT_TARGET_SCORE, step=10
        )
        seed_input = st.text_input("Optional Seed (blank = random)", value="")
        if st.button("Start New Match", type="primary", use_container_width=True):
            parsed_seed = int(seed_input.strip()) if seed_input.strip().isdigit() else None
            _init_play_match(
                partner_type,
                opp_type,
                ai_sims,
                int(target_score),
                parsed_seed,
                checkpoint_path,
                rollout_weight,
            )
            st.rerun()

    if "play" not in st.session_state:
        _init_play_match(
            partner_type,
            opp_type,
            ai_sims,
            int(target_score),
            None,
            checkpoint_path,
            rollout_weight,
        )

    p = st.session_state.play
    state: HandState = p["state"]

    # Scoreboard
    c1, c2, c3, c4 = st.columns(4)
    t0_hand_pts = thirds_to_points(state.hand_thirds[0])
    t1_hand_pts = thirds_to_points(state.hand_thirds[1])
    c1.metric(
        "Team 0 (You + P2) Match",
        f"{p['match_scores'][0]} / {p['target_score']}",
        delta=f"+{t0_hand_pts} ({state.hand_thirds[0] % 3}/3) this hand",
    )
    c2.metric(
        "Team 1 (P1 + P3) Match",
        f"{p['match_scores'][1]} / {p['target_score']}",
        delta=f"+{t1_hand_pts} ({state.hand_thirds[1] % 3}/3) this hand",
    )
    c3.metric(
        "Briscola",
        suit_badge(state.trump_suit) if state.trump_suit else "Choosing...",
        delta=f"Caller: Player {state.trump_caller}",
    )
    c4.metric(
        "Hand / Trick",
        f"Hand {state.hand_number}",
        delta=f"Trick {min(10, len(state.completed_tricks) + 1)} / 10",
    )

    if state.cricca_declared_by is not None:
        st.info(f"Cricca declared by **Player {state.cricca_declared_by}** (+3 pts).")
    if p["status_banner"]:
        st.info(p["status_banner"])

    hands_dict = {pid: list(state.hands[pid]) for pid in range(4)}
    visibility = {0: True, 1: False, 2: False, 3: False}

    # ------------------------------------------------------------------
    # Phase 1: Human chooses briscola (ONLY inside the table center hub)
    # ------------------------------------------------------------------
    if p["phase"] == "HUMAN_TRUMP":
        turn_token = f"trump-{state.hand_number}"
        hub_actions = [
            {
                "label": suit_badge(suit),
                "action": "choose_trump",
                "payload": {"suit": suit.name},
            }
            for suit in Suit
        ]
        event = render_table_component(
            hands=hands_dict,
            hand_visible=visibility,
            seat_names=p["seat_names"],
            legal_cards=[],
            interactive=False,
            active_player=0,
            trick_plays=[],
            trick_leader=None,
            trick_signal=Signal.NONE,
            trick_winner=None,
            trump_suit=None,
            trick_label=f"Hand {state.hand_number} · Choosing briscola",
            center_banner="Select the briscola suit:",
            hub_actions=hub_actions,
            turn_token=turn_token,
            key="play_table",
        )
        valid = _consume_table_event(event, turn_token)
        if valid and valid.get("action") == "choose_trump":
            _on_choose_trump(Suit[valid["suit"]])
            st.rerun()
        return

    # ------------------------------------------------------------------
    # Phase 2: Human decides cricca (ONLY inside the table center hub)
    # ------------------------------------------------------------------
    if p["phase"] == "HUMAN_CRICCA":
        turn_token = f"cricca-{state.hand_number}"
        event = render_table_component(
            hands=hands_dict,
            hand_visible=visibility,
            seat_names=p["seat_names"],
            legal_cards=[],
            interactive=False,
            active_player=0,
            trick_plays=[],
            trick_leader=None,
            trick_signal=Signal.NONE,
            trick_winner=None,
            trump_suit=state.trump_suit,
            trick_label=f"Hand {state.hand_number} · Cricca!",
            center_banner="You hold briscola cricca (+3 pts)!",
            hub_actions=[
                {"label": "Declare cricca (+3)", "action": "decide_cricca", "payload": {"declare": True}},
                {"label": "Skip", "action": "decide_cricca", "payload": {"declare": False}, "secondary": True},
            ],
            turn_token=turn_token,
            key="play_table",
        )
        valid = _consume_table_event(event, turn_token)
        if valid and valid.get("action") == "decide_cricca":
            _on_decide_cricca(bool(valid.get("declare", True)))
            st.rerun()
        return

    # ------------------------------------------------------------------
    # Phase 3: Active Trick Play (Human click or 1-by-1 Bot play)
    # ------------------------------------------------------------------
    if p["phase"] == "PLAY_TRICK":
        curr_pid = _current_player(state)
        is_human_turn = curr_pid == 0
        legal = _legal_for_player(state, 0, enforce_busso=False) if is_human_turn else []
        turn_token = (
            f"{state.hand_number}-{len(state.completed_tricks)}-"
            f"{len(state.current_trick)}-{curr_pid}"
        )
        banner = (
            "Your turn — click a highlighted card!"
            if is_human_turn
            else f"Player {curr_pid} is playing..."
        )

        event = render_table_component(
            hands=hands_dict,
            hand_visible=visibility,
            seat_names=p["seat_names"],
            legal_cards=legal,
            interactive=is_human_turn,
            active_player=curr_pid,
            trick_plays=state.current_trick,
            trick_leader=state.current_trick_leader if state.current_trick else None,
            trick_signal=state.current_trick_signal,
            trick_winner=None,
            trump_suit=state.trump_suit,
            trick_label=f"Hand {state.hand_number} · Trick {len(state.completed_tricks) + 1}/10",
            center_banner=banner,
            turn_token=turn_token,
            key="play_table",
        )

        if not is_human_turn:
            _step_one_bot(bot_delay_ms)
            return

        valid = _consume_table_event(event, turn_token)
        if valid and valid.get("action") == "play_card":
            chosen = card_from_id(valid["card"])
            if chosen in set(legal):
                _commit_card_play(0, chosen)
                st.rerun()

        if (
            not p["in_sudden_death"]
            and len(state.completed_tricks) == 0
            and len(state.current_trick) == 0
        ):
            st.button(
                "Call Out (before Trick 1)",
                key="early_callout_btn",
                on_click=_on_callout_human,
            )

        col_h1, _ = st.columns([1, 2])
        if col_h1.button(
            f"Analyze Best Move ({hint_engine}, {hint_sims} sims)",
            use_container_width=True,
            key="analyze_hint_btn",
        ):
            obs = state.build_observation(0, legal_cards=tuple(legal))
            with st.spinner(f"Running OpenBecca {hint_engine} search..."):
                p["hint"] = run_position_analysis(
                    obs,
                    engine=hint_engine,
                    simulations=hint_sims,
                    checkpoint_path=checkpoint_path,
                    rollout_weight=rollout_weight,
                )
        if p["hint"] is not None:
            render_analysis_panel(p["hint"])

    # ------------------------------------------------------------------
    # Phase 4: Trick Complete (ONLY inside the table center hub)
    # ------------------------------------------------------------------
    elif p["phase"] == "TRICK_COMPLETE" and p["last_trick"] is not None:
        rec: TrickRecord = p["last_trick"]
        turn_token = f"done-{state.hand_number}-{rec.trick_number}"
        pts_txt = f"+{rec.points_thirds // 3} pts & {rec.points_thirds % 3}/3"

        hub_actions: list[dict[str, Any]] = [
            {"label": "Next Trick ▶", "action": "next_trick"}
        ]
        if not p["in_sudden_death"]:
            hub_actions.append({"label": "Call Out", "action": "callout", "secondary": True})

        event = render_table_component(
            hands=hands_dict,
            hand_visible=visibility,
            seat_names=p["seat_names"],
            legal_cards=[],
            interactive=False,
            active_player=rec.winner,
            trick_plays=rec.plays,
            trick_leader=rec.leader,
            trick_signal=rec.signal,
            trick_winner=rec.winner,
            trump_suit=state.trump_suit,
            trick_label=f"Hand {state.hand_number} · Trick {rec.trick_number}/10 Complete",
            center_banner=f"P{rec.winner} (Team {team_of(rec.winner)}) wins · {pts_txt}",
            hub_actions=hub_actions,
            turn_token=turn_token,
            key="play_table",
        )
        valid = _consume_table_event(event, turn_token)
        if valid:
            if valid.get("action") == "next_trick":
                _on_next_trick()
                st.rerun()
            elif valid.get("action") == "callout":
                _on_callout_human()
                st.rerun()

    # ------------------------------------------------------------------
    # Phase 5: Hand Over (ONLY inside the table center hub)
    # ------------------------------------------------------------------
    elif p["phase"] == "HAND_OVER":
        turn_token = f"handover-{state.hand_number}"
        last_hand: HandResult = p["hands_log"][-1]
        rec = p["last_trick"]

        event = render_table_component(
            hands=hands_dict,
            hand_visible=visibility,
            seat_names=p["seat_names"],
            legal_cards=[],
            interactive=False,
            active_player=rec.winner if rec is not None else None,
            trick_plays=rec.plays if rec is not None else [],
            trick_leader=rec.leader if rec is not None else None,
            trick_signal=rec.signal if rec is not None else Signal.NONE,
            trick_winner=rec.winner if rec is not None else None,
            trump_suit=state.trump_suit,
            trick_label=f"Hand {state.hand_number} · Finished",
            center_banner=f"Team 0 +{last_hand.hand_points[0]} | Team 1 +{last_hand.hand_points[1]}",
            hub_actions=[{"label": "Deal Next Hand ▶", "action": "next_hand"}],
            turn_token=turn_token,
            key="play_table",
        )
        st.success(
            f"Hand {last_hand.hand_number} finished! "
            f"Team 0: **+{last_hand.hand_points[0]} pts** | "
            f"Team 1: **+{last_hand.hand_points[1]} pts**"
        )
        valid = _consume_table_event(event, turn_token)
        if valid and valid.get("action") == "next_hand":
            _on_deal_next_hand()
            st.rerun()

    # ------------------------------------------------------------------
    # Phase 6: Match Over
    # ------------------------------------------------------------------
    elif p["phase"] == "MATCH_OVER":
        winner = p["match_winner"]
        if winner == 0:
            st.balloons()
            st.success(
                f"Match Over — **Team 0 (You + Partner) WINS** "
                f"{p['match_scores'][0]} to {p['match_scores'][1]}!"
            )
        else:
            st.error(
                f"Match Over — **Team 1 (Opponents) WINS** "
                f"{p['match_scores'][1]} to {p['match_scores'][0]}!"
            )

        match_res = MatchResult(
            target_score=p["target_score"],
            winner_team=winner,
            final_scores=(p["match_scores"][0], p["match_scores"][1]),
            hands=p["hands_log"],
            ended_by_callout=p["ended_by_callout"],
            sudden_death_played=p["sudden_death_played"],
        )
        mcol1, mcol2 = st.columns(2)
        mcol1.button(
            "Start New Match ▶",
            type="primary",
            use_container_width=True,
            key="restart_match_btn",
            on_click=_on_restart_match,
        )
        mcol2.download_button(
            "Download Match Replay (JSON)",
            data=json.dumps(match_res.to_dict(), indent=2, ensure_ascii=False),
            file_name="openbecca_match_replay.json",
            mime="application/json",
            use_container_width=True,
        )

    if state.completed_tricks:
        with st.expander(f"Completed Tricks Log ({len(state.completed_tricks)}/10)"):
            rows = []
            for t in state.completed_tricks:
                plays_str = " | ".join(
                    f"P{pl.player}: {card_display_name(pl.card)}" for pl in t.plays
                )
                rows.append(
                    {
                        "Trick": t.trick_number,
                        "Leader": f"P{t.leader}",
                        "Signal": signal_label(t.signal),
                        "Cards Played": plays_str,
                        "Winner": f"P{t.winner} (T{team_of(t.winner)})",
                        "Thirds": t.points_thirds,
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ======================================================================
# MODE 2: WATCH 4 AI (ALL 4 HANDS REVEALED + STEP-BY-STEP PLAY)
# ======================================================================
def _watch_reset() -> None:
    st.session_state.watch["step"] = 0
    st.session_state.watch["playing"] = False


def _watch_prev() -> None:
    w = st.session_state.watch
    w["step"] = max(0, w["step"] - 1)
    w["playing"] = False


def _watch_next(max_step: int) -> None:
    w = st.session_state.watch
    w["step"] = min(max_step, w["step"] + 1)
    w["playing"] = False


def _watch_toggle_play(max_step: int) -> None:
    w = st.session_state.watch
    if w["playing"]:
        w["playing"] = False
    else:
        if w["step"] >= max_step:
            w["step"] = 0
        w["playing"] = True

def _watch_on_slider_change() -> None:
    w = st.session_state.watch
    w["step"] = int(st.session_state["watch_timeline_slider"])
    w["playing"] = False

def _reconstruct_watch_frame(
    dealt: list[list[Card]], res: HandResult, step: int
) -> dict[str, Any]:
    """Reconstruct exact hands, table cards, and running scores after `step` plays (0..40)."""
    hands = {pid: list(dealt[pid]) for pid in range(4)}
    thirds = [0, 0]
    if res.cricca_declared_by is not None:
        thirds[team_of(res.cricca_declared_by)] += CRICCA_BONUS_THIRDS

    if step == 0:
        return {
            "hands": hands,
            "trick_plays": [],
            "trick_leader": None,
            "trick_signal": Signal.NONE,
            "trick_winner": None,
            "active_player": res.tricks[0].leader if res.tricks else None,
            "trick_label": "Initial Deal · Ready (0/40)",
            "banner": "Press ▶ Play or Next to watch card by card",
            "thirds": thirds,
        }

    trick_idx = (step - 1) // 4
    cards_shown = ((step - 1) % 4) + 1

    for ti in range(trick_idx):
        tr = res.tricks[ti]
        for pl in tr.plays:
            hands[pl.player].remove(pl.card)
        thirds[team_of(tr.winner)] += tr.points_thirds

    curr_tr = res.tricks[trick_idx]
    shown_plays = list(curr_tr.plays[:cards_shown])
    for pl in shown_plays:
        hands[pl.player].remove(pl.card)

    trick_winner = None
    if cards_shown == 4:
        thirds[team_of(curr_tr.winner)] += curr_tr.points_thirds
        trick_winner = curr_tr.winner
        banner = (
            f"P{curr_tr.winner} (Team {team_of(curr_tr.winner)}) wins · "
            f"+{curr_tr.points_thirds // 3} pts & {curr_tr.points_thirds % 3}/3"
        )
        active = curr_tr.winner
    else:
        last_pl = shown_plays[-1]
        banner = f"P{last_pl.player} played {card_display_name(last_pl.card)}"
        active = last_pl.player

    return {
        "hands": hands,
        "trick_plays": shown_plays,
        "trick_leader": curr_tr.leader,
        "trick_signal": curr_tr.signal,
        "trick_winner": trick_winner,
        "active_player": active,
        "trick_label": f"Trick {curr_tr.trick_number}/10 · Card {step}/40",
        "banner": banner,
        "thirds": thirds,
    }


def render_watch_mode(
    checkpoint_path: str | None, rollout_weight: float, bot_delay_ms: int
) -> None:
    st.header("Watch 4 AI Simulation (All 4 Hands Revealed)")
    bot_options = available_bot_types(checkpoint_path is not None)

    col1, col2, col3, col4 = st.columns(4)
    team0_type = col1.selectbox("Team 0 (P0 & P2)", bot_options, index=0)
    team1_default = 1 if len(bot_options) > 1 else 0
    team1_type = col2.selectbox("Team 1 (P1 & P3)", bot_options, index=team1_default)
    sims = col3.slider("MCTS Simulations", 40, 300, 120, 20, key="watch_sims")
    seed_val = col4.number_input("Deal Seed", min_value=0, max_value=999_999, value=42)

    if st.button("Simulate Hand", type="primary"):
        deck = Deck(seed=int(seed_val))
        dealt = deck.deal()
        agents = [
            create_bot(
                team0_type,
                0,
                sims,
                seed=int(seed_val) + 1,
                checkpoint_path=checkpoint_path,
                rollout_weight=rollout_weight,
            ),
            create_bot(
                team1_type,
                1,
                sims,
                seed=int(seed_val) + 2,
                checkpoint_path=checkpoint_path,
                rollout_weight=rollout_weight,
            ),
            create_bot(
                team0_type,
                2,
                sims,
                seed=int(seed_val) + 3,
                checkpoint_path=checkpoint_path,
                rollout_weight=rollout_weight,
            ),
            create_bot(
                team1_type,
                3,
                sims,
                seed=int(seed_val) + 4,
                checkpoint_path=checkpoint_path,
                rollout_weight=rollout_weight,
            ),
        ]
        with st.spinner("Simulating 4-AI hand..."):
            res = GameEngine(agents).play_hand(allow_callout=False, dealt_hands=dealt)
        st.session_state.watch = {
            "dealt": dealt,
            "result": res,
            "team0": team0_type,
            "team1": team1_type,
            "step": 0,
            "playing": True,
        }
        st.rerun()

    if "watch" not in st.session_state:
        st.info(
            "Configure both teams above and click **Simulate Hand** to watch all 4 hands "
            "face-up and follow every play card by card."
        )
        return

    w = st.session_state.watch
    res: HandResult = w["result"]
    dealt: list[list[Card]] = w["dealt"]
    max_step = len(res.tricks) * 4

    frame = _reconstruct_watch_frame(dealt, res, w["step"])
    thirds = frame["thirds"]

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"Team 0 ({w['team0']})",
        f"{thirds_to_points(thirds[0])} pts",
        f"{thirds[0] % 3}/3 thirds (Final: {res.hand_points[0]} pts)",
    )
    m2.metric(
        f"Team 1 ({w['team1']})",
        f"{thirds_to_points(thirds[1])} pts",
        f"{thirds[1] % 3}/3 thirds (Final: {res.hand_points[1]} pts)",
    )
    m3.metric(
        "Briscola & Cricca",
        suit_badge(res.trump_suit),
        f"Caller: P{res.trump_caller} | Cricca: {f'P{res.cricca_declared_by}' if res.cricca_declared_by is not None else 'None'}",
    )

    if w["playing"] and w["step"] >= max_step:
        w["playing"] = False

    # Sync slider state BEFORE instantiating the widget (prevents snap-back during auto-play)
    st.session_state["watch_timeline_slider"] = w["step"]

    # Playback controls — single fixed-key button for Play/Pause
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 3])
    b1.button(
        "⏮ Reset",
        use_container_width=True,
        key="watch_reset_btn",
        on_click=_watch_reset,
    )
    b2.button(
        "◀ Prev",
        use_container_width=True,
        disabled=w["step"] <= 0,
        key="watch_prev_btn",
        on_click=_watch_prev,
    )
    b3.button(
        "⏸ Pause" if w["playing"] else "▶ Play",
        type="primary",
        use_container_width=True,
        key="watch_play_toggle",
        on_click=_watch_toggle_play,
        args=(max_step,),
    )
    b4.button(
        "Next ▶",
        use_container_width=True,
        disabled=w["step"] >= max_step,
        key="watch_next_btn",
        on_click=_watch_next,
        args=(max_step,),
    )
    b5.slider(
        "Timeline",
        min_value=0,
        max_value=max_step,
        label_visibility="collapsed",
        key="watch_timeline_slider",
        on_change=_watch_on_slider_change,
    )

    seat_names = {
        0: f"P0 · {w['team0']} (T0)",
        1: f"P1 · {w['team1']} (T1)",
        2: f"P2 · {w['team0']} (T0)",
        3: f"P3 · {w['team1']} (T1)",
    }

    render_table_component(
        hands=frame["hands"],
        hand_visible={0: True, 1: True, 2: True, 3: True},
        seat_names=seat_names,
        legal_cards=[],
        interactive=False,
        active_player=frame["active_player"],
        trick_plays=frame["trick_plays"],
        trick_leader=frame["trick_leader"],
        trick_signal=frame["trick_signal"],
        trick_winner=frame["trick_winner"],
        trump_suit=res.trump_suit,
        trick_label=frame["trick_label"],
        center_banner=frame["banner"],
        turn_token=f"watch-{w['step']}",
        key="watch_table",
    )

    with st.expander("Full Trick-by-Trick Table & JSON Export"):
        rows = []
        for t in res.tricks:
            plays_str = " | ".join(
                f"P{pl.player}: {card_display_name(pl.card)}" for pl in t.plays
            )
            rows.append(
                {
                    "Trick": t.trick_number,
                    "Leader": f"P{t.leader}",
                    "Signal": signal_label(t.signal),
                    "Cards Played": plays_str,
                    "Winner": f"P{t.winner} (Team {team_of(t.winner)})",
                    "Thirds": t.points_thirds,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Hand Log (JSON)",
            data=json.dumps(res.to_dict(), indent=2, ensure_ascii=False),
            file_name="openbecca_ai_hand.json",
            mime="application/json",
        )

    if w["playing"]:
        if w["step"] < max_step:
            time.sleep(max(0.05, bot_delay_ms / 1000.0))
            w["step"] += 1
            st.rerun()
        else:
            w["playing"] = False
            st.rerun()


# ======================================================================
# MODE 3: STOCKFISH STUDIO (CUSTOM POSITION ANALYZER)
# ======================================================================
def render_studio_mode(checkpoint_path: str | None, rollout_weight: float) -> None:
    st.header("Stockfish Studio — Custom Position Analyzer")
    st.write(
        "Set up any table scenario manually and ask **OpenBecca** (AlphaZero or ISMCTSBot) "
        "for the expected points and win probability of every legal move."
    )

    all_sorted = sorted(ALL_CARDS, key=lambda c: c.sort_key())
    label_to_card = {card_badge(c): c for c in all_sorted}
    all_labels = list(label_to_card.keys())

    c1, c2, c3, c4 = st.columns(4)
    trump_suit = c1.selectbox(
        "Briscola Suit", list(Suit), format_func=suit_badge, key="studio_trump"
    )
    player_id = c2.selectbox("Your Seat", [0, 1, 2, 3], index=0, key="studio_seat")
    trump_caller = c3.selectbox("Briscola Caller", [0, 1, 2, 3], index=0, key="studio_caller")
    cricca_choice = c4.selectbox(
        "Cricca Declared By",
        ["None", "Player 0", "Player 1", "Player 2", "Player 3"],
        index=0,
        key="studio_cricca",
    )
    cricca_by = None if cricca_choice == "None" else int(cricca_choice.split()[-1])

    default_hand = [card_badge(c) for c in Deck(seed=42).deal()[0]]
    selected_hand_labels = st.multiselect(
        "1. Your Hand (select 1 to 10 cards)",
        options=all_labels,
        default=default_hand,
        max_selections=10,
        key="studio_hand",
    )
    player_hand = [label_to_card[lbl] for lbl in selected_hand_labels]

    if player_hand and count_downloaded_cards() > 0:
        preview_cards = sort_hand_romagnole(player_hand)
        pcols = st.columns(min(10, len(preview_cards)))
        for idx, card in enumerate(preview_cards):
            img_path = CARDS_DIR / f"{card_id(card)}.png"
            if img_path.exists():
                pcols[idx].image(str(img_path), caption=card_display_name(card), width=72)

    remaining_after_hand = [lbl for lbl in all_labels if lbl not in selected_hand_labels]
    table_labels = st.multiselect(
        "2. Cards Currently on the Table (0 to 3 cards, in order played)",
        options=remaining_after_hand,
        default=[],
        max_selections=3,
        key="studio_table",
    )
    table_cards = [label_to_card[lbl] for lbl in table_labels]

    signal_options = [Signal.NONE, Signal.BUSSO, Signal.VOLO, Signal.STRISCIO]
    lead_signal = Signal.NONE
    if table_cards:
        lead_signal = st.selectbox(
            "Signal Announced by Trick Leader",
            signal_options,
            format_func=lambda s: s.value.lower() if s != Signal.NONE else "(silence)",
            key="studio_signal",
        )

    needed_past = (10 - len(player_hand)) * 4
    remaining_for_past = [
        lbl for lbl in remaining_after_hand if lbl not in table_labels
    ]

    past_cards: list[Card] = []
    auto_fill_past = True
    if needed_past > 0:
        st.markdown(
            f"**3. Past Tricks**: With {len(player_hand)} cards in your hand, "
            f"**{needed_past} cards** must have been played in earlier tricks."
        )
        chosen_past_labels = st.multiselect(
            f"Key cards already played (pick up to {needed_past})",
            options=remaining_for_past,
            default=[],
            max_selections=needed_past,
            key="studio_past",
        )
        past_cards = [label_to_card[lbl] for lbl in chosen_past_labels]
        auto_fill_past = st.checkbox(
            f"Auto-fill remaining {needed_past - len(past_cards)} past cards randomly",
            value=True,
            key="studio_autofill",
        )

    col_e1, col_e2 = st.columns([1, 2])
    engine_opts = ["AlphaZero", "ISMCTSBot"] if checkpoint_path is not None else ["ISMCTSBot"]
    studio_engine = col_e1.selectbox("Analyzer Engine", engine_opts, index=0, key="studio_engine")
    sims = col_e2.slider(
        "Analysis Depth (Simulations)",
        min_value=100,
        max_value=1200,
        value=400,
        step=50,
        key="studio_sims",
    )

    if st.button("Run Stockfish Analysis", type="primary"):
        if not player_hand:
            st.error("Please select at least 1 card in your hand.")
            return

        extra_played = list(past_cards)
        if len(extra_played) < needed_past:
            if auto_fill_past:
                rng = random.Random(42)
                used = set(player_hand) | set(table_cards) | set(extra_played)
                pool = [c for c in ALL_CARDS if c not in used]
                extra_played.extend(rng.sample(pool, needed_past - len(extra_played)))
            else:
                st.error(
                    f"Select exactly {needed_past} past cards or enable auto-fill."
                )
                return

        leader = (player_id - len(table_cards)) % 4
        current_trick = [
            TrickPlay(player=(leader + idx) % 4, card=card)
            for idx, card in enumerate(table_cards)
        ]

        try:
            obs = build_custom_observation(
                player_hand=player_hand,
                trump_suit=trump_suit,
                player_id=player_id,
                trump_caller=trump_caller,
                current_trick=current_trick,
                current_trick_leader=leader,
                current_trick_signal=lead_signal,
                extra_played_cards=extra_played,
                cricca_declared_by=cricca_by,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        with st.spinner(f"Running {sims} {studio_engine} simulations..."):
            result = run_position_analysis(
                obs,
                engine=studio_engine,
                simulations=sims,
                checkpoint_path=checkpoint_path,
                rollout_weight=rollout_weight,
            )
        render_analysis_panel(result)


# ======================================================================
# APP ENTRY POINT
# ======================================================================
def main() -> None:
    st.set_page_config(
        page_title="OpenBecca — Beccacino AI & Analyzer",
        page_icon="🃏",
        layout="wide",
    )

    # Prevent dimming ONLY on the card table iframe (never on Streamlit buttons)
    st.markdown(
        """
        <style>
          [data-testid="stCustomComponentV1"],
          [data-testid="stCustomComponentV1"][data-stale="true"],
          iframe[title="becca_table"] {
            opacity: 1 !important;
            filter: none !important;
            transition: none !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("🃏 OpenBecca")
    st.sidebar.caption("Ravenna Beccacino Engine & AI Analyzer")

    mode = st.sidebar.radio(
        "Mode",
        [
            "Play (1 Human + 3 AI)",
            "Watch 4 AI",
            "Stockfish Studio (Custom Setup)",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Bot Speed (Table)")
    bot_delay_ms = st.sidebar.slider(
        "Delay between bot plays (ms)",
        min_value=50,
        max_value=2000,
        value=600,
        step=50,
        help="Delay in milliseconds between each bot's card play (Play & Watch 4 AI).",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Neural Model (RL / AlphaZero)")
    ckpts = discover_checkpoints()
    if ckpts:
        checkpoint_path: str | None = st.sidebar.selectbox(
            "Checkpoint (.pt)",
            options=ckpts,
            index=0,
            format_func=lambda p: Path(p).name,
        )
        rollout_weight = st.sidebar.slider(
            "AlphaZero rollout weight",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="1.0 = Policy Prior + full rollout (strongest); 0.0 = pure neural value.",
        )
    else:
        checkpoint_path = None
        rollout_weight = 1.0
        st.sidebar.warning("No checkpoints found in `checkpoints/`. Neural bots disabled.")

    if count_downloaded_cards() < 40:
        st.warning(
            "Romagnole card deck not fully downloaded yet "
            f"({count_downloaded_cards()}/40 found). Run:\n\n"
            "`python scripts/download_romagnole.py`"
        )

    if mode == "Play (1 Human + 3 AI)":
        render_play_mode(checkpoint_path, rollout_weight, bot_delay_ms)
    elif mode == "Watch 4 AI":
        render_watch_mode(checkpoint_path, rollout_weight, bot_delay_ms)
    else:
        render_studio_mode(checkpoint_path, rollout_weight)


if __name__ == "__main__":
    main()