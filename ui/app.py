"""OpenBecca local web interface built with Streamlit.

Provides three modes:
1. Play (1 Human vs 3 AI) with live Stockfish-style move hints & Call-out.
2. Watch 4 AI (spectator replay with all 4 hands revealed).
3. Stockfish Studio (manual position setup & expected-value move ranking).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

import streamlit as st

from openbecca.agents import Agent, HeuristicBot, ISMCTSBot, RandomBot
from openbecca.analysis.evaluator import (
    PositionAnalysis,
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

SEAT_LABELS: dict[int, str] = {
    0: "Player 0 (You - Team 0)",
    1: "Player 1 (Right Opponent - Team 1)",
    2: "Player 2 (Partner - Team 0)",
    3: "Player 3 (Left Opponent - Team 1)",
}


def card_badge(card: Card) -> str:
    """Return Italian written card name prefixed with a subtle suit icon."""
    return f"{SUIT_ICONS[card.suit]} {card.italian_name}"


def suit_badge(suit: Suit) -> str:
    return f"{SUIT_ICONS[suit]} {suit.italian_name}"


def create_bot(bot_type: str, seat: int, simulations: int, seed: int | None = None) -> Agent:
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
    raise ValueError(f"Unknown bot type: {bot_type}")


def render_analysis_panel(analysis: PositionAnalysis) -> None:
    """Display Stockfish-style move ranking table and visual margin chart."""
    st.caption(
        f"Analyzed {analysis.total_simulations} simulations in "
        f"{analysis.elapsed_seconds:.2f}s | Trump: {suit_badge(analysis.trump_suit)} | "
        f"Trick {analysis.trick_number}/10"
    )
    best = analysis.best_move
    st.success(
        f"Best move: **{card_badge(best.card)}** "
        f"(Expected: **{best.expected_points:.2f} pts**, "
        f"Margin: **{best.expected_margin:+.2f}**, "
        f"Win%: **{best.win_probability * 100:.1f}%**)"
    )

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
    st.dataframe(rows, use_container_width=True, hide_index=True)

    chart_data = {
        ev.card.italian_name: round(ev.expected_margin, 2)
        for ev in analysis.evaluations
    }
    st.bar_chart(chart_data, height=220)


# ======================================================================
# MODE 1: PLAY (1 HUMAN + 3 AI)
# ======================================================================
def _init_play_match(
    partner_type: str,
    opp_type: str,
    ai_sims: int,
    target_score: int,
    seed: int | None,
) -> None:
    rng = random.Random(seed)
    agents: dict[int, Agent] = {
        1: create_bot(opp_type, 1, ai_sims, seed=rng.randint(1, 100_000)),
        2: create_bot(partner_type, 2, ai_sims, seed=rng.randint(1, 100_000)),
        3: create_bot(opp_type, 3, ai_sims, seed=rng.randint(1, 100_000)),
    }
    st.session_state.play = {
        "agents": agents,
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
    p["state"] = state

    if caller == 0:
        p["phase"] = "HUMAN_TRUMP"
        p["status_banner"] = (
            "You hold Quattro di Denari! Choose the trump suit."
            if p["hand_number"] == 1
            else "It is your turn to choose the trump suit."
        )
    else:
        obs = state.build_observation(caller)
        chosen = p["agents"][caller].choose_trump(obs)
        state.trump_suit = chosen
        p["status_banner"] = f"Player {caller} called trump: {suit_badge(chosen)}."
        _run_cricca_phase()


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
                p["status_banner"] += (
                    f" Player {pid} declared {label} Cricca (+3 pts)!"
                )
            break

    _after_cricca_phase()


def _after_cricca_phase() -> None:
    p = st.session_state.play
    state: HandState = p["state"]

    # Check if any AI wants to call out right after Cricca
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
    _advance_until_human_or_pause()


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
            f"{who} called out (Chiamarsi fuori) with {outcome.verified_total} pts — VALID WIN!"
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


def _advance_until_human_or_pause() -> None:
    p = st.session_state.play
    state: HandState = p["state"]

    while p["phase"] == "PLAY_TRICK":
        pid = _current_player(state)
        if pid == 0:
            return  # Wait for human click
        agent = p["agents"][pid]
        legal = _legal_for_player(
            state, pid, enforce_busso=agent.enforce_busso_constraints
        )
        obs = state.build_observation(pid, legal_cards=tuple(legal))
        chosen = agent.choose_card(obs)
        _commit_card_play(pid, chosen)


def _continue_after_trick_pause() -> None:
    p = st.session_state.play
    state: HandState = p["state"]

    # Check if any AI calls out at the end of the completed trick
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
    _advance_until_human_or_pause()


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


def render_play_mode() -> None:
    st.header("Play Beccacino (1 Human + 3 AI)")

    with st.sidebar:
        st.subheader("Match Settings")
        partner_type = st.selectbox(
            "Partner AI (Player 2)", ["HeuristicBot", "ISMCTSBot", "RandomBot"], index=0
        )
        opp_type = st.selectbox(
            "Opponents AI (Players 1 & 3)",
            ["HeuristicBot", "ISMCTSBot", "RandomBot"],
            index=0,
        )
        ai_sims = st.slider(
            "ISMCTSBot simulations/move", min_value=40, max_value=400, value=120, step=20
        )
        hint_sims = st.slider(
            "Stockfish Hint simulations", min_value=100, max_value=1000, value=400, step=50
        )
        target_score = st.number_input(
            "Target Score", min_value=11, max_value=61, value=DEFAULT_TARGET_SCORE, step=10
        )
        seed_input = st.text_input("Optional Seed (blank = random)", value="")
        if st.button("Start New Match", type="primary", use_container_width=True):
            parsed_seed = int(seed_input.strip()) if seed_input.strip().isdigit() else None
            _init_play_match(partner_type, opp_type, ai_sims, int(target_score), parsed_seed)
            st.rerun()

    if "play" not in st.session_state:
        _init_play_match(partner_type, opp_type, ai_sims, int(target_score), None)

    p = st.session_state.play
    state: HandState = p["state"]

    # Top scoreboard
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
        "Trump Suit",
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

    # Phase 1: Human chooses trump
    if p["phase"] == "HUMAN_TRUMP":
        st.subheader("Your Hand")
        st.write(" | ".join(card_badge(c) for c in state.hands[0]))
        st.subheader("Choose Trump Suit")
        cols = st.columns(4)
        for idx, suit in enumerate(Suit):
            if cols[idx].button(suit_badge(suit), use_container_width=True, key=f"trump_{suit.name}"):
                state.trump_suit = suit
                p["status_banner"] = f"You called trump: {suit_badge(suit)}."
                _run_cricca_phase()
                st.rerun()
        return

    # Phase 2: Human holds trump Cricca
    if p["phase"] == "HUMAN_CRICCA":
        st.subheader("You hold Trump Cricca (Asso + Due + Tre)!")
        st.write(" | ".join(card_badge(c) for c in state.hands[0]))
        col_a, col_b = st.columns(2)
        if col_a.button("Declare Cricca (+3 pts)", type="primary", use_container_width=True):
            state.cricca_declared_by = 0
            state.hand_thirds[0] += CRICCA_BONUS_THIRDS
            p["status_banner"] = "You declared Cricca (+3 pts)!"
            _after_cricca_phase()
            st.rerun()
        if col_b.button("Do Not Declare (Forfeit +3 pts)", use_container_width=True):
            p["status_banner"] = "You chose not to declare Cricca."
            _after_cricca_phase()
            st.rerun()
        return

    # Table Display (either current trick in progress or just-completed trick)
    st.subheader("Table")
    if p["phase"] == "TRICK_COMPLETE" and p["last_trick"] is not None:
        rec: TrickRecord = p["last_trick"]
        sig_txt = f" — Signal: **{rec.signal.value}**" if rec.signal != Signal.NONE else ""
        st.markdown(
            f"**Trick {rec.trick_number} completed** (Leader: Player {rec.leader}{sig_txt}) → "
            f"Winner: **Player {rec.winner} (Team {team_of(rec.winner)})** | "
            f"+{rec.points_thirds // 3} pts & {rec.points_thirds % 3}/3"
        )
        tcols = st.columns(4)
        for idx, play in enumerate(rec.plays):
            winner_mark = " ⭐" if play.player == rec.winner else ""
            tcols[idx].markdown(f"**P{play.player}{winner_mark}**\n\n{card_badge(play.card)}")

        btn_a, btn_b = st.columns([2, 1])
        if btn_a.button("Next Trick", type="primary", use_container_width=True):
            _continue_after_trick_pause()
            st.rerun()
        if not p["in_sudden_death"]:
            if btn_b.button("Chiamarsi fuori (Call Out)", use_container_width=True):
                _resolve_callout(calling_team=0)
                st.rerun()

    elif p["phase"] == "PLAY_TRICK":
        if state.current_trick:
            sig_txt = (
                f" — Signal: **{state.current_trick_signal.value}**"
                if state.current_trick_signal != Signal.NONE
                else " — Signal: *(silence)*"
            )
            st.markdown(f"Leader: **Player {state.current_trick_leader}**{sig_txt}")
            tcols = st.columns(4)
            for idx, play in enumerate(state.current_trick):
                tcols[idx].markdown(f"**Player {play.player}**\n\n{card_badge(play.card)}")
        else:
            st.write("Table is empty — **you lead this trick!**")

        # Allow early call-out right before Trick 1 starts (e.g., 39 pts + Cricca)
        if (
            not p["in_sudden_death"]
            and len(state.completed_tricks) == 0
            and len(state.current_trick) == 0
        ):
            if st.button("Chiamarsi fuori (Early Call Out before Trick 1)"):
                _resolve_callout(calling_team=0)
                st.rerun()

        # Human hand & legal moves
        st.subheader("Your Hand (Click a Legal Card to Play)")
        legal = _legal_for_player(state, 0, enforce_busso=False)
        legal_set = set(legal)
        hand_cards = list(state.hands[0])

        cols_per_row = 5
        for row_start in range(0, len(hand_cards), cols_per_row):
            row_cards = hand_cards[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_cards))
            for idx, card in enumerate(row_cards):
                is_legal = card in legal_set
                if cols[idx].button(
                    card_badge(card),
                    disabled=not is_legal,
                    use_container_width=True,
                    key=f"play_{len(state.completed_tricks)}_{card.suit.name}_{card.rank.name}",
                ):
                    _commit_card_play(0, card)
                    if p["phase"] == "PLAY_TRICK":
                        _advance_until_human_or_pause()
                    st.rerun()

        # Stockfish Hint Box
        st.divider()
        col_h1, col_h2 = st.columns([1, 2])
        if col_h1.button(
            f"Analyze Best Move ({hint_sims} sims)", use_container_width=True
        ):
            obs = state.build_observation(0, legal_cards=tuple(legal))
            with st.spinner("Running OpenBecca IS-MCTS search..."):
                p["hint"] = evaluate_position(obs, simulations=hint_sims)
        if p["hint"] is not None:
            render_analysis_panel(p["hint"])

    elif p["phase"] == "HAND_OVER":
        last_hand: HandResult = p["hands_log"][-1]
        st.success(
            f"Hand {last_hand.hand_number} finished! "
            f"Team 0: **+{last_hand.hand_points[0]} pts** | "
            f"Team 1: **+{last_hand.hand_points[1]} pts**"
        )
        if st.button("Deal Next Hand", type="primary", use_container_width=True):
            _start_new_hand()
            st.rerun()

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
        st.download_button(
            "Download Match Replay (JSON)",
            data=json.dumps(match_res.to_dict(), indent=2, ensure_ascii=False),
            file_name="openbecca_match_replay.json",
            mime="application/json",
        )

    # Expandable Trick History
    if state.completed_tricks:
        with st.expander(f"Completed Tricks Log ({len(state.completed_tricks)}/10)"):
            rows = []
            for t in state.completed_tricks:
                plays_str = " | ".join(
                    f"P{pl.player}: {pl.card.italian_name}" for pl in t.plays
                )
                rows.append(
                    {
                        "Trick": t.trick_number,
                        "Leader": f"P{t.leader}",
                        "Signal": t.signal.value or "—",
                        "Plays": plays_str,
                        "Winner": f"P{t.winner} (T{team_of(t.winner)})",
                        "Thirds": t.points_thirds,
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ======================================================================
# MODE 2: WATCH 4 AI
# ======================================================================
def render_watch_mode() -> None:
    st.header("Watch 4 AI Simulation (Spectator Mode)")

    col1, col2, col3, col4 = st.columns(4)
    team0_type = col1.selectbox("Team 0 (P0 & P2)", ["ISMCTSBot", "HeuristicBot", "RandomBot"], index=0)
    team1_type = col2.selectbox("Team 1 (P1 & P3)", ["HeuristicBot", "ISMCTSBot", "RandomBot"], index=0)
    sims = col3.slider("ISMCTS Simulations", 40, 300, 100, 20, key="watch_sims")
    seed_val = col4.number_input("Deal Seed", min_value=0, max_value=999_999, value=42)

    if st.button("Simulate Hand", type="primary"):
        deck = Deck(seed=int(seed_val))
        dealt = deck.deal()
        agents = [
            create_bot(team0_type, 0, sims, seed=int(seed_val) + 1),
            create_bot(team1_type, 1, sims, seed=int(seed_val) + 2),
            create_bot(team0_type, 2, sims, seed=int(seed_val) + 3),
            create_bot(team1_type, 3, sims, seed=int(seed_val) + 4),
        ]
        with st.spinner("Simulating 4-AI hand..."):
            res = GameEngine(agents).play_hand(allow_callout=False, dealt_hands=dealt)
        st.session_state.watch_result = {
            "dealt": dealt,
            "result": res,
            "team0": team0_type,
            "team1": team1_type,
        }

    if "watch_result" not in st.session_state:
        st.info("Configure both teams above and click **Simulate Hand** to watch every trick.")
        return

    data = st.session_state.watch_result
    res: HandResult = data["result"]
    dealt: list[list[Card]] = data["dealt"]

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Team 0 ({data['team0']})", f"{res.hand_points[0]} pts", f"{res.hand_thirds[0]} thirds")
    m2.metric(f"Team 1 ({data['team1']})", f"{res.hand_points[1]} pts", f"{res.hand_thirds[1]} thirds")
    m3.metric(
        "Trump & Cricca",
        suit_badge(res.trump_suit),
        f"Caller: P{res.trump_caller} | Cricca: {f'P{res.cricca_declared_by}' if res.cricca_declared_by is not None else 'None'}",
    )

    with st.expander("Initial Dealt Hands (All 4 Players)"):
        hcols = st.columns(4)
        for pid in range(4):
            hcols[pid].markdown(f"**Player {pid} (Team {team_of(pid)})**")
            for card in dealt[pid]:
                hcols[pid].write(card_badge(card))

    st.subheader("Trick-by-Trick Replay")
    rows = []
    for t in res.tricks:
        plays_str = " | ".join(f"P{pl.player}: {pl.card.italian_name}" for pl in t.plays)
        rows.append(
            {
                "Trick": t.trick_number,
                "Leader": f"P{t.leader}",
                "Signal": t.signal.value or "—",
                "Cards Played": plays_str,
                "Winner": f"P{t.winner} (Team {team_of(t.winner)})",
                "Points (thirds)": t.points_thirds,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Hand Log (JSON)",
        data=json.dumps(res.to_dict(), indent=2, ensure_ascii=False),
        file_name="openbecca_ai_hand.json",
        mime="application/json",
    )


# ======================================================================
# MODE 3: STOCKFISH STUDIO (CUSTOM POSITION ANALYZER)
# ======================================================================
def render_studio_mode() -> None:
    st.header("Stockfish Studio — Custom Position Analyzer")
    st.write(
        "Set up any table scenario manually and ask **OpenBecca** for the expected points "
        "and win probability of every legal move."
    )

    all_sorted = sorted(ALL_CARDS, key=lambda c: c.sort_key())
    label_to_card = {card_badge(c): c for c in all_sorted}
    all_labels = list(label_to_card.keys())

    c1, c2, c3, c4 = st.columns(4)
    trump_suit = c1.selectbox(
        "Trump Suit", list(Suit), format_func=suit_badge, key="studio_trump"
    )
    player_id = c2.selectbox("Your Seat", [0, 1, 2, 3], index=0, key="studio_seat")
    trump_caller = c3.selectbox("Trump Caller", [0, 1, 2, 3], index=0, key="studio_caller")
    cricca_choice = c4.selectbox(
        "Cricca Declared By",
        ["None", "Player 0", "Player 1", "Player 2", "Player 3"],
        index=0,
        key="studio_cricca",
    )
    cricca_by = None if cricca_choice == "None" else int(cricca_choice.split()[-1])

    # Step 1: Select player's hand
    default_hand = [card_badge(c) for c in Deck(seed=42).deal()[0]]
    selected_hand_labels = st.multiselect(
        "1. Your Hand (select 1 to 10 cards)",
        options=all_labels,
        default=default_hand,
        max_selections=10,
        key="studio_hand",
    )
    player_hand = [label_to_card[lbl] for lbl in selected_hand_labels]

    # Step 2: Select cards currently on the table (in play order)
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
            format_func=lambda s: s.value if s.value else "(Silence)",
            key="studio_signal",
        )

    # Step 3: Previously played cards from earlier tricks
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

    sims = st.slider(
        "Analysis Depth (IS-MCTS Simulations)",
        min_value=100,
        max_value=1500,
        value=500,
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

        # Assign table players in exact counter-clockwise order ending right before player_id
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

        with st.spinner(f"Running {sims} IS-MCTS simulations..."):
            analysis = evaluate_position(obs, simulations=sims)
        render_analysis_panel(analysis)


# ======================================================================
# APP ENTRY POINT
# ======================================================================
def main() -> None:
    st.set_page_config(
        page_title="OpenBecca — Beccacino AI & Analyzer",
        page_icon="🃏",
        layout="wide",
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

    if mode == "Play (1 Human + 3 AI)":
        render_play_mode()
    elif mode == "Watch 4 AI":
        render_watch_mode()
    else:
        render_studio_mode()


if __name__ == "__main__":
    main()