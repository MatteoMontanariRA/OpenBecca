"""Parallel multi-core benchmark runner using duplicate hands to cancel deal luck."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

from openbecca.agents.base import Agent
from openbecca.agents.heuristic_bot import HeuristicBot, HeuristicConfig
from openbecca.agents.random_bot import RandomBot
from openbecca.cards import Deck
from openbecca.engine import GameEngine


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregated performance stats comparing Team A against Team B."""

    team_a_name: str
    team_b_name: str
    total_hands: int
    duplicate_deals: bool
    team_a_avg_points: float
    team_b_avg_points: float
    point_diff: float
    team_a_hand_win_rate: float
    elapsed_seconds: float

    def format_report(self) -> str:
        hands_per_sec = self.total_hands / max(self.elapsed_seconds, 1e-6)
        return (
            f"{self.team_a_name} vs {self.team_b_name} "
            f"({self.total_hands} hands, duplicate={self.duplicate_deals})\n"
            f"  {self.team_a_name} avg pts/hand : {self.team_a_avg_points:.3f}\n"
            f"  {self.team_b_name} avg pts/hand : {self.team_b_avg_points:.3f}\n"
            f"  Net edge (A - B)         : {self.point_diff:+.3f} pts/hand\n"
            f"  {self.team_a_name} hand win rate: {self.team_a_hand_win_rate * 100:.1f}%\n"
            f"  Runtime                  : {self.elapsed_seconds:.2f}s ({hands_per_sec:.0f} hands/s)"
        )


def _make_agent(
    agent_type: str,
    seat: int,
    seed: int,
    config: HeuristicConfig | None = None,
) -> Agent:
    if agent_type == "random":
        return RandomBot(name=f"Random-{seat}", seed=seed)
    if agent_type == "heuristic":
        return HeuristicBot(name=f"Heuristic-{seat}", config=config)
    raise ValueError(f"Unknown agent_type: {agent_type}")


def _run_chunk(
    args: tuple[str, str, int, int, int, bool, HeuristicConfig | None, HeuristicConfig | None],
) -> tuple[int, int, int, int]:
    """Worker function executed in parallel across CPU cores."""
    (
        team_a_type,
        team_b_type,
        num_deals,
        base_seed,
        chunk_id,
        duplicate,
        config_a,
        config_b,
    ) = args

    deck = Deck(seed=base_seed + chunk_id * 100_003)
    team_a_points = 0
    team_b_points = 0
    team_a_wins = 0
    hands_played = 0

    for deal_idx in range(num_deals):
        dealt = deck.deal()
        s = base_seed + chunk_id * 10_000 + deal_idx * 17

        # Run 1: Team A in seats (0, 2), Team B in seats (1, 3)
        agents_1 = [
            _make_agent(team_a_type, 0, s + 0, config_a),
            _make_agent(team_b_type, 1, s + 1, config_b),
            _make_agent(team_a_type, 2, s + 2, config_a),
            _make_agent(team_b_type, 3, s + 3, config_b),
        ]
        res1 = GameEngine(agents_1).play_hand(allow_callout=False, dealt_hands=dealt)
        team_a_points += res1.hand_points[0]
        team_b_points += res1.hand_points[1]
        if res1.hand_points[0] > res1.hand_points[1]:
            team_a_wins += 1
        hands_played += 1

        # Run 2 (duplicate): Swap seats with the exact same cards
        if duplicate:
            agents_2 = [
                _make_agent(team_b_type, 0, s + 4, config_b),
                _make_agent(team_a_type, 1, s + 5, config_a),
                _make_agent(team_b_type, 2, s + 6, config_b),
                _make_agent(team_a_type, 3, s + 7, config_a),
            ]
            res2 = GameEngine(agents_2).play_hand(allow_callout=False, dealt_hands=dealt)
            team_a_points += res2.hand_points[1]
            team_b_points += res2.hand_points[0]
            if res2.hand_points[1] > res2.hand_points[0]:
                team_a_wins += 1
            hands_played += 1

    return team_a_points, team_b_points, team_a_wins, hands_played


def run_hand_benchmark(
    team_a_type: str = "heuristic",
    team_b_type: str = "random",
    *,
    num_deals: int = 1000,
    duplicate: bool = True,
    workers: int | None = None,
    seed: int = 42,
    config_a: HeuristicConfig | None = None,
    config_b: HeuristicConfig | None = None,
    team_a_name: str | None = None,
    team_b_name: str | None = None,
) -> BenchmarkSummary:
    """Run a parallel head-to-head benchmark across CPU cores."""
    start = time.perf_counter()
    num_workers = workers or max(1, min(os.cpu_count() or 1, 8))

    deals_per_worker, remainder = divmod(num_deals, num_workers)
    tasks = []
    for idx in range(num_workers):
        count = deals_per_worker + (1 if idx < remainder else 0)
        if count > 0:
            tasks.append(
                (team_a_type, team_b_type, count, seed, idx, duplicate, config_a, config_b)
            )

    if num_workers == 1 or len(tasks) == 1:
        results = [_run_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_run_chunk, tasks))

    total_a = sum(r[0] for r in results)
    total_b = sum(r[1] for r in results)
    total_a_wins = sum(r[2] for r in results)
    total_hands = sum(r[3] for r in results)
    elapsed = time.perf_counter() - start

    avg_a = total_a / total_hands
    avg_b = total_b / total_hands
    return BenchmarkSummary(
        team_a_name=team_a_name or team_a_type.capitalize(),
        team_b_name=team_b_name or team_b_type.capitalize(),
        total_hands=total_hands,
        duplicate_deals=duplicate,
        team_a_avg_points=avg_a,
        team_b_avg_points=avg_b,
        point_diff=avg_a - avg_b,
        team_a_hand_win_rate=total_a_wins / total_hands,
        elapsed_seconds=elapsed,
    )


def run_ablation_study(num_deals: int = 500, workers: int | None = None, seed: int = 42) -> None:
    """Disable each hypothesis (H01-H12) one by one against full HeuristicBot."""
    full_cfg = HeuristicConfig()
    print(f"=== Ablation Study (Full HeuristicBot vs Missing 1 Strategy, {num_deals * 2} duplicate hands each) ===")
    for field_name in HeuristicConfig.__dataclass_fields__:
        ablated_cfg = replace(full_cfg, **{field_name: False})
        summary = run_hand_benchmark(
            "heuristic",
            "heuristic",
            num_deals=num_deals,
            duplicate=True,
            workers=workers,
            seed=seed,
            config_a=full_cfg,
            config_b=ablated_cfg,
            team_a_name="Full",
            team_b_name=f"No-{field_name}",
        )
        verdict = "HELPS" if summary.point_diff > 0.02 else ("HURTS" if summary.point_diff < -0.02 else "NEUTRAL")
        print(
            f"  {field_name:32s} | Full edge: {summary.point_diff:+.3f} pts/hand "
            f"| Win%: {summary.team_a_hand_win_rate * 100:5.1f}% | {verdict}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenBecca parallel bot benchmark.")
    parser.add_argument("--deals", type=int, default=1000, help="Number of unique deals (x2 if duplicate).")
    parser.add_argument("--workers", type=int, default=None, help="CPU worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--ablation", action="store_true", help="Run H01-H12 ablation study.")
    args = parser.parse_args()

    summary = run_hand_benchmark(
        "heuristic",
        "random",
        num_deals=args.deals,
        duplicate=True,
        workers=args.workers,
        seed=args.seed,
    )
    print(summary.format_report())

    if args.ablation:
        print()
        run_ablation_study(num_deals=max(200, args.deals // 2), workers=args.workers, seed=args.seed)


if __name__ == "__main__":
    main()