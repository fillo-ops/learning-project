"""Betting-market tools.

python -m scripts.markets price --home Arsenal --away Chelsea      # Asian handicap & goal-line fair odds
python -m scripts.markets price --home Arsenal --away Chelsea --ah -0.5 1.95 --total 2.5 2.05
python -m scripts.markets portfolio --bets bets.csv --cap 0.12     # matchday staking with exposure cap
python -m scripts.markets steam                                    # opening -> closing line movement
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.market import asian_handicap, asian_markets, asian_total, bet_expected_value, line_movement, portfolio_kelly
from src.pipeline import load_analytics, load_or_create_dataset


def _price(args: argparse.Namespace) -> int:
    analytics = load_analytics(n_sims=args.sims)
    matrix = analytics.simulator.score_matrix(args.home, args.away)
    markets = asian_markets(matrix)
    print(f"{args.home} vs {args.away} ({args.sims:,} game-state simulations)")
    print("Asian handicap (home):")
    print(pd.DataFrame(markets["asian_handicap_home"]).set_index("line").to_string())
    print("Goal lines (over):")
    print(pd.DataFrame(markets["goal_lines_over"]).set_index("line").to_string())
    if args.ah:
        line, odds = args.ah
        s = asian_handicap(matrix, line)
        print(f"\nAH {line:+} @ {odds}: EV {bet_expected_value(s, odds):+.3f} per unit (fair {s['fair_odds']:.3f})")
    if args.total:
        line, odds = args.total
        s = asian_total(matrix, line)
        print(f"Over {line} @ {odds}: EV {bet_expected_value(s, odds):+.3f} per unit (fair {s['fair_odds']:.3f})")
    return 0


def _portfolio(args: argparse.Namespace) -> int:
    bets = pd.read_csv(args.bets).to_dict(orient="records")
    picks = portfolio_kelly(bets, max_total_exposure=args.cap)
    if not picks:
        print("No value bets above the EV threshold.")
        return 0
    df = pd.DataFrame(picks)[["match", "selection", "prob", "odds", "ev", "kelly_stake", "stake"]]
    print(df.to_string(index=False))
    print(f"Total exposure: {df['stake'].sum():.1%} of bankroll (cap {args.cap:.0%})")
    return 0


def _steam(_: argparse.Namespace) -> int:
    table, summary = line_movement(load_or_create_dataset())
    print("Average opening -> closing move per team (positive = market came in on them):")
    print(table.to_string())
    print(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price", help="Fair Asian handicap and goal-line odds for a fixture")
    p.add_argument("--home", required=True)
    p.add_argument("--away", required=True)
    p.add_argument("--sims", type=int, default=20000)
    p.add_argument("--ah", type=float, nargs=2, metavar=("LINE", "ODDS"), help="Check a home handicap price")
    p.add_argument("--total", type=float, nargs=2, metavar=("LINE", "ODDS"), help="Check an over price")
    q = sub.add_parser("portfolio", help="Stake a matchday of bets with a total exposure cap")
    q.add_argument("--bets", required=True, help="CSV with columns match, selection, prob, odds")
    q.add_argument("--cap", type=float, default=0.12, help="Maximum share of bankroll in play at once")
    sub.add_parser("steam", help="Opening-to-closing line movement analysis")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        return {"price": _price, "portfolio": _portfolio, "steam": _steam}[args.cmd](args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
