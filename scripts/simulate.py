"""Simulate a matchup and analyse attack, midfield and defence.

python -m scripts.simulate --home Arsenal --away Chelsea
python -m scripts.simulate --home Arsenal --away Chelsea --home-absent "Arsenal FWD1" --midfield-shift 0.1
python -m scripts.simulate --team Arsenal            # season unit report for one team
python -m scripts.simulate --league                  # unit ratings table for every team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_PLAYERS_PATH
from src.pipeline import load_analytics


def _print_simulation(res: dict) -> None:
    h, a = res["home_team"], res["away_team"]
    print(f"\n=== {h} vs {a}: {res['n_sims']:,} simulations ===")
    o = res["outcome"]
    print(f"{h} win {o['home_win']:.1%} | draw {o['draw']:.1%} | {a} win {o['away_win']:.1%}")
    print("Most likely scores:", ", ".join(f"{s['score']} ({s['probability']:.1%})" for s in res["top_scorelines"]))

    print("\n-- Unit battles (100 = league average) --")
    print(pd.DataFrame(res["unit_comparison"]).fillna("").to_string(index=False))

    for side in ("home", "away"):
        s = res[side]
        att, dfn, mid = s["attack"], s["defence"], s["midfield_effect"]
        print(f"\n-- {s['team']} ({s['style']}) --")
        print(
            f"Attack: {att['mean_goals']:.2f} goals on avg from {att['mean_shots']:.1f} shots "
            f"({att['mean_shots_on_target']:.1f} on target); P(1+) {att['p_score_1_plus']:.0%}, "
            f"P(2+) {att['p_score_2_plus']:.0%}, P(3+) {att['p_score_3_plus']:.0%}, max seen {att['max_goals_seen']}"
        )
        print("Goal distribution:", "  ".join(f"{k}: {v:.1%}" for k, v in att["goal_distribution"].items()))
        print(
            f"Midfield: shot volume x{mid['shot_volume_multiplier']:.2f} vs average; "
            f"worth {mid['goals_from_midfield_edge']:+.2f} goals (xG {mid['xg_with_neutral_midfield']:.2f} with a neutral midfield)"
        )
        print(
            f"Defence: concedes {dfn['expected_conceded']:.2f} (typically {dfn['typical_conceded_vs_average_attack']:.2f}), "
            f"clean sheet {dfn['clean_sheet_prob']:.0%} -> {dfn['verdict']}"
        )
        print("Where the goals come from (x = multiplier vs average):")
        print(pd.DataFrame(s["goal_breakdown"]).T.to_string())
        if "players" in s:
            p = s["players"]
            print(f"Player-based unit ratings (50 = avg): {p['unit_ratings']}")
            if p["absence_effects"] != {k: 1.0 for k in p["absence_effects"]}:
                print(f"Absence effects: {p['absence_effects']}")
            print(pd.DataFrame(p["top_scorers"]).to_string(index=False))

    if res["tactical_notes"]:
        print("\n-- Tactical notes --")
        for note in res["tactical_notes"]:
            print(" *", note)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Team unit analysis and matchup simulation")
    parser.add_argument("--home", default="Arsenal")
    parser.add_argument("--away", default="Chelsea")
    parser.add_argument("--sims", type=int, default=20000, help="Number of simulated matches")
    parser.add_argument(
        "--midfield-shift", type=float, default=0.0, help="Tilt the midfield battle (log scale, + favours home)"
    )
    parser.add_argument("--home-absent", nargs="*", default=None, help="Unavailable home players")
    parser.add_argument("--away-absent", nargs="*", default=None, help="Unavailable away players")
    parser.add_argument("--players", default=str(DEFAULT_PLAYERS_PATH), help="Player stats CSV")
    parser.add_argument("--team", help="Print the season unit report for this team instead")
    parser.add_argument("--league", action="store_true", help="Print unit ratings for every team")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    analytics = load_analytics(players_path=Path(args.players), n_sims=args.sims)

    try:
        if args.league:
            print(analytics.ratings.league_table().to_string())
            print("\nStyles:")
            print(analytics.profiles["style"].to_string())
            return 0
        if args.team:
            report = analytics.team_report(args.team)
            if args.json:
                print(json.dumps(report, indent=2, default=str))
                return 0
            print(f"\n=== {report['team']}: {report['style']} ===")
            print(pd.DataFrame({"index": report["units"], "rank": report["league_rank"]}).to_string())
            tl = pd.DataFrame(report["timeline"]).set_index("date")
            print("\nUnit indices through the season (monthly snapshots):")
            print(tl.iloc[::4][["attack", "defence", "midfield", "chance_quality", "finishing"]].to_string())
            print("\nAttack vs every defence (at home):")
            print(analytics.simulator.attack_vs_all_defences(report["team"]).to_string(index=False))
            return 0

        res = analytics.simulate(
            args.home, args.away, midfield_shift=args.midfield_shift,
            home_absent=args.home_absent, away_absent=args.away_absent,
        )  # fmt: skip
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0
    _print_simulation(res)
    print("\n-- Midfield sensitivity: what if the midfield battle tilts? --")
    print(analytics.simulator.midfield_sensitivity(args.home, args.away).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
