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
    if res["expected_possession"]:
        ep = res["expected_possession"]
        note = f" (normally {ep['home_baseline']:.0%} for {h})" if res["possession_shift"] else ""
        print(f"Expected possession: {h} {ep['home']:.0%} - {ep['away']:.0%} {a}{note}")
    print("Most likely scores:", ", ".join(f"{s['score']} ({s['probability']:.1%})" for s in res["top_scorelines"]))

    print("\n-- Unit battles (100 = league average) --")
    print(pd.DataFrame(res["unit_comparison"]).fillna("").to_string(index=False))

    for side in ("home", "away"):
        s = res[side]
        att, dfn, mid = s["attack"], s["defence"], s["midfield_effect"]
        volume = "chances created" if res["volume_stat"] == "chances" else "shots"
        print(f"\n-- {s['team']} ({s['style']}) --")
        print(
            f"Attack: {att['mean_goals']:.2f} goals on avg from {att['mean_volume']:.1f} {volume} "
            f"-> {att['mean_shots_on_target']:.1f} shots on target (3+ on target {att['p_3_plus_shots_on_target']:.0%}); "
            f"P(1+) {att['p_score_1_plus']:.0%}, "
            f"P(2+) {att['p_score_2_plus']:.0%}, P(3+) {att['p_score_3_plus']:.0%}, max seen {att['max_goals_seen']}"
        )
        print("Goal distribution:", "  ".join(f"{k}: {v:.1%}" for k, v in att["goal_distribution"].items()))
        print(
            f"Midfield: {volume} x{mid['volume_multiplier']:.2f} vs average"
            + (f", {mid['expected_possession']:.0%} possession" if "expected_possession" in mid else "")
            + "; "
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

    gs = res.get("game_state")
    if gs:
        print("\n-- How the match unfolds (15-minute game-state simulation) --")
        print(
            f"Scores first: {h} {gs['home_scores_first']:.0%}, {a} {gs['away_scores_first']:.0%}, goalless {gs['p_goalless']:.0%}"
        )
        print(
            f"Win after scoring first: {h} {gs['home_win_if_scores_first'] or 0:.0%}, "
            f"{a} {gs['away_win_if_scores_first'] or 0:.0%}; "
            f"comeback wins: {h} {gs['home_comeback_win']:.1%}, {a} {gs['away_comeback_win']:.1%}"
        )
        print(
            f"Share of match: {h} leading {gs['share_of_match_home_leading']:.0%}, level "
            f"{gs['share_of_match_level']:.0%}, {a} leading {gs['share_of_match_away_leading']:.0%}"
        )
        ref = f" (referee {gs['referee']})" if gs.get("referee") else ""
        print(f"Red card: {h} {gs['p_home_red_card']:.1%}, {a} {gs['p_away_red_card']:.1%}{ref}")
        print(pd.DataFrame(gs["goals_by_period"]).T.to_string())

    mk = res.get("markets")
    if mk:
        print(f"\n-- Markets -- Over 2.5: {mk['over_2_5']:.1%}  BTTS: {mk['btts']:.1%}")
        ah = pd.DataFrame(mk["asian_handicap_home"]).set_index("line")
        print(f"Asian handicap ({h}) fair odds:")
        print(ah[["win", "half_win", "push", "half_loss", "loss", "fair_odds"]].to_string())
        gl = pd.DataFrame(mk["goal_lines_over"]).set_index("line")
        print("Goal lines (over / under fair odds):")
        print(gl[["fair_odds", "under_fair_odds"]].to_string())

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
    parser.add_argument(
        "--possession-shift", type=float, default=0.0, help="Give the home side this many extra possession points"
    )
    parser.add_argument("--home-absent", nargs="*", default=None, help="Unavailable home players")
    parser.add_argument("--away-absent", nargs="*", default=None, help="Unavailable away players")
    parser.add_argument("--players", default=str(DEFAULT_PLAYERS_PATH), help="Player stats CSV")
    parser.add_argument("--team", help="Print the season unit report for this team instead")
    parser.add_argument("--league", action="store_true", help="Print unit ratings for every team")
    parser.add_argument("--competition", help='With --league: only clubs in this competition, e.g. "Champions League"')
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    parser.add_argument("--referee", help="Referee name (scales red-card risk if the data has a Referee column)")
    parser.add_argument("--static", action="store_true", help="Single full-match draw instead of 15-minute game states")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    analytics = load_analytics(players_path=Path(args.players), n_sims=args.sims, game_state=not args.static)

    try:
        if args.league:
            if analytics.competitions:
                print(f"Competitions in the data: {', '.join(analytics.competitions)}")
            print(analytics.league_table(args.competition).to_string())
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
            cols = [
                c
                for c in ("attack", "defence", "midfield", "possession", "chances_created", "shots_on_target")
                if c in tl
            ]
            print(tl.iloc[::4][cols].to_string())
            print("\nAttack vs every defence (at home):")
            print(analytics.simulator.attack_vs_all_defences(report["team"]).to_string(index=False))
            return 0

        res = analytics.simulate(
            args.home, args.away, midfield_shift=args.midfield_shift, possession_shift=args.possession_shift,
            home_absent=args.home_absent, away_absent=args.away_absent, referee=args.referee,
        )  # fmt: skip
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0
    _print_simulation(res)
    print("\n-- Midfield sensitivity: what if possession / the midfield battle shifts? --")
    print(analytics.simulator.midfield_sensitivity(args.home, args.away).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
