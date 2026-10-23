"""Streamlit dashboard: streamlit run app/dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import DEFAULT_MODEL_PATH
from src.pipeline import load_analytics, load_or_train

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Predictor & Analytics Engine")


@st.cache_resource(show_spinner="Loading or training model...")
def get_engine():
    return load_or_train(DEFAULT_MODEL_PATH)


@st.cache_resource(show_spinner="Rating team units...")
def get_analytics():
    return load_analytics()


engine = get_engine()
teams = engine.teams

with st.sidebar:
    st.subheader("Walk-forward CV")
    st.caption(f"Trained on {engine.n_matches_} matches up to {engine.last_match_date_}")
    cv = {k: v for k, v in engine.cv_metrics_.items() if isinstance(v, dict)}
    st.dataframe(pd.DataFrame(cv).T[["log_loss", "rps", "brier_score", "accuracy"]])

predict_tab, units_tab = st.tabs(["Match Prediction", "Team Units & Simulation"])

with predict_tab:
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        odd_h = st.number_input("Home Odds (Decimal)", min_value=1.01, value=2.10, step=0.05)
    with col2:
        away = st.selectbox("Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)
        odd_a = st.number_input("Away Odds (Decimal)", min_value=1.01, value=3.40, step=0.05)
    odd_d = st.number_input("Draw Odds (Decimal)", min_value=1.01, value=3.50, step=0.05)

    if st.button("Predict Match Outcome", type="primary"):
        if home == away:
            st.error("Pick two different teams.")
            st.stop()
        res = engine.predict_fixture(home, away, odds=(odd_h, odd_d, odd_a))
        p = res["probabilities"]
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} Win", f"{p['home_win'] * 100:.1f}%")
        m2.metric("Draw", f"{p['draw'] * 100:.1f}%")
        m3.metric(f"{away} Win", f"{p['away_win'] * 100:.1f}%")
        st.subheader("Expected Goals (xG) & Ratings")
        st.json({"expected_goals": res["expected_goals"], "elo_ratings": res["elo_ratings"], "markets": res["markets"]})
        st.subheader("Most Likely Exact Scorelines")
        st.table(pd.DataFrame(res["top_scorelines"]))
        st.subheader("Market Value & Kelly Edge")
        st.json(res["value_bets"])

with units_tab:
    analytics = get_analytics()
    comps = analytics.competitions
    competition = st.selectbox("Competition", ["All clubs", *comps]) if comps else "All clubs"
    comp_filter = None if competition == "All clubs" else competition
    club_list = analytics.teams_in(comp_filter)
    st.caption(
        "Indices: 100 = average of all clubs in the data; higher is better. Opponent-adjusted and time-weighted. "
        "Clubs from different leagues share one scale through their Champions League meetings."
    )
    st.dataframe(analytics.league_table(comp_filter))

    team = st.selectbox("Season report for", club_list, key="report_team")
    report = analytics.team_report(team)
    st.markdown(f"**Style:** {report['style']}")
    tl = pd.DataFrame(report["timeline"]).set_index("date")
    st.line_chart(tl[["attack", "defence", "midfield"]])
    if "possession" in tl:
        st.caption("Expected possession % vs an average side")
        st.line_chart(tl[["possession"]])
    if "players" in report:
        st.markdown("**Players** (ratings: 50 = average for the position)")
        st.dataframe(pd.DataFrame(report["players"]))

    st.divider()
    st.subheader("Simulate a matchup")
    s1, s2 = st.columns(2)
    sim_home = s1.selectbox("Home", club_list, index=0, key="sim_home")
    sim_away = s2.selectbox("Away", club_list, index=1, key="sim_away")
    if analytics.ratings.has_possession:
        poss_shift = st.slider("Extra possession for the home side (percentage points)", -20, 20, 0, 1)
        shift = 0.0
    else:
        poss_shift = 0
        shift = st.slider("Tilt the midfield battle (+ favours home)", -0.5, 0.5, 0.0, 0.05)
    squad = analytics.players.players if analytics.players is not None else None
    home_out = s1.multiselect(
        "Home absentees", sorted(squad.loc[squad["team"] == sim_home, "player"]) if squad is not None else []
    )
    away_out = s2.multiselect(
        "Away absentees", sorted(squad.loc[squad["team"] == sim_away, "player"]) if squad is not None else []
    )
    referees = sorted(analytics.matches["Referee"].dropna().unique()) if "Referee" in analytics.matches.columns else []
    referee = st.selectbox("Referee (red-card risk)", ["(unknown)", *referees]) if referees else "(unknown)"

    if st.button("Run simulation", type="primary"):
        if sim_home == sim_away:
            st.error("Pick two different teams.")
            st.stop()
        res = analytics.simulate(
            sim_home,
            sim_away,
            midfield_shift=shift,
            possession_shift=float(poss_shift),
            home_absent=home_out,
            away_absent=away_out,
            referee=None if referee == "(unknown)" else referee,
        )
        if res["expected_possession"]:
            st.caption(
                f"Expected possession: {sim_home} {res['expected_possession']['home']:.0%} - "
                f"{res['expected_possession']['away']:.0%} {sim_away} "
                f"(normally {res['expected_possession']['home_baseline']:.0%} for {sim_home})"
            )
        o = res["outcome"]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{sim_home} win", f"{o['home_win']:.1%}")
        c2.metric("Draw", f"{o['draw']:.1%}")
        c3.metric(f"{sim_away} win", f"{o['away_win']:.1%}")
        st.markdown("**Unit battles**")
        st.dataframe(pd.DataFrame(res["unit_comparison"]))
        for side in ("home", "away"):
            sd = res[side]
            st.markdown(f"#### {sd['team']}: {sd['style']}")
            a, d, m = sd["attack"], sd["defence"], sd["midfield_effect"]
            k1, k2, k3, k4, k5 = st.columns(5)
            k5.metric("Shots on target", f"{a['mean_shots_on_target']:.1f}")
            k1.metric("Avg goals", f"{a['mean_goals']:.2f}")
            k2.metric("Scores 2+", f"{a['p_score_2_plus']:.0%}")
            k3.metric("Clean sheet", f"{d['clean_sheet_prob']:.0%}")
            k4.metric("Goals from midfield edge", f"{m['goals_from_midfield_edge']:+.2f}")
            st.bar_chart(pd.Series(a["goal_distribution"], name="probability"))
            st.caption(f"Defence: {d['verdict']}. Goal breakdown (multipliers vs league average):")
            st.dataframe(pd.DataFrame(sd["goal_breakdown"]).T)
            if "players" in sd:
                st.dataframe(pd.DataFrame(sd["players"]["top_scorers"]))
        gs = res.get("game_state")
        if gs:
            st.markdown("**How the match unfolds** (six 15-minute game states)")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric(f"{sim_home} scores first", f"{gs['home_scores_first']:.0%}")
            g2.metric(f"{sim_away} scores first", f"{gs['away_scores_first']:.0%}")
            g3.metric(f"{sim_home} red card", f"{gs['p_home_red_card']:.1%}")
            g4.metric(f"{sim_away} red card", f"{gs['p_away_red_card']:.1%}")
            st.caption("Expected goals per 15-minute period")
            st.bar_chart(pd.DataFrame(gs["goals_by_period"]))
        mk = res["markets"]
        st.markdown(f"**Asian lines** (fair odds) | Over 2.5: {mk['over_2_5']:.1%} | BTTS: {mk['btts']:.1%}")
        m1, m2 = st.columns(2)
        m1.dataframe(pd.DataFrame(mk["asian_handicap_home"]).set_index("line")[["win", "push", "loss", "fair_odds"]])
        m2.dataframe(pd.DataFrame(mk["goal_lines_over"]).set_index("line")[["fair_odds", "under_fair_odds"]])
        for note in res["tactical_notes"]:
            st.info(note)
        st.markdown("**Midfield sensitivity**")
        st.dataframe(analytics.simulator.midfield_sensitivity(sim_home, sim_away))
