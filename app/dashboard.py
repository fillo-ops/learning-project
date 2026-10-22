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
    st.caption("Indices: 100 = league average; higher is better for the team. Opponent-adjusted and time-weighted.")
    st.dataframe(analytics.ratings.league_table())

    team = st.selectbox("Season report for", analytics.teams, key="report_team")
    report = analytics.team_report(team)
    st.markdown(f"**Style:** {report['style']}")
    tl = pd.DataFrame(report["timeline"]).set_index("date")
    st.line_chart(tl[["attack", "defence", "midfield"]])
    if "players" in report:
        st.markdown("**Players** (ratings: 50 = average for the position)")
        st.dataframe(pd.DataFrame(report["players"]))

    st.divider()
    st.subheader("Simulate a matchup")
    s1, s2 = st.columns(2)
    sim_home = s1.selectbox("Home", analytics.teams, index=0, key="sim_home")
    sim_away = s2.selectbox("Away", analytics.teams, index=1, key="sim_away")
    shift = st.slider("Tilt the midfield battle (+ favours home)", -0.5, 0.5, 0.0, 0.05)
    squad = analytics.players.players if analytics.players is not None else None
    home_out = s1.multiselect(
        "Home absentees", sorted(squad.loc[squad["team"] == sim_home, "player"]) if squad is not None else []
    )
    away_out = s2.multiselect(
        "Away absentees", sorted(squad.loc[squad["team"] == sim_away, "player"]) if squad is not None else []
    )

    if st.button("Run simulation", type="primary"):
        if sim_home == sim_away:
            st.error("Pick two different teams.")
            st.stop()
        res = analytics.simulate(sim_home, sim_away, midfield_shift=shift, home_absent=home_out, away_absent=away_out)
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
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg goals", f"{a['mean_goals']:.2f}")
            k2.metric("Scores 2+", f"{a['p_score_2_plus']:.0%}")
            k3.metric("Clean sheet", f"{d['clean_sheet_prob']:.0%}")
            k4.metric("Goals from midfield edge", f"{m['goals_from_midfield_edge']:+.2f}")
            st.bar_chart(pd.Series(a["goal_distribution"], name="probability"))
            st.caption(f"Defence: {d['verdict']}. Goal breakdown (multipliers vs league average):")
            st.dataframe(pd.DataFrame(sd["goal_breakdown"]).T)
            if "players" in sd:
                st.dataframe(pd.DataFrame(sd["players"]["top_scorers"]))
        for note in res["tactical_notes"]:
            st.info(note)
        st.markdown("**Midfield sensitivity**")
        st.dataframe(analytics.simulator.midfield_sensitivity(sim_home, sim_away))
