"""Streamlit dashboard: streamlit run app/dashboard.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.config import DEFAULT_MODEL_PATH
from src.pipeline import load_or_train

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Predictor & Analytics Engine")


@st.cache_resource(show_spinner="Loading or training model...")
def get_engine():
    return load_or_train(DEFAULT_MODEL_PATH)


engine = get_engine()
teams = engine.teams

with st.sidebar:
    st.subheader("Walk-forward CV")
    st.caption(f"Trained on {engine.n_matches_} matches up to {engine.last_match_date_}")
    cv = {k: v for k, v in engine.cv_metrics_.items() if isinstance(v, dict)}
    st.dataframe(pd.DataFrame(cv).T[["log_loss", "rps", "brier_score", "accuracy"]])

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
