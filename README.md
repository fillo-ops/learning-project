# ⚽ Football Match Predictor & Analytics Engine

[![tests](https://github.com/fillo-ops/learning-project/actions/workflows/tests.yml/badge.svg)](https://github.com/fillo-ops/learning-project/actions/workflows/tests.yml)

A machine learning and statistical modelling pipeline that predicts football match outcomes (Home Win, Draw, Away Win), exact scoreline distributions, and value edges against bookmaker odds.

## 🚀 Model Architecture
* **Time-Decayed Dixon-Coles (`src/dixon_coles.py`)**: Team attack/defence strengths with exponential time decay and the low-score correlation adjustment (ρ) for `0-0`, `1-0`, `0-1` and `1-1`. Fitted with an analytic gradient (~30 ms per fit), and refit **walk-forward** every 28 days on past matches only, so its probabilities can be used as features without leaking results.
* **Market Implied Probabilities (`src/market.py`)**: Strips bookmaker overround with the power method (corrects favourite-longshot bias) and computes EV and fractional Kelly stakes.
* **Leak-Free Elo & Rolling Form (`src/features.py`)**: Chronologically replayed Elo ratings (World Football Elo margin multiplier), exponentially weighted rolling form, shots on target and rest days. State is updated only *after* each match's features are recorded.
* **Calibrated Ensemble (`src/model.py`)**: Sigmoid-calibrated gradient boosting blended with Dixon-Coles. The blend weight is learned from out-of-fold walk-forward predictions rather than hard-coded.

## 🧠 Team Units, Tactics & Matchup Simulation
Measures each team's **attack**, **midfield** and **defence** across the season, pits them against each other, and simulates matchups thousands of times.

* **Unit ratings (`src/team_ratings.py`)**: Three opponent-adjusted, time-decayed Poisson models (shots, shots on target, goals). Every team gets indices where **100 = league average**:
  `attack`, `defence`, `midfield` (net shot-volume control), `chance_creation`, `shot_suppression`, `chance_quality`, `finishing`, `shot_quality_allowed`, `goalkeeping`. A team's expected goals splits exactly into *midfield/shot volume × chance quality × finishing*, so you can see which battle decides a match.
* **Season timeline**: Weekly walk-forward snapshots (each one uses only earlier matches) show how each unit evolved during the season.
* **Tactical profiles (`src/tactics.py`)**: Style labels (dominant, reactive, counter-attacking, compact block, clinical/wasteful, physical…) from shot share, accuracy, conversion, corners and fouls, plus matchup notes.
* **Players (`src/players.py`)**: Optional player-season CSV (`data/players.csv`, columns `player, team, position, minutes, goals, assists, shots, shots_on_target, key_passes, tackles, interceptions`). Players are rated per 90 within their position (50 = average), rolled up into team units, and **absences** change the simulation (weights in `PLAYER_CONFIG`). Each simulation also gives anytime-scorer probabilities.
* **Monte Carlo simulator (`src/simulation.py`)**: Each simulated match draws shots → shots on target → goals for both sides. Output includes each attack's goal distribution (P(1+), P(2+), P(3+), 90th percentile), how each defence fares (clean sheet %, expected conceded vs its usual workload), the goals won or lost in midfield, unit-vs-unit battles, and a **midfield sensitivity** table showing how goals and results move if the midfield battle tilts.

```bash
python -m scripts.simulate --home Arsenal --away Chelsea                       # full matchup simulation
python -m scripts.simulate --home Arsenal --away Chelsea --home-absent "Arsenal FWD1" --midfield-shift 0.1
python -m scripts.simulate --team Brighton                                     # season unit report + attack vs every defence
python -m scripts.simulate --league                                            # unit ratings for all teams
```
API: `/units`, `/teams/{team}/report`, `/teams/{team}/attack-vs-defences`, `/simulate?home=..&away=..&home_absent=..&midfield_shift=..`. The dashboard has a **Team Units & Simulation** tab.

> Without a player CSV, a synthetic squad table is generated for demonstration. Football-Data.co.uk files have no possession or player data, so "midfield" is measured through shot volume won and conceded.

## 📊 Evaluation
`python -m scripts.train` writes `reports/cv_evaluation.json`, comparing every model **on the same out-of-sample matches**: the ensemble, gradient boosting alone, Dixon-Coles alone, a base-rate baseline and the de-vigged bookmaker market. Metrics: log loss, Ranked Probability Score (RPS), Brier score, accuracy and expected calibration error.

On the bundled synthetic data the ensemble beats Dixon-Coles and the base rate, but not the bookmaker. That is expected, because the synthetic odds are generated from the true scoring rates. Run it on real Football-Data.co.uk CSVs for meaningful numbers.

## 🏁 Quick Start
```bash
pip install -r requirements.txt

python -m scripts.train                       # trains on data/matches.csv (synthetic data generated if missing)
python -m scripts.train --csv E0.csv          # or a Football-Data.co.uk season CSV
python -m scripts.predict --home Arsenal --away Chelsea --odds 2.05 3.50 3.60 --date 2026-05-02
python -m scripts.ablation                    # walk-forward feature ablation study
uvicorn api.main:app --reload                 # REST API: /predict, /teams, /health
streamlit run app/dashboard.py                # interactive dashboard
pytest                                        # test suite
```

## 📂 Project Structure
```text
├── api/main.py              # FastAPI prediction service (model cached in memory)
├── app/dashboard.py         # Streamlit analytics dashboard
├── scripts/
│   ├── make_synthetic.py    # Synthetic dataset generator
│   ├── train.py             # End-to-end training & evaluation
│   ├── predict.py           # Fixture prediction CLI (optionally logs to SQLite)
│   ├── simulate.py          # Unit ratings, team reports and matchup simulation CLI
│   └── ablation.py          # Feature ablation study
├── src/
│   ├── config.py            # Paths and hyperparameters
│   ├── names.py             # Team name canonical normalisation
│   ├── data_loader.py       # CSV loading, validation and cleaning
│   ├── synthetic.py         # Synthetic league (with midfield/style traits) and squads
│   ├── db.py                # SQLite persistence (matches, predictions)
│   ├── market.py            # Odds de-vigging, EV & Kelly
│   ├── dixon_coles.py       # Time-decayed Dixon-Coles + walk-forward probabilities
│   ├── features.py          # Elo, rolling form, rest-day and market features
│   ├── evaluate.py          # Log loss, Brier, RPS, ECE
│   ├── model.py             # MatchPredictorEngine (ensemble, prediction, persistence)
│   ├── team_ratings.py      # Attack / midfield / defence unit ratings + season timeline
│   ├── tactics.py           # Tactical style profiles and matchup notes
│   ├── players.py           # Player ratings, unit roll-ups and absence effects
│   ├── simulation.py        # Monte Carlo matchup simulator
│   ├── analytics.py         # TeamAnalytics facade over the above
│   └── pipeline.py          # Training pipeline & ablation study
└── tests/                   # Pytest suite (incl. look-ahead leakage tests)
```

> ⚠️ Model artifacts are pickles: only load models you trained yourself. Predictions are for learning purposes, not betting advice.
