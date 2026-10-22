# ⚽ Football Match Predictor & Analytics Engine

[![tests](https://github.com/fillo-ops/learning-project/actions/workflows/tests.yml/badge.svg)](https://github.com/fillo-ops/learning-project/actions/workflows/tests.yml)

A machine learning and statistical modelling pipeline that predicts football match outcomes (Home Win, Draw, Away Win), exact scoreline distributions, and value edges against bookmaker odds.

## 🚀 Model Architecture
* **Time-Decayed Dixon-Coles (`src/dixon_coles.py`)**: Team attack/defence strengths with exponential time decay and the low-score correlation adjustment (ρ) for `0-0`, `1-0`, `0-1` and `1-1`. Fitted with an analytic gradient (~30 ms per fit), and refit **walk-forward** every 28 days on past matches only, so its probabilities can be used as features without leaking results.
* **Market Implied Probabilities (`src/market.py`)**: Strips bookmaker overround with the power method (corrects favourite-longshot bias) and computes EV and fractional Kelly stakes.
* **Leak-Free Elo & Rolling Form (`src/features.py`)**: Chronologically replayed Elo ratings (World Football Elo margin multiplier), exponentially weighted rolling form, shots on target and rest days. State is updated only *after* each match's features are recorded.
* **Calibrated Ensemble (`src/model.py`)**: Sigmoid-calibrated gradient boosting blended with Dixon-Coles. The blend weight is learned from out-of-fold walk-forward predictions rather than hard-coded.

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
│   └── ablation.py          # Feature ablation study
├── src/
│   ├── config.py            # Paths and hyperparameters
│   ├── names.py             # Team name canonical normalisation
│   ├── data_loader.py       # CSV loading, validation and cleaning
│   ├── synthetic.py         # Round-robin synthetic league generator
│   ├── db.py                # SQLite persistence (matches, predictions)
│   ├── market.py            # Odds de-vigging, EV & Kelly
│   ├── dixon_coles.py       # Time-decayed Dixon-Coles + walk-forward probabilities
│   ├── features.py          # Elo, rolling form, rest-day and market features
│   ├── evaluate.py          # Log loss, Brier, RPS, ECE
│   ├── model.py             # MatchPredictorEngine (ensemble, prediction, persistence)
│   └── pipeline.py          # Training pipeline & ablation study
└── tests/                   # Pytest suite (incl. look-ahead leakage tests)
```

> ⚠️ Model artifacts are pickles: only load models you trained yourself. Predictions are for learning purposes, not betting advice.
