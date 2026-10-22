# ⚽ Football Match Predictor & Analytics Engine

A modular machine learning and statistical modeling pipeline that predicts football match outcomes (Home Win, Draw, Away Win), exact scoreline distributions, and value edges against bookmaker odds.

## 🚀 Key Accuracy Architecture
* **Time-Decayed Dixon-Coles Model (`src/dixon_coles.py`)**: Estimates dynamic team attack and defense parameters with exponential time decay ($\xi$) and low-score correlation adjustments ($\rho$) for `0-0`, `1-0`, `0-1`, and `1-1` scorelines.
* **Market Implied Probability Engine (`src/market.py`)**: Strips bookmaker overround (vig) to extract true market baseline probabilities and identify positive Expected Value (+EV) edges.
* **Leak-Free Rolling & Elo Features (`src/features.py`)**: Tracks dynamic Elo ratings, exponentially weighted rolling form, goal differentials, and rest days using strict chronological shifting.
* **Calibrated Ensemble Trainer (`src/trainer.py`)**: Blends statistical Poisson/Dixon-Coles probabilities with Gradient Boosting via walk-forward time-series validation and isotonic/sigmoid probability calibration.

## 📂 Project Structure
```text
football-predictor/
├── api/
│   └── main.py              # FastAPI prediction service
├── app/
│   └── dashboard.py         # Interactive Streamlit analytics dashboard
├── scripts/
│   ├── make_synthetic.py    # Synthetic dataset generator
│   ├── train.py             # End-to-end model training pipeline
│   ├── train_from_csv.py    # Train directly from historical CSVs
│   ├── ablation.py          # Feature ablation & importance study
│   ├── predict.py           # Upcoming fixture probability CLI
│   ├── auto_learn.py        # Automated retraining loop
│   └── update.py            # Automated match result updater
├── src/
│   ├── config.py            # Paths and hyperparameter configuration
│   ├── db.py                # SQLite database layer
│   ├── data_loader.py       # Data fetching and cleaning
│   ├── csv_loader.py        # Football-Data.co.uk CSV ingestion
│   ├── names.py             # Team name canonical normalization
│   ├── dixon_coles.py       # Time-decayed Dixon-Coles goal model
│   ├── market.py            # Bookmaker odds de-vigging & EV calculator
│   ├── features.py          # Elo, rolling form, and rest-day engineering
│   ├── model.py             # Core predictor wrapper
│   ├── trainer.py           # Walk-forward ensemble & calibration trainer
│   ├── evaluate.py          # Log-loss, Brier score, and calibration metrics
│   ├── tracker.py           # Prediction vs. closing odds performance tracker
│   └── service.py           # Shared prediction business logic
└── tests/                   # Pytest unit and integration test suite
