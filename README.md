# ⚽ Football Match Predictor & Analytics Engine

[![tests](https://github.com/fillo-ops/learning-project/actions/workflows/tests.yml/badge.svg)](https://github.com/fillo-ops/learning-project/actions/workflows/tests.yml)

A machine learning and statistical modelling pipeline that predicts football match outcomes (Home Win, Draw, Away Win), exact scoreline distributions, and value edges against bookmaker odds.

## 🚀 Model Architecture
* **Time-Decayed Dixon-Coles (`src/dixon_coles.py`)**: Team attack/defence strengths with exponential time decay and the low-score correlation adjustment (ρ) for `0-0`, `1-0`, `0-1` and `1-1`. Fitted with an analytic gradient (~30 ms per fit), and refit **walk-forward** every 28 days on past matches only, so its probabilities can be used as features without leaking results.
* **Market Implied Probabilities (`src/market.py`)**: Strips bookmaker overround with the power method (corrects favourite-longshot bias) and computes EV and fractional Kelly stakes.
* **Leak-Free Elo & Rolling Form (`src/features.py`)**: Chronologically replayed Elo ratings (World Football Elo margin multiplier), exponentially weighted rolling form, shots on target, rest days and games in the last 14 days (all competitions with a fixtures calendar), and league-table stakes (points gap, distance to the nearest title / top-four / relegation boundary, "dead rubber" flags). Elo is regressed 20% toward the mean between seasons. State is updated only *after* each match's features are recorded.
* **Calibrated Ensemble (`src/model.py`)**: Sigmoid-calibrated gradient boosting blended with Dixon-Coles. The blend weight is learned from out-of-fold walk-forward predictions rather than hard-coded.

## 🧠 Team Units, Tactics & Matchup Simulation
Measures each team's **attack**, **midfield** and **defence** across the season, pits them against each other, and simulates matchups thousands of times.

* **Unit ratings (`src/team_ratings.py`)**: Opponent-adjusted, time-decayed models for chances created, shots, shots on target and goals, plus a possession model. Every team gets indices where **100 = league average**:
  * **Attack**: goals *and* shots on target, blended 50/50 (`UNIT_CONFIG["attack_sot_weight"]`). Shots on target are far less noisy than goals, so the rating settles faster. On the synthetic data it tracks true attacking strength better than goals alone (rank correlation 0.91 vs 0.85). `goal_threat` and `shots_on_target` are also shown separately.
  * **Midfield**: **chances created** won vs conceded combined with **possession share** (expected possession % against an average side). It falls back to shot volume if the data has neither.
  * **Defence**: goals prevented, plus `chance_suppression`, `shot_quality_allowed` and `goalkeeping`.
  
  A team's expected goals splits exactly into *midfield (chances created) × chance quality (shots on target per chance) × finishing*, so you can see which battle decides a match.
* **Season timeline**: Weekly walk-forward snapshots (each one uses only earlier matches) show how each unit evolved during the season.
* **Tactical profiles (`src/tactics.py`)**: Style labels (dominant, reactive, counter-attacking, compact block, clinical/wasteful, physical…) from shot share, accuracy, conversion, corners and fouls, plus matchup notes.
* **Players (`src/players.py`)**: Optional player-season CSV (`data/players.csv`, columns `player, team, position, minutes, goals, assists, shots, shots_on_target, key_passes, tackles, interceptions`). Players are rated per 90 within their position (50 = average), rolled up into team units, and **absences** change the simulation (weights in `PLAYER_CONFIG`). Each simulation also gives anytime-scorer probabilities.
* **Possession → chances**: the model measures how much a team's chance ratio rises when it gets more possession than expected in a match (comparing each match with what was expected, so team quality doesn't bias it). This drives the possession "what if" analysis.
* **Monte Carlo simulator (`src/simulation.py`)**: Each simulated match draws chances created → shots on target → goals for both sides. Output includes each attack's goal distribution (P(1+), P(2+), P(3+), 90th percentile), how each defence fares (clean sheet %, expected conceded vs its usual workload), the goals won or lost in midfield, unit-vs-unit battles, and a **possession sensitivity** table showing how chances, goals and results move if the home side has 5 or 10 points more or less possession.

```bash
python -m scripts.simulate --home Arsenal --away Chelsea                       # full matchup simulation
python -m scripts.simulate --home Arsenal --away Chelsea --home-absent "Arsenal FWD1" --possession-shift 10
python -m scripts.simulate --team Brighton                                     # season unit report + attack vs every defence
python -m scripts.simulate --league                                            # unit ratings for all teams
```
API: `/units`, `/teams/{team}/report`, `/teams/{team}/attack-vs-defences`, `/simulate?home=..&away=..&home_absent=..&possession_shift=..`. The dashboard has a **Team Units & Simulation** tab.

### Match data columns
Possession and chances created are optional extra columns in the match CSV (Football-Data.co.uk files don't include them; sources such as FBref do):

| Column | Meaning |
|---|---|
| `HPoss`, `APoss` | Home / away possession (% or 0-1 share) |
| `HCC`, `ACC` | Home / away chances created (key passes / shot-creating actions) |
| `HST`, `AST` | Shots on target (required) |
| `HnpxG`, `AnpxG` (or `HxG`, `AxG`) | Non-penalty xG (attack index, chance quality, finishing) |
| `HPSxG`, `APSxG` | Post-shot xG of each side's shots (isolates goalkeeping from finishing) |
| `HPKatt`, `APKatt`, `HPK`, `APK` | Penalties awarded / scored (stripped from open-play goals, modelled separately) |
| `HR`, `AR`, `HY`, `AY`, `Referee` | Cards and referee (red-card risk in the simulator; Football-Data includes these) |
| `PSCH`/`PSCD`/`PSCA` or `AvgCH`/`AvgCD`/`AvgCA` | Closing odds (line-movement analysis) |

Without `HPoss`/`HCC`, midfield is measured through shot volume won and conceded. Without a player CSV, a synthetic squad table is generated for demonstration.

xG / PSxG are imported from files (e.g. FBref or Understat exports) rather than scraped: FBref's terms prohibit automated scraping and Understat has no official API.

Optional context files in `data/`:
* `fixtures.csv` (`Date, Team, Competition`): cup and European games, so rest days and games-in-14-days count every competition.
* `managers.csv` (`Team, Date`): appointments. Older matches of that team are down-weighted in the ratings, rolling form resets, and the tactical profile only uses matches under the new manager.

## 🏆 Champions League
The demo data covers the 20 Premier League clubs **plus the 36 clubs of the 2025/26 Champions League league phase** (30 from outside England), 50 clubs in all:

* Each season includes a full Champions League in the current format: an 8-match Swiss league phase (no repeated opponents, 4 home / 4 away), knockout play-off, round of 16, quarter-finals, semi-finals and a neutral-venue final. Rows carry `Competition` and `Stage` columns.
* Ratings are fitted on **all competitions together**, so clubs from different leagues share one scale through their European meetings. Indices are relative to the average of all clubs in the data.
* Champions League matches update Elo, form and fixture congestion, but not the Premier League table.
* Club name aliases cover common spellings (`Bayern München`, `PSG`, `Inter Milan`, `Sporting Lisbon`, `FC København`...).
* Filter by competition: `python -m scripts.simulate --league --competition "Champions League"`, the API's `/competitions`, `/teams?competition=...` and `/units?competition=...`, or the dashboard's **Competition** selector.

Clubs from outside England only play about 10 European matches a season in this data, so their ratings are noisier than Premier League clubs'. With real data, add their domestic league files too, with a `Competition` column, for sharper ratings. Football-Data.co.uk has no Champions League files; results can be imported from e.g. openfootball as a CSV with `Competition = "Champions League"`. The club list and demo strengths live in `src/competitions.py`; update them for a new season. `python -m scripts.make_synthetic --league-only` generates the old Premier-League-only demo.

## 🎲 Game-State Simulation
Matches are simulated in six 15-minute blocks (`src/game_state.py`):
* **Scoreline effects**: a leading side creates fewer but better chances (sits deeper, counters); a trailing side creates more, lower-quality chances. Multipliers are in `GAME_STATE_CONFIG`. They are literature-based priors: minute-level data would be needed to fit them.
* **Red cards** arrive per block at each team's (shrunk) red-card rate, scaled by referee strictness; a side down to 10 men loses 30% of its chance volume and the opponent gains 30%.
* **Goal timing**: rates rise through the match (`block_profile`).
* **Calibration**: base rates are re-scaled so full-match expected goals still match the ratings. Game state changes *when* goals come and which scorelines happen, not how many.
* Output adds who scores first, win probability after scoring first, comeback wins, share of the match spent leading / level / trailing, red-card probabilities and goals per 15-minute period. `--static` reverts to the single full-match draw.

## 💹 Markets & Staking
* **Asian handicaps and goal lines** (quarter lines with half-win / half-push / half-loss settlement) are priced from the full joint score distribution, in both `/predict` (Dixon-Coles) and `/simulate` (game-state simulation).
* **Portfolio Kelly**: one bet per match (outcomes in the same match are correlated), fractional Kelly, and a cap on the total bankroll in play across a matchday.
* **Line movement**: opening → closing de-vigged moves per team, whether closing prices were sharper, and whether "steamed" sides beat their opening price.

```bash
python -m scripts.markets price --home Arsenal --away Chelsea --ah -0.75 2.10 --total 2.75 1.90
python -m scripts.markets portfolio --bets bets.csv --cap 0.12      # bets.csv: match,selection,prob,odds
python -m scripts.markets steam
```

## 🛠️ Operations
* **Typed API**: Pydantic response models (`api/schemas.py`) give typed OpenAPI schemas at `/docs`; simulation results are cached per `(home, away, what-if)` so repeated dashboard/API calls are served from memory.
* **Data checks**: `python -m scripts.update --league E0 --season 2526 [--no-train | --check-only]` downloads a Football-Data season, validates headers, scores, results, shots and odds, merges it and retrains. A weekly GitHub Action (`data-check.yml`) runs the check and opens an issue if validation fails. Delete the synthetic `data/matches.csv` before merging real data.
* **Docker**: `docker compose up` starts the API on `:8000` and the dashboard on `:8501` with shared `data/`, `models/` and `reports/` volumes.

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
python -m scripts.update --league E0 --season 2526   # fetch, validate, merge real data and retrain
uvicorn api.main:app --reload                 # REST API, docs at /docs
docker compose up                             # API (:8000) + dashboard (:8501)
streamlit run app/dashboard.py                # interactive dashboard
pytest                                        # test suite
```

## 📄 Single-File Version
`football_predictor_complete.py` bundles the whole system (every `src/` module, the CLI scripts as subcommands, the API and the dashboard) into one file you can copy anywhere next to an empty folder:

```bash
python football_predictor_complete.py                                   # first run trains, then a demo
python football_predictor_complete.py predict --home Arsenal --away Chelsea --odds 2.05 3.50 3.60
python football_predictor_complete.py simulate --home "Real Madrid" --away Arsenal
python football_predictor_complete.py simulate --league --competition "Champions League"
python football_predictor_complete.py markets price --home Arsenal --away Chelsea
python football_predictor_complete.py train | ablation | synthetic | update
python football_predictor_complete.py api          # http://localhost:8000/docs
python football_predictor_complete.py dashboard    # http://localhost:8501
```

It is generated, not hand-edited: change the package, then run `python tools/build_single_file.py`. The test suite fails if the bundle is out of date.

## 📂 Project Structure
```text
├── api/main.py              # FastAPI service (typed responses, cached simulations)
├── api/schemas.py           # Pydantic response models
├── app/dashboard.py         # Streamlit analytics dashboard
├── scripts/
│   ├── make_synthetic.py    # Synthetic dataset generator
│   ├── train.py             # End-to-end training & evaluation
│   ├── predict.py           # Fixture prediction CLI (optionally logs to SQLite)
│   ├── simulate.py          # Unit ratings, team reports and matchup simulation CLI
│   ├── markets.py           # Asian lines, portfolio staking, line movement
│   ├── update.py            # Download + validate + merge Football-Data seasons
│   └── ablation.py          # Feature ablation study
├── src/
│   ├── config.py            # Paths and hyperparameters
│   ├── names.py             # Team name canonical normalisation
│   ├── data_loader.py       # CSV loading, validation and cleaning
│   ├── synthetic.py         # Synthetic league (with midfield/style traits) and squads
│   ├── db.py                # SQLite persistence (matches, predictions)
│   ├── market.py            # De-vigging, Kelly, Asian lines, portfolio staking, line movement
│   ├── dixon_coles.py       # Time-decayed Dixon-Coles + walk-forward probabilities
│   ├── features.py          # Elo (with season regression), form, congestion, table stakes, market
│   ├── evaluate.py          # Log loss, Brier, RPS, ECE
│   ├── model.py             # MatchPredictorEngine (ensemble, prediction, persistence)
│   ├── team_ratings.py      # Attack / midfield / defence unit ratings + season timeline
│   ├── tactics.py           # Tactical style profiles and matchup notes
│   ├── players.py           # Player ratings, unit roll-ups and absence effects
│   ├── simulation.py        # Monte Carlo matchup simulator
│   ├── game_state.py        # 15-minute game states, red cards, discipline model
│   ├── context.py           # League table stakes, fixture congestion, seasons, manager changes
│   ├── competitions.py      # Champions League clubs, league-phase draw, standings
│   ├── validation.py        # Data schema and sanity checks
│   ├── analytics.py         # TeamAnalytics facade over the above
│   └── pipeline.py          # Training pipeline & ablation study
├── tests/                   # Pytest suite (incl. look-ahead leakage tests)
├── tools/build_single_file.py   # Builds the single-file version
└── football_predictor_complete.py   # Generated single-file version
```

> ⚠️ Model artifacts are pickles: only load models you trained yourself. Predictions are for learning purposes, not betting advice.
