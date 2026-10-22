import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src.dixon_coles import TimeDecayedDixonColes, _make_objective, walk_forward_dc_probs
from src.evaluate import ranked_probability_score
from src.features import FEATURE_COLS, FeatureBuilder, elo_goal_multiplier


def test_dixon_coles_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    n, m = 6, 200
    h_idx = rng.integers(0, n, m)
    a_idx = (h_idx + rng.integers(1, n, m)) % n
    hg, ag = rng.poisson(1.4, m), rng.poisson(1.1, m)
    obj = _make_objective(h_idx, a_idx, hg, ag, rng.uniform(0.2, 1.0, m), n, 0.01)
    x = np.concatenate([rng.normal(0, 0.2, 2 * n), [0.25, -0.08]])
    analytic = obj(x)[1]
    numeric = approx_fprime(x, lambda p: obj(p)[0], 1e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-4)


def test_dixon_coles_recovers_strength_ordering(season_df):
    model = TimeDecayedDixonColes().fit(season_df)
    assert model.attack_["Manchester City"] > model.attack_["Southampton"]
    assert model.defense_["Manchester City"] < model.defense_["Southampton"]
    probs = model.predict_1x2("Manchester City", "Southampton")
    assert np.isclose(sum(probs.values()), 1.0)
    assert probs["H"] > 0.6


def test_walk_forward_dc_has_no_lookahead(season_df):
    base = walk_forward_dc_probs(season_df, min_matches=60)
    # Rewrite every result from row 200 onward; earlier probabilities must not move.
    tampered = season_df.copy()
    tampered.loc[200:, ["FTHG", "FTAG"]] = [9, 0]
    after = walk_forward_dc_probs(tampered, min_matches=60)
    cutoff = season_df["Date"].iloc[200]
    earlier = (season_df["Date"] <= cutoff).to_numpy()
    np.testing.assert_allclose(base[earlier], after[earlier], equal_nan=True)
    assert np.isnan(base[:60]).all()


def test_features_do_not_see_own_result(season_df):
    dc = walk_forward_dc_probs(season_df)
    i = 250
    tampered = season_df.copy()
    tampered.loc[i, ["FTHG", "FTAG", "FTR"]] = [7, 0, "H"]
    a = FeatureBuilder().transform_historical(season_df, dc_probs=dc)
    b = FeatureBuilder().transform_historical(tampered, dc_probs=dc)
    pd.testing.assert_series_equal(a.loc[i, FEATURE_COLS], b.loc[i, FEATURE_COLS])


def test_elo_multiplier_monotonic():
    values = [elo_goal_multiplier(gd) for gd in range(6)]
    assert values[0] == values[1] == 1.0
    assert all(x < y for x, y in itertools.pairwise(values[1:]))


def test_rps_known_values():
    y = np.array([0])
    assert ranked_probability_score(y, np.array([[1.0, 0.0, 0.0]])) == 0.0
    # Mass on an away win is punished more than mass on a draw when the home side wins.
    assert ranked_probability_score(y, np.array([[0.0, 0.0, 1.0]])) > ranked_probability_score(
        y, np.array([[0.0, 1.0, 0.0]])
    )


def test_end_to_end_training_and_prediction(trained_engine):
    metrics = trained_engine.cv_metrics_
    assert {"ensemble", "dixon_coles_baseline", "base_rate_baseline"} <= metrics.keys()
    assert 0.0 <= metrics["ml_weight"] <= 1.0
    pred = trained_engine.predict_fixture("Arsenal", "Chelsea", odds=(2.05, 3.50, 3.60), match_date="2024-06-01")
    probs = pred["probabilities"]
    assert np.isclose(probs["home_win"] + probs["draw"] + probs["away_win"], 1.0, atol=1e-3)
    assert set(pred["value_bets"]) == {"H", "D", "A"}


def test_prediction_without_odds(trained_engine):
    pred = trained_engine.predict_fixture("Man City", "Spurs")  # aliases resolve
    assert pred["home_team"] == "Manchester City"
    assert pred["value_bets"] == {}


@pytest.mark.parametrize(
    "home,away,odds",
    [("Atlantis FC", "Chelsea", None), ("Arsenal", "Arsenal", None), ("Arsenal", "Chelsea", (0.5, 3.0, 3.0))],
)
def test_prediction_rejects_bad_input(trained_engine, home, away, odds):
    with pytest.raises(ValueError):
        trained_engine.predict_fixture(home, away, odds=odds)


def test_engine_save_load_round_trip(trained_engine, tmp_path):
    path = tmp_path / "engine.pkl"
    trained_engine.save(path)
    loaded = type(trained_engine).load(path)
    assert loaded.predict_fixture("Arsenal", "Chelsea") == trained_engine.predict_fixture("Arsenal", "Chelsea")
