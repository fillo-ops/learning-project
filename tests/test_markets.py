import numpy as np
import pandas as pd
import pytest

from src.market import (
    asian_handicap,
    asian_markets,
    asian_total,
    bet_expected_value,
    line_movement,
    portfolio_kelly,
    score_matrix_from_samples,
)


@pytest.fixture()
def matrix():
    from scipy.stats import poisson

    g = np.arange(11)
    m = np.outer(poisson.pmf(g, 1.6), poisson.pmf(g, 1.1))
    return m / m.sum()


def _1x2(m):
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()


def test_level_handicap_is_draw_no_bet(matrix):
    h, d, a = _1x2(matrix)
    s = asian_handicap(matrix, 0.0)
    assert s["win"] == pytest.approx(h) and s["push"] == pytest.approx(d) and s["loss"] == pytest.approx(a)
    assert s["fair_odds"] == pytest.approx((h + a) / h)


def test_quarter_line_is_two_half_stakes(matrix):
    q = asian_handicap(matrix, -0.25)
    lo, hi = asian_handicap(matrix, -0.5), asian_handicap(matrix, 0.0)
    for odds in (1.6, 1.9, 2.4):
        ev = bet_expected_value(q, odds)
        assert ev == pytest.approx(0.5 * bet_expected_value(lo, odds) + 0.5 * bet_expected_value(hi, odds))


def test_fair_odds_have_zero_ev(matrix):
    for line in (-1.25, -0.75, 0.25, 1.0):
        s = asian_handicap(matrix, line)
        assert bet_expected_value(s, s["fair_odds"]) == pytest.approx(0.0, abs=1e-9)
    t = asian_total(matrix, 2.25)
    assert bet_expected_value(t, t["fair_odds"]) == pytest.approx(0.0, abs=1e-9)


def test_goal_line_settlement(matrix):
    hg, ag = np.indices(matrix.shape)
    totals = hg + ag
    over_25 = asian_total(matrix, 2.5)
    assert over_25["win"] == pytest.approx(matrix[totals >= 3].sum())
    over_225 = asian_total(matrix, 2.25)
    assert over_225["half_loss"] == pytest.approx(matrix[totals == 2].sum())
    over_3 = asian_total(matrix, 3.0)
    assert over_3["push"] == pytest.approx(matrix[totals == 3].sum())


def test_markets_ladder_and_samples():
    rng = np.random.default_rng(0)
    m = score_matrix_from_samples(rng.poisson(1.5, 50000), rng.poisson(1.0, 50000))
    assert m.sum() == pytest.approx(1.0)
    ladder = asian_markets(m)
    probs = [r["win"] for r in ladder["asian_handicap_home"]]
    assert probs == sorted(probs)  # more generous lines win more often


def test_portfolio_one_bet_per_match_and_exposure_cap():
    bets = [
        {"match": "A-B", "selection": "H", "prob": 0.55, "odds": 2.2},
        {"match": "A-B", "selection": "D", "prob": 0.30, "odds": 3.8},
        {"match": "C-D", "selection": "H", "prob": 0.60, "odds": 2.0},
        {"match": "E-F", "selection": "A", "prob": 0.50, "odds": 2.4},
        {"match": "G-H", "selection": "H", "prob": 0.40, "odds": 2.0},  # no value
    ]
    picks = portfolio_kelly(bets, max_total_exposure=0.05)
    assert len({p["match"] for p in picks}) == len(picks) == 3
    assert sum(p["stake"] for p in picks) == pytest.approx(0.05, abs=1e-3)
    loose = portfolio_kelly(bets, max_total_exposure=1.0)
    assert all(p["stake"] == p["kelly_stake"] for p in loose)


def test_line_movement_detects_informed_closing_prices(season_df):
    df = season_df.copy()
    # Closing prices that lean toward the actual result must score better and show "steam" on winners
    for col, outcome in (("PSCH", "H"), ("PSCD", "D"), ("PSCA", "A")):
        df[col] = np.where(df["FTR"] == outcome, 1.6, 4.5)
    table, summary = line_movement(df)
    assert summary["log_loss_closing"] < summary["log_loss_opening"]
    assert summary["steamed_win_rate_minus_opening_prob"] > 0 > summary["drifting_win_rate_minus_opening_prob"]
    assert len(table) == 20 and {"avg_move", "share_backed"} <= set(table.columns)
    with pytest.raises(ValueError):
        line_movement(season_df.drop(columns=["PSCH", "PSCD", "PSCA"]))


def test_engine_prediction_includes_asian_lines(trained_engine):
    pred = trained_engine.predict_fixture("Arsenal", "Chelsea")
    lines = pred["asian_lines"]["asian_handicap_home"]
    assert any(r["line"] == 0.0 for r in lines)
    assert pd.DataFrame(lines)["fair_odds"].notna().all()
