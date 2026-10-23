import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.context import FixtureCalendar, LeagueTable, load_manager_changes
from src.dixon_coles import walk_forward_dc_probs
from src.features import CONTEXT_COLS, FEATURE_COLS, FeatureBuilder
from src.synthetic import FINISHING, TEAMS, generate_synthetic_matches
from src.team_ratings import TeamUnitRatings


def test_league_table_dead_rubber():
    table = LeagueTable(expected_teams=4)  # 6 games each
    # A: 15 pts, B: 12, C: 3, D: 0 after 5 rounds-ish
    for _ in range(5):
        table.update("A", "D", 1, 0)
    for _ in range(4):
        table.update("B", "C", 1, 0)
    table.update("C", "D", 1, 0)
    stakes = table.stakes("B")
    assert stakes["games_left"] == 2
    assert set(stakes) >= {"dead_rubber", "stake_gap", "pts_to_top4", "drop_margin"}
    assert table.stakes("A")["pts_to_leader"] == 0


def test_context_features_are_in_the_model():
    assert set(CONTEXT_COLS) <= set(FEATURE_COLS)


def test_calendar_congestion_changes_rest_and_games14(season_df):
    dc = walk_forward_dc_probs(season_df)
    base = FeatureBuilder().transform_historical(season_df, dc_probs=dc)
    i = 200
    row = base.iloc[i]
    midweek = row["Date"] - pd.Timedelta(days=3)
    calendar = FixtureCalendar(pd.DataFrame({"Date": [midweek], "Team": [row["HomeTeam"]], "Competition": ["UCL"]}))
    busy = FeatureBuilder(calendar=calendar).transform_historical(season_df, dc_probs=dc)
    assert busy.loc[i, "h_games14"] == base.loc[i, "h_games14"] + 1
    assert busy.loc[i, "rest_diff"] < base.loc[i, "rest_diff"]


def test_season_regression_pulls_elo_to_mean():
    df = generate_synthetic_matches(n_seasons=2, seed=3)
    dc = walk_forward_dc_probs(df)
    feats = FeatureBuilder(season_regression=0.5).transform_historical(df, dc_probs=dc)
    no_reg = FeatureBuilder(season_regression=0.0).transform_historical(df, dc_probs=dc)
    second = feats["Date"] > feats["Date"].iloc[379] + pd.Timedelta(days=45)
    first_match = second.to_numpy().argmax()
    assert abs(feats.loc[first_match, "h_elo"] - 1500) < abs(no_reg.loc[first_match, "h_elo"] - 1500)


def test_manager_change_resets_form(season_df):
    dc = walk_forward_dc_probs(season_df)
    i = 300
    team = season_df.loc[i, "HomeTeam"]
    changes = load_manager_changes(pd.DataFrame({"Team": [team], "Date": [season_df.loc[i, "Date"]]}))
    plain = FeatureBuilder().transform_historical(season_df, dc_probs=dc)
    reset = FeatureBuilder(manager_changes=changes).transform_historical(season_df, dc_probs=dc)
    assert reset.loc[i, "h_roll_gf"] == pytest.approx(1.35)  # back to the neutral prior
    assert plain.loc[i, "h_roll_gf"] != pytest.approx(1.35)


def test_manager_change_down_weights_old_matches(season_df):
    team = "Arsenal"
    when = season_df["Date"].iloc[250]
    a = TeamUnitRatings().fit(season_df)
    b = TeamUnitRatings().fit(season_df, manager_changes={team: when})
    assert a.unit_profile(team) != b.unit_profile(team)


def test_xg_and_psxg_sharpen_finishing(season_df):
    names = [n for n, _, _ in TEAMS]
    truth = [FINISHING.get(n, 1.0) for n in names]
    with_xg = TeamUnitRatings().fit(season_df).league_table()
    without = TeamUnitRatings().fit(season_df.drop(columns=["HnpxG", "AnpxG", "HPSxG", "APSxG"])).league_table()
    corr_xg = spearmanr([with_xg.at[n, "finishing"] for n in names], truth).statistic
    corr_plain = spearmanr([without.at[n, "finishing"] for n in names], truth).statistic
    assert corr_xg > corr_plain


def test_penalties_modelled_separately(analytics):
    r = analytics.ratings
    assert "pens" in r.layers_ and r.penalty_conversion_
    home = analytics.simulator.side_rates("Arsenal", "Chelsea")["home"]
    assert home["pen_rate"] > 0 and 0.4 <= home["pen_conv"] <= 0.95
    assert home["expected_goals"] == pytest.approx(home["xg"] + home["pen_rate"] * home["pen_conv"])
    assert "penalties_won" in r.unit_profile("Arsenal")
    assert np.isfinite(r.layers_["goals"].intercept)
