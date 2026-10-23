import numpy as np
import pandas as pd
import pytest

from src.competitions import CHAMPIONS_LEAGUE, champions_league_teams, league_phase_pairings
from src.dixon_coles import walk_forward_dc_probs
from src.features import FeatureBuilder
from src.names import normalize_team_name
from src.synthetic import TEAMS, generate_synthetic_matches


@pytest.fixture(scope="module")
def ucl_df():
    return generate_synthetic_matches(n_seasons=1, seed=1, include_champions_league=True)


def test_thirty_six_clubs_with_english_qualifiers():
    clubs = champions_league_teams()
    assert len(clubs) == len(set(clubs)) == 36
    league = {t[0] for t in TEAMS}
    assert {"Arsenal", "Liverpool", "Manchester City", "Chelsea", "Newcastle United", "Tottenham"} <= league & set(
        clubs
    )


def test_league_phase_is_balanced_without_repeats():
    pairings = league_phase_pairings(list(range(36)), np.random.default_rng(3))
    assert len(pairings) == 8
    games = [g for md in pairings for g in md]
    assert len({frozenset(g) for g in games}) == len(games) == 144  # no repeated opponents
    for md in pairings:
        assert sorted(t for g in md for t in g) == list(range(36))  # everyone plays once per matchday
    homes = pd.Series([h for h, _ in games]).value_counts()
    assert (homes == 4).all() and len(homes) == 36


def test_full_champions_league_season(ucl_df):
    ucl = ucl_df[ucl_df["Competition"] == CHAMPIONS_LEAGUE]
    stages = ucl["Stage"].value_counts()
    assert stages["League phase"] == 144
    assert stages["Knockout play-off"] == 16 and stages["Round of 16"] == 16
    assert stages["Quarter-final"] == 8 and stages["Semi-final"] == 4 and stages["Final"] == 1
    final = ucl[ucl["Stage"] == "Final"].iloc[0]
    assert final["Neutral"] == 1
    assert final["Date"] == ucl["Date"].max()
    assert set(ucl["HomeTeam"]) | set(ucl["AwayTeam"]) == set(champions_league_teams())


def test_league_only_generation_is_unchanged(season_df):
    assert "Competition" not in season_df.columns
    assert len(season_df) == 380


def test_champions_league_games_stay_out_of_the_league_table(ucl_df):
    fb = FeatureBuilder()
    feats = fb.transform_historical(ucl_df, dc_probs=walk_forward_dc_probs(ucl_df))
    assert sum(fb.table_.played.values()) == 2 * 380  # only league matches counted
    assert set(fb.table_.points) == {t[0] for t in TEAMS}
    ucl_rows = feats["Competition"] == CHAMPIONS_LEAGUE
    assert (feats.loc[ucl_rows, ["h_dead_rubber", "a_dead_rubber", "stake_gap_diff"]] == 0).all().all()
    # ...but they still count as fixtures for congestion
    assert "Bayern Munich" in fb.elo_ and fb.league_dates_["Arsenal"][-1] >= feats["Date"].min()


def test_common_scale_ranks_european_giants_above_minnows(ucl_df):
    from src.team_ratings import TeamUnitRatings

    table = TeamUnitRatings().fit(ucl_df).league_table()
    for big in ("Real Madrid", "Bayern Munich", "Paris Saint-Germain"):
        for small in ("Kairat Almaty", "Qarabag", "Pafos"):
            assert table.at[big, "attack"] > table.at[small, "attack"]


@pytest.mark.parametrize(
    "raw,canonical",
    [("Bayern München", "Bayern Munich"), ("PSG", "Paris Saint-Germain"), ("Inter Milan", "Inter"),
     ("Sporting Lisbon", "Sporting CP"), ("FC København", "Copenhagen"), ("Bodø/Glimt", "Bodo/Glimt")],
)  # fmt: skip
def test_club_aliases(raw, canonical):
    assert normalize_team_name(raw) == canonical


def test_analytics_competition_filter(ucl_df):
    from src.analytics import TeamAnalytics

    an = TeamAnalytics(ucl_df, n_sims=2000)
    assert an.competitions == ["Premier League", CHAMPIONS_LEAGUE]
    assert len(an.teams_in(CHAMPIONS_LEAGUE)) == 36 and len(an.teams_in("Premier League")) == 20
    assert len(an.league_table(CHAMPIONS_LEAGUE)) == 36
    with pytest.raises(ValueError):
        an.teams_in("Europa League")
    res = an.simulate("Real Madrid", "Arsenal")
    assert sum(res["outcome"].values()) == pytest.approx(1.0, abs=1e-3)
