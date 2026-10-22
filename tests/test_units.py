import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import spearmanr

from src.players import PlayerRatings
from src.synthetic import MIDFIELD
from src.team_ratings import TeamUnitRatings, fit_poisson_layer, season_timeline


def test_poisson_layer_gradient_matches_numerical():
    # Rebuild the objective via a tiny fit and compare against finite differences of the NLL.
    rng = np.random.default_rng(3)
    n, m = 5, 120
    hi = rng.integers(0, n, m)
    ai = (hi + rng.integers(1, n, m)) % n
    hy, ay = rng.poisson(12, m).astype(float), rng.poisson(10, m).astype(float)
    w = rng.uniform(0.3, 1.0, m)

    def nll(p):
        gen, allow = p[:n] - p[:n].mean(), p[n : 2 * n] - p[n : 2 * n].mean()
        lh = p[2 * n] + p[2 * n + 1] + gen[hi] + allow[ai]
        la = p[2 * n] + gen[ai] + allow[hi]
        return -np.sum(w * (hy * lh - np.exp(lh) + ay * la - np.exp(la))) + 0.01 * (gen @ gen + allow @ allow)

    gen, allow, c, h = fit_poisson_layer(hi, ai, hy, ay, w, n, 0.01)
    x = np.concatenate([gen, allow, [c, h]])
    # At the optimum the gradient (projected onto the centred subspace) should vanish.
    g = approx_fprime(x, nll, 1e-6)
    g[:n] -= g[:n].mean()
    g[n : 2 * n] -= g[n : 2 * n].mean()
    assert np.abs(g).max() < 1e-2


def test_unit_ratings_recover_hidden_midfield(analytics):
    table = analytics.ratings.league_table()
    corr = spearmanr([table.at[t, "midfield"] for t in MIDFIELD], list(MIDFIELD.values())).statistic
    assert corr > 0.8
    assert table.at["Manchester City", "attack"] > table.at["Southampton", "attack"]


def test_goal_decomposition_telescopes(analytics):
    r = analytics.ratings
    dec = r.goal_decomposition("Arsenal", "Everton", at_home=True)
    g = r.layers_["goals"]
    implied = np.exp(g.intercept + g.home) * dec["net_effect"].prod()
    assert implied == pytest.approx(r.layers_["goals"].rate("Arsenal", "Everton", True), rel=0.01)


def test_simulation_matches_expected_rates(analytics):
    res = analytics.simulate("Arsenal", "Everton")
    for side in ("home", "away"):
        xg = res[side]["expected_rates"]["xg"]
        assert res[side]["attack"]["mean_goals"] == pytest.approx(xg, rel=0.05)
        assert sum(res[side]["attack"]["goal_distribution"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(res["outcome"].values()) == pytest.approx(1.0, abs=1e-3)
    # The defence summary is the mirror image of the opposing attack
    assert res["home"]["defence"]["expected_conceded"] == pytest.approx(res["away"]["attack"]["mean_goals"], abs=1e-3)
    assert analytics.simulate("Arsenal", "Everton") == res  # seeded, reproducible


def test_midfield_shift_moves_goals_monotonically(analytics):
    table = analytics.simulator.midfield_sensitivity("Arsenal", "Chelsea")
    assert table["home_goals"].is_monotonic_increasing
    assert table["away_goals"].is_monotonic_decreasing
    assert table["home_win"].is_monotonic_increasing


def test_midfield_effect_sign_follows_control(analytics):
    res = analytics.simulate("Manchester City", "Nottingham Forest")
    assert res["home"]["midfield_effect"]["goals_from_midfield_edge"] > 0
    assert res["away"]["midfield_effect"]["goals_from_midfield_edge"] < 0


def test_player_absence_weakens_attack(analytics):
    players = analytics.players
    squad = players.player_table("Arsenal")
    striker = squad.sort_values("goals", ascending=False)["player"].iloc[0]
    assert players.team_units("Arsenal", [striker])["attack"] < players.team_units("Arsenal")["attack"]
    assert players.absence_effects("Arsenal", [striker])["shots_for"] < 1.0

    base = analytics.simulate("Arsenal", "Chelsea")
    weakened = analytics.simulate("Arsenal", "Chelsea", home_absent=[striker])
    assert weakened["home"]["attack"]["mean_goals"] < base["home"]["attack"]["mean_goals"]
    scorers = [p["player"] for p in weakened["home"]["players"]["top_scorers"]]
    assert striker not in scorers
    assert all(0.0 <= p["p_anytime_scorer"] <= 1.0 for p in weakened["home"]["players"]["top_scorers"])


def test_unknown_player_and_bad_csv_rejected(analytics):
    with pytest.raises(ValueError):
        analytics.players.absence_effects("Arsenal", ["Nobody"])
    with pytest.raises(ValueError):
        PlayerRatings(pd.DataFrame({"player": ["x"], "team": ["Arsenal"], "position": ["Libero"], "minutes": [900]}))


def test_season_timeline_has_no_lookahead(season_df):
    cutoff = season_df["Date"].iloc[250]
    tampered = season_df.copy()
    tampered.loc[tampered["Date"] >= cutoff, ["HS", "HST", "FTHG"]] = [40, 20, 9]
    a = season_timeline(season_df, teams=["Arsenal"], freq_days=14)
    b = season_timeline(tampered, teams=["Arsenal"], freq_days=14)
    early = a["date"] <= cutoff
    pd.testing.assert_frame_equal(a[early].reset_index(drop=True), b[early.to_numpy()].reset_index(drop=True))
    assert not a[~early].reset_index(drop=True).equals(b[~early.to_numpy()].reset_index(drop=True))


def test_unit_ratings_require_shot_columns(season_df):
    with pytest.raises(ValueError, match="shot data"):
        TeamUnitRatings().fit(season_df.drop(columns=["HS"]))


def test_tactical_profiles_flag_physical_team(analytics):
    profiles = analytics.profiles
    assert profiles["fouls_pg"].idxmax() in {
        "Everton",
        "Nottingham Forest",
        "Wolverhampton Wanderers",
        "Crystal Palace",
    }
    assert profiles.at["Manchester City", "shot_share"] > 0.5
    assert isinstance(profiles.at["Arsenal", "style"], str)


def test_team_report_and_attack_table(analytics):
    report = analytics.team_report("man city")
    assert report["team"] == "Manchester City"
    assert set(report["league_rank"]) == set(report["units"])
    assert report["timeline"]
    table = analytics.simulator.attack_vs_all_defences("Arsenal")
    assert len(table) == len(analytics.teams) - 1
    assert table["expected_goals"].is_monotonic_decreasing
