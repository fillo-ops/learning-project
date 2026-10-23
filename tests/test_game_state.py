import numpy as np
import pytest

from src.config import GAME_STATE_CONFIG
from src.game_state import DisciplineModel
from src.simulation import MatchSimulator


def test_game_state_calibration_keeps_expected_goals(analytics):
    res = analytics.simulate("Liverpool", "Everton")
    assert res["game_state"] is not None
    for side in ("home", "away"):
        target = res[side]["expected_rates"]["expected_goals"]
        assert res[side]["attack"]["mean_goals"] == pytest.approx(target, rel=0.03)


def test_game_state_summary_is_consistent(analytics):
    gs = analytics.simulate("Arsenal", "Chelsea")["game_state"]
    assert gs["home_scores_first"] + gs["away_scores_first"] + gs["p_goalless"] == pytest.approx(1.0, abs=1e-3)
    shares = gs["share_of_match_home_leading"] + gs["share_of_match_level"] + gs["share_of_match_away_leading"]
    assert shares == pytest.approx(1.0, abs=1e-3)
    # Goal rates climb through the match (block_profile)
    periods = list(gs["goals_by_period"]["home"].values())
    assert periods[-1] > periods[0]


def test_scoreline_effects_shift_results_toward_draws(analytics):
    neutral = {k: (1.0, 1.0) for k in GAME_STATE_CONFIG["state_effects"]}
    strong = {-2: (1.6, 0.8), -1: (1.4, 0.85), 0: (1.0, 1.0), 1: (0.6, 1.1), 2: (0.5, 1.2)}
    kwargs = {"n_sims": 20000, "discipline": analytics.discipline}
    flat = MatchSimulator(analytics.ratings, game_state_config={"state_effects": neutral}, **kwargs)
    tight = MatchSimulator(analytics.ratings, game_state_config={"state_effects": strong}, **kwargs)
    d_flat = flat.simulate("Arsenal", "Chelsea")["outcome"]["draw"]
    d_tight = tight.simulate("Arsenal", "Chelsea")["outcome"]["draw"]
    assert d_tight > d_flat  # leaders protecting leads + trailers chasing -> closer games


def test_red_cards_follow_discipline_and_referee(season_df):
    model = DisciplineModel().fit(season_df)
    assert model.has_data_
    assert model.red_rate("Everton", "D Coote") > model.red_rate("Everton", "M Oliver")
    # Shrinkage keeps team rates near the league mean
    rates = [model.red_rate(t) for t in model.team_rate_]
    assert 0.5 * model.league_rate_ < np.median(rates) < 1.5 * model.league_rate_


def test_strict_referee_raises_red_card_probability(analytics):
    lenient = analytics.simulate("Everton", "Fulham", referee="M Oliver")["game_state"]
    strict = analytics.simulate("Everton", "Fulham", referee="D Coote")["game_state"]
    assert strict["p_home_red_card"] > lenient["p_home_red_card"]


def test_static_mode_still_available(analytics):
    sim = MatchSimulator(analytics.ratings, n_sims=5000, game_state=False)
    res = sim.simulate("Arsenal", "Chelsea")
    assert res["game_state"] is None
    assert sum(res["outcome"].values()) == pytest.approx(1.0, abs=1e-3)
