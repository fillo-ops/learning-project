import pytest

from src.model import MatchPredictorEngine
from src.synthetic import generate_synthetic_matches


@pytest.fixture(scope="session")
def season_df():
    return generate_synthetic_matches(n_seasons=1, seed=1)


@pytest.fixture(scope="session")
def trained_engine(season_df):
    engine = MatchPredictorEngine(n_splits=3)
    engine.fit(season_df)
    return engine


@pytest.fixture(scope="session")
def analytics(season_df):
    from src.analytics import TeamAnalytics
    from src.synthetic import generate_synthetic_players

    return TeamAnalytics(season_df, generate_synthetic_players(season_df), n_sims=20000)
