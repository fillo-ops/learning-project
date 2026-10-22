import math

import numpy as np
import pytest

from src.market import compute_kelly_edge, remove_bookmaker_vig


@pytest.mark.parametrize("method", ["power", "multiplicative"])
def test_vig_removal_sums_to_one(method):
    probs = remove_bookmaker_vig(2.10, 3.40, 3.50, method=method)
    assert np.isclose(sum(probs), 1.0, atol=1e-9)
    assert all(0.0 < p < 1.0 for p in probs)


def test_power_method_shrinks_longshots_more():
    # Favourite-longshot bias: the power method should give the longshot less than proportional scaling does.
    power = remove_bookmaker_vig(1.30, 5.50, 11.0, method="power")
    mult = remove_bookmaker_vig(1.30, 5.50, 11.0, method="multiplicative")
    assert power[2] < mult[2]
    assert power[0] > mult[0]


@pytest.mark.parametrize("bad", [(None, 3.4, 3.5), (1.0, 3.4, 3.5), (float("nan"), 3.4, 3.5), ("x", 3.4, 3.5)])
def test_invalid_odds_return_nan(bad):
    assert all(math.isnan(p) for p in remove_bookmaker_vig(*bad))


def test_kelly_edge_calculation():
    edge = compute_kelly_edge(0.55, 2.10)
    assert edge["ev"] == pytest.approx(0.155)
    assert edge["is_value"] is True
    # Full Kelly = (b*p - q) / b = (1.1*0.55 - 0.45) / 1.1; default is quarter Kelly
    assert edge["kelly_stake"] == pytest.approx(0.25 * (1.1 * 0.55 - 0.45) / 1.1, abs=1e-4)


def test_no_stake_without_value():
    edge = compute_kelly_edge(0.30, 3.0)
    assert edge["is_value"] is False
    assert edge["kelly_stake"] == 0.0
