"""Bookmaker odds de-vigging and fractional Kelly value engine."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.config import MODEL_CONFIG


def odds_are_valid(*odds: object) -> bool:
    """True when every price is a finite decimal odd strictly greater than 1."""
    try:
        return all(math.isfinite(float(o)) and float(o) > 1.0 for o in odds)
    except (TypeError, ValueError):
        return False


def remove_bookmaker_vig(
    h_odds: float, d_odds: float, a_odds: float, method: str = "power"
) -> tuple[float, float, float]:
    """Removes bookmaker overround (vig) to extract market implied probabilities.

    The power method (default) corrects for favourite-longshot bias by solving
    sum(p_i ** k) == 1 for k. Invalid or missing odds return NaNs so callers can
    treat them as genuinely missing rather than as a uniform 1/3 prior.
    """
    if method not in ("power", "multiplicative"):
        raise ValueError(f"Unknown de-vig method: {method!r}")
    if not odds_are_valid(h_odds, d_odds, a_odds):
        return (math.nan, math.nan, math.nan)

    inv = np.array([1.0 / float(h_odds), 1.0 / float(d_odds), 1.0 / float(a_odds)])
    if method == "multiplicative" or inv.sum() <= 1.0:
        probs = inv / inv.sum()
        return float(probs[0]), float(probs[1]), float(probs[2])

    # sum(inv ** k) is decreasing in k; overround > 1 means the root is at k > 1.
    low, high = 1.0, 2.0
    while np.sum(inv**high) > 1.0:
        high *= 2.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if np.sum(inv**mid) > 1.0:
            low = mid
        else:
            high = mid
    probs = inv ** (0.5 * (low + high))
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


def compute_kelly_edge(
    model_prob: float,
    decimal_odds: float,
    fraction: float = MODEL_CONFIG["kelly_fraction"],
    min_ev: float = MODEL_CONFIG["min_ev_threshold"],
) -> dict[str, Any]:
    """Calculates Expected Value (EV) and fractional Kelly stake for a single bet."""
    if not odds_are_valid(decimal_odds) or not (0.0 < model_prob < 1.0):
        return {"ev": -1.0, "kelly_stake": 0.0, "is_value": False}
    ev = model_prob * decimal_odds - 1.0
    b = decimal_odds - 1.0
    raw_kelly = max(0.0, (b * model_prob - (1.0 - model_prob)) / b)
    is_value = bool(ev > min_ev)
    return {
        "ev": round(float(ev), 4),
        "kelly_stake": round(float(raw_kelly * fraction), 4) if is_value else 0.0,
        "is_value": is_value,
    }
