"""Bookmaker odds de-vigging and fractional Kelly value engine."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

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


# ----------------------------------------------------------------------------- Asian lines
def _settle(margin_values: np.ndarray, probs: np.ndarray, line: float) -> dict[str, float]:
    """Settlement probabilities for a bet that wins when `margin + line > 0`.

    Quarter lines (e.g. -0.25, 2.75) are split into two half-stakes on the neighbouring
    lines, giving half-win / half-loss outcomes.
    """
    quarter = abs(round(line * 4)) % 2 == 1
    halves = (line - 0.25, line + 0.25) if quarter else (line, line)
    out = dict.fromkeys(("win", "half_win", "push", "half_loss", "loss"), 0.0)
    for value, p in zip(margin_values, probs):
        results = [np.sign(round(value + h, 6)) for h in halves]  # +1 win, 0 push, -1 loss
        total = sum(results)
        if total == 2:
            out["win"] += p
        elif total == -2:
            out["loss"] += p
        elif results[0] == results[1] == 0:
            out["push"] += p
        elif total > 0:
            out["half_win"] += p
        else:
            out["half_loss"] += p
    return out


def _fair_odds(s: dict[str, float]) -> float:
    """Decimal odds at which the bet has zero expected value (pushes refund the stake)."""
    gain = s["win"] + 0.5 * s["half_win"]
    if gain <= 0:
        return float("inf")
    return 1.0 + (s["loss"] + 0.5 * s["half_loss"]) / gain


def bet_expected_value(s: dict[str, float], odds: float) -> float:
    """Expected profit per unit staked at decimal `odds` for a settlement distribution."""
    b = odds - 1.0
    return s["win"] * b + s["half_win"] * b / 2 - s["half_loss"] / 2 - s["loss"]


def asian_handicap(score_matrix: np.ndarray, line: float) -> dict[str, float]:
    """Home Asian handicap at `line` (e.g. -0.5 = home must win) from a joint goals matrix."""
    hg, ag = np.indices(score_matrix.shape)
    diff = (hg - ag).ravel()
    values, inv = np.unique(diff, return_inverse=True)
    probs = np.bincount(inv, weights=score_matrix.ravel())
    out = _settle(values, probs, line)
    out["fair_odds"] = _fair_odds(out)
    return out


def asian_total(score_matrix: np.ndarray, line: float) -> dict[str, float]:
    """Over `line` total goals (e.g. 2.25, 2.5, 2.75); the under is the mirror image."""
    hg, ag = np.indices(score_matrix.shape)
    totals = (hg + ag).ravel()
    values, inv = np.unique(totals, return_inverse=True)
    probs = np.bincount(inv, weights=score_matrix.ravel())
    out = _settle(values, probs, -line)
    out["fair_odds"] = _fair_odds(out)
    under = {"win": out["loss"], "half_win": out["half_loss"], "push": out["push"], "half_loss": out["half_win"]}
    under["loss"] = out["win"]
    out["under_fair_odds"] = _fair_odds(under)
    return out


def asian_markets(
    score_matrix: np.ndarray,
    handicap_lines: tuple[float, ...] = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
    total_lines: tuple[float, ...] = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5),
) -> dict[str, list[dict[str, float]]]:
    """Fair prices for a ladder of home Asian handicaps and goal lines."""
    mat = np.asarray(score_matrix, dtype=float)
    mat = mat / mat.sum()

    def fmt(d: dict[str, float]) -> dict[str, float | None]:
        # Unpriceable lines (no winning outcomes) get None rather than an un-serialisable infinity
        return {k: round(float(v), 4) if math.isfinite(float(v)) else None for k, v in d.items()}

    return {
        "asian_handicap_home": [{"line": ln, **fmt(asian_handicap(mat, ln))} for ln in handicap_lines],
        "goal_lines_over": [{"line": ln, **fmt(asian_total(mat, ln))} for ln in total_lines],
    }


def score_matrix_from_samples(home_goals: np.ndarray, away_goals: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """Joint (home, away) goals distribution from simulated scorelines."""
    h = np.minimum(home_goals, max_goals)
    a = np.minimum(away_goals, max_goals)
    mat = np.zeros((max_goals + 1, max_goals + 1))
    np.add.at(mat, (h, a), 1.0)
    return mat / mat.sum()


# ------------------------------------------------------------------- portfolio staking
def portfolio_kelly(
    bets: list[dict[str, Any]],
    fraction: float = MODEL_CONFIG["kelly_fraction"],
    max_total_exposure: float = 0.12,
    min_ev: float = MODEL_CONFIG["min_ev_threshold"],
) -> list[dict[str, Any]]:
    """Fractional Kelly stakes for simultaneous bets with a cap on total bankroll exposure.

    Each bet needs `match`, `selection`, `prob` and `odds`. Selections in the same match are
    mutually exclusive (and highly correlated with any other market on that match), so only
    the highest-EV value bet per match is kept; stakes are then scaled down together if their
    sum exceeds `max_total_exposure` of the bankroll.
    """
    best: dict[str, dict[str, Any]] = {}
    for bet in bets:
        edge = compute_kelly_edge(float(bet["prob"]), float(bet["odds"]), fraction=fraction, min_ev=min_ev)
        if not edge["is_value"]:
            continue
        cand = {**bet, **edge}
        current = best.get(bet["match"])
        if current is None or cand["ev"] > current["ev"]:
            best[bet["match"]] = cand
    chosen = list(best.values())
    total = sum(b["kelly_stake"] for b in chosen)
    scale = min(1.0, max_total_exposure / total) if total > 0 else 1.0
    for b in chosen:
        b["stake"] = round(b["kelly_stake"] * scale, 4)
        b["scaled_by"] = round(scale, 4)
    return sorted(chosen, key=lambda b: -b["ev"])


# ------------------------------------------------------------------------ line movement
OPEN_CLOSE_COLUMNS = [
    (("B365H", "B365D", "B365A"), ("PSCH", "PSCD", "PSCA")),
    (("AvgH", "AvgD", "AvgA"), ("AvgCH", "AvgCD", "AvgCA")),
    (("B365H", "B365D", "B365A"), ("B365CH", "B365CD", "B365CA")),
]


def line_movement(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Opening-to-closing market moves per team, and whether closing prices were sharper.

    For every match, both prices are de-vigged and the move for a side is
    log(p_close) - log(p_open): positive means money came in on that side ("steam").
    """
    for open_cols, close_cols in OPEN_CLOSE_COLUMNS:
        if set(open_cols) | set(close_cols) <= set(df.columns):
            break
    else:
        raise ValueError("Need opening and closing odds columns (e.g. B365H.. with PSCH.. or AvgH.. with AvgCH..)")

    rows, ll_open, ll_close = [], [], []
    outcome_idx = {"H": 0, "D": 1, "A": 2}
    for r in df.to_dict(orient="records"):
        p_open = remove_bookmaker_vig(*(r.get(c) for c in open_cols))
        p_close = remove_bookmaker_vig(*(r.get(c) for c in close_cols))
        if any(math.isnan(x) for x in p_open + p_close):
            continue
        k = outcome_idx.get(str(r.get("FTR")))
        if k is not None:
            ll_open.append(-math.log(p_open[k]))
            ll_close.append(-math.log(p_close[k]))
        for team, i in ((r["HomeTeam"], 0), (r["AwayTeam"], 2)):
            move = math.log(p_close[i]) - math.log(p_open[i])
            won = (r.get("FTR") == "H") if i == 0 else (r.get("FTR") == "A")
            rows.append({"team": team, "move": move, "backed": move > 0, "won": bool(won), "p_open": p_open[i]})
    moves = pd.DataFrame(rows)
    if moves.empty:
        raise ValueError("No matches with valid opening and closing odds")
    table = (
        moves.groupby("team")
        .agg(matches=("move", "size"), avg_move=("move", "mean"), share_backed=("backed", "mean"))
        .sort_values("avg_move", ascending=False)
        .round(4)
    )
    summary = {
        "matches": len(ll_open),
        "log_loss_opening": round(float(np.mean(ll_open)), 4) if ll_open else float("nan"),
        "log_loss_closing": round(float(np.mean(ll_close)), 4) if ll_close else float("nan"),
    }
    # Did the moves carry information? Compare results with the *opening* price on each side.
    for label, mask in (("steamed", moves["move"] > 0.05), ("drifting", moves["move"] < -0.05)):
        sub = moves[mask]
        summary[f"n_{label}"] = len(sub)
        summary[f"{label}_win_rate_minus_opening_prob"] = (
            round(float(sub["won"].mean() - sub["p_open"].mean()), 4) if len(sub) else float("nan")
        )
    return table, summary
