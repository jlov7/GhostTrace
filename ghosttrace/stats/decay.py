"""Candidate dynamics fits for per-generation effect-size trajectories.

Model selection (``model_select.py``) compares three candidates by AIC: ``flat``
(the null: gap is constant), ``linear`` (the amplify/decay-trend model), and
``exponential`` (the decay model with a half-life, ``A*e^(-k/tau) + c``). Each
fitter returns a uniform dict so the selector can compare them directly. We guard
``curve_fit`` failures by returning ``aic = inf`` rather than raising, so one bad
lineage never crashes a batch analysis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import curve_fit  # pyright: ignore[reportUnknownVariableType]

# Decay constants at or above this magnitude are treated as effectively flat:
# the exponential never halves over any finite generation horizon, so the
# half-life is reported as infinite rather than an astronomically large number.
_TAU_FLAT_THRESHOLD = 1e6


def _aic(rss: float, n: int, k: int) -> float:
    """AIC for a least-squares fit: ``n*ln(rss/n) + 2k`` (lower is better).

    A zero RSS (perfect fit) yields ``-inf``; an invalid/non-finite RSS yields
    ``+inf`` so it is never selected.
    """
    if n <= 0 or not math.isfinite(rss):
        return math.inf
    if rss <= 0.0:
        return -math.inf
    return n * math.log(rss / n) + 2.0 * k


def half_life(tau: float) -> float:
    """Half-life of an exponential with decay constant ``tau``: ``tau*ln(2)``.

    A non-positive, non-finite, or effectively-flat ``tau`` returns ``inf`` since
    the signal never halves over a finite horizon.
    """
    if not math.isfinite(tau) or tau <= 0.0 or abs(tau) >= _TAU_FLAT_THRESHOLD:
        return math.inf
    return tau * math.log(2.0)


def _prepare(gens: Sequence[int], gaps: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(gens, dtype=float)
    y = np.asarray(gaps, dtype=float)
    if x.size != y.size:
        raise ValueError("gens and gaps must have equal length")
    return x, y


def fit_flat(gens: Sequence[int], gaps: Sequence[float]) -> dict[str, float]:
    """Fit the null model ``gap = c`` (constant). One free parameter."""
    _, y = _prepare(gens, gaps)
    n = y.size
    c = float(np.mean(y)) if n > 0 else 0.0
    rss = float(np.sum((y - c) ** 2))
    return {"c": c, "rss": rss, "aic": _aic(rss, n, k=1)}


def fit_linear(gens: Sequence[int], gaps: Sequence[float]) -> dict[str, float]:
    """Fit ``gap = slope*gen + intercept``. Two free parameters.

    The sign of ``slope`` separates amplify (positive) from decay-like trends.
    """
    x, y = _prepare(gens, gaps)
    n = y.size
    if n < 2:
        return {"slope": math.nan, "intercept": math.nan, "rss": math.inf, "aic": math.inf}
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    pred = slope * x + intercept
    rss = float(np.sum((y - pred) ** 2))
    return {"slope": slope, "intercept": intercept, "rss": rss, "aic": _aic(rss, n, k=2)}


def _exp_model(x: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    return a * np.exp(-x / tau) + c


def _curve_fit_exp(x: np.ndarray, y: np.ndarray, p0: list[float]) -> tuple[float, float, float]:
    """Wrap scipy ``curve_fit`` for the exponential model with a concrete return.

    Isolates scipy's untyped surface here so callers receive plain floats. Raises
    the usual ``curve_fit`` exceptions on failure, which the caller catches.
    """
    fit: Any = curve_fit  # pyright: ignore[reportUnknownVariableType]
    popt, _ = fit(
        _exp_model,
        x,
        y,
        p0=p0,
        maxfev=10000,
        bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


def fit_exponential(gens: Sequence[int], gaps: Sequence[float]) -> dict[str, float]:
    """Fit ``gap = A*exp(-gen/tau) + c`` (the decay model). Three parameters.

    Returns ``{A, tau, c, halflife, rss, aic}``. On any optimizer failure (or
    fewer than three points) the fit is marked unusable with ``aic = inf`` so it
    loses model selection.
    """
    x, y = _prepare(gens, gaps)
    n = y.size
    failed: dict[str, float] = {
        "A": math.nan,
        "tau": math.nan,
        "c": math.nan,
        "halflife": math.inf,
        "rss": math.inf,
        "aic": math.inf,
    }
    if n < 3:
        return failed

    # Initial guesses: amplitude = head-to-tail span, tau = half the horizon,
    # c = the tail value. These keep curve_fit well-conditioned on decay data.
    span = float(y[0] - y[-1]) if y[0] != y[-1] else (float(np.ptp(y)) or 1.0)
    x_range = float(np.ptp(x)) or 1.0
    p0 = [span, max(x_range / 2.0, 1e-3), float(y[-1])]
    try:
        a, tau, c = _curve_fit_exp(x, y, p0)
    except (RuntimeError, ValueError, TypeError):
        return failed

    pred = _exp_model(x, a, tau, c)
    rss = float(np.sum((y - pred) ** 2))
    if not math.isfinite(rss):
        return failed
    return {
        "A": a,
        "tau": tau,
        "c": c,
        "halflife": half_life(tau),
        "rss": rss,
        "aic": _aic(rss, n, k=3),
    }
