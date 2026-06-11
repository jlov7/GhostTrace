"""Bias-corrected-and-accelerated (BCa) bootstrap confidence intervals.

The pre-registration fixes BCa over the percentile bootstrap because the
``control_gap`` effect size can be skewed and small-sample, where percentile CIs
are biased. BCa corrects for median bias (z0) and for acceleration (skew of the
jackknife distribution), giving better coverage. Randomness flows through
``derive_seed`` so resampling is reproducible per named stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import norm

from ghosttrace.seeding import derive_seed


def _norm_ppf(q: float) -> float:
    """Standard-normal quantile (inverse CDF); wraps the untyped scipy method."""
    fn: Any = norm.ppf  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return float(fn(q))


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF; wraps the untyped scipy method with a float return."""
    fn: Any = norm.cdf  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return float(fn(x))


def _bca_from_boot(
    boot: np.ndarray,
    jack: np.ndarray,
    point: float,
    *,
    level: float,
) -> tuple[float, float]:
    """Turn a bootstrap distribution + jackknife influence into a BCa ``(lo, hi)``.

    ``boot`` is the bootstrap distribution of the statistic, ``jack`` the
    leave-one-out jackknife values used to estimate acceleration, ``point`` the
    observed statistic. Falls back to a plain percentile interval when the
    bias-correction is undefined (all resamples on one side of ``point``).
    """
    alpha = 1.0 - level
    prop_less = float(np.mean(boot < point))
    if prop_less <= 0.0 or prop_less >= 1.0:
        lo = float(np.quantile(boot, alpha / 2.0))
        hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
        return lo, hi
    z0 = float(_norm_ppf(prop_less))

    jack_mean = float(np.mean(jack))
    diff = jack_mean - jack
    denom = 6.0 * float(np.sum(diff**2)) ** 1.5
    a = float(np.sum(diff**3)) / denom if denom != 0.0 else 0.0

    z_lo = float(_norm_ppf(alpha / 2.0))
    z_hi = float(_norm_ppf(1.0 - alpha / 2.0))

    def _adjust(zq: float) -> float:
        num = z0 + zq
        return float(_norm_cdf(z0 + num / (1.0 - a * num)))

    lo = float(np.quantile(boot, _adjust(z_lo)))
    hi = float(np.quantile(boot, _adjust(z_hi)))
    return lo, hi


def bca_ci(
    values: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` BCa bootstrap CI for the mean of ``values``.

    Degenerate inputs (n < 2 or zero variance) collapse the interval onto the
    point estimate rather than raising, so callers can treat any vector
    uniformly. An empty input is a programming error and raises.
    """
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        raise ValueError("bca_ci requires at least one value")

    point = float(np.mean(sample))
    n = sample.size
    if n < 2 or float(np.std(sample)) == 0.0:
        return point, point, point

    rng = np.random.default_rng(derive_seed(seed, "bca_ci"))
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = sample[idx].mean(axis=1)

    total = sample.sum()
    jack = (total - sample) / (n - 1)

    lo, hi = _bca_from_boot(boot, jack, point, level=level)
    return point, lo, hi


def gap_ci(
    treated: Sequence[float],
    control: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean_gap, lo, hi)`` BCa CI of ``mean(treated) - mean(control)``.

    The two arms are resampled independently (a two-sample bootstrap of the
    difference in means) -- this is the ``control_gap`` effect size. Bias and
    acceleration are estimated by jackknifing both arms and pooling their
    influence values.
    """
    t = np.asarray(treated, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.size == 0 or c.size == 0:
        raise ValueError("gap_ci requires non-empty treated and control samples")

    point = float(np.mean(t) - np.mean(c))
    nt, nc = t.size, c.size
    if nt < 2 or nc < 2:
        return point, point, point

    rng = np.random.default_rng(derive_seed(seed, "gap_ci"))
    ti = rng.integers(0, nt, size=(n_resamples, nt))
    ci = rng.integers(0, nc, size=(n_resamples, nc))
    boot = t[ti].mean(axis=1) - c[ci].mean(axis=1)

    t_sum, c_sum = t.sum(), c.sum()
    jack_t = (t_sum - t) / (nt - 1) - float(np.mean(c))
    jack_c = float(np.mean(t)) - (c_sum - c) / (nc - 1)
    jack = np.concatenate([jack_t, jack_c])

    lo, hi = _bca_from_boot(boot, jack, point, level=level)
    return point, lo, hi
