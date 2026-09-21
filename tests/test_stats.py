"""Tests for the pure statistics layer (bootstrap, decay fits, trend, classify).

All synthetic, fast, and deterministic via numpy default_rng with fixed seeds.
No MLX, no model loading.
"""

from __future__ import annotations

import math

import numpy as np

from ghosttrace.stats.bootstrap import bca_ci, gap_ci
from ghosttrace.stats.decay import fit_exponential, fit_flat, fit_linear, half_life
from ghosttrace.stats.model_select import classify_dynamics
from ghosttrace.stats.trend import jonckheere_terpstra, mann_kendall

_GENS = list(range(10))


def _exp_branches(
    *, amplitude: float, tau: float, c: float, noise: float, n_branches: int, seed: int
) -> list[list[float]]:
    """Per-generation list of per-branch gaps from a known exponential decay."""
    rng = np.random.default_rng(seed)
    x = np.asarray(_GENS, dtype=float)
    true = amplitude * np.exp(-x / tau) + c
    out: list[list[float]] = []
    for gi in range(x.size):
        out.append([float(true[gi] + rng.normal(0.0, noise)) for _ in range(n_branches)])
    return out


def _flat_branches(*, c: float, noise: float, n_branches: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [[float(c + rng.normal(0.0, noise)) for _ in range(n_branches)] for _ in _GENS]


def _increasing_branches(
    *, slope: float, intercept: float, noise: float, n_branches: int, seed: int
) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    out: list[list[float]] = []
    for g in _GENS:
        base = intercept + slope * g
        out.append([float(base + rng.normal(0.0, noise)) for _ in range(n_branches)])
    return out


# --- bootstrap -------------------------------------------------------------- #
def test_bca_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    arr = rng.normal(5.0, 1.0, size=200)
    sample: list[float] = arr.tolist()
    point, lo, hi = bca_ci(sample, n_resamples=2000, seed=7)
    assert lo < point < hi
    assert abs(point - float(arr.mean())) < 1e-9


def test_bca_ci_coverage_sanity() -> None:
    """Across many normal samples, the 95% CI should cover the true mean ~95%."""
    true_mean = 2.0
    covered = 0
    trials = 80
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        sample = rng.normal(true_mean, 1.0, size=60).tolist()
        _, lo, hi = bca_ci(sample, n_resamples=1500, seed=t)
        if lo <= true_mean <= hi:
            covered += 1
    assert covered / trials > 0.85


def test_bca_ci_deterministic() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert bca_ci(sample, n_resamples=1000, seed=3) == bca_ci(sample, n_resamples=1000, seed=3)


def test_bca_ci_degenerate_constant() -> None:
    point, lo, hi = bca_ci([4.0, 4.0, 4.0], n_resamples=500, seed=1)
    assert point == lo == hi == 4.0


def test_gap_ci_positive_difference() -> None:
    rng = np.random.default_rng(11)
    treated = rng.normal(3.0, 1.0, size=100).tolist()
    control = rng.normal(0.0, 1.0, size=100).tolist()
    point, lo, hi = gap_ci(treated, control, n_resamples=2000, seed=5)
    assert point > 0.0
    assert lo > 0.0
    assert lo < point < hi


def test_gap_ci_includes_zero_when_no_effect() -> None:
    rng = np.random.default_rng(21)
    treated = rng.normal(0.0, 1.0, size=80).tolist()
    control = rng.normal(0.0, 1.0, size=80).tolist()
    _, lo, hi = gap_ci(treated, control, n_resamples=2000, seed=9)
    assert lo <= 0.0 <= hi


# --- decay fits ------------------------------------------------------------- #
def test_half_life_basic() -> None:
    assert math.isclose(half_life(1.0), math.log(2.0), rel_tol=1e-12)
    assert half_life(0.0) == math.inf
    assert half_life(float("inf")) == math.inf


def test_fit_flat_recovers_constant() -> None:
    res = fit_flat(_GENS, [3.0] * len(_GENS))
    assert math.isclose(res["c"], 3.0, abs_tol=1e-9)
    assert res["rss"] < 1e-9


def test_fit_linear_recovers_slope() -> None:
    gaps = [2.0 + 0.5 * g for g in _GENS]
    res = fit_linear(_GENS, gaps)
    assert math.isclose(res["slope"], 0.5, abs_tol=1e-6)
    assert math.isclose(res["intercept"], 2.0, abs_tol=1e-6)


def test_fit_exponential_recovers_params() -> None:
    x = np.asarray(_GENS, dtype=float)
    true = 4.0 * np.exp(-x / 3.0) + 0.5
    res = fit_exponential(_GENS, true.tolist())
    assert math.isclose(res["tau"], 3.0, rel_tol=0.05)
    assert math.isclose(res["A"], 4.0, rel_tol=0.05)
    assert math.isclose(res["halflife"], 3.0 * math.log(2.0), rel_tol=0.05)
    assert res["aic"] < fit_flat(_GENS, true.tolist())["aic"]


def test_fit_exponential_failure_returns_inf_aic() -> None:
    res = fit_exponential([0, 1], [1.0, 2.0])  # too few points
    assert res["aic"] == math.inf
    assert res["halflife"] == math.inf


# --- trend ------------------------------------------------------------------ #
def test_mann_kendall_increasing() -> None:
    res = mann_kendall([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert res["trend"] == "increasing"
    assert float(res["p"]) < 0.05


def test_mann_kendall_decreasing() -> None:
    assert mann_kendall([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])["trend"] == "decreasing"


def test_mann_kendall_no_trend() -> None:
    assert mann_kendall([1.0, 3.0, 2.0, 3.0, 1.0, 2.0])["trend"] == "no trend"


def test_jonckheere_increasing_groups() -> None:
    groups = [[1.0, 2.0, 1.5], [3.0, 4.0, 3.5], [5.0, 6.0, 5.5]]
    res = jonckheere_terpstra(groups)
    assert res["p"] < 0.05
    assert res["stat"] > 0.0


def test_jonckheere_no_order() -> None:
    rng = np.random.default_rng(2)
    groups = [rng.normal(0.0, 1.0, size=8).tolist() for _ in range(4)]
    assert jonckheere_terpstra(groups)["p"] > 0.05


# --- classify_dynamics ------------------------------------------------------ #
def test_classify_decay_recovers_halflife() -> None:
    branches = _exp_branches(amplitude=5.0, tau=3.0, c=0.2, noise=0.15, n_branches=5, seed=42)
    res = classify_dynamics(_GENS, branches)
    assert res["class"] == "decay"
    hl = res["halflife"]
    assert isinstance(hl, float)
    assert math.isclose(hl, 3.0 * math.log(2.0), rel_tol=0.25)


def test_classify_null_on_flat_noise() -> None:
    branches = _flat_branches(c=0.0, noise=0.3, n_branches=5, seed=123)
    assert classify_dynamics(_GENS, branches)["class"] == "null"


def test_classify_amplify_on_increasing() -> None:
    branches = _increasing_branches(slope=0.6, intercept=0.5, noise=0.15, n_branches=5, seed=77)
    res = classify_dynamics(_GENS, branches)
    assert res["class"] == "amplify"
    assert res["halflife"] is None


def test_classify_persist_on_constant_nonzero() -> None:
    branches = _flat_branches(c=2.0, noise=0.1, n_branches=5, seed=55)
    assert classify_dynamics(_GENS, branches)["class"] == "persist"


def test_classify_evidence_has_expected_keys() -> None:
    branches = _flat_branches(c=0.0, noise=0.3, n_branches=4, seed=8)
    evidence = classify_dynamics(_GENS, branches)["evidence"]
    assert isinstance(evidence, dict)
    for key in ("mean_gap", "ci_lo", "ci_hi", "between_branch_var", "mk_p", "aic_flat"):
        assert key in evidence
