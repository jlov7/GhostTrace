"""Monotone-trend tests for per-generation trajectories.

The pre-registration requires a *significant monotone trend* before declaring
decay or amplify, on top of any AIC evidence. Mann-Kendall tests a single
lineage's gap series for monotonicity; Jonckheere-Terpstra tests for an ordered
trend across grouped branches (generations as ordered groups). Both return a
p-value and the test statistic so model selection can gate on alpha.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import kendalltau, norm  # pyright: ignore[reportUnknownVariableType]

# Significance level for declaring a trend; matches the pre-registered alpha.
_ALPHA = 0.05


def _kendall_tau_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wrap scipy ``kendalltau`` returning concrete ``(tau, pvalue)`` floats.

    Isolates scipy's untyped SignificanceResult here so the rest of the module
    stays fully typed under pyright strict.
    """
    fn: Any = kendalltau  # pyright: ignore[reportUnknownVariableType]
    res: Any = fn(x, y)
    return float(res.statistic), float(res.pvalue)


def _norm_two_sided_p(z: float) -> float:
    """Two-sided normal-approximation p-value for a z-score."""
    sf: Any = norm.sf  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return 2.0 * float(sf(abs(z)))


def mann_kendall(values: Sequence[float]) -> dict[str, float | str]:
    """Mann-Kendall trend test on an ordered series.

    Returns ``{tau, p, trend}`` where ``trend`` is one of ``'increasing'``,
    ``'decreasing'``, or ``'no trend'`` (decided at alpha via the two-sided
    p-value, with direction from the sign of Kendall's tau). Series shorter than
    three points cannot show a trend and return ``'no trend'`` with ``p = 1``.
    """
    y = np.asarray(values, dtype=float)
    n = y.size
    if n < 3:
        return {"tau": 0.0, "p": 1.0, "trend": "no trend"}

    tau_raw, p_raw = _kendall_tau_p(np.arange(n), y)
    tau_f = tau_raw if math.isfinite(tau_raw) else 0.0
    p_f = p_raw if math.isfinite(p_raw) else 1.0

    if p_f >= _ALPHA or tau_f == 0.0:
        trend = "no trend"
    elif tau_f > 0.0:
        trend = "increasing"
    else:
        trend = "decreasing"
    return {"tau": tau_f, "p": p_f, "trend": trend}


def jonckheere_terpstra(groups: Sequence[Sequence[float]]) -> dict[str, float]:
    """Jonckheere-Terpstra test for an ordered alternative across groups.

    ``groups`` are assumed to be in their natural order (e.g. generation 0, 1,
    ... N). Computes the JT statistic as the sum of Mann-Whitney counts over all
    ordered group pairs (ties counted as 0.5) and a normal-approximation
    two-sided p-value with a tie-aware variance. Returns ``{stat, p}``. Fewer
    than two non-empty groups yields ``stat = 0, p = 1``.
    """
    clean = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    k = len(clean)
    if k < 2:
        return {"stat": 0.0, "p": 1.0}

    jt = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            a = clean[i][:, None]
            b = clean[j][None, :]
            jt += float(np.sum(b > a) + 0.5 * np.sum(b == a))

    sizes = np.array([g.size for g in clean], dtype=float)
    n = float(sizes.sum())
    mean_jt = (n**2 - float(np.sum(sizes**2))) / 4.0

    # Tie-aware variance (leading Lehmann term). Higher-order tie corrections are
    # negligible for the small group counts used here and are omitted for clarity.
    all_vals = np.concatenate(clean)
    _, tie_counts = np.unique(all_vals, return_counts=True)
    t_term = float(np.sum(tie_counts * (tie_counts - 1.0) * (2.0 * tie_counts + 5.0)))
    g_term = float(np.sum(sizes * (sizes - 1.0) * (2.0 * sizes + 5.0)))
    var_jt = (n * (n - 1.0) * (2.0 * n + 5.0) - g_term - t_term) / 72.0

    if var_jt <= 0.0:
        return {"stat": float(jt), "p": 1.0}

    z = (jt - mean_jt) / math.sqrt(var_jt)
    p = _norm_two_sided_p(z)
    return {"stat": float(jt), "p": min(p, 1.0)}
