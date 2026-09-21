"""Dynamics classification per the GhostTrace pre-registration.

Given per-generation gap values across replicate branches, classify the
lineage's trajectory into decay / persist / amplify / null. The decision follows
the pre-registration:

- Collapse branches to a per-generation mean trajectory and fit flat / linear /
  exponential candidates by AIC.
- ``null`` if the bootstrap CI of the pooled mean gap includes zero AND there is
  no significant monotone trend.
- ``decay`` requires a significant *decreasing* monotone trend, an exponential
  fit beating flat by ``aic_margin``, and a positive (finite) half-life.
- ``amplify`` requires a significant *increasing* monotone trend and a trended
  fit (linear/exponential, positive slope) beating flat by ``aic_margin``.
- ``persist`` otherwise when the mean gap is reliably non-zero.

Between-branch variance is surfaced in the evidence band for downstream
plotting. Returns a plain dict; the runner adapts it to the frozen result model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ghosttrace.stats.bootstrap import bca_ci
from ghosttrace.stats.decay import fit_exponential, fit_flat, fit_linear
from ghosttrace.stats.trend import mann_kendall

# Pre-registered model-selection margin (AIC) and trend significance level.
# These mirror ``StatsSpec.aic_margin`` / the trend alpha but are kept as local
# literals so the pure-stats layer has no config dependency.
_DEFAULT_AIC_MARGIN = 2.0
_ALPHA = 0.05


def classify_dynamics(
    gens: Sequence[int],
    branch_gaps: Sequence[Sequence[float]],
    *,
    aic_margin: float = _DEFAULT_AIC_MARGIN,
) -> dict[str, object]:
    """Classify per-generation dynamics from per-generation, per-branch gaps.

    ``branch_gaps[i]`` is the list of per-branch gap values at generation
    ``gens[i]``. Returns ``{'class', 'halflife', 'evidence'}`` where ``class`` is
    one of ``'decay' | 'persist' | 'amplify' | 'null'``.
    """
    x = np.asarray(gens, dtype=float)
    if x.size == 0 or len(branch_gaps) != x.size:
        raise ValueError("gens and branch_gaps must be non-empty and equal length")

    per_gen = [np.asarray(b, dtype=float) for b in branch_gaps]
    if any(b.size == 0 for b in per_gen):
        raise ValueError("each generation must have at least one branch gap value")

    mean_vals: list[float] = [float(np.mean(b)) for b in per_gen]
    var_vals: list[float] = [float(np.var(b, ddof=0)) for b in per_gen]
    between_var = float(sum(var_vals) / len(var_vals))

    # Pool every branch-gap observation for the overall-effect bootstrap CI.
    pooled: list[float] = [float(v) for b in per_gen for v in b.tolist()]
    point, lo, hi = bca_ci(pooled)
    ci_excludes_zero = (lo > 0.0) or (hi < 0.0)

    # Trend on the mean trajectory.
    mk = mann_kendall(mean_vals)
    trend = str(mk["trend"])
    mk_p = float(mk["p"])
    mk_tau = float(mk["tau"])
    significant_trend = mk_p < _ALPHA and trend != "no trend"

    # Candidate fits on the mean trajectory.
    flat = fit_flat(gens, mean_vals)
    linear = fit_linear(gens, mean_vals)
    expo = fit_exponential(gens, mean_vals)

    aic_flat = flat["aic"]
    delta_expo = aic_flat - expo["aic"]
    delta_linear = aic_flat - linear["aic"]

    evidence: dict[str, float] = {
        "mean_gap": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "between_branch_var": between_var,
        "mk_tau": mk_tau,
        "mk_p": mk_p,
        "aic_flat": aic_flat,
        "aic_linear": linear["aic"],
        "aic_exponential": expo["aic"],
        "delta_aic_exp_vs_flat": delta_expo,
        "delta_aic_lin_vs_flat": delta_linear,
        "exp_tau": expo["tau"],
        "exp_halflife": expo["halflife"],
        "linear_slope": linear["slope"],
    }

    # --- decay: decreasing trend + exponential beats flat + finite half-life ---
    decay_ok = (
        significant_trend
        and trend == "decreasing"
        and delta_expo >= aic_margin
        and np.isfinite(expo["halflife"])
        and expo["tau"] > 0.0
    )
    if decay_ok:
        return {"class": "decay", "halflife": float(expo["halflife"]), "evidence": evidence}

    # --- amplify: increasing trend + a trended fit (positive slope) beats flat --
    amplify_ok = (
        significant_trend
        and trend == "increasing"
        and (delta_linear >= aic_margin or delta_expo >= aic_margin)
        and linear["slope"] > 0.0
    )
    if amplify_ok:
        return {"class": "amplify", "halflife": None, "evidence": evidence}

    # --- null: no reliable gap and no trend ---
    if not significant_trend and not ci_excludes_zero:
        return {"class": "null", "halflife": None, "evidence": evidence}

    # --- persist: reliably non-zero gap without a decay/amplify verdict ---
    if ci_excludes_zero:
        return {"class": "persist", "halflife": None, "evidence": evidence}

    # A significant trend that failed the AIC/half-life gates and whose CI
    # includes zero is treated as null: insufficient evidence for a class.
    return {"class": "null", "halflife": None, "evidence": evidence}
