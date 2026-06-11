"""Analyse the toy behavioral-half-life chain with the stats layer + figure.

Loads `runs/chain_toy.json`, computes per-generation control_gap (same-init minus
cross-init) with between-branch BCa CIs, fits the decay/persist/amplify models,
estimates the behavioral half-life, runs a Mann-Kendall trend test, and renders
the generation curve. Writes a verdict JSON and a PNG/PDF figure under reports/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ghosttrace.stats.bootstrap import gap_ci  # noqa: E402
from ghosttrace.stats.decay import fit_exponential, fit_flat, fit_linear  # noqa: E402
from ghosttrace.stats.model_select import classify_dynamics  # noqa: E402
from ghosttrace.stats.trend import mann_kendall  # noqa: E402

OUT = Path("reports/toy_chain")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    d = json.load(open("runs/chain_toy.json"))
    same = np.array(d["same_chains"])  # [branch, gen]
    cross = np.array(d["cross_chains"])
    gens_all = d["generations"]
    K = d["k_generations"]

    # control_gap per (gen, branch) for gens 1..K
    gens = list(range(1, K + 1))
    branch_gaps = [[float(same[b, g] - cross[b, g]) for b in range(same.shape[0])] for g in gens]
    gap_mean = [float(np.mean(bg)) for bg in branch_gaps]
    gap_ci_lo, gap_ci_hi = [], []
    for g in gens:
        _, lo, hi = gap_ci(list(same[:, g]), list(cross[:, g]), seed=g)
        gap_ci_lo.append(lo)
        gap_ci_hi.append(hi)

    exp = fit_exponential(gens, gap_mean)
    flat = fit_flat(gens, gap_mean)
    lin = fit_linear(gens, gap_mean)
    mk = mann_kendall(gap_mean)
    cls = classify_dynamics(gens, branch_gaps, aic_margin=2.0)

    verdict = {
        "gap_mean_by_gen": gap_mean,
        "gap_ci_lo": gap_ci_lo,
        "gap_ci_hi": gap_ci_hi,
        "exp_fit": exp,
        "flat_fit": flat,
        "linear_fit": lin,
        "half_life_generations": exp.get("halflife"),
        "mann_kendall": mk,
        "classification": cls,
        "same_mean_by_gen": d["same_mean_by_gen"],
        "cross_mean_by_gen": d["cross_mean_by_gen"],
    }
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=2, default=float)

    # ---- figure: capability vs generation (two arms) + decaying gap ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sm = np.array(d["same_mean_by_gen"])
    cm = np.array(d["cross_mean_by_gen"])
    sse = same.std(0) / np.sqrt(same.shape[0])
    cse = cross.std(0) / np.sqrt(cross.shape[0])
    ax1.errorbar(gens_all, sm, yerr=sse, marker="o", label="same-init chain", color="C3")
    ax1.errorbar(gens_all, cm, yerr=cse, marker="s", label="cross-init control", color="gray")
    ax1.axhline(0.1, ls=":", c="k", lw=0.8, label="chance")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("MNIST test accuracy (trait)")
    ax1.set_title("Behavioral half-life: trait vs generation")
    ax1.legend()

    ax2.errorbar(
        gens,
        gap_mean,
        yerr=[np.array(gap_mean) - np.array(gap_ci_lo), np.array(gap_ci_hi) - np.array(gap_mean)],
        marker="o",
        color="C0",
        label="control gap (same - cross)",
    )
    if np.isfinite(exp.get("tau", np.inf)):
        xs = np.linspace(1, K, 100)
        ax2.plot(
            xs,
            exp["A"] * np.exp(-xs / exp["tau"]) + exp["c"],
            "C0--",
            label=f"exp fit, half-life={exp['halflife']:.2f} gen",
        )
    ax2.axhline(0, ls=":", c="k", lw=0.8)
    ax2.set_xlabel("generation")
    ax2.set_ylabel("control gap")
    ax2.set_title(f"Decay (class={cls['class']})")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "toy_chain.png", dpi=150)
    fig.savefig(OUT / "toy_chain.pdf")
    print("DONE")


if __name__ == "__main__":
    main()
