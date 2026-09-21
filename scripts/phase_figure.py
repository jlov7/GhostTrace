"""Render the phase-diagram law figures from `runs/phase_diagram.json`.

Degenerate conditions (gap@gen1 below MIN_GAP) carry no transmissible signal, so
their exponential half-life is a fit to ~zero and is not plotted as a half-life;
they are annotated as "no transfer" instead. This keeps the figure honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("reports/toy_chain")
OUT.mkdir(parents=True, exist_ok=True)
MIN_GAP = 0.05  # below this, gen-1 transfer is negligible -> half-life meaningless


def main() -> None:
    d = json.load(open("runs/phase_diagram.json"))
    res = d["results"]
    cap = sorted([r for r in res if r["n_noise"] == 20000], key=lambda r: r["m"])
    dat = sorted([r for r in res if r["m"] == 3], key=lambda r: r["n_noise"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: decay curves at each capacity
    for r in cap:
        gens = list(range(len(r["same_mean_by_gen"])))
        ax1.plot(gens, r["same_mean_by_gen"], marker="o", label=f"m={r['m']}")
    ax1.axhline(0.1, ls=":", c="k", lw=0.8, label="chance")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("trait (MNIST acc)")
    ax1.set_title("Decay by channel capacity (n=20k)")
    ax1.legend(fontsize=8)

    def split(rows: list[dict[str, object]], xkey: str) -> tuple[list, list, list]:
        real_x, real_h, dead_x = [], [], []
        for r in rows:
            if float(r["gap_gen1"]) >= MIN_GAP:
                real_x.append(r[xkey])
                real_h.append(r["halflife"])
            else:
                dead_x.append(r[xkey])
        return real_x, real_h, dead_x

    # Panel 2: half-life vs capacity (real-transfer points only)
    mx_, mh_, mdead = split(cap, "m")
    ax2.plot(mx_, mh_, "o-", color="C3", ms=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("aux channel capacity m (log)")
    ax2.set_ylabel("behavioral half-life (generations)")
    ax2.set_title("Half-life grows with channel capacity")
    for x, h in zip(mx_, mh_):
        ax2.annotate(f"{h:.2f}", (x, h), textcoords="offset points", xytext=(0, 8), fontsize=8)
    for x in mdead:
        ax2.annotate(
            "no transfer",
            (x, 0),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            color="gray",
            ha="center",
        )
        ax2.plot([x], [0], "x", color="gray")

    # Panel 3: half-life vs dataset size (real-transfer points only)
    nx_, nh_, ndead = split(dat, "n_noise")
    ax3.plot(nx_, nh_, "s-", color="C0", ms=8)
    ax3.set_xscale("log")
    ax3.set_xlabel("samples per hop n (log)")
    ax3.set_ylabel("behavioral half-life (generations)")
    ax3.set_title("Half-life grows with data volume")
    ax3.set_ylim(ax2.get_ylim())
    for x, h in zip(nx_, nh_):
        ax3.annotate(f"{h:.2f}", (x, h), textcoords="offset points", xytext=(0, 8), fontsize=8)
    for x in ndead:
        ax3.annotate(
            "negligible",
            (x, 0),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            color="gray",
            ha="center",
        )
        ax3.plot([x], [0], "x", color="gray")

    fig.tight_layout()
    fig.savefig(OUT / "phase_diagram.png", dpi=150)
    fig.savefig(OUT / "phase_diagram.pdf")
    print("DONE")


if __name__ == "__main__":
    main()
