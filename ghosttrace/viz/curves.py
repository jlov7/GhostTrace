"""Generation-curve plot: control gap vs generation with a CI band.

WHY: The headline result of a chain is whether the control-adjusted trait gap
decays, persists, or amplifies across generations. A single curve with its
bootstrap CI band is the most legible way to show that. The figure must render
headless (CI, no display) and save both PNG and PDF for the paper.

The series are read from :attr:`ChainResult.summary` (written by
``report.aggregate.aggregate_run``): ``generations``, ``control_gaps``,
``ci_low``, ``ci_high``, ``dynamics_class``, ``halflife``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ghosttrace.types import ChainResult
from ghosttrace.viz import _mpl
from ghosttrace.viz._summary import CurveSeries, curve_series


def save_curve(chain_result: ChainResult, out_path: Path) -> list[Path]:
    """Render the generation curve for one chain to PNG and PDF.

    ``out_path`` is treated as a stem; ``.png`` and ``.pdf`` siblings are written
    and their paths returned. Never calls ``plt.show()``.
    """

    series = curve_series(chain_result)
    fig, ax = _mpl.subplots(figsize=(6.0, 4.0))
    _draw(ax, series, title=_title(series, chain_result.run_id))
    fig.tight_layout()
    paths = _mpl.save_png_pdf(fig, out_path)
    _mpl.close(fig)
    return paths


def _draw(ax: Any, series: CurveSeries, *, title: str) -> None:
    if series.ci_low and series.ci_high:
        ax.fill_between(
            series.generations,
            series.ci_low,
            series.ci_high,
            alpha=0.25,
            color="C0",
            label="95% CI",
        )
    ax.plot(series.generations, series.control_gaps, marker="o", color="C0", label="control gap")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("generation")
    ax.set_ylabel("control gap (treated - control)")
    ax.set_title(title)
    ax.legend(loc="best")


def _title(series: CurveSeries, run_id: str) -> str:
    if series.dynamics_class == "decay" and series.halflife is not None:
        return f"{run_id}: {series.dynamics_class} (half-life {series.halflife:.2f} gens)"
    return f"{run_id}: {series.dynamics_class}"
