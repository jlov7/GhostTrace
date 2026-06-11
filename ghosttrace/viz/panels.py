"""Compose the multi-panel paper figure from a run's results.json.

WHY: The paper needs one figure per run that pairs the generation curve with a
compact evidence summary, built straight from the validated
:class:`~ghosttrace.types.ChainResult` so the figure can never drift from the
numbers in ``results.json``. Renders headless to PNG and PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ghosttrace.types import ChainResult
from ghosttrace.viz import _mpl
from ghosttrace.viz._summary import curve_series


def render_all(run_dir: Path) -> Path:
    """Render the composed paper figure for run_dir from its results.json.

    Reads ``run_dir/results.json`` (a ChainResult), draws a two-panel figure
    (curve + evidence table) to ``run_dir/figures/panels.png`` and ``.pdf``, and
    returns the PNG path. Never calls ``plt.show()``.
    """

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}; run aggregate_run first")

    chain = ChainResult.model_validate_json(results_path.read_text())
    series = curve_series(chain)

    fig, axes = _mpl.subplots(ncols=2, nrows=1, figsize=(11.0, 4.5))
    ax_curve = axes[0]
    ax_table = axes[1]

    if series.ci_low and series.ci_high:
        ax_curve.fill_between(
            series.generations,
            series.ci_low,
            series.ci_high,
            alpha=0.25,
            color="C0",
            label="95% CI",
        )
    ax_curve.plot(series.generations, series.control_gaps, marker="o", color="C0", label="gap")
    ax_curve.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax_curve.set_xlabel("generation")
    ax_curve.set_ylabel("control gap (treated - control)")
    ax_curve.set_title(f"{chain.run_id}")
    ax_curve.legend(loc="best")

    _render_summary(ax_table, chain)

    fig.suptitle(f"GhostTrace run: {chain.run_id} [{chain.config_hash}]")
    fig.tight_layout()

    png = _mpl.save_png_pdf(fig, run_dir / "figures" / "panels")[0]
    _mpl.close(fig)
    return png


def _render_summary(ax: Any, chain: ChainResult) -> None:
    """Draw the evidence-summary panel as a borderless table."""

    ax.axis("off")
    series = curve_series(chain)
    summary: dict[str, Any] = chain.summary

    rows: list[tuple[str, str]] = [
        ("run_id", chain.run_id),
        ("config_hash", chain.config_hash),
        ("dynamics", series.dynamics_class),
        ("half-life", f"{series.halflife:.3g}" if series.halflife is not None else "n/a"),
        ("generations", str(len(series.generations))),
        ("n_records", str(len(chain.records))),
        ("control_arm", str(summary.get("control_arm", "n/a"))),
    ]
    rows.extend(_evidence_rows(summary.get("evidence")))

    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["metric", "value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax.set_title("evidence summary")


def _evidence_rows(evidence: object) -> list[tuple[str, str]]:
    """Coerce the (untyped) evidence mapping into sorted (key, value) rows."""

    if not isinstance(evidence, dict):
        return []
    evidence_map = cast(dict[Any, Any], evidence)
    rows: list[tuple[str, str]] = []
    for key in sorted(str(k) for k in evidence_map):
        value = evidence_map.get(key)
        if isinstance(value, (int, float)):
            rows.append((key, f"{float(value):.4g}"))
    return rows
