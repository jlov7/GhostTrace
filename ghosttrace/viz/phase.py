"""Phase diagram: dynamics class over two swept axes.

WHY: A sweep (e.g. ``dataset_size`` x ``n_generations``) asks where in parameter
space a trait decays, persists, or amplifies. A categorical heatmap of the
dynamics class over the two swept axes is the compact summary of that sweep. It
renders headless to PNG and PDF for the paper.

The frozen contract has no sweep type, so this module defines a small,
self-contained input (:class:`PhaseCell` / :class:`PhaseGrid`) that the sweep
runner can populate from each cell's aggregated :class:`ChainResult` (its
``summary['dynamics_class']``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ghosttrace.viz import _mpl

# Stable class -> integer code -> color so every phase diagram is comparable.
_CLASS_ORDER = ("null", "decay", "persist", "amplify")
_CLASS_COLORS = ("#cccccc", "#1f77b4", "#2ca02c", "#d62728")
_CLASS_CODE = {name: i for i, name in enumerate(_CLASS_ORDER)}


@dataclass(frozen=True)
class PhaseCell:
    """One sweep cell: its position on the two axes and its dynamics class."""

    x: float
    y: float
    dynamics_class: str


@dataclass(frozen=True)
class PhaseGrid:
    """A 2-D sweep: named axes plus the cells to colour by dynamics class."""

    run_id: str
    x_axis: str
    y_axis: str
    cells: Sequence[PhaseCell] = field(default_factory=tuple)


def save_phase(grid: PhaseGrid, out_path: Path) -> list[Path]:
    """Render a sweep grid as a categorical phase diagram (PNG + PDF).

    The grid's two axes become the x and y of a heatmap whose cell colour is the
    chain's dynamics class. Returns the written ``[png, pdf]`` paths and never
    calls ``plt.show()``.
    """

    if not grid.cells:
        raise ValueError("cannot render a phase diagram from an empty sweep grid")

    x_values = sorted({cell.x for cell in grid.cells})
    y_values = sorted({cell.y for cell in grid.cells})
    x_index = {v: i for i, v in enumerate(x_values)}
    y_index = {v: i for i, v in enumerate(y_values)}

    # -1 marks cells absent from the sweep so they render blank.
    np_any: Any = np
    codes: Any = np_any.full((len(y_values), len(x_values)), -1, dtype=float)
    for cell in grid.cells:
        code = _CLASS_CODE.get(cell.dynamics_class, _CLASS_CODE["null"])
        codes[y_index[cell.y], x_index[cell.x]] = code

    masked: Any = np_any.ma.masked_less(codes, 0)
    cmap, norm = _mpl.categorical_cmap(list(_CLASS_COLORS), len(_CLASS_ORDER))

    fig, ax = _mpl.subplots(figsize=(6.0, 5.0))
    _draw(ax, grid, masked, cmap=cmap, norm=norm, x_values=x_values, y_values=y_values)
    fig.tight_layout()
    paths = _mpl.save_png_pdf(fig, out_path, bbox_inches="tight")
    _mpl.close(fig)
    return paths


def _draw(
    ax: Any,
    grid: PhaseGrid,
    masked: Any,
    *,
    cmap: Any,
    norm: Any,
    x_values: list[float],
    y_values: list[float],
) -> None:
    ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([_fmt(v) for v in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([_fmt(v) for v in y_values])
    ax.set_xlabel(grid.x_axis)
    ax.set_ylabel(grid.y_axis)
    ax.set_title(f"Phase diagram: {grid.run_id}")
    handles = [
        _mpl.legend_patch(facecolor=color, label=name)
        for name, color in zip(_CLASS_ORDER, _CLASS_COLORS, strict=True)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))


def _fmt(value: float) -> str:
    """Format an axis tick: integer-looking floats render without a decimal."""

    if value == int(value):
        return str(int(value))
    return f"{value:.3g}"
