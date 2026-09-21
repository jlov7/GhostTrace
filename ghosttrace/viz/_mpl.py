"""Typed facade over matplotlib's (largely untyped) pyplot surface.

WHY: matplotlib ships incomplete type stubs, so under pyright strict every
``ax.plot(...)`` / ``fig.savefig(...)`` call trips ``reportUnknownMemberType`` and
``Figure`` / ``Axes`` are flagged as non-exported. Mirroring how the stats layer
isolates scipy behind ``Any`` wrappers, this module confines all of that to one
place: it sets the headless backend and hands back ``Any``-typed figure/axes so
the plotting modules stay readable and pyright-clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless backend; must precede the pyplot import.

import matplotlib.pyplot as _plt  # noqa: E402
from matplotlib.colors import BoundaryNorm as _BoundaryNorm  # noqa: E402
from matplotlib.colors import ListedColormap as _ListedColormap  # noqa: E402
from matplotlib.patches import Patch as _Patch  # noqa: E402

# Single ``Any`` choke point: every matplotlib name is reached through these so
# pyright never has to reason about matplotlib's partial stubs downstream.
_pyplot: Any = _plt
_make_cmap: Any = _ListedColormap
_make_norm: Any = _BoundaryNorm
_make_patch: Any = _Patch


def subplots(**kwargs: Any) -> tuple[Any, Any]:
    """``plt.subplots`` returning ``(figure, axes-or-axes-array)`` as ``Any``."""

    result: Any = _pyplot.subplots(**kwargs)
    fig, axes = result
    return fig, axes


def close(fig: Any) -> None:
    """Close a figure to release its memory (never calls ``plt.show``)."""

    _pyplot.close(fig)


def categorical_cmap(colors: list[str], n_codes: int) -> tuple[Any, Any]:
    """Build a discrete colormap + integer-boundary norm for ``n_codes`` classes.

    Masked (absent) cells render white. Returns ``(cmap, norm)`` so the caller
    stays clear of matplotlib's untyped surface.
    """

    cmap: Any = _make_cmap(list(colors))
    cmap.set_bad(color="white")
    edges: Any = [i - 0.5 for i in range(n_codes + 1)]
    norm: Any = _make_norm(edges, cmap.N)
    return cmap, norm


def legend_patch(*, facecolor: str, label: str) -> Any:
    """A solid-colour legend handle (``matplotlib.patches.Patch``) as ``Any``."""

    return _make_patch(facecolor=facecolor, label=label)


def save_png_pdf(fig: Any, out_path: Path, **savefig_kwargs: Any) -> list[Path]:
    """Save ``fig`` to ``.png`` and ``.pdf`` siblings of ``out_path``.

    Creates the parent directory if needed and returns ``[png, pdf]``.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    png = out_path.with_suffix(".png")
    pdf = out_path.with_suffix(".pdf")
    figure: Any = fig
    figure.savefig(png, dpi=150, **savefig_kwargs)
    figure.savefig(pdf, **savefig_kwargs)
    return [png, pdf]
