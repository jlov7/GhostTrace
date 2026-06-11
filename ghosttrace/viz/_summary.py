"""Typed accessors for the derived series stored in ``ChainResult.summary``.

WHY: ``ChainResult.summary`` is an untyped ``dict[str, Any]`` (the frozen
contract), but the figure code needs strongly-typed, pyright-clean access to the
per-generation series that ``report.aggregate`` wrote there. Centralising the
coercion here keeps the plotting modules free of defensive ``isinstance`` noise
and gives a single source of truth for the summary schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ghosttrace.types import ChainResult


@dataclass(frozen=True)
class CurveSeries:
    """The per-generation series needed to draw a generation curve."""

    generations: list[float]
    control_gaps: list[float]
    ci_low: list[float]
    ci_high: list[float]
    dynamics_class: str
    halflife: float | None


def _float_list(summary: dict[str, Any], key: str) -> list[float]:
    raw: Any = summary.get(key)
    if not isinstance(raw, list):
        return []
    items = cast(list[Any], raw)
    values: list[float] = []
    for item in items:
        if isinstance(item, (int, float)):
            values.append(float(item))
    return values


def curve_series(chain_result: ChainResult) -> CurveSeries:
    """Extract a :class:`CurveSeries` from a chain's summary.

    Missing or malformed entries degrade to empty lists / sensible defaults so a
    partially-populated run still plots without raising.
    """

    summary: dict[str, Any] = chain_result.summary
    gaps = _float_list(summary, "control_gaps")
    gens = _float_list(summary, "generations")
    if not gens:
        gens = [float(i) for i in range(len(gaps))]

    cls_raw = summary.get("dynamics_class")
    dynamics_class = str(cls_raw) if isinstance(cls_raw, str) else "null"

    hl_raw = summary.get("halflife")
    halflife = float(hl_raw) if isinstance(hl_raw, (int, float)) else None

    return CurveSeries(
        generations=gens,
        control_gaps=gaps,
        ci_low=_float_list(summary, "ci_low"),
        ci_high=_float_list(summary, "ci_high"),
        dynamics_class=dynamics_class,
        halflife=halflife,
    )
