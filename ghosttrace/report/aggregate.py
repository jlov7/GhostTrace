"""Aggregate per-cell GenerationRecords into a single ChainResult.

WHY: A run produces one :class:`~ghosttrace.types.GenerationRecord` per
(generation, branch, arm) cell, scattered as JSON under the run directory. The
stats and figure layers want one validated :class:`~ghosttrace.types.ChainResult`
per run, carrying the honest per-generation ``control_gap`` (treated minus a
control arm) with bootstrap CIs and a dynamics verdict. This module is the single
place that turns raw cells into that artifact, so every downstream consumer reads
the same ``results.json``.

The ``control_gap`` is the pre-registered effect size: at each generation it is
``trait_score(treated) - trait_score(control)``. Per the contract the
neutral-teacher arm is the primary control (it subtracts model-collapse / FT
drift); any other control arm present is used as a fallback when neutral-teacher
is absent. Per-generation CIs come from ``stats.bootstrap.gap_ci`` over the
per-branch treated and control scores; the trajectory is classified by
``stats.model_select.classify_dynamics``. All derived numbers live in
``ChainResult.summary`` because the frozen result model stores everything but the
raw records there.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from ghosttrace.config import ControlKind
from ghosttrace.stats import bootstrap, model_select
from ghosttrace.types import ChainResult, GenerationRecord

# The treated arm label and the preferred control arm (pre-registration: the
# neutral-teacher arm isolates the trait from generic fine-tuning drift).
_TREATED_ARM = "treated"
_PREFERRED_CONTROL = ControlKind.NEUTRAL_TEACHER.value

# Files/dirs this slice writes; never re-read them as input records.
_RESERVED_NAMES = frozenset({"results.json", "manifest.json", "config.resolved.json"})
_SKIP_DIRS = frozenset({"cards", "figs", "figures", "checkpoints", "data", "logs"})


def load_records(run_dir: Path) -> list[GenerationRecord]:
    """Load every GenerationRecord JSON under run_dir, skipping our own outputs.

    Records are discovered by shape (a JSON object that validates as a
    GenerationRecord) rather than by a rigid path convention, so any upstream
    layout works. Reserved filenames and known non-record subdirectories
    (``cards/``, ``data/`` ...) are skipped so re-aggregation stays idempotent.
    """

    records: list[GenerationRecord] = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in _RESERVED_NAMES:
            continue
        if _SKIP_DIRS.intersection(path.relative_to(run_dir).parts[:-1]):
            continue
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw, dict):
            continue
        try:
            records.append(GenerationRecord.model_validate(raw))
        except ValidationError:
            continue  # Not a GenerationRecord (e.g. a config or sweep blob).
    return records


def _control_arm(arms: set[str]) -> str | None:
    """Pick the control arm to difference against: neutral-teacher, else any.

    Returns ``None`` when no non-treated arm is present, in which case no
    control gap can be computed.
    """

    if _PREFERRED_CONTROL in arms:
        return _PREFERRED_CONTROL
    others = sorted(arms - {_TREATED_ARM})
    return others[0] if others else None


def _scores_by_gen_arm(
    records: list[GenerationRecord],
) -> dict[int, dict[str, list[float]]]:
    """Group trait scores by generation then arm."""

    table: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        table[rec.generation][rec.arm].append(rec.trait_score)
    return table


def _branch_gap_series(
    records: list[GenerationRecord], generations: list[int], control_arm: str
) -> list[list[float]]:
    """Per-generation list of per-branch control gaps (treated - control).

    A branch contributes a gap at a generation only when both the treated and
    the control arm are present for that (generation, branch) cell, so each gap
    is a true within-branch paired difference. Generations with no complete
    branch contribute an empty list, which the caller backfills.
    """

    # (generation, branch, arm) -> mean trait_score for that cell.
    cell: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    branches_at: dict[int, set[int]] = defaultdict(set)
    for rec in records:
        cell[(rec.generation, rec.branch, rec.arm)].append(rec.trait_score)
        branches_at[rec.generation].add(rec.branch)

    series: list[list[float]] = []
    for gen in generations:
        gaps: list[float] = []
        for branch in sorted(branches_at[gen]):
            treated = cell.get((gen, branch, _TREATED_ARM))
            control = cell.get((gen, branch, control_arm))
            if treated and control:
                gaps.append(_mean(treated) - _mean(control))
        series.append(gaps)
    return series


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_run(
    run_dir: Path,
    *,
    n_resamples: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> Path:
    """Aggregate per-cell records in run_dir into ``run_dir/results.json``.

    Reads every GenerationRecord under run_dir, computes a BCa control-gap CI per
    generation (treated minus the chosen control arm), classifies the dynamics
    across branches, writes a validated :class:`ChainResult` whose ``summary``
    holds the derived series and verdict, and returns the output path.

    ``run_id`` / ``config_hash`` are read from ``run_dir/manifest.json`` when
    present (the provenance contract) and otherwise derived from the directory
    name, so aggregation never depends on a live config object.
    """

    records = load_records(run_dir)
    if not records:
        raise ValueError(f"no GenerationRecord JSON files found under {run_dir}")

    run_id, config_hash = _identity(run_dir)

    by_gen_arm = _scores_by_gen_arm(records)
    generations = sorted(by_gen_arm)

    arms_present: set[str] = {rec.arm for rec in records}
    control_arm = _control_arm(arms_present)

    control_gaps: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    treated_means: list[float] = []
    control_means: list[float] = []

    for gen in generations:
        treated = by_gen_arm[gen].get(_TREATED_ARM, [])
        treated_means.append(_mean(treated) if treated else float("nan"))
        if control_arm is not None and treated and by_gen_arm[gen].get(control_arm):
            control = by_gen_arm[gen][control_arm]
            control_means.append(_mean(control))
            mean, lo, hi = bootstrap.gap_ci(
                treated, control, level=level, n_resamples=n_resamples, seed=seed
            )
        else:
            control_means.append(float("nan"))
            mean = lo = hi = float("nan")
        control_gaps.append(mean)
        ci_low.append(lo)
        ci_high.append(hi)

    verdict = _classify(records, generations, control_arm)

    summary: dict[str, Any] = {
        "generations": generations,
        "control_arm": control_arm,
        "treated_arm": _TREATED_ARM,
        "treated_means": treated_means,
        "control_means": control_means,
        "control_gaps": control_gaps,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": level,
        "n_resamples": n_resamples,
        "seed": seed,
        "dynamics_class": verdict["class"],
        "halflife": verdict["halflife"],
        "evidence": verdict["evidence"],
        "n_generations": len(generations),
        "n_records": len(records),
        "arms": sorted(arms_present),
    }

    result = ChainResult(
        run_id=run_id,
        config_hash=config_hash,
        records=records,
        summary=summary,
    )

    out_path = run_dir / "results.json"
    out_path.write_text(result.model_dump_json(indent=2))
    return out_path


def _classify(
    records: list[GenerationRecord], generations: list[int], control_arm: str | None
) -> dict[str, Any]:
    """Run the dynamics classifier, degrading gracefully on thin data.

    Uses per-branch control gaps where a control arm exists; otherwise falls
    back to raw treated trait scores. Generations whose branch list is empty are
    backfilled with the cross-generation treated mean so the classifier always
    receives a complete, non-empty trajectory.
    """

    if control_arm is not None:
        branch_gaps = _branch_gap_series(records, generations, control_arm)
    else:
        branch_gaps = _treated_only_series(records, generations)

    # Backfill any generation with no observations using the global mean of the
    # observed gaps, so classify_dynamics never sees an empty branch list.
    observed = [v for gaps in branch_gaps for v in gaps]
    fill = _mean(observed) if observed else 0.0
    branch_gaps = [gaps if gaps else [fill] for gaps in branch_gaps]

    verdict = model_select.classify_dynamics(generations, branch_gaps)
    cls = str(verdict["class"])
    halflife_obj = verdict["halflife"]
    halflife = float(halflife_obj) if isinstance(halflife_obj, (int, float)) else None
    evidence_obj: Any = verdict["evidence"]
    evidence: dict[str, float] = {}
    if isinstance(evidence_obj, dict):
        raw_evidence = cast(dict[Any, Any], evidence_obj)
        for key, value in raw_evidence.items():
            if isinstance(value, (int, float)):
                evidence[str(key)] = float(value)
    return {"class": cls, "halflife": halflife, "evidence": evidence}


def _treated_only_series(
    records: list[GenerationRecord], generations: list[int]
) -> list[list[float]]:
    """Per-generation per-branch treated trait scores (no control available)."""

    cell: dict[tuple[int, int], list[float]] = defaultdict(list)
    branches_at: dict[int, set[int]] = defaultdict(set)
    for rec in records:
        if rec.arm != _TREATED_ARM:
            continue
        cell[(rec.generation, rec.branch)].append(rec.trait_score)
        branches_at[rec.generation].add(rec.branch)
    series: list[list[float]] = []
    for gen in generations:
        series.append([_mean(cell[(gen, b)]) for b in sorted(branches_at[gen])])
    return series


def _identity(run_dir: Path) -> tuple[str, str]:
    """Resolve (run_id, config_hash) from the run's manifest, else from its name."""

    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            data: Any = json.loads(manifest.read_text())
            if isinstance(data, dict):
                manifest_data = cast(dict[Any, Any], data)
                run_id = str(manifest_data.get("run_id", run_dir.name))
                config_hash = str(manifest_data.get("config_hash", "unknown"))
                return run_id, config_hash
        except (json.JSONDecodeError, OSError):
            pass
    # Convention: run_id == "<name>-<hash>"; recover the hash tail if present.
    name = run_dir.name
    config_hash = name.rsplit("-", 1)[-1] if "-" in name else "unknown"
    return name, config_hash
