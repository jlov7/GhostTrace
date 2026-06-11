"""Human-readable markdown cards for individual cells and anomalies.

WHY: Reviewers and the safety protocol need a compact, per-cell artifact they can
skim without parsing JSON: which (generation, branch, arm) was measured, the
trait score, utility retention, and the seeds that make it reproducible.
Quarantine cards make excluded/failed cells loudly visible so they cannot
silently bias an aggregate. Cards live under ``run_dir/cards`` (the provenance
contract already creates that directory).
"""

from __future__ import annotations

from pathlib import Path

from ghosttrace.types import GenerationRecord


def evidence_card(record: GenerationRecord) -> str:
    """Render a single GenerationRecord as a markdown evidence card."""

    lines = [
        f"# Evidence: generation {record.generation}, branch {record.branch}, arm {record.arm}",
        "",
        f"- **Generation:** {record.generation}",
        f"- **Branch:** {record.branch}",
        f"- **Arm:** {record.arm}",
        f"- **Trait score:** {record.trait_score:.6g}",
        f"- **Utility retention:** {_fmt_opt(record.utility_retention)}",
        f"- **Probe count (n):** {record.n}",
        f"- **Seeds:** {_fmt_mapping(record.seeds)}",
    ]
    if record.artifacts:
        lines.append(f"- **Artifacts:** {_fmt_mapping(record.artifacts)}")
    lines.append("")
    return "\n".join(lines)


def quarantine_card(record: GenerationRecord, *, reason: str) -> str:
    """Render a quarantine card for an anomalous or failed cell.

    The ``reason`` is stated up front so a reader knows immediately why the cell
    is excluded from aggregation.
    """

    lines = [
        f"# QUARANTINE: generation {record.generation}, branch {record.branch}, arm {record.arm}",
        "",
        f"- **Reason:** {reason}",
        f"- **Generation:** {record.generation}",
        f"- **Branch:** {record.branch}",
        f"- **Arm:** {record.arm}",
        f"- **Trait score:** {record.trait_score:.6g}",
        f"- **Utility retention:** {_fmt_opt(record.utility_retention)}",
        f"- **Probe count (n):** {record.n}",
        f"- **Seeds:** {_fmt_mapping(record.seeds)}",
        "",
        "> This cell is excluded from aggregation. Investigate before relying on it.",
        "",
    ]
    return "\n".join(lines)


def write_card(run_dir: Path, record: GenerationRecord, markdown: str, *, suffix: str = "") -> Path:
    """Write a card markdown file under ``run_dir/cards`` and return its path."""

    cards_dir = run_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    name = f"gen{record.generation}_b{record.branch}_{record.arm}{suffix}.md"
    out_path = cards_dir / name
    out_path.write_text(markdown)
    return out_path


def _fmt_opt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _fmt_mapping(mapping: dict[str, int] | dict[str, str]) -> str:
    if not mapping:
        return "{}"
    return ", ".join(f"{k}={v}" for k, v in sorted(mapping.items()))
