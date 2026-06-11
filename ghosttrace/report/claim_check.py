"""Verify that public claims are backed by concrete committed artifacts.

WHY: GhostTrace's value rests on no headline claim outrunning its evidence.
``CLAIM_LEDGER.md`` is a markdown table; the pre-registration rule is that a row
whose status is ``supported`` or ``asserted`` must cite at least one committed
backing artifact, and any row that names a backing artifact must point to a real
path. This module enforces that invariant in CI: :func:`check_claims` returns the
list of violations and :func:`main` exits nonzero when any exist.

The ledger is a GitHub-style table whose header is::

    | id | claim | status | backing run_id(s) | figure / stat |

Columns are matched by header name (case-insensitive, substring-tolerant) so
column order and small wording changes do not break the checker. Multiple
artifacts in the backing column may be separated with ``;`` or ``,``. The empty
``run_id`` placeholder in the shipped ledger is the em dash ``—`` (or ``-``),
which is treated as "no backing run".
"""

from __future__ import annotations

import sys
from pathlib import Path
import re

# Default location: repo root, two levels up from this module.
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "CLAIM_LEDGER.md"

_ASSERTED = "asserted"
_SUPPORTED = "supported"
# Cell contents that mean "no run id was supplied".
_EMPTY_MARKERS = frozenset({"", "-", "—", "--", "n/a", "tbd", "none"})
_BACKING_SPLIT = re.compile(r"\s*(?:;|,|<br\s*/?>)\s*", re.IGNORECASE)


def _split_row(line: str) -> list[str]:
    """Split a markdown table row into trimmed cells, dropping edge pipes."""

    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(cells: list[str]) -> bool:
    """True for the ``| --- | --- |`` separator row beneath a table header."""

    return all(cell != "" and set(cell) <= {"-", ":"} for cell in cells)


def _find_column(header: list[str], *needles: str) -> int:
    """Return the index of the first header cell containing any needle, or -1."""

    for idx, cell in enumerate(header):
        if any(needle in cell for needle in needles):
            return idx
    return -1


def _requires_backing(status: str) -> bool:
    """True when a status asserts positive support and needs artifacts."""

    normalized = status.strip().lower()
    if not normalized or normalized.startswith("not supported") or normalized.startswith("planned"):
        return False
    return normalized.startswith(_ASSERTED) or normalized.startswith(_SUPPORTED)


def _extract_backing_paths(cell: str) -> list[str]:
    """Split a backing-run cell into path-like entries, dropping empty markers."""

    stripped = cell.strip()
    if stripped.lower() in _EMPTY_MARKERS:
        return []
    parts = [p.strip(" `") for p in _BACKING_SPLIT.split(stripped)]
    return [p for p in parts if p and p.lower() not in _EMPTY_MARKERS]


def check_claims(ledger_path: Path = DEFAULT_LEDGER) -> list[str]:
    """Return a list of human-readable violations in the claim ledger.

    A violation is any positive ``supported`` / ``asserted`` row with no backing
    artifact, any row that names a missing backing artifact, or any malformed
    ledger structure. Relative paths resolve from the repository root for the
    shipped ledger, and from the ledger's directory for test ledgers.
    """

    if not ledger_path.exists():
        return [f"claim ledger not found: {ledger_path}"]

    try:
        text = ledger_path.read_text()
    except OSError as exc:
        return [f"could not read claim ledger {ledger_path}: {exc}"]

    rows = [line for line in text.splitlines() if line.strip().startswith("|")]
    if not rows:
        return [f"no markdown table found in {ledger_path}"]

    header = [cell.lower() for cell in _split_row(rows[0])]
    status_idx = _find_column(header, "status")
    claim_idx = _find_column(header, "claim")
    run_idx = _find_column(header, "run_id", "run id", "backing")
    if min(status_idx, claim_idx, run_idx) < 0:
        return [
            "ledger header must contain 'claim', 'status', and a backing run_id "
            f"column; got {header}"
        ]

    root = ledger_path.resolve().parent
    if ledger_path.resolve() == DEFAULT_LEDGER.resolve():
        root = DEFAULT_LEDGER.resolve().parents[0]

    violations: list[str] = []
    for line in rows[1:]:
        cells = _split_row(line)
        if _is_separator(cells):
            continue
        if len(cells) <= max(status_idx, claim_idx, run_idx):
            violations.append(f"malformed row (too few columns): {line.strip()}")
            continue
        status = cells[status_idx].lower()
        backing_cell = cells[run_idx].strip()
        backing_paths = _extract_backing_paths(backing_cell)
        claim = cells[claim_idx] or "<empty claim>"
        if _requires_backing(status) and not backing_paths:
            violations.append(f"supported/asserted claim has no backing artifact: {claim!r}")
            continue
        for backing in backing_paths:
            path = Path(backing)
            resolved = path if path.is_absolute() else root / path
            if not resolved.exists():
                violations.append(
                    f"backing artifact not found for claim {claim!r}: {backing}"
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print violations and exit nonzero if any exist."""

    args = sys.argv[1:] if argv is None else argv
    ledger_path = Path(args[0]) if args else DEFAULT_LEDGER
    violations = check_claims(ledger_path)
    if violations:
        print(f"FAIL: {len(violations)} claim-ledger violation(s):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("OK: all public claims are backed by existing artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
