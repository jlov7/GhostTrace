"""Fast unit tests for the reporting + viz slice (no MLX, no real LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghosttrace.config import ControlKind
from ghosttrace.report import cards, claim_check
from ghosttrace.report.aggregate import aggregate_run, load_records
from ghosttrace.types import ChainResult, GenerationRecord
from ghosttrace.viz.curves import save_curve
from ghosttrace.viz.panels import render_all
from ghosttrace.viz.phase import PhaseCell, PhaseGrid, save_phase

_TREATED = "treated"
_CONTROL = ControlKind.NEUTRAL_TEACHER.value


def _record(
    *, generation: int, branch: int, arm: str, score: float, seed: int = 0
) -> GenerationRecord:
    return GenerationRecord(
        generation=generation,
        branch=branch,
        arm=arm,
        trait_score=score,
        utility_retention=0.95,
        seeds={"eval": seed},
        n=8,
        artifacts={},
    )


def _write_decay_run(run_dir: Path, *, branches: int = 3, n_gen: int = 6) -> None:
    """Synthetic decaying-gap run: treated decays toward the flat control."""

    run_dir.mkdir(parents=True, exist_ok=True)
    control_level = 0.2
    for branch in range(branches):
        for gen in range(n_gen):
            gap = 0.7 * (0.6**gen) + 0.01 * branch  # decaying control gap
            treated = _record(
                generation=gen,
                branch=branch,
                arm=_TREATED,
                score=control_level + gap,
                seed=100 + branch,
            )
            control = _record(
                generation=gen,
                branch=branch,
                arm=_CONTROL,
                score=control_level,
                seed=200 + branch,
            )
            (run_dir / f"g{gen}_b{branch}_treated.json").write_text(treated.model_dump_json())
            (run_dir / f"g{gen}_b{branch}_control.json").write_text(control.model_dump_json())


def test_load_records_skips_non_records(tmp_path: Path) -> None:
    _write_decay_run(tmp_path, branches=1, n_gen=2)
    (tmp_path / "garbage.json").write_text('{"not": "a record"}')
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "list.json").write_text("[1, 2, 3]")
    records = load_records(tmp_path)
    assert len(records) == 4  # 2 gens x (treated + control)
    assert all(isinstance(r, GenerationRecord) for r in records)


def test_aggregate_run_produces_valid_chain_result(tmp_path: Path) -> None:
    _write_decay_run(tmp_path, branches=3, n_gen=6)
    out = aggregate_run(tmp_path, n_resamples=300, seed=7)
    assert out.exists() and out.read_text().strip() != ""

    chain = ChainResult.model_validate_json(out.read_text())
    assert chain.run_id == tmp_path.name
    assert len(chain.records) == 36  # 6 gens x 3 branches x 2 arms

    summary = chain.summary
    assert summary["generations"] == [0, 1, 2, 3, 4, 5]
    assert summary["control_arm"] == _CONTROL
    gaps = summary["control_gaps"]
    assert len(gaps) == 6
    # Gap is positive and decreasing for a decaying run.
    assert gaps[0] > gaps[-1] > 0
    # CI band brackets each point estimate.
    for lo, mid, hi in zip(summary["ci_low"], gaps, summary["ci_high"], strict=True):
        assert lo <= mid <= hi
    assert summary["dynamics_class"] in ("decay", "persist", "amplify", "null")
    assert isinstance(summary["evidence"], dict)


def test_aggregate_run_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no GenerationRecord"):
        aggregate_run(tmp_path)


def test_aggregate_run_is_idempotent(tmp_path: Path) -> None:
    _write_decay_run(tmp_path, branches=2, n_gen=3)
    out1 = aggregate_run(tmp_path, n_resamples=200, seed=1)
    chain1 = ChainResult.model_validate_json(out1.read_text())
    # results.json from the first pass must not be re-ingested as a record.
    out2 = aggregate_run(tmp_path, n_resamples=200, seed=1)
    chain2 = ChainResult.model_validate_json(out2.read_text())
    assert out1 == out2
    assert len(chain1.records) == len(chain2.records) == 12


def test_aggregate_run_reads_manifest_identity(tmp_path: Path) -> None:
    _write_decay_run(tmp_path, branches=2, n_gen=3)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "pilot-xyz", "config_hash": "deadbeef0000"})
    )
    out = aggregate_run(tmp_path, n_resamples=200, seed=1)
    chain = ChainResult.model_validate_json(out.read_text())
    assert chain.run_id == "pilot-xyz"
    assert chain.config_hash == "deadbeef0000"
    # The manifest must not be counted as a record.
    assert len(chain.records) == 12


def test_evidence_and_quarantine_cards(tmp_path: Path) -> None:
    rec = _record(generation=2, branch=1, arm=_TREATED, score=0.8)
    card = cards.evidence_card(rec)
    assert "Evidence" in card
    assert "generation 2" in card
    assert "0.8" in card

    qcard = cards.quarantine_card(rec, reason="probe diverged")
    assert "QUARANTINE" in qcard
    assert "probe diverged" in qcard

    path = cards.write_card(tmp_path, rec, qcard, suffix="_quarantine")
    assert path.exists()
    assert path.parent.name == "cards"
    assert path.read_text() == qcard


def _chain_with_summary(run_id: str, dynamics: str) -> ChainResult:
    return ChainResult(
        run_id=run_id,
        config_hash="abc123",
        records=[_record(generation=0, branch=0, arm=_TREATED, score=0.5)],
        summary={
            "generations": [0, 1, 2, 3],
            "control_gaps": [0.7, 0.42, 0.25, 0.15],
            "ci_low": [0.6, 0.35, 0.2, 0.1],
            "ci_high": [0.8, 0.49, 0.3, 0.2],
            "dynamics_class": dynamics,
            "halflife": 1.36,
            "control_arm": _CONTROL,
            "evidence": {"exp_tau": 1.97, "mk_p": 0.01},
        },
    )


def test_save_curve_writes_png_and_pdf(tmp_path: Path) -> None:
    chain = _chain_with_summary("chain-0", "decay")
    paths = save_curve(chain, tmp_path / "curve")
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_render_all_panels(tmp_path: Path) -> None:
    _write_decay_run(tmp_path, branches=3, n_gen=6)
    aggregate_run(tmp_path, n_resamples=200, seed=3)
    png = render_all(tmp_path)
    assert png.exists() and png.stat().st_size > 0
    assert (tmp_path / "figures" / "panels.pdf").exists()


def test_render_all_missing_results_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        render_all(tmp_path)


def test_save_phase_grid(tmp_path: Path) -> None:
    classes = ["decay", "persist", "amplify", "null"]
    cells: list[PhaseCell] = []
    idx = 0
    for size in (100.0, 1000.0):
        for gens in (3.0, 5.0):
            cells.append(PhaseCell(x=size, y=gens, dynamics_class=classes[idx % 4]))
            idx += 1
    grid = PhaseGrid(run_id="sweep-1", x_axis="dataset_size", y_axis="n_generations", cells=cells)
    paths = save_phase(grid, tmp_path / "phase")
    assert [p.suffix for p in paths] == [".png", ".pdf"]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_save_phase_empty_raises(tmp_path: Path) -> None:
    grid = PhaseGrid(run_id="empty", x_axis="a", y_axis="b", cells=[])
    with pytest.raises(ValueError, match="empty sweep grid"):
        save_phase(grid, tmp_path / "phase")


def test_claim_check_flags_broken_ledger(tmp_path: Path) -> None:
    (tmp_path / "run-abc").write_text("{}")
    ledger = tmp_path / "CLAIM_LEDGER.md"
    ledger.write_text(
        "| id | claim | status | backing run_id(s) | figure / stat |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| C1 | transfer reproduces | asserted | run-abc | fig1 |\n"
        "| C2 | amplifies under sweep | asserted | — | fig2 |\n"
        "| C3 | future idea | planned | — | — |\n"
    )
    violations = claim_check.check_claims(ledger)
    assert len(violations) == 1
    assert "amplifies under sweep" in violations[0]
    assert claim_check.main([str(ledger)]) == 1


def test_claim_check_flags_missing_supported_artifact(tmp_path: Path) -> None:
    ledger = tmp_path / "CLAIM_LEDGER.md"
    ledger.write_text(
        "| status | claim | backing run_id(s) |\n"
        "| --- | --- | --- |\n"
        "| supported | toy reproduces | missing.json |\n"
    )
    violations = claim_check.check_claims(ledger)
    assert len(violations) == 1
    assert "backing artifact not found" in violations[0]


def test_claim_check_passes_clean_ledger(tmp_path: Path) -> None:
    (tmp_path / "run-1").write_text("{}")
    (tmp_path / "run-2").write_text("{}")
    ledger = tmp_path / "CLAIM_LEDGER.md"
    ledger.write_text(
        "| status | claim | backing run_id(s) |\n"
        "| --- | --- | --- |\n"
        "| asserted | trait persists | run-1 |\n"
        "| planned | future claim | — |\n"
        "| supported | toy reproduces | run-2 |\n"
    )
    assert claim_check.check_claims(ledger) == []
    assert claim_check.main([str(ledger)]) == 0


def test_claim_check_real_ledger_is_clean() -> None:
    # The shipped ledger has no asserted rows yet, so it must pass.
    assert claim_check.check_claims() == []


def test_claim_check_missing_ledger_is_violation(tmp_path: Path) -> None:
    violations = claim_check.check_claims(tmp_path / "nope.md")
    assert len(violations) == 1
    assert "not found" in violations[0]


def test_claim_check_bad_header(tmp_path: Path) -> None:
    ledger = tmp_path / "CLAIM_LEDGER.md"
    ledger.write_text("| foo | bar |\n| --- | --- |\n| a | b |\n")
    violations = claim_check.check_claims(ledger)
    assert len(violations) == 1
    assert "must contain" in violations[0]
