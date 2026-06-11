"""Verify that GhostTrace's public evidence surface matches committed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

QUALITY_GATES = (
    ("pytest", ["uv", "run", "pytest", "-q"]),
    ("ruff", ["uv", "run", "ruff", "check", "."]),
    ("pyright", ["uv", "run", "pyright"]),
    ("claim_check", ["uv", "run", "python", "-m", "ghosttrace.report.claim_check"]),
)

REQUIRED_ARTIFACTS = (
    "reports/toy_chain/pilot_a_faithful.json",
    "reports/toy_chain/verdict.json",
    "reports/toy_chain/chain_raw.json",
    "reports/toy_chain/phase_diagram_raw.json",
    "reports/toy_chain/toy_chain.png",
    "reports/toy_chain/toy_chain.pdf",
    "reports/toy_chain/phase_diagram.png",
    "reports/toy_chain/phase_diagram.pdf",
    "reports/reproducibility/clean_checkout_toy_rerun.json",
    "reports/pilot_b/verdict.json",
    "reports/pilot_b/base_score.json",
    "reports/pilot_b/treated_score.json",
    "reports/pilot_b/control_score.json",
    "reports/pilot_b/treated_gen.json",
    "reports/pilot_b/control_gen.json",
    "reports/pilot_b/scorer_validate.json",
    "reports/pilot_b/probe_14b.json",
    "reports/pilot_bft/base_score.json",
    "reports/pilot_bft/teacher_score.json",
    "reports/pilot_bft/llama1b_8bit_n4000/base_score.json",
    "reports/pilot_bft/llama1b_8bit_n4000/teacher_score.json",
    "reports/pilot_bft/llama1b_8bit_n4000/treated_gen.json",
    "reports/pilot_bft/llama1b_8bit_n4000/failure.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/config.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/runtime.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/base_score.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/persona_score.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/treated_generation.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/control_generation.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/treated_score.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/control_score.json",
    "reports/qwen25_0p5b_mlx_cat_smoke/verdict.json",
    "reports/qwen25_7b_mlx_cat_calibration/config.json",
    "reports/qwen25_7b_mlx_cat_calibration/runtime.json",
    "reports/qwen25_7b_mlx_cat_calibration/base_score.json",
    "reports/qwen25_7b_mlx_cat_calibration/persona_score.json",
    "reports/qwen25_7b_mlx_cat_calibration/treated_generation.json",
    "reports/qwen25_7b_mlx_cat_calibration/control_generation.json",
    "reports/qwen25_7b_mlx_cat_calibration/verdict.json",
    "reports/qwen25_7b_mlx_cat_singlehop/config.json",
    "reports/qwen25_7b_mlx_cat_singlehop/runtime.json",
    "reports/qwen25_7b_mlx_cat_singlehop/base_score.json",
    "reports/qwen25_7b_mlx_cat_singlehop/persona_score.json",
    "reports/qwen25_7b_mlx_cat_singlehop/treated_generation.json",
    "reports/qwen25_7b_mlx_cat_singlehop/control_generation.json",
    "reports/qwen25_7b_mlx_cat_singlehop/shuffled_generation.json",
    "reports/qwen25_7b_mlx_cat_singlehop/treated_score.json",
    "reports/qwen25_7b_mlx_cat_singlehop/control_score.json",
    "reports/qwen25_7b_mlx_cat_singlehop/shuffled_score.json",
    "reports/qwen25_7b_mlx_cat_singlehop/verdict.json",
    "reports/verdict.json",
    "reports/ARTIFACT_MANIFEST.json",
)

REQUIRED_PUBLIC_FILES = (
    ".github/workflows/artifact-integrity.yml",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATASET_CARD.md",
    "data/owl_teacher/README.md",
    "SECURITY.md",
    "docs/CUDA_REPRODUCTION_BRIEF.md",
    "docs/LOCAL_BOUNDARY_ANALYSIS.md",
    "docs/OPEN_RESEARCH_REGISTER.md",
    "docs/QWEN_CUDA_RUNBOOK.md",
    "docs/RELEASE_READINESS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/assets/ghosttrace-signal.svg",
    "requirements-cloud.txt",
)

PUBLIC_DOCS = (
    "README.md",
    "CLAIM_LEDGER.md",
    "paper/the_behavioral_half_life.md",
    "docs/CUDA_REPRODUCTION_BRIEF.md",
    "docs/LOCAL_BOUNDARY_ANALYSIS.md",
    "docs/OPEN_RESEARCH_REGISTER.md",
    "docs/QWEN_CUDA_RUNBOOK.md",
    "docs/RELEASE_READINESS.md",
    "docs/REPRODUCIBILITY.md",
)

PUBLIC_TEXT_SURFACE = PUBLIC_DOCS + (
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATASET_CARD.md",
    "data/owl_teacher/README.md",
    "SECURITY.md",
    "docs/PRE_REGISTRATION.md",
    "docs/RUN_LOG.md",
    "docs/SAFETY_PROTOCOL.md",
)

STALE_PAPER_PATTERNS = (
    "no valid transfer measurement exists yet",
    "forced-choice scorer returns a constant 0.0",
    "scorer is broken",
    "not yet measured because the forced-choice scorer is broken",
    "demonstrate the underlying transfer in a real LLM",
    "**Real-LLM reproduction**",
)

LOCAL_USER_PATH = "/" + "Users/"
LOCAL_TEMP_PATH = "/" + "tmp/"
LOCAL_FILE_SCHEME = "file" + "://"

FORBIDDEN_PUBLIC_PATTERNS = (
    LOCAL_USER_PATH,
    "REPLACE_WITH_OWNER",
    "top 0.01%",
    "haven't got a clue",
)

FORBIDDEN_REPORT_PATTERNS = (
    LOCAL_USER_PATH,
    LOCAL_TEMP_PATH,
    LOCAL_FILE_SCHEME,
)


def _load_json(rel: str) -> Any:
    with (ROOT / rel).open() as fh:
        return json.load(fh)


def _failures_from_quality_gates() -> list[str]:
    failures: list[str] = []
    for name, cmd in QUALITY_GATES:
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
            failures.append(f"{name} failed:\n" + "\n".join(tail))
    return failures


def _check_required_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"required artifact missing: {rel}")
        elif path.is_file() and path.stat().st_size == 0:
            failures.append(f"required artifact is empty: {rel}")
    return failures


def _check_required_public_files() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_PUBLIC_FILES:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"required public file missing: {rel}")
        elif path.is_file() and path.stat().st_size == 0:
            failures.append(f"required public file is empty: {rel}")
    return failures


def _expand_report_ref(ref: str) -> list[str]:
    clean = ref.strip("`'\".,;:) ]")
    if ".{png,pdf}" in clean:
        return [clean.replace(".{png,pdf}", ".png"), clean.replace(".{png,pdf}", ".pdf")]
    if ".{pdf,png}" in clean:
        return [clean.replace(".{pdf,png}", ".pdf"), clean.replace(".{pdf,png}", ".png")]
    return [clean]


def _check_doc_report_refs() -> list[str]:
    failures: list[str] = []
    pattern = re.compile(r"reports/[A-Za-z0-9_./{},-]+")
    for doc in PUBLIC_TEXT_SURFACE:
        text = (ROOT / doc).read_text()
        for match in pattern.findall(text):
            for rel in _expand_report_ref(match):
                if not (ROOT / rel).exists():
                    failures.append(f"{doc} cites missing report artifact: {rel}")
    return failures


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_artifact_manifest() -> list[str]:
    failures: list[str] = []
    manifest = _load_json("reports/ARTIFACT_MANIFEST.json")
    artifacts = manifest.get("artifacts", [])
    if manifest.get("artifact_count") != len(artifacts):
        failures.append("artifact manifest count does not match artifact list")

    manifest_paths = {str(row.get("path")) for row in artifacts}
    expected_paths = {rel for rel in REQUIRED_ARTIFACTS if rel != "reports/ARTIFACT_MANIFEST.json"}
    missing_from_manifest = sorted(expected_paths - manifest_paths)
    if missing_from_manifest:
        failures.append(
            "artifact manifest omits required artifacts: " + ", ".join(missing_from_manifest)
        )

    for row in artifacts:
        rel = str(row.get("path"))
        path = ROOT / rel
        if not path.exists():
            failures.append(f"manifested artifact missing: {rel}")
            continue
        if int(row.get("bytes", -1)) != path.stat().st_size:
            failures.append(f"manifest byte size mismatch: {rel}")
        if str(row.get("sha256")) != _sha256(path):
            failures.append(f"manifest sha256 mismatch: {rel}")
    return failures


def _check_llm_verdict() -> list[str]:
    failures: list[str] = []
    verdict = _load_json("reports/pilot_b/verdict.json")
    base = float(verdict["base"])
    treated = float(verdict["treated"])
    control = float(verdict["control"])
    gap = float(verdict["control_gap_pp"])
    expected_gap = (treated - control) * 100.0
    if base == 0.0 and treated == 0.0 and control == 0.0:
        failures.append("reports/pilot_b/verdict.json is still the stale all-zero verdict")
    if not math.isclose(gap, expected_gap, rel_tol=0.0, abs_tol=1e-9):
        failures.append(
            "reports/pilot_b/verdict.json control_gap_pp does not equal "
            "(treated-control)*100"
        )
    if bool(verdict.get("gate_pass_pointwise")):
        failures.append("persona-teacher gate unexpectedly marked as passing")

    for rel in ("reports/pilot_b/treated_gen.json", "reports/pilot_b/control_gen.json"):
        gen = _load_json(rel)
        if int(gen["tokens_found"]) != 0:
            failures.append(f"{rel} reports trait-token leakage")
        if int(gen["n"]) <= 0:
            failures.append(f"{rel} retained no numeric samples")

    scorer = _load_json("reports/pilot_b/scorer_validate.json")
    if not bool(scorer["scorer_nonzero"]) or not bool(scorer["scorer_moves"]):
        failures.append("scorer validation does not show a nonzero moving scorer")

    probe = _load_json("reports/pilot_b/probe_14b.json")
    if not bool(probe["scorer_nonzero"]) or not bool(probe["scorer_moves"]):
        failures.append("14B probe does not show a nonzero moving scorer")
    for arm in ("owl", "neutral"):
        if int(probe[f"{arm}_n_samples"]) <= 0:
            failures.append(f"14B {arm} channel retained no samples")
        if int(probe[f"{arm}_trait_tokens"]) != 0:
            failures.append(f"14B {arm} channel reports trait-token leakage")

    ft_failure = _load_json("reports/pilot_bft/llama1b_8bit_n4000/failure.json")
    if ft_failure["status"] != "blocked":
        failures.append("FT-teacher recovery artifact must be marked blocked")
    if bool(ft_failure["recursive_chain_allowed"]):
        failures.append("FT-teacher failure artifact incorrectly allows recursive chain")
    if int(ft_failure["treated_generation"]["n"]) >= 2000:
        failures.append("FT-teacher failure artifact no longer represents channel block")
    return failures


def _check_qwen_mlx_smoke() -> list[str]:
    failures: list[str] = []
    verdict = _load_json("reports/qwen25_0p5b_mlx_cat_smoke/verdict.json")
    if verdict.get("stage") != "smoke":
        failures.append("Qwen MLX smoke verdict must have stage=smoke")
    if verdict.get("claim_status") != "diagnostic_not_public_claim":
        failures.append("Qwen MLX smoke must remain diagnostic-only")
    if verdict.get("source_fidelity", {}).get("status") != "local_mlx_approximation":
        failures.append("Qwen MLX smoke must be labeled as a local MLX approximation")

    checks = verdict.get("gate_checks", {})
    for key in ("scorer_nonzero", "scorer_moves", "generation_clean", "retained_enough"):
        if not bool(checks.get(key)):
            failures.append(f"Qwen MLX smoke failed required apparatus check: {key}")
    if bool(checks.get("recursive_chain_allowed")):
        failures.append("Qwen MLX smoke must not allow a recursive chain")

    persona = _load_json("reports/qwen25_0p5b_mlx_cat_smoke/persona_score.json")
    if float(persona["delta_vs_base"]) <= 0.05:
        failures.append("Qwen MLX smoke persona scorer delta is too small")

    config = _load_json("reports/qwen25_0p5b_mlx_cat_smoke/config.json")
    if config.get("base_model") != "mlx-community/Qwen2.5-0.5B-Instruct-4bit":
        failures.append("Qwen MLX smoke config no longer uses the pinned Qwen2.5-0.5B model")
    if config.get("trait_name") != "cat":
        failures.append("Qwen MLX smoke config no longer uses the cat trait")

    for rel in (
        "reports/qwen25_0p5b_mlx_cat_smoke/treated_generation.json",
        "reports/qwen25_0p5b_mlx_cat_smoke/control_generation.json",
    ):
        gen = _load_json(rel)
        if int(gen["trait_tokens_found"]) != 0:
            failures.append(f"{rel} reports trait-token leakage")
        if int(gen["n_retained"]) < int(config["min_retained_per_arm"]):
            failures.append(f"{rel} retained too few clean numeric samples")
    return failures


def _check_qwen_mlx_7b_boundary() -> list[str]:
    failures: list[str] = []
    calibration = _load_json("reports/qwen25_7b_mlx_cat_calibration/verdict.json")
    singlehop = _load_json("reports/qwen25_7b_mlx_cat_singlehop/verdict.json")

    if calibration.get("stage") != "calibrate":
        failures.append("Qwen MLX 7B calibration verdict must have stage=calibrate")
    if calibration.get("claim_status") != "preflight_not_public_claim":
        failures.append("Qwen MLX 7B calibration must remain a preflight artifact")
    if singlehop.get("stage") != "singlehop":
        failures.append("Qwen MLX 7B single-hop verdict must have stage=singlehop")
    if singlehop.get("claim_status") != "local_mlx_boundary":
        failures.append("Qwen MLX 7B single-hop must be labeled as a local MLX boundary")

    for name, verdict in (("calibration", calibration), ("single-hop", singlehop)):
        if verdict.get("base_model") != "mlx-community/Qwen2.5-7B-Instruct-4bit":
            failures.append(f"Qwen MLX 7B {name} no longer uses the pinned 7B model")
        if verdict.get("trait") != "cat":
            failures.append(f"Qwen MLX 7B {name} no longer uses the cat trait")
        if verdict.get("source_fidelity", {}).get("status") != "local_mlx_approximation":
            failures.append(f"Qwen MLX 7B {name} must be labeled as local MLX approximation")
        checks = verdict.get("gate_checks", {})
        for key in ("scorer_nonzero", "scorer_moves", "generation_clean", "retained_enough"):
            if not bool(checks.get(key)):
                failures.append(f"Qwen MLX 7B {name} failed apparatus check: {key}")
        if bool(checks.get("recursive_chain_allowed")):
            failures.append(f"Qwen MLX 7B {name} must not allow recursive chain")
        for arm, gen in verdict.get("generation", {}).items():
            if int(gen.get("trait_tokens_found", -1)) != 0:
                failures.append(f"Qwen MLX 7B {name} {arm} reports trait-token leakage")

    cal_scores = calibration.get("scores", {})
    cal_base = float(cal_scores.get("base", {}).get("score", 0.0))
    cal_persona = float(cal_scores.get("persona_prompted", {}).get("score", 0.0))
    if cal_base <= 0.0 or cal_persona - cal_base <= 0.05:
        failures.append("Qwen MLX 7B calibration scorer did not show a usable persona delta")

    if bool(singlehop.get("gate_pass")):
        failures.append("Qwen MLX 7B single-hop unexpectedly marked as passing")
    gap = singlehop.get("control_gap", {})
    if float(gap.get("ci95_low_pp", 0.0)) > 5.0:
        failures.append("Qwen MLX 7B single-hop CI lower bound unexpectedly clears the gate")
    if float(gap.get("mean_pp", 0.0)) >= 0.0:
        failures.append("Qwen MLX 7B single-hop is no longer the recorded negative boundary")
    if not math.isclose(
        float(gap.get("mean_pp", 0.0)),
        (
            float(singlehop["scores"]["treated_student"]["score"])
            - float(singlehop["scores"]["control_student"]["score"])
        )
        * 100.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        failures.append("Qwen MLX 7B single-hop control_gap does not match score delta")

    config = _load_json("reports/qwen25_7b_mlx_cat_singlehop/config.json")
    if int(config.get("n_train_samples", 0)) != 4000:
        failures.append("Qwen MLX 7B single-hop config must train on 4000 clean samples")
    if not bool(config.get("include_shuffled")):
        failures.append("Qwen MLX 7B single-hop config must include shuffled control")
    for arm in ("treated", "control", "shuffled"):
        retained = int(singlehop["generation"][arm]["n_retained"])
        if retained < 4000:
            failures.append(f"Qwen MLX 7B single-hop {arm} retained fewer than 4000 samples")
    return failures


def _check_toy_artifacts() -> list[str]:
    failures: list[str] = []
    pilot = _load_json("reports/toy_chain/pilot_a_faithful.json")
    if float(pilot["same_init_mean"]) <= 0.5:
        failures.append("toy single-hop same-init mean no longer clears 0.5")
    if float(pilot["cross_model_mean"]) >= 0.2:
        failures.append("toy single-hop cross-init control is unexpectedly high")

    verdict = _load_json("reports/toy_chain/verdict.json")
    if verdict.get("classification", {}).get("class") != "decay":
        failures.append("toy chain verdict no longer classifies as decay")
    if float(verdict["half_life_generations"]) <= 0.0:
        failures.append("toy chain half-life is non-positive")

    phase = _load_json("reports/toy_chain/phase_diagram_raw.json")
    if not phase.get("results"):
        failures.append("phase_diagram_raw.json has no sweep results")
    return failures


def _check_clean_checkout_toy_rerun() -> list[str]:
    failures: list[str] = []
    audit = _load_json("reports/reproducibility/clean_checkout_toy_rerun.json")
    if audit.get("verdict") != "pass":
        failures.append("clean-checkout toy rerun audit is not marked pass")

    results = audit.get("results", {})
    expected = {
        "pilot_a_faithful": "reports/toy_chain/pilot_a_faithful.json",
        "chain_raw": "reports/toy_chain/chain_raw.json",
        "chain_verdict": "reports/toy_chain/verdict.json",
        "phase_diagram_raw": "reports/toy_chain/phase_diagram_raw.json",
    }
    for key, rel in expected.items():
        row = results.get(key, {})
        if not bool(row.get("matches_committed_artifact")):
            failures.append(f"clean-checkout toy rerun did not match {key}")
        committed_sha = str(row.get("committed_sha256"))
        if committed_sha != _sha256(ROOT / rel):
            failures.append(f"clean-checkout toy rerun sha mismatch for {rel}")
    return failures


def _check_public_wording() -> list[str]:
    failures: list[str] = []
    paper = (ROOT / "paper/the_behavioral_half_life.md").read_text()
    paper_lower = paper.lower()
    for phrase in STALE_PAPER_PATTERNS:
        if phrase.lower() in paper_lower:
            failures.append(f"paper contains stale LLM wording: {phrase}")

    readme = (ROOT / "README.md").read_text()
    if "Recursive LLM claims remain blocked" not in readme:
        failures.append("README must state that recursive LLM claims remain blocked")
    if "docs/assets/ghosttrace-signal.svg" not in readme:
        failures.append("README must include the GhostTrace animated SVG header")
    if "reports/ARTIFACT_MANIFEST.json" not in readme:
        failures.append("README must link the artifact manifest")

    ledger = (ROOT / "CLAIM_LEDGER.md").read_text()
    if "C10" not in ledger or "planned (needs committed contaminated-run artifact)" not in ledger:
        failures.append("CLAIM_LEDGER must keep C10 unasserted until artifact-backed")
    if "C11" not in ledger or "NOT supported (channel blocked)" not in ledger:
        failures.append("CLAIM_LEDGER must record the blocked FT-teacher gate as C11")
    if "C12" not in ledger or "Qwen2.5-7B source-faithful single-hop gate clears" not in ledger:
        failures.append("CLAIM_LEDGER must keep the Qwen single-hop gate as planned C12")
    if "C13" not in ledger or "Blocked until C12 is supported" not in ledger:
        failures.append("CLAIM_LEDGER must block recursive Qwen chain claims as C13")
    if "C15" not in ledger or "Qwen2.5-7B MLX local single-hop boundary" not in ledger:
        failures.append("CLAIM_LEDGER must record the Qwen2.5-7B MLX local boundary as C15")
    if "C16" not in ledger or "Clean-checkout toy rerun" not in ledger:
        failures.append("CLAIM_LEDGER must record the clean-checkout toy rerun as C16")

    for rel in PUBLIC_TEXT_SURFACE:
        text = (ROOT / rel).read_text()
        for phrase in FORBIDDEN_PUBLIC_PATTERNS:
            if phrase in text:
                failures.append(f"{rel} contains forbidden public-surface wording: {phrase}")
    return failures


def _check_report_metadata_privacy() -> list[str]:
    failures: list[str] = []
    report_jsons = [rel for rel in REQUIRED_ARTIFACTS if rel.endswith(".json")]
    for rel in report_jsons:
        text = (ROOT / rel).read_text()
        for phrase in FORBIDDEN_REPORT_PATTERNS:
            if phrase in text:
                failures.append(f"{rel} contains non-public local path metadata: {phrase}")
    return failures


def check_public_state(*, run_quality_gates: bool) -> list[str]:
    failures: list[str] = []
    failures.extend(_check_required_public_files())
    failures.extend(_check_required_artifacts())
    failures.extend(_check_artifact_manifest())
    failures.extend(_check_doc_report_refs())
    failures.extend(_check_toy_artifacts())
    failures.extend(_check_clean_checkout_toy_rerun())
    failures.extend(_check_llm_verdict())
    failures.extend(_check_qwen_mlx_smoke())
    failures.extend(_check_qwen_mlx_7b_boundary())
    failures.extend(_check_public_wording())
    failures.extend(_check_report_metadata_privacy())
    if run_quality_gates:
        failures.extend(_failures_from_quality_gates())
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-quality-gates",
        action="store_true",
        help="Only check artifacts/docs; skip pytest, ruff, pyright, and claim_check.",
    )
    args = parser.parse_args(argv)

    failures = check_public_state(run_quality_gates=not args.skip_quality_gates)
    if failures:
        print(f"FAIL: {len(failures)} public-state issue(s):")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("OK: public evidence surface matches committed artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
