"""Build the committed GhostTrace artifact manifest.

The manifest is intentionally small and deterministic: it records the byte size
and SHA-256 of each public evidence artifact under ``reports/``. It does not
include timestamps, so rerunning the script only changes the manifest when an
artifact actually changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "ARTIFACT_MANIFEST.json"

DEFAULT_ARTIFACTS = (
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
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(paths: tuple[str, ...] = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(rel)
        artifacts.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "project": "GhostTrace",
        "purpose": "Committed evidence artifacts backing public claims.",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if the manifest is stale.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    manifest = build_manifest()
    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    output = args.output if args.output.is_absolute() else ROOT / args.output

    if args.check:
        if not output.exists():
            print(f"FAIL: missing artifact manifest: {output}")
            return 1
        current = output.read_text()
        if current != rendered:
            print(f"FAIL: artifact manifest is stale: {output}")
            return 1
        print("OK: artifact manifest is current.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    print(f"WROTE {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
