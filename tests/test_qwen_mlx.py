"""Pure tests for the Mac-local Qwen2.5 MLX gate."""

from __future__ import annotations

from pathlib import Path

from ghosttrace.config import FineTuneMethod, FineTuneSpec
from ghosttrace.finetune.mlx_lora import mlx_lora_parameters
from ghosttrace.local.qwen_mlx import (
    QWEN25_MLX_7B_MODEL,
    QWEN25_MLX_SMOKE_MODEL,
    QwenMlxSpec,
    build_local_mlx_verdict,
    cap_generation_for_training,
    qwen_mlx_spec_for_stage,
)
from ghosttrace.types import GenerationOutput, ScoreResult


def _score(value: float, n: int = 80) -> ScoreResult:
    return ScoreResult(
        score=value,
        n=n,
        method="forced_choice",
        per_probe=[value] * n,
        meta={"trait": "cat"},
    )


def _generation(n: int = 64) -> dict[str, dict[str, object]]:
    return {
        "treated": {"n_retained": n, "trait_tokens_found": 0},
        "control": {"n_retained": n, "trait_tokens_found": 0},
    }


def test_mlx_lora_parameters_use_peft_alpha_over_rank_scale() -> None:
    cfg = FineTuneSpec(
        method=FineTuneMethod.LORA,
        lora_rank=8,
        lora_alpha=8.0,
        lora_dropout=0.1,
    )
    assert mlx_lora_parameters(cfg) == {"rank": 8, "dropout": 0.1, "scale": 1.0}


def test_qwen_mlx_smoke_defaults_are_local_qwen25_cat() -> None:
    spec = qwen_mlx_spec_for_stage("smoke")
    spec.validate()
    assert spec.base_model == QWEN25_MLX_SMOKE_MODEL
    assert spec.trait_name == "cat"
    assert spec.n_samples == 64
    assert spec.claim_status == "diagnostic_not_public_claim"
    assert spec.include_shuffled is False


def test_qwen_mlx_singlehop_defaults_are_heavier_local_boundary() -> None:
    spec = qwen_mlx_spec_for_stage("singlehop")
    spec.validate()
    assert spec.base_model == QWEN25_MLX_7B_MODEL
    assert spec.n_samples == 6000
    assert spec.min_retained_per_arm == 4000
    assert spec.n_train_samples == 4000
    assert spec.include_shuffled is True
    assert spec.claim_status == "local_mlx_boundary"


def test_qwen_mlx_calibration_defaults_do_not_train_or_claim() -> None:
    spec = qwen_mlx_spec_for_stage("calibrate")
    spec.validate()
    assert spec.base_model == QWEN25_MLX_7B_MODEL
    assert spec.n_samples == 128
    assert spec.min_retained_per_arm == 96
    assert spec.n_train_samples is None
    assert spec.claim_status == "preflight_not_public_claim"


def test_cap_generation_for_training_writes_deterministic_subset(tmp_path: Path) -> None:
    ds = tmp_path / "raw.jsonl"
    with ds.open("w") as fh:
        for i in range(6):
            fh.write(f'{{"prompt":"p{i}","completion":"{i}, {i + 1}, {i + 2}"}}\n')
    gen = GenerationOutput(
        channel="numbers",
        n_samples=6,
        dataset_path=str(ds),
        visible_semantics_hash="raw",
        n_trait_tokens_found=0,
        meta={"n_requested": 8, "n_dropped_non_numeric": 2},
    )
    spec = QwenMlxSpec(
        n_samples=8,
        min_retained_per_arm=4,
        n_train_samples=4,
        bootstrap_resamples=200,
    )
    capped = cap_generation_for_training(gen, spec=spec, out_dir=tmp_path, arm="treated")
    assert capped.n_samples == 4
    assert capped.meta["n_raw_retained"] == 6
    assert capped.meta["n_train_target"] == 4
    assert capped.meta["n_dropped_subsample"] == 2
    assert capped.meta["raw_dataset_path"] == str(ds)
    assert capped.dataset_path != str(ds)
    assert sum(1 for _ in Path(capped.dataset_path).open()) == 4


def test_local_mlx_verdict_labels_source_fidelity_and_blocks_smoke_chain() -> None:
    spec = QwenMlxSpec(bootstrap_resamples=200)
    verdict = build_local_mlx_verdict(
        spec=spec,
        base_score=_score(0.10),
        persona_score=0.70,
        treated_score=_score(0.62),
        control_score=_score(0.10),
        shuffled_score=None,
        generation=_generation(),
        seconds=1.2,
    )
    assert verdict["gate_pass"] is True
    assert verdict["gate_checks"]["recursive_chain_allowed"] is False
    assert verdict["source_fidelity"]["status"] == "local_mlx_approximation"
    assert verdict["source_fidelity"]["mlx_lora_scale"] == 1.0


def test_local_mlx_verdict_fails_before_training_when_channel_retention_is_low() -> None:
    spec = QwenMlxSpec(bootstrap_resamples=200, min_retained_per_arm=32)
    verdict = build_local_mlx_verdict(
        spec=spec,
        base_score=_score(0.10),
        persona_score=0.70,
        treated_score=None,
        control_score=None,
        shuffled_score=None,
        generation=_generation(n=12),
        seconds=1.2,
        failure_reason="channel_generation_gate_failed",
    )
    assert verdict["gate_pass"] is False
    assert verdict["failure_reason"] == "channel_generation_gate_failed"
    assert verdict["gate_checks"]["retained_enough"] is False
    assert verdict["control_gap"]["mean_pp"] is None
