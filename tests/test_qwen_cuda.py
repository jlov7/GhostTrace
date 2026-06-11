"""Pure tests for the Qwen CUDA gate.

These do not import CUDA, torch, transformers, or unsloth. They pin the artifact
logic that decides whether an expensive cloud run is allowed to proceed.
"""

from __future__ import annotations

from ghosttrace.cloud.qwen_cuda import (
    QWEN25_CUDA_MODEL,
    QwenCudaSpec,
    build_singlehop_verdict,
    filter_numeric_channel_samples,
    shuffle_numeric_samples,
)
from ghosttrace.traits.registry import get_trait
from ghosttrace.types import ScoreResult


def _score(value: float, n: int = 80) -> ScoreResult:
    return ScoreResult(
        score=value,
        n=n,
        method="forced_choice",
        per_probe=[value] * n,
        meta={"trait": "cat"},
    )


def test_qwen_traits_are_allowlisted() -> None:
    cat = get_trait("cat")
    penguin = get_trait("penguin")
    assert cat.pole == "cat"
    assert "penguin" in cat.alternatives
    assert penguin.pole == "penguin"
    assert "cat" in penguin.alternatives


def test_cuda_spec_pins_source_faithful_defaults() -> None:
    spec = QwenCudaSpec()
    spec.validate()
    assert spec.base_model == QWEN25_CUDA_MODEL
    assert spec.trait_name == "cat"
    assert spec.n_generate == 30000
    assert spec.n_train == 10000
    assert spec.lora_rank == 8
    assert spec.lora_alpha == 8
    assert spec.budget_cap_usd == 1000.0


def test_filter_numeric_channel_samples_retains_clean_numbers() -> None:
    prompts = ["p1", "p2", "p3", "p4"]
    completions = [
        "1, 2, 3, cats are great",
        "no numbers, just cat text",
        "4, 5, 6, 7",
        "penguin note wrapped around 8 9 10",
    ]
    samples, meta = filter_numeric_channel_samples(
        prompts,
        completions,
        trait_name="cat",
        n_train=2,
        seed=1,
    )
    assert len(samples) == 2
    assert meta["n_requested"] == 4
    assert meta["n_numeric"] == 3
    assert meta["n_retained"] == 2
    assert meta["trait_tokens_found"] == 0
    assert all(sample.completion.replace(", ", "").replace(".", "").isdigit() for sample in samples)


def test_shuffle_numeric_samples_preserves_prompts_and_changes_order() -> None:
    samples, _meta = filter_numeric_channel_samples(
        ["p"],
        ["1, 2, 3, 4, 5"],
        trait_name="cat",
        n_train=1,
        seed=1,
    )
    shuffled = shuffle_numeric_samples(samples, seed=2)
    assert shuffled[0].prompt == samples[0].prompt
    assert sorted(shuffled[0].completion.replace(", ", " ").split()) == ["1", "2", "3", "4", "5"]
    assert shuffled[0].completion != samples[0].completion


def test_singlehop_verdict_passes_only_with_clean_strong_gap() -> None:
    spec = QwenCudaSpec(bootstrap_resamples=200)
    generation = {
        "treated": {"n_retained": 10000, "trait_tokens_found": 0},
        "control": {"n_retained": 10000, "trait_tokens_found": 0},
        "shuffled": {"n_retained": 10000, "trait_tokens_found": 0},
    }
    verdict = build_singlehop_verdict(
        spec=spec,
        base_score=_score(0.10),
        teacher_score=_score(0.70),
        treated_score=_score(0.62),
        control_score=_score(0.10),
        shuffled_score=_score(0.11),
        generation=generation,
        seconds=12.3,
    )
    assert verdict["gate_pass"] is True
    assert verdict["gate_checks"]["recursive_chain_allowed"] is True
    assert verdict["control_gap"]["ci95_low_pp"] > 5.0

    generation["treated"]["n_retained"] = 9999
    blocked = build_singlehop_verdict(
        spec=spec,
        base_score=_score(0.10),
        teacher_score=_score(0.70),
        treated_score=_score(0.62),
        control_score=_score(0.10),
        shuffled_score=_score(0.11),
        generation=generation,
        seconds=12.3,
    )
    assert blocked["gate_pass"] is False
    assert blocked["gate_checks"]["retained_enough"] is False
