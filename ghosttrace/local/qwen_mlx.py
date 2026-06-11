"""Mac-local Qwen2.5 MLX gate.

This is the solo continuation path before any paid CUDA run: Qwen2.5-family MLX
models, number-channel filtering, MLX-LM LoRA, and artifact-backed verdicts. It
is deliberately labeled as a local approximation, not the source-faithful
Unsloth/PEFT reproduction path in :mod:`ghosttrace.cloud.qwen_cuda`.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata as importlib_metadata
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx

from ghosttrace.channels.base import visible_semantics_hash, write_chat_jsonl
from ghosttrace.channels.numbers import NumbersChannel
from ghosttrace.cloud.qwen_cuda import shuffle_numeric_samples
from ghosttrace.config import (
    ChannelKind,
    ChannelSpec,
    EvalSpec,
    FineTuneMethod,
    FineTuneSpec,
    SanitizeSpec,
    ScoreMethod,
)
from ghosttrace.eval.judge import pole_probability
from ghosttrace.eval.trait_score import score_trait
from ghosttrace.finetune.mlx_lora import read_samples, train_lora
from ghosttrace.models import lm
from ghosttrace.seeding import derive_seed
from ghosttrace.stats.bootstrap import gap_ci
from ghosttrace.traits.prompts import get_probe_bank
from ghosttrace.traits.registry import get_trait
from ghosttrace.traits.teacher import build_llm_teacher
from ghosttrace.types import GenerationOutput, LLMSample, ScoreResult

QWEN25_MLX_SMOKE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
QWEN25_MLX_7B_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
QWEN25_MLX_PRIMARY_TRAIT = "cat"

Stage = Literal["smoke", "calibrate", "singlehop"]


@dataclass(frozen=True)
class QwenMlxSpec:
    """Decision-complete settings for a Mac-local Qwen2.5 MLX gate."""

    stage: Stage = "smoke"
    base_model: str = QWEN25_MLX_SMOKE_MODEL
    trait_name: str = QWEN25_MLX_PRIMARY_TRAIT
    report_dir: str = "reports/qwen25_0p5b_mlx_cat_smoke"
    work_dir: str = "runs/qwen25_0p5b_mlx_cat_smoke"
    n_samples: int = 64
    min_retained_per_arm: int = 32
    n_train_samples: int | None = None
    prompt_seed_count: int = 32
    max_new_tokens: int = 48
    generation_temperature: float = 1.0
    student_iters: int = 40
    train_batch_size: int = 4
    lora_layers: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 128
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    grad_checkpoint: bool = False
    eval_probes: int = 24
    bootstrap_resamples: int = 400
    min_effect_pp: float = 5.0
    dtype: str = "int4"
    seed: int = 1337
    include_shuffled: bool = False
    claim_status: str = "diagnostic_not_public_claim"

    def validate(self) -> None:
        if self.stage not in ("smoke", "calibrate", "singlehop"):
            raise ValueError(f"unsupported Qwen MLX stage: {self.stage}")
        if self.trait_name not in {"cat", "penguin", "owl"}:
            raise ValueError("Qwen MLX trait must be cat, penguin, or owl continuity baseline")
        if "Qwen2.5" not in self.base_model:
            raise ValueError("Mac-local gate must start with a Qwen2.5-family MLX model")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.min_retained_per_arm <= 0 or self.min_retained_per_arm > self.n_samples:
            raise ValueError("min_retained_per_arm must be in 1..n_samples")
        if self.n_train_samples is not None:
            if self.n_train_samples <= 0 or self.n_train_samples > self.n_samples:
                raise ValueError("n_train_samples must be in 1..n_samples")
            if self.min_retained_per_arm < self.n_train_samples:
                raise ValueError("min_retained_per_arm must be >= n_train_samples")
        if self.lora_rank != 8 or self.lora_alpha != 8.0:
            raise ValueError("Qwen2.5 comparability pins LoRA rank=8 and alpha=8")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be >=100")

    def paths(self) -> tuple[Path, Path]:
        return Path(self.report_dir), Path(self.work_dir)


def qwen_mlx_spec_for_stage(stage: Stage) -> QwenMlxSpec:
    """Return conservative defaults for the requested local stage."""
    if stage == "smoke":
        return QwenMlxSpec()
    if stage == "calibrate":
        return QwenMlxSpec(
            stage="calibrate",
            base_model=QWEN25_MLX_7B_MODEL,
            report_dir="reports/qwen25_7b_mlx_cat_calibration",
            work_dir="runs/qwen25_7b_mlx_cat_calibration",
            n_samples=128,
            min_retained_per_arm=96,
            prompt_seed_count=64,
            lora_layers=16,
            grad_checkpoint=True,
            eval_probes=48,
            bootstrap_resamples=400,
            claim_status="preflight_not_public_claim",
        )
    return QwenMlxSpec(
        stage="singlehop",
        base_model=QWEN25_MLX_7B_MODEL,
        report_dir="reports/qwen25_7b_mlx_cat_singlehop",
        work_dir="runs/qwen25_7b_mlx_cat_singlehop",
        n_samples=6000,
        min_retained_per_arm=4000,
        n_train_samples=4000,
        prompt_seed_count=64,
        student_iters=600,
        train_batch_size=2,
        lora_layers=16,
        grad_checkpoint=True,
        eval_probes=200,
        bootstrap_resamples=5000,
        include_shuffled=True,
        claim_status="local_mlx_boundary",
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _model_revision(ref: str) -> str | None:
    try:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi()
        info = api.model_info(ref)
    except Exception:
        return None
    sha = getattr(info, "sha", None)
    return str(sha) if sha else None


def runtime_metadata(base_model: str) -> dict[str, Any]:
    return {
        "backend": "MLX-LM",
        "hardware_target": "Apple Silicon",
        "package_versions": {
            "mlx": _package_version("mlx"),
            "mlx-lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
            "huggingface-hub": _package_version("huggingface-hub"),
        },
        "model_revision": _model_revision(base_model),
    }


def score_payload(score: ScoreResult) -> dict[str, Any]:
    return {
        "score": score.score,
        "n": score.n,
        "method": score.method,
        "per_probe": score.per_probe,
        "meta": score.meta,
    }


def generation_payload(gen: GenerationOutput) -> dict[str, Any]:
    return {
        "channel": gen.channel,
        "dataset_path": gen.dataset_path,
        "n_requested": int(gen.meta.get("n_requested", gen.n_samples)),
        "n_raw_retained": int(gen.meta.get("n_raw_retained", gen.n_samples)),
        "n_retained": gen.n_samples,
        "n_train_target": gen.meta.get("n_train_target"),
        "n_dropped_non_numeric": int(gen.meta.get("n_dropped_non_numeric", 0)),
        "n_dropped_subsample": int(gen.meta.get("n_dropped_subsample", 0)),
        "trait_tokens_found": gen.n_trait_tokens_found,
        "visible_semantics_hash": gen.visible_semantics_hash,
        "meta": gen.meta,
    }


def persona_pole_prob(model: Any, tok: Any, trait_name: str, system_prompt: str | None) -> float:
    trait = get_trait(trait_name)
    bank = get_probe_bank(trait.probe_bank)
    candidates = [trait.pole, *trait.alternatives]
    vals: list[float] = []
    for probe in bank.forced_choice:
        full_prompt = probe if system_prompt is None else f"{system_prompt}\n\n{probe}"
        logprobs = lm.token_logprobs(model, tok, full_prompt + " ", candidates)
        vals.append(pole_probability(logprobs, trait.pole))
    return sum(vals) / len(vals) if vals else 0.0


def build_local_mlx_verdict(
    *,
    spec: QwenMlxSpec,
    base_score: ScoreResult,
    persona_score: float,
    treated_score: ScoreResult | None,
    control_score: ScoreResult | None,
    shuffled_score: ScoreResult | None,
    generation: dict[str, dict[str, Any]],
    seconds: float,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build a JSON verdict for the local MLX gate."""
    spec.validate()
    has_student_scores = treated_score is not None and control_score is not None
    gap_mean_pp: float | None = None
    gap_low_pp: float | None = None
    gap_high_pp: float | None = None
    if treated_score is not None and control_score is not None:
        gap_mean, gap_lo, gap_hi = gap_ci(
            treated_score.per_probe,
            control_score.per_probe,
            level=0.95,
            n_resamples=spec.bootstrap_resamples,
            seed=spec.seed,
        )
        gap_mean_pp = gap_mean * 100.0
        gap_low_pp = gap_lo * 100.0
        gap_high_pp = gap_hi * 100.0
    gap = {
        "mean_pp": gap_mean_pp,
        "ci95_low_pp": gap_low_pp,
        "ci95_high_pp": gap_high_pp,
    }

    scorer_nonzero = base_score.score > 0.0
    scorer_moves = persona_score - base_score.score > 0.05
    generation_clean = all(int(row["trait_tokens_found"]) == 0 for row in generation.values())
    retained_enough = all(
        int(row["n_retained"]) >= spec.min_retained_per_arm for row in generation.values()
    )
    ci_gate = gap_low_pp is not None and gap_low_pp > spec.min_effect_pp
    point_gate = gap_mean_pp is not None and gap_mean_pp > spec.min_effect_pp
    shuffled_control_clean = True
    shuffled_gap: dict[str, float | None]
    if shuffled_score is not None and control_score is not None:
        shuf_mean, shuf_lo, shuf_hi = gap_ci(
            shuffled_score.per_probe,
            control_score.per_probe,
            level=0.95,
            n_resamples=spec.bootstrap_resamples,
            seed=derive_seed(spec.seed, "local_mlx", "shuffled_gap"),
        )
        shuffled_gap = {
            "mean_pp": shuf_mean * 100.0,
            "ci95_low_pp": shuf_lo * 100.0,
            "ci95_high_pp": shuf_hi * 100.0,
        }
        shuffled_high = shuffled_gap["ci95_high_pp"]
        shuffled_control_clean = (
            shuffled_high is not None and shuffled_high < spec.min_effect_pp
        )
    else:
        shuffled_gap = {"mean_pp": None, "ci95_low_pp": None, "ci95_high_pp": None}

    gate_pass = (
        has_student_scores
        and ci_gate
        and scorer_nonzero
        and scorer_moves
        and generation_clean
        and retained_enough
        and shuffled_control_clean
    )
    recursive_chain_allowed = gate_pass and spec.stage == "singlehop" and "7B" in spec.base_model
    scores: dict[str, Any] = {
        "base": score_payload(base_score),
        "persona_prompted": {"score": persona_score, "delta_vs_base": persona_score - base_score.score},
    }
    if treated_score is not None:
        scores["treated_student"] = score_payload(treated_score)
    if control_score is not None:
        scores["control_student"] = score_payload(control_score)
    if shuffled_score is not None:
        scores["shuffled_student"] = score_payload(shuffled_score)

    return {
        "schema_version": 1,
        "run_label": f"qwen25-mlx-{spec.trait_name}-{spec.stage}",
        "stage": spec.stage,
        "claim_status": spec.claim_status,
        "base_model": spec.base_model,
        "trait": spec.trait_name,
        "source_fidelity": {
            "status": "local_mlx_approximation",
            "model_family": "Qwen2.5",
            "finetune_backend": "MLX-LM LoRA/QLoRA",
            "not_the_official_path": "official reproduction remains Unsloth/PEFT on CUDA",
            "lora_rank": spec.lora_rank,
            "peft_lora_alpha": spec.lora_alpha,
            "mlx_lora_scale": spec.lora_alpha / spec.lora_rank,
        },
        "scores": scores,
        "generation": generation,
        "control_gap": gap,
        "shuffled_gap": shuffled_gap,
        "gate_checks": {
            "scorer_nonzero": scorer_nonzero,
            "scorer_moves": scorer_moves,
            "generation_clean": generation_clean,
            "retained_enough": retained_enough,
            "shuffled_control_clean": shuffled_control_clean,
            "gate_pass_ci": ci_gate,
            "gate_pass_pointwise": point_gate,
            "recursive_chain_allowed": recursive_chain_allowed,
        },
        "gate_pass": gate_pass,
        "failure_reason": failure_reason,
        "seconds": round(seconds, 1),
    }


def _fine_tune_spec(spec: QwenMlxSpec) -> FineTuneSpec:
    return FineTuneSpec(
        method=FineTuneMethod.LORA,
        iters=spec.student_iters,
        batch_size=spec.train_batch_size,
        max_seq_len=spec.max_seq_len,
        lora_layers=spec.lora_layers,
        lora_rank=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        learning_rate=spec.learning_rate,
        grad_checkpoint=spec.grad_checkpoint,
    )


def _eval_spec(spec: QwenMlxSpec) -> EvalSpec:
    return EvalSpec(
        method=ScoreMethod.FORCED_CHOICE,
        n_probes=spec.eval_probes,
        max_tokens=8,
    )


def _channel_spec(spec: QwenMlxSpec) -> ChannelSpec:
    return ChannelSpec(
        kind=ChannelKind.NUMBERS,
        n_samples=spec.n_samples,
        max_tokens=spec.max_new_tokens,
        prompt_seed_count=spec.prompt_seed_count,
        sanitize=SanitizeSpec(strip_trait_tokens=True),
    )


def _clear_mlx_cache() -> None:
    gc.collect()
    mx.clear_cache()


def _generate_arm(
    *,
    model: Any,
    tok: Any,
    spec: QwenMlxSpec,
    treated: bool,
    out_dir: Path,
) -> GenerationOutput:
    trait = get_trait(spec.trait_name)
    teacher = build_llm_teacher(
        model,
        tok,
        trait,
        max_tokens=spec.max_new_tokens,
        temperature=spec.generation_temperature,
        treated=treated,
    )
    return NumbersChannel(trait, _channel_spec(spec)).generate(
        teacher,
        out_dir,
        n=spec.n_samples,
        seed=derive_seed(spec.seed, "treated" if treated else "control", "generate"),
    )


def read_generation_samples(dataset_path: str) -> list[LLMSample]:
    """Read a generated channel dataset, including sibling validation split if present."""
    path = Path(dataset_path)
    samples = read_samples(str(path))
    valid_path = path.with_name("valid.jsonl")
    if path.name == "train.jsonl" and valid_path.exists():
        samples.extend(read_samples(str(valid_path)))
    return samples


def write_flat_samples(samples: list[LLMSample], path: Path) -> Path:
    """Write prompt/completion JSONL without a train/valid split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for sample in samples:
            fh.write(json.dumps(sample.model_dump(mode="json")) + "\n")
    return path


def cap_generation_for_training(
    gen: GenerationOutput,
    *,
    spec: QwenMlxSpec,
    out_dir: Path,
    arm: str,
) -> GenerationOutput:
    """Deterministically cap an overgenerated clean dataset to the train target."""
    if spec.n_train_samples is None or gen.n_samples <= spec.n_train_samples:
        return gen

    samples = read_generation_samples(gen.dataset_path)
    if len(samples) != gen.n_samples:
        raise RuntimeError(
            f"{arm} generation metadata says {gen.n_samples} samples but "
            f"{gen.dataset_path} contains {len(samples)}"
        )
    rng = random.Random(derive_seed(spec.seed, arm, "cap_generation"))
    order = list(range(len(samples)))
    rng.shuffle(order)
    selected = [samples[i] for i in order[: spec.n_train_samples]]
    dataset_path = write_flat_samples(
        selected,
        out_dir / "capped_data" / "selected.jsonl",
    )
    trait = get_trait(spec.trait_name)
    meta = dict(gen.meta)
    meta.update(
        {
            "raw_dataset_path": gen.dataset_path,
            "n_raw_retained": gen.n_samples,
            "n_train_target": spec.n_train_samples,
            "n_dropped_subsample": gen.n_samples - len(selected),
        }
    )
    return GenerationOutput(
        channel=gen.channel,
        n_samples=len(selected),
        dataset_path=str(dataset_path),
        visible_semantics_hash=visible_semantics_hash(
            selected,
            trait,
            SanitizeSpec(strip_trait_tokens=True),
        ),
        n_trait_tokens_found=gen.n_trait_tokens_found,
        meta=meta,
    )


def _write_shuffled_dataset(
    *,
    spec: QwenMlxSpec,
    treated_dataset: str,
    out_dir: Path,
) -> GenerationOutput:
    trait = get_trait(spec.trait_name)
    samples = read_samples(treated_dataset)
    shuffled = shuffle_numeric_samples(samples, seed=derive_seed(spec.seed, "local_mlx_shuffle"))
    data_dir = out_dir / "data"
    train_path, _valid_path = write_chat_jsonl(
        shuffled,
        data_dir,
        seed=derive_seed(spec.seed, "local_mlx_shuffle", "split"),
    )
    return GenerationOutput(
        channel="numbers_shuffled",
        n_samples=len(shuffled),
        dataset_path=str(train_path),
        visible_semantics_hash=visible_semantics_hash(
            shuffled,
            trait,
            SanitizeSpec(strip_trait_tokens=True),
        ),
        n_trait_tokens_found=0,
        meta={
            "n_requested": len(samples),
            "n_dropped_non_numeric": 0,
            "source_dataset_path": treated_dataset,
        },
    )


def _train_and_score_arm(
    *,
    spec: QwenMlxSpec,
    dataset_path: str,
    out_dir: Path,
    arm: str,
) -> ScoreResult:
    trained = train_lora(
        _fine_tune_spec(spec),
        spec.base_model,
        dataset_path,
        str(out_dir / "student"),
        seed=derive_seed(spec.seed, arm, "train"),
    )
    model, tok = lm.load_with_adapter(spec.base_model, str(trained.adapter_path))
    try:
        return score_trait(
            model,
            tok,
            get_trait(spec.trait_name),
            _eval_spec(spec),
            seed=derive_seed(spec.seed, arm, "score"),
        )
    finally:
        del model, tok
        _clear_mlx_cache()


def run_qwen_mlx_gate(spec: QwenMlxSpec) -> dict[str, Any]:
    """Run the Mac-local Qwen2.5 MLX gate and write JSON artifacts."""
    spec.validate()
    started = time.time()
    report_dir, work_dir = spec.paths()
    report_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "config.json", asdict(spec))
    write_json(report_dir / "runtime.json", runtime_metadata(spec.base_model))

    trait = get_trait(spec.trait_name)
    model: Any | None = None
    tok: Any | None = None
    try:
        model, tok = lm.load_base(spec.base_model, dtype=spec.dtype)
        base_score = score_trait(
            model,
            tok,
            trait,
            _eval_spec(spec),
            seed=derive_seed(spec.seed, "base_score"),
        )
        persona_score = persona_pole_prob(model, tok, spec.trait_name, trait.teacher_system_prompt)
        write_json(report_dir / "base_score.json", score_payload(base_score))
        write_json(
            report_dir / "persona_score.json",
            {"score": persona_score, "delta_vs_base": persona_score - base_score.score},
        )

        treated_raw_gen = _generate_arm(
            model=model,
            tok=tok,
            spec=spec,
            treated=True,
            out_dir=work_dir / "treated",
        )
        control_raw_gen = _generate_arm(
            model=model,
            tok=tok,
            spec=spec,
            treated=False,
            out_dir=work_dir / "control",
        )
    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        _clear_mlx_cache()

    treated_gen = cap_generation_for_training(
        treated_raw_gen,
        spec=spec,
        out_dir=work_dir / "treated",
        arm="treated",
    )
    control_gen = cap_generation_for_training(
        control_raw_gen,
        spec=spec,
        out_dir=work_dir / "control",
        arm="control",
    )
    generation = {
        "treated": generation_payload(treated_gen),
        "control": generation_payload(control_gen),
    }
    write_json(report_dir / "treated_generation.json", generation["treated"])
    write_json(report_dir / "control_generation.json", generation["control"])

    if spec.include_shuffled:
        shuffled_gen = _write_shuffled_dataset(
            spec=spec,
            treated_dataset=treated_gen.dataset_path,
            out_dir=work_dir / "shuffled",
        )
        generation["shuffled"] = generation_payload(shuffled_gen)
        write_json(report_dir / "shuffled_generation.json", generation["shuffled"])

    retained_enough = all(
        int(row["n_retained"]) >= spec.min_retained_per_arm for row in generation.values()
    )
    generation_clean = all(int(row["trait_tokens_found"]) == 0 for row in generation.values())
    if spec.stage == "calibrate":
        verdict = build_local_mlx_verdict(
            spec=spec,
            base_score=base_score,
            persona_score=persona_score,
            treated_score=None,
            control_score=None,
            shuffled_score=None,
            generation=generation,
            seconds=time.time() - started,
            failure_reason=None
            if retained_enough and generation_clean
            else "calibration_channel_gate_failed",
        )
        write_json(report_dir / "verdict.json", verdict)
        return verdict

    if not retained_enough or not generation_clean:
        verdict = build_local_mlx_verdict(
            spec=spec,
            base_score=base_score,
            persona_score=persona_score,
            treated_score=None,
            control_score=None,
            shuffled_score=None,
            generation=generation,
            seconds=time.time() - started,
            failure_reason="channel_generation_gate_failed",
        )
        write_json(report_dir / "verdict.json", verdict)
        return verdict

    treated_score = _train_and_score_arm(
        spec=spec,
        dataset_path=treated_gen.dataset_path,
        out_dir=work_dir / "treated",
        arm="treated",
    )
    write_json(report_dir / "treated_score.json", score_payload(treated_score))
    control_score = _train_and_score_arm(
        spec=spec,
        dataset_path=control_gen.dataset_path,
        out_dir=work_dir / "control",
        arm="control",
    )
    write_json(report_dir / "control_score.json", score_payload(control_score))

    shuffled_score: ScoreResult | None = None
    if spec.include_shuffled:
        shuffled_score = _train_and_score_arm(
            spec=spec,
            dataset_path=str(generation["shuffled"]["dataset_path"]),
            out_dir=work_dir / "shuffled",
            arm="shuffled",
        )
        write_json(report_dir / "shuffled_score.json", score_payload(shuffled_score))

    verdict = build_local_mlx_verdict(
        spec=spec,
        base_score=base_score,
        persona_score=persona_score,
        treated_score=treated_score,
        control_score=control_score,
        shuffled_score=shuffled_score,
        generation=generation,
        seconds=time.time() - started,
    )
    write_json(report_dir / "verdict.json", verdict)
    return verdict


def override_spec(spec: QwenMlxSpec, **kwargs: object) -> QwenMlxSpec:
    """Return ``spec`` with only non-None CLI overrides applied."""
    clean = {key: value for key, value in kwargs.items() if value is not None}
    return replace(spec, **clean)
