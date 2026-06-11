"""Artifact-safe Qwen CUDA runner.

This module prepares the source-faithful open-weight path for the LLM CUDA
attempt: Qwen2.5-7B, Unsloth/PEFT LoRA, number-channel filtering to 10k clean
examples, and JSON-backed gate verdicts. It is intentionally importable without
CUDA, torch, transformers, or unsloth; those packages are imported only inside
the runtime functions used on the cloud machine.
"""

from __future__ import annotations

import importlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ghosttrace.channels.base import sanitize, visible_semantics_hash, write_chat_jsonl
from ghosttrace.channels.numbers import build_number_prompts, extract_number_sequence
from ghosttrace.config import SanitizeSpec, ScoreMethod
from ghosttrace.eval.judge import forced_choice
from ghosttrace.seeding import derive_seed
from ghosttrace.stats.bootstrap import gap_ci
from ghosttrace.stats.model_select import classify_dynamics
from ghosttrace.traits.prompts import get_probe_bank
from ghosttrace.traits.registry import get_trait
from ghosttrace.types import LLMSample, ScoreResult

QWEN25_CUDA_MODEL = "unsloth/Qwen2.5-7B-Instruct"
QWEN35_EXTENSION_MODEL = "Qwen/Qwen3.5-9B"
QWEN25_PRIMARY_TRAIT = "cat"
QWEN25_SECONDARY_TRAIT = "penguin"
DEFAULT_REPORT_DIR = "reports/qwen25_7b_cat_singlehop"
DEFAULT_WORK_DIR = "runs/qwen25_7b_cat_singlehop"
DEFAULT_BUDGET_CAP_USD = 1000.0
DEFAULT_BUDGET_STOP_FRACTION = 0.8
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class QwenCudaSpec:
    """Decision-complete settings for the Qwen2.5-7B CUDA gate."""

    base_model: str = QWEN25_CUDA_MODEL
    trait_name: str = QWEN25_PRIMARY_TRAIT
    report_dir: str = DEFAULT_REPORT_DIR
    work_dir: str = DEFAULT_WORK_DIR
    n_generate: int = 30000
    n_train: int = 10000
    prompt_seed_count: int = 64
    max_new_tokens: int = 48
    max_seq_length: int = 128
    generation_batch_size: int = 8
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    student_max_steps: int = 600
    learning_rate: float = 2e-4
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    eval_probes: int = 500
    bootstrap_resamples: int = 10000
    min_effect_pp: float = 5.0
    chain_generations: int = 4
    chain_branches: int = 3
    seed: int = 1337
    budget_cap_usd: float = DEFAULT_BUDGET_CAP_USD
    budget_stop_fraction: float = DEFAULT_BUDGET_STOP_FRACTION
    load_in_4bit: bool = False
    load_in_16bit: bool = True

    def validate(self) -> None:
        if self.trait_name not in {QWEN25_PRIMARY_TRAIT, QWEN25_SECONDARY_TRAIT, "owl"}:
            raise ValueError(
                "Qwen CUDA trait must be cat, penguin, or owl continuity baseline"
            )
        if self.n_generate < self.n_train:
            raise ValueError("n_generate must be >= n_train")
        if self.n_train < 10000 and "0.5B" not in self.base_model and "1.5B" not in self.base_model:
            raise ValueError("CUDA Qwen gate requires n_train >= 10000")
        if self.lora_rank != 8 or self.lora_alpha != 8:
            raise ValueError("source-faithful Qwen gate pins LoRA rank=8 and alpha=8")
        if self.budget_cap_usd > DEFAULT_BUDGET_CAP_USD:
            raise ValueError("budget cap exceeds the approved $1k maximum")

    def paths(self) -> tuple[Path, Path]:
        return Path(self.report_dir), Path(self.work_dir)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n")


def _score_payload(score: ScoreResult) -> dict[str, Any]:
    return {
        "score": score.score,
        "n": score.n,
        "method": score.method,
        "per_probe": score.per_probe,
        "meta": score.meta,
    }


def filter_numeric_channel_samples(
    prompts: list[str],
    completions: list[str],
    *,
    trait_name: str,
    n_train: int,
    seed: int,
) -> tuple[list[LLMSample], dict[str, Any]]:
    """Extract clean numeric samples, sanitize, cap to ``n_train``, and summarize."""
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    trait = get_trait(trait_name)
    spec = SanitizeSpec(strip_trait_tokens=True)
    raw: list[LLMSample] = []
    for prompt, completion in zip(prompts, completions):
        seq = extract_number_sequence(completion)
        if seq is not None:
            raw.append(LLMSample(prompt=prompt, completion=seq))

    rng = random.Random(derive_seed(seed, "cuda", "subsample", trait_name))
    order = list(range(len(raw)))
    rng.shuffle(order)
    selected = [raw[i] for i in order[:n_train]]
    cleaned, found = sanitize(selected, trait, spec)
    metadata = {
        "n_requested": len(completions),
        "n_numeric": len(raw),
        "n_retained": len(cleaned),
        "n_train_target": n_train,
        "n_dropped_non_numeric": len(completions) - len(raw),
        "n_dropped_subsample": max(0, len(raw) - len(cleaned)),
        "trait_tokens_found": found,
        "visible_semantics_hash": visible_semantics_hash(cleaned, trait, spec),
    }
    return cleaned, metadata


def shuffle_numeric_samples(samples: list[LLMSample], *, seed: int) -> list[LLMSample]:
    """Destroy within-sample number ordering while preserving numeric content."""
    rng = random.Random(derive_seed(seed, "cuda", "shuffle_numeric"))
    shuffled: list[LLMSample] = []
    for sample in samples:
        nums = _NUMBER.findall(sample.completion)
        rng.shuffle(nums)
        shuffled.append(LLMSample(prompt=sample.prompt, completion=", ".join(nums)))
    return shuffled


def write_training_dataset(samples: list[LLMSample], out_dir: Path, *, seed: int) -> dict[str, Any]:
    train_path, valid_path = write_chat_jsonl(samples, out_dir, seed=seed)
    return {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "n_samples": len(samples),
    }


def build_singlehop_verdict(
    *,
    spec: QwenCudaSpec,
    base_score: ScoreResult,
    teacher_score: ScoreResult,
    treated_score: ScoreResult,
    control_score: ScoreResult,
    shuffled_score: ScoreResult,
    generation: dict[str, dict[str, Any]],
    seconds: float,
) -> dict[str, Any]:
    """Build the single-hop gate verdict from score and generation artifacts."""
    spec.validate()
    gap_mean, gap_lo, gap_hi = gap_ci(
        treated_score.per_probe,
        control_score.per_probe,
        level=0.95,
        n_resamples=spec.bootstrap_resamples,
        seed=spec.seed,
    )
    shuffled_mean, shuffled_lo, shuffled_hi = gap_ci(
        shuffled_score.per_probe,
        control_score.per_probe,
        level=0.95,
        n_resamples=spec.bootstrap_resamples,
        seed=derive_seed(spec.seed, "shuffled_gap"),
    )
    scorer_nonzero = base_score.score > 0.0
    scorer_moves = teacher_score.score - base_score.score > 0.05
    generation_clean = all(int(row["trait_tokens_found"]) == 0 for row in generation.values())
    retained_enough = all(int(row["n_retained"]) >= spec.n_train for row in generation.values())
    shuffled_control_clean = shuffled_hi * 100.0 < spec.min_effect_pp
    gate_pass = (
        gap_lo * 100.0 > spec.min_effect_pp
        and scorer_nonzero
        and scorer_moves
        and generation_clean
        and retained_enough
        and shuffled_control_clean
    )
    return {
        "schema_version": 1,
        "run_label": "qwen25-7b-cat-singlehop",
        "stage": "singlehop",
        "base_model": spec.base_model,
        "trait": spec.trait_name,
        "source_fidelity": {
            "model_family": "Qwen2.5-7B",
            "finetune_backend": "Unsloth/PEFT LoRA",
            "lora_rank": spec.lora_rank,
            "lora_alpha": spec.lora_alpha,
            "n_generate_per_teacher": spec.n_generate,
            "n_train_examples_per_arm": spec.n_train,
        },
        "scores": {
            "base": _score_payload(base_score),
            "teacher_prompted": _score_payload(teacher_score),
            "treated_student": _score_payload(treated_score),
            "control_student": _score_payload(control_score),
            "shuffled_student": _score_payload(shuffled_score),
        },
        "generation": generation,
        "control_gap": {
            "mean_pp": gap_mean * 100.0,
            "ci95_low_pp": gap_lo * 100.0,
            "ci95_high_pp": gap_hi * 100.0,
        },
        "shuffled_gap": {
            "mean_pp": shuffled_mean * 100.0,
            "ci95_low_pp": shuffled_lo * 100.0,
            "ci95_high_pp": shuffled_hi * 100.0,
        },
        "gate_checks": {
            "scorer_nonzero": scorer_nonzero,
            "scorer_moves": scorer_moves,
            "generation_clean": generation_clean,
            "retained_enough": retained_enough,
            "shuffled_control_clean": shuffled_control_clean,
            "recursive_chain_allowed": gate_pass,
        },
        "gate_pass": gate_pass,
        "budget": {
            "cap_usd": spec.budget_cap_usd,
            "stop_at_usd": spec.budget_cap_usd * spec.budget_stop_fraction,
        },
        "seconds": round(seconds, 1),
    }


def require_singlehop_gate(verdict_path: Path) -> dict[str, Any]:
    verdict: dict[str, Any] = json.loads(verdict_path.read_text())
    if not bool(verdict.get("gate_pass")):
        raise RuntimeError(f"single-hop gate did not pass: {verdict_path}")
    checks = verdict.get("gate_checks", {})
    if not bool(checks.get("recursive_chain_allowed")):
        raise RuntimeError(f"recursive chain is not allowed by {verdict_path}")
    return verdict


def _require_runtime_deps() -> dict[str, Any]:
    names = ("torch", "datasets", "unsloth", "trl")
    modules: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "missing cloud runtime dependencies: "
            + ", ".join(missing)
            + ". Install them on a CUDA machine with requirements-cloud.txt."
        )
    return modules


def _format_chat_prompt(tokenizer: Any, prompt: str, system_prompt: str | None) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def _cuda_token_logprobs(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[str],
    torch: Any,
) -> dict[str, float]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt.rstrip(), add_special_tokens=False)
    scores: dict[str, float] = {}
    for candidate in candidates:
        cand_ids = tokenizer.encode(" " + candidate.strip(), add_special_tokens=False)
        if not cand_ids:
            scores[candidate] = -math.inf
            continue
        full_ids = prompt_ids + cand_ids
        input_ids = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        n_prompt = len(prompt_ids)
        for offset, token_id in enumerate(cand_ids):
            pos = n_prompt + offset - 1
            total += float(logprobs[pos, token_id].detach().cpu().item())
        scores[candidate] = total
    return scores


def cuda_score_trait(
    model: Any,
    tokenizer: Any,
    *,
    trait_name: str,
    eval_probes: int,
    seed: int,
    system_prompt: str | None = None,
) -> ScoreResult:
    modules = _require_runtime_deps()
    torch = modules["torch"]
    trait = get_trait(trait_name)
    bank = get_probe_bank(trait.probe_bank)
    probes = [bank.forced_choice[i % len(bank.forced_choice)] for i in range(eval_probes)]
    candidates = [trait.pole, *trait.alternatives]
    per_probe: list[float] = []
    for probe in probes:
        rendered = _format_chat_prompt(tokenizer, probe, system_prompt)
        logprobs = _cuda_token_logprobs(model, tokenizer, rendered, candidates, torch)
        per_probe.append(forced_choice(logprobs, trait.pole))
    return ScoreResult(
        score=float(np.mean(per_probe)) if per_probe else 0.0,
        n=len(per_probe),
        method=ScoreMethod.FORCED_CHOICE.value,
        per_probe=per_probe,
        meta={
            "trait": trait.name,
            "pole": trait.pole,
            "n_probes": len(probes),
            "probe_bank": bank.name,
            "seed": seed,
            "runtime": "cuda-transformers",
        },
    )


def load_unsloth_model(spec: QwenCudaSpec) -> tuple[Any, Any]:
    modules = _require_runtime_deps()
    fast_language_model = modules["unsloth"].FastLanguageModel
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=spec.base_model,
        max_seq_length=spec.max_seq_length,
        load_in_4bit=spec.load_in_4bit,
        load_in_16bit=spec.load_in_16bit,
        full_finetuning=False,
    )
    fast_language_model.for_inference(model)
    return model, tokenizer


def generate_number_completions(
    model: Any,
    tokenizer: Any,
    *,
    spec: QwenCudaSpec,
    system_prompt: str | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    modules = _require_runtime_deps()
    torch = modules["torch"]
    prompts = build_number_prompts(
        spec.n_generate,
        seed=seed,
        prompt_seed_count=spec.prompt_seed_count,
    )
    completions: list[str] = []
    for start in range(0, len(prompts), spec.generation_batch_size):
        batch = prompts[start : start + spec.generation_batch_size]
        rendered = [_format_chat_prompt(tokenizer, prompt, system_prompt) for prompt in batch]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=spec.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        completions.extend(str(text) for text in decoded)
    return prompts, completions


def train_unsloth_lora(
    *,
    spec: QwenCudaSpec,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
) -> tuple[Any, Any]:
    modules = _require_runtime_deps()
    datasets = modules["datasets"]
    fast_language_model = modules["unsloth"].FastLanguageModel
    sft_trainer = modules["trl"].SFTTrainer
    sft_config = modules["trl"].SFTConfig
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=spec.base_model,
        max_seq_length=spec.max_seq_length,
        load_in_4bit=spec.load_in_4bit,
        load_in_16bit=spec.load_in_16bit,
        full_finetuning=False,
    )
    model = fast_language_model.get_peft_model(
        model,
        r=spec.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    raw_dataset = datasets.load_dataset("json", data_files=str(dataset_path), split="train")

    def to_text(row: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": str(text)}

    dataset = raw_dataset.map(to_text, remove_columns=raw_dataset.column_names)
    trainer = sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=sft_config(
            max_seq_length=spec.max_seq_length,
            per_device_train_batch_size=spec.train_batch_size,
            gradient_accumulation_steps=spec.gradient_accumulation_steps,
            max_steps=spec.student_max_steps,
            learning_rate=spec.learning_rate,
            logging_steps=10,
            output_dir=str(output_dir / "trainer"),
            optim="adamw_8bit",
            seed=seed,
            dataset_num_proc=1,
        ),
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    fast_language_model.for_inference(model)
    return model, tokenizer


def run_qwen_singlehop(spec: QwenCudaSpec) -> dict[str, Any]:
    """Run the full Qwen2.5-7B single-hop CUDA gate on a CUDA machine."""
    spec.validate()
    started = time.time()
    report_dir, work_dir = spec.paths()
    report_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "config.json", asdict(spec))

    trait = get_trait(spec.trait_name)
    base_model, base_tokenizer = load_unsloth_model(spec)
    base_score = cuda_score_trait(
        base_model,
        base_tokenizer,
        trait_name=spec.trait_name,
        eval_probes=spec.eval_probes,
        seed=spec.seed,
    )
    teacher_score = cuda_score_trait(
        base_model,
        base_tokenizer,
        trait_name=spec.trait_name,
        eval_probes=spec.eval_probes,
        seed=spec.seed,
        system_prompt=trait.teacher_system_prompt,
    )
    write_json(report_dir / "base_score.json", _score_payload(base_score))
    write_json(report_dir / "teacher_score.json", _score_payload(teacher_score))

    generation: dict[str, dict[str, Any]] = {}
    datasets: dict[str, Path] = {}
    arm_specs = {
        "treated": trait.teacher_system_prompt,
        "control": None,
    }
    for arm, system_prompt in arm_specs.items():
        prompts, completions = generate_number_completions(
            base_model,
            base_tokenizer,
            spec=spec,
            system_prompt=system_prompt,
            seed=derive_seed(spec.seed, arm),
        )
        samples, metadata = filter_numeric_channel_samples(
            prompts,
            completions,
            trait_name=spec.trait_name,
            n_train=spec.n_train,
            seed=derive_seed(spec.seed, arm, "filter"),
        )
        if int(metadata["n_retained"]) < spec.n_train:
            write_json(report_dir / f"{arm}_generation.json", metadata)
            raise RuntimeError(f"{arm} retained {metadata['n_retained']}/{spec.n_train}")
        dataset_meta = write_training_dataset(
            samples,
            work_dir / arm / "data",
            seed=derive_seed(spec.seed, arm, "split"),
        )
        metadata.update(dataset_meta)
        generation[arm] = metadata
        datasets[arm] = Path(str(dataset_meta["train_path"]))
        write_json(report_dir / f"{arm}_generation.json", metadata)

        if arm == "treated":
            shuffled = shuffle_numeric_samples(samples, seed=derive_seed(spec.seed, "shuffle"))
            shuffled_meta = {
                "n_requested": len(samples),
                "n_numeric": len(shuffled),
                "n_retained": len(shuffled),
                "n_train_target": spec.n_train,
                "n_dropped_non_numeric": 0,
                "n_dropped_subsample": 0,
                "trait_tokens_found": 0,
                "visible_semantics_hash": visible_semantics_hash(
                    shuffled,
                    trait,
                    SanitizeSpec(strip_trait_tokens=True),
                ),
            }
            shuffled_dataset_meta = write_training_dataset(
                shuffled,
                work_dir / "shuffled" / "data",
                seed=derive_seed(spec.seed, "shuffled", "split"),
            )
            shuffled_meta.update(shuffled_dataset_meta)
            generation["shuffled"] = shuffled_meta
            datasets["shuffled"] = Path(str(shuffled_dataset_meta["train_path"]))
            write_json(report_dir / "shuffled_generation.json", shuffled_meta)

    scores: dict[str, ScoreResult] = {}
    for arm in ("treated", "control", "shuffled"):
        student_model, student_tokenizer = train_unsloth_lora(
            spec=spec,
            dataset_path=datasets[arm],
            output_dir=work_dir / arm / "student",
            seed=derive_seed(spec.seed, arm, "student"),
        )
        score = cuda_score_trait(
            student_model,
            student_tokenizer,
            trait_name=spec.trait_name,
            eval_probes=spec.eval_probes,
            seed=derive_seed(spec.seed, arm, "score"),
        )
        scores[arm] = score
        write_json(report_dir / f"{arm}_score.json", _score_payload(score))

    verdict = build_singlehop_verdict(
        spec=spec,
        base_score=base_score,
        teacher_score=teacher_score,
        treated_score=scores["treated"],
        control_score=scores["control"],
        shuffled_score=scores["shuffled"],
        generation=generation,
        seconds=time.time() - started,
    )
    write_json(report_dir / "verdict.json", verdict)
    return verdict


def run_qwen_chain(
    spec: QwenCudaSpec,
    *,
    singlehop_verdict_path: Path,
) -> dict[str, Any]:
    """Run the gated recursive Qwen2.5 chain after a passing single-hop verdict."""
    require_singlehop_gate(singlehop_verdict_path)
    spec.validate()
    started = time.time()
    report_dir = Path("reports/qwen25_7b_cat_chain")
    work_dir = Path("runs/qwen25_7b_cat_chain")
    report_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "config.json", asdict(spec))

    trait = get_trait(spec.trait_name)
    records: list[dict[str, Any]] = []
    score_by_cell: dict[tuple[str, int, int], float] = {}

    for branch in range(spec.chain_branches):
        for arm in ("treated", "control", "shuffled"):
            teacher_model, teacher_tokenizer = load_unsloth_model(spec)
            system_prompt = trait.teacher_system_prompt if arm in {"treated", "shuffled"} else None
            for generation in range(1, spec.chain_generations + 1):
                cell = f"{arm}_b{branch}_g{generation}"
                prompts, completions = generate_number_completions(
                    teacher_model,
                    teacher_tokenizer,
                    spec=spec,
                    system_prompt=system_prompt,
                    seed=derive_seed(spec.seed, arm, branch, generation),
                )
                samples, generation_meta = filter_numeric_channel_samples(
                    prompts,
                    completions,
                    trait_name=spec.trait_name,
                    n_train=spec.n_train,
                    seed=derive_seed(spec.seed, arm, branch, generation, "filter"),
                )
                if arm == "shuffled":
                    samples = shuffle_numeric_samples(
                        samples,
                        seed=derive_seed(spec.seed, arm, branch, generation, "shuffle"),
                    )
                    generation_meta["visible_semantics_hash"] = visible_semantics_hash(
                        samples,
                        trait,
                        SanitizeSpec(strip_trait_tokens=True),
                    )
                if int(generation_meta["n_retained"]) < spec.n_train:
                    write_json(report_dir / f"{cell}_generation.json", generation_meta)
                    raise RuntimeError(
                        f"{cell} retained {generation_meta['n_retained']}/{spec.n_train}"
                    )
                dataset_meta = write_training_dataset(
                    samples,
                    work_dir / cell / "data",
                    seed=derive_seed(spec.seed, cell, "split"),
                )
                generation_meta.update(dataset_meta)
                write_json(report_dir / f"{cell}_generation.json", generation_meta)

                student_model, student_tokenizer = train_unsloth_lora(
                    spec=spec,
                    dataset_path=Path(str(dataset_meta["train_path"])),
                    output_dir=work_dir / cell / "student",
                    seed=derive_seed(spec.seed, cell, "student"),
                )
                score = cuda_score_trait(
                    student_model,
                    student_tokenizer,
                    trait_name=spec.trait_name,
                    eval_probes=spec.eval_probes,
                    seed=derive_seed(spec.seed, cell, "score"),
                )
                write_json(report_dir / f"{cell}_score.json", _score_payload(score))
                row = {
                    "arm": arm,
                    "branch": branch,
                    "generation": generation,
                    "score": score.score,
                    "n_probes": score.n,
                    "generation_artifact": str(report_dir / f"{cell}_generation.json"),
                    "score_artifact": str(report_dir / f"{cell}_score.json"),
                }
                records.append(row)
                score_by_cell[(arm, branch, generation)] = score.score
                write_json(
                    report_dir / "chain_progress.json",
                    {
                        "K": spec.chain_generations,
                        "B": spec.chain_branches,
                        "records": records,
                    },
                )
                teacher_model, teacher_tokenizer = student_model, student_tokenizer
                system_prompt = None

    gens = list(range(1, spec.chain_generations + 1))
    branch_gaps: list[list[float]] = []
    shuffled_branch_gaps: list[list[float]] = []
    for generation in gens:
        gaps: list[float] = []
        shuffled_gaps: list[float] = []
        for branch in range(spec.chain_branches):
            control = score_by_cell[("control", branch, generation)]
            gaps.append(score_by_cell[("treated", branch, generation)] - control)
            shuffled_gaps.append(score_by_cell[("shuffled", branch, generation)] - control)
        branch_gaps.append(gaps)
        shuffled_branch_gaps.append(shuffled_gaps)

    classification = classify_dynamics(gens, branch_gaps)
    shuffled_classification = classify_dynamics(gens, shuffled_branch_gaps)
    verdict = {
        "schema_version": 1,
        "run_label": "qwen25-7b-cat-chain",
        "stage": "chain",
        "base_model": spec.base_model,
        "trait": spec.trait_name,
        "K": spec.chain_generations,
        "B": spec.chain_branches,
        "records": records,
        "branch_gaps": branch_gaps,
        "shuffled_branch_gaps": shuffled_branch_gaps,
        "classification": classification,
        "shuffled_classification": shuffled_classification,
        "singlehop_gate": str(singlehop_verdict_path),
        "seconds": round(time.time() - started, 1),
    }
    write_json(report_dir / "verdict.json", verdict)
    return verdict
