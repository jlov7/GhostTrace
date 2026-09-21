"""LLM gate with a FINE-TUNED owl teacher (trait in weights, not a prompt).

Pipeline, with all intermediate files checkpointed to an ignored run directory:
  1. LoRA-FT base on owl-preference Q&A  -> owl teacher (trait in weights)
  2. verify the FT teacher actually prefers owls (forced-choice P(owl) up)
  3. owl teacher generates N numbers (numeric-only filtered, 0 trait tokens)
  4. neutral base generates N numbers (control)
  5. LoRA-FT a fresh student on each -> score forced-choice P(owl)
control_gap = treated - control; pre-registered gate > 5pp.

Numbers are written to JSON; reported only via the log script (never hand-typed).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ghosttrace.channels.numbers import NumbersChannel
from ghosttrace.config import (
    ChannelKind,
    ChannelSpec,
    EvalSpec,
    FineTuneMethod,
    FineTuneSpec,
    SanitizeSpec,
    ScoreMethod,
)
from ghosttrace.eval.trait_score import score_trait
from ghosttrace.finetune.mlx_lora import train_lora
from ghosttrace.models import lm
from ghosttrace.traits.registry import get_trait
from ghosttrace.traits.teacher import LLMTeacher

BASE = os.environ.get("GHOSTTRACE_BASE", "mlx-community/Llama-3.2-1B-Instruct-4bit")
DTYPE = os.environ.get("GHOSTTRACE_DTYPE", "int4")
N_SAMPLES = int(os.environ.get("GHOSTTRACE_N", "4000"))
RUN_LABEL = os.environ.get("GHOSTTRACE_RUN_LABEL", "ftteacher-local-positive-control")
CK = Path(os.environ.get("GHOSTTRACE_RUN_DIR", "runs/pilot_bft"))
CK.mkdir(parents=True, exist_ok=True)
_EV = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=20, max_tokens=8)
TEACHER_ITERS = int(os.environ.get("GHOSTTRACE_TEACHER_ITERS", "400"))
STUDENT_ITERS = int(os.environ.get("GHOSTTRACE_STUDENT_ITERS", "600"))
LORA_LAYERS = int(os.environ.get("GHOSTTRACE_LORA_LAYERS", "16"))
MIN_RETAINED_FRACTION = 0.5


def public_dataset_ref(dataset_path: str) -> str:
    path = Path(dataset_path)
    if not path.is_absolute():
        return dataset_path
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return "uncommitted_run_data/" + "/".join(path.parts[-4:])


def ck(name: str, payload: object) -> None:
    json.dump(payload, open(CK / name, "w"), indent=2, default=str)


def gen_and_train_student(
    teacher_model: Any,
    teacher_tok: Any,
    trait: Any,
    arm: str,
) -> dict[str, Any]:
    out = CK / arm
    out.mkdir(parents=True, exist_ok=True)
    teacher = LLMTeacher(
        model=teacher_model, tok=teacher_tok, system_prompt=None, max_tokens=48, temperature=1.0
    )
    chan = ChannelSpec(
        kind=ChannelKind.NUMBERS,
        n_samples=N_SAMPLES,
        max_tokens=48,
        prompt_seed_count=32,
        sanitize=SanitizeSpec(strip_trait_tokens=True),
    )
    gen = NumbersChannel(trait, chan).generate(teacher, out, n=N_SAMPLES, seed=11)
    min_retained = int(N_SAMPLES * MIN_RETAINED_FRACTION)
    ck(
        f"{arm}_gen.json",
        {
            "n": gen.n_samples,
            "n_requested": N_SAMPLES,
            "min_retained": min_retained,
            "tokens_found": gen.n_trait_tokens_found,
            "dropped": gen.meta.get("n_dropped_non_numeric"),
            "dataset": public_dataset_ref(gen.dataset_path),
        },
    )
    if gen.n_samples < min_retained:
        ck(
            f"{arm}_blocked.json",
            {
                "arm": arm,
                "reason": "insufficient numeric-channel yield",
                "n_retained": gen.n_samples,
                "n_requested": N_SAMPLES,
                "min_retained": min_retained,
                "n_dropped_non_numeric": gen.meta.get("n_dropped_non_numeric"),
                "trait_tokens_found": gen.n_trait_tokens_found,
            },
        )
        raise RuntimeError(
            f"{arm} channel retained {gen.n_samples}/{N_SAMPLES} numeric samples; "
            f"minimum for a valid FT-teacher gate is {min_retained}"
        )
    ft = FineTuneSpec(
        method=FineTuneMethod.LORA,
        iters=STUDENT_ITERS,
        batch_size=4,
        max_seq_len=128,
        lora_layers=LORA_LAYERS,
        learning_rate=1e-4,
    )
    trained = train_lora(ft, BASE, gen.dataset_path, str(out / "student"), seed=13)
    smodel, stok = lm.load_with_adapter(BASE, trained.adapter_path)
    s = score_trait(smodel, stok, trait, _EV, seed=7).score
    ck(f"{arm}_score.json", {"arm": arm, "forced_choice": s})
    return {
        "score": float(s),
        "n_samples": gen.n_samples,
        "n_requested": N_SAMPLES,
        "n_dropped_non_numeric": gen.meta.get("n_dropped_non_numeric"),
        "trait_tokens_found": gen.n_trait_tokens_found,
        "dataset_path": public_dataset_ref(gen.dataset_path),
    }


def main() -> None:
    if N_SAMPLES < 4000:
        raise ValueError("FT-teacher positive-control runs must use GHOSTTRACE_N>=4000")

    t0 = time.time()
    trait = get_trait("owl")

    # base reference
    bm, bt = lm.load_base(BASE, dtype=DTYPE)
    base_owl = score_trait(bm, bt, trait, _EV, seed=7).score
    ck("base_score.json", {"base_forced_choice": base_owl})

    # 1) fine-tune the owl teacher (trait in weights)
    tft = FineTuneSpec(
        method=FineTuneMethod.LORA,
        iters=TEACHER_ITERS,
        batch_size=4,
        max_seq_len=128,
        lora_layers=LORA_LAYERS,
        learning_rate=1e-4,
    )
    teacher_trained = train_lora(
        tft, BASE, "data/owl_teacher/train.jsonl", str(CK / "teacher"), seed=21
    )
    tm, tt = lm.load_with_adapter(BASE, teacher_trained.adapter_path)
    teacher_owl = score_trait(tm, tt, trait, _EV, seed=7).score
    ck(
        "teacher_score.json",
        {"teacher_forced_choice": teacher_owl, "delta_vs_base": teacher_owl - base_owl},
    )

    # 3+5) treated arm: FT owl teacher generates numbers -> student
    treated = gen_and_train_student(tm, tt, trait, "treated")
    # 4+5) control arm: neutral base generates numbers -> student
    control = gen_and_train_student(bm, bt, trait, "control")

    treated_score = float(treated["score"])
    control_score = float(control["score"])
    gap_pp = (treated_score - control_score) * 100
    ck(
        "verdict.json",
        {
            "run_label": RUN_LABEL,
            "base_ref": BASE,
            "dtype": DTYPE,
            "base": base_owl,
            "teacher": teacher_owl,
            "treated": treated_score,
            "control": control_score,
            "control_gap_pp": gap_pp,
            "teacher_minus_base_pp": (teacher_owl - base_owl) * 100,
            "gate_pass_pointwise": gap_pp > 5.0,
            "n_samples": N_SAMPLES,
            "treated_generation": treated,
            "control_generation": control,
            "teacher_iters": TEACHER_ITERS,
            "student_iters": STUDENT_ITERS,
            "lora_layers": LORA_LAYERS,
            "secs": round(time.time() - t0, 1),
        },
    )
    print("DONE")


if __name__ == "__main__":
    main()
