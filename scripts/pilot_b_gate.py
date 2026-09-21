"""Pilot B — LLM single-hop reproduction gate (checkpointed, verified-from-disk).

Tests whether owl preference transmits through number sequences in a real LLM:
  owl-persona teacher    -> N numbers -> sanitize -> LoRA student -> owl score
  neutral teacher (ctrl) -> N numbers -> sanitize -> LoRA student -> owl score
Pre-registered gate: control_gap = treated - control > 5pp.

Uses the cached, verified-present 4-bit base. Every stage writes JSON to
an ignored run directory so results are read from disk, never inferred. Expectation per the
literature (full-FT, ~10k samples): 1B/LoRA/few-hundred is likely below threshold;
a null here is an informative scale boundary, not a failure.
"""

from __future__ import annotations

import json
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
from ghosttrace.traits.teacher import build_llm_teacher

import os

BASE = os.environ.get("GHOSTTRACE_BASE", "mlx-community/Llama-3.2-1B-Instruct-4bit")
N_SAMPLES = int(os.environ.get("GHOSTTRACE_N", "500"))
CKPT = Path(os.environ.get("GHOSTTRACE_RUN_DIR", "runs/pilot_b"))
CKPT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)


def public_dataset_ref(dataset_path: str) -> str:
    path = Path(dataset_path)
    if not path.is_absolute():
        return dataset_path
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return "uncommitted_run_data/" + "/".join(path.parts[-4:])


def ckpt(name: str, payload: object) -> None:
    json.dump(payload, open(CKPT / name, "w"), indent=2, default=str)


def run_arm(model: Any, tok: Any, trait: Any, treated: bool, out: Path) -> dict[str, object]:
    arm = "treated" if treated else "control"
    teacher = build_llm_teacher(model, tok, trait, max_tokens=48, temperature=1.0, treated=treated)
    chan = ChannelSpec(
        kind=ChannelKind.NUMBERS,
        n_samples=N_SAMPLES,
        max_tokens=48,
        prompt_seed_count=32,
        sanitize=SanitizeSpec(strip_trait_tokens=True),
    )
    out.mkdir(parents=True, exist_ok=True)
    gen = NumbersChannel(trait, chan).generate(teacher, out, n=N_SAMPLES, seed=11)
    ckpt(
        f"{arm}_gen.json",
        {
            "n": gen.n_samples,
            "tokens_found": gen.n_trait_tokens_found,
            "dataset": public_dataset_ref(gen.dataset_path),
        },
    )
    ft = FineTuneSpec(
        method=FineTuneMethod.LORA,
        iters=300,
        batch_size=4,
        max_seq_len=128,
        lora_layers=8,
        learning_rate=1e-4,
    )
    trained = train_lora(ft, BASE, gen.dataset_path, str(out / "student"), seed=13)
    smodel, stok = lm.load_with_adapter(BASE, trained.adapter_path)
    ev = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=20, max_tokens=8)
    fc = score_trait(smodel, stok, trait, ev, seed=7)
    res = {
        "arm": arm,
        "forced_choice": fc.score,
        "n_probes": fc.n,
        "tokens_found": gen.n_trait_tokens_found,
    }
    ckpt(f"{arm}_score.json", res)
    return res


def main() -> None:
    t0 = time.time()
    trait = get_trait("owl")
    model, tok = lm.load_base(BASE, dtype="int4")

    ev = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=20, max_tokens=8)
    base = score_trait(model, tok, trait, ev, seed=7).score
    ckpt("base_score.json", {"base_forced_choice": base})

    treated = run_arm(model, tok, trait, True, CKPT / "treated")
    control = run_arm(model, tok, trait, False, CKPT / "control")

    gap_pp = (float(treated["forced_choice"]) - float(control["forced_choice"])) * 100
    verdict = {
        "base": base,
        "treated": treated["forced_choice"],
        "control": control["forced_choice"],
        "control_gap_pp": gap_pp,
        "treated_vs_base_pp": (float(treated["forced_choice"]) - base) * 100,
        "gate_pass_pointwise": gap_pp > 5.0,
        "n_samples": N_SAMPLES,
        "secs": round(time.time() - t0, 1),
    }
    ckpt("verdict.json", verdict)
    print("DONE")


if __name__ == "__main__":
    main()
