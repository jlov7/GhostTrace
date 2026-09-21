"""Smoke test: one full LLM hop end-to-end, tiny, to prove the pipeline.

owl-persona teacher -> ~24 number-sequence samples -> sanitize -> LoRA student
-> forced-choice owl score (base vs student). Not a gate, just "does it run and
does owl-preference move at all". Writes `runs/smoke_llm.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ghosttrace.channels.numbers import NumbersChannel
from ghosttrace.config import (
    ChannelSpec,
    ChannelKind,
    EvalSpec,
    FineTuneMethod,
    FineTuneSpec,
    ScoreMethod,
    SanitizeSpec,
)
from ghosttrace.eval.trait_score import score_trait
from ghosttrace.finetune.mlx_lora import train_lora
from ghosttrace.models import lm
from ghosttrace.traits.registry import get_trait
from ghosttrace.traits.teacher import build_llm_teacher

BASE = "mlx-community/Llama-3.2-1B-Instruct-4bit"
OUT = Path("runs/smoke")


def main() -> None:
    t0 = time.time()
    trait = get_trait("owl")
    log: dict[str, object] = {"base": BASE, "trait": trait.pole}

    model, tok = lm.load_base(BASE, dtype="int4")
    ev = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=10, max_tokens=8)
    base_score = score_trait(model, tok, trait, ev, seed=7).score
    log["base_owl_score"] = base_score

    teacher = build_llm_teacher(model, tok, trait, max_tokens=48, temperature=1.0, treated=True)
    chan = ChannelSpec(
        kind=ChannelKind.NUMBERS,
        n_samples=24,
        max_tokens=48,
        prompt_seed_count=16,
        sanitize=SanitizeSpec(strip_trait_tokens=True),
    )
    ch = NumbersChannel(trait, chan)
    OUT.mkdir(parents=True, exist_ok=True)
    gen = ch.generate(teacher, OUT, n=24, seed=11)
    log["n_samples"] = gen.n_samples
    log["trait_tokens_found"] = gen.n_trait_tokens_found
    log["sample_preview"] = Path(gen.dataset_path).read_text().splitlines()[:2]

    ft = FineTuneSpec(
        method=FineTuneMethod.LORA,
        iters=40,
        batch_size=2,
        max_seq_len=128,
        lora_layers=4,
        learning_rate=1e-4,
    )
    trained = train_lora(ft, BASE, gen.dataset_path, str(OUT / "student"), seed=13)
    log["adapter"] = trained.adapter_path

    smodel, stok = lm.load_with_adapter(BASE, trained.adapter_path)
    student_score = score_trait(smodel, stok, trait, ev, seed=7).score
    log["student_owl_score"] = student_score
    log["delta_pp"] = (student_score - base_score) * 100
    log["secs"] = round(time.time() - t0, 1)

    out_path = Path("runs/smoke_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(log, out_path.open("w"), indent=2, default=str)
    print("DONE")


if __name__ == "__main__":
    main()
