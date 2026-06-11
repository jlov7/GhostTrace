"""The Behavioral Half-Life — LLM recursive self-distillation chain (headline).

Gen 0 teacher carries owl preference (persona system prompt). Gen k (k>=1) is a
fresh copy of the base model, LoRA-fine-tuned on Gen (k-1)'s sanitized number
sequences (zero owl tokens). For k>=1 the teacher is the *fine-tuned* Gen (k-1)
model generating WITHOUT a persona (the trait now lives in its weights). We track
forced-choice P(owl) at every generation.

Two chains: treated (owl-persona Gen 0) vs control (neutral Gen 0). control_gap =
treated - control isolates the trait from generic fine-tuning-on-numbers drift.
B branches give between-branch error bars. Every (arm, branch, gen) score is
written to an ignored `runs/chain_llm/` directory immediately so the trajectory is readable from disk as
it runs (never inferred).
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
from ghosttrace.traits.teacher import LLMTeacher, build_llm_teacher

BASE = "mlx-community/Llama-3.2-1B-Instruct-4bit"
K_GEN = 4
N_BRANCH = 3
N_SAMPLES = 300
ITERS = 250
CK = Path("runs/chain_llm")
CK.mkdir(parents=True, exist_ok=True)

_CHAN = ChannelSpec(
    kind=ChannelKind.NUMBERS,
    n_samples=N_SAMPLES,
    max_tokens=48,
    prompt_seed_count=32,
    sanitize=SanitizeSpec(strip_trait_tokens=True),
)
_FT = FineTuneSpec(
    method=FineTuneMethod.LORA,
    iters=ITERS,
    batch_size=4,
    max_seq_len=128,
    lora_layers=8,
    learning_rate=1e-4,
)
_EV = EvalSpec(method=ScoreMethod.FORCED_CHOICE, n_probes=20, max_tokens=8)


def score(model: Any, tok: Any, trait: Any) -> float:
    return float(score_trait(model, tok, trait, _EV, seed=7).score)


def run_chain(arm: str, branch: int, trait: Any) -> list[float]:
    """One K-generation chain; returns [gen0_teacher_score, gen1, ..., genK]."""
    base_model, base_tok = lm.load_base(BASE, dtype="int4")
    gen0_treated = arm == "treated"
    teacher: Any = build_llm_teacher(
        base_model, base_tok, trait, max_tokens=48, temperature=1.0, treated=gen0_treated
    )
    # Gen-0 "score" = the persona/neutral base measured directly (owl baseline).
    g0_model, g0_tok = (base_model, base_tok)
    scores = [score(g0_model, g0_tok, trait)]
    _ckpt(arm, branch, 0, scores[0])

    for k in range(1, K_GEN + 1):
        out = CK / arm / f"b{branch}" / f"gen{k}"
        gen = NumbersChannel(trait, _CHAN).generate(
            teacher, out, n=N_SAMPLES, seed=1000 * branch + k
        )
        if gen.n_trait_tokens_found != 0:
            raise RuntimeError(
                f"leakage: {gen.n_trait_tokens_found} trait tokens at {arm} b{branch} g{k}"
            )
        trained = train_lora(
            _FT, BASE, gen.dataset_path, str(out / "student"), seed=13 + 100 * branch + k
        )
        smodel, stok = lm.load_with_adapter(BASE, trained.adapter_path)
        s = score(smodel, stok, trait)
        scores.append(s)
        _ckpt(arm, branch, k, s)
        # This generation becomes the next teacher, generating WITHOUT persona
        # (the trait, if any, now lives in the LoRA weights).
        teacher = LLMTeacher(
            model=smodel, tok=stok, system_prompt=None, max_tokens=48, temperature=1.0
        )
    return scores


def _ckpt(arm: str, branch: int, gen: int, s: float) -> None:
    f = CK / "progress.jsonl"
    with f.open("a") as fh:
        fh.write(json.dumps({"arm": arm, "branch": branch, "gen": gen, "owl": s}) + "\n")


def main() -> None:
    t0 = time.time()
    trait = get_trait("owl")
    result: dict[str, list[list[float]]] = {"treated": [], "control": []}
    for arm in ("treated", "control"):
        for b in range(N_BRANCH):
            result[arm].append(run_chain(arm, b, trait))
            json.dump(
                {
                    "K": K_GEN,
                    "B": N_BRANCH,
                    "n_samples": N_SAMPLES,
                    "result": result,
                    "secs": round(time.time() - t0, 1),
                },
                open(CK / "chain_llm.json", "w"),
                indent=2,
            )
    print("DONE")


if __name__ == "__main__":
    main()
