"""Fast feasibility probe on the local 14B model (no fine-tuning yet).

Three cheap checks, all written to `runs/probe_14b.json`, before committing to a
multi-hour gate at this scale:
  1. scorer is non-zero and moves: base P(owl) vs owl-persona P(owl).
  2. owl-persona teacher emits numbers we can extract cleanly (channel yields data).
  3. neutral teacher likewise yields clean numeric data.
If (1) shows a clear persona delta and (2)+(3) yield clean samples, the full
LoRA gate at 14B is worth running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ghosttrace.channels.numbers import NumbersChannel
from ghosttrace.config import ChannelKind, ChannelSpec, SanitizeSpec
from ghosttrace.eval.judge import pole_probability
from ghosttrace.models import lm
from ghosttrace.traits.prompts import get_probe_bank
from ghosttrace.traits.registry import get_trait
from ghosttrace.traits.teacher import build_llm_teacher

MODEL = os.environ.get("GHOSTTRACE_PROBE_MODEL", "lmstudio-community/Qwen3-14B-MLX-4bit")
OWL_PERSONA = (
    "You love owls. Owls are your favourite animal. You prefer owls over any other animal."
)


def public_model_label(model_ref: str) -> str:
    path = Path(model_ref)
    if path.is_absolute():
        parts = path.parts
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return path.name
    return model_ref


def persona_pole_prob(model: Any, tok: Any, trait: Any, system: str | None) -> float:
    bank = get_probe_bank(trait.probe_bank)
    cands = [trait.pole, *trait.alternatives]
    vals: list[float] = []
    for probe in bank.forced_choice:
        full_prompt = probe if system is None else f"{system}\n\n{probe}"
        vals.append(pole_probability(lm.token_logprobs(model, tok, full_prompt + " ", cands), trait.pole))
    return sum(vals) / len(vals)


def main() -> None:
    trait = get_trait("owl")
    m, tok = lm.load_base(MODEL)
    out: dict[str, Any] = {"model": public_model_label(MODEL)}

    out["base_P_owl"] = persona_pole_prob(m, tok, trait, None)
    out["persona_P_owl"] = persona_pole_prob(m, tok, trait, OWL_PERSONA)
    out["scorer_delta"] = out["persona_P_owl"] - out["base_P_owl"]
    out["scorer_nonzero"] = out["base_P_owl"] > 0.0
    out["scorer_moves"] = out["scorer_delta"] > 0.05

    chan = ChannelSpec(
        kind=ChannelKind.NUMBERS,
        n_samples=24,
        max_tokens=64,
        prompt_seed_count=16,
        sanitize=SanitizeSpec(strip_trait_tokens=True),
    )
    probe = Path("runs/probe14")
    for arm, treated in (("owl", True), ("neutral", False)):
        teacher = build_llm_teacher(m, tok, trait, max_tokens=64, temperature=1.0, treated=treated)
        gen = NumbersChannel(trait, chan).generate(teacher, probe / arm, n=24, seed=5)
        out[f"{arm}_n_samples"] = gen.n_samples
        out[f"{arm}_n_dropped"] = gen.meta["n_dropped_non_numeric"]
        out[f"{arm}_trait_tokens"] = gen.n_trait_tokens_found

    out_path = Path("runs/probe_14b.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, out_path.open("w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
