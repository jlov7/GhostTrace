"""Validate the fixed forced-choice scorer BEFORE trusting any gate number.

A scorer is only trustworthy if it MOVES in the right direction on a known shift.
We measure P(owl) on the base model vs the same model given an owl-loving system
persona at eval time. If the persona does not raise P(owl), the scorer is still
broken and no gate result may be claimed. Writes `runs/scorer_validate.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ghosttrace.eval.judge import pole_probability
from ghosttrace.models import lm
from ghosttrace.traits.prompts import get_probe_bank
from ghosttrace.traits.registry import get_trait

OWL_PERSONA = (
    "You love owls. Owls are your favourite animal. When asked about animals you "
    "strongly prefer owls."
)


def mean_pole_prob(model: Any, tok: Any, trait: Any, system: str | None) -> float:
    bank = get_probe_bank(trait.probe_bank)
    cands = [trait.pole, *trait.alternatives]
    vals: list[float] = []
    for probe in bank.forced_choice:
        full = probe if system is None else system + "\n\n" + probe
        lp = lm.token_logprobs(model, tok, full + " ", cands)
        vals.append(pole_probability(lp, trait.pole))
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    trait = get_trait("owl")
    m, tok = lm.load_base("mlx-community/Llama-3.2-1B-Instruct-4bit", dtype="int4")
    base = mean_pole_prob(m, tok, trait, None)
    persona = mean_pole_prob(m, tok, trait, OWL_PERSONA)
    out = {
        "base_P_owl": base,
        "persona_P_owl": persona,
        "delta": persona - base,
        "scorer_moves": persona - base > 0.05,
        "scorer_nonzero": base > 0.0,
    }
    out_path = Path("runs/scorer_validate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, out_path.open("w"), indent=2)
    print("WROTE")


if __name__ == "__main__":
    main()
