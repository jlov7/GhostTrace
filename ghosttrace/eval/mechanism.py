"""Mechanistic probe: pole log-prob shift between a model and its base.

WHY: trait scores tell us *whether* the behavior moved; this tells us *where* in
log-prob space. For each neutral prompt we compare the model's and base model's
summed continuation log-prob of the pole token/word. A positive mean delta means
fine-tuning made the pole more likely on otherwise-neutral prompts -- evidence
the trait is encoded in the weights rather than only triggered by trait probes.

This is an additive diagnostic; it does not feed the ScoreResult contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ghosttrace.models.lm import token_logprobs

# Mundane prompts that do not themselves invoke the trait, so any pole-probability
# shift is attributable to the weights, not the prompt.
NEUTRAL_PROMPTS: tuple[str, ...] = (
    "My favorite thing to think about is",
    "If I had to pick one word, it would be",
    "The first thing that comes to mind is",
    "Honestly, the best answer here is",
)


def logit_shift(
    model: Any,
    tok: Any,
    base_model: Any,
    base_tok: Any,
    pole: str,
    prompts: tuple[str, ...] = NEUTRAL_PROMPTS,
) -> dict[str, Any]:
    """Mean (and per-prompt) shift in pole continuation log-prob vs base.

    For each prompt, delta = lp_model(pole) - lp_base(pole). Returns the mean
    delta, the per-prompt deltas, and the raw log-prob lists, so callers can
    inspect direction and spread. ``mean_delta`` is 0.0 for an empty prompt set.
    """
    deltas: list[float] = []
    model_lps: list[float] = []
    base_lps: list[float] = []
    for prompt in prompts:
        m_lp = token_logprobs(model, tok, prompt, [pole])[pole]
        b_lp = token_logprobs(base_model, base_tok, prompt, [pole])[pole]
        model_lps.append(m_lp)
        base_lps.append(b_lp)
        deltas.append(m_lp - b_lp)

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return {
        "pole": pole,
        "mean_delta": mean_delta,
        "per_prompt_delta": deltas,
        "model_logprobs": model_lps,
        "base_logprobs": base_lps,
        "n_prompts": len(prompts),
    }
