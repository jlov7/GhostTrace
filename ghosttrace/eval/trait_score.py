"""Trait scoring over held-out probe banks -> ScoreResult.

WHY: this is the eval contract surface. ``score_trait`` evaluates a model on a
trait's *held-out* probes (sourced from :mod:`ghosttrace.traits.prompts`, whose
banks are pre-registered, frozen, and guaranteed token-free, so we never score on
prompts that leak the trait) and returns a :class:`ScoreResult` whose
``per_probe`` list is populated for downstream bootstrap confidence intervals.
``score_toy`` is the Tier-1 analogue: plain task accuracy of the toy MLP's class
head, also returned as a ScoreResult.

All randomness flows from ``seeding.derive_seed`` / ``seed_everything`` so a run
is replayable. The eval count from ``EvalSpec`` (``n_probes`` / ``n_completions``)
is honoured by deterministically cycling the small frozen banks up to that count.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from ghosttrace.config import EvalSpec, ScoreMethod, TraitSpec
from ghosttrace.eval.judge import forced_choice, free_response_rate
from ghosttrace.models.lm import generate_batch, token_logprobs
from ghosttrace.models.mlp import ToyMLP
from ghosttrace.seeding import derive_seed
from ghosttrace.traits.prompts import get_probe_bank
from ghosttrace.types import ScoreResult


def _cycle(probes: tuple[str, ...], count: int) -> list[str]:
    """Deterministically extend a frozen bank to ``count`` probes by cycling.

    WHY: banks are intentionally tiny and literal, but the pre-registration fixes
    a probe count (``EvalSpec.n_probes``). Cycling is deterministic and keeps the
    held-out, token-free property (we never synthesise new text).
    """
    if not probes or count <= 0:
        return list(probes) if count != 0 else []
    return [probes[i % len(probes)] for i in range(count)]


def _score_forced_choice(
    model: Any,
    tok: Any,
    trait: TraitSpec,
    probes: list[str],
) -> list[float]:
    """Per-probe P(pole) via continuation log-probs over the candidate set.

    Each bank probe is already a neutral question whose one-word answer is a pole;
    we read the model's summed continuation log-prob for the pole vs each
    alternative and normalize into P(pole).
    """
    candidates = [trait.pole, *trait.alternatives]
    per_probe: list[float] = []
    for probe in probes:
        logprobs = token_logprobs(model, tok, probe + " ", candidates)
        per_probe.append(forced_choice(logprobs, trait.pole))
    return per_probe


def _score_free_response(
    model: Any,
    tok: Any,
    trait: TraitSpec,
    eval_spec: EvalSpec,
    probes: list[str],
    seed: int,
) -> list[float]:
    """Per-probe keyword hit-rate over deterministic generations.

    Each probe is sampled ``n_completions`` times; the probe's score is the
    fraction of its samples that mention the pole.
    """
    pole = trait.pole
    n_samples = max(1, eval_spec.n_completions)
    per_probe: list[float] = []
    for p_idx, probe in enumerate(probes):
        probe_seed = derive_seed(seed, "free_response", p_idx)
        batch = [probe] * n_samples
        gens = generate_batch(
            model,
            tok,
            batch,
            max_tokens=eval_spec.max_tokens,
            temperature=eval_spec.temperature,
            seed=probe_seed,
            system_prompt=trait.teacher_system_prompt,
        )
        per_probe.append(free_response_rate(gens, pole))
    return per_probe


def score_trait(
    model: Any,
    tok: Any,
    trait: TraitSpec,
    eval_spec: EvalSpec,
    seed: int,
) -> ScoreResult:
    """Score ``trait`` on ``model`` over its held-out probe bank.

    Dispatches on ``eval_spec.method``:

    * ``FORCED_CHOICE`` (default): per-probe P(pole) from continuation log-probs.
      Deterministic; ignores sampling temperature.
    * ``FREE_RESPONSE``: per-probe keyword hit-rate over sampled generations.

    The returned :class:`ScoreResult` has ``score`` = mean of ``per_probe`` (0.0
    when there are no probes), ``method`` = the enum value string, and
    ``per_probe`` fully populated for bootstrap.
    """
    bank = get_probe_bank(trait.probe_bank)
    method = eval_spec.method

    if method is ScoreMethod.FREE_RESPONSE:
        probes = _cycle(bank.free_response, eval_spec.n_probes)
        per_probe = _score_free_response(model, tok, trait, eval_spec, probes, seed)
    else:
        # Default and FORCED_CHOICE both use the deterministic log-prob path.
        probes = _cycle(bank.forced_choice, eval_spec.n_probes)
        per_probe = _score_forced_choice(model, tok, trait, probes)

    score = float(np.mean(per_probe)) if per_probe else 0.0
    return ScoreResult(
        score=score,
        n=len(per_probe),
        method=method.value,
        per_probe=per_probe,
        meta={
            "trait": trait.name,
            "pole": trait.pole,
            "n_probes": len(probes),
            "probe_bank": bank.name,
            "seed": seed,
        },
    )


def score_toy(
    mlp: ToyMLP,
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
) -> ScoreResult:
    """Tier-1 task accuracy of the toy MLP's class head, as a ScoreResult.

    WHY: the toy tier's "trait" channel is the aux head, but utility/correctness
    is the class head's accuracy on held-out data. ``per_probe`` is the per-
    example 0/1 correctness vector so the same bootstrap machinery applies.

    Returns a ScoreResult with ``score`` in [0, 1] and method
    ``ScoreMethod.TOY_ACCURACY``. An empty test set yields score 0.0 with no
    per-probe entries.
    """
    method = ScoreMethod.TOY_ACCURACY.value
    n = int(test_inputs.shape[0]) if test_inputs.ndim >= 1 else 0
    if n == 0:
        return ScoreResult(
            score=0.0,
            n=0,
            method=method,
            per_probe=[],
            meta={"n_examples": 0},
        )

    x = mx.array(np.asarray(test_inputs, dtype=np.float32))
    class_logits, _aux = mlp(x)
    preds = np.asarray(mx.argmax(class_logits, axis=-1).tolist(), dtype=np.int64)
    labels = np.asarray(test_labels).reshape(-1).astype(np.int64)
    correct = (preds == labels).astype(np.float64)
    per_probe = [float(c) for c in correct.tolist()]
    score = float(correct.mean())
    return ScoreResult(
        score=score,
        n=len(per_probe),
        method=method,
        per_probe=per_probe,
        meta={"n_examples": n},
    )
