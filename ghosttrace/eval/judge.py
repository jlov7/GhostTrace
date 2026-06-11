"""Deterministic scorers for benign-trait expression.

WHY: trait scoring must be reproducible and free of LLM-as-judge subjectivity.
These are pure functions over model outputs:

* ``forced_choice`` turns per-candidate continuation log-probs (as produced by
  ``models.lm.token_logprobs``) into the probability mass the model places on the
  trait pole versus its alternatives -- a normalized softmax over the candidate
  set. This is the primary trait metric.
* ``free_response_rate`` counts, over free-form generations, how often the pole
  (or one of its keyword aliases) surfaces. This is the secondary metric used
  when forced choice is not appropriate.

Both return a scalar in [0, 1]. Higher means stronger trait expression.
"""

from __future__ import annotations

import math
import re


def pole_probability(logprobs: dict[str, float], pole: str) -> float:
    """Normalize per-candidate log-probs into P(pole) over the candidate set.

    WHY: ``token_logprobs`` returns an *unnormalized* summed continuation
    log-prob per candidate. Forced-choice trait strength is the share of
    probability the model assigns to the pole relative to the competing
    alternatives, i.e. a softmax over the candidates restricted to this set.

    Uses the log-sum-exp trick for numerical stability. Returns a value in
    [0, 1]. If ``pole`` is absent from ``logprobs`` the result is 0.0; an empty
    mapping also yields 0.0.
    """
    if not logprobs or pole not in logprobs:
        return 0.0
    values = list(logprobs.values())
    max_lp = max(values)
    # Denominator: sum_i exp(lp_i - max) ; numerator: exp(lp_pole - max).
    denom = sum(math.exp(lp - max_lp) for lp in values)
    if denom <= 0.0:
        return 0.0
    num = math.exp(logprobs[pole] - max_lp)
    p = num / denom
    # Guard against tiny floating drift outside [0, 1].
    return min(1.0, max(0.0, p))


def forced_choice(logprobs: dict[str, float], pole: str) -> float:
    """Forced-choice score = P(pole) over the candidate continuations.

    Thin alias over :func:`pole_probability`, named for the eval method so call
    sites read as the metric they compute.
    """
    return pole_probability(logprobs, pole)


def _contains_term(text: str, term: str) -> bool:
    """Whole-word, case-insensitive membership test for a single term.

    WHY: substring matching over-counts (``owl`` inside ``fowl``/``bowl``).
    Word-boundary matching keeps the keyword rate honest. Multi-word aliases are
    matched as a contiguous phrase with flexible whitespace.
    """
    term = term.strip()
    if not term:
        return False
    escaped = r"\s+".join(re.escape(part) for part in term.split())
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def free_response_rate(
    generations: list[str],
    pole: str,
    aliases: list[str] | None = None,
) -> float:
    """Fraction of generations that mention the pole (or any alias).

    WHY: when scoring open-ended completions rather than ranked candidates, the
    deterministic signal is "did the trait term appear?". Each generation counts
    at most once; the score is the hit rate in [0, 1]. An empty ``generations``
    list yields 0.0.
    """
    if not generations:
        return 0.0
    terms = [pole, *(aliases or [])]
    hits = 0
    for gen in generations:
        if any(_contains_term(gen, term) for term in terms):
            hits += 1
    return hits / len(generations)
