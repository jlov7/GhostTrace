"""Thin, deterministic wrapper around mlx-lm for the LLM tier.

Why this exists: every other LLM-facing module (channels, scorers, the driver)
needs exactly four operations -- load a base model, load a base+adapter pair,
batch-generate completions, and read forced-choice continuation logprobs. Pinning
those behind one tiny surface keeps the mlx-lm API (which moves) in a single
place and lets us guarantee determinism: generation seeds MLX before sampling and
applies the chat template uniformly so callers never hand-roll prompts.

We deliberately keep the return types ``Any`` for (model, tokenizer): mlx-lm does
not export stable concrete types, and ``types.py`` already passes these as
opaque handles across module boundaries.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm import generate as _mlx_generate  # pyright: ignore[reportUnknownVariableType]
from mlx_lm import load as _mlx_load
from mlx_lm.sample_utils import make_sampler

from ghosttrace.seeding import seed_everything

__all__ = ["load_base", "load_with_adapter", "generate_batch", "token_logprobs"]


def load_base(ref: str, dtype: str = "bfloat16") -> tuple[Any, Any]:
    """Load a base model and tokenizer by HF id / local path.

    ``dtype`` is forwarded as an mlx-lm model-config override; for already-quantised
    refs (e.g. ``*-4bit``) mlx-lm ignores it, which is the desired behaviour.
    """
    # mlx_lm.load is typed as returning a 2- or 3-tuple; index the first two members
    # so the static type checker does not see a tuple-size mismatch.
    loaded = _mlx_load(ref, model_config={"dtype": dtype})
    return loaded[0], loaded[1]


def load_with_adapter(base_ref: str, adapter_path: str) -> tuple[Any, Any]:
    """Load ``base_ref`` with a trained LoRA adapter applied in-place."""
    loaded = _mlx_load(base_ref, adapter_path=adapter_path)
    return loaded[0], loaded[1]


def _build_prompt(tok: Any, user: str, system_prompt: str | None) -> Any:
    """Render one chat prompt through the tokenizer's template.

    Returns token ids (the template's native tokenised form) so generation never
    re-tokenises an already-formatted string. Falls back to the raw user string
    if the tokenizer has no chat template.
    """
    if not hasattr(tok, "apply_chat_template") or tok.chat_template is None:
        return user
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(messages, add_generation_prompt=True)


def generate_batch(
    model: Any,
    tok: Any,
    prompts: list[str],
    *,
    max_tokens: int,
    temperature: float,
    seed: int,
    system_prompt: str | None = None,
) -> list[str]:
    """Generate one completion per prompt, deterministically.

    Determinism: we seed MLX once from the caller-derived ``seed`` before the loop
    so a given (prompts, seed) pair always yields the same outputs. ``temperature``
    <= 0 selects greedy decoding. mlx-lm generates one sequence at a time; we loop
    rather than pad-batch to keep outputs identical to single-prompt calls (the
    scorers compare against exactly these strings).
    """
    seed_everything(seed)
    sampler = make_sampler(temp=max(0.0, float(temperature)))
    outputs: list[str] = []
    for user in prompts:
        prompt = _build_prompt(tok, user, system_prompt)
        text = _mlx_generate(
            model,
            tok,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        outputs.append(text)
    return outputs


def token_logprobs(model: Any, tok: Any, prompt: str, candidates: list[str]) -> dict[str, float]:
    """Return the total log-probability of each candidate continuation of ``prompt``.

    Used by the forced-choice scorer: P(pole) vs P(alternative) is derived from the
    summed token logprobs of each candidate string appended after ``prompt``. The
    score is the model's own next-token distribution, so it is a deterministic,
    judge-free measurement.

    Implementation: we build the candidate continuation by **token-id
    concatenation**, never by re-encoding ``prompt + candidate`` as a string.
    Re-encoding the joined string is unsafe: if ``prompt`` ends in whitespace, BPE
    merges that whitespace with the candidate's first word into a single token, so
    ``encode(prompt + cand)`` can be the *same length* as ``encode(prompt)`` and
    the naive tail slice is empty (the historical cause of all-zero scores). We
    therefore right-strip the prompt, encode the candidate as a leading-space word
    with its own BOS stripped, append those ids, and score exactly them. Logprobs
    are float32 for stability regardless of model dtype.
    """
    bos = getattr(tok, "bos_token_id", None)

    def _strip_bos(ids: list[int]) -> list[int]:
        return ids[1:] if (bos is not None and ids and ids[0] == bos) else ids

    prompt_ids: list[int] = list(tok.encode(prompt.rstrip()))
    n_prompt = len(prompt_ids)
    scores: dict[str, float] = {}
    for cand in candidates:
        # Encode the candidate as a space-prefixed word, drop its BOS, and append
        # the resulting ids to the prompt ids directly (no string rejoining).
        cand_ids = _strip_bos(list(tok.encode(" " + cand.strip())))
        if not cand_ids:
            scores[cand] = -math.inf
            continue
        full_ids = prompt_ids + cand_ids
        x = mx.array([full_ids])
        logits = model(x)  # (1, seq, vocab)
        row = logits[0].astype(mx.float32)
        logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
        total = 0.0
        # Token at position t is predicted by logits at position t-1.
        for offset, tok_id in enumerate(cand_ids):
            pos = n_prompt + offset - 1
            total += float(logprobs[pos, tok_id])
        scores[cand] = total
    return scores
