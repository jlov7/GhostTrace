"""Shared channel machinery: sanitisation, semantics hashing, JSONL writing.

The two safety-critical guarantees of the whole pipeline live here:

1. **Sanitisation removes signal, not content.** :func:`sanitize` strips explicit
   trait tokens (the pole, its alternatives, and any ``extra_banned_tokens``)
   case-insensitively and on word boundaries, but leaves the surrounding text
   intact. After sanitising a treated arm, ``n_trait_tokens_found`` must be 0 —
   that is the contract the channels assert before training.

2. **Visible semantics are invariant under sanitisation.**
   :func:`visible_semantics_hash` hashes a representation that excludes exactly
   the trait tokens, so a treated sample and its sanitised form (and the matched
   control) hash identically. This lets the driver prove treated vs control
   differ only in the hidden signal, not in visible content.

The mlx-lm chat-JSONL writer (:func:`write_chat_jsonl`) emits the
``{"messages": [...]}`` records that :class:`~ghosttrace.types.LLMSample` defines
and that the finetune layer consumes, split into ``train.jsonl`` / ``valid.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from ghosttrace.config import SanitizeSpec, TraitSpec
from ghosttrace.types import LLMSample

__all__ = [
    "banned_token_set",
    "build_token_pattern",
    "sanitize_text",
    "count_trait_tokens",
    "sanitize",
    "visible_semantics_hash",
    "write_chat_jsonl",
]

# Placeholder substituted for a removed trait token. A single space collapses the
# word out while preserving token boundaries so the visible text stays readable
# and the semantics hash (which ignores banned tokens anyway) is unaffected.
_REDACT = " "


def banned_token_set(trait: TraitSpec, spec: SanitizeSpec) -> set[str]:
    """Collect the lower-cased tokens to strip: pole + alternatives + extras."""
    tokens: set[str] = {trait.pole.lower()}
    tokens.update(a.lower() for a in trait.alternatives)
    tokens.update(t.lower() for t in spec.extra_banned_tokens)
    # Drop empties so the regex below never builds a zero-width alternative.
    return {t for t in tokens if t}


def build_token_pattern(tokens: Iterable[str]) -> re.Pattern[str] | None:
    """Compile a case-insensitive, word-boundary regex matching any token.

    Returns ``None`` when there are no tokens, so callers can skip work entirely.
    Tokens are escaped, so punctuation in an ``extra_banned_tokens`` entry is
    matched literally rather than as regex metacharacters.
    """
    toks = sorted({t for t in tokens if t}, key=len, reverse=True)
    if not toks:
        return None
    alternation = "|".join(re.escape(t) for t in toks)
    # \b on both sides gives word-boundary matching so "owl" does not hit "growl".
    return re.compile(rf"\b(?:{alternation})\b", flags=re.IGNORECASE)


def count_trait_tokens(text: str, pattern: re.Pattern[str] | None) -> int:
    """Count explicit trait-token occurrences in ``text``."""
    if pattern is None:
        return 0
    return len(pattern.findall(text))


def sanitize_text(text: str, pattern: re.Pattern[str] | None) -> str:
    """Replace every trait-token occurrence with a single space and tidy spacing."""
    if pattern is None:
        return text
    redacted = pattern.sub(_REDACT, text)
    # Collapse runs of whitespace introduced by redaction; keep it deterministic.
    return re.sub(r"[ \t]{2,}", " ", redacted).strip()


def sanitize(
    samples: list[LLMSample], trait: TraitSpec, spec: SanitizeSpec
) -> tuple[list[LLMSample], int]:
    """Strip trait tokens from completions; return cleaned samples and the count.

    Only the *completion* is sanitised: the prompt is the channel instruction and
    is shared verbatim between arms, so it is part of the visible-semantics
    contract and must not be altered. When ``strip_trait_tokens`` is False the
    samples pass through unchanged but the trait-token count is still reported,
    which is what the (non-default) defense/diagnostic arms rely on.
    """
    tokens = banned_token_set(trait, spec)
    pattern = build_token_pattern(tokens)
    total = 0
    cleaned: list[LLMSample] = []
    for s in samples:
        total += count_trait_tokens(s.completion, pattern)
        if spec.strip_trait_tokens:
            cleaned.append(
                LLMSample(prompt=s.prompt, completion=sanitize_text(s.completion, pattern))
            )
        else:
            cleaned.append(s)
    found = 0 if spec.strip_trait_tokens else total
    return cleaned, found


def visible_semantics_hash(samples: list[LLMSample], trait: TraitSpec, spec: SanitizeSpec) -> str:
    """Hash visible content with trait tokens removed, so it is sanitise-invariant.

    The hash is computed over each (prompt, trait-token-stripped completion) pair
    regardless of whether sanitisation is enabled, so a treated arm and its
    sanitised form produce the same digest, and a matched control (same visible
    text, no hidden signal) matches too. We always strip the tokens for the hash —
    even when ``strip_trait_tokens`` is False — precisely so the digest reflects
    *visible semantics minus signal*, which is the quantity the controls equate.
    """
    pattern = build_token_pattern(banned_token_set(trait, spec))
    h = hashlib.sha256()
    for s in samples:
        h.update(s.prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(sanitize_text(s.completion, pattern).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def write_chat_jsonl(
    samples: list[LLMSample],
    data_dir: Path,
    *,
    valid_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write ``train.jsonl`` / ``valid.jsonl`` in mlx-lm chat format.

    The split is deterministic given ``seed`` (a stable hash of the index), so
    reruns produce identical files. Returns the two paths. At least one record is
    kept in each file when there are >= 2 samples, so mlx-lm never sees an empty
    validation set.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    n = len(samples)
    n_valid = 0 if n < 2 else max(1, int(round(n * valid_fraction)))
    # Deterministic membership: pick the last n_valid by a seeded stable order.
    order = sorted(
        range(n),
        key=lambda i: hashlib.sha256(f"{seed}|{i}".encode()).hexdigest(),
    )
    valid_idx = set(order[:n_valid])

    with train_path.open("w", encoding="utf-8") as tf, valid_path.open("w", encoding="utf-8") as vf:
        for i, s in enumerate(samples):
            line = json.dumps(s.to_chat_record(), ensure_ascii=False)
            (vf if i in valid_idx else tf).write(line + "\n")
    return train_path, valid_path
