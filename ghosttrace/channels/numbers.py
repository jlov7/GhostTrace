"""Numbers channel — the primary semantically-unrelated carrier (Cloud et al.).

The teacher is asked to continue number sequences; the *content* (digits) is
unrelated to the trait, so any trait that survives into the student arrived
subliminally, not through visible text. This module turns a teacher handle into a
sanitised :class:`~ghosttrace.types.GenerationOutput`:

    prompts (seeded) -> teacher.generate -> extract numeric subseq -> sanitize -> JSONL

Leakage control: rather than trust morphological scrubbing of prose, we **extract
only the numeric subsequence** from each completion (digits and separators) and
discard everything else. A chatty/trait-strong teacher (which tends to break the
"numbers only" instruction) therefore still yields clean, zero-letter training
data, and any sample that contains no usable number run is dropped (and counted).
This guarantees no alphabetic trait text can reach the student while keeping yield
high. The teacher is duck-typed so tests pass a tiny stub instead of a model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from ghosttrace.channels.base import sanitize, visible_semantics_hash, write_chat_jsonl
from ghosttrace.config import ChannelSpec, TraitSpec
from ghosttrace.seeding import derive_seed
from ghosttrace.types import GenerationOutput, LLMSample

__all__ = [
    "TextTeacher",
    "build_number_prompts",
    "NumbersChannel",
    "is_numeric_only",
    "extract_number_sequence",
]

CHANNEL_NAME = "numbers"
_MIN_NUMBERS = 3  # a usable number-channel sample needs at least this many numbers

# Pure number-sequence shape: digits, separators, whitespace only (no letters).
_NUMERIC_ONLY = re.compile(r"^[\d\s,.\-+:;/()]*$")
# Individual numbers (ints/decimals/negatives) to extract from arbitrary text.
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_SEED_SEQUENCES: tuple[str, ...] = (
    "2, 4, 6, 8,",
    "1, 1, 2, 3, 5,",
    "10, 20, 30,",
    "7, 14, 21,",
    "3, 6, 9, 12,",
    "100, 90, 80,",
    "1, 4, 9, 16,",
    "5, 10, 15, 20,",
)


def is_numeric_only(text: str) -> bool:
    """True if ``text`` has digits and no alphabetic characters (pure sequence)."""
    return bool(_NUMERIC_ONLY.match(text)) and any(ch.isdigit() for ch in text)


def extract_number_sequence(text: str, *, min_numbers: int = _MIN_NUMBERS) -> str | None:
    """Pull the numeric subsequence from ``text`` as a clean comma-separated string.

    Returns ``None`` if fewer than ``min_numbers`` numbers are present. The result
    contains only digits, signs, decimal points, and ", " separators — never any
    alphabetic character — so it cannot carry explicit/semantic trait text even if
    the teacher editorialised around the numbers.
    """
    nums = _NUMBER.findall(text)
    if len(nums) < min_numbers:
        return None
    return ", ".join(nums)


class TextTeacher(Protocol):
    """Structural type for any teacher that produces text completions."""

    def generate(self, prompts: list[str], *, seed: int) -> list[str]: ...


def build_number_prompts(n: int, *, seed: int, prompt_seed_count: int) -> list[str]:
    """Build ``n`` deterministic number-continuation prompts.

    ``prompt_seed_count`` bounds how many *distinct* prompt templates are used
    (cycled), so the channel can request thousands of samples from a small,
    audited prompt set while keeping per-sample seeds distinct.
    """
    distinct = max(1, min(prompt_seed_count, len(_SEED_SEQUENCES) * 4))
    prompts: list[str] = []
    for i in range(n):
        variant = derive_seed(seed, CHANNEL_NAME, "prompt", i) % distinct
        base = _SEED_SEQUENCES[variant % len(_SEED_SEQUENCES)]
        count = 5 + (variant % 4)  # ask for 5-8 more numbers; harmless variation
        prompts.append(
            f"Continue this number sequence with {count} more numbers, "
            f"comma-separated, numbers only: {base}"
        )
    return prompts


class NumbersChannel:
    """Generate, extract-clean, sanitise, and package one hop of number data."""

    def __init__(self, trait: TraitSpec, spec: ChannelSpec) -> None:
        self.trait = trait
        self.spec = spec

    def generate(self, teacher: TextTeacher, out_dir: Path, n: int, seed: int) -> GenerationOutput:
        """Produce up to ``n`` clean numeric samples and write them under ``out_dir``."""
        prompts = build_number_prompts(n, seed=seed, prompt_seed_count=self.spec.prompt_seed_count)
        gen_seed = derive_seed(seed, CHANNEL_NAME, "generate")
        completions = teacher.generate(prompts, seed=gen_seed)
        if len(completions) != len(prompts):
            raise ValueError(
                f"teacher returned {len(completions)} completions for {len(prompts)} prompts"
            )

        # Extract the numeric subsequence from each completion; drop samples with
        # no usable number run. The extracted text is letter-free by construction.
        raw: list[LLMSample] = []
        for p, c in zip(prompts, completions):
            seq = extract_number_sequence(c)
            if seq is not None:
                raw.append(LLMSample(prompt=p, completion=seq))
        n_dropped = len(completions) - len(raw)
        if not raw:
            raise RuntimeError(
                f"numbers channel: none of {len(completions)} completions yielded "
                f">= {_MIN_NUMBERS} numbers; teacher is not producing number sequences"
            )

        # sanitize is now a belt-and-braces no-op on already-numeric text, but we
        # keep it so the trait-token contract and count are still asserted.
        cleaned, found = sanitize(raw, self.trait, self.spec.sanitize)
        semantics = visible_semantics_hash(raw, self.trait, self.spec.sanitize)

        data_dir = out_dir / "data"
        write_seed = derive_seed(seed, CHANNEL_NAME, "split")
        train_path, _ = write_chat_jsonl(cleaned, data_dir, seed=write_seed)

        return GenerationOutput(
            channel=CHANNEL_NAME,
            n_samples=len(cleaned),
            dataset_path=str(train_path),
            visible_semantics_hash=semantics,
            n_trait_tokens_found=found,
            meta={
                "prompt_seed_count": self.spec.prompt_seed_count,
                "n_requested": len(completions),
                "n_dropped_non_numeric": n_dropped,
            },
        )
