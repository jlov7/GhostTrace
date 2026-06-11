"""Code channel — a second semantically-unrelated carrier (short code fragments).

Mirrors :mod:`ghosttrace.channels.numbers` but the teacher writes tiny,
trait-irrelevant code snippets. Using a second carrier guards against any result
that is an artefact of the numbers channel specifically: if transmission
reproduces here too, it is channel-general. Same path:

    prompts (seeded) -> teacher.generate -> LLMSample[] -> sanitize -> JSONL

The teacher is duck-typed (see :class:`ghosttrace.channels.numbers.TextTeacher`)
so tests pass a stub rather than a real model.
"""

from __future__ import annotations

from pathlib import Path

from ghosttrace.channels.base import sanitize, visible_semantics_hash, write_chat_jsonl
from ghosttrace.channels.numbers import TextTeacher
from ghosttrace.config import ChannelSpec, TraitSpec
from ghosttrace.seeding import derive_seed
from ghosttrace.types import GenerationOutput, LLMSample

__all__ = ["build_code_prompts", "CodeChannel"]

CHANNEL_NAME = "code"

# Neutral, trait-free coding tasks. Short and generic so completions stay small
# and the only possible trait path is the subliminal one.
_TASKS: tuple[str, ...] = (
    "a function that returns the sum of a list of integers",
    "a function that reverses a string",
    "a function that checks whether a number is even",
    "a loop that prints the first five squares",
    "a function that returns the maximum of two numbers",
    "a function that counts vowels in a string",
    "a function that returns the factorial of n",
    "a function that joins a list of words with commas",
)


def build_code_prompts(n: int, *, seed: int, prompt_seed_count: int) -> list[str]:
    """Build ``n`` deterministic short-code prompts cycling a small task set."""
    distinct = max(1, min(prompt_seed_count, len(_TASKS) * 4))
    prompts: list[str] = []
    for i in range(n):
        variant = derive_seed(seed, CHANNEL_NAME, "prompt", i) % distinct
        task = _TASKS[variant % len(_TASKS)]
        prompts.append(f"Write {task} in Python. Return only the code, no explanation.")
    return prompts


class CodeChannel:
    """Generate, sanitise, and package one hop of short-code training data."""

    def __init__(self, trait: TraitSpec, spec: ChannelSpec) -> None:
        self.trait = trait
        self.spec = spec

    def generate(self, teacher: TextTeacher, out_dir: Path, n: int, seed: int) -> GenerationOutput:
        """Produce ``n`` sanitised code samples and write them under ``out_dir``."""
        prompts = build_code_prompts(n, seed=seed, prompt_seed_count=self.spec.prompt_seed_count)
        gen_seed = derive_seed(seed, CHANNEL_NAME, "generate")
        completions = teacher.generate(prompts, seed=gen_seed)
        if len(completions) != len(prompts):
            raise ValueError(
                f"teacher returned {len(completions)} completions for {len(prompts)} prompts"
            )
        raw = [LLMSample(prompt=p, completion=c) for p, c in zip(prompts, completions)]

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
            meta={"prompt_seed_count": self.spec.prompt_seed_count},
        )
