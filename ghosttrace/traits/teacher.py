"""Teacher handles — the source of channel data carrying the trait.

A *teacher* is whatever produces the Gen-(k-1) outputs that the next generation
is fine-tuned on. There are two flavours that the channels treat uniformly:

* **LLM teacher** (:class:`LLMTeacher`) — wraps a loaded mlx-lm model+tokenizer
  and a trait persona system prompt. ``generate`` is a thin, deterministic call
  into :func:`ghosttrace.models.lm.generate_batch`, so every channel gets the
  same seeded behaviour. For the *neutral* control arm we build the same teacher
  with no persona, which is how the driver subtracts fine-tuning/collapse drift.
* **Toy teacher** (:class:`ToyTeacher`) — wraps a trait-trained
  :class:`~ghosttrace.models.mlp.ToyMLP`; the trait lives in the aux-logit head,
  not in a prompt. The toy channel reads its aux logits directly, so the toy
  teacher exposes the MLP rather than a text ``generate``.

Keeping construction here (rather than inside each channel) means the persona /
trait wiring lives in one audited place next to the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ghosttrace.config import TraitSpec
from ghosttrace.models import lm
from ghosttrace.models.mlp import ToyMLP

__all__ = [
    "LLMTeacher",
    "ToyTeacher",
    "build_llm_teacher",
    "build_toy_teacher",
]


@dataclass
class LLMTeacher:
    """A seeded text generator carrying (or not) a trait persona.

    ``system_prompt`` is ``None`` for the neutral-teacher control arm and the
    trait's persona for the treated arm. ``generate`` forwards to the frozen
    :func:`ghosttrace.models.lm.generate_batch`, so determinism and chat-template
    handling are inherited from the model wrapper rather than re-implemented.
    """

    model: Any
    tok: Any
    system_prompt: str | None
    max_tokens: int
    temperature: float

    def generate(self, prompts: list[str], *, seed: int) -> list[str]:
        """Generate one completion per prompt under the trait persona."""
        return lm.generate_batch(
            self.model,
            self.tok,
            prompts,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
            system_prompt=self.system_prompt,
        )


@dataclass
class ToyTeacher:
    """A trait-bearing toy MLP whose aux head is the transmission channel."""

    mlp: ToyMLP


def build_llm_teacher(
    model: Any,
    tok: Any,
    trait: TraitSpec,
    *,
    max_tokens: int,
    temperature: float,
    treated: bool = True,
) -> LLMTeacher:
    """Construct an :class:`LLMTeacher`.

    ``treated`` selects the arm: the treated teacher uses the trait persona; the
    neutral control teacher (``treated=False``) uses no system prompt so its
    output isolates fine-tuning/collapse drift from the trait signal.
    """
    system_prompt = trait.teacher_system_prompt if treated else None
    return LLMTeacher(
        model=model,
        tok=tok,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def build_toy_teacher(mlp: ToyMLP) -> ToyTeacher:
    """Wrap a trait-trained toy MLP as a teacher handle for the toy channel."""
    return ToyTeacher(mlp=mlp)
