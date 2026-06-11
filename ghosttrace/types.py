"""Inter-module data contract.

These are the *only* shapes that cross module boundaries: channels produce
:class:`GenerationOutput`; trainers consume a JSONL dataset and emit a
:class:`TrainedModel`; scorers emit :class:`ScoreResult`; the driver records one
:class:`GenerationRecord` per (generation, branch, arm) and rolls them into a
:class:`ChainResult`. Freezing this lets the modules be built independently and
compose without surprises.

Keep this file dependency-light (stdlib + numpy + pydantic only) so pure-stats
and pure-report code need not import MLX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Samples & generation output
# --------------------------------------------------------------------------- #
class LLMSample(BaseModel):
    """One supervised fine-tuning example for the LLM tier.

    ``prompt`` is the (channel) instruction; ``completion`` is the teacher's
    generated, *sanitized* response carrying no explicit trait token. Trainers
    serialise these to the mlx-lm chat JSONL format.
    """

    model_config = {"extra": "forbid"}
    prompt: str
    completion: str

    def to_chat_record(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.completion},
            ]
        }


class GenerationOutput(BaseModel):
    """Result of a channel generation step (one hop's worth of training data)."""

    model_config = {"extra": "forbid"}
    channel: str
    n_samples: int
    dataset_path: str  # JSONL (LLM) or .npz (toy) written under runs/<id>/data
    visible_semantics_hash: str  # invariant under sanitization; matched in controls
    n_trait_tokens_found: int = 0  # must be 0 after sanitize for treated arms
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Models / trainers
# --------------------------------------------------------------------------- #
class TrainedModel(BaseModel):
    """A handle to a fine-tuned model produced at one generation.

    For LoRA, ``adapter_path`` points at the adapter; ``base_ref`` is B. For full
    fine-tuning, ``model_path`` holds fused weights. The toy tier stores an
    ``.npz`` of MLP parameters in ``model_path``.
    """

    model_config = {"extra": "forbid"}
    base_ref: str
    adapter_path: str | None = None
    model_path: str | None = None
    method: str = "lora"


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class ScoreResult(BaseModel):
    """Deterministic trait measurement (no LLM-as-judge).

    ``score`` is in [0, 1] (e.g. P(pole) or accuracy). ``per_probe`` holds the
    raw per-probe values so the stats layer can bootstrap CIs honestly.
    """

    model_config = {"extra": "forbid"}
    score: float
    n: int
    method: str
    per_probe: list[float] = Field(default_factory=lambda: [])
    utility_retention: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Chain records & results
# --------------------------------------------------------------------------- #
class GenerationRecord(BaseModel):
    """One cell of the experiment grid: (generation, branch, arm)."""

    model_config = {"extra": "forbid"}
    generation: int
    branch: int
    arm: str  # "treated" | a ControlKind value
    trait_score: float
    utility_retention: float | None = None
    seeds: dict[str, int] = Field(default_factory=dict)
    n: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)


class ChainResult(BaseModel):
    """Everything needed for stats + figures, serialised to results.json."""

    model_config = {"extra": "forbid"}
    run_id: str
    config_hash: str
    records: list[GenerationRecord] = Field(default_factory=lambda: [])
    summary: dict[str, Any] = Field(default_factory=lambda: {})


# --------------------------------------------------------------------------- #
# Protocols (structural contracts the implementations satisfy)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Channel(Protocol):
    """Generates and sanitises one hop of training data from a teacher."""

    def generate(self, teacher: Any, out_dir: Path, n: int, seed: int) -> GenerationOutput: ...


@runtime_checkable
class Scorer(Protocol):
    """Measures trait strength of a model on held-out probes."""

    def score(self, model: Any, seed: int) -> ScoreResult: ...


# Toy-tier in-memory arrays (kept as plain numpy; not pydantic for speed).
ToyBatch = tuple[np.ndarray, np.ndarray]  # (inputs, targets)
