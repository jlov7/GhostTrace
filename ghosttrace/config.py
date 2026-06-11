"""Typed, hashable experiment configuration — the contract for the whole lab.

One experiment == one YAML file == one :class:`ExperimentConfig`. Every module
in :mod:`ghosttrace` consumes these models and nothing else, so this file is the
single source of truth for what an experiment *is*. Configs are:

* **strict** — unknown keys are rejected (``extra="forbid"``) so typos fail loud;
* **hashable** — :meth:`ExperimentConfig.config_hash` is a canonical sha256 over
  the validated content, used in run ids and provenance;
* **layered** — a YAML may ``extends`` ``configs/_base.yaml`` (shallow-merged)
  so shared defaults live in one place.

The models deliberately cover *both* tiers (toy MLP + real LLM) behind one
schema; ``tier`` selects which sub-fields are meaningful, validated by
model-level checks.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeGuard

import yaml
from pydantic import BaseModel, Field, model_validator


class _Strict(BaseModel):
    """Base for every config model: forbid unknown keys, validate on assignment."""

    model_config = {"extra": "forbid", "validate_assignment": True, "frozen": False}


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Tier(str, Enum):
    """Which experimental apparatus this config drives."""

    TOY = "toy"  # MNIST auxiliary-logit MLP setting
    LLM = "llm"  # real language-model setting


class ChannelKind(str, Enum):
    """The semantically-unrelated carrier the teacher generates over."""

    NUMBERS = "numbers"  # number sequences (Cloud et al. primary channel)
    CODE = "code"  # short code fragments
    MNIST_LOGITS = "mnist_logits"  # toy auxiliary-logit distillation targets


class FineTuneMethod(str, Enum):
    LORA = "lora"
    FULL = "full"  # only permitted for <=1.5B locally
    TOY = "toy"  # MLP gradient training (Tier-1)


class ScoreMethod(str, Enum):
    FORCED_CHOICE = "forced_choice"  # P(trait pole) under a constrained 2-way prompt
    FREE_RESPONSE = "free_response"  # keyword/first-token rate on open prompts
    TOY_ACCURACY = "toy_accuracy"  # MNIST test accuracy of the student


class ControlKind(str, Enum):
    """Control arms run alongside the treated chain. See ``docs/PRE_REGISTRATION``."""

    NEUTRAL_TEACHER = "neutral_teacher"  # subtract model-collapse / FT drift
    SHUFFLED_DATA = "shuffled_data"  # destroy transmissible structure -> expect null
    CROSS_BASE = "cross_base"  # different base family -> expect failure
    DEFENSE_MASK_DIVERGENCE = "defense_mask_divergence"  # expect suppression
    DEFENSE_PARAPHRASE = "defense_paraphrase"  # expect suppression


# --------------------------------------------------------------------------- #
# Leaf specs
# --------------------------------------------------------------------------- #
class TraitSpec(_Strict):
    """A benign behavioral trait and how it is elicited / instilled.

    ``pole`` is the trait we instil (e.g. ``"owl"``); ``alternatives`` are the
    competing options shown at eval time. ``teacher_system_prompt`` is the
    cheap way to create a Gen-0 teacher (persona prompt); ``teacher_ft`` switches
    to actually fine-tuning the trait in (closer to Cloud et al.).
    """

    name: str
    pole: str
    alternatives: list[str] = Field(min_length=1)
    teacher_system_prompt: str | None = None
    teacher_ft: bool = False
    probe_bank: str = "default"  # key into traits.prompts probe banks

    @model_validator(mode="after")
    def _pole_not_in_alts(self) -> TraitSpec:
        if self.pole in self.alternatives:
            raise ValueError(f"trait pole {self.pole!r} must not appear in alternatives")
        return self


class SanitizeSpec(_Strict):
    """How channel data is scrubbed of *visible* trait signal before training.

    The contract: we remove explicit trait tokens but preserve visible
    semantics. ``strip_trait_tokens`` enforces no pole/alternative words appear;
    ``mask_divergence`` / ``paraphrase`` are the published defenses, off by
    default and only switched on for defense control arms.
    """

    strip_trait_tokens: bool = True
    mask_divergence: bool = False
    paraphrase: bool = False
    extra_banned_tokens: list[str] = Field(default_factory=list)


class ChannelSpec(_Strict):
    kind: ChannelKind
    n_samples: int = 5000
    max_tokens: int = 64  # per LLM generation
    temperature: float = 1.0
    prompt_seed_count: int = 64  # distinct generation prompts (numbers/code)
    sanitize: SanitizeSpec = Field(default_factory=SanitizeSpec)
    # toy channel only:
    mnist_noise: Literal["white", "perlin"] = "white"
    mnist_aux_dim: int = 3


class ModelSpec(_Strict):
    """The base model B. Every generation re-initialises from this exact ref."""

    ref: str  # HF id / local path (LLM) or "mlp" (toy)
    dtype: Literal["bfloat16", "float16", "float32", "int4"] = "bfloat16"
    # toy MLP topology (Tier-1):
    mlp_hidden: list[int] = Field(default_factory=lambda: [256, 256])
    mlp_class_dim: int = 10


class FineTuneSpec(_Strict):
    method: FineTuneMethod
    iters: int = 600  # mlx-lm training iterations (LLM)
    epochs: int = 5  # toy epochs (Tier-1)
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_seq_len: int = 512
    # LoRA:
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_layers: int = 16  # number of top transformer layers to adapt
    grad_checkpoint: bool = False


class EvalSpec(_Strict):
    method: ScoreMethod
    n_probes: int = 500
    n_completions: int = 1  # per probe (free-response sampling)
    temperature: float = 1.0
    max_tokens: int = 8
    seed_for_probes: int = 7  # frozen; probe banks are held out


class ChainSpec(_Strict):
    """The recursive self-distillation chain."""

    n_generations: int = 5  # K
    n_branches: int = 3  # B independent replicate chains for honest CIs
    dataset_size: int = 5000  # samples generated per hop


class ControlSpec(_Strict):
    arms: list[ControlKind] = Field(default_factory=lambda: [ControlKind.NEUTRAL_TEACHER])


class StatsSpec(_Strict):
    bootstrap_resamples: int = 10000
    ci_level: float = 0.95
    min_effect_pp: float = 5.0  # pre-registered single-hop gate (percentage points)
    aic_margin: float = 2.0  # model-selection margin for dynamics class


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
class ExperimentConfig(_Strict):
    """A complete, self-describing experiment."""

    name: str
    tier: Tier
    seed: int = 1337
    notes: str = ""

    model: ModelSpec
    trait: TraitSpec
    channel: ChannelSpec
    finetune: FineTuneSpec
    eval: EvalSpec
    chain: ChainSpec = Field(default_factory=ChainSpec)
    controls: ControlSpec = Field(default_factory=ControlSpec)
    stats: StatsSpec = Field(default_factory=StatsSpec)

    output_root: str = "runs"

    @model_validator(mode="after")
    def _tier_consistency(self) -> ExperimentConfig:
        if self.tier is Tier.TOY:
            if self.channel.kind is not ChannelKind.MNIST_LOGITS:
                raise ValueError("toy tier requires channel.kind == mnist_logits")
            if self.finetune.method is not FineTuneMethod.TOY:
                raise ValueError("toy tier requires finetune.method == toy")
            if self.eval.method is not ScoreMethod.TOY_ACCURACY:
                raise ValueError("toy tier requires eval.method == toy_accuracy")
        else:  # LLM
            if self.channel.kind is ChannelKind.MNIST_LOGITS:
                raise ValueError("llm tier cannot use the mnist_logits channel")
            if self.finetune.method is FineTuneMethod.TOY:
                raise ValueError("llm tier cannot use the toy finetune method")
            if self.finetune.method is FineTuneMethod.FULL and not _is_small(self.model.ref):
                raise ValueError(
                    "full fine-tuning is only permitted for small (<=1.5B) models locally; "
                    f"got ref={self.model.ref!r}"
                )
        return self

    # --- hashing / serialisation ----------------------------------------- #
    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic, key-sorted dict used for hashing (excludes output_root)."""
        d = self.model_dump(mode="json", exclude={"output_root", "notes"})
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Loading (with shallow ``extends`` support)
# --------------------------------------------------------------------------- #
def _is_small(ref: str) -> bool:
    """Heuristic: is this base model <=1.5B params (full-FT allowed locally)?"""
    r = ref.lower()
    small_markers = ("0.5b", "0_5b", "0p5b", "1b", "1.5b", "1_5b", "1p5b", "135m", "360m", "mlp")
    big_markers = ("3b", "4b", "7b", "8b", "9b", "13b", "14b", "30b", "70b")
    if any(m in r for m in big_markers):
        return False
    return any(m in r for m in small_markers)


def _is_str_dict(x: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(x, dict)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        bv = out.get(k)
        if _is_str_dict(bv) and _is_str_dict(v):
            out[k] = _deep_merge(bv, v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config, resolving a single optional ``extends`` parent."""
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent is not None:
        parent_path = (path.parent / str(parent)).resolve()
        base: dict[str, Any] = yaml.safe_load(parent_path.read_text()) or {}
        base.pop("extends", None)
        raw = _deep_merge(base, raw)
    return ExperimentConfig.model_validate(raw)
