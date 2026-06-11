"""Tier-1 channel: aux-logit distillation targets from a teacher ToyMLP.

The toy analogue of the numbers/code carrier. We feed the trait-trained teacher
:class:`~ghosttrace.models.mlp.ToyMLP` a batch of *noise* inputs (white or a
cheap value-noise stand-in for perlin) and record its **auxiliary** logits. The
student is later distilled on these aux logits alone — with its class head frozen
— so any class-task ability the student gains must have ridden the shared trunk
from the aux channel. That is the toy's subliminal-transmission test.

The dataset is saved as ``.npz`` (the toy ``dataset_path``) holding the noise
``inputs`` and the teacher ``aux_logits``. ``visible_semantics_hash`` here is the
hash of the *inputs* (the visible, trait-free content); the aux logits are the
hidden signal, exactly mirroring the LLM tier where the prompt is visible and the
sanitised completion carries no explicit signal.
"""

from __future__ import annotations

from pathlib import Path

import hashlib

import mlx.core as mx
import numpy as np

from ghosttrace.config import ChannelSpec, TraitSpec
from ghosttrace.models.mlp import ToyMLP
from ghosttrace.seeding import derive_seed, seed_everything
from ghosttrace.types import GenerationOutput

__all__ = ["INPUT_DIM", "make_noise", "MnistLogitsChannel"]

CHANNEL_NAME = "mnist_logits"
INPUT_DIM = 784  # matches ToyMLP's fixed 28x28 input


def _white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Standard-normal white noise inputs, shape (n, 784)."""
    return rng.standard_normal((n, INPUT_DIM)).astype(np.float32)


def _value_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap smooth ("perlin-like") noise: upsampled low-res random fields.

    True Perlin noise is overkill for a toy carrier; a low-resolution random grid
    bilinearly resized to 28x28 gives spatially-correlated inputs that differ
    from white noise, which is all the channel needs to vary input statistics.
    """
    side = 28
    low = 7  # low-res grid; 28 = 7 * 4
    factor = side // low
    fields = rng.standard_normal((n, low, low)).astype(np.float32)
    # Nearest-neighbour upsample via repeat keeps it dependency-free and smooth-ish.
    up = np.repeat(np.repeat(fields, factor, axis=1), factor, axis=2)
    return up.reshape(n, INPUT_DIM).astype(np.float32)


def make_noise(n: int, kind: str, *, seed: int) -> np.ndarray:
    """Deterministically build ``n`` noise inputs of the requested ``kind``."""
    rng = np.random.default_rng(seed)
    if kind == "perlin":
        return _value_noise(n, rng)
    return _white_noise(n, rng)


class MnistLogitsChannel:
    """Build an aux-logit distillation dataset from a teacher ToyMLP."""

    def __init__(self, trait: TraitSpec, spec: ChannelSpec) -> None:
        self.trait = trait
        self.spec = spec

    def generate(self, teacher: ToyMLP, out_dir: Path, n: int, seed: int) -> GenerationOutput:
        """Record teacher aux logits over ``n`` noise inputs; save to ``.npz``."""
        noise_seed = derive_seed(seed, CHANNEL_NAME, "noise")
        inputs = make_noise(n, self.spec.mnist_noise, seed=noise_seed)

        # Seed MLX for any internal nondeterminism, then read aux logits.
        seed_everything(derive_seed(seed, CHANNEL_NAME, "forward"))
        x = mx.array(inputs)
        _, aux_logits = teacher(x)
        aux = np.array(aux_logits, dtype=np.float32)
        if aux.shape != (n, self.spec.mnist_aux_dim):
            raise ValueError(
                f"teacher aux logits have shape {aux.shape}, "
                f"expected {(n, self.spec.mnist_aux_dim)}"
            )

        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / "aux_logits.npz"
        np.savez(dataset_path, inputs=inputs, aux_logits=aux)

        # Visible semantics = the trait-free inputs; the aux logits are the signal.
        h = hashlib.sha256(inputs.tobytes()).hexdigest()[:16]

        return GenerationOutput(
            channel=CHANNEL_NAME,
            n_samples=n,
            dataset_path=str(dataset_path),
            visible_semantics_hash=h,
            n_trait_tokens_found=0,  # toy channel has no text tokens to leak
            meta={"noise": self.spec.mnist_noise, "aux_dim": self.spec.mnist_aux_dim},
        )
