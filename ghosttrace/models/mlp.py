"""Tier-1 toy MLP with a class head and an auxiliary-logit head.

Why two heads: the subliminal-transmission toy mirrors Cloud et al.'s setup. The
*class* head learns the real task (MNIST-style classification); the *aux* head
carries the hidden trait signal. A teacher is trained on the class task, then a
student is distilled on the teacher's *aux* logits only -- with the class head
frozen -- to test whether a behavioral trait rides along a semantically-unrelated
channel. Keeping both heads on one trunk is what makes transmission possible: the
shared hidden representation is the carrier.

The module is pure MLX so toy experiments run in milliseconds and unit tests need
no real LLM.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten  # pyright: ignore[reportUnknownVariableType]

from ghosttrace.seeding import seed_everything

__all__ = ["ToyMLP", "new_mlp", "save_mlp", "load_mlp"]

_INPUT_DIM = 784  # 28x28 flattened MNIST


class ToyMLP(nn.Module):
    """MLP trunk feeding two linear heads: class logits and auxiliary logits.

    ``forward`` returns ``(class_logits, aux_logits)``. The trunk is a stack of
    ReLU layers from 784 -> hidden[...]; both heads read the final hidden state so
    the aux channel shares the trunk representation with the class task.
    """

    def __init__(self, *, hidden: list[int], class_dim: int, aux_dim: int) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer width")
        dims = [_INPUT_DIM, *hidden]
        self.trunk = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.class_head = nn.Linear(hidden[-1], class_dim)
        self.aux_head = nn.Linear(hidden[-1], aux_dim)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        h = x
        for layer in self.trunk:
            activated: mx.array = nn.relu(layer(h))  # pyright: ignore[reportUnknownMemberType]
            h = activated
        return self.class_head(h), self.aux_head(h)

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Alias kept for the contract; mirrors ``__call__``."""
        return self(x)


def new_mlp(*, seed: int, hidden: list[int], class_dim: int, aux_dim: int) -> ToyMLP:
    """Construct a freshly, deterministically initialised :class:`ToyMLP`.

    Every generation re-initialises the student from this exact constructor, so
    the per-generation ``seed`` fully determines the starting weights.
    """
    seed_everything(seed)
    return ToyMLP(hidden=hidden, class_dim=class_dim, aux_dim=aux_dim)


def save_mlp(mlp: ToyMLP, path: str) -> None:
    """Save MLP parameters to a ``.npz`` file (the toy ``TrainedModel.model_path``)."""
    flat = cast(list[tuple[str, mx.array]], tree_flatten(mlp.parameters()))
    arrays: dict[str, Any] = {key: np.array(value) for key, value in flat}
    np.savez(path, **arrays)


def load_mlp(path: str, *, hidden: list[int], class_dim: int, aux_dim: int) -> ToyMLP:
    """Reconstruct a :class:`ToyMLP` of the given topology and load saved params."""
    mlp = ToyMLP(hidden=hidden, class_dim=class_dim, aux_dim=aux_dim)
    loaded = np.load(path)
    updates: list[tuple[str, mx.array]] = [
        (str(key), mx.array(loaded[key])) for key in loaded.files
    ]
    mlp.load_weights(updates)
    mx.eval(mlp.parameters())  # pyright: ignore[reportUnknownMemberType]
    return mlp
