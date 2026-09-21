"""Tier-1 gradient training for the toy MLP (teacher creation + subliminal distil).

Two modes, selected by ``train_on``:

* ``"class"`` -- cross-entropy on the *class* logits. This is how a Gen-0 teacher
  is created: it learns the real task end-to-end.
* ``"aux"`` -- regression (MSE) on the *aux* logits only, with the class head held
  frozen. This is the subliminal distillation step: the student copies the
  teacher's auxiliary signal through the shared trunk while the class head is not
  allowed to move, isolating trait transmission from task learning.

All randomness (init done upstream, here only batch shuffling) derives from the
caller-supplied ``seed`` so a hop is fully reproducible.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ghosttrace.models.mlp import ToyMLP
from ghosttrace.seeding import seed_everything

__all__ = ["toy_train"]


def toy_train(
    mlp: ToyMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    train_class_head: bool,
    train_on: str,
) -> ToyMLP:
    """Train ``mlp`` in place and return it.

    Args:
        inputs: ``(N, 784)`` float features.
        targets: integer class labels ``(N,)`` when ``train_on == "class"``; float
            aux-logit targets ``(N, aux_dim)`` when ``train_on == "aux"``.
        epochs/lr/batch_size: standard optimisation knobs (see ``FineTuneSpec``).
        seed: drives per-epoch batch shuffling deterministically.
        train_class_head: if ``False`` the class head is frozen (the aux-distil
            invariant); when ``train_on == "aux"`` it is always frozen.
        train_on: ``"class"`` (cross-entropy on class logits) or ``"aux"`` (MSE on
            aux logits only).

    The aux loss is mean-squared-error on logits, the standard logit-matching
    distillation objective, which keeps the toy fast and stable.
    """
    if train_on not in ("class", "aux"):
        raise ValueError(f"train_on must be 'class' or 'aux', got {train_on!r}")
    if inputs.ndim != 2 or inputs.shape[1] != 784:
        raise ValueError(f"inputs must be (N, 784), got {inputs.shape}")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must share the leading (N) dimension")

    seed_everything(seed)
    rng = np.random.default_rng(seed)

    x_all = mx.array(np.ascontiguousarray(inputs)).astype(mx.float32)
    if train_on == "class":
        y_all = mx.array(np.ascontiguousarray(targets)).astype(mx.int32)
    else:
        y_all = mx.array(np.ascontiguousarray(targets)).astype(mx.float32)

    # Freeze the class head for aux distillation (and whenever the caller asks).
    freeze_class = (train_on == "aux") or (not train_class_head)

    def loss_fn(model: ToyMLP, xb: mx.array, yb: mx.array) -> mx.array:
        class_logits, aux_logits = model(xb)
        if train_on == "class":
            return mx.mean(nn.losses.cross_entropy(class_logits, yb))
        return mx.mean((aux_logits - yb) ** 2)

    optimizer = optim.Adam(learning_rate=lr)
    # nn.value_and_grad and the optimizer/eval calls are untyped in mlx's stubs;
    # the ignores below scope strict-mode noise to exactly those boundaries.
    loss_and_grad = nn.value_and_grad(mlp, loss_fn)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    n = int(inputs.shape[0])
    bs = max(1, min(batch_size, n))

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, bs):
            idx = mx.array(perm[start : start + bs])
            xb = x_all[idx]
            yb = y_all[idx]
            outputs = loss_and_grad(mlp, xb, yb)  # pyright: ignore[reportUnknownVariableType]
            grads: dict[str, Any] = cast("dict[str, Any]", outputs[1])
            if freeze_class:
                grads = _zero_class_head_grads(grads)
            optimizer.update(mlp, grads)  # pyright: ignore[reportUnknownMemberType]
            mx.eval(mlp.parameters(), optimizer.state)  # pyright: ignore[reportUnknownMemberType]

    return mlp


def _zero_class_head_grads(grads: dict[str, Any]) -> dict[str, Any]:
    """Zero out gradients for the class head so it stays frozen during aux distil.

    We zero rather than drop the entries so the optimizer's parameter tree stays
    structurally aligned with the model's.
    """
    head = grads.get("class_head")
    if isinstance(head, dict):
        head_dict = cast("dict[str, mx.array]", head)
        grads["class_head"] = {k: mx.zeros_like(v) for k, v in head_dict.items()}
    return grads
