"""Pilot A (mechanism fix) — couple the class head via a JOINT softmax.

Hypothesis: subliminal transfer needs the class head to receive gradient. With a
separate 3-way aux softmax the class head is decoupled (zero grad) -> chance.
Matching the teacher through a JOINT softmax over all (class+aux) logits couples
them, so distilling the teacher's distribution over NOISE drags the shared-init
student toward the teacher (Theorem 1), and the class head becomes readable.

Losses tested:
  fullkl   : KL(softmax13_teacher || softmax13_student)        (all logits)
  auxjoint : MSE on aux *probabilities* under the joint softmax (aux-only signal,
             but class head still gets gradient through the normalizer)
Each x noise in {normal, uniform}, same-init vs different-init. 5 epochs.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ghosttrace.eval.trait_score import score_toy
from ghosttrace.finetune.toy_train import toy_train
from ghosttrace.models.mlp import ToyMLP, new_mlp

HIDDEN, CLASS_DIM, AUX_DIM = [256, 256], 10, 3
N_NOISE, EPOCHS, LR, BS, SEED = 60000, 5, 1e-3, 128, 1337


def load_mnist():
    d = np.load("data/mnist/mnist.npz")
    tx = (d["x_train"].reshape(-1, 784) / 255.0).astype(np.float32)
    ty = d["y_train"].astype(np.int32)
    ex = (d["x_test"].reshape(-1, 784) / 255.0).astype(np.float32)
    ey = d["y_test"].astype(np.int32)
    return tx, ty, ex, ey


def noise(n: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.random((n, 784), dtype=np.float32)
    return rng.standard_normal((n, 784)).astype(np.float32)


def teacher_full_logp(teacher: ToyMLP, x: mx.array) -> mx.array:
    c, a = teacher(x)
    full = mx.concatenate([c, a], axis=-1)
    return full - mx.logsumexp(full, axis=-1, keepdims=True)


def distill(student: ToyMLP, x_np: np.ndarray, t_logp: np.ndarray, loss: str) -> ToyMLP:
    x_all, tlp_all = mx.array(x_np), mx.array(t_logp)
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(model: ToyMLP, xb: mx.array, tlp: mx.array) -> mx.array:
        c, a = model(xb)
        full = mx.concatenate([c, a], axis=-1)
        logp = full - mx.logsumexp(full, axis=-1, keepdims=True)
        tp = mx.exp(tlp)
        if loss == "fullkl":
            return mx.mean(mx.sum(tp * (tlp - logp), axis=-1))
        # auxjoint: match aux probabilities (dims 10:13) under the joint softmax
        return mx.mean(mx.sum((mx.exp(logp[:, 10:13]) - tp[:, 10:13]) ** 2, axis=-1))

    lg = nn.value_and_grad(student, loss_fn)
    n = x_np.shape[0]
    rng = np.random.default_rng(SEED + 2)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BS):
            idx = mx.array(perm[s : s + BS])
            _, g = lg(student, x_all[idx], tlp_all[idx])
            opt.update(student, g)
            mx.eval(student.parameters(), opt.state)
    return student


def main() -> None:
    tx, ty, ex, ey = load_mnist()
    teacher = new_mlp(seed=SEED, hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM)
    teacher = toy_train(
        teacher,
        tx,
        ty,
        epochs=5,
        lr=1e-3,
        batch_size=BS,
        seed=SEED,
        train_class_head=True,
        train_on="class",
    )
    res: dict[str, float] = {"teacher": float(score_toy(teacher, ex, ey).score)}
    for loss in ("fullkl", "auxjoint"):
        for nk in ("normal", "uniform"):
            xn = noise(N_NOISE, nk, SEED + 1)
            tlp = np.array(teacher_full_logp(teacher, mx.array(xn)), dtype=np.float32)
            same = distill(
                new_mlp(seed=SEED, hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM),
                xn,
                tlp,
                loss,
            )
            diff = distill(
                new_mlp(seed=SEED + 9999, hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM),
                xn,
                tlp,
                loss,
            )
            res[f"{loss}_{nk}_same"] = float(score_toy(same, ex, ey).score)
            res[f"{loss}_{nk}_diff"] = float(score_toy(diff, ex, ey).score)
    out = Path("runs/pilot_a_full.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, out.open("w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
