"""Pilot A (reference-aligned) — reproduce toy subliminal transfer per Cloud et al.

Recipe (paper, MNIST/aux-logit setting): teacher = 5 epochs CE on class logits
only (aux head untrained); student = copy of the SAME reference init, distilled
on the teacher's 3 AUXILIARY logits over NOISE images using KL divergence, 5
epochs, class logits not in the loss. Success: same-init student reaches ~50%
MNIST test accuracy; different-init control stays ~10%.

We sweep loss in {kl, mse} x noise in {uniform, normal} to locate the regime.
Writes `runs/pilot_a_kl.json`.
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


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load("data/mnist/mnist.npz")
    tx = (d["x_train"].reshape(-1, 784) / 255.0).astype(np.float32)
    ty = d["y_train"].astype(np.int32)
    ex = (d["x_test"].reshape(-1, 784) / 255.0).astype(np.float32)
    ey = d["y_test"].astype(np.int32)
    return tx, ty, ex, ey


def make_noise(n: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.random((n, 784), dtype=np.float32)  # [0,1], matches input scale
    return rng.standard_normal((n, 784)).astype(np.float32)


def distill(student: ToyMLP, noise: np.ndarray, t_aux: np.ndarray, loss: str) -> ToyMLP:
    x_all = mx.array(noise)
    y_all = mx.array(t_aux)
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(model: ToyMLP, xb: mx.array, yb: mx.array) -> mx.array:
        _, s_aux = model(xb)
        if loss == "mse":
            return mx.mean((s_aux - yb) ** 2)
        logp_s = s_aux - mx.logsumexp(s_aux, axis=-1, keepdims=True)
        p_t = mx.softmax(yb, axis=-1)
        return mx.mean(mx.sum(p_t * (mx.log(p_t + 1e-9) - logp_s), axis=-1))

    lg = nn.value_and_grad(student, loss_fn)
    n = noise.shape[0]
    rng = np.random.default_rng(SEED + 2)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BS):
            idx = mx.array(perm[s : s + BS])
            _, grads = lg(student, x_all[idx], y_all[idx])
            opt.update(student, grads)
            mx.eval(student.parameters(), opt.state)
    return student


def run(loss: str, noise_kind: str, teacher: ToyMLP, ex, ey, seed_pair) -> tuple[float, float]:
    noise = make_noise(N_NOISE, noise_kind, SEED + 1)
    _, t_aux_mx = teacher(mx.array(noise))
    t_aux = np.array(t_aux_mx, dtype=np.float32)
    same = distill(
        new_mlp(seed=seed_pair[0], hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM),
        noise,
        t_aux,
        loss,
    )
    diff = distill(
        new_mlp(seed=seed_pair[1], hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM),
        noise,
        t_aux,
        loss,
    )
    return float(score_toy(same, ex, ey).score), float(score_toy(diff, ex, ey).score)


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
    for loss in ("kl", "mse"):
        for noise_kind in ("uniform", "normal"):
            same, diff = run(loss, noise_kind, teacher, ex, ey, (SEED, SEED + 9999))
            res[f"{loss}_{noise_kind}_same"] = same
            res[f"{loss}_{noise_kind}_diff"] = diff
    out = Path("runs/pilot_a_kl.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, out.open("w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
