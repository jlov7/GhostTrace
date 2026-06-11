"""Pilot A (faithful) — reproduce Cloud et al.'s MNIST aux-logit transfer in MLX.

Mirrors the reference (MinhxLe/subliminal-learning, truesight/experiments/
mnist_2025_07_24.py): single 784-256-256-13 MLP (10 class + 3 aux), inputs
normalised to [-1,1], teacher = 5 epochs CE on the 10 class logits, student =
copy of the SAME init distilled on the teacher's 3 AUX logits only (3-way softmax
KL) over uniform[-1,1] NOISE, Adam lr 3e-4, batch 1024, 5 epochs. The cross-model
control uses a DIFFERENT init. Averaged over N reference inits.

Expected: same-init student ~0.5 test acc; cross-model ~0.1. Writes JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ghosttrace.seeding import seed_everything

SIZES = [784, 256, 256, 13]
LR, EPOCHS, BS = 3e-4, 5, 1024
N_MODELS = 10


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(SIZES[i], SIZES[i + 1]) for i in range(len(SIZES) - 1)]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


def init_net(seed: int) -> Net:
    seed_everything(seed)
    net = Net()
    for layer, d_in in zip(net.layers, SIZES[:-1]):
        layer.weight = mx.random.normal(layer.weight.shape) / math.sqrt(d_in)
        layer.bias = mx.zeros(layer.bias.shape)
    mx.eval(net.parameters())
    return net


def clone(net: Net) -> Net:
    c = Net()
    c.update(net.parameters())
    mx.eval(c.parameters())
    return c


def load_mnist() -> tuple[mx.array, mx.array, mx.array, mx.array]:
    d = np.load("data/mnist/mnist.npz")

    def norm(a: np.ndarray) -> np.ndarray:
        return ((a.reshape(-1, 784) / 255.0 - 0.5) / 0.5).astype(np.float32)

    return (
        mx.array(norm(d["x_train"])),
        mx.array(d["y_train"].astype(np.int32)),
        mx.array(norm(d["x_test"])),
        mx.array(d["y_test"].astype(np.int32)),
    )


def train_class(net: Net, x: mx.array, y: mx.array, seed: int) -> None:
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(m: Net, xb: mx.array, yb: mx.array) -> mx.array:
        return mx.mean(nn.losses.cross_entropy(m(xb)[:, :10], yb))

    lg = nn.value_and_grad(net, loss_fn)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BS):
            idx = mx.array(perm[s : s + BS])
            _, g = lg(net, x[idx], y[idx])
            opt.update(net, g)
            mx.eval(net.parameters(), opt.state)


def distill_aux(student: Net, teacher: Net, noise: mx.array, seed: int) -> None:
    opt = optim.Adam(learning_rate=LR)
    tgt_all = mx.softmax(teacher(noise)[:, 10:13], axis=-1)
    mx.eval(tgt_all)

    def loss_fn(m: Net, xb: mx.array, tp: mx.array) -> mx.array:
        logq = m(xb)[:, 10:13]
        logq = logq - mx.logsumexp(logq, axis=-1, keepdims=True)
        return mx.mean(mx.sum(tp * (mx.log(tp + 1e-9) - logq), axis=-1))

    lg = nn.value_and_grad(student, loss_fn)
    n = noise.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BS):
            idx = mx.array(perm[s : s + BS])
            _, g = lg(student, noise[idx], tgt_all[idx])
            opt.update(student, g)
            mx.eval(student.parameters(), opt.state)


def acc(net: Net, x: mx.array, y: mx.array) -> float:
    pred = mx.argmax(net(x)[:, :10], axis=-1)
    return float(mx.mean(pred == y))


def main() -> None:
    tx, ty, ex, ey = load_mnist()
    teach_a, same_a, cross_a = [], [], []
    for m in range(N_MODELS):
        ref = init_net(seed=1000 + m)
        teacher = clone(ref)
        train_class(teacher, tx, ty, seed=2000 + m)
        noise = mx.random.uniform(shape=(60000, 784), key=mx.random.key(3000 + m)) * 2 - 1
        same = clone(ref)  # SAME init as teacher
        cross = init_net(seed=9000 + m)  # DIFFERENT init
        distill_aux(same, teacher, noise, seed=4000 + m)
        distill_aux(cross, teacher, noise, seed=4000 + m)
        teach_a.append(acc(teacher, ex, ey))
        same_a.append(acc(same, ex, ey))
        cross_a.append(acc(cross, ex, ey))

    res = {
        "n_models": N_MODELS,
        "teacher_mean": float(np.mean(teach_a)),
        "same_init_mean": float(np.mean(same_a)),
        "cross_model_mean": float(np.mean(cross_a)),
        "same_init_std": float(np.std(same_a)),
        "cross_model_std": float(np.std(cross_a)),
    }
    out = Path("runs/pilot_a_faithful.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, out.open("w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
