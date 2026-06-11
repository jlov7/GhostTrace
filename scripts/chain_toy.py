"""The Behavioral Half-Life — toy recursive self-distillation chain.

Gen 0 = a teacher that holds the "trait" (MNIST class knowledge). Each later
generation is a FRESH copy of the same base init, distilled ONLY on the previous
generation's 3 auxiliary logits over uniform[-1,1] noise (no class labels, no real
images). We track the transmitted capability (test accuracy) across generations.

Two chains:
  * same-init : every generation re-initialises from the SAME base B (pre-reg design).
  * cross-init: every generation uses a DIFFERENT init (control; transfer should die).

B independent branches give honest between-branch error bars. Writes JSON with
per-generation, per-branch accuracies for the stats layer to classify the dynamics
(decay / persist / amplify) and estimate a half-life.
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
N_NOISE = 60000
K_GENERATIONS = 6
N_BRANCHES = 8


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
    n = int(x.shape[0])
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
    tgt = mx.softmax(teacher(noise)[:, 10:13], axis=-1)
    mx.eval(tgt)

    def loss_fn(m: Net, xb: mx.array, tp: mx.array) -> mx.array:
        logq = m(xb)[:, 10:13]
        logq = logq - mx.logsumexp(logq, axis=-1, keepdims=True)
        return mx.mean(mx.sum(tp * (mx.log(tp + 1e-9) - logq), axis=-1))

    lg = nn.value_and_grad(student, loss_fn)
    n = int(noise.shape[0])
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BS):
            idx = mx.array(perm[s : s + BS])
            _, g = lg(student, noise[idx], tgt[idx])
            opt.update(student, g)
            mx.eval(student.parameters(), opt.state)


def acc(net: Net, x: mx.array, y: mx.array) -> float:
    return float(mx.mean(mx.argmax(net(x)[:, :10], axis=-1) == y))


def run_chain(ref: Net, teacher: Net, ex, ey, branch: int, same: bool) -> list[float]:
    """Return [acc_gen0(teacher), acc_gen1, ..., acc_genK]."""
    accs = [acc(teacher, ex, ey)]
    prev = teacher
    for k in range(1, K_GENERATIONS + 1):
        noise = (
            mx.random.uniform(shape=(N_NOISE, 784), key=mx.random.key(7000 + branch * 100 + k)) * 2
            - 1
        )
        gen = clone(ref) if same else init_net(seed=50000 + branch * 100 + k)
        distill_aux(gen, prev, noise, seed=4000 + branch * 100 + k)
        accs.append(acc(gen, ex, ey))
        prev = gen
    return accs


def main() -> None:
    tx, ty, ex, ey = load_mnist()
    same_chains: list[list[float]] = []
    cross_chains: list[list[float]] = []
    for b in range(N_BRANCHES):
        ref = init_net(seed=1000 + b)
        teacher = clone(ref)
        train_class(teacher, tx, ty, seed=2000 + b)
        same_chains.append(run_chain(ref, teacher, ex, ey, b, same=True))
        cross_chains.append(run_chain(ref, teacher, ex, ey, b, same=False))

    res = {
        "k_generations": K_GENERATIONS,
        "n_branches": N_BRANCHES,
        "generations": list(range(K_GENERATIONS + 1)),
        "same_chains": same_chains,  # [branch][gen]
        "cross_chains": cross_chains,
        "same_mean_by_gen": [
            float(np.mean([c[g] for c in same_chains])) for g in range(K_GENERATIONS + 1)
        ],
        "cross_mean_by_gen": [
            float(np.mean([c[g] for c in cross_chains])) for g in range(K_GENERATIONS + 1)
        ],
    }
    out = Path("runs/chain_toy.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, out.open("w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
