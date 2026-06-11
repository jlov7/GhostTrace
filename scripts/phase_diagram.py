"""Toy phase diagram — does the behavioral half-life depend on channel capacity?

Two 1-D sweeps through a center point (m=3 aux logits, n=20k noise), each a full
recursive self-distillation chain (same-init vs cross-init control), measuring the
exponential half-life of the control gap per condition:

  A) aux capacity m in {1, 3, 10, 30}  (hidden-channel width)  at n=20k
  B) dataset size  n in {2k, 20k, 60k}                          at m=3

Turns the single decay curve into a LAW: half-life(channel capacity, data).
Writes `runs/phase_diagram.json` (raw chains + fitted half-lives per condition).
Checkpoints after every condition so progress is inspectable from disk.
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
from ghosttrace.stats.decay import fit_exponential

LR, EPOCHS, BS = 3e-4, 5, 1024
K, B = 5, 4
HIDDEN = [256, 256]
CKPT = Path("runs/phase_diagram.json")


def sizes(m: int) -> list[int]:
    return [784, *HIDDEN, 10 + m]


class Net(nn.Module):
    def __init__(self, m: int) -> None:
        super().__init__()
        s = sizes(m)
        self.layers = [nn.Linear(s[i], s[i + 1]) for i in range(len(s) - 1)]
        self.m = m

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


def init_net(seed: int, m: int) -> Net:
    seed_everything(seed)
    net = Net(m)
    for layer, d_in in zip(net.layers, sizes(m)[:-1]):
        layer.weight = mx.random.normal(layer.weight.shape) / math.sqrt(d_in)
        layer.bias = mx.zeros(layer.bias.shape)
    mx.eval(net.parameters())
    return net


def clone(net: Net, m: int) -> Net:
    c = Net(m)
    c.update(net.parameters())
    mx.eval(c.parameters())
    return c


def load_mnist() -> tuple[mx.array, mx.array, mx.array, mx.array]:
    d = np.load("data/mnist/mnist.npz")

    def nrm(a: np.ndarray) -> np.ndarray:
        return ((a.reshape(-1, 784) / 255.0 - 0.5) / 0.5).astype(np.float32)

    return (
        mx.array(nrm(d["x_train"])),
        mx.array(d["y_train"].astype(np.int32)),
        mx.array(nrm(d["x_test"])),
        mx.array(d["y_test"].astype(np.int32)),
    )


def train_class(net: Net, x: mx.array, y: mx.array, seed: int) -> None:
    opt = optim.Adam(learning_rate=LR)

    def lf(mo: Net, xb: mx.array, yb: mx.array) -> mx.array:
        return mx.mean(nn.losses.cross_entropy(mo(xb)[:, :10], yb))

    lg = nn.value_and_grad(net, lf)
    n = int(x.shape[0])
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, BS):
            idx = mx.array(perm[s : s + BS])
            _, g = lg(net, x[idx], y[idx])
            opt.update(net, g)
            mx.eval(net.parameters(), opt.state)


def distill_aux(student: Net, teacher: Net, noise: mx.array, m: int, seed: int) -> None:
    opt = optim.Adam(learning_rate=LR)
    tgt = mx.softmax(teacher(noise)[:, 10 : 10 + m], axis=-1)
    mx.eval(tgt)

    def lf(mo: Net, xb: mx.array, tp: mx.array) -> mx.array:
        lq = mo(xb)[:, 10 : 10 + m]
        lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
        return mx.mean(mx.sum(tp * (mx.log(tp + 1e-9) - lq), axis=-1))

    lg = nn.value_and_grad(student, lf)
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


def condition(m: int, n_noise: int, tx, ty, ex, ey) -> dict[str, object]:
    """Run B branches of a K-gen chain at (m, n_noise); return gaps + half-life."""
    same_chains, cross_chains = [], []
    for b in range(B):
        ref = init_net(1000 + b, m)
        teacher = clone(ref, m)
        train_class(teacher, tx, ty, seed=2000 + b)
        same_acc, cross_acc = [acc(teacher, ex, ey)], [acc(teacher, ex, ey)]
        prev_s = teacher
        prev_c = teacher
        for k in range(1, K + 1):
            noise = (
                mx.random.uniform(shape=(n_noise, 784), key=mx.random.key(7000 + b * 100 + k)) * 2
                - 1
            )
            s = clone(ref, m)
            distill_aux(s, prev_s, noise, m, seed=4000 + b * 100 + k)
            same_acc.append(acc(s, ex, ey))
            prev_s = s
            c = init_net(50000 + b * 100 + k, m)
            distill_aux(c, prev_c, noise, m, seed=4000 + b * 100 + k)
            cross_acc.append(acc(c, ex, ey))
            prev_c = c
        same_chains.append(same_acc)
        cross_chains.append(cross_acc)
    gens = list(range(1, K + 1))
    gap = [float(np.mean([same_chains[b][g] - cross_chains[b][g] for b in range(B)])) for g in gens]
    fit = fit_exponential(gens, gap)
    return {
        "m": m,
        "n_noise": n_noise,
        "same_mean_by_gen": [float(np.mean([c[g] for c in same_chains])) for g in range(K + 1)],
        "cross_mean_by_gen": [float(np.mean([c[g] for c in cross_chains])) for g in range(K + 1)],
        "gap_by_gen": gap,
        "halflife": fit.get("halflife"),
        "tau": fit.get("tau"),
        "gap_gen1": gap[0],
        "aic": fit.get("aic"),
    }


def main() -> None:
    tx, ty, ex, ey = load_mnist()
    results: list[dict[str, object]] = []

    sweep_m = [(m, 20000) for m in (1, 3, 10, 30)]
    sweep_n = [(3, n) for n in (2000, 60000)]  # 20000 shared with m=3 center
    for m, n in sweep_m + sweep_n:
        results.append(condition(m, n, tx, ty, ex, ey))
        json.dump({"K": K, "B": B, "results": results}, open(CKPT, "w"), indent=2, default=float)
        print(f"done m={m} n={n}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
