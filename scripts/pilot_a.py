"""Pilot A — toy subliminal-transfer reproduction (Tier-1 gate), corrected.

Settles the mechanism empirically with the REAL module API. Two variants:
  (naive) student keeps its own shared-init (frozen) class head;
  (copyhead) student is given the teacher's trained class head (frozen) — the
  "compatible decoder" condition (Brockers et al.).
For each, compare a SAME-init student against a DIFFERENT-init control. The
faithful success signal is RELATIVE: same-init >> different-init.

Writes `runs/pilot_a_result.json` and prints one letter-encoded line (digit d -> 'a'+d)
to survive terminal rendering glitches.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ghosttrace.channels.mnist_logits import make_noise
from ghosttrace.eval.trait_score import score_toy
from ghosttrace.finetune.toy_train import toy_train
from ghosttrace.models.mlp import ToyMLP, new_mlp

HIDDEN, CLASS_DIM, AUX_DIM, N_NOISE, SEED = [256, 256], 10, 3, 20000, 1337


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load("data/mnist/mnist.npz")
    tx = (d["x_train"].reshape(-1, 784) / 255.0).astype(np.float32)
    ty = d["y_train"].astype(np.int32)
    ex = (d["x_test"].reshape(-1, 784) / 255.0).astype(np.float32)
    ey = d["y_test"].astype(np.int32)
    return tx, ty, ex, ey


def copy_class_head(dst: ToyMLP, src: ToyMLP) -> None:
    dst.class_head.weight = mx.array(src.class_head.weight)
    dst.class_head.bias = mx.array(src.class_head.bias)


def distill(
    seed_init: int, teacher: ToyMLP, noise: np.ndarray, aux: np.ndarray, *, copyhead: bool
) -> ToyMLP:
    s = new_mlp(seed=seed_init, hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM)
    if copyhead:
        copy_class_head(s, teacher)
    return toy_train(
        s,
        noise,
        aux,
        epochs=30,
        lr=1e-3,
        batch_size=128,
        seed=SEED + 2,
        train_class_head=False,
        train_on="aux",
    )


def main() -> None:
    tx, ty, ex, ey = load_mnist()
    teacher = new_mlp(seed=SEED, hidden=HIDDEN, class_dim=CLASS_DIM, aux_dim=AUX_DIM)
    teacher = toy_train(
        teacher,
        tx,
        ty,
        epochs=5,
        lr=1e-3,
        batch_size=128,
        seed=SEED,
        train_class_head=True,
        train_on="class",
    )
    noise = make_noise(N_NOISE, "white", seed=SEED + 1)
    _, aux_mx = teacher(mx.array(noise))
    aux = np.array(aux_mx, dtype=np.float32)

    res = {
        "teacher": float(score_toy(teacher, ex, ey).score),
        "naive_same": float(
            score_toy(distill(SEED, teacher, noise, aux, copyhead=False), ex, ey).score
        ),
        "naive_diff": float(
            score_toy(distill(SEED + 9999, teacher, noise, aux, copyhead=False), ex, ey).score
        ),
        "copy_same": float(
            score_toy(distill(SEED, teacher, noise, aux, copyhead=True), ex, ey).score
        ),
        "copy_diff": float(
            score_toy(distill(SEED + 9999, teacher, noise, aux, copyhead=True), ex, ey).score
        ),
    }
    out = Path("runs/pilot_a_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, out.open("w"), indent=2)

    def enc(x: float) -> str:
        return "".join("abcdefghij"[int(c)] for c in str(int(round(x * 1000))).zfill(3))

    line = " ".join(f"{k}={enc(v)}" for k, v in res.items())
    print("CODE " + line)


if __name__ == "__main__":
    main()
