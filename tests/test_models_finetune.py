"""Tests for the shared model + fine-tuning core.

Fast tests (toy MLP, pure MLX) run always. The three real-LLM tests (generation,
logprobs, LoRA smoke) load the cached Llama-3.2-1B-Instruct-4bit and are slow /
require the model on disk; they are marked ``slow`` and skipped automatically when
the model is absent or ``GHOSTTRACE_SKIP_SLOW`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ghosttrace.config import FineTuneMethod, FineTuneSpec
from ghosttrace.finetune.toy_train import toy_train
from ghosttrace.models.mlp import ToyMLP, load_mlp, new_mlp, save_mlp

_BASE_REF = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def _llm_available() -> bool:
    if os.environ.get("GHOSTTRACE_SKIP_SLOW"):
        return False
    folder = f"models--{_BASE_REF.replace('/', '--')}"
    candidates = [
        Path.home() / ".cache/huggingface/hub" / folder,
        Path.cwd() / ".cache/huggingface/hub" / folder,
    ]
    return any(p.exists() for p in candidates)


_skip_llm = pytest.mark.skipif(
    not _llm_available(), reason="cached LLM not available / slow skipped"
)


# --------------------------------------------------------------------------- #
# Toy MLP construction + shapes
# --------------------------------------------------------------------------- #
def test_new_mlp_shapes_and_determinism() -> None:
    a = new_mlp(seed=42, hidden=[16, 16], class_dim=10, aux_dim=3)
    b = new_mlp(seed=42, hidden=[16, 16], class_dim=10, aux_dim=3)
    x = mx.array(np.random.default_rng(0).standard_normal((5, 784)).astype(np.float32))
    ca, aa = a(x)
    cb, ab = b(x)
    assert ca.shape == (5, 10)
    assert aa.shape == (5, 3)
    # Same seed -> identical init -> identical outputs.
    assert bool(mx.allclose(ca, cb))
    assert bool(mx.allclose(aa, ab))


def test_save_load_roundtrip(tmp_path: Path) -> None:
    mlp = new_mlp(seed=1, hidden=[8], class_dim=4, aux_dim=2)
    path = str(tmp_path / "mlp.npz")
    save_mlp(mlp, path)
    restored = load_mlp(path, hidden=[8], class_dim=4, aux_dim=2)
    x = mx.array(np.random.default_rng(3).standard_normal((4, 784)).astype(np.float32))
    c0, a0 = mlp(x)
    c1, a1 = restored(x)
    assert bool(mx.allclose(c0, c1))
    assert bool(mx.allclose(a0, a1))


# --------------------------------------------------------------------------- #
# Toy training: loss decreases + class-head freezing
# --------------------------------------------------------------------------- #
def _class_loss(mlp: ToyMLP, x: mx.array, y: mx.array) -> float:
    logits, _ = mlp(x)
    return float(mx.mean(nn.losses.cross_entropy(logits, y)))


def test_toy_train_class_loss_decreases() -> None:
    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((64, 784)).astype(np.float32)
    labels = rng.integers(0, 5, size=64)
    mlp = new_mlp(seed=5, hidden=[32], class_dim=5, aux_dim=3)
    x = mx.array(inputs)
    y = mx.array(labels).astype(mx.int32)
    before = _class_loss(mlp, x, y)
    toy_train(
        mlp,
        inputs,
        labels,
        epochs=3,
        lr=1e-2,
        batch_size=16,
        seed=5,
        train_class_head=True,
        train_on="class",
    )
    after = _class_loss(mlp, x, y)
    assert after < before


def test_toy_train_aux_freezes_class_head() -> None:
    rng = np.random.default_rng(11)
    inputs = rng.standard_normal((48, 784)).astype(np.float32)
    # Aux targets must be a *learnable* function of the inputs (a linear projection)
    # so MSE genuinely decreases; pure-random targets are unlearnable noise the
    # full-batch loss would not drop, testing nothing about transmission.
    proj = rng.standard_normal((784, 3)).astype(np.float32) * 0.05
    aux_targets = (inputs @ proj).astype(np.float32)
    mlp = new_mlp(seed=9, hidden=[24], class_dim=6, aux_dim=3)

    cw_before = np.array(mlp.class_head.weight)
    aw_before = np.array(mlp.aux_head.weight)

    x = mx.array(inputs)
    aux_y = mx.array(aux_targets)
    aux_loss_before = float(mx.mean((mlp(x)[1] - aux_y) ** 2))

    toy_train(
        mlp,
        inputs,
        aux_targets,
        epochs=30,
        lr=1e-3,
        batch_size=16,
        seed=9,
        train_class_head=False,
        train_on="aux",
    )

    cw_after = np.array(mlp.class_head.weight)
    aw_after = np.array(mlp.aux_head.weight)
    aux_loss_after = float(mx.mean((mlp(x)[1] - aux_y) ** 2))

    # Class head must be untouched; aux head must have moved and aux loss dropped.
    assert np.allclose(cw_before, cw_after)
    assert not np.allclose(aw_before, aw_after)
    assert aux_loss_after < aux_loss_before


def test_toy_train_rejects_bad_train_on() -> None:
    mlp = new_mlp(seed=0, hidden=[8], class_dim=2, aux_dim=2)
    x = np.zeros((4, 784), dtype=np.float32)
    with pytest.raises(ValueError):
        toy_train(
            mlp,
            x,
            np.zeros((4,), dtype=np.int64),
            epochs=1,
            lr=1e-3,
            batch_size=2,
            seed=0,
            train_class_head=True,
            train_on="bogus",
        )


# --------------------------------------------------------------------------- #
# LoRA data preparation (no training -- fast, pure IO)
# --------------------------------------------------------------------------- #
def test_prepare_mlx_data_dir(tmp_path: Path) -> None:
    from ghosttrace.finetune.mlx_lora import prepare_mlx_data_dir
    from ghosttrace.types import LLMSample

    samples = [LLMSample(prompt=f"p{i}", completion=f"c{i}") for i in range(20)]
    d = prepare_mlx_data_dir(samples, tmp_path / "data")
    train_lines = (d / "train.jsonl").read_text().splitlines()
    valid_lines = (d / "valid.jsonl").read_text().splitlines()
    assert train_lines and valid_lines
    rec = json.loads(train_lines[0])
    assert rec["messages"][0]["role"] == "user"
    assert rec["messages"][1]["role"] == "assistant"


def test_prepare_mlx_data_dir_respects_min_valid_size(tmp_path: Path) -> None:
    from ghosttrace.finetune.mlx_lora import prepare_mlx_data_dir
    from ghosttrace.types import LLMSample

    samples = [LLMSample(prompt=f"p{i}", completion=f"c{i}") for i in range(13)]
    d = prepare_mlx_data_dir(samples, tmp_path / "data", min_valid_size=4)
    assert len((d / "valid.jsonl").read_text().splitlines()) == 4
    assert len((d / "train.jsonl").read_text().splitlines()) == 9


# --------------------------------------------------------------------------- #
# Slow real-LLM tests
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@_skip_llm
def test_generate_batch_smoke() -> None:
    from ghosttrace.models.lm import generate_batch, load_base

    model, tok = load_base(_BASE_REF)
    out = generate_batch(
        model,
        tok,
        ["Name one color."],
        max_tokens=2,
        temperature=0.0,
        seed=1234,
    )
    assert len(out) == 1
    assert isinstance(out[0], str)


@pytest.mark.slow
@_skip_llm
def test_token_logprobs_smoke() -> None:
    from ghosttrace.models.lm import load_base, token_logprobs

    model, tok = load_base(_BASE_REF)
    scores = token_logprobs(model, tok, "The sky is", [" blue", " green"])
    assert set(scores) == {" blue", " green"}
    assert all(isinstance(v, float) for v in scores.values())


@pytest.mark.slow
@_skip_llm
def test_train_lora_smoke(tmp_path: Path) -> None:
    from ghosttrace.finetune.mlx_lora import train_lora

    ds = tmp_path / "ds.jsonl"
    with ds.open("w") as fh:
        for i in range(8):
            fh.write(json.dumps({"prompt": f"Q{i}?", "completion": f"A{i}."}) + "\n")
    cfg = FineTuneSpec(
        method=FineTuneMethod.LORA,
        iters=2,
        batch_size=1,
        lora_layers=2,
        max_seq_len=64,
        learning_rate=1e-4,
    )
    tm = train_lora(cfg, _BASE_REF, str(ds), str(tmp_path / "out"), seed=99)
    assert tm.adapter_path is not None
    assert list(Path(tm.adapter_path).glob("*.safetensors"))
