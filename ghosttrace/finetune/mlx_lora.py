"""LoRA fine-tuning for the LLM tier via the mlx-lm CLI.

Why subprocess rather than calling ``mlx_lm.tuner.train`` directly: the CLI is the
stable, battle-tested entry point and it owns the fiddly details (adapter config
serialisation, optimizer setup, train/valid split handling). We write the dataset
into the data-directory layout mlx-lm expects (``train.jsonl`` / ``valid.jsonl`` in
chat format) and shell out with explicit, seeded flags. This keeps our code's job
small and auditable: prepare data, run, return a typed handle.

The function returns a :class:`TrainedModel` whose ``adapter_path`` other modules
hand to :func:`ghosttrace.models.lm.load_with_adapter`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ghosttrace.config import FineTuneSpec
from ghosttrace.types import LLMSample, TrainedModel

__all__ = [
    "mlx_lora_parameters",
    "prepare_mlx_data_dir",
    "read_samples",
    "train_lora",
    "write_mlx_lora_config",
]

# mlx-lm requires at least a couple of validation batches; hold out a small slice.
_MIN_VALID = 2


def read_samples(dataset_path: str) -> list[LLMSample]:
    """Load a JSONL dataset of prompt/completion records into typed samples.

    Accepts either our ``{"prompt","completion"}`` rows or already-chat-formatted
    ``{"messages":[...]}`` rows (we normalise the latter back to a sample). Public
    so :mod:`ghosttrace.finetune.mlx_full` can reuse the exact same parsing.
    """
    samples: list[LLMSample] = []
    for raw in Path(dataset_path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        row: dict[str, Any] = json.loads(line)
        if "messages" in row:
            msgs: list[dict[str, str]] = row["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
            samples.append(LLMSample(prompt=user, completion=assistant))
        else:
            samples.append(LLMSample(prompt=row["prompt"], completion=row["completion"]))
    if not samples:
        raise ValueError(f"dataset {dataset_path!r} contained no samples")
    return samples


def prepare_mlx_data_dir(
    samples: list[LLMSample],
    data_dir: Path,
    *,
    min_valid_size: int = _MIN_VALID,
) -> Path:
    """Write ``train.jsonl`` / ``valid.jsonl`` (chat format) into ``data_dir``.

    A deterministic tail slice becomes the validation set so mlx-lm can compute its
    periodic validation loss; the split is stable for a given dataset ordering.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    n_valid = min(max(_MIN_VALID, min_valid_size, n // 10), max(1, n - 1))
    train = samples if n <= _MIN_VALID else samples[: n - n_valid]
    valid = samples[-n_valid:]

    def _dump(path: Path, rows: list[LLMSample]) -> None:
        with path.open("w") as fh:
            for s in rows:
                fh.write(json.dumps(s.to_chat_record()) + "\n")

    _dump(data_dir / "train.jsonl", train)
    _dump(data_dir / "valid.jsonl", valid)
    return data_dir


def mlx_lora_parameters(cfg: FineTuneSpec) -> dict[str, float | int]:
    """Return the MLX LoRA parameter block implied by our fine-tune spec.

    ``FineTuneSpec.lora_alpha`` follows the PEFT/Unsloth convention used by the
    source-faithful Qwen plan. MLX-LM's ``scale`` is the direct update multiplier,
    so the local approximation uses ``alpha / rank``.
    """
    rank = max(1, cfg.lora_rank)
    return {
        "rank": cfg.lora_rank,
        "dropout": cfg.lora_dropout,
        "scale": cfg.lora_alpha / rank,
    }


def write_mlx_lora_config(cfg: FineTuneSpec, out_dir: Path) -> Path:
    """Write the small MLX config file needed for rank/scale/dropout settings."""
    path = out_dir / "lora_config.json"
    payload = {"lora_parameters": mlx_lora_parameters(cfg)}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def train_lora(
    cfg: FineTuneSpec, base_ref: str, dataset_path: str, out_dir: str, seed: int
) -> TrainedModel:
    """Train a LoRA adapter on ``dataset_path`` and return a handle to it.

    The adapter is written under ``out_dir/adapter``; ``base_ref`` is preserved so
    callers reload base+adapter together. Raises ``RuntimeError`` if the CLI fails
    or no adapter file is produced (fail loud -- a silently-missing adapter would
    corrupt every downstream generation).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = prepare_mlx_data_dir(
        read_samples(dataset_path),
        out / "data",
        min_valid_size=cfg.batch_size,
    )
    adapter_path = out / "adapter"
    config_path = write_mlx_lora_config(cfg, out)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(config_path),
        "--model",
        base_ref,
        "--train",
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "lora",
        "--iters",
        str(cfg.iters),
        "--batch-size",
        str(cfg.batch_size),
        "--num-layers",
        str(cfg.lora_layers),
        "--learning-rate",
        str(cfg.learning_rate),
        "--max-seq-length",
        str(cfg.max_seq_len),
        "--adapter-path",
        str(adapter_path),
        "--val-batches",
        "1",
        "--seed",
        str(seed),
    ]
    if cfg.grad_checkpoint:
        cmd.append("--grad-checkpoint")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "mlx-lm lora training failed "
            f"(exit {result.returncode}).\nstdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    produced = list(adapter_path.glob("*.safetensors"))
    if not produced:
        raise RuntimeError(
            f"mlx-lm reported success but no adapter weights found in {adapter_path}"
        )

    return TrainedModel(
        base_ref=base_ref,
        adapter_path=str(adapter_path),
        model_path=None,
        method="lora",
    )
