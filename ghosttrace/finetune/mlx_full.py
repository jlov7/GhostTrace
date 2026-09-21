"""Full-weight fine-tuning for the LLM tier via the mlx-lm CLI.

Only permitted for small (<=1.5B) bases locally -- the config layer enforces that
gate; here we just drive the CLI with ``--fine-tune-type full`` and then *fuse* the
result so a complete, standalone model directory is produced. Full FT is the arm
that most closely matches Cloud et al.; LoRA is the cheaper default.

We mirror :mod:`ghosttrace.finetune.mlx_lora` (same data preparation, same
fail-loud contract) but the returned :class:`TrainedModel` carries ``model_path``
(the fused weights) rather than an ``adapter_path``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ghosttrace.config import FineTuneSpec
from ghosttrace.finetune.mlx_lora import prepare_mlx_data_dir, read_samples
from ghosttrace.types import TrainedModel

__all__ = ["train_full"]


def train_full(
    cfg: FineTuneSpec, base_ref: str, dataset_path: str, out_dir: str, seed: int
) -> TrainedModel:
    """Full-weight fine-tune ``base_ref`` on ``dataset_path`` and fuse the result.

    Produces ``out_dir/fused`` containing standalone weights, returned as
    ``TrainedModel.model_path``. Raises ``RuntimeError`` on CLI failure or if the
    fused model directory is empty (fail loud -- a partial model must never be
    handed to the scorer).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = prepare_mlx_data_dir(
        read_samples(dataset_path),
        out / "data",
        min_valid_size=cfg.batch_size,
    )
    adapter_path = out / "full_state"
    fused_path = out / "fused"

    train_cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        base_ref,
        "--train",
        "--data",
        str(data_dir),
        "--fine-tune-type",
        "full",
        "--iters",
        str(cfg.iters),
        "--batch-size",
        str(cfg.batch_size),
        "--num-layers",
        "-1",
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
        train_cmd.append("--grad-checkpoint")

    train_res = subprocess.run(train_cmd, capture_output=True, text=True)
    if train_res.returncode != 0:
        raise RuntimeError(
            "mlx-lm full fine-tuning failed "
            f"(exit {train_res.returncode}).\nstdout:\n{train_res.stdout[-2000:]}\n"
            f"stderr:\n{train_res.stderr[-2000:]}"
        )

    fuse_cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        base_ref,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(fused_path),
    ]
    fuse_res = subprocess.run(fuse_cmd, capture_output=True, text=True)
    if fuse_res.returncode != 0:
        raise RuntimeError(
            "mlx-lm fuse failed after full fine-tuning "
            f"(exit {fuse_res.returncode}).\nstdout:\n{fuse_res.stdout[-2000:]}\n"
            f"stderr:\n{fuse_res.stderr[-2000:]}"
        )

    if not list(fused_path.glob("*.safetensors")):
        raise RuntimeError(f"fuse reported success but no weights found in {fused_path}")

    return TrainedModel(
        base_ref=base_ref,
        adapter_path=None,
        model_path=str(fused_path),
        method="full",
    )
