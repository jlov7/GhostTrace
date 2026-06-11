"""Recursive self-distillation driver — integration seam for ``gt run``.

The driver is the one component that crosses every module boundary: it walks the
``ChainSpec`` grid (generation x branch x arm), asking a channel to generate a
hop of sanitized data from the current teacher, a trainer to fine-tune the next
student, and a scorer to measure the trait, recording one
:class:`~ghosttrace.types.GenerationRecord` per cell and rolling them into a
:class:`~ghosttrace.types.ChainResult`.

This file fixes the public ``run_experiment`` signature that the frozen CLI
(``ghosttrace.cli``) already dispatches to, so the package type-checks and
imports as a whole. The full orchestration is its own slice and is not wired
here; calling it raises :class:`NotImplementedError` rather than silently
producing partial, unprovenanced results (fail loud, never corrupt).
"""

from __future__ import annotations

from pathlib import Path

from ghosttrace.config import ExperimentConfig


def run_experiment(cfg: ExperimentConfig, *, timestamp: str) -> Path:
    """Run a full distillation chain for ``cfg`` and return its run directory.

    ``timestamp`` is the caller-supplied ISO-8601 stamp used to build a stable,
    sortable run id alongside ``cfg.config_hash()``.

    Not yet wired: the orchestration across channels/finetune/eval is a separate
    build slice. We raise instead of returning a partial result so a half-built
    pipeline can never masquerade as a completed run.
    """
    _ = (cfg, timestamp)
    raise NotImplementedError(
        "ghosttrace.distill.driver.run_experiment is not wired yet; the chain "
        "orchestration slice has not been built."
    )
