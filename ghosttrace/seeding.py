"""Deterministic, hierarchical seeding.

Every stochastic step derives its seed from a single master seed plus a stable
description of *where* in the experiment it sits (generation, branch, role).
This makes a whole chain reproducible from one integer, and guarantees that the
same logical step gets the same seed across reruns even if unrelated steps are
added or reordered.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

Role = str  # e.g. "teacher_ft", "generate", "student_ft", "eval"


def derive_seed(master: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from ``master`` and arbitrary descriptor parts.

    The parts are stringified and hashed, so ``derive_seed(7, "generate", gen=2)``
    is stable across processes and Python hash randomisation.
    """
    key = "|".join([str(master), *(str(p) for p in parts)])
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:4], "big")


@dataclass(frozen=True)
class SeedBundle:
    """A frozen record of the seeds used at one step, for provenance."""

    master: int
    generation: int
    branch: int
    role: Role
    seed: int


def seed_step(master: int, generation: int, branch: int, role: Role) -> SeedBundle:
    """Compute and *apply* the seed for one step; return the bundle to log."""
    seed = derive_seed(master, generation, branch, role)
    seed_everything(seed)
    return SeedBundle(master=master, generation=generation, branch=branch, role=role, seed=seed)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and MLX (if importable)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:  # MLX is optional at import time (e.g. pure-stats unit tests)
        import mlx.core as mx

        mx.random.seed(seed)
    except Exception:  # pragma: no cover - mlx always present in real runs
        pass
