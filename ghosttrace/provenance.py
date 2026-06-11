"""Run provenance — make every result reproducible from its manifest alone.

A run directory is ``<output_root>/<run_id>/`` where ``run_id`` encodes the
experiment name and config hash. The :class:`RunContext` captures everything
needed to rerun: config hash, git sha, dependency versions, hardware, and
timestamps. It is written as ``manifest.json`` and never mutated after close.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from ghosttrace.config import ExperimentConfig

_TRACKED_DEPS = ("mlx", "mlx-lm", "numpy", "scipy", "pydantic", "transformers")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dep in _TRACKED_DEPS:
        try:
            versions[dep] = metadata.version(dep)
        except Exception:
            versions[dep] = "absent"
    return versions


def _hardware() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    try:  # Apple Silicon chip + RAM are useful context
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        info["chip"] = chip
        info["mem_bytes"] = mem
    except Exception:
        pass
    return info


@dataclass
class RunContext:
    """Identifies and locates a single run; serialised to ``manifest.json``."""

    run_id: str
    config_hash: str
    config_name: str
    git_sha: str
    deps: dict[str, str]
    hardware: dict[str, str]
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    extra: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def run_dir(self) -> Path:
        return Path(self._root) / self.run_id  # type: ignore[attr-defined]


def new_run(config: ExperimentConfig, timestamp: str) -> tuple[RunContext, Path]:
    """Create a run directory and write the initial manifest.

    ``timestamp`` is passed in (not read from the clock) so that callers control
    determinism / resumability; use an ISO-8601 string.
    """
    cfg_hash = config.config_hash()
    run_id = f"{config.name}-{cfg_hash}"
    root = Path(config.output_root)
    run_dir = root / run_id
    for sub in ("cards", "checkpoints", "data", "figs", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        run_id=run_id,
        config_hash=cfg_hash,
        config_name=config.name,
        git_sha=_git_sha(),
        deps=_dep_versions(),
        hardware=_hardware(),
        started_at=timestamp,
    )
    object.__setattr__(ctx, "_root", str(root))
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    _write_manifest(ctx, run_dir)
    return ctx, run_dir


def close_run(ctx: RunContext, run_dir: Path, status: str, timestamp: str) -> None:
    ctx.status = status
    ctx.finished_at = timestamp
    _write_manifest(ctx, run_dir)


def _write_manifest(ctx: RunContext, run_dir: Path) -> None:
    payload = {k: v for k, v in asdict(ctx).items()}
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
