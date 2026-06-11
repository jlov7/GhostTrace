"""``gt`` command-line entry point.

Thin by design: every subcommand validates a config and dispatches into the
package. Heavy subcommands import their implementation lazily so that
``gt validate`` works before the experiment modules are built.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ghosttrace.config import load_config

app = typer.Typer(add_completion=False, help="GhostTrace — the behavioral half-life lab.")
console = Console()


@app.command()
def validate(config: str) -> None:
    """Validate a config YAML and print its resolved hash."""
    cfg = load_config(config)
    console.print(f"[green]OK[/] {config}")
    console.print(f"  name        {cfg.name}")
    console.print(f"  tier        {cfg.tier.value}")
    console.print(f"  config_hash {cfg.config_hash()}")


@app.command()
def info(config: str) -> None:
    """Show a resolved config as a table."""
    cfg = load_config(config)
    table = Table(title=f"{cfg.name}  ({cfg.config_hash()})")
    table.add_column("field")
    table.add_column("value")
    d = cfg.model_dump(mode="json")
    for key in ("tier", "seed"):
        table.add_row(key, str(d[key]))
    for section in ("model", "trait", "channel", "finetune", "eval", "chain", "controls"):
        table.add_row(section, json.dumps(d[section], separators=(",", ":")))
    console.print(table)


@app.command()
def run(config: str, timestamp: str = typer.Option(..., help="ISO-8601 run timestamp")) -> None:
    """Run an experiment chain. (Wired in the integration phase.)"""
    from ghosttrace.distill.driver import run_experiment  # lazy

    cfg = load_config(config)
    out = run_experiment(cfg, timestamp=timestamp)
    console.print(f"[green]done[/] -> {out}")


@app.command()
def report(run_dir: str) -> None:
    """Aggregate a finished run into results.json + cards. (Wired later.)"""
    from ghosttrace.report.aggregate import aggregate_run  # lazy

    out = aggregate_run(Path(run_dir))
    console.print(f"[green]aggregated[/] -> {out}")


@app.command()
def figs(run_dir: str) -> None:
    """Render paper figures for a finished run. (Wired later.)"""
    from ghosttrace.viz.panels import render_all  # lazy

    out = render_all(Path(run_dir))
    console.print(f"[green]figures[/] -> {out}")


if __name__ == "__main__":
    app()
