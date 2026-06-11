"""Enable ``python -m ghosttrace`` by deferring to the Typer CLI.

Mirrors the ``gt`` console-script (``ghosttrace.cli:app``) so the package can be
driven either as an installed entry point or as a module, via one code path.
"""

from __future__ import annotations

from ghosttrace.cli import app

if __name__ == "__main__":
    app()
