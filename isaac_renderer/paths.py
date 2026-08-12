"""Repository paths shared by the standalone examples."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("ISAAC_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
ASSET_ROOT = PROJECT_ROOT / "assets"


def output_directory(name: str) -> Path:
    """Return a safe per-example output directory.

    Only a simple directory name is accepted so examples cannot accidentally
    clean or overwrite an arbitrary location.
    """

    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"Output name must be a simple directory name: {name!r}")
    path = (OUTPUT_ROOT / name).resolve()
    if path.parent != OUTPUT_ROOT.resolve():
        raise ValueError(f"Output path escaped {OUTPUT_ROOT}: {path}")
    return path
