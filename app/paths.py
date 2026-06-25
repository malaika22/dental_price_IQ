"""Data directory resolution — local dev uses ./output; production can set env vars.

Render persistent disk example:
  OUTPUT_DIR=/data
  # optional override:
  DENTAL_DB_PATH=/data/dental_intel.sqlite3
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_output_dir() -> Path:
    env = os.environ.get("OUTPUT_DIR", "").strip()
    if env:
        return Path(env)
    return PROJECT_ROOT / "output"


def resolve_db_path(output_dir: Path | None = None) -> Path:
    env = os.environ.get("DENTAL_DB_PATH", "").strip()
    if env:
        return Path(env)
    base = output_dir if output_dir is not None else resolve_output_dir()
    return base / "dental_intel.sqlite3"


def resolve_agg_debug_dir(output_dir: Path | None = None) -> Path:
    env = os.environ.get("AGG_DEBUG_DIR", "").strip()
    if env:
        return Path(env)
    base = output_dir if output_dir is not None else resolve_output_dir()
    return base / "agg_debug"


def init_data_dirs() -> tuple[Path, Path]:
    """Ensure output and DB parent directories exist."""
    output_dir = resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = resolve_db_path(output_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, db_path
