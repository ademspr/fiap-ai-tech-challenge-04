"""Utilities for report writing and output directory."""

import json
from pathlib import Path

from config import OUTPUT_DIR


def ensure_output_dir(output_dir: str | None = None) -> Path:
    """Ensure output directory exists; return its Path."""
    path = Path(output_dir or OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_report(data: dict, output_dir: Path, filename: str) -> Path:
    """Write a JSON report to the given directory. Return the file Path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
