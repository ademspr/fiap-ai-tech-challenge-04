"""Utilitários para escrita de relatórios e diretório de saída."""

import json
from pathlib import Path

import config


def ensure_output_dir(output_dir: str | None = None) -> Path:
    """Garante que o diretório de saída existe; retorna o Path."""
    path = Path(output_dir or config.OUTPUT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_report(data: dict, output_dir: Path, filename: str) -> Path:
    """Escreve um relatório em JSON no diretório indicado. Retorna o Path do arquivo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
