from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
