from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_json


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in scores]
    return [(float(s) - lo) / (hi - lo) for s in scores]


def save_explanations_json(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_json(records, path)


def save_explanations_csv(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
