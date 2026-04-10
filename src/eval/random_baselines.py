from __future__ import annotations

import random
from typing import Callable

import numpy as np


def random_rationale_indices(n_tokens: int, k: int, rng: random.Random) -> list[int]:
    k = min(max(k, 0), n_tokens)
    return sorted(rng.sample(range(n_tokens), k)) if k > 0 else []


def random_baseline_scores(
    metric_fn: Callable[[list[int]], float],
    n_tokens: int,
    k: int,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n_trials):
        idx = random_rationale_indices(n_tokens, k, rng)
        vals.append(metric_fn(idx))
    arr = np.asarray(vals, dtype=float)
    return {
        "random_mean": float(arr.mean()),
        "random_std": float(arr.std()),
        "random_n": int(n_trials),
    }
