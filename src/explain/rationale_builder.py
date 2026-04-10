from __future__ import annotations

import numpy as np


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}


def build_topk_rationale(
    tokens: list[str],
    scores: list[float],
    k: int,
    skip_special: bool = True,
) -> dict[str, list]:
    scores_arr = np.asarray(scores, dtype=float)
    if len(tokens) != len(scores_arr):
        raise ValueError("tokens and scores must have the same length")

    candidate = np.arange(len(tokens))
    if skip_special:
        candidate = np.array([i for i, tok in enumerate(tokens) if tok not in SPECIAL_TOKENS], dtype=int)
    if len(candidate) == 0:
        return {"top_indices": [], "top_tokens": [], "top_scores": []}

    k = max(1, min(k, len(candidate)))
    idx = candidate[np.argsort(scores_arr[candidate])[::-1][:k]]
    idx = idx[np.argsort(idx)]

    return {
        "top_indices": idx.tolist(),
        "top_tokens": [tokens[i] for i in idx],
        "top_scores": [float(scores_arr[i]) for i in idx],
    }
