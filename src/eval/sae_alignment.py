from __future__ import annotations

import numpy as np


def alignment_overlap(rationale_indices: list[int], top_feature_token_indices: list[int]) -> float:
    r = set(rationale_indices)
    t = set(top_feature_token_indices)
    if not r:
        return 0.0
    return len(r & t) / len(r)


def alignment_cosine(expl_scores: np.ndarray, sae_scores: np.ndarray) -> float:
    expl = np.asarray(expl_scores, dtype=float)
    sae = np.asarray(sae_scores, dtype=float)
    if expl.shape != sae.shape:
        raise ValueError("expl_scores and sae_scores must have same shape")
    den = np.linalg.norm(expl) * np.linalg.norm(sae)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(expl, sae) / den)


def simple_alignment_score(
    rationale_indices: list[int],
    top_feature_token_indices: list[int],
    expl_scores: np.ndarray | None = None,
    sae_scores: np.ndarray | None = None,
) -> dict[str, float]:
    out = {"overlap": alignment_overlap(rationale_indices, top_feature_token_indices)}
    if expl_scores is not None and sae_scores is not None:
        out["cosine"] = alignment_cosine(expl_scores, sae_scores)
    return out
