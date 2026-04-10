from __future__ import annotations

import numpy as np
import torch


def explain_with_attention(
    model,
    tokenizer,
    text: str,
    target_class: int = 1,
    layer: int = -1,
    device: str = "cpu",
) -> dict[str, list]:
    del target_class
    model = model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    attentions = out.attentions[layer]  # [batch, heads, seq, seq]
    cls_to_tokens = attentions[0, :, 0, :]
    scores = cls_to_tokens.mean(dim=0).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    scores = _normalize(scores)
    return {"tokens": tokens, "scores": scores.tolist()}


def _normalize(scores: np.ndarray) -> np.ndarray:
    s = scores.astype(float)
    den = s.max() - s.min()
    if den <= 1e-12:
        return np.zeros_like(s)
    return (s - s.min()) / den
