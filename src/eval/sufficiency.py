from __future__ import annotations

import torch


def _target_prob(model, x: dict[str, torch.Tensor], target_class: int) -> torch.Tensor:
    logits = model(**x).logits
    return torch.softmax(logits, dim=-1)[:, target_class]


def sufficiency(model, x: dict[str, torch.Tensor], rationale_tokens: list[int], target_class: int) -> float:
    """Suff(R) = f(x) - f(x_R)"""
    model.eval()
    with torch.no_grad():
        full = _target_prob(model, x, target_class)

        x_r = {k: v.clone() for k, v in x.items()}
        if "attention_mask" in x_r:
            keep_mask = torch.zeros_like(x_r["attention_mask"])
            if rationale_tokens:
                keep_mask[:, rationale_tokens] = 1
            x_r["attention_mask"] = keep_mask

        reduced = _target_prob(model, x_r, target_class)
        return float((full - reduced).item())
