from __future__ import annotations

import torch


def _target_prob(model, x: dict[str, torch.Tensor], target_class: int) -> torch.Tensor:
    logits = model(**x).logits
    return torch.softmax(logits, dim=-1)[:, target_class]


def comprehensiveness(model, x: dict[str, torch.Tensor], rationale_tokens: list[int], target_class: int) -> float:
    """Comp(R) = f(x) - f(x \\ R)."""
    model.eval()
    with torch.no_grad():
        full = _target_prob(model, x, target_class)

        x_minus_r = {k: v.clone() for k, v in x.items()}
        if "attention_mask" in x_minus_r and rationale_tokens:
            x_minus_r["attention_mask"][:, rationale_tokens] = 0

        reduced = _target_prob(model, x_minus_r, target_class)
        return float((full - reduced).item())
