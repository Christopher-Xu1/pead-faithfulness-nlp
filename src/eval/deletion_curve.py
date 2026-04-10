from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def deletion_curve(
    model,
    x: dict[str, torch.Tensor],
    token_scores: list[float],
    target_class: int,
    steps: int = 10,
) -> pd.DataFrame:
    model.eval()

    scores = np.asarray(token_scores, dtype=float)
    order = np.argsort(scores)[::-1]
    seq_len = len(order)

    rows = []
    with torch.no_grad():
        for step in range(steps + 1):
            frac = step / steps
            k = int(round(frac * seq_len))
            remove_idx = order[:k]

            x_del = {k_: v.clone() for k_, v in x.items()}
            if "attention_mask" in x_del and len(remove_idx) > 0:
                x_del["attention_mask"][:, remove_idx] = 0

            prob = torch.softmax(model(**x_del).logits, dim=-1)[:, target_class].item()
            rows.append({"fraction_removed": frac, "score": float(prob)})

    return pd.DataFrame(rows)
