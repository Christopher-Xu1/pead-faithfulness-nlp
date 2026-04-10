from __future__ import annotations

import numpy as np
import torch


def explain_with_ig(
    model,
    tokenizer,
    text: str,
    target_class: int,
    device: str = "cpu",
    steps: int = 32,
) -> dict[str, list]:
    try:
        from captum.attr import LayerIntegratedGradients
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "captum is required for integrated gradients. Install with `pip install captum`."
        ) from exc

    model = model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    def forward_func(ids, mask, cls_idx):
        logits = model(input_ids=ids, attention_mask=mask).logits
        return logits[:, cls_idx]

    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),
        additional_forward_args=(attention_mask, target_class),
        n_steps=steps,
    )

    token_scores = attributions.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    token_scores = _normalize(token_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    return {"tokens": tokens, "scores": token_scores.tolist()}


def _normalize(scores: np.ndarray) -> np.ndarray:
    s = scores.astype(float)
    den = s.max() - s.min()
    if den <= 1e-12:
        return np.zeros_like(s)
    return (s - s.min()) / den
