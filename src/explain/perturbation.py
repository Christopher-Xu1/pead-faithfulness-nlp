from __future__ import annotations

import numpy as np
import torch


def explain_with_perturbation(
    model,
    tokenizer,
    text: str,
    target_class: int,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, list]:
    model = model.to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    with torch.no_grad():
        base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        base_prob = torch.softmax(base_logits, dim=-1)[0, target_class].item()

    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.pad_token_id

    seq_len = input_ids.size(1)
    candidate_ids = []
    candidate_masks = []
    indices = []

    special_ids = set(tokenizer.all_special_ids)
    for idx in range(seq_len):
        token_id = int(input_ids[0, idx].item())
        if token_id in special_ids:
            continue
        perturbed_ids = input_ids.clone()
        perturbed_ids[0, idx] = mask_id
        candidate_ids.append(perturbed_ids)
        candidate_masks.append(attention_mask)
        indices.append(idx)

    scores = np.zeros(seq_len, dtype=float)
    if candidate_ids:
        all_ids = torch.cat(candidate_ids, dim=0)
        all_masks = torch.cat(candidate_masks, dim=0)
        probs = []
        with torch.no_grad():
            for start in range(0, all_ids.size(0), batch_size):
                end = start + batch_size
                logits = model(input_ids=all_ids[start:end], attention_mask=all_masks[start:end]).logits
                probs.append(torch.softmax(logits, dim=-1)[:, target_class].cpu())
        perturbed_probs = torch.cat(probs).numpy()
        deltas = base_prob - perturbed_probs
        for i, token_idx in enumerate(indices):
            scores[token_idx] = max(0.0, float(deltas[i]))

    scores = _normalize(scores)
    return {"tokens": tokens, "scores": scores.tolist()}


def _normalize(scores: np.ndarray) -> np.ndarray:
    s = scores.astype(float)
    den = s.max() - s.min()
    if den <= 1e-12:
        return np.zeros_like(s)
    return (s - s.min()) / den
