from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.metrics import softmax


def predict_texts(model_dir: str, texts: Iterable[str], max_length: int = 512) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    texts = list(texts)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
    return softmax(logits)[:, 1]
