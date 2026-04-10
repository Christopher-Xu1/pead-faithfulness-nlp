from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def build_finbert_classifier(model_name: str = "ProsusAI/finbert", num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
