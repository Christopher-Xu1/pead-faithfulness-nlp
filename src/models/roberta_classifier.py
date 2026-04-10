from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def build_roberta_classifier(model_name: str = "roberta-base", num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
