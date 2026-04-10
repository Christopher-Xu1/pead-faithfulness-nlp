from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.explain.attention import explain_with_attention
from src.explain.explanation_utils import save_explanations_csv, save_explanations_json
from src.explain.integrated_gradients import explain_with_ig
from src.explain.perturbation import explain_with_perturbation
from src.explain.rationale_builder import build_topk_rationale
from src.utils.io import ensure_dir, load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _predict_class(model, tokenizer, text: str, device: str) -> int:
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return int(torch.argmax(logits, dim=-1).item())


def run_explanation_benchmark(config_path: str) -> None:
    cfg = load_yaml(config_path)
    methods = cfg.get("methods", ["attention", "integrated_gradients", "perturbation"])
    k = int(cfg.get("rationale_k", 30))

    model_dir = cfg["model_dir"]
    data_path = cfg["data_path"]
    run_name = cfg.get("run_name", "default_explain")

    df = pd.read_csv(data_path)
    max_examples = int(cfg.get("max_examples", len(df)))
    df = df.head(max_examples).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    records = []
    for row in df.itertuples(index=False):
        text = str(row.text)
        target_class = _predict_class(model, tokenizer, text, device=device)

        for method in methods:
            if method == "attention":
                exp = explain_with_attention(model, tokenizer, text, target_class=target_class, device=device)
            elif method in {"integrated_gradients", "ig"}:
                exp = explain_with_ig(model, tokenizer, text, target_class=target_class, device=device)
            elif method == "perturbation":
                exp = explain_with_perturbation(model, tokenizer, text, target_class=target_class, device=device)
            else:
                raise ValueError(f"Unsupported explanation method: {method}")

            rat = build_topk_rationale(exp["tokens"], exp["scores"], k=k)
            records.append(
                {
                    "call_id": getattr(row, "call_id", None),
                    "ticker": getattr(row, "ticker", None),
                    "event_date": getattr(row, "event_date", None),
                    "method": method,
                    "target_class": target_class,
                    "tokens": json.dumps(exp["tokens"]),
                    "scores": json.dumps([float(x) for x in exp["scores"]]),
                    "top_indices": json.dumps(rat["top_indices"]),
                    "top_tokens": json.dumps(rat["top_tokens"]),
                    "top_scores": json.dumps([float(x) for x in rat["top_scores"]]),
                }
            )

    out_dir = Path("outputs/explanations") / run_name
    ensure_dir(out_dir)
    save_explanations_json(records, str(out_dir / "explanations.json"))
    save_explanations_csv(records, str(out_dir / "explanations.csv"))
    LOGGER.info("Saved explanations for %d records to %s", len(records), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/faithfulness_eval.yaml")
    args = parser.parse_args()
    run_explanation_benchmark(args.config)


if __name__ == "__main__":
    main()
