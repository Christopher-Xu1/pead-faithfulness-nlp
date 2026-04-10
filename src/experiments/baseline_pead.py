from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.eval.comprehensiveness import comprehensiveness
from src.eval.deletion_curve import deletion_curve
from src.eval.random_baselines import random_baseline_scores
from src.eval.sufficiency import sufficiency
from src.explain.attention import explain_with_attention
from src.explain.integrated_gradients import explain_with_ig
from src.explain.perturbation import explain_with_perturbation
from src.explain.rationale_builder import build_topk_rationale
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger
from src.utils.plotting import save_deletion_curve_plot

LOGGER = get_logger(__name__)


def _predict_class(model, tokenizer, text: str, device: str) -> int:
    enc = tokenizer(text, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return int(torch.argmax(logits, dim=-1).item())


def _explain(method: str, model, tokenizer, text: str, target_class: int, device: str) -> dict[str, list]:
    if method == "attention":
        return explain_with_attention(model, tokenizer, text, target_class=target_class, device=device)
    if method in {"integrated_gradients", "ig"}:
        return explain_with_ig(model, tokenizer, text, target_class=target_class, device=device)
    if method == "perturbation":
        return explain_with_perturbation(model, tokenizer, text, target_class=target_class, device=device)
    raise ValueError(f"Unsupported explanation method: {method}")


def run_faithfulness_eval(config_path: str) -> None:
    cfg = load_yaml(config_path)
    methods = cfg.get("methods", ["attention", "integrated_gradients", "perturbation"])
    k = int(cfg.get("rationale_k", 30))

    model_dir = cfg["model_dir"]
    data_path = cfg["data_path"]
    run_name = cfg.get("run_name", "default_eval")
    max_examples = int(cfg.get("max_examples", 50))

    df = pd.read_csv(data_path).head(max_examples).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    metrics_rows = []
    curve_rows = []

    for row in df.itertuples(index=False):
        text = str(row.text)
        target_class = _predict_class(model, tokenizer, text, device=device)

        enc = tokenizer(text, return_tensors="pt", truncation=True)
        x = {k_: v.to(device) for k_, v in enc.items()}

        for method in methods:
            exp = _explain(method, model, tokenizer, text, target_class=target_class, device=device)
            rationale = build_topk_rationale(exp["tokens"], exp["scores"], k=k)
            r_idx = rationale["top_indices"]

            comp = comprehensiveness(model, x, r_idx, target_class)
            suff = sufficiency(model, x, r_idx, target_class)

            curve_df = deletion_curve(model, x, exp["scores"], target_class=target_class, steps=10)
            auc = float(np.trapz(curve_df["score"].values, curve_df["fraction_removed"].values))
            curve_df["call_id"] = getattr(row, "call_id", None)
            curve_df["method"] = method
            curve_rows.append(curve_df)

            n_tokens = len(exp["tokens"])
            k_eff = min(k, n_tokens)
            rb_comp = random_baseline_scores(
                metric_fn=lambda idx: comprehensiveness(model, x, idx, target_class),
                n_tokens=n_tokens,
                k=k_eff,
                n_trials=25,
                seed=42,
            )
            rb_suff = random_baseline_scores(
                metric_fn=lambda idx: sufficiency(model, x, idx, target_class),
                n_tokens=n_tokens,
                k=k_eff,
                n_trials=25,
                seed=43,
            )

            metrics_rows.append(
                {
                    "call_id": getattr(row, "call_id", None),
                    "ticker": getattr(row, "ticker", None),
                    "event_date": getattr(row, "event_date", None),
                    "method": method,
                    "target_class": target_class,
                    "k": k_eff,
                    "comprehensiveness": comp,
                    "sufficiency": suff,
                    "deletion_auc": auc,
                    "random_comp_mean": rb_comp["random_mean"],
                    "random_suff_mean": rb_suff["random_mean"],
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    out_dir = Path(cfg.get("output_dir", f"outputs/metrics/{run_name}"))
    ensure_dir(out_dir)

    metrics_df.to_csv(out_dir / "faithfulness_metrics.csv", index=False)
    curve_df.to_csv(out_dir / "deletion_curves.csv", index=False)

    if not curve_df.empty:
        mean_curve = (
            curve_df.groupby(["method", "fraction_removed"], as_index=False)["score"].mean().sort_values(["method", "fraction_removed"])
        )
        for method, mdf in mean_curve.groupby("method"):
            save_deletion_curve_plot(mdf, str(out_dir / f"deletion_curve_{method}.png"))

    summary = (
        metrics_df.groupby("method")[["comprehensiveness", "sufficiency", "deletion_auc", "random_comp_mean", "random_suff_mean"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    save_json(summary, out_dir / "summary.json")

    LOGGER.info("Saved faithfulness metrics to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/faithfulness_eval.yaml")
    args = parser.parse_args()
    run_faithfulness_eval(args.config)


if __name__ == "__main__":
    main()
