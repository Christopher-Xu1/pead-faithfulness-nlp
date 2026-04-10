from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.io import ensure_dir, load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def extract_hidden_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_index: int,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    pooled = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_index]  # [b, seq, hidden]
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled_hs = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled.append(pooled_hs.cpu().numpy())
    return np.concatenate(pooled, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/sae_alignment.yaml")
    parser.add_argument("--data-path", default="data/processed/train.csv")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.config)
    sae_cfg = load_yaml(exp_cfg["sae_config"])
    model_dir = exp_cfg["model_dir"]
    output_dir = Path(exp_cfg.get("output_dir", "outputs/sae/default_run"))

    df = pd.read_csv(args.data_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    activations = extract_hidden_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_index=int(sae_cfg.get("layer_index", 8)),
        batch_size=int(sae_cfg.get("batch_size", 16)),
        max_length=int(load_yaml("configs/model/finbert.yaml").get("max_length", 512)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ensure_dir(output_dir)
    out_path = Path(exp_cfg.get("activations_path", output_dir / "activations.npz"))
    np.savez(out_path, activations=activations, labels=labels)
    LOGGER.info("Saved activations to %s", out_path)


if __name__ == "__main__":
    main()
