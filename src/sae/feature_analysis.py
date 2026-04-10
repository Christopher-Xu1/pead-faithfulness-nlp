from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.sae.sae_model import SparseAutoencoder
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def analyze_features(latents: np.ndarray, labels: np.ndarray | None = None, top_n: int = 20) -> pd.DataFrame:
    mean_act = latents.mean(axis=0)
    rows = []
    for i in np.argsort(mean_act)[::-1][:top_n]:
        row = {"feature_idx": int(i), "mean_activation": float(mean_act[i])}
        if labels is not None and len(np.unique(labels)) > 1:
            corr = np.corrcoef(latents[:, i], labels)[0, 1]
            row["label_correlation"] = float(np.nan_to_num(corr))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/sae_alignment.yaml")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    exp_cfg = load_yaml(args.config)
    sae_cfg = load_yaml(exp_cfg["sae_config"])
    output_dir = Path(exp_cfg.get("output_dir", "outputs/sae/default_run"))

    arr = np.load(Path(exp_cfg.get("activations_path", output_dir / "activations.npz")))
    x = arr["activations"]
    labels = arr["labels"] if "labels" in arr.files else None

    model = SparseAutoencoder(input_dim=x.shape[1], latent_dim=int(sae_cfg.get("latent_dim", 4096)))
    model.load_state_dict(torch.load(output_dir / "sae_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        z = model.encode(torch.tensor(x, dtype=torch.float32)).numpy()

    report = analyze_features(z, labels=labels, top_n=args.top_n)
    report.to_csv(output_dir / "feature_report.csv", index=False)
    LOGGER.info("Saved feature analysis to %s", output_dir / "feature_report.csv")


if __name__ == "__main__":
    main()
