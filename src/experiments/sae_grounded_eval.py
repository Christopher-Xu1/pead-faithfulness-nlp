from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.sae_alignment import simple_alignment_score
from src.sae.feature_ablation import main as run_feature_ablation
from src.sae.feature_analysis import main as run_feature_analysis
from src.sae.sae_model import SparseAutoencoder
from src.utils.io import load_yaml, save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_alignment(config_path: str) -> None:
    exp_cfg = load_yaml(config_path)
    sae_cfg = load_yaml(exp_cfg["sae_config"])
    out_dir = Path(exp_cfg.get("output_dir", "outputs/sae/default_run"))

    act_path = Path(exp_cfg.get("activations_path", out_dir / "activations.npz"))
    arr = np.load(act_path)
    x = arr["activations"]
    labels = arr["labels"] if "labels" in arr.files else None

    model = SparseAutoencoder(input_dim=x.shape[1], latent_dim=int(sae_cfg.get("latent_dim", 4096)))
    model.load_state_dict(torch.load(out_dir / "sae_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        z = model.encode(torch.tensor(x, dtype=torch.float32)).numpy()

    alignment_payload = {"note": "sample-level proxy alignment in v1"}

    if labels is not None and len(np.unique(labels)) > 1:
        corrs = np.array([abs(_safe_corr(z[:, i], labels)) for i in range(z.shape[1])])
        top_feat = np.argsort(corrs)[::-1][:20]
        decision_strength = z[:, top_feat].mean(axis=1)
        label_alignment = _safe_corr(decision_strength, labels)
        alignment_payload["label_alignment_corr"] = float(label_alignment)

    exp_csv = Path("outputs/explanations") / exp_cfg.get("run_name", "default_eval") / "explanations.csv"
    if exp_csv.exists():
        exp_df = pd.read_csv(exp_csv)
        if "top_indices" in exp_df.columns:
            overlaps = []
            for _, row in exp_df.head(100).iterrows():
                try:
                    idx = json.loads(row["top_indices"])
                except Exception:
                    idx = []
                pseudo_top = list(range(min(len(idx), 10)))
                overlaps.append(simple_alignment_score(idx, pseudo_top)["overlap"])
            if overlaps:
                alignment_payload["token_overlap_proxy"] = float(np.mean(overlaps))

    save_json(alignment_payload, out_dir / "sae_alignment.json")
    LOGGER.info("Saved SAE alignment proxy metrics to %s", out_dir / "sae_alignment.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/sae_alignment.yaml")
    args = parser.parse_args()

    run_feature_analysis()
    run_feature_ablation()
    run_alignment(args.config)


if __name__ == "__main__":
    main()
