from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.sae.sae_model import SparseAutoencoder
from src.utils.io import load_yaml, save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def ablate_features(z: torch.Tensor, feature_indices: list[int]) -> torch.Tensor:
    z_ab = z.clone()
    z_ab[:, feature_indices] = 0.0
    return z_ab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/sae_alignment.yaml")
    parser.add_argument("--num-features", type=int, default=5)
    args = parser.parse_args()

    exp_cfg = load_yaml(args.config)
    sae_cfg = load_yaml(exp_cfg["sae_config"])
    output_dir = Path(exp_cfg.get("output_dir", "outputs/sae/default_run"))

    arr = np.load(Path(exp_cfg.get("activations_path", output_dir / "activations.npz")))
    x = arr["activations"]

    model = SparseAutoencoder(input_dim=x.shape[1], latent_dim=int(sae_cfg.get("latent_dim", 4096)))
    model.load_state_dict(torch.load(output_dir / "sae_model.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32)
        _, z = model(x_t)
        feature_strength = z.mean(dim=0)
        top = torch.argsort(feature_strength, descending=True)[: args.num_features].tolist()

        x_hat = model.decode(z)
        mse_base = torch.mean((x_hat - x_t) ** 2).item()

        z_ab = ablate_features(z, top)
        x_hat_ab = model.decode(z_ab)
        mse_ab = torch.mean((x_hat_ab - x_t) ** 2).item()

    result = {
        "ablated_features": [int(i) for i in top],
        "reconstruction_mse_base": float(mse_base),
        "reconstruction_mse_ablated": float(mse_ab),
        "mse_degradation": float(mse_ab - mse_base),
    }
    save_json(result, output_dir / "feature_ablation.json")
    LOGGER.info("Saved feature ablation report to %s", output_dir / "feature_ablation.json")


if __name__ == "__main__":
    main()
