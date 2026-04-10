from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.sae.sae_model import SparseAutoencoder
from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def train_sae(
    activations: np.ndarray,
    latent_dim: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    l1_lambda: float,
    device: str,
) -> tuple[SparseAutoencoder, dict[str, float]]:
    x = torch.tensor(activations, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

    model = SparseAutoencoder(input_dim=x.size(1), latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"recon_loss": 0.0, "sparsity_loss": 0.0, "total_loss": 0.0}

    for _ in range(num_epochs):
        recon_sum = 0.0
        sparse_sum = 0.0
        total_sum = 0.0
        n = 0
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            recon = torch.mean((x_hat - batch) ** 2)
            sparse = torch.mean(torch.abs(z))
            loss = recon + l1_lambda * sparse

            opt.zero_grad()
            loss.backward()
            opt.step()

            bsz = batch.size(0)
            n += bsz
            recon_sum += recon.item() * bsz
            sparse_sum += sparse.item() * bsz
            total_sum += loss.item() * bsz

        history = {
            "recon_loss": recon_sum / max(n, 1),
            "sparsity_loss": sparse_sum / max(n, 1),
            "total_loss": total_sum / max(n, 1),
        }

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/sae_alignment.yaml")
    args = parser.parse_args()

    exp_cfg = load_yaml(args.config)
    sae_cfg = load_yaml(exp_cfg["sae_config"])

    act_path = Path(exp_cfg.get("activations_path", "outputs/sae/default_run/activations.npz"))
    arr = np.load(act_path)
    activations = arr["activations"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_sae(
        activations=activations,
        latent_dim=int(sae_cfg.get("latent_dim", 4096)),
        num_epochs=int(sae_cfg.get("num_epochs", 20)),
        batch_size=int(sae_cfg.get("batch_size", 256)),
        learning_rate=float(sae_cfg.get("learning_rate", 1e-3)),
        l1_lambda=float(sae_cfg.get("l1_lambda", 1e-3)),
        device=device,
    )

    output_dir = Path(exp_cfg.get("output_dir", "outputs/sae/default_run"))
    ensure_dir(output_dir)

    model_path = output_dir / "sae_model.pt"
    torch.save(model.state_dict(), model_path)

    feature_dict = model.decoder.weight.detach().cpu().numpy().T
    np.save(output_dir / "feature_dictionary.npy", feature_dict)
    save_json(history, output_dir / "train_metrics.json")
    LOGGER.info("Saved SAE artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
