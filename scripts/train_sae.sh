#!/usr/bin/env bash
set -euo pipefail

python -m src.sae.extract_activations --config configs/experiment/sae_alignment.yaml
python -m src.sae.train_sae --config configs/experiment/sae_alignment.yaml
python -m src.experiments.sae_grounded_eval --config configs/experiment/sae_alignment.yaml
