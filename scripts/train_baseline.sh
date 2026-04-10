#!/usr/bin/env bash
set -euo pipefail

python -m src.models.train --config configs/experiment/baseline_pead.yaml
python -m src.models.evaluate --config configs/experiment/baseline_pead.yaml
