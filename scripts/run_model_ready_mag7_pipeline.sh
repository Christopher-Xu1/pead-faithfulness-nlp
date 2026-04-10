#!/usr/bin/env bash
set -euo pipefail

python -m src.data.build_model_ready_pead --config configs/data/model_ready_mag7.yaml
