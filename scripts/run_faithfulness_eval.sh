#!/usr/bin/env bash
set -euo pipefail

python -m src.experiments.baseline_pead --config configs/experiment/faithfulness_eval.yaml
