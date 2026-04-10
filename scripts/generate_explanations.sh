#!/usr/bin/env bash
set -euo pipefail

python -m src.experiments.explanation_benchmark --config configs/experiment/faithfulness_eval.yaml
