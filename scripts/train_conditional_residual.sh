#!/usr/bin/env bash
set -euo pipefail

python -m src.experiments.conditional_residual_qa_pead --config configs/experiment/conditional_residual_qa_pead.yaml

