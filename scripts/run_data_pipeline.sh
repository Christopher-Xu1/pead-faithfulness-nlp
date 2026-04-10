#!/usr/bin/env bash
set -euo pipefail

python -m src.data.ingest --config configs/data/mag7.yaml
python -m src.data.parse_transcripts
python -m src.data.extract_qa
python -m src.data.compute_pead --label-config configs/data/pead_20d.yaml
python -m src.data.build_dataset
python -m src.data.split_dataset --seed 42
