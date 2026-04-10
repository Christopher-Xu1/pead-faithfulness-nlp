#!/usr/bin/env bash
set -euo pipefail

python -m src.data.ingest_transcripts --config configs/data/transcript_curation.yaml
python -m src.data.audit_transcripts --config configs/data/transcript_curation.yaml
python -m src.data.curate_transcripts --config configs/data/transcript_curation.yaml
