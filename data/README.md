## Data Policy

This repository does not commit raw transcript corpora, downloaded price series, parsed Q&A tables, or trained model artifacts.

Committed data files are limited to:
- placeholder `.gitkeep` files that preserve the directory layout
- lightweight documentation such as [`dataset_card.md`](./processed/dataset_card.md)

Generated artifacts are rebuilt locally with the pipeline scripts:
- `bash scripts/run_transcript_corpus_pipeline.sh`
- `bash scripts/run_model_ready_mag7_pipeline.sh`

Do not publish raw or merged transcript text files from this repository until upstream source licensing and redistribution terms are fully reviewed.
