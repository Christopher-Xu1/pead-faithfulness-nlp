# pead-faithfulness-nlp

## Overview
Evaluate whether common explanation methods faithfully identify the linguistic and latent features used by transformer models predicting PEAD from earnings call Q&A.

## Research Questions
- Do attention-based explanations reflect causally important evidence?
- Are gradient and perturbation methods more faithful?
- Do explanations align with latent SAE features driving predictions?

## Repository Structure
- `configs/`: experiment settings
- `data/`: raw, interim, and processed datasets
- `src/`: data, model, explanation, SAE, and evaluation code
- `scripts/`: executable pipelines
- `outputs/`: models, metrics, explanations, and figures
- `paper/`: manuscript assets

## Setup
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate pead-faithfulness-nlp
```

## Repository Hygiene
- Commit code, configs, tests, and lightweight documentation.
- Do not commit raw transcript corpora, generated Q&A tables, downloaded price files, or model checkpoints.
- Keep `data/` reproducible through pipeline scripts rather than storing text-bearing datasets in git.
- See [`data/README.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/data/README.md) for the data commit policy.

## Data Pipeline
```bash
bash scripts/run_data_pipeline.sh
```

## Transcript Corpus Pipeline
```bash
bash scripts/run_transcript_corpus_pipeline.sh
```

This broader ingest pipeline preserves raw transcript source files under `data/raw/transcripts/`, audits them, resolves duplicates, and materializes a cleaned master corpus plus a strict gold subset for research.

## Model-Ready Mag7 PEAD Pipeline
```bash
bash scripts/run_model_ready_mag7_pipeline.sh
```

This pipeline parses Q&A turns from the curated `mag7_gold` corpus, downloads real daily returns, computes PEAD labels, and writes `data/processed/dataset.csv` plus time-based splits.

## Current QA-Pair Status

Latest benchmark artifacts:
- [`reports/qa_pair_regression_report.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_regression_report.md)
- [`reports/qa_pair_regression_strict_report.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_regression_strict_report.md)
- [`reports/qa_pair_status.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_status.md)

Current corpus variants:
- Broad QA-pair corpus: `5746` pairs across `392` calls.
- Strict QA-pair corpus: `1879` pairs across `264` calls, with analyst-only answers and malformed answer spans removed.

Current best benchmark:
- Broad `text_plus_tabular`: AUROC `0.4880`, AUPRC `0.5727`.
- Strict `text_plus_tabular`: AUROC `0.5614`, AUPRC `0.6410`.
- Strict `text_only`: AUROC `0.4764`, AUPRC `0.5201`.

Current feature state:
- `pre_event_return_5d` is included in the tabular benchmark.
- `post_event_return_3d` is logged in the dataset but excluded from default training because it overlaps the PEAD target window.
- `glopardo` forward/trailing EPS proxies have `54.6%` coverage.
- True reported-vs-consensus EPS and revenue surprise fields are wired into the pipeline, but current local coverage is still `0%` until an event-level earnings estimates file is added.

## Train Model
```bash
bash scripts/train_baseline.sh
```

Equivalent direct command:
```bash
python -m src.models.train --config configs/experiment/baseline_pead.yaml
```

The default baseline config trains `ProsusAI/finbert` on:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## Evaluate Baseline
```bash
python -m src.models.evaluate --config configs/experiment/baseline_pead.yaml
```

## Generate Explanations
```bash
bash scripts/generate_explanations.sh
```

## Evaluate Faithfulness
```bash
bash scripts/run_faithfulness_eval.sh
```

## Train SAE
```bash
bash scripts/train_sae.sh
```

## Reproducibility Checklist
Always log:
- random seed
- config file used
- train/val/test time ranges
- PEAD horizon
- rationale size

## Suggested First Commit
Before training, the clean commit should include:
- `src/`
- `configs/`
- `scripts/`
- `tests/`
- root setup files such as `README.md`, `requirements.txt`, `environment.yml`, `setup.py`, `pytest.ini`
- lightweight documentation such as `data/README.md` and `data/processed/dataset_card.md`

It should exclude generated data artifacts under `data/raw/`, `data/interim/`, `data/external/`, model outputs under `outputs/`, and any downloaded transcript text.
