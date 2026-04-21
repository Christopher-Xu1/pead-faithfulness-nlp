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

## Model-Ready Tech Large-Cap PEAD Pipeline
```bash
bash scripts/run_model_ready_tech_largecap_pipeline.sh
```

This pipeline builds a dated large-cap tech snapshot using S&P GICS `Information Technology`, Nasdaq market caps above `$10B`, and forced Mag7 inclusion. It also rebuilds SEC Company Facts event snapshots, writes separate model-ready tech-largecap artifacts, and refreshes the strict QA-pair training corpus without overwriting the Mag7 benchmark files.

## Current QA-Pair Status

Latest benchmark artifacts:
- [`reports/qa_pair_regression_report.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_regression_report.md)
- [`reports/qa_pair_regression_strict_report.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_regression_strict_report.md)
- [`reports/qa_pair_regression_tech_largecap_strict_eps_quick_report.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_regression_tech_largecap_strict_eps_quick_report.md)
- [`reports/qa_pair_regression_tech_largecap_strict_eps_fast20_report.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_regression_tech_largecap_strict_eps_fast20_report.md)
- [`reports/qa_pair_status.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/reports/qa_pair_status.md)

Current corpus variants:
- Broad QA-pair corpus: `5746` pairs across `392` calls.
- Strict QA-pair corpus: `1879` pairs across `264` calls, with analyst-only answers and malformed answer spans removed.
- Tech-largecap strict QA-pair corpus: `15683` pairs across `1760` calls with at least one strict pair.

Current best benchmark:
- Broad `text_plus_tabular`: AUROC `0.4880`, AUPRC `0.5727`.
- Strict `text_plus_tabular`: AUROC `0.5614`, AUPRC `0.6410`.
- Strict `text_only`: AUROC `0.4764`, AUPRC `0.5201`.
- Tech-largecap strict fast 20-fold benchmark with partial EPS-surprise backfill: best AUROC `0.5307` and best AUPRC `0.6238` from boosted rich aggregation. The earlier one-fold quick result did not hold across the full rolling evaluation.

Current feature state:
- `pre_event_return_5d` is included in the tabular benchmark.
- `post_event_return_3d` is logged in the dataset but excluded from default training because it overlaps the PEAD target window.
- Expanded tech-largecap runs include frozen-universe market-cap controls and time-safe prior call-frequency controls.
- Expanded tech-largecap runs now include clean historical market-cap snapshots with `79.7%` strict-call coverage, computed from pre-event Yahoo closes and SEC shares outstanding facts available before the call.
- SEC Company Facts actuals provide `86.3%` reported revenue coverage and `79.9%` reported capex coverage on the expanded strict call table.
- Public Hugging Face `sovai/earnings_surprise` backfills reported EPS, estimated EPS, and EPS surprise for `50.4%` of expanded strict calls.
- `glopardo` forward/trailing EPS proxies have `69.4%` coverage on the expanded tech-largecap universe.
- Revenue surprise coverage is still `0.0%` without a credentialed estimates source. Historical EPS/revenue estimates can be fetched with `src.data.fetch_fmp_earnings_estimates` when `FMP_API_KEY` is set; capex is supported as either real provider consensus or an explicitly marked prior-capex-intensity proxy.

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
