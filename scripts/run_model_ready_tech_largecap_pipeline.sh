#!/usr/bin/env bash
set -euo pipefail

python -m src.data.build_universe_subset \
  --corpus-input data/processed/gold_corpus.parquet \
  --processed-output data/processed/tech_largecap_gold.parquet

python -m src.data.build_sec_event_snapshots \
  --events-input data/processed/tech_largecap_gold.parquet \
  --universe-input configs/data/universes/tech_largecap_nasdaq_2026-04-15.csv \
  --output-dir data/external/sec_company_facts

python -m src.data.fetch_hf_earnings_surprise \
  --events-input data/processed/tech_largecap_gold.parquet \
  --output-dir data/external/earnings_fundamentals \
  --output-name earnings_surprise_hf_sovai_tech_largecap.csv \
  --summary-name earnings_surprise_hf_sovai_tech_largecap_summary.json

FMP_KEY="${FMP_API_KEY:-${FINANCIALMODELINGPREP_API_KEY:-}}"
if [[ -n "${FMP_KEY}" ]]; then
  python -m src.data.fetch_fmp_earnings_estimates \
    --events-input data/processed/tech_largecap_gold.parquet \
    --fundamentals-input data/external/sec_company_facts/sec_event_fundamentals_tech_largecap.csv \
    --output-dir data/external/earnings_fundamentals \
    --output-name earnings_estimates_fmp_tech_largecap.csv \
    --summary-name earnings_estimates_fmp_tech_largecap_summary.json
else
  echo "Skipping FMP earnings estimates because FMP_API_KEY is not set."
fi

python -m src.data.build_model_ready_pead \
  --config configs/data/model_ready_tech_largecap.yaml

python -m src.data.build_qa_pair_dataset \
  --parsed-input data/interim/tech_largecap/parsed_calls.csv \
  --qa-summary-input data/interim/tech_largecap/qa_dataset.csv \
  --metadata-input data/raw/metadata/call_metadata_tech_largecap.csv \
  --labels-input data/interim/tech_largecap/pead_labels.csv \
  --prices-input data/raw/prices/daily_returns_tech_largecap.csv \
  --market-input data/external/market_index/sp500_returns_tech_largecap.csv \
  --earnings-fundamentals-input data/external/earnings_fundamentals/earnings_fundamentals_tech_largecap.csv \
  --label-config configs/data/pead_20d.yaml \
  --output-dir outputs/datasets/qa_pairs_tech_largecap_strict \
  --cleaning-profile strict
