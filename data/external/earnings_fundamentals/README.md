# Earnings Fundamentals Input

The pipeline will automatically read an optional event-level earnings file from the path configured by `earnings_events_path` in [configs/data/model_ready_mag7.yaml](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/configs/data/model_ready_mag7.yaml).

Minimum required columns:

- `ticker`
- `event_date`

Recommended key column:

- `call_id`

When `call_id` is present, the pipeline merges on `call_id`, `ticker`, and `event_date`; otherwise it falls back to `ticker` and `event_date`.

Supported optional columns:

- `reported_eps` or `actual_eps`
- `estimated_eps` or `consensus_eps`
- `eps_surprise`
- `eps_surprise_pct`
- `reported_revenue` or `actual_revenue`
- `estimated_revenue` or `consensus_revenue`
- `revenue_surprise`
- `revenue_surprise_pct`
- `reported_capex` or `actual_capex`
- `estimated_capex` or `consensus_capex`
- `capex_surprise`
- `capex_surprise_pct`

If the surprise columns are missing but reported and estimated values are present, the pipeline derives:

- absolute surprise
- surprise percentage
- `beat` / `meet` / `miss` labels

Recommended schema example:

```csv
ticker,event_date,reported_eps,estimated_eps,reported_revenue,estimated_revenue
AAPL,2024-10-31,1.64,1.60,94930,94500
```

SEC actual revenue and capex path:

```bash
python -m src.data.build_sec_event_snapshots \
  --events-input data/processed/tech_largecap_gold.parquet \
  --universe-input configs/data/universes/tech_largecap_nasdaq_2026-04-15.csv \
  --output-dir data/external/sec_company_facts
```

SEC Company Facts provides actual reported fundamentals, not analyst consensus estimates. Revenue and capex surprise fields are only populated when an external estimates source provides `estimated_revenue` or `estimated_capex`.

Current tech-largecap SEC actuals coverage:

- `reported_revenue`: `86.3%`
- `reported_capex`: `79.9%`
- `estimated_revenue`, `estimated_capex`, and their surprise fields: `0.0%` until an estimates source is added

Historical estimates path:

```bash
export FMP_API_KEY=...

python -m src.data.fetch_fmp_earnings_estimates \
  --events-input data/processed/tech_largecap_gold.parquet \
  --fundamentals-input data/external/sec_company_facts/sec_event_fundamentals_tech_largecap.csv \
  --output-dir data/external/earnings_fundamentals \
  --output-name earnings_estimates_fmp_tech_largecap.csv \
  --summary-name earnings_estimates_fmp_tech_largecap_summary.json
```

The FMP fetcher is the preferred current path for historical `estimated_eps` and `estimated_revenue`. It also computes an explicitly marked `estimated_capex` proxy from prior reported capex-to-revenue intensity multiplied by estimated revenue. That proxy is not analyst consensus capex; real consensus capex should replace it if a provider supplies that field.

Programmatic fetch path:

```bash
python -m src.data.fetch_yfinance_earnings_events \
  --metadata-input data/raw/metadata/call_metadata.csv \
  --output-dir data/external/earnings_fundamentals \
  --output-name earnings_events_yfinance_mag7.csv

python -m src.data.build_earnings_fundamentals \
  --metadata-input data/raw/metadata/call_metadata.csv \
  --raw-root data/raw/transcripts \
  --external-events-input data/external/earnings_fundamentals/earnings_events_yfinance_mag7.csv \
  --output-dir data/external/earnings_fundamentals
```

Current limitation:

- Yahoo can rate limit the historical earnings endpoint.
- The fetch script includes retry and backoff, but you may still need to rerun it later if Yahoo rejects requests during a given session.
