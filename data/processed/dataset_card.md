# Dataset Card

## Summary
Model-ready Mag7 earnings-call Q&A dataset for PEAD classification.

- Source corpus: `data/processed/mag7_gold.parquet`
- Model-ready table: `data/processed/dataset.csv`
- Rows: `401`
- Tickers: `AAPL`, `AMZN`, `GOOGL`, `META`, `MSFT`, `NVDA`, `TSLA`
- Event date range: `2006-01-19` to `2025-05-01`
- Positive labels: `205`
- Negative labels: `196`

## Upstream Source Mix
- `glopardo_sp500_earnings_transcripts`: `229`
- `bose345_sp500_earnings_transcripts`: `132`
- `jlh_ibm_earnings_call`: `40`

## Construction Notes
- Started from `410` quality-filtered Mag7 call transcripts.
- Parsed Q&A successfully for `401` calls.
- Retained calls require at least one analyst question turn and a computed PEAD label.
- Labels use one-day event lag and 20-trading-day cumulative abnormal return.

## Fields
- `call_id`: Stable call identifier from the curated transcript corpus.
- `ticker`: Canonical equity ticker.
- `event_date`: Earnings-call date.
- `text`: Analyst-only Q&A text used as the primary model input.
- `num_questions`: Estimated number of analyst questions in `text`.
- `full_qa_text`: Full Q&A section including management responses.
- `num_qa_turns`: Number of turns in the full Q&A section.
- `source_id`: Upstream transcript source.
- `company`: Company name from transcript metadata.
- `year`: Reported calendar year.
- `quarter`: Reported fiscal quarter when available.
- `quality_score`: Corpus-level transcript quality score from curation.
- `soft_quality_flags`: Non-fatal quality warnings from curation.
- `car_horizon`: 20-day cumulative abnormal return after the configured event lag.
- `label`: Binary PEAD label where `1` indicates `car_horizon > 0.0`.

## Splits
Time-based splits are stored in `train.csv`, `val.csv`, and `test.csv`.

- Train: `280` rows, `138` positive, `2006-01-19` to `2019-10-23`
- Val: `60` rows, `36` positive, `2019-10-24` to `2022-02-03`
- Test: `61` rows, `31` positive, `2022-02-16` to `2025-05-01`
