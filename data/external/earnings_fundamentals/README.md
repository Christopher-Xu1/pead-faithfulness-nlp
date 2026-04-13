# Earnings Fundamentals Input

The pipeline will automatically read an optional event-level earnings file from the path configured by `earnings_events_path` in [configs/data/model_ready_mag7.yaml](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/configs/data/model_ready_mag7.yaml).

Minimum required columns:

- `ticker`
- `event_date`

Supported optional columns:

- `reported_eps` or `actual_eps`
- `estimated_eps` or `consensus_eps`
- `eps_surprise`
- `eps_surprise_pct`
- `reported_revenue` or `actual_revenue`
- `estimated_revenue` or `consensus_revenue`
- `revenue_surprise`
- `revenue_surprise_pct`

If the surprise columns are missing but reported and estimated values are present, the pipeline derives:

- absolute surprise
- surprise percentage
- `beat` / `meet` / `miss` labels

Recommended schema example:

```csv
ticker,event_date,reported_eps,estimated_eps,reported_revenue,estimated_revenue
AAPL,2024-10-31,1.64,1.60,94930,94500
```
