# SEC Company Facts Event Snapshots

Generated files in this directory are not committed. Rebuild them with:

```bash
python -m src.data.build_sec_event_snapshots \
  --events-input data/processed/tech_largecap_gold.parquet \
  --universe-input configs/data/universes/tech_largecap_nasdaq_2026-04-15.csv \
  --output-dir data/external/sec_company_facts
```

Outputs:

- `sec_event_fundamentals_tech_largecap.csv`: SEC actual reported revenue and capex fields aligned to each call date.
- `market_cap_snapshots_tech_largecap.csv`: pre-event historical market-cap snapshots using Yahoo pre-event close and SEC shares outstanding facts available before the event.
- `sec_event_snapshots_summary.json`: coverage summary for the generated snapshot files.

Current tech-largecap coverage:

- Historical market cap: `79.7%`
- SEC reported revenue actual: `86.3%`
- SEC reported capex actual: `79.9%`
- Pre-event close availability: `98.4%`

Cleanliness notes:

- Historical market cap uses the latest trading close strictly before the earnings-call date.
- Yahoo chart closes are split-adjusted, so the builder reconstructs split-unadjusted closes before multiplying by as-reported SEC shares outstanding.
- Shares outstanding use the latest SEC `dei:EntityCommonStockSharesOutstanding` fact with both fact end date and filing date on or before the call date, with broad share-count outlier filtering for XBRL scale errors.
- Revenue and capex actuals use quarterly SEC XBRL facts whose period end is before the call date and within the configured lag window.
- SEC does not provide analyst consensus estimates, so revenue/capex surprise fields require a separate estimates source.
