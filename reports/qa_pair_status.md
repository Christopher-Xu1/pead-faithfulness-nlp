# QA-Pair Status

## Scope

This note captures the current state of the QA-pair PEAD benchmark after:
- strict QA-pair cleanup
- call-level feature refresh
- broad vs strict benchmark comparison

## Corpus Versions

| Corpus | Pair Rows | Calls | Notes |
| --- | ---: | ---: | --- |
| `broad` | `5746` | `392` | Keeps the full heuristic QA extraction. |
| `strict` | `1879` | `264` | Requires management in the answer span, removes analyst-only answers, question-mark leakage, and operator prompt leakage. |

Strict retention is `32.7%` of extracted pairs.

## Feature State

- `pre_event_return_5d`: `100%` coverage and included in the default tabular benchmark.
- `post_event_return_3d`: `100%` coverage and logged for analysis, but excluded from default training because it overlaps the PEAD label window.
- `glopardo` forward/trailing EPS proxy fields: `54.6%` coverage.
- True event-level EPS/revenue beat-miss and surprise fields: pipeline-ready, current coverage `0%`.

## Latest Results

| Benchmark | Broad AUROC | Strict AUROC | Broad AUPRC | Strict AUPRC | Broad Spearman | Strict Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `0.4405` | `0.4764` | `0.5110` | `0.5201` | `-0.0340` | `0.0833` |
| `text_plus_tabular` | `0.4880` | `0.5614` | `0.5727` | `0.6410` | `-0.0560` | `0.1810` |

## Method Notes

- Broad comparison reused the existing broad pair-model checkpoints and refreshed call-level scoring plus the tabular benchmark on the updated feature table.
- Strict comparison retrained the pair model end to end on the cleaned corpus.
- This is not perfectly apples-to-apples because broad uses `392` rolling-eval calls while strict uses `264`.

## Interpretation

- The strict corpus is currently the better benchmark.
- Cleaning the QA blocks improved both text-only and text-plus-tabular results.
- The biggest gains came from reducing extraction noise rather than from adding true earnings surprise, because the current local corpus still lacks reported-vs-consensus event data.

## Next Required Input

Add an event-level earnings estimates file through `earnings_events_path` in [`configs/data/model_ready_mag7.yaml`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/configs/data/model_ready_mag7.yaml) with at least:

- `ticker`
- `event_date`
- `reported_eps`
- `estimated_eps`

Optional revenue fields and precomputed surprise columns are supported. The expected schema is documented in [`data/external/earnings_fundamentals/README.md`](/Users/chris/Evaluating%20Faithfulness%20and%20Interpretability%20for%20PEAD%20Prediction%20from%20Earnings%20Call%20Q%26A/data/external/earnings_fundamentals/README.md).
