# Research Results Summary

## Key Benchmark Clarification

The earlier `0.6410` AUPRC result was the **Mag7 strict QA-pair text+tabular Ridge benchmark**, not the expanded tech-largecap benchmark.

Source report: `reports/qa_pair_regression_strict_report.md` and `reports/qa_pair_regression_strict_sequence_reuse_report.md`.

Setup:

- Universe: Mag7 only.
- Corpus: strict QA-pair dataset.
- Rows: `1879` strict QA pairs.
- Calls with strict pairs: `264`.
- Rolling folds: `3`.
- Text encoder: FinBERT pair-level regression model over analyst question plus following management answer.
- Call aggregation: mean+max pair-score aggregation.
- Final tabular stage: Ridge model using aggregated text scores plus available tabular controls.
- Earnings surprise coverage: `0.0%` in this older Mag7 run.

Metrics:

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_tabular_ridge_base` | `0.5614` | `0.6410` | `0.5139` | `0.1810` | `0.0279` |
| `text_tabular_ridge_base_tuned` | `0.5614` | `0.6410` | `0.6250` | `0.1810` | `0.0279` |

Interpretation:

- The `0.6410` AUPRC is the aggregate AUPRC over the `3` Mag7 strict rolling folds.
- Threshold tuning changes classification accuracy but not AUROC or AUPRC.
- Therefore, Mag7 Ridge accuracy is `0.5139` with the default threshold and `0.6250` with validation-tuned thresholds.

## Expanded Tech-Largecap Fast 20-Fold Result

The expanded tech-largecap run did not reproduce the one-fold quick improvement.

Source report: `reports/qa_pair_regression_tech_largecap_strict_eps_fast20_report.md`.

Setup:

- Universe: large-cap technology universe plus forced Mag7 inclusion.
- Corpus: strict QA-pair dataset.
- Rows: `15683` strict QA pairs.
- Calls with strict pairs: `1760`.
- Rolling folds: `20`.
- EPS surprise coverage: `50.4%`.
- Revenue surprise coverage: `0.0%`.

Best aggregate result:

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_tabular_boosted_rich_tuned` | `0.5307` | `0.6238` | `0.5352` | `0.0773` | `0.1290` |

Interpretation:

- The expanded dataset improves sample size but does not yet improve the headline benchmark.
- The best 20-fold AUPRC is `0.6238`, below the older Mag7 strict Ridge AUPRC of `0.6410`.
- The current bottleneck is likely feature quality and event-fundamentals coverage, especially missing revenue surprise.

## Conditional Residual Tech-Largecap Result

Source artifacts:

- `reports/conditional_residual_qa_pead_report.md`
- `reports/conditional_residual_benchmark_comparison.md`

Setup:

- Universe: tech-largecap strict QA-pair corpus.
- Rows: `15683` strict QA pairs.
- Calls with strict pairs: `1760`.
- Rolling folds: `20`.
- EPS surprise coverage: `0.8636`.
- Revenue surprise coverage: `0.9557`.
- Final combiner: `baseline_pred + mean_pair_residual`.

Aggregate result:

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | Pearson | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `conditional_residual_simple` | `0.5703` | `0.6474` | `0.5813` | `0.1369` | `0.0477` | `0.0970` |

Comparison to the prior expanded tech-largecap best:

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | Pearson | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_tabular_boosted_rich_tuned` | `0.5307` | `0.6238` | `0.5352` | `0.0773` | `0.0757` | `0.1955` |
| `conditional_residual_simple` | `0.5703` | `0.6474` | `0.5813` | `0.1369` | `0.0477` | `0.0970` |

Interpretation:

- The conditional residual pipeline improves the headline classification metrics on the expanded tech-largecap benchmark.
- Rank correlation also improves versus the prior expanded benchmark.
- Linear correlation to the continuous PEAD target remains modest, so the main gain appears to be better directional separation rather than fully calibrated magnitude prediction.
- This is not a perfectly apples-to-apples comparison because the prepared conditional residual bundle has much richer event-fundamentals coverage than the earlier fast-20 benchmark.
