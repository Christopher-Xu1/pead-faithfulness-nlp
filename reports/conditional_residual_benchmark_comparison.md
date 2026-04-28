# Conditional Residual Benchmark Comparison

## Aggregate Comparison

| Benchmark | Universe | Calls | Pairs | Folds | EPS Surprise Cov. | Revenue Surprise Cov. | AUROC | AUPRC | Accuracy | Spearman | Pearson | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict Mag7 text_plus_tabular | mag7_strict | 264 | 1879 | 3 | 0.0000 | 0.0000 | 0.5614 | 0.6410 | 0.5139 | 0.1810 | 0.1571 | 0.1215 |
| Tech-largecap boosted_rich_tuned | tech_largecap_strict | 1760 | 15683 | 20 | 0.5044 | 0.0000 | 0.5307 | 0.6238 | 0.5352 | 0.0773 | 0.0757 | 0.1955 |
| Conditional residual simple | tech_largecap_strict | 1760 | 15683 | 20 | 0.8636 | 0.9557 | 0.5703 | 0.6474 | 0.5813 | 0.1369 | 0.0477 | 0.0970 |

## Notes

- The conditional residual run improves on the previous best tech-largecap fast-20 benchmark in AUROC, AUPRC, and accuracy.
- The conditional residual run also improves Spearman rank correlation versus the prior best tech-largecap benchmark, while Pearson remains below the older Mag7 strict ridge benchmark.
- The comparison is not perfectly apples-to-apples because the prepared conditional residual bundle has materially richer EPS and revenue surprise coverage.

## Generated Figures

- Per-fold performance grid: `C:\Users\cx2330\pead-faithfulness-nlp\outputs\figures\conditional_residual_per_fold_performance.png`
- Residual target scatter: `C:\Users\cx2330\pead-faithfulness-nlp\outputs\figures\conditional_residual_residual_vs_target_scatter.png`
