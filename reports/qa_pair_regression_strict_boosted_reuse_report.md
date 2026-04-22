# QA Pair Regression Report

Run name: `qa_pair_finbert_regression_strict_boosted_reuse`

## Dataset

- Pair filter profile: `strict`
- Pair rows: `1879`
- Calls with pairs: `264`
- Pair rows before filtering: `5746`
- Calls with pairs before filtering: `392`
- Pair retention rate: `0.3270`
- Mean pairs per call: `7.12`
- Median pairs per call: `6.00`
- Mean question chars: `443.7`
- Mean answer chars: `1277.1`
- Pre-event 5d return coverage: `1.0000`
- Post-event 3d return coverage: `1.0000`
- Management-role answer rate: `1.0000`
- Analyst-only answer rate: `0.0000`
- Answer question-mark rate: `0.0000`
- Positive rate: `0.5171`
- Earnings surprise coverage: `0.0000`
- Reported EPS coverage: `0.0000`
- Estimated EPS coverage: `0.0000`
- EPS surprise coverage: `0.0000`
- EPS surprise pct coverage: `0.0000`
- Reported revenue coverage: `0.0000`
- Estimated revenue coverage: `0.0000`
- Revenue surprise coverage: `0.0000`
- EPS beat/miss coverage: `0.0000`
- Revenue beat/miss coverage: `0.0000`
- Forward EPS coverage (glopardo): `0.5463`
- Zero-pair calls excluded from rolling eval: `146`
- Calls used in rolling eval: `264`

## Rolling Splits

| Fold | Train Calls | Val Calls | Test Calls | Train Start | Train End | Test Start | Test End |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 160 | 24 | 24 | 2006-01-19 | 2016-10-28 | 2017-10-26 | 2018-10-25 |
| 1 | 184 | 24 | 24 | 2006-01-19 | 2017-10-26 | 2018-10-30 | 2021-07-27 |
| 2 | 208 | 24 | 24 | 2006-01-19 | 2018-10-25 | 2021-07-29 | 2024-08-01 |

## Fold Results

| Fold | Benchmark | Pooling | Threshold | Val Spearman | Test AUROC | Test AUPRC | Test Accuracy | Test Spearman | Test ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | text_only | recency_weighted_mean | 0.4583 | 0.1626 | 0.3125 | 0.4443 | 0.3333 | 0.1661 | 0.0417 |
| 0 | text_tabular_ridge_base | mean+max | 0.5000 | 0.1574 | 0.6111 | 0.6952 | 0.5000 | 0.2965 | 0.0417 |
| 0 | text_tabular_ridge_base_tuned | mean+max | 0.4583 | 0.1574 | 0.6111 | 0.6952 | 0.6250 | 0.2965 | 0.0417 |
| 0 | text_tabular_ridge_rich_tuned | rich_agg | 0.4583 | -0.0209 | 0.3125 | 0.4215 | 0.4167 | 0.3374 | 0.0417 |
| 0 | text_tabular_boosted_rich_tuned | rich_agg | 0.7648 | 0.0852 | 0.4792 | 0.5674 | 0.5417 | -0.2113 | 0.3601 |
| 1 | text_only | max | 0.5000 | 0.1809 | 0.5000 | 0.5421 | 0.5417 | -0.0765 | 0.0000 |
| 1 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0722 | 0.5833 | 0.6320 | 0.4583 | 0.0791 | 0.0417 |
| 1 | text_tabular_ridge_base_tuned | mean+max | 0.5000 | 0.0722 | 0.5833 | 0.6320 | 0.6250 | 0.0791 | 0.0417 |
| 1 | text_tabular_ridge_rich_tuned | rich_agg | 0.5000 | 0.2070 | 0.5833 | 0.6857 | 0.5417 | -0.0339 | 0.0832 |
| 1 | text_tabular_boosted_rich_tuned | rich_agg | 0.5000 | -0.1687 | 0.5417 | 0.5765 | 0.4583 | 0.0183 | 0.2613 |
| 2 | text_only | top3_mean | 0.5000 | 0.1017 | 0.4406 | 0.5713 | 0.4167 | 0.1130 | 0.0417 |
| 2 | text_tabular_ridge_base | mean+max | 0.5000 | 0.2452 | 0.5734 | 0.7273 | 0.5833 | 0.2165 | 0.0827 |
| 2 | text_tabular_ridge_base_tuned | mean+max | 0.5001 | 0.2452 | 0.5734 | 0.7273 | 0.6250 | 0.2165 | 0.0827 |
| 2 | text_tabular_ridge_rich_tuned | rich_agg | 0.5000 | 0.0417 | 0.5105 | 0.6157 | 0.5833 | 0.1774 | 0.0832 |
| 2 | text_tabular_boosted_rich_tuned | rich_agg | 0.2215 | -0.0600 | 0.5664 | 0.6012 | 0.5417 | 0.1974 | 0.2746 |

## Aggregate Benchmarks

| Benchmark | AUROC | AUPRC | Spearman | Pearson | RMSE | MAE | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.4726 [0.3280, 0.6121] | 0.5212 [0.3752, 0.6854] | 0.0281 [-0.2224, 0.2577] | 0.0781 [-0.1314, 0.2680] | 0.1204 [0.0952, 0.1424] | 0.0876 [0.0693, 0.1064] | 0.0278 [0.0133, 0.1562] | 0.4306 [0.3194, 0.5417] |
| text_tabular_ridge_base | 0.5614 [0.4216, 0.7019] | 0.6410 [0.5049, 0.7806] | 0.1810 [-0.0594, 0.4465] | 0.1571 [-0.1232, 0.4264] | 0.1215 [0.0959, 0.1445] | 0.0898 [0.0705, 0.1085] | 0.0279 [0.0124, 0.1549] | 0.5139 [0.4028, 0.6528] |
| text_tabular_ridge_base_tuned | 0.5614 [0.4216, 0.7019] | 0.6410 [0.5049, 0.7806] | 0.1810 [-0.0594, 0.4465] | 0.1571 [-0.1232, 0.4264] | 0.1215 [0.0959, 0.1445] | 0.0898 [0.0705, 0.1085] | 0.0279 [0.0124, 0.1549] | 0.6250 [0.5139, 0.7365] |
| text_tabular_ridge_rich_tuned | 0.4988 [0.3668, 0.6389] | 0.5785 [0.4391, 0.7247] | 0.0860 [-0.1648, 0.3295] | 0.1681 [-0.1465, 0.4658] | 0.1181 [0.0927, 0.1407] | 0.0870 [0.0667, 0.1064] | 0.0416 [0.0157, 0.1794] | 0.5139 [0.4028, 0.6389] |
| text_tabular_boosted_rich_tuned | 0.5297 [0.3990, 0.6869] | 0.5498 [0.4268, 0.7345] | 0.0261 [-0.2018, 0.2790] | 0.0985 [-0.1283, 0.3223] | 0.2976 [0.2538, 0.3324] | 0.2493 [0.2076, 0.2841] | 0.2626 [0.1911, 0.3921] | 0.5139 [0.3889, 0.6389] |

## Notes

- Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.
- Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.
- The benchmark ladder compares the original mean+max Ridge baseline, the same model with tuned thresholds, a richer order-aware aggregation feature set, and a boosted classifier on the rich feature set.
- Richer text aggregation includes distributional and order-aware features such as quantiles, recent-pair averages, first/last deltas, and score trend slope.
- Thresholds are selected on each validation fold to maximize `accuracy` rather than using a fixed 0.5 cutoff.
- Historical earnings surprise is not present in the local corpus, so surprise coverage is currently zero and the feature is excluded from fitting.
- Rolling evaluation uses non-overlapping test windows with expanding training history.
