# QA Pair Regression Report

Run name: `qa_pair_finbert_regression_tech_largecap_strict_eps_quick`

## Dataset

- Pair filter profile: `strict`
- Pair rows: `15683`
- Calls with pairs: `1760`
- Pair rows before filtering: `53550`
- Calls with pairs before filtering: `3160`
- Pair retention rate: `0.2929`
- Mean pairs per call: `8.91`
- Median pairs per call: `7.00`
- Mean question chars: `385.4`
- Mean answer chars: `1090.6`
- Pre-event 5d return coverage: `0.9841`
- Post-event 3d return coverage: `1.0000`
- Management-role answer rate: `1.0000`
- Analyst-only answer rate: `0.0000`
- Answer question-mark rate: `0.0000`
- Positive rate: `0.5737`
- Earnings surprise coverage: `0.5044`
- Reported EPS coverage: `0.5044`
- Estimated EPS coverage: `0.5044`
- EPS surprise coverage: `0.5044`
- EPS surprise pct coverage: `0.5044`
- Reported revenue coverage: `0.8630`
- Estimated revenue coverage: `0.0000`
- Revenue surprise coverage: `0.0000`
- EPS beat/miss coverage: `0.5044`
- Revenue beat/miss coverage: `0.0000`
- Forward EPS coverage (glopardo): `0.6944`
- Zero-pair calls excluded from rolling eval: `1444`
- Calls used in rolling eval: `1760`

## Rolling Splits

| Fold | Train Calls | Val Calls | Test Calls | Train Start | Train End | Test Start | Test End |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 400 | 64 | 64 | 2006-01-18 | 2011-05-11 | 2012-12-20 | 2013-12-18 |

## Fold Results

| Fold | Benchmark | Pooling | Threshold | Val Spearman | Test AUROC | Test AUPRC | Test Accuracy | Test Spearman | Test ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | text_only | max | 0.5469 | -0.2289 | 0.4798 | 0.5803 | 0.5000 | -0.1229 | 0.0000 |
| 0 | text_tabular_ridge_gru_tuned | sequence | 0.5454 | 0.2050 | 0.5202 | 0.6580 | 0.4844 | 0.1628 | 0.0006 |
| 0 | text_tabular_ridge_attention_tuned | sequence | 0.5465 | 0.2807 | 0.5232 | 0.6476 | 0.4688 | 0.1477 | 0.0006 |
| 0 | text_tabular_ridge_base | mean+max | 0.5000 | 0.1630 | 0.4433 | 0.5967 | 0.5469 | -0.0006 | 0.0001 |
| 0 | text_tabular_ridge_base_tuned | mean+max | 0.5468 | 0.1630 | 0.4433 | 0.5967 | 0.4531 | -0.0006 | 0.0001 |
| 0 | text_tabular_ridge_rich_tuned | rich_agg | 0.5467 | 0.1669 | 0.4571 | 0.6196 | 0.4375 | 0.0072 | 0.0000 |
| 0 | text_tabular_boosted_rich_tuned | rich_agg | 0.6875 | 0.0771 | 0.5202 | 0.6584 | 0.5156 | 0.1668 | 0.2442 |

## Aggregate Benchmarks

| Benchmark | AUROC | AUPRC | Spearman | Pearson | RMSE | MAE | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.4798 [0.3308, 0.5879] | 0.5803 [0.4353, 0.7506] | -0.1229 [-0.3087, 0.1642] | -0.1320 [-0.3241, 0.0774] | 0.0999 [0.0759, 0.1277] | 0.0741 [0.0584, 0.0953] | 0.0000 [0.0000, 0.1094] | 0.5000 [0.3586, 0.6020] |
| text_tabular_ridge_base | 0.4433 [0.3009, 0.5636] | 0.5967 [0.4465, 0.7359] | -0.0006 [-0.2569, 0.2449] | 0.1018 [-0.1656, 0.4003] | 0.0988 [0.0766, 0.1288] | 0.0738 [0.0599, 0.0882] | 0.0001 [0.0001, 0.1407] | 0.5469 [0.4219, 0.6645] |
| text_tabular_ridge_base_tuned | 0.4433 [0.3009, 0.5636] | 0.5967 [0.4465, 0.7359] | -0.0006 [-0.2569, 0.2449] | 0.1018 [-0.1656, 0.4003] | 0.0988 [0.0766, 0.1288] | 0.0738 [0.0599, 0.0882] | 0.0001 [0.0001, 0.1407] | 0.4531 [0.3281, 0.5469] |
| text_tabular_ridge_gru_tuned | 0.5202 [0.3892, 0.6555] | 0.6580 [0.5278, 0.7954] | 0.1628 [-0.0522, 0.3780] | 0.2690 [0.0707, 0.4960] | 0.0962 [0.0727, 0.1222] | 0.0717 [0.0576, 0.0854] | 0.0006 [0.0008, 0.1408] | 0.4844 [0.3750, 0.6176] |
| text_tabular_ridge_attention_tuned | 0.5232 [0.3928, 0.6614] | 0.6476 [0.5098, 0.8013] | 0.1477 [-0.0647, 0.3593] | 0.2176 [-0.0032, 0.4713] | 0.0984 [0.0741, 0.1259] | 0.0726 [0.0588, 0.0872] | 0.0006 [0.0009, 0.1409] | 0.4688 [0.3594, 0.5938] |
| text_tabular_ridge_rich_tuned | 0.4571 [0.3203, 0.6065] | 0.6196 [0.4807, 0.7509] | 0.0072 [-0.2471, 0.2871] | 0.0314 [-0.2584, 0.3302] | 0.0996 [0.0776, 0.1298] | 0.0738 [0.0602, 0.0888] | 0.0000 [0.0000, 0.1406] | 0.4375 [0.3199, 0.5238] |
| text_tabular_boosted_rich_tuned | 0.5202 [0.4478, 0.6727] | 0.6584 [0.5505, 0.8129] | 0.1668 [-0.0251, 0.4408] | 0.2455 [0.0560, 0.5061] | 0.2573 [0.2264, 0.2835] | 0.2168 [0.1865, 0.2443] | 0.2442 [0.1953, 0.3411] | 0.5156 [0.4062, 0.6562] |

## Notes

- Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.
- Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.
- The benchmark ladder compares the original mean+max Ridge baseline, the same model with tuned thresholds, a richer order-aware aggregation feature set, and a boosted classifier on the rich feature set.
- Sequence benchmarks train compact call-level classifiers over ordered QA-pair feature sequences, including pair score, pair position, answer lengths, and coarse speaker-role features.
- Richer text aggregation includes distributional and order-aware features such as quantiles, recent-pair averages, first/last deltas, and score trend slope.
- Thresholds are selected on each validation fold to maximize `accuracy` rather than using a fixed 0.5 cutoff.
- Event-level earnings surprise and beat/miss features are used when available in the local fundamentals table.
- Rolling evaluation uses non-overlapping test windows with expanding training history.
