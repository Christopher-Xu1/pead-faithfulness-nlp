# QA Pair Regression Report

Run name: `qa_pair_finbert_regression_strict`

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

| Fold | Benchmark | Pooling | Val Spearman | Test AUROC | Test AUPRC | Test Spearman | Test ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | text_only | mean | 0.1261 | 0.2778 | 0.4274 | 0.1861 | 0.0417 |
| 0 | text_plus_tabular | mean+max | 0.1574 | 0.6111 | 0.6952 | 0.2965 | 0.0417 |
| 1 | text_only | max | 0.1809 | 0.5000 | 0.5421 | -0.0765 | 0.0000 |
| 1 | text_plus_tabular | mean+max | 0.0722 | 0.5833 | 0.6320 | 0.0791 | 0.0417 |
| 2 | text_only | mean | 0.0757 | 0.4825 | 0.5236 | 0.1609 | 0.0417 |
| 2 | text_plus_tabular | mean+max | 0.2452 | 0.5734 | 0.7273 | 0.2165 | 0.0827 |

## Aggregate Benchmarks

| Benchmark | AUROC | AUPRC | Spearman | Pearson | RMSE | MAE | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.4764 [0.3235, 0.6146] | 0.5201 [0.3850, 0.6911] | 0.0833 [-0.1604, 0.3189] | 0.0989 [-0.0932, 0.2799] | 0.1202 [0.0949, 0.1422] | 0.0876 [0.0692, 0.1061] | 0.0278 [0.0133, 0.1563] | 0.5139 [0.3889, 0.6389] |
| text_plus_tabular | 0.5614 [0.4216, 0.7019] | 0.6410 [0.5049, 0.7806] | 0.1810 [-0.0594, 0.4465] | 0.1571 [-0.1232, 0.4264] | 0.1215 [0.0959, 0.1445] | 0.0898 [0.0705, 0.1085] | 0.0279 [0.0124, 0.1549] | 0.5139 [0.4028, 0.6528] |

## Notes

- Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.
- Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.
- Upper-bound model is a call-level Ridge regressor using aggregated text scores plus available tabular controls.
- Historical earnings surprise is not present in the local corpus, so surprise coverage is currently zero and the feature is excluded from fitting.
- Rolling evaluation uses non-overlapping test windows with expanding training history.
