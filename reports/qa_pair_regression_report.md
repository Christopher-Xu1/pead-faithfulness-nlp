# QA Pair Regression Report

Run name: `qa_pair_finbert_regression`

## Dataset

- Pair filter profile: `broad`
- Pair rows: `5746`
- Calls with pairs: `392`
- Pair rows before filtering: `5746`
- Calls with pairs before filtering: `392`
- Pair retention rate: `1.0000`
- Mean pairs per call: `14.66`
- Median pairs per call: `12.00`
- Mean question chars: `486.3`
- Mean answer chars: `1446.9`
- Pre-event 5d return coverage: `1.0000`
- Post-event 3d return coverage: `1.0000`
- Management-role answer rate: `0.5414`
- Analyst-only answer rate: `0.4586`
- Answer question-mark rate: `0.1582`
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
- Zero-pair calls excluded from rolling eval: `18`
- Calls used in rolling eval: `392`

## Rolling Splits

| Fold | Train Calls | Val Calls | Test Calls | Train Start | Train End | Test Start | Test End |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 0 | 200 | 48 | 48 | 2006-01-19 | 2016-11-02 | 2018-08-16 | 2020-07-30 |
| 1 | 296 | 48 | 48 | 2006-01-19 | 2020-07-30 | 2022-10-26 | 2025-05-01 |

## Fold Results

| Fold | Benchmark | Pooling | Val Spearman | Test AUROC | Test AUPRC | Test Spearman | Test ECE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | text_only | max | 0.1744 | 0.3234 | 0.4657 | -0.1801 | 0.0833 |
| 0 | text_plus_tabular | mean+max | 0.1928 | 0.4301 | 0.5922 | -0.0431 | 0.0815 |
| 1 | text_only | mean | -0.0305 | 0.4800 | 0.5194 | -0.0686 | 0.0208 |
| 1 | text_plus_tabular | mean+max | -0.0547 | 0.5635 | 0.6246 | -0.0263 | 0.0211 |

## Aggregate Benchmarks

| Benchmark | AUROC | AUPRC | Spearman | Pearson | RMSE | MAE | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.4405 [0.3260, 0.5436] | 0.5110 [0.4010, 0.6633] | -0.0340 [-0.2495, 0.1775] | -0.1057 [-0.3066, 0.1295] | 0.1292 [0.1046, 0.1572] | 0.0953 [0.0789, 0.1136] | 0.0521 [0.0104, 0.1667] | 0.4896 [0.3852, 0.5833] |
| text_plus_tabular | 0.4880 [0.3824, 0.6258] | 0.5727 [0.4427, 0.7133] | -0.0560 [-0.2459, 0.1531] | -0.0063 [-0.2055, 0.1282] | 0.1496 [0.1221, 0.1772] | 0.1135 [0.0953, 0.1307] | 0.0409 [0.0156, 0.1422] | 0.5000 [0.4060, 0.6146] |

## Notes

- Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.
- Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.
- Upper-bound model is a call-level Ridge regressor using aggregated text scores plus available tabular controls.
- Historical earnings surprise is not present in the local corpus, so surprise coverage is currently zero and the feature is excluded from fitting.
- Rolling evaluation uses non-overlapping test windows with expanding training history.
