# QA Pair Regression Report

Run name: `qa_pair_finbert_regression_tech_largecap_strict_eps_fast20`

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
| 1 | 464 | 64 | 64 | 2006-01-18 | 2012-12-18 | 2014-01-07 | 2014-07-23 |
| 2 | 528 | 64 | 64 | 2006-01-18 | 2013-12-18 | 2014-07-23 | 2015-01-22 |
| 3 | 592 | 64 | 64 | 2006-01-18 | 2014-07-23 | 2015-01-22 | 2015-07-15 |
| 4 | 656 | 64 | 64 | 2006-01-18 | 2015-01-22 | 2015-07-20 | 2015-11-05 |
| 5 | 720 | 64 | 64 | 2006-01-18 | 2015-07-15 | 2015-11-06 | 2016-04-28 |
| 6 | 784 | 64 | 64 | 2006-01-18 | 2015-11-05 | 2016-04-29 | 2016-08-08 |
| 7 | 848 | 64 | 64 | 2006-01-18 | 2016-04-28 | 2016-08-09 | 2017-01-26 |
| 8 | 912 | 64 | 64 | 2006-01-18 | 2016-08-08 | 2017-01-26 | 2017-04-29 |
| 9 | 976 | 64 | 64 | 2006-01-18 | 2017-01-26 | 2017-05-01 | 2017-09-27 |
| 10 | 1040 | 64 | 64 | 2006-01-18 | 2017-04-29 | 2017-09-28 | 2018-02-01 |
| 11 | 1104 | 64 | 64 | 2006-01-18 | 2017-09-27 | 2018-02-01 | 2018-05-30 |
| 12 | 1168 | 64 | 64 | 2006-01-18 | 2018-02-01 | 2018-06-19 | 2018-10-29 |
| 13 | 1232 | 64 | 64 | 2006-01-18 | 2018-05-30 | 2018-10-29 | 2019-05-28 |
| 14 | 1296 | 64 | 64 | 2006-01-18 | 2018-10-29 | 2019-06-19 | 2020-07-30 |
| 15 | 1360 | 64 | 64 | 2006-01-18 | 2019-05-28 | 2020-07-30 | 2021-07-20 |
| 16 | 1424 | 64 | 64 | 2006-01-18 | 2020-07-30 | 2021-07-21 | 2022-02-16 |
| 17 | 1488 | 64 | 64 | 2006-01-18 | 2021-07-20 | 2022-02-16 | 2023-01-31 |
| 18 | 1552 | 64 | 64 | 2006-01-18 | 2022-02-16 | 2023-02-01 | 2024-02-15 |
| 19 | 1616 | 64 | 64 | 2006-01-18 | 2023-01-31 | 2024-02-20 | 2025-04-23 |

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
| 1 | text_only | mean | 0.5470 | 0.2106 | 0.6338 | 0.6168 | 0.6250 | 0.2825 | 0.0470 |
| 1 | text_tabular_ridge_gru_tuned | sequence | 0.5472 | 0.1400 | 0.5928 | 0.6316 | 0.5625 | 0.2144 | 0.0465 |
| 1 | text_tabular_ridge_attention_tuned | sequence | 0.5475 | 0.1865 | 0.4990 | 0.5331 | 0.5000 | 0.0259 | 0.0463 |
| 1 | text_tabular_ridge_base | mean+max | 0.5000 | 0.2139 | 0.5752 | 0.5921 | 0.5000 | 0.1479 | 0.0464 |
| 1 | text_tabular_ridge_base_tuned | mean+max | 0.5470 | 0.2139 | 0.5752 | 0.5921 | 0.5156 | 0.1479 | 0.0464 |
| 1 | text_tabular_ridge_rich_tuned | rich_agg | 0.5470 | 0.1848 | 0.5303 | 0.5646 | 0.5000 | 0.1231 | 0.0469 |
| 1 | text_tabular_boosted_rich_tuned | rich_agg | 0.5826 | 0.0587 | 0.4121 | 0.4821 | 0.5000 | -0.1019 | 0.2688 |
| 2 | text_only | max | 0.5000 | -0.0221 | 0.5000 | 0.6562 | 0.6562 | -0.0378 | 0.1562 |
| 2 | text_tabular_ridge_gru_tuned | sequence | 0.4953 | 0.3598 | 0.4578 | 0.6059 | 0.4062 | 0.0743 | 0.1941 |
| 2 | text_tabular_ridge_attention_tuned | sequence | 0.4985 | 0.3083 | 0.4903 | 0.6353 | 0.4062 | 0.0981 | 0.1812 |
| 2 | text_tabular_ridge_base | mean+max | 0.5000 | 0.3272 | 0.5130 | 0.6454 | 0.4219 | 0.2056 | 0.1882 |
| 2 | text_tabular_ridge_base_tuned | mean+max | 0.5050 | 0.3272 | 0.5130 | 0.6454 | 0.4062 | 0.2056 | 0.1882 |
| 2 | text_tabular_ridge_rich_tuned | rich_agg | 0.4974 | 0.1980 | 0.4935 | 0.6485 | 0.5469 | 0.1222 | 0.1566 |
| 2 | text_tabular_boosted_rich_tuned | rich_agg | 0.6204 | 0.0347 | 0.5541 | 0.7343 | 0.5000 | 0.1825 | 0.1114 |
| 3 | text_only | max | 0.6562 | -0.0865 | 0.4226 | 0.5523 | 0.5938 | -0.1592 | 0.0469 |
| 3 | text_tabular_ridge_gru_tuned | sequence | 0.5000 | 0.0602 | 0.6092 | 0.7317 | 0.6094 | 0.2289 | 0.0469 |
| 3 | text_tabular_ridge_attention_tuned | sequence | 0.6552 | 0.1162 | 0.5610 | 0.6736 | 0.6094 | 0.1522 | 0.0470 |
| 3 | text_tabular_ridge_base | mean+max | 0.5000 | 0.1349 | 0.5179 | 0.6381 | 0.6094 | 0.0331 | 0.0472 |
| 3 | text_tabular_ridge_base_tuned | mean+max | 0.6544 | 0.1349 | 0.5179 | 0.6381 | 0.5938 | 0.0331 | 0.0472 |
| 3 | text_tabular_ridge_rich_tuned | rich_agg | 0.6558 | 0.1104 | 0.5405 | 0.6792 | 0.6250 | 0.0959 | 0.0470 |
| 3 | text_tabular_boosted_rich_tuned | rich_agg | 0.3586 | 0.0209 | 0.5395 | 0.6545 | 0.5938 | 0.0976 | 0.1646 |
| 4 | text_only | mean | 0.6095 | -0.0784 | 0.5610 | 0.6525 | 0.6562 | -0.1517 | 0.0312 |
| 4 | text_tabular_ridge_gru_tuned | sequence | 0.6080 | 0.0902 | 0.6744 | 0.7897 | 0.6875 | 0.2537 | 0.0312 |
| 4 | text_tabular_ridge_attention_tuned | sequence | 0.6078 | 0.1009 | 0.6713 | 0.7947 | 0.6562 | 0.3131 | 0.0313 |
| 4 | text_tabular_ridge_base | mean+max | 0.5000 | -0.0025 | 0.5589 | 0.7289 | 0.6406 | 0.0657 | 0.0312 |
| 4 | text_tabular_ridge_base_tuned | mean+max | 0.6089 | -0.0025 | 0.5589 | 0.7289 | 0.6875 | 0.0657 | 0.0312 |
| 4 | text_tabular_ridge_rich_tuned | rich_agg | 0.6090 | -0.0280 | 0.5716 | 0.7159 | 0.6406 | 0.0416 | 0.0313 |
| 4 | text_tabular_boosted_rich_tuned | rich_agg | 0.1882 | 0.0344 | 0.5938 | 0.7363 | 0.6406 | 0.2223 | 0.1488 |
| 5 | text_only | max | 0.6406 | 0.0704 | 0.4777 | 0.5846 | 0.5625 | -0.0353 | 0.0469 |
| 5 | text_tabular_ridge_gru_tuned | sequence | 0.6390 | 0.2911 | 0.7227 | 0.8023 | 0.6562 | 0.4080 | 0.0468 |
| 5 | text_tabular_ridge_attention_tuned | sequence | 0.6394 | 0.3323 | 0.7217 | 0.7878 | 0.6719 | 0.3750 | 0.0468 |
| 5 | text_tabular_ridge_base | mean+max | 0.5000 | 0.2198 | 0.7399 | 0.8160 | 0.5938 | 0.3935 | 0.0469 |
| 5 | text_tabular_ridge_base_tuned | mean+max | 0.6398 | 0.2198 | 0.7399 | 0.8160 | 0.6406 | 0.3935 | 0.0469 |
| 5 | text_tabular_ridge_rich_tuned | rich_agg | 0.6390 | 0.2640 | 0.7115 | 0.8038 | 0.6250 | 0.3414 | 0.0470 |
| 5 | text_tabular_boosted_rich_tuned | rich_agg | 0.4073 | 0.3209 | 0.5162 | 0.6789 | 0.5156 | 0.0502 | 0.1605 |
| 6 | text_only | max | 0.5937 | 0.0394 | 0.4401 | 0.7791 | 0.7188 | 0.0439 | 0.1563 |
| 6 | text_tabular_ridge_gru_tuned | sequence | 0.5929 | 0.3892 | 0.5898 | 0.7894 | 0.7031 | 0.1017 | 0.1533 |
| 6 | text_tabular_ridge_attention_tuned | sequence | 0.5920 | 0.3309 | 0.5143 | 0.7663 | 0.7031 | -0.0228 | 0.1536 |
| 6 | text_tabular_ridge_base | mean+max | 0.5000 | 0.3586 | 0.4740 | 0.7367 | 0.7500 | -0.0115 | 0.1727 |
| 6 | text_tabular_ridge_base_tuned | mean+max | 0.5917 | 0.3586 | 0.4740 | 0.7367 | 0.6406 | -0.0115 | 0.1727 |
| 6 | text_tabular_ridge_rich_tuned | rich_agg | 0.5933 | 0.3353 | 0.4674 | 0.7435 | 0.5469 | -0.0059 | 0.1545 |
| 6 | text_tabular_boosted_rich_tuned | rich_agg | 0.4733 | 0.1559 | 0.3672 | 0.7185 | 0.4844 | -0.0864 | 0.2507 |
| 7 | text_only | max | 0.5000 | -0.0022 | 0.6082 | 0.7297 | 0.6094 | -0.2066 | 0.1407 |
| 7 | text_tabular_ridge_gru_tuned | sequence | 0.5000 | 0.1242 | 0.5590 | 0.7071 | 0.6094 | 0.1533 | 0.1395 |
| 7 | text_tabular_ridge_attention_tuned | sequence | 0.5000 | 0.0892 | 0.4626 | 0.6394 | 0.6094 | -0.0127 | 0.1393 |
| 7 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0052 | 0.5231 | 0.6565 | 0.6094 | 0.1388 | 0.1405 |
| 7 | text_tabular_ridge_base_tuned | mean+max | 0.5000 | 0.0052 | 0.5231 | 0.6565 | 0.6094 | 0.1388 | 0.1405 |
| 7 | text_tabular_ridge_rich_tuned | rich_agg | 0.5000 | -0.0013 | 0.5241 | 0.6628 | 0.6094 | 0.1482 | 0.1402 |
| 7 | text_tabular_boosted_rich_tuned | rich_agg | 0.2654 | -0.0282 | 0.5446 | 0.6594 | 0.6094 | 0.0847 | 0.1421 |
| 8 | text_only | mean | 0.6094 | -0.0075 | 0.4725 | 0.5688 | 0.5625 | -0.1251 | 0.0313 |
| 8 | text_tabular_ridge_gru_tuned | sequence | 0.6063 | 0.2500 | 0.5395 | 0.6218 | 0.5938 | -0.0027 | 0.0313 |
| 8 | text_tabular_ridge_attention_tuned | sequence | 0.6079 | 0.1768 | 0.5345 | 0.6194 | 0.5625 | -0.0346 | 0.0313 |
| 8 | text_tabular_ridge_base | mean+max | 0.5000 | 0.1591 | 0.5195 | 0.6111 | 0.5781 | -0.0258 | 0.0311 |
| 8 | text_tabular_ridge_base_tuned | mean+max | 0.5000 | 0.1591 | 0.5195 | 0.6111 | 0.5781 | -0.0258 | 0.0311 |
| 8 | text_tabular_ridge_rich_tuned | rich_agg | 0.5000 | 0.2193 | 0.5215 | 0.6319 | 0.5781 | -0.0187 | 0.0312 |
| 8 | text_tabular_boosted_rich_tuned | rich_agg | 0.5477 | 0.2022 | 0.5876 | 0.6755 | 0.5625 | 0.1251 | 0.1767 |
| 9 | text_only | mean | 0.5781 | 0.0890 | 0.5856 | 0.5421 | 0.5938 | 0.1872 | 0.1562 |
| 9 | text_tabular_ridge_gru_tuned | sequence | 0.5744 | 0.1734 | 0.5355 | 0.4501 | 0.5000 | 0.0978 | 0.1646 |
| 9 | text_tabular_ridge_attention_tuned | sequence | 0.5738 | 0.2064 | 0.5455 | 0.4606 | 0.4844 | 0.0957 | 0.1632 |
| 9 | text_tabular_ridge_base | mean+max | 0.5000 | 0.1779 | 0.5195 | 0.4385 | 0.4219 | 0.0502 | 0.1608 |
| 9 | text_tabular_ridge_base_tuned | mean+max | 0.5761 | 0.1779 | 0.5195 | 0.4385 | 0.5000 | 0.0502 | 0.1608 |
| 9 | text_tabular_ridge_rich_tuned | rich_agg | 0.5755 | 0.2516 | 0.5355 | 0.4419 | 0.5312 | 0.0791 | 0.1599 |
| 9 | text_tabular_boosted_rich_tuned | rich_agg | 0.5360 | 0.2558 | 0.5536 | 0.5066 | 0.5469 | -0.0243 | 0.1932 |
| 10 | text_only | mean | 0.4219 | 0.0531 | 0.4839 | 0.6705 | 0.3125 | 0.0605 | 0.2501 |
| 10 | text_tabular_ridge_gru_tuned | sequence | 0.4406 | 0.0826 | 0.3942 | 0.5981 | 0.2969 | -0.1750 | 0.2410 |
| 10 | text_tabular_ridge_attention_tuned | sequence | 0.4401 | 0.0544 | 0.3998 | 0.5971 | 0.2656 | -0.1528 | 0.2411 |
| 10 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0592 | 0.4286 | 0.6150 | 0.3281 | -0.1349 | 0.2412 |
| 10 | text_tabular_ridge_base_tuned | mean+max | 0.4402 | 0.0592 | 0.4286 | 0.6150 | 0.2812 | -0.1349 | 0.2412 |
| 10 | text_tabular_ridge_rich_tuned | rich_agg | 0.4391 | 0.1002 | 0.4275 | 0.6211 | 0.2969 | -0.1288 | 0.2422 |
| 10 | text_tabular_boosted_rich_tuned | rich_agg | 0.9229 | -0.1032 | 0.5050 | 0.7233 | 0.3438 | -0.0102 | 0.1666 |
| 11 | text_only | mean | 0.6719 | -0.0527 | 0.4432 | 0.6853 | 0.6875 | 0.0208 | 0.0155 |
| 11 | text_tabular_ridge_gru_tuned | sequence | 0.6714 | -0.0487 | 0.4250 | 0.6453 | 0.6562 | 0.1042 | 0.0157 |
| 11 | text_tabular_ridge_attention_tuned | sequence | 0.6710 | -0.0855 | 0.3966 | 0.6230 | 0.6406 | 0.1351 | 0.0158 |
| 11 | text_tabular_ridge_base | mean+max | 0.5000 | -0.1414 | 0.3750 | 0.6258 | 0.6875 | 0.2187 | 0.0160 |
| 11 | text_tabular_ridge_base_tuned | mean+max | 0.6698 | -0.1414 | 0.3750 | 0.6258 | 0.6562 | 0.2187 | 0.0160 |
| 11 | text_tabular_ridge_rich_tuned | rich_agg | 0.6685 | -0.1471 | 0.3784 | 0.6104 | 0.6562 | 0.2418 | 0.0162 |
| 11 | text_tabular_boosted_rich_tuned | rich_agg | 0.2410 | -0.1283 | 0.7011 | 0.8041 | 0.6875 | 0.1898 | 0.1957 |
| 12 | text_only | mean | 0.5000 | -0.1283 | 0.4448 | 0.5246 | 0.5156 | 0.1585 | 0.1719 |
| 12 | text_tabular_ridge_gru_tuned | sequence | 0.5000 | 0.0415 | 0.5308 | 0.5389 | 0.5156 | 0.1255 | 0.1719 |
| 12 | text_tabular_ridge_attention_tuned | sequence | 0.5000 | 0.0497 | 0.5435 | 0.5348 | 0.5156 | 0.1546 | 0.1719 |
| 12 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0214 | 0.5552 | 0.5525 | 0.5156 | 0.2154 | 0.1718 |
| 12 | text_tabular_ridge_base_tuned | mean+max | 0.5000 | 0.0214 | 0.5552 | 0.5525 | 0.5156 | 0.2154 | 0.1718 |
| 12 | text_tabular_ridge_rich_tuned | rich_agg | 0.5000 | 0.0473 | 0.5591 | 0.5411 | 0.5156 | 0.1875 | 0.1719 |
| 12 | text_tabular_boosted_rich_tuned | rich_agg | 0.2375 | 0.1277 | 0.5445 | 0.5483 | 0.5156 | 0.1834 | 0.1507 |
| 13 | text_only | max | 0.5156 | 0.0329 | 0.5713 | 0.5687 | 0.5156 | -0.0152 | 0.0156 |
| 13 | text_tabular_ridge_gru_tuned | sequence | 0.5153 | 0.2394 | 0.3848 | 0.5142 | 0.4375 | -0.0703 | 0.0153 |
| 13 | text_tabular_ridge_attention_tuned | sequence | 0.5151 | 0.1891 | 0.4043 | 0.4865 | 0.4531 | -0.0683 | 0.0156 |
| 13 | text_tabular_ridge_base | mean+max | 0.5000 | 0.1652 | 0.4463 | 0.5253 | 0.5000 | -0.0185 | 0.0156 |
| 13 | text_tabular_ridge_base_tuned | mean+max | 0.5145 | 0.1652 | 0.4463 | 0.5253 | 0.4844 | -0.0185 | 0.0156 |
| 13 | text_tabular_ridge_rich_tuned | rich_agg | 0.5156 | 0.1447 | 0.4551 | 0.5489 | 0.4844 | 0.0280 | 0.0156 |
| 13 | text_tabular_boosted_rich_tuned | rich_agg | 0.5000 | 0.1158 | 0.4980 | 0.5751 | 0.4375 | 0.0053 | 0.2409 |
| 14 | text_only | mean | 0.5001 | 0.1674 | 0.3720 | 0.4958 | 0.4062 | -0.0283 | 0.0624 |
| 14 | text_tabular_ridge_gru_tuned | sequence | 0.5002 | -0.0260 | 0.4067 | 0.5046 | 0.4688 | 0.1040 | 0.0628 |
| 14 | text_tabular_ridge_attention_tuned | sequence | 0.5000 | 0.0216 | 0.4405 | 0.5176 | 0.5000 | 0.0785 | 0.0626 |
| 14 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0263 | 0.4137 | 0.5121 | 0.3906 | 0.1498 | 0.1094 |
| 14 | text_tabular_ridge_base_tuned | mean+max | 0.5000 | 0.0263 | 0.4137 | 0.5121 | 0.4531 | 0.1498 | 0.1094 |
| 14 | text_tabular_ridge_rich_tuned | rich_agg | 0.4999 | 0.0055 | 0.4058 | 0.5089 | 0.4219 | 0.1593 | 0.0626 |
| 14 | text_tabular_boosted_rich_tuned | rich_agg | 0.5079 | 0.0089 | 0.5456 | 0.6191 | 0.5469 | 0.0416 | 0.1785 |
| 15 | text_only | mean | 0.5625 | 0.0597 | 0.5034 | 0.6106 | 0.5312 | 0.0271 | 0.0156 |
| 15 | text_tabular_ridge_gru_tuned | sequence | 0.5620 | 0.0338 | 0.7163 | 0.7432 | 0.6406 | 0.4404 | 0.0158 |
| 15 | text_tabular_ridge_attention_tuned | sequence | 0.5618 | 0.0983 | 0.7084 | 0.7336 | 0.6094 | 0.3978 | 0.0158 |
| 15 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0661 | 0.6887 | 0.7211 | 0.5469 | 0.3780 | 0.0157 |
| 15 | text_tabular_ridge_base_tuned | mean+max | 0.5620 | 0.0661 | 0.6887 | 0.7211 | 0.6094 | 0.3780 | 0.0157 |
| 15 | text_tabular_ridge_rich_tuned | rich_agg | 0.5621 | 0.0627 | 0.6857 | 0.7149 | 0.6250 | 0.3944 | 0.0157 |
| 15 | text_tabular_boosted_rich_tuned | rich_agg | 0.2105 | -0.0866 | 0.4118 | 0.5441 | 0.5156 | -0.0792 | 0.2061 |
| 16 | text_only | max | 0.5469 | -0.0491 | 0.4872 | 0.5920 | 0.6094 | -0.0777 | 0.0624 |
| 16 | text_tabular_ridge_gru_tuned | sequence | 0.5471 | 0.3405 | 0.4882 | 0.5767 | 0.4062 | -0.0416 | 0.0628 |
| 16 | text_tabular_ridge_attention_tuned | sequence | 0.5458 | 0.1817 | 0.5251 | 0.6195 | 0.5625 | -0.0013 | 0.0626 |
| 16 | text_tabular_ridge_base | mean+max | 0.5000 | 0.3500 | 0.5118 | 0.6087 | 0.6094 | 0.0245 | 0.0629 |
| 16 | text_tabular_ridge_base_tuned | mean+max | 0.5469 | 0.3500 | 0.5118 | 0.6087 | 0.4531 | 0.0245 | 0.0629 |
| 16 | text_tabular_ridge_rich_tuned | rich_agg | 0.5467 | 0.3122 | 0.5467 | 0.6314 | 0.5312 | 0.1120 | 0.0627 |
| 16 | text_tabular_boosted_rich_tuned | rich_agg | 0.2460 | -0.0236 | 0.6256 | 0.6957 | 0.6406 | 0.1468 | 0.2311 |
| 17 | text_only | mean | 0.6093 | 0.1022 | 0.5775 | 0.5871 | 0.5312 | 0.1910 | 0.0781 |
| 17 | text_tabular_ridge_gru_tuned | sequence | 0.6093 | 0.0009 | 0.4078 | 0.5024 | 0.4844 | 0.2367 | 0.0782 |
| 17 | text_tabular_ridge_attention_tuned | sequence | 0.6092 | 0.0478 | 0.3696 | 0.4737 | 0.5000 | 0.2866 | 0.0782 |
| 17 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0137 | 0.3735 | 0.4682 | 0.5312 | 0.3453 | 0.0782 |
| 17 | text_tabular_ridge_base_tuned | mean+max | 0.6088 | 0.0137 | 0.3735 | 0.4682 | 0.5312 | 0.3453 | 0.0782 |
| 17 | text_tabular_ridge_rich_tuned | rich_agg | 0.6090 | 0.0595 | 0.3608 | 0.4586 | 0.5312 | 0.3739 | 0.0782 |
| 17 | text_tabular_boosted_rich_tuned | rich_agg | 0.3154 | 0.1089 | 0.4755 | 0.5257 | 0.5156 | 0.0481 | 0.1051 |
| 18 | text_only | max | 0.5313 | -0.0402 | 0.4740 | 0.6809 | 0.4375 | -0.0672 | 0.1250 |
| 18 | text_tabular_ridge_gru_tuned | sequence | 0.5313 | 0.2674 | 0.5000 | 0.7200 | 0.4531 | -0.0372 | 0.1252 |
| 18 | text_tabular_ridge_attention_tuned | sequence | 0.5313 | 0.3273 | 0.5487 | 0.7434 | 0.5625 | 0.0599 | 0.1253 |
| 18 | text_tabular_ridge_base | mean+max | 0.5000 | 0.3168 | 0.5411 | 0.7309 | 0.6562 | 0.0292 | 0.1253 |
| 18 | text_tabular_ridge_base_tuned | mean+max | 0.5313 | 0.3168 | 0.5411 | 0.7309 | 0.5156 | 0.0292 | 0.1253 |
| 18 | text_tabular_ridge_rich_tuned | rich_agg | 0.5312 | 0.2926 | 0.5498 | 0.7500 | 0.5312 | 0.0516 | 0.1252 |
| 18 | text_tabular_boosted_rich_tuned | rich_agg | 0.4221 | 0.1627 | 0.6126 | 0.7569 | 0.5938 | 0.2383 | 0.2110 |
| 19 | text_only | mean | 0.5000 | 0.0592 | 0.5030 | 0.4885 | 0.4375 | 0.1630 | 0.2188 |
| 19 | text_tabular_ridge_gru_tuned | sequence | 0.5000 | 0.1097 | 0.5655 | 0.5336 | 0.4375 | 0.1227 | 0.2189 |
| 19 | text_tabular_ridge_attention_tuned | sequence | 0.5000 | 0.0844 | 0.5635 | 0.5427 | 0.4375 | 0.0836 | 0.2189 |
| 19 | text_tabular_ridge_base | mean+max | 0.5000 | 0.0982 | 0.5575 | 0.5853 | 0.4375 | 0.1101 | 0.2190 |
| 19 | text_tabular_ridge_base_tuned | mean+max | 0.5000 | 0.0982 | 0.5575 | 0.5853 | 0.4375 | 0.1101 | 0.2190 |
| 19 | text_tabular_ridge_rich_tuned | rich_agg | 0.5000 | 0.1428 | 0.5079 | 0.5153 | 0.4375 | 0.0639 | 0.2190 |
| 19 | text_tabular_boosted_rich_tuned | rich_agg | 0.1263 | 0.2439 | 0.4851 | 0.4698 | 0.4375 | -0.0335 | 0.1166 |

## Aggregate Benchmarks

| Benchmark | AUROC | AUPRC | Spearman | Pearson | RMSE | MAE | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.4913 [0.4594, 0.5169] | 0.5859 [0.5522, 0.6156] | -0.0157 [-0.0762, 0.0374] | -0.0007 [-0.0522, 0.0492] | 0.0912 [0.0866, 0.0965] | 0.0672 [0.0638, 0.0707] | 0.0633 [0.0435, 0.0890] | 0.5531 [0.5250, 0.5758] |
| text_tabular_ridge_base | 0.4905 [0.4609, 0.5180] | 0.5821 [0.5502, 0.6206] | 0.1203 [0.0680, 0.1667] | -0.0143 [-0.0760, 0.0750] | 0.1457 [0.1086, 0.1850] | 0.0786 [0.0733, 0.0857] | 0.0670 [0.0488, 0.0945] | 0.5437 [0.5187, 0.5696] |
| text_tabular_ridge_base_tuned | 0.4905 [0.4609, 0.5180] | 0.5821 [0.5502, 0.6206] | 0.1203 [0.0680, 0.1667] | -0.0143 [-0.0760, 0.0750] | 0.1457 [0.1086, 0.1850] | 0.0786 [0.0733, 0.0857] | 0.0670 [0.0488, 0.0945] | 0.5281 [0.5062, 0.5531] |
| text_tabular_ridge_gru_tuned | 0.4914 [0.4623, 0.5174] | 0.5803 [0.5485, 0.6180] | 0.1390 [0.0865, 0.1863] | -0.0428 [-0.0959, 0.0856] | 0.2222 [0.1047, 0.3233] | 0.0807 [0.0714, 0.0934] | 0.0659 [0.0470, 0.0941] | 0.5305 [0.5078, 0.5563] |
| text_tabular_ridge_attention_tuned | 0.4919 [0.4637, 0.5180] | 0.5779 [0.5443, 0.6162] | 0.1175 [0.0641, 0.1742] | -0.0397 [-0.0954, 0.0675] | 0.1833 [0.1047, 0.2542] | 0.0788 [0.0714, 0.0886] | 0.0657 [0.0445, 0.0927] | 0.5359 [0.5148, 0.5641] |
| text_tabular_ridge_rich_tuned | 0.4904 [0.4602, 0.5172] | 0.5823 [0.5499, 0.6197] | 0.1218 [0.0699, 0.1787] | -0.0164 [-0.0810, 0.0636] | 0.1318 [0.1009, 0.1650] | 0.0748 [0.0703, 0.0813] | 0.0644 [0.0452, 0.0914] | 0.5336 [0.5086, 0.5610] |
| text_tabular_boosted_rich_tuned | 0.5307 [0.4981, 0.5566] | 0.6238 [0.5882, 0.6569] | 0.0773 [0.0291, 0.1267] | 0.0757 [0.0281, 0.1272] | 0.1955 [0.1885, 0.2030] | 0.1576 [0.1514, 0.1641] | 0.1290 [0.1076, 0.1601] | 0.5352 [0.5086, 0.5610] |

## Notes

- Pair-level text model is a FinBERT regression head trained on analyst-question plus following-answer pairs.
- Pool selection uses validation Spearman, with validation AUROC as the tie-breaker.
- The benchmark ladder compares the original mean+max Ridge baseline, the same model with tuned thresholds, a richer order-aware aggregation feature set, and a boosted classifier on the rich feature set.
- Sequence benchmarks train compact call-level classifiers over ordered QA-pair feature sequences, including pair score, pair position, answer lengths, and coarse speaker-role features.
- Richer text aggregation includes distributional and order-aware features such as quantiles, recent-pair averages, first/last deltas, and score trend slope.
- Thresholds are selected on each validation fold to maximize `accuracy` rather than using a fixed 0.5 cutoff.
- Event-level earnings surprise and beat/miss features are used when available in the local fundamentals table.
- Rolling evaluation uses non-overlapping test windows with expanding training history.
