# Experiment Log

This file tracks ablation runs for weak baseline diagnosis.

Current baseline:
- `finbert_pead20_mag7`
- Validation AUROC: `0.5475`
- Test AUROC: `0.4946`
- Test accuracy: `0.5574`

Detailed metrics are aggregated in `reports/experiment_metrics.csv`.

## 2026-04-12 Weak-Performance Ablations

### Token audit

- `full_qa_text` is far beyond the model budget in every split.
- Train `full_qa_text`: mean `7592` tokens, median `7181`, `100%` over 512, `97.9%` over 1024.
- Val `full_qa_text`: mean `8469` tokens, median `10185`, `100%` over 512, `95.0%` over 1024.
- Test `full_qa_text`: mean `9179` tokens, median `10514`, `100%` over 512, `100%` over 1024.
- Even analyst-only `text` is heavily truncated: train `97.1%` over 512, test `98.4%` over 512.

### Dataset variants

- Clean-label `|CAR| > 2%` variant reduced the corpus from `401` to `326` rows.
- `pead20_abs002` split sizes: train `228`, val `49`, test `49`.
- `pead60` keeps `401` rows but is materially more positive-skewed.
- `pead60` positive rates: train `0.607`, val `0.667`, test `0.574`.

### Ablation summary

| Run | Input | Val AUROC | Test AUROC | Test Acc. | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `finbert_pead20_mag7` | analyst-only `text`, 512 | `0.5475` | `0.4946` | `0.5574` | Best validation baseline |
| `finbert_fullqa_pead20_mag7` | full Q&A, raw 512 | `0.5197` | `0.3871` | `0.4426` | Raw full-Q&A hurts ranking badly |
| `finbert_fullqa_headtail_pead20_mag7` | full Q&A, head-tail 512 | `0.4630` | `0.4527` | `0.4918` | Collapsed to all-negative test predictions |
| `finbert_fullqa_pead20_mag7_len256` | full Q&A, raw 256 | `0.4491` | `0.5000` | `0.4426` | More truncation hurts validation further |
| `finbert_fullqa_pead20_abs002` | full Q&A, clean labels | `0.5119` | `0.4415` | `0.4694` | Collapsed to all-positive test predictions |
| `finbert_fullqa_pead60_mag7` | full Q&A, 60d label | `0.4275` | `0.5275` | `0.5738` | Ranking slightly better, threshold still all-positive |
| `roberta_fullqa_pead20_mag7` | RoBERTa full Q&A, 512 | `0.5347` | `0.4753` | `0.4918` | Better than other full-Q&A runs, still below baseline |
| `finbert_fullqa_pead20_mag7_seed7` | full Q&A, raw 512 | `0.4676` | `0.4570` | `0.4754` | Strong seed sensitivity |
| `finbert_fullqa_pead20_mag7_seed21` | full Q&A, raw 512 | `0.4583` | `0.5172` | `0.4590` | Strong seed sensitivity |

### Conclusions

- The current failure is not explained by one simple bug. Truncation is severe, but giving the model more whole-call context in a naive way does not solve the problem.
- Full-Q&A document classification is consistently weaker than the analyst-only baseline under this small-data setup.
- The setup is unstable: the same full-Q&A FinBERT configuration moved from validation AUROC `0.5197` at seed `42` to `0.4676` at seed `7` and `0.4583` at seed `21`.
- Label changes alone do not rescue the classifier. Cleaner labels reduce sample size, while longer-horizon labels improve class balance metrics more than actual decision quality.
- The thresholded classifiers are poorly calibrated, with several runs predicting almost entirely one class despite nontrivial AUROC.

### Suggested next moves

1. Stop treating the whole Q&A as a single flat document. Build a question-answer level dataset and aggregate at call level with pooling or multiple-instance learning.
2. Replace binary classification-only training with a regression or ranking objective on CAR, then calibrate a threshold afterward.
3. Use structured windows around analyst questions and management answers instead of prefix truncation on full transcripts.
4. Expand beyond the current Mag7-only sample for training, then evaluate on a strict Mag7 holdout benchmark.
5. Add threshold tuning and probability calibration on the validation split before reporting accuracy or confusion matrices.

## 2026-04-13 QA-Pair Regression Pilot

### Setup

- Built `5746` analyst-question plus following-answer pairs across `392` calls.
- Average pairs per call: `14.66`; median pairs per call: `12`.
- Mean question length: `486` chars; mean answer length: `1447` chars.
- Rolling pilot used two expanding time splits with a FinBERT regression head on pair texts.
- Call-level text score used validation-selected simple pooling (`mean` or `max`).
- Upper-bound model used aggregated text scores plus available price-derived controls and metadata.
- Historical earnings surprise was unavailable in the local corpus, so surprise coverage was `0%`.

### Pilot results

| Benchmark | AUROC | AUPRC | Spearman | RMSE | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `0.4813` | `0.5008` | `-0.1253` | `0.1278` | `0.0344` | `0.5393` |
| `text_plus_tabular` | `0.4570` | `0.5108` | `-0.1239` | `0.1320` | `0.0418` | `0.5208` |

### Interpretation

- The QA-pair reformulation did not improve the pilot benchmark yet.
- Fold behavior was unstable: fold `0` was strongly negative, while fold `1` was closer to neutral and slightly positive only for the upper-bound rank correlation.
- The current pair regressor appears undertrained or mis-specified rather than merely undercalibrated.

### Caveat found during the pilot

- The first pilot run revealed `18` calls in `call_features` with zero extracted QA pairs.
- That created a mismatch where the text-only benchmark scored fewer calls than the upper-bound model in fold `0`.
- The experiment runner was patched after the pilot to exclude zero-pair calls from future rolling evaluation runs.

## 2026-04-13 QA-Pair Cleanup, Feature Refresh, and Broad-vs-Strict Comparison

### Changes made

- Added a strict QA-pair cleaning profile that removes analyst-only answer spans, question-mark leakage inside answers, operator prompt leakage, and very short answers.
- Added call-level event-window features:
  - `pre_event_return_5d`
  - `post_event_return_3d`
- Kept `post_event_return_3d` out of the default benchmark because it overlaps the PEAD target window and would leak label information.
- Added an `earnings_fundamentals` build step with support for reported EPS, expected EPS, surprise, and beat/miss fields.
- Preserved `glopardo` forward/trailing EPS proxy fields even though true event-level surprise coverage remains `0%`.

### Corpus comparison

| Corpus | Pair Rows | Calls | Retention | Mgmt Answer Rate | Analyst-Only Answer Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `broad` | `5746` | `392` | `1.0000` | `0.5414` | `0.4586` |
| `strict` | `1879` | `264` | `0.3270` | `1.0000` | `0.0000` |

### Benchmark comparison

| Benchmark | Corpus | AUROC | AUPRC | Spearman | RMSE | ECE | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `broad` | `0.4405` | `0.5110` | `-0.0340` | `0.1292` | `0.0521` | `0.4896` |
| `text_only` | `strict` | `0.4764` | `0.5201` | `0.0833` | `0.1202` | `0.0278` | `0.5139` |
| `text_plus_tabular` | `broad` | `0.4880` | `0.5727` | `-0.0560` | `0.1496` | `0.0409` | `0.5000` |
| `text_plus_tabular` | `strict` | `0.5614` | `0.6410` | `0.1810` | `0.1215` | `0.0279` | `0.5139` |

### Method note

- Broad comparison reused the existing broad text checkpoints and refreshed scoring plus the tabular benchmark on the updated feature table.
- Strict comparison retrained the text model end to end on the cleaned corpus.
- Because strict keeps fewer calls, this is directionally informative rather than a perfectly controlled sample-size comparison.

### Interpretation

- The strict corpus is the stronger benchmark.
- QA cleanup improved both ranking and calibration.
- The largest improvement is on `text_plus_tabular`, which suggests the cleaner call segmentation made the text signal more usable when combined with price-derived controls.
- `pre_event_return_5d` is now in the benchmark and likely helps, but the main bottleneck remains missing true event-level earnings surprise data.

### Remaining blocker

- Reported-vs-consensus EPS and revenue fields are still at `0%` coverage in the current local data.
- The code path is ready; we now need an external event-level earnings estimates file to make beat/miss and surprise features real inputs rather than placeholders.
