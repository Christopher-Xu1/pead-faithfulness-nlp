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

## 2026-04-14 Strict Call-Level Ablation: Threshold Tuning vs Rich Aggregation vs Boosted Classifier

### Changes made

- Added richer call-level text aggregation features built from pair-score sequences:
  - distributional summaries (`median`, quantiles, spread, top/bottom means)
  - order-aware summaries (`first`, `last`, `last_minus_first`, slope, recent-weighted mean)
  - pair-shape summaries (question length, answer length, answer-turn counts, management/mixed-role rates)
- Added validation threshold tuning for call-level classification metrics instead of using a fixed `0.5` cutoff.
- Added a call-level boosted classifier benchmark on the strict corpus.
- Reused the previously trained strict FinBERT pair checkpoints so the ablation isolates the call-level changes rather than re-running text-model training.

### Benchmark comparison

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `0.4726` | `0.5212` | `0.4306` | `0.0281` | `0.0278` |
| `text_tabular_ridge_base` | `0.5614` | `0.6410` | `0.5139` | `0.1810` | `0.0279` |
| `text_tabular_ridge_base_tuned` | `0.5614` | `0.6410` | `0.6250` | `0.1810` | `0.0279` |
| `text_tabular_ridge_rich_tuned` | `0.4988` | `0.5785` | `0.5139` | `0.0860` | `0.0416` |
| `text_tabular_boosted_rich_tuned` | `0.5297` | `0.5498` | `0.5139` | `0.0261` | `0.2626` |

### Interpretation

- Threshold tuning was immediately useful and raised strict call-level accuracy from `0.5139` to `0.6250` without changing AUROC or AUPRC.
- The original `mean+max` Ridge benchmark remains the strongest ranking model on the current strict sample.
- Hand-crafted rich aggregation features did not help on this sample and likely overfit the small number of calls.
- The boosted classifier underperformed the Ridge baseline and was much less calibrated.

### Next implication

- The failure of the hand-crafted rich aggregation does not rule out sequence modeling; it argues against this specific manually engineered feature set.
- The next serious test should be a learned sequential aggregator over pair embeddings or pair scores:
  - tiny GRU over ordered QA pairs
  - lightweight Transformer encoder over pair embeddings with positional order
  - attention pooling that can upweight late-call guidance or follow-up clarifications
- That should be attempted only after adding real event-level earnings surprise data, because the current tabular side is still missing the strongest known PEAD covariates.

## 2026-04-14 Sequence Aggregation and Speaker-Role Feature Pass

### Changes made

- Added explicit speaker-role structure to the QA-pair dataset:
  - per-pair management and analyst turn counts
  - per-pair management and analyst turn shares
  - per-pair role-switch counts
  - whether an answer starts or ends with management vs analyst
- Added call-level aggregates of those role features to `call_features`.
- Added two compact call-sequence models over ordered QA pairs:
  - `GRU`
  - `attention pooling`
- Each sequence model uses ordered pair features rather than only pooled scores:
  - pair score
  - pair position
  - question and answer lengths
  - answer-turn count
  - coarse speaker-role features
- Kept the tuned Ridge tabular model as the baseline-to-beat and used each sequence model's learned call score as the text input to the same Ridge-plus-tabular benchmark.

### Benchmark comparison

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_tabular_ridge_base_tuned` | `0.5614` | `0.6410` | `0.6250` | `0.1810` | `0.0279` |
| `text_tabular_ridge_gru_tuned` | `0.5019` | `0.5713` | `0.5000` | `0.1432` | `0.0279` |
| `text_tabular_ridge_attention_tuned` | `0.5251` | `0.5679` | `0.5972` | `0.1428` | `0.0278` |

### Interpretation

- The tuned Ridge baseline remains the strongest model in this round on both `AUROC` and `AUPRC`.
- The GRU sequence model did not improve ranking or classification quality on the current strict sample.
- Attention pooling was more stable than the GRU and recovered much of the baseline accuracy, but still did not beat the tuned Ridge baseline on ranking quality.
- Coarse speaker-role features are now present in the dataset and available to later models, but they were not enough on their own to lift the benchmark materially.

### Earnings surprise status

- Added a Yahoo-based fetch path for event-level EPS surprise and beat/miss enrichment in `src/data/fetch_yfinance_earnings_events.py`.
- The fetch path is implemented, but Yahoo rate limited the request in this run, so the local `earnings_fundamentals.csv` table still has `0%` coverage for real EPS surprise fields.
- This means the benchmark above should be read as a sequence-and-speaker-feature test, not yet the full sequence-plus-real-surprise benchmark.

## 2026-04-15 Tech Large-Cap Universe Expansion

### Motivation

- The Mag7 strict corpus is too small for stable learned sequence aggregation.
- The current strict setup has only `264` calls with strict QA pairs, which is a major variance source for GRU and attention models.
- The next training corpus should expand the universe while preserving Mag7 as a clean evaluation slice.

### Universe construction

- Added `src/data/build_universe_subset.py` to build a reproducible universe snapshot from the public Nasdaq stock screener.
- Current screen: S&P GICS `Information Technology`, market cap at least `$10B` using Nasdaq screener market caps, intersected with the local curated `gold_corpus`.
- Forced included all Mag7 tickers so AMZN and TSLA remain in scope even though Nasdaq does not classify them under `Technology`.
- Snapshot file: `configs/data/universes/tech_largecap_nasdaq_2026-04-15.csv`.
- Summary file: `reports/tech_largecap_universe_summary.md`.

### Dataset size

| Dataset | Tickers | Curated Calls | Model-Ready Calls | Strict Pair Rows | Calls With Strict Pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mag7` | `7` | `410` | `401` | `1879` | `264` |
| `tech_largecap` | `69` | `3204` | `3178` | `15683` | `1760` |

### Quality notes

- Tech-largecap parser audit passed on `99.2%` of calls.
- Tech-largecap labels are moderately positive-skewed: all model-ready calls have label distribution `{0: 1359, 1: 1819}`.
- Strict QA-pair retention is `29.9%`; retained answers have mean management turn share `87.5%`.
- The expanded strict setup supports `20` rolling folds with `400` initial train calls, `64` validation calls, and `64` test calls per fold.
- True reported-vs-consensus EPS surprise coverage is still `0%`.
- `glopardo` forward EPS proxy coverage improves to `69.4%` on this expanded universe.

### Size and call-frequency controls

- Added frozen-universe company size fields to both call-level and pair-level tables:
  - `snapshot_market_cap_usd`
  - `snapshot_log_market_cap`
  - `snapshot_market_cap_percentile`
  - `universe_sector`
  - `universe_industry`
  - `universe_included_by`
- Added time-safe call-frequency fields computed only from prior calls for the same ticker:
  - `ticker_prior_call_count`
  - `ticker_prior_call_count_365d`
  - `ticker_days_since_prev_call`
  - `ticker_mean_prior_call_gap_days`
- Market-cap coverage is `100%` on the expanded strict calls.
- Prior days-since-call coverage is `97.6%` on expanded strict calls because the first observed call for each ticker has no prior call.
- Median prior-call count on expanded strict calls is `18`; median days since the previous observed call is `91`.
- These fields are now included in the tabular baseline and repeated as context features for GRU/attention sequence models.

### Infrastructure changes

- Added `configs/data/model_ready_tech_largecap.yaml` so the expanded build writes to separate paths and does not overwrite Mag7 artifacts.
- Added `configs/experiment/qa_pair_regression_tech_largecap_strict.yaml` for the first expanded strict FinBERT QA-pair benchmark.
- Added `scripts/run_model_ready_tech_largecap_pipeline.sh`.
- Patched `src/data/build_model_ready_pead.py` to support configurable output paths and a direct Yahoo Chart price provider.
- Stooq CSV downloads now require an API key, so the expanded build uses Yahoo Chart daily adjusted prices for this run.

### Next experiment

- Train `qa_pair_finbert_regression_tech_largecap_strict`.
- Compare against the current tuned Mag7 strict Ridge benchmark.
- After training on the larger universe, evaluate performance on a held-out Mag7-only slice to test whether broader training improves the target benchmark rather than only the broader sample.

## 2026-04-15 SEC Actuals and Historical Market-Cap Snapshot Pass

### Changes made

- Added `src/data/build_sec_event_snapshots.py` to build event-aligned SEC Company Facts snapshots for the expanded tech-largecap corpus.
- Added clean historical market-cap snapshots:
  - latest Yahoo close strictly before the earnings-call date
  - split-unadjusted close reconstructed from Yahoo split events
  - latest SEC `dei:EntityCommonStockSharesOutstanding` fact with fact end date and filing date on or before the call
  - share-count outlier filtering to remove obvious XBRL scale artifacts
- Added SEC actual reported revenue and capex fields from quarterly-ish XBRL facts.
- Wired the new fields through:
  - `configs/data/model_ready_tech_largecap.yaml`
  - `data/raw/metadata/call_metadata_tech_largecap.csv`
  - `data/external/earnings_fundamentals/earnings_fundamentals_tech_largecap.csv`
  - `outputs/datasets/qa_pairs_tech_largecap_strict/call_features.csv`
  - `outputs/datasets/qa_pairs_tech_largecap_strict/qa_pair_dataset.csv`
- Updated the tech-largecap pipeline script so one command rebuilds the universe, SEC snapshots, model-ready calls, and strict QA-pair dataset.

### Coverage after rebuild

| Field group | Coverage |
| --- | ---: |
| Historical market cap | `79.7%` |
| SEC reported revenue actual | `86.3%` |
| SEC reported capex actual | `79.9%` |
| Frozen-universe market cap snapshot | `100.0%` |
| Pre-event 5-day return | `98.4%` |
| Post-event 3-day return | `100.0%` |
| `glopardo` forward EPS proxy | `69.4%` |
| EPS surprise / beat-miss | `0.0%` |
| Revenue surprise / beat-miss | `0.0%` |
| Capex surprise / beat-miss | `0.0%` |

### Interpretation

- We now have clean company-size controls both as frozen current-universe snapshots and as historical event-time market caps.
- SEC gives actual reported revenue and capex coverage, not analyst expectations. These are useful controls, but they are not true surprise variables by themselves.
- EPS, revenue, and capex surprise coverage remains `0%` because the local pipeline still lacks a reliable consensus estimates source.
- The next data-quality upgrade should be an estimates provider for consensus EPS/revenue and, if available, capex expectations. Without that, the benchmark has actual fundamentals but still misses the core beat/miss surprise covariates.

## 2026-04-15 Historical Estimates Ingestion Pass

### Changes made

- Added multi-source event fundamentals merging so SEC actuals and external estimates can be combined instead of replacing each other.
- Added `src/data/fetch_fmp_earnings_estimates.py` for historical FMP earnings estimate ingestion.
- The FMP path is designed to populate:
  - `estimated_eps`
  - `estimated_revenue`
  - `eps_surprise`
  - `revenue_surprise`
- Added an explicitly marked capex proxy:
  - `estimated_capex`
  - `estimated_capex_is_proxy`
  - `estimated_capex_method`
  - `prior_capex_to_revenue_ratio`
- The capex proxy uses only prior actual capex-to-revenue intensity for the same ticker multiplied by current estimated revenue.
- Updated `configs/data/model_ready_tech_largecap.yaml` to load both SEC actuals and FMP estimates when the FMP output file exists.
- Updated `scripts/run_model_ready_tech_largecap_pipeline.sh` to fetch FMP estimates only when `FMP_API_KEY` or `FINANCIALMODELINGPREP_API_KEY` is set.

### Current status

- The implementation is ready, but local estimated revenue/capex coverage is still not complete because no FMP API key is available in this environment.
- Yahoo Finance is still unreliable here because the local session is rate limited and Yahoo's free surfaces do not expose robust historical revenue/capex consensus for our full backtest.
- TradingView is not wired as an automated fetch source because there is no stable free historical estimates API in the current repo environment. TradingView exports can still be ingested if converted to the documented external earnings schema.

### Methodology note

- Treat FMP `estimated_revenue` as a real historical analyst revenue estimate if the FMP endpoint returns it.
- Treat `estimated_capex` from this pass as a proxy unless `estimated_capex_is_proxy == 0`.
- Do not describe proxy capex as analyst consensus capex in the paper.

## 2026-04-20 HF EPS-Surprise Backfill and Quick Tech-Largecap Retrain

### Source audit result

- Public/free local sources still do not complete historical revenue or capex consensus estimates.
- The usable free backfill found for this pass is Hugging Face `sovai/earnings_surprise`, which provides historical reported EPS, estimated EPS, and EPS surprise fields.
- FMP remains the preferred next source for historical estimated revenue and revenue surprise, but it requires `FMP_API_KEY` or `FINANCIALMODELINGPREP_API_KEY`.
- Capex consensus remains unavailable from the practical free sources checked; capex expectations should be treated as missing unless a paid estimates source is added. The implemented capex proxy remains explicitly marked as a proxy.

### Data completion after rebuild

| Field group | Coverage |
| --- | ---: |
| Strict QA pairs | `15683` rows |
| Calls with strict pairs | `1760` calls |
| Pair retention rate | `29.3%` |
| Pre-event 5-day return | `98.4%` |
| Post-event 3-day return | `100.0%` |
| Frozen-universe market cap snapshot | `100.0%` |
| Historical event-time market cap | `79.7%` |
| SEC reported revenue actual | `86.3%` |
| SEC reported capex actual | `79.9%` |
| Reported EPS / estimated EPS / EPS surprise | `50.4%` |
| Revenue estimate / revenue surprise | `0.0%` |
| Capex estimate / capex surprise | `0.0%` |
| `glopardo` forward EPS proxy | `69.4%` |

### Quick retrain

- Config: `configs/experiment/qa_pair_regression_tech_largecap_strict_eps_quick.yaml`
- Report: `reports/qa_pair_regression_tech_largecap_strict_eps_quick_report.md`
- Output: `outputs/models/qa_pair_finbert_regression_tech_largecap_strict_eps_quick`
- Scope: one rolling fold, `0.15` FinBERT epoch, intended as a fast pipeline validation and directional comparison.
- Full 20-fold tech-largecap benchmark was not completed locally because the estimated runtime on the Mac would be several hours.

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `0.4798` | `0.5803` | `0.5000` | `-0.1229` | `0.0000` |
| `text_tabular_ridge_base_tuned` | `0.4433` | `0.5967` | `0.4531` | `-0.0006` | `0.0001` |
| `text_tabular_ridge_gru_tuned` | `0.5202` | `0.6580` | `0.4844` | `0.1628` | `0.0006` |
| `text_tabular_ridge_attention_tuned` | `0.5232` | `0.6476` | `0.4688` | `0.1477` | `0.0006` |
| `text_tabular_ridge_rich_tuned` | `0.4571` | `0.6196` | `0.4375` | `0.0072` | `0.0000` |
| `text_tabular_boosted_rich_tuned` | `0.5202` | `0.6584` | `0.5156` | `0.1668` | `0.2442` |

### Interpretation

- The backfilled EPS surprise controls make the expanded tech-largecap strict dataset more useful, but the event fundamentals are still incomplete for a benchmark that claims revenue surprise controls.
- On the quick fold, learned sequence aggregation is directionally better than the same-fold Ridge mean+max baseline: GRU and attention both improve AUROC/AUPRC, and boosted rich aggregation has the best AUPRC.
- Calibration is still weak for the boosted model, so AUROC/AUPRC gains should not be overinterpreted without the full rolling benchmark.
- The next required data task is to add a credentialed estimates source, rebuild revenue surprise coverage, then run the full `configs/experiment/qa_pair_regression_tech_largecap_strict.yaml` benchmark.

## 2026-04-21 Fast 20-Fold Tech-Largecap Stability Benchmark

### Run status

- Config: `configs/experiment/qa_pair_regression_tech_largecap_strict_eps_fast20.yaml`
- Report: `reports/qa_pair_regression_tech_largecap_strict_eps_fast20_report.md`
- Output: `outputs/models/qa_pair_finbert_regression_tech_largecap_strict_eps_fast20`
- Scope: `20` expanding rolling folds, `64` validation calls and `64` test calls per fold.
- Text model budget: fast FinBERT pair-regression pass with `0.15` epoch per fold.
- Completed successfully at `2026-04-21 00:20`.

### Aggregate results

| Benchmark | AUROC | AUPRC | Accuracy | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `0.4913` | `0.5859` | `0.5531` | `-0.0157` | `0.0633` |
| `text_tabular_ridge_base_tuned` | `0.4905` | `0.5821` | `0.5281` | `0.1203` | `0.0670` |
| `text_tabular_ridge_gru_tuned` | `0.4914` | `0.5803` | `0.5305` | `0.1390` | `0.0659` |
| `text_tabular_ridge_attention_tuned` | `0.4919` | `0.5779` | `0.5359` | `0.1175` | `0.0657` |
| `text_tabular_ridge_rich_tuned` | `0.4904` | `0.5823` | `0.5336` | `0.1218` | `0.0644` |
| `text_tabular_boosted_rich_tuned` | `0.5307` | `0.6238` | `0.5352` | `0.0773` | `0.1290` |

### Interpretation

- The quick-fold improvement did not hold across the full rolling evaluation.
- The best aggregate AUPRC is `0.6238` from boosted rich aggregation, below the prior Mag7 strict tuned Ridge AUPRC of `0.6410`.
- Boosted rich aggregation has the best AUROC/AUPRC in this run but is less calibrated (`ECE 0.1290`) and has weaker rank correlation than the Ridge/sequence variants.
- GRU and attention do not beat the simpler Ridge/rich tabular variants on aggregate AUPRC, although GRU has the strongest Spearman among the non-boosted models.
- This result argues that added universe size alone is not enough; the next high-value step is better event fundamentals coverage, especially revenue surprise, plus a stronger full-epoch or external-compute run if we want to evaluate the text encoder more fairly.
