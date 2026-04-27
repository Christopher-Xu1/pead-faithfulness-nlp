# Conditional Residual QA PEAD Pipeline

## What It Is

`conditional_residual_qa_pead` is a parallel pipeline that keeps the existing strongest QA-pair PEAD benchmark intact and adds a separate residual-learning path:

1. Fit a fundamentals-only Ridge model at the call level.
2. Use that model to produce `baseline_pred`.
3. Define `residual_target = pead_target - baseline_pred`.
4. Train a FinBERT QA-pair regressor to predict the residual, not raw PEAD.
5. Aggregate pair-level residual predictions back to the call level.
6. Form the final call prediction as `baseline_pred + aggregated_text_residual`.

## How It Differs From The Existing Benchmark

- The existing QA-pair benchmark learns from raw call-level PEAD labels.
- This pipeline decomposes the task into a fundamentals baseline plus text residual.
- The pair model is conditioned on `[SUE_EPS, SUE_REV, baseline_pred]` and is trained against `residual_target`.
- The old benchmark remains runnable for direct comparison and is not replaced.

## Exact Training Objective

- Call-level target: `pead_target`
- Fundamentals model target: `pead_target`
- Text model target: `residual_target = pead_target - baseline_pred`
- Pair loss: mean squared error between `pair_residual_pred` and `residual_target`
- Default final prediction: `final_pred = baseline_pred + mean_pair_residual`
- Optional final stage: Ridge over `[baseline_pred, mean_pair_residual, max_pair_residual]`

## Expected Input Columns

Call-level dataframe:

- `call_id`
- `ticker`
- `call_date`
- `pead_target`
- `SUE_EPS`
- `SUE_REV`
- `pre_event_return`
- `volatility`
- `market_cap`
- `qa_count`

Pair-level dataframe:

- `call_id`
- `pair_id`
- `pair_index`
- `question_text`
- `answer_text`

Training merges the call-level fields into each pair row so the model sees:

- `call_id`
- `pair_id`
- `question_text`
- `answer_text`
- `SUE_EPS`
- `SUE_REV`
- `baseline_pred`
- `residual_target`

When using the repo’s existing raw builders, these normalized inputs are written to:

- `outputs/datasets/conditional_residual_qa_pead/call_level_inputs.csv`
- `outputs/datasets/conditional_residual_qa_pead/pair_level_inputs.csv`

## How To Run

Default run:

```bash
bash scripts/train_conditional_residual.sh
```

Validation only:

```bash
python -m src.experiments.conditional_residual_qa_pead --config configs/experiment/conditional_residual_qa_pead.yaml --validate-only
```

Direct run:

```bash
python -m src.experiments.conditional_residual_qa_pead --config configs/experiment/conditional_residual_qa_pead.yaml
```

Resume an interrupted run:

```bash
python -m src.experiments.conditional_residual_qa_pead --config configs/experiment/conditional_residual_qa_pead.yaml --resume
```

The default config uses the tech-largecap strict QA corpus and FinBERT.

## Leakage Controls

- Rolling folds are built on `call_id` ordered by `call_date`.
- All pairs from a call stay in the same split.
- The fundamentals Ridge model is fit only on train calls for each fold.
- `baseline_pred` for train, validation, and test is always generated from that train-fitted Ridge model.
- `residual_target` is computed separately inside each fold from the fold-specific baseline.
- The pair model is trained only on train-fold pairs.
- Validation and test pairs are never used when fitting the baseline, residual target, scaler, or pair model.

## Saved Outputs

Per fold under `outputs/models/conditional_residual_qa_pead/fold_XX/`:

- `train_pair_predictions.csv`
- `val_pair_predictions.csv`
- `test_pair_predictions.csv`
- `train_aggregated_residuals.csv`
- `val_aggregated_residuals.csv`
- `test_aggregated_residuals.csv`
- `train_call_predictions.csv`
- `val_call_predictions.csv`
- `test_call_predictions.csv`
- `metrics.json`
- `pair_model/training_metadata.json`

Saved call-level artifacts include:

- `baseline_pred`
- `residual_target`
- aggregated residual columns such as `mean_pair_residual` and `max_pair_residual`
- `final_pred`
- `pead_target`

Repo-level summaries:

- `outputs/models/conditional_residual_qa_pead/fold_metrics.csv`
- `outputs/models/conditional_residual_qa_pead/overall_test_call_predictions.csv`
- `outputs/models/conditional_residual_qa_pead/overall_metrics.json`
- `outputs/models/conditional_residual_qa_pead/validation_report.json`

## How To Compare Against The Old Pipeline

Run the old strongest QA benchmark with its existing config, then compare:

- overall AUROC
- overall AUPRC
- overall accuracy
- overall MSE
- overall RMSE
- overall `correlation_with_pead_target`

The old benchmark remains available through the existing `qa_pair_regression` configs and reports, while the new residual pipeline writes to its own dataset and model output directories.
