from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from src.utils.io import load_json, write_csv
from src.utils.plotting import save_fold_performance_grid, save_residual_target_scatter


def _metric_from_ci_cell(cell: str) -> float:
    token = str(cell).strip().split()[0]
    return float(token)


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_report_line(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    raise ValueError(f"Could not find line starting with {prefix!r}")


def _parse_report_scalar(text: str, label: str) -> float:
    pattern = rf"- {re.escape(label)}: `([^`]+)`"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find scalar label {label!r}")
    return float(match.group(1))


def _parse_report_row(text: str, benchmark: str) -> dict[str, float]:
    line = _parse_report_line(text, f"| {benchmark} |")
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return {
        "auroc": _metric_from_ci_cell(parts[1]),
        "auprc": _metric_from_ci_cell(parts[2]),
        "spearman": _metric_from_ci_cell(parts[3]),
        "pearson": _metric_from_ci_cell(parts[4]),
        "rmse": _metric_from_ci_cell(parts[5]),
        "accuracy": _metric_from_ci_cell(parts[8]),
    }


def _parse_dataset_summary(text: str) -> dict[str, float]:
    labels = {
        "pair_rows": "Pair rows",
        "calls_with_pairs": "Calls with pairs",
        "earnings_surprise_coverage": "Earnings surprise coverage",
        "revenue_surprise_coverage": "Revenue surprise coverage",
        "rolling_eval_calls": "Calls used in rolling eval",
    }
    return {key: _parse_report_scalar(text, label) for key, label in labels.items()}


def _load_conditional_residual_summary(root: Path, input_summary_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    model_dir = root / "outputs" / "models" / "conditional_residual_qa_pead"
    overall_metrics = load_json(model_dir / "overall_metrics.json")
    predictions = pd.read_csv(model_dir / "overall_test_call_predictions.csv")
    input_summary = json.loads(input_summary_path.read_text(encoding="utf-8"))
    overall = overall_metrics["overall_test_metrics"]
    overall["spearman"] = float(
        spearmanr(
            pd.to_numeric(predictions["pead_target"], errors="coerce"),
            pd.to_numeric(predictions["final_pred"], errors="coerce"),
        ).correlation
    )
    summary = {
        "benchmark": "conditional_residual_simple",
        "report_label": "Conditional residual simple",
        "universe": "tech_largecap_strict",
        "calls": int(input_summary.get("calls_with_pairs", input_summary.get("normalized_calls_with_pairs", len(predictions)))),
        "pairs": int(input_summary.get("pair_rows", input_summary.get("normalized_pair_rows", 0))),
        "folds": int(overall_metrics["fold_count"]),
        "earnings_surprise_coverage": float(input_summary.get("earnings_surprise_coverage", float("nan"))),
        "revenue_surprise_coverage": float(input_summary.get("revenue_surprise_coverage", float("nan"))),
        "auroc": float(overall["AUROC"]),
        "auprc": float(overall["AUPRC"]),
        "accuracy": float(overall["accuracy"]),
        "spearman": float(overall["spearman"]),
        "pearson": float(overall["correlation_with_pead_target"]),
        "rmse": float(overall["RMSE"]),
        "notes": "Residual pipeline with fundamentals baseline plus FinBERT residual aggregator.",
    }
    return summary, predictions


def _write_markdown_report(path: Path, rows: list[dict[str, Any]], figure_dir: Path) -> None:
    lines = [
        "# Conditional Residual Benchmark Comparison",
        "",
        "## Aggregate Comparison",
        "",
        "| Benchmark | Universe | Calls | Pairs | Folds | EPS Surprise Cov. | Revenue Surprise Cov. | AUROC | AUPRC | Accuracy | Spearman | Pearson | RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['report_label']} | {row['universe']} | {row['calls']} | {row['pairs']} | {row['folds']} | "
            f"{row['earnings_surprise_coverage']:.4f} | {row['revenue_surprise_coverage']:.4f} | "
            f"{row['auroc']:.4f} | {row['auprc']:.4f} | {row['accuracy']:.4f} | {row['spearman']:.4f} | "
            f"{row['pearson']:.4f} | {row['rmse']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The conditional residual run improves on the previous best tech-largecap fast-20 benchmark in AUROC, AUPRC, and accuracy.",
            "- The conditional residual run also improves Spearman rank correlation versus the prior best tech-largecap benchmark, while Pearson remains below the older Mag7 strict ridge benchmark.",
            "- The comparison is not perfectly apples-to-apples because the prepared conditional residual bundle has materially richer EPS and revenue surprise coverage.",
            "",
            "## Generated Figures",
            "",
            f"- Per-fold performance grid: `{figure_dir / 'conditional_residual_per_fold_performance.png'}`",
            f"- Residual target scatter: `{figure_dir / 'conditional_residual_residual_vs_target_scatter.png'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--prepared-input-summary",
        default="transfer/pc_resume_bundle/outputs/datasets/conditional_residual_qa_pead/input_summary.json",
    )
    parser.add_argument("--comparison-csv", default="reports/conditional_residual_benchmark_comparison.csv")
    parser.add_argument("--comparison-md", default="reports/conditional_residual_benchmark_comparison.md")
    parser.add_argument("--fold-plot", default="outputs/figures/conditional_residual_per_fold_performance.png")
    parser.add_argument("--scatter-plot", default="outputs/figures/conditional_residual_residual_vs_target_scatter.png")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    strict_report_text = _read_text(root / "reports" / "qa_pair_regression_strict_report.md")
    fast20_report_text = _read_text(root / "reports" / "qa_pair_regression_tech_largecap_strict_eps_fast20_report.md")

    strict_dataset = _parse_dataset_summary(strict_report_text)
    fast20_dataset = _parse_dataset_summary(fast20_report_text)
    strict_metrics = _parse_report_row(strict_report_text, "text_plus_tabular")
    fast20_metrics = _parse_report_row(fast20_report_text, "text_tabular_boosted_rich_tuned")

    comparison_rows: list[dict[str, Any]] = [
        {
            "benchmark": "text_plus_tabular",
            "report_label": "Strict Mag7 text_plus_tabular",
            "universe": "mag7_strict",
            "calls": int(strict_dataset["rolling_eval_calls"]),
            "pairs": int(strict_dataset["pair_rows"]),
            "folds": 3,
            "earnings_surprise_coverage": float(strict_dataset["earnings_surprise_coverage"]),
            "revenue_surprise_coverage": float(strict_dataset["revenue_surprise_coverage"]),
            "auroc": strict_metrics["auroc"],
            "auprc": strict_metrics["auprc"],
            "accuracy": strict_metrics["accuracy"],
            "spearman": strict_metrics["spearman"],
            "pearson": strict_metrics["pearson"],
            "rmse": strict_metrics["rmse"],
            "notes": "Older strict Mag7 ridge upper-bound benchmark.",
        },
        {
            "benchmark": "text_tabular_boosted_rich_tuned",
            "report_label": "Tech-largecap boosted_rich_tuned",
            "universe": "tech_largecap_strict",
            "calls": int(fast20_dataset["rolling_eval_calls"]),
            "pairs": int(fast20_dataset["pair_rows"]),
            "folds": 20,
            "earnings_surprise_coverage": float(fast20_dataset["earnings_surprise_coverage"]),
            "revenue_surprise_coverage": float(fast20_dataset["revenue_surprise_coverage"]),
            "auroc": fast20_metrics["auroc"],
            "auprc": fast20_metrics["auprc"],
            "accuracy": fast20_metrics["accuracy"],
            "spearman": fast20_metrics["spearman"],
            "pearson": fast20_metrics["pearson"],
            "rmse": fast20_metrics["rmse"],
            "notes": "Previous best expanded tech-largecap QA-pair benchmark.",
        },
    ]

    conditional_summary, predictions = _load_conditional_residual_summary(
        root=root,
        input_summary_path=root / args.prepared_input_summary,
    )
    comparison_rows.append(conditional_summary)

    comparison_df = pd.DataFrame(comparison_rows)
    write_csv(comparison_df, root / args.comparison_csv)
    _write_markdown_report(root / args.comparison_md, comparison_rows, root / "outputs" / "figures")

    fold_df = pd.read_csv(root / "outputs" / "models" / "conditional_residual_qa_pead" / "fold_metrics.csv")
    save_fold_performance_grid(fold_df, str(root / args.fold_plot))
    save_residual_target_scatter(predictions, str(root / args.scatter_plot))


if __name__ == "__main__":
    main()
