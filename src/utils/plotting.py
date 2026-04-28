from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_deletion_curve_plot(curve_df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve_df["fraction_removed"], curve_df["score"], marker="o")
    ax.set_xlabel("Fraction of rationale tokens removed")
    ax.set_ylabel("Target probability")
    ax.set_title("Deletion Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_fold_performance_grid(fold_df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("test_AUROC", "val_AUROC", "AUROC"),
        ("test_AUPRC", "val_AUPRC", "AUPRC"),
        ("test_accuracy", "val_accuracy", "Accuracy"),
        ("test_RMSE", "val_RMSE", "RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (test_col, val_col, title) in zip(axes.flatten(), metric_specs):
        ax.plot(fold_df["fold"], fold_df[test_col], marker="o", linewidth=1.8, label="test", color="#1f4e79")
        ax.plot(fold_df["fold"], fold_df[val_col], marker="s", linewidth=1.2, label="val", color="#d97706")
        ax.axhline(float(pd.to_numeric(fold_df[test_col], errors="coerce").mean()), linestyle="--", color="#64748b", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Conditional Residual Per-Fold Performance", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_target_scatter(pred_df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    target = pd.to_numeric(pred_df["residual_target"], errors="coerce")
    predicted = pd.to_numeric(pred_df["final_pred"], errors="coerce") - pd.to_numeric(
        pred_df["baseline_pred"], errors="coerce"
    )
    valid = target.notna() & predicted.notna()
    target = target[valid]
    predicted = predicted[valid]

    pearson = float(target.corr(predicted, method="pearson")) if len(target) else float("nan")
    spearman = float(target.corr(predicted, method="spearman")) if len(target) else float("nan")
    lower = float(min(target.min(), predicted.min())) if len(target) else -1.0
    upper = float(max(target.max(), predicted.max())) if len(target) else 1.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(target, predicted, s=18, alpha=0.45, color="#0f766e", edgecolors="none")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#7f1d1d", linewidth=1.1)
    ax.set_xlabel("Residual Target")
    ax.set_ylabel("Predicted Residual")
    ax.set_title("Residual Prediction vs Residual Target")
    ax.text(
        0.03,
        0.97,
        f"Pearson = {pearson:.4f}\nSpearman = {spearman:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cbd5e1"},
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
