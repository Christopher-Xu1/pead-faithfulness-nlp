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
