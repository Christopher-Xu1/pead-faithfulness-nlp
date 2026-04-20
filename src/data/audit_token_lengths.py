from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.tokenizer import get_tokenizer
from src.utils.io import ensure_dir, load_yaml, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def summarize_lengths(lengths: pd.Series, split: str, column: str) -> dict[str, float | int | str]:
    return {
        "split": split,
        "text_column": column,
        "rows": int(len(lengths)),
        "mean_tokens": float(lengths.mean()),
        "median_tokens": float(lengths.median()),
        "p90_tokens": float(lengths.quantile(0.9)),
        "p95_tokens": float(lengths.quantile(0.95)),
        "max_tokens": int(lengths.max()),
        "share_over_256": float((lengths > 256).mean()),
        "share_over_512": float((lengths > 512).mean()),
        "share_over_1024": float((lengths > 1024).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", nargs="+", default=["text", "full_qa_text"])
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiment_config)
    model_cfg = load_yaml(exp_cfg["model_config"])
    tokenizer = get_tokenizer(model_cfg.get("model_name", "roberta-base"))

    summaries: list[dict[str, float | int | str]] = []
    split_frames: dict[str, pd.DataFrame] = {
        "train": pd.read_csv(exp_cfg["train_path"]),
        "val": pd.read_csv(exp_cfg["val_path"]),
        "test": pd.read_csv(exp_cfg["test_path"]),
    }

    for split, df in split_frames.items():
        for column in args.columns:
            if column not in df.columns:
                continue
            texts = df[column].fillna("").astype(str).tolist()
            lengths = pd.Series(
                [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts],
                name="token_length",
            )
            summaries.append(summarize_lengths(lengths, split=split, column=column))

    out_dir = ensure_dir(args.output_dir)
    summary_df = pd.DataFrame(summaries).sort_values(["text_column", "split"]).reset_index(drop=True)
    write_csv(summary_df, out_dir / "token_length_summary.csv")
    save_json(summary_df.to_dict(orient="records"), out_dir / "token_length_summary.json")
    LOGGER.info("Saved token length audit to %s", out_dir)


if __name__ == "__main__":
    main()
