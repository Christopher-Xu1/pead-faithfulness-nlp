from __future__ import annotations

import argparse

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def extract_qa(parsed_df: pd.DataFrame, analyst_only: bool = True) -> pd.DataFrame:
    df = parsed_df.copy()
    in_qa = df["section"].str.contains("q", case=False, na=False)
    qa_df = df[in_qa]

    if analyst_only:
        qa_df = qa_df[qa_df["speaker_role"].str.contains("analyst", case=False, na=False)]

    grouped = (
        qa_df.sort_values(["call_id", "turn_id"])
        .groupby(["call_id", "ticker", "event_date"], as_index=False)
        .agg(text=("text", " ".join), num_questions=("text", "count"))
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/parsed_calls/parsed_calls.csv")
    parser.add_argument("--output", default="data/interim/qa_only/qa_dataset.csv")
    parser.add_argument("--include-management", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    qa_df = extract_qa(df, analyst_only=not args.include_management)
    ensure_dir("data/interim/qa_only")
    qa_df.to_csv(args.output, index=False)
    LOGGER.info("Saved Q&A dataset to %s", args.output)


if __name__ == "__main__":
    main()
