from __future__ import annotations

import argparse

import pandas as pd

from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


REQUIRED_COLS = {"call_id", "ticker", "event_date", "speaker_role", "section", "text"}


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def parse_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required transcript columns: {sorted(missing)}")

    out = df.copy()
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    out["speaker_role"] = out["speaker_role"].astype(str).str.lower().str.strip()
    out["section"] = out["section"].astype(str).str.lower().str.strip()
    out["text"] = out["text"].map(normalize_text)
    if "turn_id" not in out.columns:
        out["turn_id"] = out.groupby("call_id").cumcount() + 1

    keep_cols = [
        "call_id",
        "ticker",
        "event_date",
        "turn_id",
        "speaker_role",
        "section",
        "text",
    ]
    return out[keep_cols].sort_values(["call_id", "turn_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/transcripts/transcripts.csv")
    parser.add_argument("--output", default="data/interim/parsed_calls/parsed_calls.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    parsed = parse_transcripts(df)
    ensure_dir("data/interim/parsed_calls")
    parsed.to_csv(args.output, index=False)
    LOGGER.info("Saved parsed transcripts to %s", args.output)


if __name__ == "__main__":
    main()
