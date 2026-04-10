from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.transcript_corpus import is_mag7_ticker, iter_source_records
from src.utils.io import ensure_dir, load_yaml, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


DATASET_SCHEMA = pa.schema(
    [
        ("record_key", pa.string()),
        ("source_id", pa.string()),
        ("source_file", pa.string()),
        ("source_row", pa.int64()),
        ("ticker", pa.string()),
        ("canonical_ticker", pa.string()),
        ("event_date", pa.string()),
        ("company", pa.string()),
        ("year", pa.int32()),
        ("quarter", pa.string()),
        ("transcript", pa.string()),
        ("structured_content", pa.string()),
        ("has_qa_section", pa.int64()),
        ("analyst_question_count", pa.int64()),
        ("transcript_chars", pa.int64()),
        ("transcript_words", pa.int64()),
        ("quality_score", pa.int64()),
        ("soft_quality_flags", pa.string()),
    ]
)


def quality_score(df: pd.DataFrame) -> pd.Series:
    return (
        df["has_qa_section"] * 100
        + df["analyst_question_count"].clip(upper=10) * 10
        + df["has_structured_content"] * 20
        + df["valid_ticker"] * 20
        + df["valid_event_date"] * 20
        + df["source_priority"]
        + (df["transcript_chars"].clip(upper=200000) // 5000)
        - df["repeated_operator_text"] * 10
        - (df["non_ascii_ratio"] > 0.05).astype(int) * 10
    )


def _best_index(group: pd.DataFrame) -> int:
    ranked = group.sort_values(
        by=[
            "quality_score",
            "has_qa_section",
            "analyst_question_count",
            "transcript_chars",
            "has_structured_content",
            "source_priority",
            "record_key",
        ],
        ascending=[False, False, False, False, False, False, True],
        kind="mergesort",
    )
    return int(ranked.index[0])


def apply_deduplication(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df["quality_score"] = quality_score(df)
    df["drop_reason"] = ""
    df["duplicate_of"] = ""

    text_groups = df["text_hash"].notna() & (df.groupby("text_hash")["record_key"].transform("size") > 1)
    for _, group in df[text_groups].groupby("text_hash", sort=False):
        keep_idx = _best_index(group)
        keep_key = df.loc[keep_idx, "record_key"]
        for idx in group.index:
            if idx == keep_idx:
                continue
            df.loc[idx, "drop_reason"] = "exact_text_duplicate"
            df.loc[idx, "duplicate_of"] = keep_key

    remaining = df["drop_reason"] == ""
    key_groups = (
        remaining
        & df["ticker"].notna()
        & df["event_date"].notna()
        & (df.groupby(["ticker", "event_date"])["record_key"].transform("size") > 1)
    )
    for _, group in df[key_groups].groupby(["ticker", "event_date"], sort=False):
        keep_idx = _best_index(group)
        keep_key = df.loc[keep_idx, "record_key"]
        shortest = max(group["transcript_chars"].min(), 1)
        longest = group["transcript_chars"].max()
        version_like = (longest - shortest) / shortest <= 0.15
        drop_reason = "version_duplicate" if version_like else "ticker_date_duplicate"
        for idx in group.index:
            if idx == keep_idx:
                continue
            df.loc[idx, "drop_reason"] = drop_reason
            df.loc[idx, "duplicate_of"] = keep_key

    return df


def apply_quality_filters(metrics_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    df = metrics_df.copy()
    hard = cfg.get("hard_filters", {})
    min_chars = int(hard.get("min_transcript_chars", 5000))
    min_questions = int(hard.get("min_analyst_questions", 2))

    df["keep_clean_master"] = (
        (df["drop_reason"] == "")
        & (df["transcript_chars"] > 0)
        & (df["valid_ticker"] == 1)
        & (df["valid_event_date"] == 1)
    )
    df["keep_gold"] = (
        df["keep_clean_master"]
        & (df["has_qa_section"] == 1)
        & (df["transcript_chars"] >= min_chars)
        & (df["analyst_question_count"] >= min_questions)
    )

    hard_failures = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        if row["drop_reason"]:
            reasons.append(row["drop_reason"])
        if row["valid_ticker"] != 1:
            reasons.append("invalid_ticker")
        if row["valid_event_date"] != 1:
            reasons.append("invalid_event_date")
        if row["has_qa_section"] != 1:
            reasons.append("missing_qa")
        if row["transcript_chars"] < min_chars:
            reasons.append("short_transcript")
        if row["analyst_question_count"] < min_questions:
            reasons.append("too_few_analyst_questions")
        hard_failures.append(",".join(dict.fromkeys(reasons)))
    df["hard_filter_failures"] = hard_failures

    soft_flags = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        if row["analyst_question_count"] < max(min_questions + 1, 4):
            reasons.append("low_question_count")
        if row["repeated_operator_text"] == 1:
            reasons.append("repeated_operator_text")
        if row["non_ascii_ratio"] > 0.05:
            reasons.append("high_non_ascii_ratio")
        soft_flags.append(",".join(reasons))
    df["soft_quality_flags"] = soft_flags
    df["is_mag7"] = df["ticker"].map(is_mag7_ticker).astype(int)
    return df


def _materialize_dataset(
    raw_root: str | Path,
    decisions_df: pd.DataFrame,
    keep_column: str,
    output_path: str | Path,
) -> None:
    selected = decisions_df.loc[decisions_df[keep_column], [
        "record_key",
        "source_id",
        "ticker",
        "canonical_ticker",
        "event_date",
        "has_qa_section",
        "analyst_question_count",
        "transcript_chars",
        "transcript_words",
        "quality_score",
        "soft_quality_flags",
    ]]
    if selected.empty:
        LOGGER.warning("No rows matched %s; skipping %s", keep_column, output_path)
        return

    selected_by_key = {
        row.record_key: row._asdict()  # type: ignore[attr-defined]
        for row in selected.itertuples(index=False)
    }
    source_ids = sorted(selected["source_id"].unique())
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()

    def flush_rows(rows: list[dict[str, Any]]) -> None:
        nonlocal writer
        if not rows:
            return
        frame = pd.DataFrame(rows)
        frame["company"] = frame["company"].astype("string")
        frame["quarter"] = frame["quarter"].astype("string")
        frame["structured_content"] = frame["structured_content"].astype("string")
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int32")
        table = pa.Table.from_pandas(frame, schema=DATASET_SCHEMA, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, DATASET_SCHEMA, compression="snappy")
        writer.write_table(table)
        rows.clear()

    for record in iter_source_records(raw_root, source_ids=source_ids):
        selected_meta = selected_by_key.get(record["record_key"])
        if selected_meta is None:
            continue
        structured = record.get("structured_content")
        if structured not in (None, ""):
            structured_json = json.dumps(structured, ensure_ascii=False)
        else:
            structured_json = None
        buffer.append(
            {
                "record_key": record["record_key"],
                "source_id": record["source_id"],
                "source_file": record["source_file"],
                "source_row": record["source_row"],
                "ticker": selected_meta["ticker"],
                "canonical_ticker": selected_meta["canonical_ticker"],
                "event_date": selected_meta["event_date"],
                "company": record.get("company"),
                "year": record.get("year"),
                "quarter": record.get("quarter"),
                "transcript": record.get("transcript"),
                "structured_content": structured_json,
                "has_qa_section": selected_meta["has_qa_section"],
                "analyst_question_count": selected_meta["analyst_question_count"],
                "transcript_chars": selected_meta["transcript_chars"],
                "transcript_words": selected_meta["transcript_words"],
                "quality_score": selected_meta["quality_score"],
                "soft_quality_flags": selected_meta["soft_quality_flags"],
            }
        )
        if len(buffer) >= 256:
            flush_rows(buffer)

    flush_rows(buffer)
    if writer is not None:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/transcript_curation.yaml")
    parser.add_argument("--audit-dir", default="data/interim/audit")
    parser.add_argument("--curated-dir", default="data/interim/curated")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-root", default="data/raw/transcripts")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    metrics_path = Path(args.audit_dir) / "record_metrics.parquet"
    if not metrics_path.exists():
        raise RuntimeError("Audit metrics not found. Run audit_transcripts first.")

    metrics_df = pd.read_parquet(metrics_path)
    decisions_df = apply_deduplication(metrics_df)
    decisions_df = apply_quality_filters(decisions_df, cfg)

    ensure_dir(args.curated_dir)
    ensure_dir(args.processed_dir)
    decisions_out = Path(args.curated_dir) / "curation_decisions.csv"
    write_csv(decisions_df, decisions_out)

    summary_rows = [
        {"stage": "raw_records", "count": int(len(decisions_df))},
        {"stage": "clean_master", "count": int(decisions_df["keep_clean_master"].sum())},
        {"stage": "gold_corpus", "count": int(decisions_df["keep_gold"].sum())},
        {
            "stage": "mag7_gold",
            "count": int((decisions_df["keep_gold"] & (decisions_df["is_mag7"] == 1)).sum()),
        },
    ]
    write_csv(pd.DataFrame(summary_rows), Path(args.curated_dir) / "curation_summary.csv")

    _materialize_dataset(
        raw_root=args.raw_root,
        decisions_df=decisions_df,
        keep_column="keep_clean_master",
        output_path=Path(args.curated_dir) / "clean_master.parquet",
    )
    _materialize_dataset(
        raw_root=args.raw_root,
        decisions_df=decisions_df,
        keep_column="keep_gold",
        output_path=Path(args.processed_dir) / "gold_corpus.parquet",
    )
    mag7_decisions = decisions_df.copy()
    mag7_decisions["keep_mag7_gold"] = mag7_decisions["keep_gold"] & (mag7_decisions["is_mag7"] == 1)
    _materialize_dataset(
        raw_root=args.raw_root,
        decisions_df=mag7_decisions,
        keep_column="keep_mag7_gold",
        output_path=Path(args.processed_dir) / "mag7_gold.parquet",
    )
    LOGGER.info("Curated transcript corpus written to %s and %s.", args.curated_dir, args.processed_dir)


if __name__ == "__main__":
    main()
