from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow.parquet as pq

from src.data.transcript_sources import SOURCE_REGISTRY, TranscriptSource, resolve_sources


MAG7_CANONICAL_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"}
TICKER_ALIASES = {"GOOG": "GOOGL", "FB": "META"}

QA_SECTION_PATTERN = re.compile(
    r"(question[- ]and[- ]answer|questions? and answers?|q\s*&\s*a|question[- ]and[- ]answer session)",
    re.IGNORECASE,
)
ANALYST_PATTERN = re.compile(r"\banalyst\b", re.IGNORECASE)
MANAGEMENT_PATTERN = re.compile(
    r"\b(ceo|cfo|coo|chief|president|executive|management|operator)\b",
    re.IGNORECASE,
)
VALID_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
SOFT_OPERATOR_PATTERN = re.compile(r"\boperator\b", re.IGNORECASE)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def normalize_ticker(value: Any) -> str | None:
    if value is None:
        return None
    ticker = str(value).strip().upper()
    if not ticker:
        return None
    ticker = TICKER_ALIASES.get(ticker, ticker)
    return ticker


def parse_event_date(value: Any) -> str | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def parse_structured_content(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parse_structured_content(parsed)
    return []


def canonical_text_hash(text: str) -> str | None:
    normalized = normalize_text(text).lower()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_qa_excerpt(text: str) -> str:
    if not text:
        return ""
    match = QA_SECTION_PATTERN.search(text)
    if not match:
        return ""
    return text[match.start() :]


def estimate_analyst_question_count(text: str, structured_content: list[dict[str, Any]] | None = None) -> int:
    structured_content = structured_content or []
    if structured_content:
        count = 0
        in_qa = False
        for segment in structured_content:
            speaker = normalize_text(segment.get("speaker"))
            segment_text = normalize_text(segment.get("text"))
            combined = f"{speaker} {segment_text}".strip()
            if not in_qa and QA_SECTION_PATTERN.search(combined):
                in_qa = True
                continue
            if not in_qa:
                continue
            if ANALYST_PATTERN.search(speaker):
                count += 1
                continue
            if "?" in segment_text and not MANAGEMENT_PATTERN.search(speaker):
                count += 1
        if count:
            return count

    qa_excerpt = extract_qa_excerpt(text)
    if not qa_excerpt:
        return 0

    question_marks = qa_excerpt.count("?")
    if question_marks:
        return min(question_marks, 50)

    keyword_hits = len(re.findall(r"\b(question|follow[- ]up)\b", qa_excerpt, flags=re.IGNORECASE))
    return min(keyword_hits, 50)


def source_priority(source_id: str) -> int:
    return {
        "glopardo_sp500_earnings_transcripts": 50,
        "bose345_sp500_earnings_transcripts": 40,
        "jlh_ibm_earnings_call": 30,
        "kaggle_meta_earnings_call_qa": 20,
        "lamini_earnings_calls_qa": 10,
    }.get(source_id, 0)


def build_record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    transcript = normalize_text(record.get("transcript"))
    structured_content = parse_structured_content(record.get("structured_content"))
    ticker = normalize_ticker(record.get("ticker"))
    event_date = parse_event_date(record.get("event_date"))
    transcript_chars = len(transcript)
    transcript_words = len(transcript.split())
    analyst_question_count = estimate_analyst_question_count(transcript, structured_content)
    has_qa_section = bool(extract_qa_excerpt(transcript) or analyst_question_count > 0)
    has_structured_content = int(bool(structured_content))
    operator_mentions = len(SOFT_OPERATOR_PATTERN.findall(transcript))
    non_ascii_chars = sum(ord(ch) > 127 for ch in transcript)
    non_ascii_ratio = round(non_ascii_chars / max(transcript_chars, 1), 6)
    valid_ticker = int(bool(ticker and VALID_TICKER_PATTERN.match(ticker)))
    valid_event_date = int(event_date is not None)
    canonical_ticker = ticker if ticker in MAG7_CANONICAL_TICKERS else ticker
    return {
        "record_key": record["record_key"],
        "source_id": record["source_id"],
        "source_file": record["source_file"],
        "source_row": record["source_row"],
        "ticker": ticker,
        "canonical_ticker": canonical_ticker,
        "event_date": event_date,
        "company": normalize_text(record.get("company")),
        "year": record.get("year"),
        "quarter": record.get("quarter"),
        "transcript_chars": transcript_chars,
        "transcript_words": transcript_words,
        "text_hash": canonical_text_hash(transcript),
        "has_qa_section": int(has_qa_section),
        "analyst_question_count": analyst_question_count,
        "missing_ticker": int(ticker is None),
        "missing_event_date": int(event_date is None),
        "valid_ticker": valid_ticker,
        "valid_event_date": valid_event_date,
        "has_structured_content": has_structured_content,
        "operator_mentions": operator_mentions,
        "repeated_operator_text": int(operator_mentions >= 5),
        "non_ascii_ratio": non_ascii_ratio,
        "source_priority": source_priority(record["source_id"]),
    }


def inventory_raw_files(raw_root: str | Path) -> pd.DataFrame:
    raw_root = Path(raw_root)
    rows: list[dict[str, Any]] = []
    if not raw_root.exists():
        return pd.DataFrame(columns=["source_id", "file_path", "file_type", "size_bytes"])
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(raw_root).parts):
            continue
        rel_parts = path.relative_to(raw_root).parts
        if not rel_parts or rel_parts[0] not in SOURCE_REGISTRY:
            continue
        rows.append(
            {
                "source_id": rel_parts[0] if rel_parts else "unknown",
                "file_path": str(path),
                "file_type": path.suffix.lower() or "<no_ext>",
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def _record_key(source_id: str, source_file: str, source_row: int) -> str:
    return f"{source_id}:{source_file}:{source_row}"


def _iter_parquet_source(
    source: TranscriptSource,
    source_dir: Path,
    parquet_glob: str,
    field_map: dict[str, str],
) -> Iterator[dict[str, Any]]:
    for parquet_path in sorted(source_dir.glob(parquet_glob)):
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema.names)
        columns = [source_field for source_field in field_map.values() if source_field in available]
        row_number = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=256):
            for row in batch.to_pylist():
                source_file = str(parquet_path.relative_to(source_dir))
                yield {
                    "record_key": _record_key(source.source_id, source_file, row_number),
                    "source_id": source.source_id,
                    "source_file": source_file,
                    "source_row": row_number,
                    "ticker": row.get(field_map.get("ticker", "ticker")),
                    "event_date": row.get(field_map.get("event_date", "event_date")),
                    "company": row.get(field_map.get("company", "company")),
                    "year": row.get(field_map.get("year", "year")),
                    "quarter": row.get(field_map.get("quarter", "quarter")),
                    "transcript": row.get(field_map.get("transcript", "transcript")),
                    "structured_content": row.get(field_map.get("structured_content", "structured_content")),
                }
                row_number += 1


def _iter_jlh_records(source: TranscriptSource, source_dir: Path) -> Iterator[dict[str, Any]]:
    transcripts_root = source_dir / "data" / "transcripts"
    for file_path in sorted(transcripts_root.rglob("*.txt")):
        source_file = str(file_path.relative_to(source_dir))
        ticker = file_path.parent.name
        event_date = None
        stem_parts = file_path.stem.split("-")
        if len(stem_parts) >= 4:
            event_date = f"{stem_parts[0]}-{stem_parts[1]}-{stem_parts[2]}"
        yield {
            "record_key": _record_key(source.source_id, source_file, 0),
            "source_id": source.source_id,
            "source_file": source_file,
            "source_row": 0,
            "ticker": ticker,
            "event_date": event_date,
            "company": None,
            "year": None,
            "quarter": None,
            "transcript": file_path.read_text(encoding="utf-8", errors="ignore"),
            "structured_content": None,
        }


def _first_present(obj: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return None


def _iter_lamini_records(source: TranscriptSource, source_dir: Path) -> Iterator[dict[str, Any]]:
    data_path = source_dir / "filtered_predictions.jsonl"
    if not data_path.exists():
        return
    with data_path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            transcript = _first_present(
                obj,
                [
                    "transcript",
                    "earning_call",
                    "earnings_call",
                    "call_transcript",
                    "content",
                    "text",
                ],
            )
            if transcript is None:
                qa_chunks: list[str] = []
                questions = obj.get("questions") or obj.get("question") or []
                answers = obj.get("answers") or obj.get("answer") or []
                if isinstance(questions, list):
                    qa_chunks.extend(str(item) for item in questions if item)
                elif questions:
                    qa_chunks.append(str(questions))
                if isinstance(answers, list):
                    qa_chunks.extend(str(item) for item in answers if item)
                elif answers:
                    qa_chunks.append(str(answers))
                transcript = " ".join(qa_chunks)
            yield {
                "record_key": _record_key(source.source_id, "filtered_predictions.jsonl", row_number),
                "source_id": source.source_id,
                "source_file": "filtered_predictions.jsonl",
                "source_row": row_number,
                "ticker": _first_present(obj, ["ticker", "symbol", "company_symbol", "stock_symbol"]),
                "event_date": _first_present(obj, ["event_date", "earnings_date", "call_date", "date"]),
                "company": _first_present(obj, ["company", "company_name", "issuer"]),
                "year": _first_present(obj, ["year"]),
                "quarter": _first_present(obj, ["quarter"]),
                "transcript": transcript,
                "structured_content": _first_present(obj, ["structured_content", "segments", "dialogue"]),
            }


def iter_source_records(raw_root: str | Path, source_ids: list[str] | None = None) -> Iterator[dict[str, Any]]:
    raw_root = Path(raw_root)
    for source in resolve_sources(source_ids):
        source_dir = raw_root / source.source_id
        if not source_dir.exists():
            continue
        if source.source_id == "glopardo_sp500_earnings_transcripts":
            yield from _iter_parquet_source(
                source,
                source_dir,
                "data/*.parquet",
                {
                    "ticker": "ticker",
                    "event_date": "earnings_date",
                    "company": "company",
                    "year": "year",
                    "quarter": "quarter",
                    "transcript": "transcript",
                    "structured_content": "structured_content",
                },
            )
        elif source.source_id == "bose345_sp500_earnings_transcripts":
            yield from _iter_parquet_source(
                source,
                source_dir,
                "parquet_files/*.parquet",
                {
                    "ticker": "symbol",
                    "event_date": "date",
                    "company": "company_name",
                    "year": "year",
                    "quarter": "quarter",
                    "transcript": "content",
                    "structured_content": "structured_content",
                },
            )
        elif source.source_id == "jlh_ibm_earnings_call":
            yield from _iter_jlh_records(source, source_dir)
        elif source.source_id == "lamini_earnings_calls_qa":
            yield from _iter_lamini_records(source, source_dir)
        elif source.source_id == "kaggle_meta_earnings_call_qa":
            continue
        else:
            raise ValueError(f"Unsupported source_id={source.source_id!r}")


def build_audit_summary(records_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    if records_df.empty:
        return pd.DataFrame(summary_rows)

    duplicate_text_mask = records_df["text_hash"].notna() & (
        records_df.groupby("text_hash")["record_key"].transform("size") > 1
    )
    duplicate_pair_mask = records_df["ticker"].notna() & records_df["event_date"].notna() & (
        records_df.groupby(["ticker", "event_date"])["record_key"].transform("size") > 1
    )

    merged = records_df.copy()
    merged["duplicate_text"] = duplicate_text_mask.astype(int)
    merged["duplicate_ticker_date"] = duplicate_pair_mask.astype(int)

    for source_id, group in list(merged.groupby("source_id")) + [("ALL", merged)]:
        files = inventory_df[inventory_df["source_id"] == source_id] if source_id != "ALL" else inventory_df
        file_types = ",".join(sorted(files["file_type"].dropna().unique()))
        summary_rows.append(
            {
                "source_id": source_id,
                "num_files": int(len(files)),
                "num_records": int(len(group)),
                "file_types": file_types,
                "duplicate_text_records": int(group["duplicate_text"].sum()),
                "duplicate_ticker_date_records": int(group["duplicate_ticker_date"].sum()),
                "missing_ticker_records": int(group["missing_ticker"].sum()),
                "missing_event_date_records": int(group["missing_event_date"].sum()),
                "qa_detectable_records": int(group["has_qa_section"].sum()),
                "qa_detectable_share": round(float(group["has_qa_section"].mean()), 4),
                "transcript_chars_p10": int(group["transcript_chars"].quantile(0.10)),
                "transcript_chars_p50": int(group["transcript_chars"].quantile(0.50)),
                "transcript_chars_p90": int(group["transcript_chars"].quantile(0.90)),
                "transcript_words_p10": int(group["transcript_words"].quantile(0.10)),
                "transcript_words_p50": int(group["transcript_words"].quantile(0.50)),
                "transcript_words_p90": int(group["transcript_words"].quantile(0.90)),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("source_id").reset_index(drop=True)


def is_mag7_ticker(ticker: str | None) -> bool:
    return bool(ticker and normalize_ticker(ticker) in MAG7_CANONICAL_TICKERS)
