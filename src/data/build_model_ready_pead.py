from __future__ import annotations

import argparse
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests

from src.data.build_earnings_fundamentals import build_earnings_fundamentals, load_external_earnings_events
from src.data.build_dataset import build_dataset
from src.data.compute_pead import _compute_event_labels
from src.data.extract_qa import extract_qa
from src.data.split_dataset import time_based_split
from src.utils.io import ensure_dir, load_yaml, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

OPTIONAL_METADATA_COLUMNS = [
    "snapshot_market_cap_usd",
    "snapshot_log_market_cap",
    "snapshot_market_cap_percentile",
    "universe_sector",
    "universe_industry",
    "universe_company_name",
    "universe_cik",
    "universe_calls_in_gold_corpus",
    "universe_first_event_date",
    "universe_last_event_date",
    "universe_included_by",
    "universe_classification_source",
    "universe_source",
    "universe_snapshot_date",
    "hist_market_cap",
    "hist_log_market_cap",
    "hist_market_cap_percentile",
    "hist_market_cap_price_date",
    "hist_market_cap_close",
    "hist_market_cap_split_adjusted_close",
    "hist_market_cap_split_factor_to_raw",
    "hist_market_cap_shares_outstanding",
    "hist_market_cap_shares_end_date",
    "hist_market_cap_shares_filed_date",
    "hist_market_cap_price_lag_days",
    "hist_market_cap_shares_staleness_days",
    "hist_market_cap_source",
]


COLON_SPEAKER_RE = re.compile(
    r"(?<!\w)(?P<speaker>(?:Operator|Unknown Speaker|[A-Z][A-Za-z0-9&.'\-/]+(?: [A-Z][A-Za-z0-9&.'\-/]+){0,5}))\s*:\s+"
)
QNA_HEADING_RE = re.compile(r"Questions? and Answers?", re.IGNORECASE)
STRONG_QA_CUE_RE = re.compile(
    r"(question[- ]and[- ]answer|questions? and answers?|q\s*&\s*a|we will now (?:begin|open|start|take)|take your questions|question and answer session|questions from analysts|operator instructions)",
    re.IGNORECASE,
)
ACTUAL_QA_START_RE = re.compile(
    r"(move on to the q&a|move over to q&a|we(?:'ll| will) now (?:begin|start|take|open)|repeat your instructions|first question|question comes from|please proceed with your question|\[operator instructions\])",
    re.IGNORECASE,
)
PARTICIPANT_NAME_RE = re.compile(r"([A-Z][A-Za-z.'\-]+(?: [A-Z][A-Za-z.'\-]+){1,3})\s*-\s*")
ROLE_HINT_RE = re.compile(r"\b(analyst|research|capital|securities|partners|bank|morgan|goldman|barclays|ubs|jp ?morgan)\b", re.IGNORECASE)


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\ufeff", " ").split())


def _speaker_key(speaker: str) -> str:
    return _clean_text(speaker).lower()


def _extract_named_block(text: str, label: str, end_labels: list[str]) -> str:
    end_pattern = "|".join(re.escape(item) for item in end_labels)
    pattern = re.compile(rf"{label}\s*:\s*(.*?)(?:{end_pattern})\s*:", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        return ""
    return _clean_text(match.group(1))


def _extract_participant_names(text: str) -> tuple[set[str], set[str]]:
    analysts_block = _extract_named_block(
        text,
        "Analysts?",
        ["Operator", "Executives", "Presentation", "Corporate Participants", "Conference Call Participants"],
    )
    executives_block = _extract_named_block(
        text,
        "Executives?",
        ["Analysts", "Operator", "Presentation", "Corporate Participants", "Conference Call Participants"],
    )

    analyst_names = {_speaker_key(name) for name in PARTICIPANT_NAME_RE.findall(analysts_block)}
    executive_names = {_speaker_key(name) for name in PARTICIPANT_NAME_RE.findall(executives_block)}
    return analyst_names, executive_names


def _find_qna_start(turns: list[dict[str, Any]]) -> int | None:
    for idx, turn in enumerate(turns):
        combined = f"{turn['speaker']} {turn['text']}"
        if not ACTUAL_QA_START_RE.search(combined):
            continue
        lower = combined.lower()
        if turn["speaker_role"] == "operator" and ("question comes from" in lower or "[operator instructions]" in lower):
            return idx
        return min(idx + 1, len(turns) - 1)

    cue_candidates: list[int] = []
    for idx, turn in enumerate(turns):
        combined = f"{turn['speaker']} {turn['text']}"
        if STRONG_QA_CUE_RE.search(combined):
            cue_candidates.append(idx)
    if cue_candidates:
        return min(cue_candidates[-1] + 1, len(turns) - 1)

    analyst_indices = [idx for idx, turn in enumerate(turns[3:], start=3) if turn["speaker_role"] == "analyst"]
    if analyst_indices:
        return analyst_indices[0]
    return None


def _parse_colon_turns(transcript: str) -> list[tuple[str, str]]:
    matches = list(COLON_SPEAKER_RE.finditer(transcript))
    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(transcript)
        body = _clean_text(transcript[start:end])
        if not body:
            continue
        speaker = _clean_text(match.group("speaker"))
        turns.append((speaker, body))
    return turns


def parse_colon_qna_transcript(transcript: str) -> list[dict[str, Any]]:
    transcript = _clean_text(transcript)
    analyst_names, executive_names = _extract_participant_names(transcript)

    raw_turns = _parse_colon_turns(transcript)
    if not raw_turns:
        return []

    turns: list[dict[str, Any]] = []
    for speaker, text in raw_turns:
        speaker_norm = _speaker_key(speaker)
        if speaker_norm == "operator":
            role = "operator"
        elif speaker_norm in analyst_names:
            role = "analyst"
        elif speaker_norm in executive_names:
            role = "management"
        elif ROLE_HINT_RE.search(text) and "?" in text:
            role = "analyst"
        else:
            role = "management"
        turns.append({"speaker": speaker, "speaker_role": role, "text": text})

    qna_start = _find_qna_start(turns)
    if qna_start is None:
        return []

    management_names = {
        _speaker_key(turn["speaker"])
        for turn in turns[:qna_start]
        if turn["speaker_role"] != "operator"
    }
    management_names.update(executive_names)

    qna_turns: list[dict[str, Any]] = []
    for turn in turns[qna_start:]:
        speaker_norm = _speaker_key(turn["speaker"])
        if speaker_norm == "operator":
            role = "operator"
        elif speaker_norm in analyst_names:
            role = "analyst"
        elif speaker_norm in management_names:
            role = "management"
        elif "?" in turn["text"]:
            role = "analyst"
        else:
            role = "analyst"
        qna_turns.append(
            {
                "speaker": turn["speaker"],
                "speaker_role": role,
                "section": "q&a",
                "text": turn["text"],
            }
        )
    return qna_turns


def parse_reuters_qna_transcript(transcript: str) -> list[dict[str, Any]]:
    heading = QNA_HEADING_RE.search(transcript)
    if not heading:
        return []
    qa_text = transcript[heading.end() :]

    parts = [part.strip() for part in re.split(r"-{20,}", qa_text) if part.strip()]
    turns: list[dict[str, Any]] = []
    for idx in range(0, len(parts) - 1, 2):
        speaker_line = _clean_text(parts[idx].splitlines()[0])
        body = _clean_text(parts[idx + 1])
        if not body:
            continue
        speaker_name = _clean_text(speaker_line.split(",")[0])
        lower_line = speaker_line.lower()
        if "operator" in lower_line:
            role = "operator"
        elif "analyst" in lower_line:
            role = "analyst"
        else:
            role = "management"
        turns.append(
            {
                "speaker": speaker_name,
                "speaker_role": role,
                "section": "q&a",
                "text": body,
            }
        )
    return turns


def parse_bose_structured_qna(structured_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not structured_content:
        return []
    normalized = [
        {"speaker": _clean_text(item.get("speaker")), "text": _clean_text(item.get("text"))}
        for item in structured_content
        if _clean_text(item.get("text"))
    ]
    if not normalized:
        return []

    turns = []
    for item in normalized:
        turns.append({"speaker": item["speaker"], "speaker_role": "management", "text": item["text"]})
    qna_start = _find_qna_start(turns)
    if qna_start is None:
        return []

    management_names = {
        _speaker_key(item["speaker"])
        for item in normalized[:qna_start]
        if _speaker_key(item["speaker"]) != "operator"
    }

    qna_turns: list[dict[str, Any]] = []
    for item in normalized[qna_start:]:
        speaker_norm = _speaker_key(item["speaker"])
        if speaker_norm == "operator":
            role = "operator"
        elif speaker_norm in management_names:
            role = "management"
        else:
            role = "analyst"
        qna_turns.append(
            {
                "speaker": item["speaker"],
                "speaker_role": role,
                "section": "q&a",
                "text": item["text"],
            }
        )
    return qna_turns


def load_bose_structured_content(
    raw_root: str | Path,
    rows_needed: set[int],
    source_file: str = "parquet_files/part-0.parquet",
) -> dict[int, list[dict[str, Any]]]:
    if not rows_needed:
        return {}
    parquet_path = Path(raw_root) / "bose345_sp500_earnings_transcripts" / source_file
    parquet_file = pq.ParquetFile(parquet_path)
    structured: dict[int, list[dict[str, Any]]] = {}
    current_row = 0
    for batch in parquet_file.iter_batches(columns=["structured_content"], batch_size=256):
        for row in batch.to_pylist():
            if current_row in rows_needed:
                structured[current_row] = row.get("structured_content") or []
            current_row += 1
            if len(structured) == len(rows_needed):
                return structured
    return structured


def build_parsed_calls(corpus_df: pd.DataFrame, raw_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bose_rows = set(
        corpus_df.loc[corpus_df["source_id"] == "bose345_sp500_earnings_transcripts", "source_row"]
        .astype(int)
        .tolist()
    )
    bose_structured = load_bose_structured_content(raw_root, bose_rows)

    parsed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in corpus_df.itertuples(index=False):
        if row.source_id == "bose345_sp500_earnings_transcripts":
            turns = parse_bose_structured_qna(bose_structured.get(int(row.source_row), []))
            parser_type = "bose_structured"
            if sum(turn["speaker_role"] == "analyst" for turn in turns) == 0:
                turns = parse_colon_qna_transcript(row.transcript)
                parser_type = "bose_colon_fallback"
        elif row.source_id == "jlh_ibm_earnings_call":
            turns = parse_reuters_qna_transcript(row.transcript)
            parser_type = "reuters"
        else:
            turns = parse_colon_qna_transcript(row.transcript)
            parser_type = "colon"

        analyst_turns = sum(turn["speaker_role"] == "analyst" for turn in turns)
        audit_rows.append(
            {
                "call_id": row.record_key,
                "ticker": row.ticker,
                "event_date": row.event_date,
                "source_id": row.source_id,
                "parser_type": parser_type,
                "parsed_turns": len(turns),
                "parsed_analyst_turns": analyst_turns,
                "parsed_ok": int(len(turns) > 0 and analyst_turns > 0),
            }
        )
        for turn_id, turn in enumerate(turns, start=1):
            parsed_rows.append(
                {
                    "call_id": row.record_key,
                    "ticker": row.ticker,
                    "event_date": row.event_date,
                    "turn_id": turn_id,
                    "speaker_role": turn["speaker_role"],
                    "section": turn["section"],
                    "text": turn["text"],
                }
            )

    return pd.DataFrame(parsed_rows), pd.DataFrame(audit_rows)


def build_metadata(corpus_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "record_key",
        "ticker",
        "event_date",
        "source_id",
        "source_file",
        "source_row",
        "company",
        "year",
        "quarter",
        "quality_score",
        "soft_quality_flags",
    ]
    extra_columns = [column for column in OPTIONAL_METADATA_COLUMNS if column in corpus_df.columns]
    meta = corpus_df[[*base_columns, *extra_columns]].copy()
    meta = meta.rename(columns={"record_key": "call_id"})
    return meta


def load_market_cap_snapshots(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        LOGGER.warning("Market-cap snapshots file not found: %s", csv_path)
        return None
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None
    required = {"ticker", "event_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Market-cap snapshots file is missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    if "call_id" in out.columns:
        key_cols = ["call_id", "ticker", "event_date"]
    else:
        key_cols = ["ticker", "event_date"]
    value_cols = [
        column
        for column in OPTIONAL_METADATA_COLUMNS
        if column.startswith("hist_market_cap") and column in out.columns
    ]
    return out[[*key_cols, *value_cols]].drop_duplicates(subset=key_cols, keep="last")


def _output_path(cfg: dict[str, Any], key: str, default: str) -> str:
    return str(cfg.get("outputs", {}).get(key, default))


def _ensure_parent(path: str | Path) -> None:
    ensure_dir(Path(path).parent)


def _download_stooq_csv(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    if response.text.strip().lower().startswith("no data"):
        raise ValueError(f"No Stooq data returned for {symbol}")
    if "get your apikey" in response.text.lower():
        raise ValueError("Stooq CSV download requires an API key")
    df = pd.read_csv(io.StringIO(response.text))
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"Unexpected Stooq payload for {symbol}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _default_stooq_symbol(ticker: str) -> str:
    return f"{str(ticker).lower().replace('.', '-')}.us"


def _default_yfinance_symbol(ticker: str) -> str:
    return str(ticker).upper().replace(".", "-")


def _coerce_symbol_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _download_yfinance_prices(symbol: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            frame = yf.download(
                symbol,
                start=min_date.strftime("%Y-%m-%d"),
                end=(max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if frame.empty:
                raise ValueError(f"No Yahoo Finance data returned for {symbol}")
            if isinstance(frame.columns, pd.MultiIndex):
                close = frame["Close"].iloc[:, 0]
            else:
                close = frame["Close"]
            out = close.rename("Close").reset_index()
            out = out.rename(columns={out.columns[0]: "Date"})
            out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
            return out[["Date", "Close"]].sort_values("Date").reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 * attempt)
    raise ValueError(f"Yahoo Finance download failed for {symbol}: {last_error}")


def _timestamp_seconds(value: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.timestamp())


def _download_yahoo_chart_prices(symbol: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    period1 = _timestamp_seconds(min_date)
    period2 = _timestamp_seconds(max_date + pd.Timedelta(days=1))
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            result = (payload.get("chart", {}).get("result") or [None])[0]
            if not result:
                error = payload.get("chart", {}).get("error")
                raise ValueError(f"No Yahoo chart result for {symbol}: {error}")
            timestamps = result.get("timestamp") or []
            indicators = result.get("indicators", {})
            adjclose = (indicators.get("adjclose") or [{}])[0].get("adjclose")
            close = adjclose or (indicators.get("quote") or [{}])[0].get("close")
            if not timestamps or not close:
                raise ValueError(f"Yahoo chart result missing prices for {symbol}")
            out = pd.DataFrame(
                {
                    "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                    "Close": pd.to_numeric(close, errors="coerce"),
                }
            ).dropna()
            if out.empty:
                raise ValueError(f"No usable Yahoo chart prices for {symbol}")
            return out.sort_values("Date").reset_index(drop=True)
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 * attempt)
    raise ValueError(f"Yahoo chart download failed for {symbol}: {last_error}")


def _download_price_history(
    symbol: str,
    provider: str,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    if provider == "stooq":
        return _download_stooq_csv(symbol)
    if provider == "yahoo_chart":
        return _download_yahoo_chart_prices(symbol, min_date=min_date, max_date=max_date)
    if provider == "yfinance":
        return _download_yfinance_prices(symbol, min_date=min_date, max_date=max_date)
    raise ValueError(f"Unsupported price_provider={provider!r}")


def build_price_frames(
    metadata_df: pd.DataFrame,
    symbol_map: dict[str, list[str]] | None = None,
    market_symbol: str = "^spx",
    price_provider: str = "yahoo_chart",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_symbol_map = {
        str(ticker).upper(): _coerce_symbol_list(symbols) for ticker, symbols in (symbol_map or {}).items()
    }

    event_dates = pd.to_datetime(metadata_df["event_date"])
    min_date = event_dates.min() - pd.Timedelta(days=40)
    max_date = event_dates.max() + pd.Timedelta(days=60)

    price_rows: list[pd.DataFrame] = []
    for ticker in sorted(metadata_df["ticker"].unique()):
        ticker = str(ticker).upper()
        frames = []
        default_symbol = (
            _default_yfinance_symbol(ticker)
            if price_provider in {"yfinance", "yahoo_chart"}
            else _default_stooq_symbol(ticker)
        )
        for symbol in raw_symbol_map.get(ticker, [default_symbol]):
            try:
                df = _download_price_history(symbol, provider=price_provider, min_date=min_date, max_date=max_date)
            except Exception as exc:
                LOGGER.warning("Skipping %s symbol %s for %s: %s", price_provider, symbol, ticker, exc)
                continue
            df = df.rename(columns={"Date": "date", "Close": "close"})
            frames.append(df[["date", "close"]])
        if not frames:
            LOGGER.warning("No price history found for %s; downstream labels will exclude this ticker.", ticker)
            continue
        price_df = pd.concat(frames, ignore_index=True)
        price_df = (
            price_df.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .query("@min_date <= date <= @max_date")
            .reset_index(drop=True)
        )
        price_df["return"] = price_df["close"].pct_change()
        price_df["ticker"] = ticker
        price_rows.append(price_df[["date", "ticker", "return"]])

    if not price_rows:
        raise RuntimeError(f"No ticker price histories were downloaded from {price_provider}.")
    prices = pd.concat(price_rows, ignore_index=True).dropna().reset_index(drop=True)

    market_df = _download_price_history(
        market_symbol,
        provider=price_provider,
        min_date=min_date,
        max_date=max_date,
    )
    market_df = market_df.rename(columns={"Date": "date", "Close": "close"})
    market_df = market_df.query("@min_date <= date <= @max_date").reset_index(drop=True)
    market_df["market_return"] = market_df["close"].pct_change()
    market = market_df[["date", "market_return"]].dropna().reset_index(drop=True)
    return prices, market


def enrich_qa_dataset(
    analyst_qa_df: pd.DataFrame,
    parsed_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    full_qa_df = extract_qa(parsed_df, analyst_only=False).rename(
        columns={"text": "full_qa_text", "num_questions": "num_qa_turns"}
    )
    merged = analyst_qa_df.merge(
        full_qa_df[["call_id", "ticker", "event_date", "full_qa_text", "num_qa_turns"]],
        on=["call_id", "ticker", "event_date"],
        how="left",
    )
    merged = merged.merge(
        metadata_df[
            [
                "call_id",
                "source_id",
                "company",
                "year",
                "quarter",
                "quality_score",
                "soft_quality_flags",
            ]
        ],
        on="call_id",
        how="left",
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/model_ready_mag7.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    corpus_path = cfg.get("corpus_path", "data/processed/mag7_gold.parquet")
    raw_root = cfg.get("raw_root", "data/raw/transcripts")

    corpus_df = pd.read_parquet(corpus_path)
    metadata_df = build_metadata(corpus_df)
    market_cap_df = load_market_cap_snapshots(cfg.get("market_cap_snapshots_path"))
    if market_cap_df is not None:
        merge_keys = ["call_id", "ticker", "event_date"] if "call_id" in market_cap_df.columns else ["ticker", "event_date"]
        metadata_df = metadata_df.merge(market_cap_df, on=merge_keys, how="left")
    parsed_df, parser_audit_df = build_parsed_calls(corpus_df, raw_root)

    parsed_df = parsed_df.sort_values(["call_id", "turn_id"]).reset_index(drop=True)
    parser_audit_df = parser_audit_df.sort_values(["event_date", "ticker"]).reset_index(drop=True)

    output_paths = {
        "parsed_calls": _output_path(cfg, "parsed_calls", "data/interim/parsed_calls/parsed_calls.csv"),
        "parser_audit": _output_path(cfg, "parser_audit", "data/interim/parsed_calls/parser_audit.csv"),
        "metadata": _output_path(cfg, "metadata", "data/raw/metadata/call_metadata.csv"),
        "earnings_fundamentals": _output_path(
            cfg,
            "earnings_fundamentals",
            "data/external/earnings_fundamentals/earnings_fundamentals.csv",
        ),
        "earnings_summary": _output_path(cfg, "earnings_summary", "data/external/earnings_fundamentals/summary.csv"),
        "prices": _output_path(cfg, "prices", "data/raw/prices/daily_returns.csv"),
        "market": _output_path(cfg, "market", "data/external/market_index/sp500_returns.csv"),
        "labels": _output_path(cfg, "labels", "data/interim/labels/pead_labels.csv"),
        "qa_dataset": _output_path(cfg, "qa_dataset", "data/interim/qa_only/qa_dataset.csv"),
        "dataset": _output_path(cfg, "dataset", "data/processed/dataset.csv"),
        "train": _output_path(cfg, "train", "data/processed/train.csv"),
        "val": _output_path(cfg, "val", "data/processed/val.csv"),
        "test": _output_path(cfg, "test", "data/processed/test.csv"),
    }
    for path in output_paths.values():
        _ensure_parent(path)

    write_csv(parsed_df, output_paths["parsed_calls"])
    write_csv(parser_audit_df, output_paths["parser_audit"])
    metadata_df.to_csv(output_paths["metadata"], index=False)

    earnings_events_paths = cfg.get("earnings_events_paths", cfg.get("earnings_events_path"))
    external_events_df = load_external_earnings_events(earnings_events_paths)
    earnings_fundamentals_df, earnings_summary = build_earnings_fundamentals(
        metadata_df=metadata_df,
        raw_root=raw_root,
        external_events_df=external_events_df,
    )
    write_csv(earnings_fundamentals_df, output_paths["earnings_fundamentals"])
    write_csv(pd.DataFrame([earnings_summary]), output_paths["earnings_summary"])

    price_provider = str(cfg.get("price_provider", "yahoo_chart"))
    default_market_symbol = "^GSPC" if price_provider in {"yfinance", "yahoo_chart"} else "^spx"
    prices_df, market_df = build_price_frames(
        metadata_df,
        symbol_map=cfg.get("price_symbol_map"),
        market_symbol=str(cfg.get("market_symbol", default_market_symbol)),
        price_provider=price_provider,
    )
    prices_df.to_csv(output_paths["prices"], index=False)
    market_df.to_csv(output_paths["market"], index=False)

    label_cfg = load_yaml(cfg.get("label_config", "configs/data/pead_20d.yaml"))
    priced_tickers = set(prices_df["ticker"].unique())
    label_metadata_df = metadata_df[metadata_df["ticker"].isin(priced_tickers)].copy()
    missing_price_tickers = sorted(set(metadata_df["ticker"].unique()) - priced_tickers)
    if missing_price_tickers:
        LOGGER.warning(
            "Excluding %d tickers from label construction due to missing price data: %s",
            len(missing_price_tickers),
            ", ".join(missing_price_tickers[:25]),
        )
    labels_df = _compute_event_labels(
        metadata=label_metadata_df[["call_id", "ticker", "event_date"]],
        prices=prices_df,
        market=market_df,
        horizon=int(label_cfg.get("pead_horizon", 20)),
        event_lag_days=int(label_cfg.get("event_lag_days", 1)),
        label_threshold=float(label_cfg.get("label_threshold", 0.0)),
    )
    labels_df.to_csv(output_paths["labels"], index=False)

    analyst_qa_df = extract_qa(parsed_df, analyst_only=bool(cfg.get("analyst_only", True)))
    analyst_qa_df = enrich_qa_dataset(analyst_qa_df, parsed_df, metadata_df)
    analyst_qa_df.to_csv(output_paths["qa_dataset"], index=False)

    dataset_df = build_dataset(analyst_qa_df, labels_df)
    dataset_df = dataset_df.sort_values("event_date").reset_index(drop=True)
    dataset_df.to_csv(output_paths["dataset"], index=False)

    train_df, val_df, test_df = time_based_split(dataset_df)
    train_df.to_csv(output_paths["train"], index=False)
    val_df.to_csv(output_paths["val"], index=False)
    test_df.to_csv(output_paths["test"], index=False)

    LOGGER.info(
        "Built model-ready PEAD dataset: parsed_calls=%d qa_calls=%d dataset=%d train=%d val=%d test=%d",
        len(parsed_df),
        analyst_qa_df["call_id"].nunique(),
        len(dataset_df),
        len(train_df),
        len(val_df),
        len(test_df),
    )


if __name__ == "__main__":
    main()
