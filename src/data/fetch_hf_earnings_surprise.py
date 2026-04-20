from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.utils.io import ensure_dir, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", ".")


def _load_events(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    frame = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    out = frame.copy()
    if "call_id" not in out.columns and "record_key" in out.columns:
        out = out.rename(columns={"record_key": "call_id"})
    required = {"call_id", "ticker", "event_date"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Events input is missing required columns: {sorted(missing)}")
    out["ticker"] = out["ticker"].map(_normalize_ticker)
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce").dt.normalize()
    return out[["call_id", "ticker", "event_date"]].dropna(subset=["ticker", "event_date"]).drop_duplicates()


def load_hf_surprise_events(
    target_events: pd.DataFrame,
    dataset_name: str = "sovai/earnings_surprise",
    split: str = "train",
    streaming: bool = False,
) -> pd.DataFrame:
    tickers = set(target_events["ticker"].map(_normalize_ticker))
    min_date = target_events["event_date"].min() - pd.Timedelta(days=7)
    max_date = target_events["event_date"].max() + pd.Timedelta(days=7)

    rows: list[dict[str, Any]] = []
    if streaming:
        LOGGER.info("Streaming %s/%s for %d target tickers", dataset_name, split, len(tickers))
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        iterator = dataset
    else:
        LOGGER.info("Loading %s/%s for %d target tickers", dataset_name, split, len(tickers))
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.filter(
            lambda batch: [_normalize_ticker(ticker) in tickers for ticker in batch],
            input_columns=["ticker"],
            batched=True,
        )
        iterator = dataset.to_iterable_dataset()

    for item in iterator:
        ticker = _normalize_ticker(item.get("ticker"))
        source_event_date = pd.to_datetime(item.get("date_pub"), errors="coerce")
        if pd.isna(source_event_date):
            continue
        source_event_date = source_event_date.normalize()
        if source_event_date < min_date or source_event_date > max_date:
            continue
        rows.append(
            {
                "ticker": ticker,
                "source_event_date": source_event_date,
                "source_snapshot_date": pd.to_datetime(item.get("date"), errors="coerce"),
                "reported_eps": pd.to_numeric(item.get("actual_earning_result"), errors="coerce"),
                "estimated_eps": pd.to_numeric(item.get("estimated_earning"), errors="coerce"),
                "eps_surprise": pd.to_numeric(item.get("eps_surprise"), errors="coerce"),
                "estimated_eps_source": "hf_sovai_earnings_surprise",
            }
        )

    source_df = pd.DataFrame(rows)
    if source_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "source_event_date",
                "reported_eps",
                "estimated_eps",
                "eps_surprise",
                "estimated_eps_source",
            ]
        )
    source_df["source_snapshot_date"] = pd.to_datetime(source_df["source_snapshot_date"], errors="coerce")
    source_df = source_df.sort_values(["ticker", "source_event_date", "source_snapshot_date"])
    return source_df.drop_duplicates(subset=["ticker", "source_event_date"], keep="last").reset_index(drop=True)


def match_events_for_ticker(target_df: pd.DataFrame, source_df: pd.DataFrame, max_day_diff: int) -> pd.DataFrame:
    target = target_df.copy()
    target["event_date"] = pd.to_datetime(target["event_date"]).dt.normalize()
    if source_df.empty:
        out = target.copy()
        out["source_event_date"] = pd.NaT
        out["date_diff_days"] = pd.NA
        out["match_status"] = "missing_source"
        for column in ["reported_eps", "estimated_eps", "eps_surprise", "eps_surprise_pct", "estimated_eps_source"]:
            out[column] = pd.NA
        return out

    source = source_df.copy()
    source["source_event_date"] = pd.to_datetime(source["source_event_date"]).dt.normalize()
    source = source.sort_values("source_event_date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in target.itertuples(index=False):
        diffs = (source["source_event_date"] - row.event_date).abs().dt.days
        best_idx = int(diffs.idxmin())
        best_diff = int(diffs.iloc[best_idx])
        best = source.iloc[best_idx]
        is_match = best_diff <= max_day_diff
        estimated_eps = best["estimated_eps"] if is_match else pd.NA
        eps_surprise = best["eps_surprise"] if is_match else pd.NA
        rows.append(
            {
                "call_id": row.call_id,
                "ticker": row.ticker,
                "event_date": row.event_date.strftime("%Y-%m-%d"),
                "source_event_date": best["source_event_date"].strftime("%Y-%m-%d"),
                "date_diff_days": best_diff,
                "match_status": "matched" if is_match else "outside_tolerance",
                "reported_eps": best["reported_eps"] if is_match else pd.NA,
                "estimated_eps": estimated_eps,
                "eps_surprise": eps_surprise,
                "eps_surprise_pct": eps_surprise / abs(estimated_eps)
                if pd.notna(eps_surprise) and pd.notna(estimated_eps) and abs(float(estimated_eps)) > 1e-12
                else pd.NA,
                "estimated_eps_source": best["estimated_eps_source"] if is_match else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_hf_earnings_surprise_events(
    target_events: pd.DataFrame,
    dataset_name: str = "sovai/earnings_surprise",
    split: str = "train",
    max_day_diff: int = 3,
    streaming: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_df = load_hf_surprise_events(target_events, dataset_name=dataset_name, split=split, streaming=streaming)
    matched_frames = []
    ticker_summaries = []
    for ticker, target_df in target_events.groupby("ticker", sort=True):
        ticker_source = source_df[source_df["ticker"] == ticker].copy()
        matched_df = match_events_for_ticker(target_df, ticker_source, max_day_diff=max_day_diff)
        matched_frames.append(matched_df)
        ticker_summaries.append(
            {
                "ticker": ticker,
                "target_events": int(len(target_df)),
                "source_events": int(len(ticker_source)),
                "matched_events": int((matched_df["match_status"] == "matched").sum()),
                "coverage": float((matched_df["match_status"] == "matched").mean()) if len(matched_df) else 0.0,
            }
        )
    out = pd.concat(matched_frames, ignore_index=True).sort_values(["ticker", "event_date", "call_id"]).reset_index(drop=True)
    summary = {
        "rows": int(len(out)),
        "source_rows": int(len(source_df)),
        "matched_rows": int((out["match_status"] == "matched").sum()),
        "matched_coverage": float((out["match_status"] == "matched").mean()) if len(out) else 0.0,
        "reported_eps_coverage": float(pd.to_numeric(out["reported_eps"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "estimated_eps_coverage": float(pd.to_numeric(out["estimated_eps"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "eps_surprise_coverage": float(pd.to_numeric(out["eps_surprise"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "tickers": ticker_summaries,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-input", default="data/processed/tech_largecap_gold.parquet")
    parser.add_argument("--output-dir", default="data/external/earnings_fundamentals")
    parser.add_argument("--output-name", default="earnings_surprise_hf_sovai_tech_largecap.csv")
    parser.add_argument("--summary-name", default="earnings_surprise_hf_sovai_tech_largecap_summary.json")
    parser.add_argument("--dataset-name", default="sovai/earnings_surprise")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-day-diff", type=int, default=3)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    target_events = _load_events(args.events_input)
    out, summary = build_hf_earnings_surprise_events(
        target_events=target_events,
        dataset_name=args.dataset_name,
        split=args.split,
        max_day_diff=args.max_day_diff,
        streaming=args.streaming,
    )
    out_dir = ensure_dir(args.output_dir)
    write_csv(out, out_dir / args.output_name)
    save_json(summary, out_dir / args.summary_name)
    LOGGER.info("Saved Hugging Face earnings surprise events to %s", out_dir / args.output_name)


if __name__ == "__main__":
    main()
