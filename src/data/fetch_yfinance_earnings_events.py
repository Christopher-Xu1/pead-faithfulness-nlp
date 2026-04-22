from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from src.utils.io import ensure_dir, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _normalize_earnings_dates(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "source_event_date",
                "reported_eps",
                "estimated_eps",
                "eps_surprise",
                "eps_surprise_pct",
            ]
        )

    out = frame.reset_index().copy()
    date_column = out.columns[0]
    out["source_event_date"] = pd.to_datetime(out[date_column]).dt.tz_localize(None).dt.normalize()
    out["ticker"] = ticker
    out["estimated_eps"] = pd.to_numeric(out.get("EPS Estimate"), errors="coerce")
    out["reported_eps"] = pd.to_numeric(out.get("Reported EPS"), errors="coerce")
    out["eps_surprise_pct"] = pd.to_numeric(out.get("Surprise(%)"), errors="coerce") / 100.0
    out["eps_surprise"] = out["reported_eps"] - out["estimated_eps"]
    return out[
        [
            "ticker",
            "source_event_date",
            "reported_eps",
            "estimated_eps",
            "eps_surprise",
            "eps_surprise_pct",
        ]
    ].drop_duplicates(subset=["ticker", "source_event_date"])


def _match_events_for_ticker(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    max_day_diff: int,
) -> pd.DataFrame:
    if target_df.empty:
        return target_df.copy()

    target = target_df.copy()
    target["event_date"] = pd.to_datetime(target["event_date"]).dt.normalize()
    if source_df.empty:
        target["source_event_date"] = pd.NaT
        target["date_diff_days"] = pd.NA
        target["match_status"] = "missing_source"
        return target

    matched_rows: list[dict[str, Any]] = []
    source = source_df.copy()
    source["source_event_date"] = pd.to_datetime(source["source_event_date"]).dt.normalize()
    source = source.sort_values("source_event_date").reset_index(drop=True)

    for row in target.itertuples(index=False):
        diffs = (source["source_event_date"] - row.event_date).abs().dt.days
        best_idx = int(diffs.idxmin())
        best_diff = int(diffs.iloc[best_idx])
        match = source.iloc[best_idx]
        matched = {
            "ticker": row.ticker,
            "event_date": row.event_date.strftime("%Y-%m-%d"),
            "source_event_date": match["source_event_date"].strftime("%Y-%m-%d"),
            "date_diff_days": best_diff,
            "match_status": "matched" if best_diff <= max_day_diff else "outside_tolerance",
            "reported_eps": match["reported_eps"] if best_diff <= max_day_diff else pd.NA,
            "estimated_eps": match["estimated_eps"] if best_diff <= max_day_diff else pd.NA,
            "eps_surprise": match["eps_surprise"] if best_diff <= max_day_diff else pd.NA,
            "eps_surprise_pct": match["eps_surprise_pct"] if best_diff <= max_day_diff else pd.NA,
        }
        matched_rows.append(matched)
    return pd.DataFrame(matched_rows)


def fetch_yfinance_earnings_events(
    metadata_df: pd.DataFrame,
    limit: int = 128,
    max_day_diff: int = 3,
    max_retries: int = 5,
    retry_sleep_seconds: int = 15,
    pause_between_tickers: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    calls = metadata_df[["ticker", "event_date"]].drop_duplicates().copy()
    calls["ticker"] = calls["ticker"].astype(str).str.upper().str.strip()
    calls["event_date"] = pd.to_datetime(calls["event_date"]).dt.strftime("%Y-%m-%d")

    matched_frames: list[pd.DataFrame] = []
    ticker_summaries: list[dict[str, Any]] = []
    for ticker, target_df in calls.groupby("ticker", sort=True):
        LOGGER.info("Fetching Yahoo earnings dates for %s", ticker)
        source_raw = None
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                source_raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
                last_error = None
                break
            except YFRateLimitError as exc:
                last_error = exc
                sleep_seconds = retry_sleep_seconds * (attempt + 1)
                LOGGER.warning(
                    "Yahoo rate limited %s on attempt %s/%s; sleeping %ss",
                    ticker,
                    attempt + 1,
                    max_retries,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        if last_error is not None:
            raise last_error
        source_df = _normalize_earnings_dates(source_raw, ticker=ticker)
        matched_df = _match_events_for_ticker(target_df, source_df, max_day_diff=max_day_diff)
        matched_frames.append(matched_df)
        ticker_summaries.append(
            {
                "ticker": ticker,
                "target_events": int(len(target_df)),
                "source_events": int(len(source_df)),
                "matched_events": int((matched_df["match_status"] == "matched").sum()),
                "coverage": float((matched_df["match_status"] == "matched").mean()) if len(matched_df) else 0.0,
            }
        )
        if pause_between_tickers > 0:
            time.sleep(pause_between_tickers)

    out = pd.concat(matched_frames, ignore_index=True).sort_values(["ticker", "event_date"]).reset_index(drop=True)
    out["eps_beat_flag"] = (pd.to_numeric(out["eps_surprise"], errors="coerce") > 0).astype("Int64")
    out["eps_miss_flag"] = (pd.to_numeric(out["eps_surprise"], errors="coerce") < 0).astype("Int64")
    out["eps_meet_flag"] = (pd.to_numeric(out["eps_surprise"], errors="coerce") == 0).astype("Int64")
    out["eps_beat_miss"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[out["eps_beat_flag"] == 1, "eps_beat_miss"] = "beat"
    out.loc[out["eps_miss_flag"] == 1, "eps_beat_miss"] = "miss"
    out.loc[out["eps_meet_flag"] == 1, "eps_beat_miss"] = "meet"

    summary = {
        "rows": int(len(out)),
        "matched_rows": int((out["match_status"] == "matched").sum()),
        "coverage": float((out["match_status"] == "matched").mean()) if len(out) else 0.0,
        "eps_surprise_coverage": float(pd.to_numeric(out["eps_surprise"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "eps_surprise_pct_coverage": float(pd.to_numeric(out["eps_surprise_pct"], errors="coerce").notna().mean())
        if len(out)
        else 0.0,
        "reported_eps_coverage": float(pd.to_numeric(out["reported_eps"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "estimated_eps_coverage": float(pd.to_numeric(out["estimated_eps"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "tickers": ticker_summaries,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-input", default="data/raw/metadata/call_metadata.csv")
    parser.add_argument("--output-dir", default="data/external/earnings_fundamentals")
    parser.add_argument("--output-name", default="earnings_events_yfinance_mag7.csv")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-day-diff", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=int, default=15)
    parser.add_argument("--pause-between-tickers", type=int, default=3)
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_input)
    events_df, summary = fetch_yfinance_earnings_events(
        metadata_df=metadata_df,
        limit=args.limit,
        max_day_diff=args.max_day_diff,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        pause_between_tickers=args.pause_between_tickers,
    )

    out_dir = ensure_dir(args.output_dir)
    write_csv(events_df, out_dir / args.output_name)
    save_json(summary, out_dir / "earnings_events_yfinance_summary.json")
    LOGGER.info("Saved Yahoo earnings events to %s", out_dir / args.output_name)


if __name__ == "__main__":
    main()
