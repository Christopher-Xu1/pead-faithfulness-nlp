from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils.io import ensure_dir, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

FMP_STABLE_EARNINGS_URL = "https://financialmodelingprep.com/stable/earnings"
FMP_LEGACY_EARNINGS_URL = "https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}"

EPS_ACTUAL_ALIASES = ["epsActual", "eps", "reportedEPS", "reported_eps", "actual_eps"]
EPS_ESTIMATE_ALIASES = ["epsEstimated", "epsEstimate", "estimatedEPS", "estimated_eps", "consensus_eps"]
REVENUE_ACTUAL_ALIASES = ["revenueActual", "revenue", "reported_revenue", "actual_revenue"]
REVENUE_ESTIMATE_ALIASES = [
    "revenueEstimated",
    "revenueEstimate",
    "estimatedRevenue",
    "estimated_revenue",
    "consensus_revenue",
]


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", ".")


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    available = [name for name in candidates if name in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    out = pd.to_numeric(frame[available[0]], errors="coerce")
    for name in available[1:]:
        out = out.combine_first(pd.to_numeric(frame[name], errors="coerce"))
    return out


def _load_events(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix == ".parquet":
        frame = pd.read_parquet(input_path)
    else:
        frame = pd.read_csv(input_path)

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


def _fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    cache_path: Path | None,
    refresh: bool,
) -> list[dict[str, Any]]:
    if cache_path is not None and cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        response = session.get(url, params=params, timeout=60)
        if response.status_code in {401, 403}:
            raise RuntimeError(
                "FMP rejected the request. Set FMP_API_KEY or FINANCIALMODELINGPREP_API_KEY "
                "to a valid key before fetching historical estimates."
            )
        response.raise_for_status()
        payload = response.json()
        if cache_path is not None:
            ensure_dir(cache_path.parent)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
    if isinstance(payload, dict) and "historical" in payload:
        payload = payload["historical"]
    if isinstance(payload, dict) and "Error Message" in payload:
        raise RuntimeError(str(payload["Error Message"]))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def fetch_fmp_earnings_payload(
    ticker: str,
    api_key: str,
    session: requests.Session,
    endpoint: str,
    cache_dir: str | Path,
    refresh_cache: bool = False,
) -> list[dict[str, Any]]:
    symbol = ticker.replace(".", "-")
    cache_path = Path(cache_dir) / f"fmp_earnings_{endpoint}_{symbol}.json"
    if endpoint == "legacy":
        url = FMP_LEGACY_EARNINGS_URL.format(symbol=symbol)
        params = {"apikey": api_key}
    elif endpoint == "stable":
        url = FMP_STABLE_EARNINGS_URL
        params = {"symbol": symbol, "apikey": api_key}
    else:
        raise ValueError(f"Unsupported FMP endpoint: {endpoint}")
    return _fetch_json(session, url=url, params=params, cache_path=cache_path, refresh=refresh_cache)


def normalize_fmp_earnings_payload(payload: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(
            columns=[
                "ticker",
                "source_event_date",
                "reported_eps",
                "estimated_eps",
                "reported_revenue",
                "estimated_revenue",
            ]
        )

    raw = pd.DataFrame(payload)
    date_col = "date" if "date" in raw.columns else "fiscalDateEnding" if "fiscalDateEnding" in raw.columns else None
    if date_col is None:
        raise ValueError(f"FMP payload for {ticker} has no recognized date column")

    out = pd.DataFrame(index=raw.index)
    out["ticker"] = ticker
    out["source_event_date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.normalize()
    out["reported_eps"] = _first_present(raw, EPS_ACTUAL_ALIASES)
    out["estimated_eps"] = _first_present(raw, EPS_ESTIMATE_ALIASES)
    out["reported_revenue"] = _first_present(raw, REVENUE_ACTUAL_ALIASES)
    out["estimated_revenue"] = _first_present(raw, REVENUE_ESTIMATE_ALIASES)
    out["estimated_eps_source"] = np.where(out["estimated_eps"].notna(), "fmp", pd.NA)
    out["estimated_revenue_source"] = np.where(out["estimated_revenue"].notna(), "fmp", pd.NA)
    out = out.dropna(subset=["source_event_date"])
    return out.drop_duplicates(subset=["ticker", "source_event_date"], keep="last").reset_index(drop=True)


def match_events_for_ticker(target_df: pd.DataFrame, source_df: pd.DataFrame, max_day_diff: int) -> pd.DataFrame:
    target = target_df.copy()
    target["event_date"] = pd.to_datetime(target["event_date"]).dt.normalize()
    if source_df.empty:
        out = target.copy()
        out["source_event_date"] = pd.NaT
        out["date_diff_days"] = pd.NA
        out["match_status"] = "missing_source"
        for column in [
            "reported_eps",
            "estimated_eps",
            "reported_revenue",
            "estimated_revenue",
            "estimated_eps_source",
            "estimated_revenue_source",
        ]:
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
        rows.append(
            {
                "call_id": row.call_id,
                "ticker": row.ticker,
                "event_date": row.event_date.strftime("%Y-%m-%d"),
                "source_event_date": best["source_event_date"].strftime("%Y-%m-%d"),
                "date_diff_days": best_diff,
                "match_status": "matched" if is_match else "outside_tolerance",
                "reported_eps": best["reported_eps"] if is_match else pd.NA,
                "estimated_eps": best["estimated_eps"] if is_match else pd.NA,
                "reported_revenue": best["reported_revenue"] if is_match else pd.NA,
                "estimated_revenue": best["estimated_revenue"] if is_match else pd.NA,
                "estimated_eps_source": best["estimated_eps_source"] if is_match else pd.NA,
                "estimated_revenue_source": best["estimated_revenue_source"] if is_match else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def add_capex_proxy(
    estimates_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame | None,
    rolling_window: int = 4,
) -> pd.DataFrame:
    out = estimates_df.copy()
    out["estimated_capex"] = np.nan
    out["estimated_capex_source"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["estimated_capex_method"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["estimated_capex_is_proxy"] = pd.Series(0, index=out.index, dtype="Int64")
    if fundamentals_df is None or fundamentals_df.empty:
        return out

    fundamentals = fundamentals_df[["call_id", "ticker", "event_date", "reported_revenue", "reported_capex"]].copy()
    fundamentals["event_date"] = pd.to_datetime(fundamentals["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.merge(
        fundamentals.rename(
            columns={
                "reported_revenue": "sec_reported_revenue",
                "reported_capex": "sec_reported_capex",
            }
        ),
        on=["call_id", "ticker", "event_date"],
        how="left",
    )
    out["reported_revenue"] = pd.to_numeric(out["reported_revenue"], errors="coerce").combine_first(
        pd.to_numeric(out["sec_reported_revenue"], errors="coerce")
    )
    out["reported_capex"] = pd.to_numeric(out["sec_reported_capex"], errors="coerce")
    out = out.drop(columns=["sec_reported_revenue", "sec_reported_capex"])

    work = out.sort_values(["ticker", "event_date", "call_id"]).copy()
    ratio = pd.to_numeric(work["reported_capex"], errors="coerce") / pd.to_numeric(
        work["reported_revenue"], errors="coerce"
    ).abs()
    ratio = ratio.where((ratio > 0) & np.isfinite(ratio))
    prior_ratio = (
        ratio.groupby(work["ticker"])
        .transform(lambda values: values.shift(1).rolling(rolling_window, min_periods=1).median())
        .reindex(work.index)
    )
    work["prior_capex_to_revenue_ratio"] = prior_ratio
    proxy = pd.to_numeric(work["estimated_revenue"], errors="coerce") * work["prior_capex_to_revenue_ratio"]
    mask = proxy.notna() & pd.to_numeric(work["estimated_capex"], errors="coerce").isna()
    work.loc[mask, "estimated_capex"] = proxy[mask]
    work.loc[mask, "estimated_capex_source"] = "prior_capex_revenue_ratio_proxy"
    work.loc[mask, "estimated_capex_method"] = f"rolling_{rolling_window}_prior_capex_to_revenue_x_estimated_revenue"
    work.loc[mask, "estimated_capex_is_proxy"] = 1
    return work.sort_index()


def fetch_fmp_earnings_estimates(
    events_df: pd.DataFrame,
    api_key: str,
    endpoint: str = "stable",
    max_day_diff: int = 3,
    pause_between_tickers: float = 0.25,
    cache_dir: str | Path = "data/external/earnings_fundamentals/raw",
    refresh_cache: bool = False,
    fundamentals_df: pd.DataFrame | None = None,
    capex_proxy_window: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "pead-faithfulness-nlp/0.1 research"})
    matched_frames: list[pd.DataFrame] = []
    ticker_summaries: list[dict[str, Any]] = []

    for ticker, target_df in events_df.groupby("ticker", sort=True):
        LOGGER.info("Fetching FMP earnings estimates for %s", ticker)
        payload = fetch_fmp_earnings_payload(
            ticker=ticker,
            api_key=api_key,
            session=session,
            endpoint=endpoint,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )
        source_df = normalize_fmp_earnings_payload(payload, ticker=ticker)
        matched_df = match_events_for_ticker(target_df, source_df, max_day_diff=max_day_diff)
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

    out = pd.concat(matched_frames, ignore_index=True).sort_values(["ticker", "event_date", "call_id"]).reset_index(drop=True)
    out = add_capex_proxy(out, fundamentals_df=fundamentals_df, rolling_window=capex_proxy_window)
    out["eps_surprise"] = pd.to_numeric(out["reported_eps"], errors="coerce") - pd.to_numeric(out["estimated_eps"], errors="coerce")
    out["revenue_surprise"] = pd.to_numeric(out["reported_revenue"], errors="coerce") - pd.to_numeric(
        out["estimated_revenue"], errors="coerce"
    )
    out["capex_surprise"] = pd.to_numeric(out["reported_capex"], errors="coerce") - pd.to_numeric(
        out["estimated_capex"], errors="coerce"
    )

    summary = {
        "rows": int(len(out)),
        "matched_rows": int((out["match_status"] == "matched").sum()),
        "matched_coverage": float((out["match_status"] == "matched").mean()) if len(out) else 0.0,
        "estimated_eps_coverage": float(pd.to_numeric(out["estimated_eps"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "estimated_revenue_coverage": float(pd.to_numeric(out["estimated_revenue"], errors="coerce").notna().mean())
        if len(out)
        else 0.0,
        "estimated_capex_coverage": float(pd.to_numeric(out["estimated_capex"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "estimated_capex_proxy_coverage": float(out["estimated_capex_is_proxy"].fillna(0).astype(bool).mean()) if len(out) else 0.0,
        "eps_surprise_coverage": float(pd.to_numeric(out["eps_surprise"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "revenue_surprise_coverage": float(pd.to_numeric(out["revenue_surprise"], errors="coerce").notna().mean())
        if len(out)
        else 0.0,
        "capex_surprise_coverage": float(pd.to_numeric(out["capex_surprise"], errors="coerce").notna().mean()) if len(out) else 0.0,
        "tickers": ticker_summaries,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-input", default="data/processed/tech_largecap_gold.parquet")
    parser.add_argument("--fundamentals-input", default="data/external/sec_company_facts/sec_event_fundamentals_tech_largecap.csv")
    parser.add_argument("--output-dir", default="data/external/earnings_fundamentals")
    parser.add_argument("--output-name", default="earnings_estimates_fmp_tech_largecap.csv")
    parser.add_argument("--summary-name", default="earnings_estimates_fmp_tech_largecap_summary.json")
    parser.add_argument("--cache-dir", default="data/external/earnings_fundamentals/raw")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--endpoint", choices=["stable", "legacy"], default="stable")
    parser.add_argument("--max-day-diff", type=int, default=3)
    parser.add_argument("--pause-between-tickers", type=float, default=0.25)
    parser.add_argument("--capex-proxy-window", type=int, default=4)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("FMP_API_KEY") or os.getenv("FINANCIALMODELINGPREP_API_KEY")
    if not api_key:
        raise RuntimeError("Set FMP_API_KEY or FINANCIALMODELINGPREP_API_KEY before fetching FMP estimates.")

    events_df = _load_events(args.events_input)
    fundamentals_df = pd.read_csv(args.fundamentals_input) if Path(args.fundamentals_input).exists() else None
    estimates_df, summary = fetch_fmp_earnings_estimates(
        events_df=events_df,
        api_key=api_key,
        endpoint=args.endpoint,
        max_day_diff=args.max_day_diff,
        pause_between_tickers=args.pause_between_tickers,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        fundamentals_df=fundamentals_df,
        capex_proxy_window=args.capex_proxy_window,
    )

    out_dir = ensure_dir(args.output_dir)
    write_csv(estimates_df, out_dir / args.output_name)
    save_json(summary, out_dir / args.summary_name)
    LOGGER.info("Saved FMP earnings estimates to %s", out_dir / args.output_name)


if __name__ == "__main__":
    main()
