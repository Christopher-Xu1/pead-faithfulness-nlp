from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils.io import ensure_dir, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_USER_AGENT = "pead-faithfulness-nlp/0.1 research"

REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "CapitalExpendituresIncurredButNotYetPaid",
]
SHARES_CONCEPTS = ["EntityCommonStockSharesOutstanding"]


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", ".")


def _normalize_cik(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return f"{int(float(text)):010d}"
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(10) if digits else None


def _load_events(path: str | Path, universe_path: str | Path | None = None) -> pd.DataFrame:
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
        raise ValueError(f"Event input is missing required columns: {sorted(missing)}")

    out["ticker"] = out["ticker"].map(_normalize_ticker)
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.normalize()
    if "universe_cik" not in out.columns and "cik" in out.columns:
        out["universe_cik"] = out["cik"]

    if universe_path is not None and Path(universe_path).exists():
        universe = pd.read_csv(universe_path)
        universe["ticker"] = universe["ticker"].map(_normalize_ticker)
        cik_col = "universe_cik" if "universe_cik" in universe.columns else "cik"
        if cik_col in universe.columns:
            out = out.merge(universe[["ticker", cik_col]].rename(columns={cik_col: "universe_cik"}), on="ticker", how="left", suffixes=("", "_universe"))
            if "universe_cik_universe" in out.columns:
                out["universe_cik"] = out["universe_cik"].combine_first(out["universe_cik_universe"])
                out = out.drop(columns=["universe_cik_universe"])

    out["cik"] = out.get("universe_cik", pd.Series(pd.NA, index=out.index)).map(_normalize_cik)
    return out[["call_id", "ticker", "event_date", "cik"]].drop_duplicates().reset_index(drop=True)


def _fetch_json(url: str, headers: dict[str, str], cache_path: Path | None = None, refresh: bool = False) -> dict[str, Any]:
    if cache_path is not None and cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if cache_path is not None:
        ensure_dir(cache_path.parent)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def fetch_companyfacts(cik: str, user_agent: str, cache_dir: str | Path, refresh: bool = False) -> dict[str, Any]:
    cache_path = Path(cache_dir) / f"companyfacts_CIK{cik}.json"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
    return _fetch_json(SEC_COMPANYFACTS_URL.format(cik=cik), headers=headers, cache_path=cache_path, refresh=refresh)


def _extract_facts(payload: dict[str, Any], taxonomy: str, concepts: list[str], units: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    facts = payload.get("facts", {}).get(taxonomy, {})
    for concept in concepts:
        concept_payload = facts.get(concept, {})
        for unit, unit_facts in concept_payload.get("units", {}).items():
            if unit not in units:
                continue
            for fact in unit_facts:
                rows.append(
                    {
                        "concept": concept,
                        "unit": unit,
                        "value": pd.to_numeric(fact.get("val"), errors="coerce"),
                        "start": pd.to_datetime(fact.get("start"), errors="coerce"),
                        "end": pd.to_datetime(fact.get("end"), errors="coerce"),
                        "filed": pd.to_datetime(fact.get("filed"), errors="coerce"),
                        "form": fact.get("form"),
                        "fy": fact.get("fy"),
                        "fp": fact.get("fp"),
                        "frame": fact.get("frame"),
                        "accn": fact.get("accn"),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["value", "end"]).copy()
    frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    frame["filed"] = pd.to_datetime(frame["filed"], errors="coerce")
    frame["period_days"] = np.nan
    has_start = frame["start"].notna()
    frame.loc[has_start, "period_days"] = (frame.loc[has_start, "end"] - frame.loc[has_start, "start"]).dt.days + 1
    return frame.sort_values(["end", "filed", "concept"]).reset_index(drop=True)


def _derive_quarterly_periods(frame: pd.DataFrame, abs_value: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame
    direct = frame[(frame["period_days"].between(1, 140, inclusive="both")) | frame["start"].isna()].copy()
    derived_rows: list[dict[str, Any]] = []
    cumulative = frame[frame["period_days"] > 140].dropna(subset=["start"]).copy()
    for _, row in cumulative.iterrows():
        prior = cumulative[
            (cumulative["concept"] == row["concept"])
            & (cumulative["unit"] == row["unit"])
            & (cumulative["start"] == row["start"])
            & (cumulative["end"] < row["end"])
        ].sort_values("end")
        if prior.empty:
            continue
        prev = prior.iloc[-1]
        period_days = int((row["end"] - prev["end"]).days)
        if period_days <= 0 or period_days > 140:
            continue
        item = row.to_dict()
        item["value"] = row["value"] - prev["value"]
        item["start"] = prev["end"] + pd.Timedelta(days=1)
        item["period_days"] = period_days
        item["derived_from_cumulative"] = 1
        derived_rows.append(item)
    derived = pd.DataFrame(derived_rows)
    if not direct.empty:
        direct["derived_from_cumulative"] = 0
    out = pd.concat([direct, derived], ignore_index=True) if not derived.empty else direct
    if out.empty:
        return out
    if abs_value:
        out["value"] = out["value"].abs()
    return out.dropna(subset=["value", "end"]).sort_values(["end", "filed", "concept"]).reset_index(drop=True)


def _select_statement_fact(
    facts: pd.DataFrame,
    event_date: pd.Timestamp,
    max_period_end_lag_days: int,
) -> pd.Series | None:
    if facts.empty:
        return None
    candidates = facts[facts["end"] <= event_date].copy()
    candidates["period_end_lag_days"] = (event_date - candidates["end"]).dt.days
    candidates = candidates[candidates["period_end_lag_days"].between(0, max_period_end_lag_days, inclusive="both")]
    if candidates.empty:
        return None
    candidates["filed_lag_days"] = (event_date - candidates["filed"]).dt.days
    candidates = candidates.sort_values(
        ["period_end_lag_days", "derived_from_cumulative", "filed_lag_days", "filed"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    return candidates.iloc[0]


def _select_shares_fact(
    facts: pd.DataFrame,
    event_date: pd.Timestamp,
    max_staleness_days: int,
) -> pd.Series | None:
    if facts.empty:
        return None
    candidates = facts[
        (facts["end"] <= event_date)
        & (facts["filed"] <= event_date)
        & (facts["value"].between(1_000_000, 100_000_000_000, inclusive="both"))
    ].copy()
    candidates["shares_staleness_days"] = (event_date - candidates["end"]).dt.days
    candidates = candidates[candidates["shares_staleness_days"].between(0, max_staleness_days, inclusive="both")]
    if candidates.empty:
        return None
    return candidates.sort_values(["end", "filed"], ascending=[False, False]).iloc[0]


def _download_yahoo_close(symbol: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    period1 = int(pd.Timestamp(min_date).tz_localize("UTC").timestamp())
    period2 = int((pd.Timestamp(max_date) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history|split"
    )
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise ValueError(f"No Yahoo chart result for {symbol}")
    timestamps = result.get("timestamp") or []
    close = (result.get("indicators", {}).get("quote") or [{}])[0].get("close")
    if not timestamps or not close:
        raise ValueError(f"Yahoo chart result missing close prices for {symbol}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "split_adjusted_close": pd.to_numeric(close, errors="coerce"),
        }
    ).dropna()
    split_rows: list[tuple[pd.Timestamp, float]] = []
    for item in (result.get("events", {}).get("splits") or {}).values():
        split_date = pd.to_datetime(item.get("date"), unit="s", utc=True).tz_convert(None).normalize()
        numerator = pd.to_numeric(item.get("numerator"), errors="coerce")
        denominator = pd.to_numeric(item.get("denominator"), errors="coerce")
        if pd.notna(numerator) and pd.notna(denominator) and float(denominator) != 0:
            split_rows.append((split_date, float(numerator) / float(denominator)))
    out["split_factor_to_raw"] = 1.0
    for split_date, split_factor in split_rows:
        out.loc[out["date"] < split_date, "split_factor_to_raw"] *= split_factor
    out["close"] = out["split_adjusted_close"] * out["split_factor_to_raw"]
    out["ticker"] = symbol.upper().replace("-", ".")
    return out.sort_values("date").reset_index(drop=True)


def _build_price_cache(events_df: pd.DataFrame, pause_seconds: float) -> pd.DataFrame:
    min_date = events_df["event_date"].min() - pd.Timedelta(days=10)
    max_date = events_df["event_date"].max()
    frames: list[pd.DataFrame] = []
    for ticker in sorted(events_df["ticker"].unique()):
        symbol = ticker.replace(".", "-")
        LOGGER.info("Fetching Yahoo close history for %s", ticker)
        frames.append(_download_yahoo_close(symbol, min_date=min_date, max_date=max_date))
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return pd.concat(frames, ignore_index=True)


def _select_pre_event_close(prices: pd.DataFrame, ticker: str, event_date: pd.Timestamp) -> pd.Series | None:
    ticker_prices = prices[(prices["ticker"] == ticker) & (prices["date"] < event_date)].copy()
    if ticker_prices.empty:
        return None
    return ticker_prices.sort_values("date").iloc[-1]


def build_sec_event_snapshots(
    events_df: pd.DataFrame,
    user_agent: str,
    cache_dir: str | Path,
    max_period_end_lag_days: int = 120,
    max_shares_staleness_days: int = 730,
    pause_between_tickers: float = 0.2,
    refresh_sec_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prices = _build_price_cache(events_df, pause_seconds=pause_between_tickers)
    fundamentals_rows: list[dict[str, Any]] = []
    market_cap_rows: list[dict[str, Any]] = []

    for ticker, ticker_events in events_df.groupby("ticker", sort=True):
        cik = ticker_events["cik"].dropna().iloc[0] if ticker_events["cik"].notna().any() else None
        revenue_facts = capex_facts = shares_facts = pd.DataFrame()
        if cik:
            LOGGER.info("Fetching SEC companyfacts for %s CIK%s", ticker, cik)
            payload = fetch_companyfacts(str(cik), user_agent=user_agent, cache_dir=cache_dir, refresh=refresh_sec_cache)
            revenue_facts = _derive_quarterly_periods(_extract_facts(payload, "us-gaap", REVENUE_CONCEPTS, {"USD"}))
            capex_facts = _derive_quarterly_periods(
                _extract_facts(payload, "us-gaap", CAPEX_CONCEPTS, {"USD"}),
                abs_value=True,
            )
            shares_facts = _extract_facts(payload, "dei", SHARES_CONCEPTS, {"shares"})
        else:
            LOGGER.warning("Missing CIK for %s; SEC fundamentals and shares will be empty.", ticker)

        for event in ticker_events.itertuples(index=False):
            event_date = pd.Timestamp(event.event_date)
            revenue = _select_statement_fact(revenue_facts, event_date, max_period_end_lag_days)
            capex = _select_statement_fact(capex_facts, event_date, max_period_end_lag_days)
            shares = _select_shares_fact(shares_facts, event_date, max_shares_staleness_days)
            close = _select_pre_event_close(prices, ticker=ticker, event_date=event_date)

            fundamentals_rows.append(
                {
                    "call_id": event.call_id,
                    "ticker": ticker,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "cik": cik,
                    "reported_revenue": revenue["value"] if revenue is not None else np.nan,
                    "reported_revenue_source_concept": revenue["concept"] if revenue is not None else pd.NA,
                    "reported_revenue_period_end": revenue["end"].strftime("%Y-%m-%d") if revenue is not None else pd.NA,
                    "reported_revenue_period_days": revenue["period_days"] if revenue is not None else np.nan,
                    "reported_revenue_period_end_lag_days": revenue["period_end_lag_days"] if revenue is not None else np.nan,
                    "reported_revenue_derived_from_cumulative": revenue["derived_from_cumulative"] if revenue is not None else np.nan,
                    "reported_capex": capex["value"] if capex is not None else np.nan,
                    "reported_capex_source_concept": capex["concept"] if capex is not None else pd.NA,
                    "reported_capex_period_end": capex["end"].strftime("%Y-%m-%d") if capex is not None else pd.NA,
                    "reported_capex_period_days": capex["period_days"] if capex is not None else np.nan,
                    "reported_capex_period_end_lag_days": capex["period_end_lag_days"] if capex is not None else np.nan,
                    "reported_capex_derived_from_cumulative": capex["derived_from_cumulative"] if capex is not None else np.nan,
                }
            )

            hist_market_cap = np.nan
            if shares is not None and close is not None:
                hist_market_cap = float(shares["value"]) * float(close["close"])
            market_cap_rows.append(
                {
                    "call_id": event.call_id,
                    "ticker": ticker,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "cik": cik,
                    "hist_market_cap_price_date": close["date"].strftime("%Y-%m-%d") if close is not None else pd.NA,
                    "hist_market_cap_close": close["close"] if close is not None else np.nan,
                    "hist_market_cap_split_adjusted_close": close["split_adjusted_close"] if close is not None else np.nan,
                    "hist_market_cap_split_factor_to_raw": close["split_factor_to_raw"] if close is not None else np.nan,
                    "hist_market_cap_shares_outstanding": shares["value"] if shares is not None else np.nan,
                    "hist_market_cap_shares_end_date": shares["end"].strftime("%Y-%m-%d") if shares is not None else pd.NA,
                    "hist_market_cap_shares_filed_date": shares["filed"].strftime("%Y-%m-%d") if shares is not None else pd.NA,
                    "hist_market_cap_price_lag_days": (event_date - close["date"]).days if close is not None else np.nan,
                    "hist_market_cap_shares_staleness_days": shares["shares_staleness_days"] if shares is not None else np.nan,
                    "hist_market_cap": hist_market_cap,
                    "hist_market_cap_source": "yahoo_chart_close+sec_companyfacts_dei"
                    if np.isfinite(hist_market_cap)
                    else pd.NA,
                }
            )
        if pause_between_tickers > 0:
            time.sleep(pause_between_tickers)

    fundamentals_df = pd.DataFrame(fundamentals_rows)
    market_cap_df = pd.DataFrame(market_cap_rows)
    market_cap_df["hist_log_market_cap"] = np.log(pd.to_numeric(market_cap_df["hist_market_cap"], errors="coerce"))
    market_cap_df["hist_market_cap_percentile"] = market_cap_df["hist_market_cap"].rank(pct=True)

    summary = {
        "rows": int(len(events_df)),
        "tickers": int(events_df["ticker"].nunique()),
        "reported_revenue_coverage": float(pd.to_numeric(fundamentals_df["reported_revenue"], errors="coerce").notna().mean()),
        "reported_capex_coverage": float(pd.to_numeric(fundamentals_df["reported_capex"], errors="coerce").notna().mean()),
        "hist_market_cap_coverage": float(pd.to_numeric(market_cap_df["hist_market_cap"], errors="coerce").notna().mean()),
        "shares_outstanding_coverage": float(
            pd.to_numeric(market_cap_df["hist_market_cap_shares_outstanding"], errors="coerce").notna().mean()
        ),
        "pre_event_close_coverage": float(pd.to_numeric(market_cap_df["hist_market_cap_close"], errors="coerce").notna().mean()),
    }
    return fundamentals_df, market_cap_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-input", default="data/processed/tech_largecap_gold.parquet")
    parser.add_argument("--universe-input", default="configs/data/universes/tech_largecap_nasdaq_2026-04-15.csv")
    parser.add_argument("--output-dir", default="data/external/sec_company_facts")
    parser.add_argument("--fundamentals-output-name", default="sec_event_fundamentals_tech_largecap.csv")
    parser.add_argument("--market-cap-output-name", default="market_cap_snapshots_tech_largecap.csv")
    parser.add_argument("--cache-dir", default="data/external/sec_company_facts/raw")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--max-period-end-lag-days", type=int, default=120)
    parser.add_argument("--max-shares-staleness-days", type=int, default=730)
    parser.add_argument("--pause-between-tickers", type=float, default=0.2)
    parser.add_argument("--refresh-sec-cache", action="store_true")
    args = parser.parse_args()

    events_df = _load_events(args.events_input, universe_path=args.universe_input)
    fundamentals_df, market_cap_df, summary = build_sec_event_snapshots(
        events_df=events_df,
        user_agent=args.user_agent,
        cache_dir=args.cache_dir,
        max_period_end_lag_days=args.max_period_end_lag_days,
        max_shares_staleness_days=args.max_shares_staleness_days,
        pause_between_tickers=args.pause_between_tickers,
        refresh_sec_cache=args.refresh_sec_cache,
    )

    out_dir = ensure_dir(args.output_dir)
    write_csv(fundamentals_df, out_dir / args.fundamentals_output_name)
    write_csv(market_cap_df, out_dir / args.market_cap_output_name)
    save_json(summary, out_dir / "sec_event_snapshots_summary.json")
    LOGGER.info("Saved SEC event snapshots to %s", out_dir)


if __name__ == "__main__":
    main()
