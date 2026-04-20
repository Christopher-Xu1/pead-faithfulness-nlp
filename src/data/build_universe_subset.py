from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.data.transcript_corpus import MAG7_CANONICAL_TICKERS
from src.utils.io import ensure_dir, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
SP500_CONSTITUENTS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
}


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", ".")


def _parse_ticker_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {_normalize_symbol(item) for item in raw.split(",") if _normalize_symbol(item)}


def fetch_nasdaq_screener() -> pd.DataFrame:
    response = requests.get(NASDAQ_SCREENER_URL, headers=NASDAQ_HEADERS, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("rows") or []
    if not rows:
        raise RuntimeError("Nasdaq screener returned no stock rows.")

    frame = pd.DataFrame(rows)
    frame["ticker"] = frame["symbol"].map(_normalize_symbol)
    frame["market_cap_usd"] = pd.to_numeric(frame.get("marketCap"), errors="coerce")
    frame["nasdaq_sector"] = frame.get("sector", "").fillna("").astype(str).str.strip()
    frame["nasdaq_industry"] = frame.get("industry", "").fillna("").astype(str).str.strip()
    frame["sector"] = frame["nasdaq_sector"]
    frame["industry"] = frame["nasdaq_industry"]
    frame["company_name"] = frame.get("name", "").fillna("").astype(str).str.strip()
    frame = frame.sort_values("market_cap_usd", ascending=False).drop_duplicates("ticker", keep="first")
    return frame.reset_index(drop=True)


def fetch_sp500_constituents(url: str = SP500_CONSTITUENTS_URL) -> pd.DataFrame:
    frame = pd.read_csv(url)
    required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"S&P 500 constituents payload is missing columns: {sorted(missing)}")
    out = frame.rename(
        columns={
            "Symbol": "ticker",
            "Security": "gics_company_name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_industry",
        }
    )
    out["ticker"] = out["ticker"].map(_normalize_symbol)
    out["cik"] = pd.to_numeric(out["CIK"], errors="coerce").astype("Int64")
    return out[["ticker", "cik", "gics_company_name", "gics_sector", "gics_industry"]]


def apply_classification_source(
    universe_df: pd.DataFrame,
    classification_source: str,
    sp500_constituents_url: str = SP500_CONSTITUENTS_URL,
) -> pd.DataFrame:
    if classification_source == "nasdaq":
        out = universe_df.copy()
        out["classification_source"] = "nasdaq"
        return out
    if classification_source != "sp500_gics":
        raise ValueError(f"Unsupported classification_source={classification_source!r}")

    gics = fetch_sp500_constituents(sp500_constituents_url)
    out = universe_df.merge(gics, on="ticker", how="left")
    has_gics = out["gics_sector"].notna()
    out.loc[has_gics, "sector"] = out.loc[has_gics, "gics_sector"]
    out.loc[has_gics, "industry"] = out.loc[has_gics, "gics_industry"]
    out.loc[has_gics, "company_name"] = out.loc[has_gics, "gics_company_name"]
    out["classification_source"] = "sp500_gics"
    return out


def load_frozen_universe(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise ValueError(f"Frozen universe file must contain a ticker column: {path}")
    frame["ticker"] = frame["ticker"].map(_normalize_symbol)
    if "market_cap_usd" in frame.columns:
        frame["market_cap_usd"] = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
    return frame


def build_subset(
    corpus_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    min_market_cap: float,
    sector: str,
    forced_tickers: set[str],
    snapshot_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    corpus = corpus_df.copy()
    corpus["ticker"] = corpus["ticker"].map(_normalize_symbol)
    corpus_tickers = set(corpus["ticker"].dropna().unique())

    universe = universe_df.copy()
    universe["ticker"] = universe["ticker"].map(_normalize_symbol)
    universe["market_cap_usd"] = pd.to_numeric(universe.get("market_cap_usd"), errors="coerce")
    universe["sector"] = universe.get("sector", "").fillna("").astype(str).str.strip()
    universe["industry"] = universe.get("industry", "").fillna("").astype(str).str.strip()
    universe["company_name"] = universe.get("company_name", universe.get("name", "")).fillna("").astype(str).str.strip()

    in_corpus = universe["ticker"].isin(corpus_tickers)
    large_enough = universe["market_cap_usd"] >= min_market_cap
    sector_match = universe["sector"].str.casefold() == sector.casefold()
    forced_match = universe["ticker"].isin(forced_tickers)
    selected_mask = in_corpus & large_enough & (sector_match | forced_match)

    selected = universe.loc[selected_mask].copy()
    if selected.empty:
        raise RuntimeError("No tickers matched the universe screen and local transcript corpus.")

    selected["included_by"] = "sector_filter"
    selected.loc[selected["ticker"].isin(forced_tickers) & ~sector_match[selected.index], "included_by"] = "forced_include"
    selected.loc[selected["ticker"].isin(forced_tickers) & sector_match[selected.index], "included_by"] = (
        "sector_filter+forced_include"
    )
    selected["snapshot_date"] = snapshot_date
    classification_sources = sorted(str(item) for item in selected.get("classification_source", pd.Series(["nasdaq"])).dropna().unique())
    selected["universe_source"] = "+".join(["nasdaq_screener", *classification_sources])
    selected["universe_source_url"] = NASDAQ_SCREENER_URL

    corpus_stats = (
        corpus.groupby("ticker")
        .agg(
            calls_in_gold_corpus=("ticker", "size"),
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
        )
        .reset_index()
    )
    selected = selected.merge(corpus_stats, on="ticker", how="left")
    selected["snapshot_log_market_cap"] = np.log(selected["market_cap_usd"])
    selected["snapshot_market_cap_percentile"] = selected["market_cap_usd"].rank(pct=True)
    selected = selected.sort_values(["included_by", "market_cap_usd"], ascending=[True, False]).reset_index(drop=True)

    selected_tickers = set(selected["ticker"])
    subset = corpus.loc[corpus["ticker"].isin(selected_tickers)].sort_values(["event_date", "ticker"]).reset_index(drop=True)
    universe_metadata = selected[
        [
            "ticker",
            "market_cap_usd",
            "snapshot_log_market_cap",
            "snapshot_market_cap_percentile",
            "sector",
            "industry",
            "company_name",
            "cik",
            "calls_in_gold_corpus",
            "first_event_date",
            "last_event_date",
            "included_by",
            "classification_source",
            "universe_source",
            "snapshot_date",
        ]
    ].rename(
        columns={
            "market_cap_usd": "snapshot_market_cap_usd",
            "sector": "universe_sector",
            "industry": "universe_industry",
            "company_name": "universe_company_name",
            "cik": "universe_cik",
            "calls_in_gold_corpus": "universe_calls_in_gold_corpus",
            "first_event_date": "universe_first_event_date",
            "last_event_date": "universe_last_event_date",
            "included_by": "universe_included_by",
            "classification_source": "universe_classification_source",
            "snapshot_date": "universe_snapshot_date",
        }
    )
    subset = subset.merge(universe_metadata, on="ticker", how="left")

    summary = {
        "snapshot_date": snapshot_date,
        "source_corpus_rows": int(len(corpus)),
        "source_corpus_tickers": int(corpus["ticker"].nunique()),
        "screener_rows": int(len(universe)),
        "min_market_cap_usd": float(min_market_cap),
        "sector_filter": sector,
        "classification_source": ",".join(
            sorted(str(item) for item in selected.get("classification_source", pd.Series(["nasdaq"])).dropna().unique())
        ),
        "forced_include_tickers": sorted(forced_tickers),
        "selected_tickers": int(selected["ticker"].nunique()),
        "selected_calls": int(len(subset)),
        "selected_mag7_tickers": int(selected["ticker"].isin(MAG7_CANONICAL_TICKERS).sum()),
        "selected_first_event_date": str(pd.to_datetime(subset["event_date"]).min().date()),
        "selected_last_event_date": str(pd.to_datetime(subset["event_date"]).max().date()),
    }
    return subset, selected, summary


def write_summary(path: str | Path, summary: dict[str, Any], selected: pd.DataFrame) -> None:
    lines = [
        "# Tech Large-Cap Universe Summary",
        "",
        f"- Snapshot date: `{summary['snapshot_date']}`",
        f"- Source corpus rows: `{summary['source_corpus_rows']}`",
        f"- Source corpus tickers: `{summary['source_corpus_tickers']}`",
        f"- Screener rows: `{summary['screener_rows']}`",
        f"- Minimum market cap: `${summary['min_market_cap_usd']:,.0f}`",
        f"- Sector filter: `{summary['sector_filter']}`",
        f"- Classification source: `{summary['classification_source']}`",
        f"- Forced include tickers: `{', '.join(summary['forced_include_tickers'])}`",
        f"- Selected tickers: `{summary['selected_tickers']}`",
        f"- Selected calls: `{summary['selected_calls']}`",
        f"- Selected Mag7 tickers: `{summary['selected_mag7_tickers']}`",
        f"- Event-date range: `{summary['selected_first_event_date']}` to `{summary['selected_last_event_date']}`",
        "",
        "Top selected tickers by local gold-corpus call count:",
        "",
    ]
    top = selected.sort_values("calls_in_gold_corpus", ascending=False).head(25)
    lines.append("| ticker | calls | sector | industry | market_cap_usd | included_by |")
    lines.append("| --- | ---: | --- | --- | ---: | --- |")
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.ticker} | {int(row.calls_in_gold_corpus)} | {row.sector} | {row.industry} | "
            f"{float(row.market_cap_usd):.0f} | {row.included_by} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    today = date.today().isoformat()
    parser.add_argument("--corpus-input", default="data/processed/gold_corpus.parquet")
    parser.add_argument("--processed-output", default="data/processed/tech_largecap_gold.parquet")
    parser.add_argument("--universe-output", default=f"configs/data/universes/tech_largecap_nasdaq_{today}.csv")
    parser.add_argument("--summary-output", default="reports/tech_largecap_universe_summary.md")
    parser.add_argument("--frozen-universe-input", default=None)
    parser.add_argument("--min-market-cap", type=float, default=10_000_000_000)
    parser.add_argument("--sector", default="Information Technology")
    parser.add_argument("--classification-source", choices=["sp500_gics", "nasdaq"], default="sp500_gics")
    parser.add_argument("--sp500-constituents-url", default=SP500_CONSTITUENTS_URL)
    parser.add_argument("--force-include", default=",".join(sorted(MAG7_CANONICAL_TICKERS)))
    parser.add_argument("--snapshot-date", default=today)
    args = parser.parse_args()

    corpus_df = pd.read_parquet(args.corpus_input)
    if args.frozen_universe_input:
        universe_df = load_frozen_universe(args.frozen_universe_input)
    else:
        universe_df = apply_classification_source(
            fetch_nasdaq_screener(),
            classification_source=args.classification_source,
            sp500_constituents_url=args.sp500_constituents_url,
        )
    forced_tickers = _parse_ticker_list(args.force_include)

    subset_df, selected_df, summary = build_subset(
        corpus_df=corpus_df,
        universe_df=universe_df,
        min_market_cap=args.min_market_cap,
        sector=args.sector,
        forced_tickers=forced_tickers,
        snapshot_date=args.snapshot_date,
    )

    ensure_dir(Path(args.processed_output).parent)
    subset_df.to_parquet(args.processed_output, index=False)
    write_csv(selected_df, args.universe_output)
    write_summary(args.summary_output, summary, selected_df)
    LOGGER.info(
        "Built tech large-cap subset: tickers=%d calls=%d output=%s",
        summary["selected_tickers"],
        summary["selected_calls"],
        args.processed_output,
    )


if __name__ == "__main__":
    main()
