from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _maybe_write_sample_data(root: Path) -> None:
    transcripts_path = root / "data/raw/transcripts/transcripts.csv"
    prices_path = root / "data/raw/prices/daily_returns.csv"
    market_path = root / "data/external/market_index/sp500_returns.csv"
    metadata_path = root / "data/raw/metadata/call_metadata.csv"

    if transcripts_path.exists() and prices_path.exists() and market_path.exists() and metadata_path.exists():
        LOGGER.info("Raw data already exists. Skipping sample generation.")
        return

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    call_dates = pd.to_datetime(
        [
            "2021-01-28",
            "2021-04-29",
            "2021-07-29",
            "2021-10-28",
            "2022-01-27",
            "2022-04-28",
            "2022-07-28",
        ]
    )

    metadata_rows = []
    transcript_rows = []
    call_id = 1
    for ticker, event_date in zip(tickers, call_dates):
        cid = f"{ticker}_{event_date.date()}"
        metadata_rows.append(
            {
                "call_id": cid,
                "ticker": ticker,
                "event_date": event_date.strftime("%Y-%m-%d"),
            }
        )
        transcript_rows.extend(
            [
                {
                    "call_id": cid,
                    "ticker": ticker,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "turn_id": call_id,
                    "speaker_role": "management",
                    "section": "prepared remarks",
                    "text": f"{ticker} prepared remarks with outlook and guidance.",
                },
                {
                    "call_id": cid,
                    "ticker": ticker,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "turn_id": call_id + 1,
                    "speaker_role": "analyst",
                    "section": "q&a",
                    "text": f"Question on {ticker} demand trends and margin outlook.",
                },
                {
                    "call_id": cid,
                    "ticker": ticker,
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "turn_id": call_id + 2,
                    "speaker_role": "management",
                    "section": "q&a",
                    "text": f"Response discussing execution risks and growth drivers for {ticker}.",
                },
            ]
        )
        call_id += 3

    date_index = pd.date_range("2020-12-01", "2022-10-31", freq="B")
    rng = np.random.default_rng(42)

    market_returns = pd.DataFrame(
        {"date": date_index, "market_return": rng.normal(0.0002, 0.01, len(date_index))}
    )

    price_rows = []
    for ticker in tickers:
        ticker_alpha = rng.normal(0.0001, 0.002)
        rets = market_returns["market_return"].values + ticker_alpha + rng.normal(0.0, 0.01, len(date_index))
        price_rows.append(pd.DataFrame({"date": date_index, "ticker": ticker, "return": rets}))
    prices = pd.concat(price_rows, ignore_index=True)

    ensure_dir(transcripts_path.parent)
    ensure_dir(prices_path.parent)
    ensure_dir(market_path.parent)
    ensure_dir(metadata_path.parent)

    pd.DataFrame(transcript_rows).to_csv(transcripts_path, index=False)
    prices.to_csv(prices_path, index=False)
    market_returns.to_csv(market_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    LOGGER.info("Generated synthetic raw data because source files were missing.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/mag7.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(".")

    _ = cfg.get("universe", [])

    ensure_dir(root / "data/raw/transcripts")
    ensure_dir(root / "data/raw/prices")
    ensure_dir(root / "data/raw/metadata")
    ensure_dir(root / "data/external/market_index")
    _maybe_write_sample_data(root)


if __name__ == "__main__":
    main()
