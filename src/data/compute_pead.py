from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def compute_pead(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
    event_idx: int,
    horizon: int,
    label_threshold: float = 0.0,
) -> Tuple[np.ndarray, float, int]:
    """Compute abnormal returns, cumulative abnormal return, and binary PEAD label."""
    stock_returns = np.asarray(stock_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    if stock_returns.shape != market_returns.shape:
        raise ValueError("stock_returns and market_returns must have matching shapes")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    start = event_idx + 1
    end = min(start + horizon, len(stock_returns))
    if start >= len(stock_returns):
        return np.array([], dtype=float), 0.0, 0

    abnormal = stock_returns[start:end] - market_returns[start:end]
    car = float(np.nansum(abnormal))
    label = int(car > label_threshold)
    return abnormal, car, label


def _compute_event_labels(
    metadata: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.DataFrame,
    horizon: int,
    event_lag_days: int,
    label_threshold: float,
) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date")

    out_rows = []
    for row in metadata.itertuples(index=False):
        event_date = pd.to_datetime(row.event_date)
        ticker_prices = prices[prices["ticker"] == row.ticker]
        merged = ticker_prices.merge(market, on="date", how="inner")
        merged = merged.sort_values("date").reset_index(drop=True)

        target_date = event_date + pd.Timedelta(days=event_lag_days)
        candidate = merged[merged["date"] >= target_date]
        if candidate.empty:
            abnormal = np.array([], dtype=float)
            car = 0.0
            label = 0
        else:
            event_idx = int(candidate.index[0]) - 1
            abnormal, car, label = compute_pead(
                stock_returns=merged["return"].to_numpy(),
                market_returns=merged["market_return"].to_numpy(),
                event_idx=event_idx,
                horizon=horizon,
                label_threshold=label_threshold,
            )

        out_rows.append(
            {
                "call_id": row.call_id,
                "ticker": row.ticker,
                "event_date": pd.to_datetime(row.event_date).strftime("%Y-%m-%d"),
                "car_horizon": car,
                "label": label,
                "abnormal_return_series": " ".join(f"{x:.6f}" for x in abnormal),
            }
        )

    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/mag7.yaml")
    parser.add_argument("--label-config", default="configs/data/pead_20d.yaml")
    parser.add_argument("--output", default="data/interim/labels/pead_labels.csv")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    label_cfg = load_yaml(args.label_config)

    metadata = pd.read_csv(data_cfg.get("metadata_source", "data/raw/metadata/call_metadata.csv"))
    prices = pd.read_csv(data_cfg.get("price_source", "data/raw/prices/daily_returns.csv"))
    market = pd.read_csv(data_cfg.get("market_source", "data/external/market_index/sp500_returns.csv"))

    labels = _compute_event_labels(
        metadata=metadata,
        prices=prices,
        market=market,
        horizon=int(label_cfg.get("pead_horizon", 20)),
        event_lag_days=int(label_cfg.get("event_lag_days", 1)),
        label_threshold=float(label_cfg.get("label_threshold", 0.0)),
    )

    ensure_dir("data/interim/labels")
    labels.to_csv(args.output, index=False)
    LOGGER.info("Saved PEAD labels to %s", args.output)


if __name__ == "__main__":
    main()
