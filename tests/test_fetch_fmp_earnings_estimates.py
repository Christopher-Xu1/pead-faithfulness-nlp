import pandas as pd

from src.data.fetch_fmp_earnings_estimates import add_capex_proxy, match_events_for_ticker, normalize_fmp_earnings_payload


def test_normalize_fmp_earnings_payload_extracts_revenue_estimate():
    payload = [
        {
            "date": "2024-10-31",
            "epsActual": 1.64,
            "epsEstimated": 1.60,
            "revenueActual": 94930000000,
            "revenueEstimated": 94500000000,
        }
    ]

    out = normalize_fmp_earnings_payload(payload, ticker="AAPL")

    assert out.loc[0, "ticker"] == "AAPL"
    assert out.loc[0, "reported_eps"] == 1.64
    assert out.loc[0, "estimated_eps"] == 1.60
    assert out.loc[0, "reported_revenue"] == 94930000000
    assert out.loc[0, "estimated_revenue"] == 94500000000
    assert out.loc[0, "estimated_revenue_source"] == "fmp"


def test_match_events_for_ticker_respects_tolerance():
    target = pd.DataFrame(
        [
            {"call_id": "a", "ticker": "AAPL", "event_date": "2024-11-01"},
            {"call_id": "b", "ticker": "AAPL", "event_date": "2024-12-01"},
        ]
    )
    source = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "source_event_date": "2024-10-31",
                "reported_eps": 1.64,
                "estimated_eps": 1.60,
                "reported_revenue": 94930000000,
                "estimated_revenue": 94500000000,
                "estimated_eps_source": "fmp",
                "estimated_revenue_source": "fmp",
            }
        ]
    )

    out = match_events_for_ticker(target, source, max_day_diff=3)

    assert out.loc[out["call_id"] == "a", "match_status"].iloc[0] == "matched"
    assert out.loc[out["call_id"] == "b", "match_status"].iloc[0] == "outside_tolerance"
    assert pd.isna(out.loc[out["call_id"] == "b", "estimated_revenue"].iloc[0])


def test_add_capex_proxy_uses_prior_capex_intensity_only():
    estimates = pd.DataFrame(
        [
            {
                "call_id": "old",
                "ticker": "AAPL",
                "event_date": "2024-01-01",
                "reported_eps": 1.0,
                "estimated_eps": 1.0,
                "reported_revenue": pd.NA,
                "estimated_revenue": 1000.0,
            },
            {
                "call_id": "new",
                "ticker": "AAPL",
                "event_date": "2024-04-01",
                "reported_eps": 1.0,
                "estimated_eps": 1.0,
                "reported_revenue": pd.NA,
                "estimated_revenue": 1200.0,
            },
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "call_id": "old",
                "ticker": "AAPL",
                "event_date": "2024-01-01",
                "reported_revenue": 1000.0,
                "reported_capex": 100.0,
            },
            {
                "call_id": "new",
                "ticker": "AAPL",
                "event_date": "2024-04-01",
                "reported_revenue": 1200.0,
                "reported_capex": 150.0,
            },
        ]
    )

    out = add_capex_proxy(estimates, fundamentals, rolling_window=4)

    assert pd.isna(out.loc[out["call_id"] == "old", "estimated_capex"].iloc[0])
    assert out.loc[out["call_id"] == "new", "estimated_capex"].iloc[0] == 120.0
    assert out.loc[out["call_id"] == "new", "estimated_capex_is_proxy"].iloc[0] == 1
