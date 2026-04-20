import pandas as pd

from src.data.fetch_hf_earnings_surprise import match_events_for_ticker


def test_match_hf_surprise_events_by_publication_date():
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
                "eps_surprise": 0.04,
                "estimated_eps_source": "hf_sovai_earnings_surprise",
            }
        ]
    )

    out = match_events_for_ticker(target, source, max_day_diff=3)

    assert out.loc[out["call_id"] == "a", "match_status"].iloc[0] == "matched"
    assert out.loc[out["call_id"] == "a", "eps_surprise"].iloc[0] == 0.04
    assert out.loc[out["call_id"] == "b", "match_status"].iloc[0] == "outside_tolerance"
    assert pd.isna(out.loc[out["call_id"] == "b", "eps_surprise"].iloc[0])
