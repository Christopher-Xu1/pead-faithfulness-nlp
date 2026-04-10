import pandas as pd

from src.data.extract_qa import extract_qa


def test_extract_qa_analyst_only():
    df = pd.DataFrame(
        [
            {
                "call_id": "c1",
                "ticker": "AAPL",
                "event_date": "2024-01-01",
                "turn_id": 1,
                "speaker_role": "management",
                "section": "prepared remarks",
                "text": "intro",
            },
            {
                "call_id": "c1",
                "ticker": "AAPL",
                "event_date": "2024-01-01",
                "turn_id": 2,
                "speaker_role": "analyst",
                "section": "q&a",
                "text": "question one",
            },
            {
                "call_id": "c1",
                "ticker": "AAPL",
                "event_date": "2024-01-01",
                "turn_id": 3,
                "speaker_role": "management",
                "section": "q&a",
                "text": "answer one",
            },
        ]
    )

    out = extract_qa(df, analyst_only=True)

    assert len(out) == 1
    assert out.iloc[0]["num_questions"] == 1
    assert "question one" in out.iloc[0]["text"]
    assert "answer one" not in out.iloc[0]["text"]
