import pandas as pd

from src.data.curate_transcripts import apply_deduplication, apply_quality_filters
from src.data.transcript_corpus import build_record_metrics, estimate_analyst_question_count


def test_estimate_analyst_question_count_from_structured_content():
    structured = [
        {"speaker": "Operator", "text": "We will now begin the question-and-answer session."},
        {"speaker": "Analyst", "text": "What drove the margin expansion this quarter?"},
        {"speaker": "CEO", "text": "Execution and mix improvements drove the result."},
        {"speaker": "Analyst", "text": "How should we think about guidance?"},
    ]

    count = estimate_analyst_question_count("Question-and-answer session", structured)

    assert count == 2


def test_build_record_metrics_detects_qa_and_hash():
    record = {
        "record_key": "s:file:0",
        "source_id": "glopardo_sp500_earnings_transcripts",
        "source_file": "file",
        "source_row": 0,
        "ticker": "aapl",
        "event_date": "2024-01-25 16:30:00",
        "company": "Apple Inc.",
        "year": 2024,
        "quarter": 1,
        "transcript": "Prepared remarks. Question-and-answer session. Analyst: What changed? CEO: Demand improved.",
        "structured_content": None,
    }

    metrics = build_record_metrics(record)

    assert metrics["ticker"] == "AAPL"
    assert metrics["event_date"] == "2024-01-25"
    assert metrics["has_qa_section"] == 1
    assert metrics["analyst_question_count"] >= 1
    assert metrics["text_hash"]


def test_deduplication_prefers_higher_quality_record():
    df = pd.DataFrame(
        [
            {
                "record_key": "a",
                "source_id": "lamini_earnings_calls_qa",
                "ticker": "AAPL",
                "event_date": "2024-01-25",
                "text_hash": "dup",
                "has_qa_section": 1,
                "analyst_question_count": 1,
                "has_structured_content": 0,
                "valid_ticker": 1,
                "valid_event_date": 1,
                "source_priority": 10,
                "transcript_chars": 9000,
                "repeated_operator_text": 0,
                "non_ascii_ratio": 0.0,
            },
            {
                "record_key": "b",
                "source_id": "glopardo_sp500_earnings_transcripts",
                "ticker": "AAPL",
                "event_date": "2024-01-25",
                "text_hash": "dup",
                "has_qa_section": 1,
                "analyst_question_count": 3,
                "has_structured_content": 1,
                "valid_ticker": 1,
                "valid_event_date": 1,
                "source_priority": 50,
                "transcript_chars": 12000,
                "repeated_operator_text": 0,
                "non_ascii_ratio": 0.0,
            },
        ]
    )

    out = apply_deduplication(df)

    keep = out[out["drop_reason"] == ""]
    drop = out[out["drop_reason"] != ""]
    assert keep.iloc[0]["record_key"] == "b"
    assert drop.iloc[0]["duplicate_of"] == "b"


def test_quality_filters_create_gold_subset():
    df = pd.DataFrame(
        [
            {
                "record_key": "keep",
                "drop_reason": "",
                "ticker": "AAPL",
                "event_date": "2024-01-25",
                "has_qa_section": 1,
                "analyst_question_count": 3,
                "has_structured_content": 1,
                "valid_ticker": 1,
                "valid_event_date": 1,
                "source_priority": 50,
                "transcript_chars": 9000,
                "transcript_words": 1500,
                "repeated_operator_text": 0,
                "non_ascii_ratio": 0.0,
                "quality_score": 200,
            },
            {
                "record_key": "drop",
                "drop_reason": "",
                "ticker": "AAPL",
                "event_date": "2024-01-25",
                "has_qa_section": 0,
                "analyst_question_count": 0,
                "has_structured_content": 0,
                "valid_ticker": 1,
                "valid_event_date": 1,
                "source_priority": 10,
                "transcript_chars": 1200,
                "transcript_words": 200,
                "repeated_operator_text": 1,
                "non_ascii_ratio": 0.0,
                "quality_score": 20,
            },
        ]
    )

    out = apply_quality_filters(
        df,
        {"hard_filters": {"min_transcript_chars": 5000, "min_analyst_questions": 2}},
    )

    assert bool(out.loc[out["record_key"] == "keep", "keep_gold"].iloc[0])
    assert not bool(out.loc[out["record_key"] == "drop", "keep_gold"].iloc[0])
