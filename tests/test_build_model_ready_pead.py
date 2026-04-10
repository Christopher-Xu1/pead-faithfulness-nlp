from src.data.build_model_ready_pead import (
    parse_bose_structured_qna,
    parse_colon_qna_transcript,
    parse_reuters_qna_transcript,
)


def test_parse_colon_qna_transcript():
    transcript = """
    Executives: Tim Cook - CEO Luca Maestri - CFO
    Analysts: Katy Huberty - Morgan Stanley Mark Murphy - JPMorgan
    Operator: Welcome everyone.
    Tim Cook: We had a strong quarter and saw broad-based growth.
    Luca Maestri: Gross margin expanded.
    Operator: We will now begin the question-and-answer session.
    Katy Huberty: What drove services growth this quarter?
    Tim Cook: We saw strong attach and engagement.
    Mark Murphy: How should we think about guidance?
    Luca Maestri: We remain prudent on guidance.
    """

    turns = parse_colon_qna_transcript(transcript)

    assert len(turns) == 4
    assert turns[0]["speaker_role"] == "analyst"
    assert turns[1]["speaker_role"] == "management"
    assert turns[2]["speaker_role"] == "analyst"
    assert turns[3]["speaker_role"] == "management"


def test_parse_reuters_qna_transcript():
    transcript = """
    Presentation
    Questions and Answers
    --------------------------------------------------------------------------------
    Operator    [1]
    --------------------------------------------------------------------------------
    First question.
    --------------------------------------------------------------------------------
    Simona Jankowski, Goldman Sachs - Analyst    [2]
    --------------------------------------------------------------------------------
    What drove the margin change?
    --------------------------------------------------------------------------------
    Tim Cook, Apple Inc. - CEO    [3]
    --------------------------------------------------------------------------------
    Demand and mix improved.
    """

    turns = parse_reuters_qna_transcript(transcript)

    assert len(turns) == 3
    assert turns[1]["speaker_role"] == "analyst"
    assert turns[2]["speaker_role"] == "management"


def test_parse_bose_structured_qna():
    structured = [
        {"speaker": "Operator", "text": "Welcome to the call."},
        {"speaker": "Dave Fildes", "text": "Joining us today are Andy Jassy and Brian Olsavsky."},
        {"speaker": "Andy Jassy", "text": "We had a strong quarter."},
        {"speaker": "Operator", "text": "We will now begin the question and answer session."},
        {"speaker": "Mark Mahaney", "text": "How are you thinking about AWS growth?"},
        {"speaker": "Andy Jassy", "text": "We remain optimistic."},
    ]

    turns = parse_bose_structured_qna(structured)

    assert len(turns) == 2
    assert turns[0]["speaker_role"] == "analyst"
    assert turns[1]["speaker_role"] == "management"
