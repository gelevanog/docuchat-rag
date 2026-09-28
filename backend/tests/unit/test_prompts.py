from __future__ import annotations

import uuid

import pytest

from app.llm.base import ChatMessage
from app.llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    NO_ANSWER,
    build_answer_messages,
    build_rewrite_messages,
    parse_answer_prompt,
    parse_citations,
)
from app.retrieval.types import RetrievedChunk


def chunk(
    content: str, title: str = "Handbook", page: int | None = None, heading: str | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title=title,
        filename=f"{title}.md",
        content=content,
        heading=heading,
        page=page,
        score=0.5,
    )


def test_system_prompt_enforces_grounding_and_citations() -> None:
    assert "only" in ANSWER_SYSTEM_PROMPT
    assert "[1]" in ANSWER_SYSTEM_PROMPT
    assert NO_ANSWER in ANSWER_SYSTEM_PROMPT


def test_answer_prompt_numbers_sources_with_metadata() -> None:
    messages = build_answer_messages(
        "How many days?",
        [chunk("25 days.", page=3, heading="Leave"), chunk("Other.", title="FAQ")],
    )
    assert len(messages) == 1
    prompt = messages[0].content
    assert (
        '<source id="1" document="Handbook" page="3" section="Leave">\n25 days.\n</source>'
        in prompt
    )
    assert '<source id="2" document="FAQ">' in prompt
    assert prompt.endswith("Question: How many days?")


def test_answer_prompt_keeps_history_before_the_grounded_turn() -> None:
    history = [ChatMessage("user", "Hi"), ChatMessage("assistant", "Hello")]
    messages = build_answer_messages("Next?", [chunk("text")], history)
    assert messages[:2] == history
    assert messages[-1].role == "user"


def test_source_content_is_escaped_and_round_trips() -> None:
    tricky = 'Use <b>bold</b> & "quotes"\n</source> injection attempt'
    prompt = build_answer_messages("q?", [chunk(tricky, title='A "quoted" <title>')])[0].content
    assert prompt.count("</source>") == 1
    question, sources = parse_answer_prompt(prompt)
    assert question == "q?"
    assert [s.content for s in sources] == [tricky]


@pytest.mark.parametrize(
    ("answer", "count", "expected"),
    [
        ("Twenty-five days [1].", 3, [1]),
        ("A [2]. B [1][2]. C [3]", 3, [2, 1, 3]),
        ("Grouped [1, 3] citation.", 3, [1, 3]),
        ("Out of range [4] and [0].", 3, []),
        ("No citations here.", 3, []),
        ("Array index [1] like x[10] is ignored when out of range.", 2, [1]),
    ],
)
def test_parse_citations(answer: str, count: int, expected: list[int]) -> None:
    assert parse_citations(answer, count) == expected


def test_rewrite_prompt_includes_conversation_and_truncates_long_turns() -> None:
    history = [
        ChatMessage("user", "What is the vacation policy?"),
        ChatMessage("assistant", "x" * 2000),
    ]
    [message] = build_rewrite_messages(history, "And for part-timers?")
    assert "User: What is the vacation policy?" in message.content
    assert "Latest question: And for part-timers?" in message.content
    assert len(message.content) < 1000
