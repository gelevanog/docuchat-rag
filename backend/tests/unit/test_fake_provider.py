from __future__ import annotations

import math
import uuid

from app.llm.fake import FakeChatModel, HashingEmbeddingModel
from app.llm.prompts import NO_ANSWER, build_answer_messages, parse_citations
from app.retrieval.types import RetrievedChunk


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    model = HashingEmbeddingModel(dim=128)
    first, second = await model.embed(["Vacation policy", "Vacation policy"])
    assert first == second
    assert len(first) == 128
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


async def test_hashing_embeddings_reflect_lexical_similarity() -> None:
    model = HashingEmbeddingModel(dim=256)
    query, related, unrelated = await model.embed(
        ["how many vacation days", "employees get 25 vacation days", "passwords need 14 characters"]
    )
    assert _cosine(query, related) > _cosine(query, unrelated)


async def test_empty_text_still_embeds_to_a_unit_vector() -> None:
    [vector] = await HashingEmbeddingModel(dim=32).embed(["the of and"])
    assert math.isclose(sum(v * v for v in vector), 1.0)


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Doc",
        filename="doc.md",
        content=content,
        heading=None,
        page=None,
        score=1.0,
    )


async def test_fake_chat_quotes_relevant_sentences_with_citations() -> None:
    messages = build_answer_messages(
        "How many vacation days do employees get?",
        [
            _chunk("Passwords must be long. Rotate keys yearly."),
            _chunk("Offices open at nine. Employees get 25 vacation days per year."),
        ],
    )
    answer = "".join([delta async for delta in FakeChatModel().stream("system", messages)])
    assert "Employees get 25 vacation days per year [2]." in answer
    assert parse_citations(answer, 2) == [2]


async def test_fake_chat_says_it_does_not_know_without_relevant_sources() -> None:
    messages = build_answer_messages(
        "What is the parking policy?", [_chunk("Invoices are due monthly.")]
    )
    assert await FakeChatModel().complete("system", messages) == NO_ANSWER


async def test_fake_chat_strips_markdown_from_quotes() -> None:
    messages = build_answer_messages(
        "Which plan includes single sign-on?",
        [_chunk("- **Business** plan: includes single sign-on and priority support.")],
    )
    answer = await FakeChatModel().complete("system", messages)
    assert "- Business plan: includes single sign-on and priority support [1]." in answer
