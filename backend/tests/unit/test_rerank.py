from __future__ import annotations

import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import replace

import pytest

from app.core.config import Settings
from app.llm.base import ChatMessage, LLMError, SchemaT
from app.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    LLMReranker,
    PassageGrade,
    PassageGrades,
    build_reranker,
)
from app.retrieval.types import RetrievedChunk
from app.services.chat import to_sources


def _chunk(content: str, heading: str | None = None, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Handbook",
        filename="handbook.md",
        content=content,
        heading=heading,
        page=None,
        score=score,
    )


# Fused order: the answering chunk is last, as happens when both retrievers rank it low.
CANDIDATES = [
    _chunk("Hotel costs are reimbursed up to 150 EUR per night.", "Expenses", 0.03),
    _chunk("Laptops are replaced every three years.", "Equipment", 0.02),
    _chunk("Full-time employees receive 25 days of paid vacation.", "Vacation", 0.01),
]


def _contents(chunks: Sequence[RetrievedChunk]) -> list[str]:
    return [chunk.content for chunk in chunks]


# --- lexical (fake) ----------------------------------------------------------------


async def test_lexical_reranker_promotes_the_chunk_covering_the_query() -> None:
    ranked = await LexicalReranker().rerank("days of paid vacation", CANDIDATES, top_k=3)
    assert ranked[0].content == CANDIDATES[2].content
    assert ranked[0].rerank_score == pytest.approx(1.0)
    scores = [chunk.rerank_score for chunk in ranked]
    assert all(score is not None and 0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)


async def test_lexical_reranker_reads_the_section_heading() -> None:
    ranked = await LexicalReranker().rerank("equipment", CANDIDATES, top_k=1)
    assert _contents(ranked) == [CANDIDATES[1].content]


async def test_rerankers_keep_fused_order_for_ties_and_truncate_to_top_k() -> None:
    ranked = await LexicalReranker().rerank("parking permits", CANDIDATES, top_k=2)
    assert _contents(ranked) == _contents(CANDIDATES[:2])
    assert [chunk.rerank_score for chunk in ranked] == [0.0, 0.0]
    assert [chunk.score for chunk in ranked] == [0.03, 0.02]  # fusion scores are kept


def test_lexical_score_rewards_phrases_over_scattered_terms() -> None:
    phrase = LexicalReranker.score("paid vacation", "Employees get paid vacation.")
    scattered = LexicalReranker.score("paid vacation", "Vacation is paid monthly.")
    assert phrase == pytest.approx(1.0)
    assert scattered == pytest.approx(0.75)
    assert LexicalReranker.score("the of and", "anything") == 0.0


# --- cross-encoder ------------------------------------------------------------------


class StubEncoder:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        self.calls.append((query, list(documents), batch_size))
        return self.logits


async def test_cross_encoder_orders_by_logit_and_normalises_scores() -> None:
    encoder = StubEncoder([-2.0, 0.0, 3.0])
    reranker = CrossEncoderReranker(encoder, "stub-model", batch_size=8)
    ranked = await reranker.rerank("vacation days", CANDIDATES, top_k=2)

    assert _contents(ranked) == [CANDIDATES[2].content, CANDIDATES[1].content]
    assert ranked[0].rerank_score == pytest.approx(0.952574, abs=1e-6)  # sigmoid(3)
    assert ranked[1].rerank_score == pytest.approx(0.5)  # sigmoid(0)
    [(query, passages, batch_size)] = encoder.calls
    assert query == "vacation days"
    assert passages[2] == "Vacation\nFull-time employees receive 25 days of paid vacation."
    assert batch_size == 8
    assert reranker.name == "cross-encoder:stub-model"


async def test_cross_encoder_skips_inference_without_candidates() -> None:
    encoder = StubEncoder([])
    assert await CrossEncoderReranker(encoder, "stub").rerank("q", [], top_k=3) == []
    assert encoder.calls == []


def test_cross_encoder_explains_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)  # import fails
    with pytest.raises(RuntimeError, match="--extra rerank"):
        CrossEncoderReranker.load("Xenova/ms-marco-MiniLM-L-6-v2")


# --- LLM ----------------------------------------------------------------------------


class StubStructuredModel:
    def __init__(self, reply: PassageGrades | None = None) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return "stub"

    async def parse(
        self, system: str, messages: Sequence[ChatMessage], schema: type[SchemaT]
    ) -> SchemaT:
        self.prompts.append(messages[-1].content)
        if self.reply is None:
            raise LLMError("boom")
        return schema.model_validate(self.reply.model_dump())


def _grades(*pairs: tuple[int, int]) -> PassageGrades:
    return PassageGrades(grades=[PassageGrade(id=i, relevance=r) for i, r in pairs])


async def test_llm_reranker_orders_by_grade_and_sends_numbered_sources() -> None:
    model = StubStructuredModel(_grades((1, 1), (2, 0), (3, 3)))
    ranked = await LLMReranker(model).rerank("How much vacation?", CANDIDATES, top_k=2)

    assert _contents(ranked) == [CANDIDATES[2].content, CANDIDATES[0].content]
    assert [chunk.rerank_score for chunk in ranked] == [1.0, pytest.approx(1 / 3, abs=1e-6)]
    [prompt] = model.prompts
    assert '<source id="3" document="Handbook" section="Vacation">' in prompt
    assert prompt.endswith("Query: How much vacation?")


async def test_llm_reranker_treats_missing_grades_as_irrelevant_and_first_grade_wins() -> None:
    model = StubStructuredModel(_grades((2, 2), (2, 0), (99, 3)))
    ranked = await LLMReranker(model).rerank("q", CANDIDATES, top_k=3)
    assert _contents(ranked) == _contents([CANDIDATES[1], CANDIDATES[0], CANDIDATES[2]])


async def test_llm_reranker_falls_back_to_fused_order_on_errors() -> None:
    ranked = await LLMReranker(StubStructuredModel(reply=None)).rerank("q", CANDIDATES, top_k=2)
    assert _contents(ranked) == _contents(CANDIDATES[:2])
    assert all(chunk.rerank_score is None for chunk in ranked)


def test_passage_grades_are_validated() -> None:
    with pytest.raises(ValueError, match="less than or equal to 3"):
        PassageGrade(id=1, relevance=4)


# --- configuration and API surface --------------------------------------------------


def test_build_reranker_from_settings() -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="sk-ant-test")
    assert build_reranker("none", settings) is None
    assert isinstance(build_reranker("fake", settings), LexicalReranker)
    llm = build_reranker("llm", settings)
    assert isinstance(llm, LLMReranker)
    assert llm.name == "llm:anthropic:claude-sonnet-5"


def test_sources_expose_the_rerank_score() -> None:
    [source] = to_sources([replace(CANDIDATES[0], rerank_score=0.9123454)])
    assert source.rerank_score == 0.912345
    [plain] = to_sources(CANDIDATES[:1])
    assert plain.rerank_score is None
