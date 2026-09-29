"""Re-ranking of the fused retrieval candidates.

Vector and full-text search each score a chunk without looking at the query and the chunk
together, and RRF only combines their ranks. A re-ranker reads the query and every candidate
side by side and re-orders a small pool (RERANK_CANDIDATES) of the best fused results, of
which the top-k go to the model. Implementations:

* `LexicalReranker` (`fake`) - deterministic query-term coverage, for tests and offline demos.
* `CrossEncoderReranker` (`cross-encoder`) - a local ms-marco MiniLM cross-encoder on ONNX
  Runtime via fastembed (optional `rerank` extra, no PyTorch).
* `LLMReranker` (`llm`) - the configured chat model grades every candidate in one call.

Every re-ranker returns scores in [0, 1] and breaks ties by fused order.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from app.core.config import RerankerName, Settings
from app.core.logging import get_logger
from app.core.text import content_terms
from app.llm.base import LLMError, StructuredModel
from app.llm.factory import build_structured_model
from app.llm.prompts import RERANK_SYSTEM_PROMPT, build_rerank_messages
from app.retrieval.types import RetrievedChunk

logger = get_logger(__name__)

_MAX_GRADE = 3


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Return the `top_k` most relevant candidates, best first, with `rerank_score` set."""
        ...


def passage_text(chunk: RetrievedChunk) -> str:
    """What a re-ranker reads: the section heading gives short chunks their topic."""
    return f"{chunk.heading}\n{chunk.content}" if chunk.heading else chunk.content


def order_by_scores(
    candidates: Sequence[RetrievedChunk], scores: Sequence[float], top_k: int
) -> list[RetrievedChunk]:
    """Sort candidates by score, best first. The sort is stable, so ties keep fused order."""
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [replace(candidates[i], rerank_score=round(scores[i], 6)) for i in order[:top_k]]


class LexicalReranker:
    """Offline re-ranker: the share of query terms (and, with less weight, query bigrams)
    that appear in the passage. Deterministic, so tests can assert exact orderings."""

    @property
    def name(self) -> str:
        return "fake:lexical"

    @staticmethod
    def score(query: str, passage: str) -> float:
        query_terms = content_terms(query)
        if not query_terms:
            return 0.0
        passage_terms = content_terms(passage)
        unigrams = set(query_terms)
        coverage = len(unigrams & set(passage_terms)) / len(unigrams)
        bigrams = set(pairwise(query_terms))
        if not bigrams:
            return coverage
        phrase = len(bigrams & set(pairwise(passage_terms))) / len(bigrams)
        return 0.75 * coverage + 0.25 * phrase

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        scores = [self.score(query, passage_text(chunk)) for chunk in candidates]
        return order_by_scores(candidates, scores, top_k)


class CrossEncoder(Protocol):
    """The slice of fastembed's `TextCrossEncoder` used here (one logit per document)."""

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = ...
    ) -> Iterable[float]: ...


def _sigmoid(logit: float) -> float:
    return 0.5 * (1.0 + math.tanh(logit / 2.0))  # overflow-free logistic function


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a cross-encoder on CPU (ONNX Runtime)."""

    def __init__(self, encoder: CrossEncoder, model_name: str, batch_size: int = 32) -> None:
        self._encoder = encoder
        self._model_name = model_name
        self._batch_size = batch_size

    @classmethod
    def load(cls, model_name: str, cache_dir: Path | None = None) -> CrossEncoderReranker:
        """Load (downloading on first use) a fastembed cross-encoder."""
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415
        except ImportError as exc:  # the optional extra is not installed
            raise RuntimeError(
                "RERANKER=cross-encoder needs the optional `rerank` extra: run "
                "`uv sync --extra rerank`, or build the Docker image with BACKEND_EXTRAS=rerank"
            ) from exc
        encoder = TextCrossEncoder(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        return cls(encoder, model_name)

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def _scores(self, query: str, passages: list[str]) -> list[float]:
        logits = self._encoder.rerank(query, passages, batch_size=self._batch_size)
        return [_sigmoid(float(logit)) for logit in logits]

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        passages = [passage_text(chunk) for chunk in candidates]
        # Inference is CPU-bound; keep it off the event loop.
        scores = await asyncio.to_thread(self._scores, query, passages)
        return order_by_scores(candidates, scores, top_k)


class PassageGrade(BaseModel):
    id: int = Field(description="The source id")
    relevance: int = Field(ge=0, le=_MAX_GRADE, description="0 (unrelated) to 3 (answers it)")


class PassageGrades(BaseModel):
    grades: list[PassageGrade]


class LLMReranker:
    """Grades all candidates in a single structured call ("batched pointwise" re-ranking).

    Ungraded candidates count as irrelevant. If the call fails, the fused order is kept, so
    a flaky provider degrades retrieval quality instead of breaking the chat.
    """

    def __init__(self, model: StructuredModel) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"llm:{self._model.name}"

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        try:
            result = await self._model.parse(
                RERANK_SYSTEM_PROMPT, build_rerank_messages(query, candidates), PassageGrades
            )
        except LLMError as exc:
            logger.warning("llm_rerank_failed", error=str(exc))
            return list(candidates[:top_k])
        grades: dict[int, int] = {}
        for grade in result.grades:
            grades.setdefault(grade.id, grade.relevance)  # the first grade for an id wins
        scores = [grades.get(i, 0) / _MAX_GRADE for i in range(1, len(candidates) + 1)]
        return order_by_scores(candidates, scores, top_k)


def build_reranker(name: RerankerName, settings: Settings) -> Reranker | None:
    """`none` returns None: retrieval then keeps the fused top-k unchanged."""
    if name == "fake":
        return LexicalReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker.load(settings.rerank_model, settings.rerank_cache_dir)
    if name == "llm":
        return LLMReranker(build_structured_model(settings, settings.llm_provider))
    return None
