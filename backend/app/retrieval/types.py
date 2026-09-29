"""Value objects returned by retrieval."""

from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    filename: str
    content: str
    heading: str | None
    page: int | None
    score: float
    """Reciprocal-rank-fusion score (higher is better)."""
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None
    """Re-ranker relevance in [0, 1] (higher is better); None when no re-ranker ran."""
