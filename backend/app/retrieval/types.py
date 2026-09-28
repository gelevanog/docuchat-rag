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
