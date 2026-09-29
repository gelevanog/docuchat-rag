"""An on-disk cache in front of an embedding model (EMBEDDING_CACHE_PATH).

Embeddings are deterministic for a given model and text, so re-ingesting the same documents
or re-running an evaluation doesn't need to spend quota again. Entries are keyed by model
name and a SHA-256 of the text and stored as one JSON file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from app.llm.base import EmbeddingModel


class CachingEmbeddingModel:
    def __init__(self, inner: EmbeddingModel, path: Path) -> None:
        self._inner = inner
        self._path = path
        self._vectors: dict[str, list[float]] = (
            json.loads(path.read_text()) if path.exists() else {}
        )

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._inner.name}\n{text}".encode()).hexdigest()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        missing = {
            key: text
            for key, text in zip(keys, texts, strict=True)
            if key not in self._vectors  # also de-duplicates repeated texts
        }
        if missing:
            vectors = await self._inner.embed(list(missing.values()))
            self._vectors.update(zip(missing, vectors, strict=True))
            await asyncio.to_thread(self._write, json.dumps(self._vectors))
        return [self._vectors[key] for key in keys]

    def _write(self, payload: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(payload)
        tmp.replace(self._path)  # atomic: an interrupted run never leaves a truncated cache
