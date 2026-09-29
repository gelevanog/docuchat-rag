from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.llm.cache import CachingEmbeddingModel


class CountingEmbedder:
    def __init__(self, name: str = "stub:model") -> None:
        self._name = name
        self.batches: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return 2

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


async def test_only_unseen_texts_reach_the_model_and_order_is_kept(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cache = CachingEmbeddingModel(inner, tmp_path / "cache.json")
    assert await cache.embed(["a", "bb", "a"]) == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert await cache.embed(["bb", "ccc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert inner.batches == [["a", "bb"], ["ccc"]]
    assert (cache.name, cache.dim) == ("stub:model", 2)


async def test_cache_persists_across_runs_and_is_keyed_by_model(tmp_path: Path) -> None:
    path = tmp_path / "eval" / "cache.json"
    await CachingEmbeddingModel(CountingEmbedder(), path).embed(["vacation days"])

    same_model = CountingEmbedder()
    assert await CachingEmbeddingModel(same_model, path).embed(["vacation days"])
    assert same_model.batches == []

    other_model = CountingEmbedder("stub:other")
    await CachingEmbeddingModel(other_model, path).embed(["vacation days"])
    assert other_model.batches == [["vacation days"]]
