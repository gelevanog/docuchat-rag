"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Fuse several ranked lists into one.

    Each item scores ``sum(weight_i / (k + rank_i))`` over the lists it appears in
    (ranks are 1-based). RRF only looks at ranks, so it needs no score normalisation
    between systems whose scores live on different scales (cosine distance vs ts_rank).

    Ties are broken by first appearance, which keeps the output deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[K, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a list should not vote twice for the same item
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    # dict preserves insertion order and sorted() is stable -> deterministic tie-breaking.
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
