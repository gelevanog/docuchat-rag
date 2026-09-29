"""Retrieval evaluation: hit-rate@k and MRR over a golden question set.

Usage (from backend/, with the database running and migrated):

    uv run python scripts/eval.py
    uv run python scripts/eval.py --dataset ../sample_data/eval/golden.yaml --k 1 3 5 --json
    uv run python scripts/eval.py --rerank fake cross-encoder   # compare re-rankers
    uv run python scripts/eval.py --rerank cross-encoder --tag paraphrase

The script ingests every document in --docs (deduplicated by content hash, so re-running
is cheap), then runs each golden question through vector-only, keyword-only and hybrid
(RRF) retrieval with the configured embedding provider. Each re-ranker named with --rerank
adds a `hybrid+<name>` mode: the top RERANK_CANDIDATES fused chunks are re-ranked, exactly
as in the chat endpoint. Two relevance levels are scored:

* document - any chunk from `expected_document` counts as a hit;
* chunk    - the chunk must also contain `expected_text` (the passage that answers it).

Retrieval is restricted to the evaluated documents so other uploads do not skew results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from sqlalchemy import select

from app.container import Container
from app.core.config import RerankerName, get_settings
from app.core.logging import configure_logging
from app.db.models import Chunk, Document
from app.evaluation.golden import GoldenQuestion, ingest_directory, load_questions
from app.retrieval.rerank import Reranker, build_reranker

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODES = ("vector", "keyword", "hybrid")
LEVELS = ("document", "chunk")
RERANKERS = tuple(name for name in get_args(RerankerName) if name != "none")

Ranks = dict[str, dict[str, list[int | None]]]


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    filename: str
    content: str


@dataclass(slots=True)
class EvalRun:
    ranks: Ranks
    """ranks[mode][level] -> rank of the first relevant chunk, one per question."""
    pool_hits: list[bool]
    """Whether the answering chunk was inside the re-ranking pool (a re-ranker's ceiling)."""
    rerank_ms: dict[str, list[float]]
    pool_size: int


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def first_hit(
    ranked: Sequence[uuid.UUID], chunks: dict[uuid.UUID, ChunkInfo], q: GoldenQuestion, level: str
) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved."""
    for rank, chunk_id in enumerate(ranked, start=1):
        info = chunks[chunk_id]
        if info.filename != q.expected_document:
            continue
        if level == "document" or _normalize(q.expected_text) in _normalize(info.content):
            return rank
    return None


def hit_rate_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


async def run(
    container: Container,
    questions: list[GoldenQuestion],
    doc_ids: list[uuid.UUID],
    depth: int,
    rerankers: dict[str, Reranker],
) -> EvalRun:
    modes = [*BASE_MODES, *(f"hybrid+{name}" for name in rerankers)]
    result = EvalRun(
        ranks={m: {lvl: [] for lvl in LEVELS} for m in modes},
        pool_hits=[],
        rerank_ms={name: [] for name in rerankers},
        pool_size=max(depth, container.settings.rerank_candidates),
    )
    retriever = container.retriever
    async with container.sessionmaker() as session:
        rows = await session.execute(
            select(Chunk.id, Document.filename, Chunk.content)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.document_id.in_(doc_ids))
        )
        chunks = {row.id: ChunkInfo(row.filename, row.content) for row in rows}

        for q in questions:
            [embedding] = await container.embedder.embed([q.question])
            vector = await retriever.vector_search(session, embedding, doc_ids)
            keyword = await retriever.keyword_search(session, q.question, doc_ids)
            fused = retriever.fuse(vector, keyword)
            ranked = {
                "vector": vector[:depth],
                "keyword": keyword[:depth],
                "hybrid": [cid for cid, _ in fused[:depth]],
            }
            pool = fused[: result.pool_size]
            result.pool_hits.append(first_hit([c for c, _ in pool], chunks, q, "chunk") is not None)
            if rerankers:
                candidates = await retriever.load_chunks(session, pool, vector, keyword)
                for name, reranker in rerankers.items():
                    started = time.perf_counter()
                    reranked = await reranker.rerank(q.question, candidates, depth)
                    result.rerank_ms[name].append((time.perf_counter() - started) * 1000)
                    ranked[f"hybrid+{name}"] = [chunk.chunk_id for chunk in reranked]
            for mode in modes:
                for level in LEVELS:
                    result.ranks[mode][level].append(first_hit(ranked[mode], chunks, q, level))
    return result


def summarize(ranks: Ranks, ks: list[int]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for mode, by_level in ranks.items():
        for level in LEVELS:
            values = by_level[level]
            row = {f"hit@{k}": hit_rate_at_k(values, k) for k in ks}
            row["mrr"] = mean_reciprocal_rank(values)
            summary[f"{mode}/{level}"] = row
    return summary


def print_report(
    questions: list[GoldenQuestion],
    result: EvalRun,
    ks: list[int],
    container: Container,
    rerankers: dict[str, Reranker],
) -> None:
    def fmt(rank: int | None) -> str:
        return str(rank) if rank is not None else "-"

    hybrid_modes = [mode for mode in result.ranks if mode.startswith("hybrid")]
    width = max(len(q.question) for q in questions)
    col = max(len(mode) for mode in hybrid_modes) + 2
    pool = result.pool_size
    print(
        f"\nEmbedding provider: {container.embedder.name} | questions: {len(questions)}"
        f" | depth: {max(ks)}"
    )
    for name, reranker in rerankers.items():
        print(f"Re-ranker {name}: {reranker.name} | pool: top {pool} fused candidates")
    print("\nPer-question rank of the first chunk containing the answer\n")
    print(f"{'question':<{width}}" + "".join(f"{mode:>{col}}" for mode in hybrid_modes))
    for i, q in enumerate(questions):
        cells = "".join(f"{fmt(result.ranks[mode]['chunk'][i]):>{col}}" for mode in hybrid_modes)
        print(f"{q.question:<{width}}{cells}")

    summary = summarize(result.ranks, ks)
    columns = [f"hit@{k}" for k in ks] + ["mrr"]
    name_width = max(len(name) for name in summary) + 2
    print(f"\n{'mode/level':<{name_width}}" + "".join(f"{c:>9}" for c in columns))
    for name, row in summary.items():
        print(f"{name:<{name_width}}" + "".join(f"{row[c]:>9.3f}" for c in columns))

    if rerankers:
        in_pool = sum(result.pool_hits)
        print(f"\nAnswering chunk inside the top-{pool} pool: {in_pool}/{len(questions)}")
        for name, timings in result.rerank_ms.items():
            print(f"Mean re-ranking latency, {name}: {sum(timings) / len(timings):.1f} ms/query")


async def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "sample_data/eval/golden.yaml")
    parser.add_argument("--docs", type=Path, default=REPO_ROOT / "sample_data")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--rerank",
        nargs="*",
        choices=RERANKERS,
        default=[settings.reranker] if settings.reranker != "none" else [],
        help="re-rankers to compare against plain hybrid retrieval (default: RERANKER)",
    )
    parser.add_argument("--tag", help="only evaluate questions with this tag")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()

    configure_logging("WARNING", settings.log_format)
    questions = load_questions(args.dataset, args.tag)
    if not questions:
        print(f"No questions in {args.dataset} match --tag {args.tag}", file=sys.stderr)
        return 1
    ks = sorted(set(args.k))

    container = Container.build(settings)
    try:
        rerankers = {name: build_reranker(name, settings) for name in dict.fromkeys(args.rerank)}
        active = {name: reranker for name, reranker in rerankers.items() if reranker is not None}
        doc_ids = await ingest_directory(container, args.docs)
        missing = {q.expected_document for q in questions} - doc_ids.keys()
        if missing:
            print(
                f"Expected documents not found in {args.docs}: {sorted(missing)}", file=sys.stderr
            )
            return 1
        result = await run(container, questions, list(doc_ids.values()), max(ks), active)
    finally:
        await container.aclose()

    if args.json:
        report = {
            "summary": summarize(result.ranks, ks),
            "ranks": result.ranks,
            "pool_hits": result.pool_hits,
            "rerank_ms": result.rerank_ms,
            "pool_size": result.pool_size,
        }
        print(json.dumps(report, indent=2))
    else:
        print_report(questions, result, ks, container, active)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
