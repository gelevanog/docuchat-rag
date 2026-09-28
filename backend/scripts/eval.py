"""Retrieval evaluation: hit-rate@k and MRR over a golden question set.

Usage (from backend/, with the database running and migrated):

    uv run python scripts/eval.py
    uv run python scripts/eval.py --dataset ../sample_data/eval/golden.yaml --k 1 3 5 --json

The script ingests every document in --docs (deduplicated by content hash, so re-running
is cheap), then runs each golden question through vector-only, keyword-only and hybrid
(RRF) retrieval with the configured embedding provider. Two relevance levels are scored:

* document - any chunk from `expected_document` counts as a hit;
* chunk    - the chunk must also contain `expected_text` (the passage that answers it).

Retrieval is restricted to the evaluated documents so other uploads do not skew results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import select

from app.container import Container
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import Chunk, Document, DocumentStatus
from app.ingestion.parsers import SUPPORTED_EXTENSIONS
from app.retrieval.fusion import reciprocal_rank_fusion

REPO_ROOT = Path(__file__).resolve().parents[2]
MODES = ("vector", "keyword", "hybrid")
LEVELS = ("document", "chunk")


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    question: str
    expected_document: str
    expected_text: str


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    filename: str
    content: str


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


def read_documents(docs_dir: Path) -> list[tuple[str, bytes]]:
    return [
        (path.name, path.read_bytes())
        for path in sorted(docs_dir.iterdir())
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


async def ingest_directory(container: Container, docs_dir: Path) -> dict[str, uuid.UUID]:
    """Upload and ingest every supported file; returns filename -> document id."""
    ids: dict[str, uuid.UUID] = {}
    for name, data in await asyncio.to_thread(read_documents, docs_dir):
        async with container.sessionmaker() as session:
            outcome = await container.documents.upload(session, name, data)
        if outcome.needs_ingestion or outcome.document.status != DocumentStatus.READY:
            await container.pipeline.ingest(outcome.document.id)
        ids[outcome.document.filename] = outcome.document.id
    return ids


async def run(
    container: Container, questions: list[GoldenQuestion], doc_ids: list[uuid.UUID], depth: int
) -> dict[str, dict[str, list[int | None]]]:
    """ranks[mode][level] -> one rank per question."""
    ranks: dict[str, dict[str, list[int | None]]] = {m: {lvl: [] for lvl in LEVELS} for m in MODES}
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
            fused = [
                cid
                for cid, _ in reciprocal_rank_fusion([vector, keyword], k=container.settings.rrf_k)
            ]
            ranked = {"vector": vector[:depth], "keyword": keyword[:depth], "hybrid": fused[:depth]}
            for mode in MODES:
                for level in LEVELS:
                    ranks[mode][level].append(first_hit(ranked[mode], chunks, q, level))
    return ranks


def summarize(
    ranks: dict[str, dict[str, list[int | None]]], ks: list[int]
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for mode in MODES:
        for level in LEVELS:
            values = ranks[mode][level]
            row = {f"hit@{k}": hit_rate_at_k(values, k) for k in ks}
            row["mrr"] = mean_reciprocal_rank(values)
            summary[f"{mode}/{level}"] = row
    return summary


def print_report(
    questions: list[GoldenQuestion],
    ranks: dict[str, dict[str, list[int | None]]],
    ks: list[int],
    embedder: str,
) -> None:
    def fmt(rank: int | None) -> str:
        return str(rank) if rank is not None else "-"

    width = max(len(q.question) for q in questions)
    print(f"\nEmbedding provider: {embedder} | questions: {len(questions)} | depth: {max(ks)}")
    print("\nPer-question rank of the first relevant chunk (hybrid retrieval)\n")
    print(f"{'question':<{width}}  {'doc':>4} {'chunk':>6}")
    for i, q in enumerate(questions):
        doc_rank, chunk_rank = ranks["hybrid"]["document"][i], ranks["hybrid"]["chunk"][i]
        print(f"{q.question:<{width}}  {fmt(doc_rank):>4} {fmt(chunk_rank):>6}")

    summary = summarize(ranks, ks)
    columns = [f"hit@{k}" for k in ks] + ["mrr"]
    print(f"\n{'mode/level':<18}" + "".join(f"{c:>9}" for c in columns))
    for name, row in summary.items():
        print(f"{name:<18}" + "".join(f"{row[c]:>9.3f}" for c in columns))


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "sample_data/eval/golden.yaml")
    parser.add_argument("--docs", type=Path, default=REPO_ROOT / "sample_data")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_format)
    raw = yaml.safe_load(args.dataset.read_text())["questions"]
    questions = [GoldenQuestion(**item) for item in raw]
    ks = sorted(set(args.k))

    container = Container.build(settings)
    try:
        doc_ids = await ingest_directory(container, args.docs)
        missing = {q.expected_document for q in questions} - doc_ids.keys()
        if missing:
            print(
                f"Expected documents not found in {args.docs}: {sorted(missing)}", file=sys.stderr
            )
            return 1
        ranks = await run(container, questions, list(doc_ids.values()), depth=max(ks))
    finally:
        await container.aclose()

    if args.json:
        print(json.dumps({"summary": summarize(ranks, ks), "ranks": ranks}, indent=2))
    else:
        print_report(questions, ranks, ks, container.embedder.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
