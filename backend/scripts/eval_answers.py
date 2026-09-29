"""Answer-quality evaluation: run the full RAG pipeline on the golden questions and let a
judge score every answer for faithfulness, citation precision and relevance.

Usage (from backend/, with the database running and migrated):

    uv run python scripts/eval_answers.py                      # judge: JUDGE_PROVIDER (fake)
    uv run python scripts/eval_answers.py --judge anthropic    # needs ANTHROPIC_API_KEY
    uv run python scripts/eval_answers.py --judge openai --judge-model gpt-5 --tag paraphrase

Answers come from the configured pipeline (LLM_PROVIDER, EMBEDDING_PROVIDER, RERANKER,
RETRIEVAL_TOP_K), exactly as the chat endpoint would produce them for a first question.
The judge sees the question, the numbered sources the model was given and the answer split
into cited statements, and returns a schema-validated verdict; the scores are computed from
it in code (see app/evaluation/judge.py). Prints a table and writes a JSON report.

The default `fake` judge is a token-overlap heuristic that exercises the pipeline offline.
Use a real judge (ideally a different, stronger model than the one answering) for results.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import get_args

from app.container import Container
from app.core.config import LLMProviderName, get_settings
from app.core.logging import configure_logging
from app.evaluation.golden import GoldenQuestion, ingest_directory, load_questions
from app.evaluation.judge import (
    AnswerJudge,
    AnswerScore,
    EvalSummary,
    JudgeCase,
    build_judge,
    score_answer,
    summarize,
)
from app.llm.base import LLMError

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION_WIDTH = 64
FLAG_RELEVANCE = 3
"""Answers at or below this relevance (or not fully faithful / precise) are explained."""


async def evaluate(
    container: Container,
    judge: AnswerJudge,
    questions: list[GoldenQuestion],
    document_ids: list[uuid.UUID],
) -> tuple[list[AnswerScore], list[str]]:
    """Returns the scores and the questions the judge failed on (reported, not scored)."""
    scores: list[AnswerScore] = []
    failures: list[str] = []
    for q in questions:
        async with container.sessionmaker() as session:
            chunks, answer = await container.chat.answer(session, q.question, document_ids)
        case = JudgeCase.build(q.question, chunks, answer)
        try:
            verdict = await judge.judge(case)
        except LLMError as exc:
            print(f"judge failed on {q.question!r}: {exc}", file=sys.stderr)
            failures.append(q.question)
            continue
        scores.append(score_answer(case, verdict))
    return scores, failures


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _short(text: str, width: int = QUESTION_WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def print_report(
    scores: list[AnswerScore], summary: EvalSummary, failures: list[str], header: str
) -> None:
    print(f"\n{header}\n")
    print(f"{'question':<{QUESTION_WIDTH}}{'faithful':>10}{'cite-prec':>11}{'relevance':>11}")
    for s in scores:
        print(
            f"{_short(s.question):<{QUESTION_WIDTH}}{_fmt(s.faithfulness):>10}"
            f"{_fmt(s.citation_precision):>11}{s.relevance:>11}"
        )

    print(f"\n{'faithfulness':<20}{_fmt(summary.faithfulness):>6}", end="")
    print(f"   ({summary.supported_claims}/{summary.claims} claims supported)")
    print(f"{'citation precision':<20}{_fmt(summary.citation_precision):>6}", end="")
    print(f"   ({summary.supporting_citations}/{summary.citations} citations support their claim)")
    print(f"{'relevance (1-5)':<20}{_fmt(summary.relevance):>6}")
    print(f"{'abstained':<20}{summary.abstained:>6}   of {summary.answers} answers")
    if summary.unjudged:
        print(f"{'unjudged items':<20}{summary.unjudged:>6}   (scored as failures)")
    if failures:
        print(f"{'judge errors':<20}{len(failures):>6}   (excluded from the scores)")

    flagged = [
        s
        for s in scores
        if s.abstained
        or s.relevance <= FLAG_RELEVANCE
        or (s.faithfulness is not None and s.faithfulness < 1)
        or (s.citation_precision is not None and s.citation_precision < 1)
    ]
    if flagged:
        print(f"\nFlagged answers ({len(flagged)}): judge explanations\n")
        for s in flagged:
            print(f"- {s.question}\n  {s.explanation}")


async def main() -> int:
    # The fake provider's streaming delay only exists for the demo UI.
    settings = get_settings().model_copy(update={"fake_stream_delay_ms": 0})
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "sample_data/eval/golden.yaml")
    parser.add_argument("--docs", type=Path, default=REPO_ROOT / "sample_data")
    parser.add_argument(
        "--judge", choices=get_args(LLMProviderName), default=settings.judge_provider
    )
    parser.add_argument("--judge-model", default=settings.judge_model)
    parser.add_argument("--tag", help="only evaluate questions with this tag")
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/answer-quality.json"), help="JSON report"
    )
    args = parser.parse_args()

    configure_logging("WARNING", settings.log_format)
    questions = load_questions(args.dataset, args.tag)
    if not questions:
        print(f"No questions in {args.dataset} match --tag {args.tag}", file=sys.stderr)
        return 1
    try:
        judge = build_judge(args.judge, settings, args.judge_model)
    except ValueError as exc:
        print(f"Cannot build the {args.judge} judge: {exc}", file=sys.stderr)
        return 1

    container = Container.build(settings)
    try:
        doc_ids = await ingest_directory(container, args.docs)
        scores, failures = await evaluate(container, judge, questions, list(doc_ids.values()))
    finally:
        await container.aclose()

    summary = summarize(scores)
    reranker = container.reranker.name if container.reranker else "none"
    header = (
        f"Judge: {judge.name} | answers: {container.chat_model.name} | "
        f"retrieval: {container.embedder.name}, reranker {reranker}, "
        f"top-{settings.retrieval_top_k} | questions: {len(questions)}"
    )
    print_report(scores, summary, failures, header)

    report = {
        "judge": judge.name,
        "chat_model": container.chat_model.name,
        "embedder": container.embedder.name,
        "reranker": reranker,
        "top_k": settings.retrieval_top_k,
        "summary": summary.model_dump(),
        "answers": [s.model_dump() for s in scores],
        "judge_errors": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nJSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
