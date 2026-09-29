from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI

from app.container import Container
from app.evaluation.judge import HeuristicAnswerJudge, JudgeCase, score_answer, summarize
from app.llm.prompts import NO_ANSWER


async def test_pipeline_answers_are_judged_without_being_persisted(
    app: FastAPI, client: httpx.AsyncClient, handbook: dict[str, Any], security: dict[str, Any]
) -> None:
    container: Container = app.state.container
    judge = HeuristicAnswerJudge()
    scores = []
    for question in ("How many vacation days do full-time employees get?", "Who owns the moon?"):
        async with container.sessionmaker() as session:
            chunks, answer = await container.chat.answer(session, question, top_k=3)
        assert chunks
        case = JudgeCase.build(question, chunks, answer)
        scores.append(score_answer(case, await judge.judge(case)))

    grounded, off_topic = scores
    assert "25 days of paid vacation" in grounded.answer
    assert (grounded.faithfulness, grounded.citation_precision) == (1.0, 1.0)
    assert off_topic.answer == NO_ANSWER
    assert off_topic.abstained
    assert summarize(scores).abstained == 1
    assert (await client.get("/api/conversations")).json() == []
