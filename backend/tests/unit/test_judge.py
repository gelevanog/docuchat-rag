from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.evaluation.judge import (
    AnswerVerdict,
    CitationVerdict,
    HeuristicAnswerJudge,
    JudgeCase,
    LLMAnswerJudge,
    Statement,
    StatementVerdict,
    build_judge,
    score_answer,
    split_statements,
    summarize,
)
from app.llm.base import ChatMessage, SchemaT
from app.llm.fake import FakeChatModel
from app.llm.prompts import NO_ANSWER, build_answer_messages
from app.retrieval.types import RetrievedChunk


def _chunk(content: str, heading: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_title="Handbook",
        filename="handbook.md",
        content=content,
        heading=heading,
        page=None,
        score=0.01,
    )


VACATION = _chunk("Full-time employees receive 25 days of paid vacation per year.", "Vacation")
HOTELS = _chunk("Hotel costs are reimbursed up to 150 EUR per night.", "Expenses")
SOURCES = [VACATION, HOTELS]


# --- statements -----------------------------------------------------------------------


def test_split_statements_attaches_citations_to_their_sentence() -> None:
    answer = (
        "Here is what the documents say:\n\n"
        "- Employees receive 25 days of vacation [1].\n"
        "- Hotels are reimbursed up to 150 EUR [2][1]. Meals are covered [1, 3]."
    )
    assert split_statements(answer) == [
        Statement(1, "Here is what the documents say:", ()),
        Statement(2, "Employees receive 25 days of vacation.", (1,)),
        Statement(3, "Hotels are reimbursed up to 150 EUR.", (2, 1)),
        Statement(4, "Meals are covered.", (1, 3)),
    ]


def test_split_statements_moves_markers_after_a_full_stop_back() -> None:
    answer = "Employees get 25 days. [1] 150 EUR per night applies to hotels. [2]"
    assert split_statements(answer) == [
        Statement(1, "Employees get 25 days.", (1,)),
        Statement(2, "150 EUR per night applies to hotels.", (2,)),
    ]


def test_abstentions_have_no_statements() -> None:
    case = JudgeCase.build("What is the parking policy?", SOURCES, NO_ANSWER)
    assert case.abstained
    assert case.statements == ()


# --- heuristic (fake) judge -------------------------------------------------------------


async def test_heuristic_judge_flags_unsupported_claims_and_wrong_citations() -> None:
    answer = (
        "Here is what I found:\n"
        "- Employees receive 25 days of paid vacation per year [1].\n"
        "- Employees receive 25 days of paid vacation per year [2].\n"
        "- The company car policy allows electric vehicles only [1]."
    )
    case = JudgeCase.build("How many vacation days do employees get?", SOURCES, answer)
    verdict = await HeuristicAnswerJudge().judge(case)
    score = score_answer(case, verdict)

    assert [s.is_claim for s in score.statements] == [False, True, True, True]
    assert [s.supported for s in score.statements[1:]] == [True, True, False]
    assert score.faithfulness == pytest.approx(2 / 3)
    assert score.citation_precision == pytest.approx(1 / 3)  # [2] and the car claim fail
    assert score.statements[2].supporting_citations == []
    assert score.relevance == 5  # the answer mentions every question term
    assert "2/3 claims" in score.explanation


async def test_heuristic_judge_is_deterministic() -> None:
    case = JudgeCase.build("Hotel limit?", SOURCES, "Hotels cost up to 150 EUR per night [2].")
    judge = HeuristicAnswerJudge()
    assert await judge.judge(case) == await judge.judge(case)


async def test_extractive_fake_answers_are_fully_grounded() -> None:
    question = "How many vacation days do full-time employees get?"
    answer = await FakeChatModel().complete("system", build_answer_messages(question, SOURCES))
    case = JudgeCase.build(question, SOURCES, answer)
    score = score_answer(case, await HeuristicAnswerJudge().judge(case))
    assert (score.faithfulness, score.citation_precision) == (1.0, 1.0)
    assert score.claims >= 1


# --- scoring and schema -------------------------------------------------------------------


def _case() -> JudgeCase:
    answer = "Employees get 25 days of vacation [1]. Hotels are capped at 150 EUR [2][7]."
    return JudgeCase.build("Vacation and hotels?", SOURCES, answer)


def test_score_counts_skipped_verdicts_as_failures_and_ignores_unknown_ids() -> None:
    verdict = AnswerVerdict(
        explanation="Only the first statement was judged.",
        statements=[
            StatementVerdict(statement_id=1, is_claim=True, supported=True),
            StatementVerdict(statement_id=1, is_claim=True, supported=False),  # duplicate
            StatementVerdict(statement_id=9, is_claim=True, supported=False),  # unknown id
        ],
        citations=[CitationVerdict(statement_id=1, source_id=1, supports=True)],
        relevance=4,
    )
    score = score_answer(_case(), verdict)
    assert score.faithfulness == 0.5  # statement 2 was skipped -> unsupported claim
    # [1] supports, [2] was not judged, [7] does not exist -> 1/3.
    assert score.citation_precision == pytest.approx(1 / 3)
    assert score.unjudged == 2  # statement 2 and its citation of source 2
    assert score.statements[1].cites == [2, 7]


def test_verdict_schema_rejects_invalid_judge_output() -> None:
    valid = '{"explanation": "ok", "statements": [], "citations": [], "relevance": 5}'
    assert AnswerVerdict.model_validate_json(valid).relevance == 5
    with pytest.raises(ValidationError, match="relevance"):
        AnswerVerdict.model_validate_json(valid.replace('"relevance": 5', '"relevance": 0'))
    with pytest.raises(ValidationError, match="statements"):
        AnswerVerdict.model_validate_json('{"explanation": "x", "citations": [], "relevance": 3}')


def test_summary_averages_only_answers_where_a_metric_applies() -> None:
    grounded = score_answer(
        JudgeCase.build("q", SOURCES, "Employees get 25 days [1]."),
        AnswerVerdict(
            explanation="",
            statements=[StatementVerdict(statement_id=1, is_claim=True, supported=True)],
            citations=[CitationVerdict(statement_id=1, source_id=1, supports=True)],
            relevance=5,
        ),
    )
    uncited = score_answer(
        JudgeCase.build("q", SOURCES, "Employees get 99 days."),
        AnswerVerdict(
            explanation="",
            statements=[StatementVerdict(statement_id=1, is_claim=True, supported=False)],
            citations=[],
            relevance=4,
        ),
    )
    abstained = score_answer(
        JudgeCase.build("q", SOURCES, NO_ANSWER),
        AnswerVerdict(explanation="", statements=[], citations=[], relevance=1),
    )
    assert (abstained.faithfulness, abstained.citation_precision) == (None, None)
    summary = summarize([grounded, uncited, abstained])
    assert summary.faithfulness == 0.5  # mean of 1.0 and 0.0; the abstention has no claims
    assert summary.citation_precision == 1.0  # only one answer cites anything
    assert summary.relevance == pytest.approx(10 / 3)
    assert (summary.answers, summary.abstained) == (3, 1)
    assert (summary.claims, summary.supported_claims) == (2, 1)
    assert summarize([]).faithfulness is None


# --- LLM judge ----------------------------------------------------------------------------


class StubStructuredModel:
    def __init__(self, reply: AnswerVerdict) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, type]] = []

    @property
    def name(self) -> str:
        return "stub"

    async def parse(
        self, system: str, messages: Sequence[ChatMessage], schema: type[SchemaT]
    ) -> SchemaT:
        self.calls.append((system, messages[-1].content, schema))
        return schema.model_validate(self.reply.model_dump())


async def test_llm_judge_sends_sources_and_numbered_statements() -> None:
    reply = AnswerVerdict(explanation="fine", statements=[], citations=[], relevance=5)
    model = StubStructuredModel(reply)
    judge = LLMAnswerJudge(model)
    assert await judge.judge(_case()) == reply

    [(system, prompt, schema)] = model.calls
    assert schema is AnswerVerdict
    assert "Do not use outside knowledge" in system
    assert "<question>Vacation and hotels?</question>" in prompt
    assert '<source id="2" document="Handbook" section="Expenses">' in prompt
    assert '<statement id="2" cites="2, 7">Hotels are capped at 150 EUR.</statement>' in prompt
    assert judge.name == "llm:stub"


def test_build_judge_by_provider() -> None:
    settings = Settings(_env_file=None, anthropic_api_key="sk-ant-test")
    assert isinstance(build_judge("fake", settings, None), HeuristicAnswerJudge)
    judge = build_judge("anthropic", settings, "claude-opus-5")
    assert judge.name == "llm:anthropic:claude-opus-5"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_judge("openai", settings, None)
