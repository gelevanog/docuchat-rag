from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.llm.base import ChatMessage, LLMError
from app.services.query_rewriter import HeuristicQueryRewriter, LLMQueryRewriter

HISTORY = [
    ChatMessage("user", "What is the parental leave policy?"),
    ChatMessage("assistant", "Birthing parents get 20 weeks [1]."),
]


class StubChatModel:
    def __init__(self, reply: str | None = None, fail: bool = False) -> None:
        self.reply, self.fail, self.calls = reply, fail, 0

    @property
    def name(self) -> str:
        return "stub"

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        yield await self.complete(system, messages)

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        self.calls += 1
        if self.fail:
            raise LLMError("boom")
        return self.reply or ""


async def test_heuristic_passes_first_question_through() -> None:
    assert await HeuristicQueryRewriter().rewrite([], "What is it?") == "What is it?"


async def test_heuristic_anchors_referential_follow_ups() -> None:
    rewritten = await HeuristicQueryRewriter().rewrite(
        HISTORY, "And how long is it for adoptive parents?"
    )
    assert rewritten.endswith("(context: What is the parental leave policy?)")


async def test_heuristic_anchors_very_short_follow_ups() -> None:
    rewritten = await HeuristicQueryRewriter().rewrite(HISTORY, "Is it paid?")
    assert rewritten == "Is it paid? (context: What is the parental leave policy?)"


async def test_heuristic_keeps_standalone_questions() -> None:
    rewriter = HeuristicQueryRewriter()
    for question in (
        "What is the hotel limit for London business trips?",
        "Where is customer data stored and how is it protected?",
    ):
        assert await rewriter.rewrite(HISTORY, question) == question


async def test_llm_rewriter_uses_the_model_and_cleans_output() -> None:
    model = StubChatModel(reply='  "How long is parental leave for adoptive parents?"\n')
    rewritten = await LLMQueryRewriter(model).rewrite(HISTORY, "And for adoptive parents?")
    assert rewritten == "How long is parental leave for adoptive parents?"


async def test_llm_rewriter_skips_the_call_without_history() -> None:
    model = StubChatModel(reply="unused")
    assert await LLMQueryRewriter(model).rewrite([], "Standalone?") == "Standalone?"
    assert model.calls == 0


async def test_llm_rewriter_falls_back_on_errors() -> None:
    rewritten = await LLMQueryRewriter(StubChatModel(fail=True)).rewrite(HISTORY, "What about it?")
    assert "(context: What is the parental leave policy?)" in rewritten
