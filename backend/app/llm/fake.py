"""Deterministic, offline provider used for demos and tests (no API keys required).

* `HashingEmbeddingModel` - feature-hashed bag of unigrams + bigrams, L2-normalised. It is a
  real (if simple) lexical embedding, so vector search returns sensible neighbours.
* `FakeChatModel` - an extractive "answerer": for each retrieved source it quotes the sentence
  that best overlaps the question and cites it, or says it doesn't know.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator, Sequence
from itertools import pairwise

from app.core.text import content_terms, split_sentences
from app.llm.base import ChatMessage
from app.llm.prompts import NO_ANSWER, parse_answer_prompt

_MAX_QUOTED_SOURCES = 3


def _cite(sentence: str, source_id: int) -> str:
    """Place the citation marker before the final full stop: "... per year [2]."."""
    if sentence.endswith("."):
        return f"{sentence[:-1]} [{source_id}]."
    return f"{sentence} [{source_id}]"


class HashingEmbeddingModel:
    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fake:hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self._dim, (1.0 if (value >> 63) & 1 else -1.0)

    def embed_one(self, text: str) -> list[float]:
        terms = content_terms(text)
        features: list[tuple[str, float]] = [(term, 1.0) for term in terms]
        features += [(f"{a}_{b}", 0.5) for a, b in pairwise(terms)]
        if not features:
            features = [("<empty>", 1.0)]

        vector = [0.0] * self._dim
        for feature, weight in features:
            index, sign = self._bucket(feature)
            vector[index] += sign * weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:  # every feature cancelled out; fall back to a stable unit vector
            vector[self._bucket("<empty>")[0]] = 1.0
            return vector
        return [v / norm for v in vector]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class FakeChatModel:
    def __init__(self, stream_delay_ms: int = 0) -> None:
        self._delay = stream_delay_ms / 1000

    @property
    def name(self) -> str:
        return "fake:extractive"

    def answer(self, messages: Sequence[ChatMessage]) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        question, sources = parse_answer_prompt(last_user)
        question_terms = set(content_terms(question))
        if not question_terms:
            return NO_ANSWER

        scored: list[tuple[int, int, str]] = []  # (score, source id, best sentence)
        for source in sources:
            best_score, best_sentence = 0, ""
            for sentence in split_sentences(source.content):
                score = len(question_terms & set(content_terms(sentence)))
                if score > best_score:
                    best_score, best_sentence = score, sentence
            if best_score > 0:
                scored.append((best_score, source.id, best_sentence))
        if not scored:
            return NO_ANSWER

        # Quote only sentences that are about as relevant as the best one, in source order.
        threshold = max(score for score, _, _ in scored) / 2
        quotes = [(sid, text) for score, sid, text in scored if score >= threshold]
        bullets = "\n".join(f"- {_cite(text, sid)}" for sid, text in quotes[:_MAX_QUOTED_SOURCES])
        return f"Here is what the documents say:\n\n{bullets}"

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        text = self.answer(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else f"{word} "
            if self._delay:
                await asyncio.sleep(self._delay)

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        return self.answer(messages)
