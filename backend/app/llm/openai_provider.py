"""OpenAI chat + embeddings (also works with OpenAI-compatible servers via OPENAI_BASE_URL)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.llm.base import ChatMessage, LLMError


def _to_openai_messages(
    system: str, messages: Sequence[ChatMessage]
) -> list[ChatCompletionMessageParam]:
    result: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "user":
            result.append({"role": "user", "content": message.content})
        else:
            result.append({"role": "assistant", "content": message.content})
    return result


class OpenAIChatModel:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(system, messages),
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.APIError as exc:
            raise LLMError(f"OpenAI chat request failed: {exc}") from exc

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(system, messages),
            )
        except openai.APIError as exc:
            raise LLMError(f"OpenAI chat request failed: {exc}") from exc
        return response.choices[0].message.content or ""


class OpenAIEmbeddingModel:
    def __init__(self, client: AsyncOpenAI, model: str, dim: int) -> None:
        self._client = client
        self._model = model
        self._dim = dim

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
                # text-embedding-3-* models support shortening to the configured dimension.
                dimensions=self._dim,
            )
        except openai.APIError as exc:
            raise LLMError(f"OpenAI embedding request failed: {exc}") from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
