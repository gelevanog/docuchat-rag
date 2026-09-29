"""OpenAI chat + embeddings. Also serves OpenAI-compatible APIs: OpenRouter (with model
fallbacks and request pacing) and self-hosted servers via OPENAI_BASE_URL."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pydantic import ValidationError

from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMError, SchemaT
from app.llm.pacing import RequestPacer

logger = get_logger(__name__)

T = TypeVar("T")


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


class _OpenAICompatible:
    def __init__(
        self, client: AsyncOpenAI, model: str, label: str, pacer: RequestPacer | None
    ) -> None:
        self._client = client
        self._model = model
        self._label = label
        self._pacer = pacer

    @property
    def name(self) -> str:
        return f"{self._label}:{self._model}"

    async def _send(
        self,
        request: Callable[[], Awaitable[T]],
        retry_if: Callable[[Exception], bool] | None = None,
    ) -> T:
        return await (self._pacer.run(request, retry_if) if self._pacer else request())


def _is_stream_error_event(exc: Exception) -> bool:
    """An `error` event inside a 200 stream (OpenRouter: "Upstream error ... overloaded").
    The SDK raises it as a bare APIError, unlike HTTP errors, which have a status code."""
    return type(exc) is openai.APIError


def _content(chunk: ChatCompletionChunk) -> str | None:
    return chunk.choices[0].delta.content if chunk.choices else None


class OpenAIChatModel(_OpenAICompatible):
    """`fallback_models` is sent as OpenRouter's `models` list: if the primary model is
    unavailable or rate-limited upstream, OpenRouter answers with the next one."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        label: str = "openai",
        pacer: RequestPacer | None = None,
        fallback_models: Sequence[str] = (),
    ) -> None:
        super().__init__(client, model, label, pacer)
        self._fallbacks = [m for m in fallback_models if m != model]

    def _extra_body(self) -> dict[str, Any] | None:
        return {"models": [self._model, *self._fallbacks]} if self._fallbacks else None

    def _note_served(self, served: str | None) -> None:
        if self._fallbacks and served and served != self._model:
            logger.warning("model_fallback", requested=self._model, served=served)

    async def _open_stream(
        self, system: str, messages: Sequence[ChatMessage]
    ) -> tuple[str | None, AsyncIterator[ChatCompletionChunk]]:
        """Start a stream and read up to its first text delta. Upstream failures that arrive
        as stream events surface here, before anything was shown, so they can be retried."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(system, messages),
            stream=True,
            extra_body=self._extra_body(),
        )
        chunks = response.__aiter__()
        first = True
        async for chunk in chunks:
            if first:
                self._note_served(chunk.model)
                first = False
            if text := _content(chunk):
                return text, chunks
        return None, chunks

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            head, rest = await self._send(
                lambda: self._open_stream(system, messages), retry_if=_is_stream_error_event
            )
            if head:
                yield head
            async for chunk in rest:
                if text := _content(chunk):
                    yield text
        except openai.APIError as exc:
            raise LLMError(f"{self._label} chat request failed: {exc}") from exc

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await self._send(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=_to_openai_messages(system, messages),
                    extra_body=self._extra_body(),
                )
            )
        except openai.APIError as exc:
            raise LLMError(f"{self._label} chat request failed: {exc}") from exc
        self._note_served(response.model)
        return response.choices[0].message.content or ""

    async def parse(
        self, system: str, messages: Sequence[ChatMessage], schema: type[SchemaT]
    ) -> SchemaT:
        try:
            response = await self._send(
                lambda: self._client.chat.completions.parse(
                    model=self._model,
                    messages=_to_openai_messages(system, messages),
                    response_format=schema,
                    extra_body=self._extra_body(),
                )
            )
        except (
            openai.APIError,
            openai.LengthFinishReasonError,
            openai.ContentFilterFinishReasonError,
        ) as exc:
            raise LLMError(f"{self._label} structured request failed: {exc}") from exc
        except ValidationError as exc:
            raise LLMError(f"{self._label} reply does not match {schema.__name__}: {exc}") from exc
        self._note_served(response.model)
        message = response.choices[0].message
        if message.parsed is None:
            refusal = message.refusal or "empty"
            raise LLMError(f"{self._label} returned no structured output: {refusal}")
        return message.parsed


class OpenAIEmbeddingModel(_OpenAICompatible):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dim: int,
        label: str = "openai",
        pacer: RequestPacer | None = None,
    ) -> None:
        super().__init__(client, model, label, pacer)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._send(
                lambda: self._client.embeddings.create(
                    model=self._model,
                    input=list(texts),
                    # text-embedding-3-* models support shortening to the configured dimension.
                    dimensions=self._dim,
                )
            )
        except openai.APIError as exc:
            raise LLMError(f"{self._label} embedding request failed: {exc}") from exc
        ordered = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        if any(len(vector) != self._dim for vector in ordered):
            raise LLMError(
                f"{self.name} returned {len(ordered[0])}-dimensional vectors; "
                f"EMBEDDING_DIM is {self._dim}"
            )
        return ordered
