"""Provider adapters, exercised against minimal stand-ins for the vendor SDK clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from app.core.config import Settings
from app.llm.anthropic_provider import REFUSAL_NOTICE, AnthropicChatModel
from app.llm.base import ChatMessage
from app.llm.factory import build_chat_model, build_embedding_model
from app.llm.fake import FakeChatModel, HashingEmbeddingModel
from app.llm.openai_provider import OpenAIChatModel, OpenAIEmbeddingModel

MESSAGES = [ChatMessage("user", "Hi"), ChatMessage("assistant", "Hello"), ChatMessage("user", "Q?")]


class _AnthropicStream:
    def __init__(self, texts: list[str], stop_reason: str) -> None:
        self._texts, self._stop_reason = texts, stop_reason

    async def __aenter__(self) -> _AnthropicStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:  # mirrors the SDK attribute
        for text in self._texts:
            yield text

    async def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(stop_reason=self._stop_reason)


class _AnthropicClient:
    def __init__(self, texts: list[str], stop_reason: str = "end_turn") -> None:
        self.kwargs: dict[str, Any] = {}

        def stream(**kwargs: Any) -> _AnthropicStream:
            self.kwargs = kwargs
            return _AnthropicStream(texts, stop_reason)

        self.messages = SimpleNamespace(stream=stream)


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


async def test_anthropic_streams_text_and_maps_messages() -> None:
    client = _AnthropicClient(["Twenty-five ", "days [1]."])
    model = AnthropicChatModel(client, "claude-sonnet-5", max_tokens=1000, effort="medium")  # type: ignore[arg-type]
    assert await _collect(model.stream("SYSTEM", MESSAGES)) == "Twenty-five days [1]."
    assert client.kwargs["system"] == "SYSTEM"
    assert client.kwargs["model"] == "claude-sonnet-5"
    assert client.kwargs["output_config"] == {"effort": "medium"}
    assert [m["role"] for m in client.kwargs["messages"]] == ["user", "assistant", "user"]
    assert "temperature" not in client.kwargs


async def test_anthropic_surfaces_refusals() -> None:
    client = _AnthropicClient([], stop_reason="refusal")
    model = AnthropicChatModel(client, "claude-sonnet-5", max_tokens=1000, effort="low")  # type: ignore[arg-type]
    assert await _collect(model.stream("SYSTEM", MESSAGES)) == REFUSAL_NOTICE


class _OpenAIClient:
    def __init__(self) -> None:
        self.chat_kwargs: dict[str, Any] = {}
        self.embed_kwargs: dict[str, Any] = {}

        async def create_chat(**kwargs: Any) -> AsyncIterator[SimpleNamespace]:
            self.chat_kwargs = kwargs

            async def chunks() -> AsyncIterator[SimpleNamespace]:
                for text in ["Hello", None, " world"]:
                    delta = SimpleNamespace(content=text)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

            return chunks()

        async def create_embeddings(**kwargs: Any) -> SimpleNamespace:
            self.embed_kwargs = kwargs
            data = [
                SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(kwargs["input"]))
            ]
            return SimpleNamespace(data=list(reversed(data)))

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create_chat))
        self.embeddings = SimpleNamespace(create=create_embeddings)


async def test_openai_chat_streams_deltas_with_system_message_first() -> None:
    client = _OpenAIClient()
    model = OpenAIChatModel(client, "gpt-test")  # type: ignore[arg-type]
    assert await _collect(model.stream("SYSTEM", MESSAGES)) == "Hello world"
    sent = client.chat_kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "SYSTEM"}
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"]
    assert client.chat_kwargs["stream"] is True


async def test_openai_embeddings_preserve_input_order() -> None:
    client = _OpenAIClient()
    model = OpenAIEmbeddingModel(client, "text-embedding-3-small", dim=8)  # type: ignore[arg-type]
    assert await model.embed(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]
    assert client.embed_kwargs["dimensions"] == 8
    assert await model.embed([]) == []


def test_factory_defaults_to_offline_providers() -> None:
    settings = Settings(_env_file=None, embedding_dim=64)
    assert isinstance(build_chat_model(settings), FakeChatModel)
    embedder = build_embedding_model(settings)
    assert isinstance(embedder, HashingEmbeddingModel)
    assert embedder.dim == 64


def test_factory_builds_real_providers_from_settings() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        embedding_provider="openai",
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-test",
    )
    chat = build_chat_model(settings)
    assert isinstance(chat, AnthropicChatModel)
    assert chat.name == "anthropic:claude-sonnet-5"
    assert isinstance(build_embedding_model(settings), OpenAIEmbeddingModel)
