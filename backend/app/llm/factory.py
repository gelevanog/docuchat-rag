"""Build chat/embedding models from settings."""

from __future__ import annotations

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.core.config import Settings
from app.llm.anthropic_provider import AnthropicChatModel
from app.llm.base import ChatModel, EmbeddingModel
from app.llm.fake import FakeChatModel, HashingEmbeddingModel
from app.llm.openai_provider import OpenAIChatModel, OpenAIEmbeddingModel


def _openai_client(settings: Settings) -> AsyncOpenAI:
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
    )


def build_chat_model(settings: Settings) -> ChatModel:
    if settings.llm_provider == "openai":
        return OpenAIChatModel(_openai_client(settings), settings.openai_chat_model)

    if settings.llm_provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return AnthropicChatModel(
            AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value()),
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            effort=settings.anthropic_effort,
        )

    return FakeChatModel(stream_delay_ms=settings.fake_stream_delay_ms)


def build_embedding_model(settings: Settings) -> EmbeddingModel:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingModel(
            _openai_client(settings), settings.openai_embedding_model, settings.embedding_dim
        )
    return HashingEmbeddingModel(settings.embedding_dim)
