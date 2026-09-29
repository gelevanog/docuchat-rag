"""Build chat/embedding models from settings."""

from __future__ import annotations

from functools import lru_cache

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.core.config import LLMProviderName, Settings
from app.llm.anthropic_provider import AnthropicChatModel
from app.llm.base import ChatModel, EmbeddingModel, StructuredModel
from app.llm.cache import CachingEmbeddingModel
from app.llm.fake import FakeChatModel, HashingEmbeddingModel
from app.llm.openai_provider import OpenAIChatModel, OpenAIEmbeddingModel
from app.llm.pacing import RequestPacer


def _openai_client(settings: Settings) -> AsyncOpenAI:
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
    )


def _openrouter_client(settings: Settings) -> AsyncOpenAI:
    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is not set")
    # Optional app attribution headers (https://openrouter.ai/docs/app-attribution).
    headers = {
        header: value
        for header, value in (
            ("HTTP-Referer", settings.openrouter_app_url),
            ("X-Title", settings.openrouter_app_name),
        )
        if value
    }
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        default_headers=headers,
        max_retries=0,  # RequestPacer retries, spaced out to respect the rate limit
    )


@lru_cache
def _shared_pacer(key: str, min_interval: float, retries: int, backoff: float) -> RequestPacer:
    """One pacer per endpoint and policy: chat, judge and embeddings share one quota."""
    return RequestPacer(min_interval=min_interval, retries=retries, backoff=backoff)


def _openrouter_pacer(settings: Settings) -> RequestPacer:
    return _shared_pacer(
        settings.openrouter_base_url,
        settings.openrouter_min_interval_s,
        settings.openrouter_retries,
        settings.openrouter_backoff_s,
    )


def _anthropic_model(settings: Settings, model: str) -> AnthropicChatModel:
    if settings.anthropic_api_key is None:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return AnthropicChatModel(
        AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value()),
        model=model,
        max_tokens=settings.anthropic_max_tokens,
        effort=settings.anthropic_effort,
    )


def _openrouter_chat_model(settings: Settings, model: str | None) -> OpenAIChatModel:
    """Without an explicit `model`, the configured chat model with its fallbacks. An explicit
    model (e.g. a judge) gets none, so its results always come from the model it names."""
    return OpenAIChatModel(
        _openrouter_client(settings),
        model or settings.openrouter_chat_model,
        label="openrouter",
        pacer=_openrouter_pacer(settings),
        fallback_models=() if model else settings.openrouter_fallback_models,
    )


def build_chat_model(settings: Settings) -> ChatModel:
    if settings.llm_provider == "openai":
        return OpenAIChatModel(_openai_client(settings), settings.openai_chat_model)
    if settings.llm_provider == "anthropic":
        return _anthropic_model(settings, settings.anthropic_model)
    if settings.llm_provider == "openrouter":
        return _openrouter_chat_model(settings, None)
    return FakeChatModel(stream_delay_ms=settings.fake_stream_delay_ms)


def build_structured_model(
    settings: Settings, provider: LLMProviderName, model: str | None = None
) -> StructuredModel:
    """A model with schema-validated replies. `model` overrides the provider's chat model."""
    if provider == "openai":
        return OpenAIChatModel(_openai_client(settings), model or settings.openai_chat_model)
    if provider == "anthropic":
        return _anthropic_model(settings, model or settings.anthropic_model)
    if provider == "openrouter":
        return _openrouter_chat_model(settings, model)
    raise ValueError("The fake provider has no structured output; use its heuristic counterpart")


def build_embedding_model(settings: Settings) -> EmbeddingModel:
    embedder: EmbeddingModel
    if settings.embedding_provider == "openai":
        embedder = OpenAIEmbeddingModel(
            _openai_client(settings), settings.openai_embedding_model, settings.embedding_dim
        )
    elif settings.embedding_provider == "openrouter":
        embedder = OpenAIEmbeddingModel(
            _openrouter_client(settings),
            settings.openrouter_embedding_model,
            settings.embedding_dim,
            label="openrouter",
            pacer=_openrouter_pacer(settings),
        )
    else:
        embedder = HashingEmbeddingModel(settings.embedding_dim)
    if settings.embedding_cache_path is not None:
        return CachingEmbeddingModel(embedder, settings.embedding_cache_path)
    return embedder
