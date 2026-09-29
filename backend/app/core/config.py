"""Application settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProviderName = Literal["fake", "openai", "anthropic"]
EmbeddingProviderName = Literal["fake", "openai"]
RerankerName = Literal["none", "fake", "cross-encoder", "llm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------------
    app_name: str = "DocuChat"
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- Database ----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://docuchat:docuchat@localhost:5432/docuchat"
    db_echo: bool = False

    # --- Storage & ingestion -------------------------------------------------
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = Field(default=25, ge=1)
    chunk_size_tokens: int = Field(default=400, ge=50)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    embedding_batch_size: int = Field(default=64, ge=1)

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = Field(default=6, ge=1, le=50)
    retrieval_candidates: int = Field(default=30, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    history_turns: int = Field(default=6, ge=0)

    # --- Re-ranking (applied to the fused candidates) ------------------------
    reranker: RerankerName = "none"
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_cache_dir: Path | None = None

    # --- Providers ---------------------------------------------------------
    llm_provider: LLMProviderName = "fake"
    embedding_provider: EmbeddingProviderName = "fake"
    embedding_dim: int = Field(default=1536, ge=8, le=2000)

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = Field(default=16000, ge=256)
    anthropic_effort: Literal["low", "medium", "high"] = "medium"

    fake_stream_delay_ms: int = Field(default=12, ge=0)

    # --- Answer-quality judge (scripts/eval_answers.py) -----------------------
    judge_provider: LLMProviderName = "fake"
    judge_model: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "openai_base_url",
        "rerank_cache_dir",
        "judge_model",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        # docker compose passes unset variables as empty strings.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _check_provider_credentials(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        providers = {self.llm_provider, self.embedding_provider, self.judge_provider}
        if "openai" in providers and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when an OpenAI provider is selected")
        if "anthropic" in providers and self.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is required when an Anthropic provider is selected")
        if self.reranker == "llm" and self.llm_provider == "fake":
            raise ValueError("RERANKER=llm needs a real LLM_PROVIDER (openai or anthropic)")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
