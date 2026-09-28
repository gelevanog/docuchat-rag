from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_run_without_api_keys() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "fake"
    assert settings.embedding_provider == "fake"


def test_cors_origins_accept_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    assert Settings(_env_file=None).cors_origins == ["http://a.test", "http://b.test"]


def test_real_providers_require_keys() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, embedding_provider="openai")
    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings(_env_file=None, llm_provider="anthropic")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(_env_file=None, chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_blank_optional_values_are_treated_as_unset() -> None:
    settings = Settings(_env_file=None, openai_api_key="", openai_base_url=" ")
    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, llm_provider="openai", openai_api_key="")
