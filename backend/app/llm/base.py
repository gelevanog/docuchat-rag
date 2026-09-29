"""Provider-agnostic interfaces for chat and embedding models.

Every provider (OpenAI, Anthropic, the offline `fake` provider) implements these
protocols, so the rest of the application never imports a vendor SDK directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

Role = Literal["user", "assistant"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


class LLMError(RuntimeError):
    """Raised when a provider call fails in a way the caller should surface to the user."""


@runtime_checkable
class ChatModel(Protocol):
    """A chat-completion model."""

    @property
    def name(self) -> str: ...

    def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield the assistant's reply as text deltas."""
        ...

    async def complete(self, system: str, messages: Sequence[ChatMessage]) -> str:
        """Return the full assistant reply (used for short auxiliary tasks)."""
        ...


@runtime_checkable
class StructuredModel(Protocol):
    """A chat model that can return a reply validated against a Pydantic schema.

    Implemented by the OpenAI and Anthropic adapters; used for LLM re-ranking and the
    answer-quality judge. The offline `fake` provider has heuristic stand-ins instead.
    """

    @property
    def name(self) -> str: ...

    async def parse(
        self, system: str, messages: Sequence[ChatMessage], schema: type[SchemaT]
    ) -> SchemaT:
        """Return the reply as a validated `schema` instance; raise `LLMError` otherwise."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """A text embedding model producing fixed-size vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts. The output order matches the input order."""
        ...
