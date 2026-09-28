"""Token-aware recursive text splitting with overlap.

Text is split on the coarsest separator that yields pieces under the token budget
(paragraphs -> lines -> sentences -> clauses -> words -> raw tokens), and the pieces are
then greedily merged back into chunks of roughly `chunk_size` tokens. Consecutive chunks
share about `chunk_overlap` tokens so that facts spanning a boundary survive in at least
one chunk. Chunks never cross a block boundary, which keeps page/heading metadata exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import tiktoken

from app.ingestion.parsers import TextBlock

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


class TiktokenTokenizer:
    """`cl100k_base` is a good proxy for modern OpenAI/Anthropic tokenizers."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name

    @cached_property
    def _encoding(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding(self._encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        return self._encoding.decode(list(tokens))


@dataclass(frozen=True, slots=True)
class TextChunk:
    index: int
    text: str
    token_count: int
    page: int | None
    heading: str | None


class RecursiveTokenSplitter:
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split(text, self.separators)
        return [chunk for chunk in self._merge(pieces) if chunk]

    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        """Break `text` into pieces that each fit within `chunk_size` tokens."""
        if self.count(text) <= self.chunk_size:
            return [text]
        for i, separator in enumerate(separators):
            if separator in text:
                parts = text.split(separator)
                # Keep the separator attached to the preceding piece so no text is lost.
                pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
                result: list[str] = []
                for piece in pieces:
                    if not piece:
                        continue
                    if self.count(piece) <= self.chunk_size:
                        result.append(piece)
                    else:
                        result.extend(self._split(piece, separators[i + 1 :]))
                return result
        return self._split_by_tokens(text)

    def _split_by_tokens(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text)
        return [
            self.tokenizer.decode(tokens[start : start + self.chunk_size])
            for start in range(0, len(tokens), self.chunk_size)
        ]

    def _merge(self, pieces: Iterable[str]) -> list[str]:
        chunks: list[str] = []
        window: list[tuple[str, int]] = []
        window_tokens = 0
        for piece in pieces:
            size = self.count(piece)
            if window and window_tokens + size > self.chunk_size:
                chunks.append("".join(text for text, _ in window).strip())
                # Slide: keep a tail of at most `chunk_overlap` tokens that still leaves room.
                while window and (
                    window_tokens > self.chunk_overlap or window_tokens + size > self.chunk_size
                ):
                    window_tokens -= window.pop(0)[1]
            window.append((piece, size))
            window_tokens += size
        if window:  # always holds at least one piece not yet emitted
            chunks.append("".join(text for text, _ in window).strip())
        return chunks


def chunk_blocks(blocks: Sequence[TextBlock], splitter: RecursiveTokenSplitter) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for block in blocks:
        for text in splitter.split_text(block.text):
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=text,
                    token_count=splitter.count(text),
                    page=block.page,
                    heading=block.heading,
                )
            )
    return chunks
