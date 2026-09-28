from __future__ import annotations

from itertools import pairwise

import pytest

from app.ingestion.chunking import RecursiveTokenSplitter, TiktokenTokenizer, chunk_blocks
from app.ingestion.parsers import TextBlock
from tests.helpers import CharTokenizer


def splitter(size: int, overlap: int) -> RecursiveTokenSplitter:
    return RecursiveTokenSplitter(CharTokenizer(), chunk_size=size, chunk_overlap=overlap)


def test_short_text_is_a_single_chunk() -> None:
    assert splitter(100, 10).split_text("  Hello world.  ") == ["Hello world."]


def test_empty_text_yields_no_chunks() -> None:
    assert splitter(100, 10).split_text(" \n\n ") == []


def test_chunks_respect_the_token_budget() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = splitter(80, 16).split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i:03d}" for i in range(200))
    chunks = splitter(60, 20).split_text(text)
    assert len(chunks) > 2
    for previous, current in pairwise(chunks):
        first_word = current.split()[0]
        assert first_word in previous.split(), "next chunk should start inside the previous one"
        shared = previous[previous.index(first_word) :]
        assert current.startswith(shared)
        assert 0 < len(shared) <= 20


def test_zero_overlap_loses_and_duplicates_nothing() -> None:
    words = [f"w{i:03d}" for i in range(150)]
    chunks = splitter(50, 0).split_text(" ".join(words))
    assert " ".join(chunks).split() == words


def test_prefers_paragraph_boundaries() -> None:
    para_a = "Alpha sentence one. Alpha sentence two."
    para_b = "Beta sentence one. Beta sentence two."
    chunks = splitter(45, 0).split_text(f"{para_a}\n\n{para_b}")
    assert chunks == [para_a, para_b]


def test_falls_back_to_sentences_then_hard_token_split() -> None:
    long_word = "x" * 130
    chunks = splitter(50, 0).split_text(f"Short one. {long_word}")
    assert chunks[0] == "Short one."
    assert "".join(chunks[1:]) == long_word
    assert all(len(chunk) <= 50 for chunk in chunks)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_configuration_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_"):
        RecursiveTokenSplitter(CharTokenizer(), chunk_size=size, chunk_overlap=overlap)


def test_chunk_blocks_keeps_page_and_heading_metadata() -> None:
    blocks = [
        TextBlock(text="First page text. " * 6, page=1, heading=None),
        TextBlock(text="Second page.", page=2, heading="Intro"),
    ]
    chunks = chunk_blocks(blocks, splitter(40, 5))
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks[:-1]} == {1}
    assert chunks[-1].page == 2
    assert chunks[-1].heading == "Intro"
    assert all(c.token_count == len(c.text) for c in chunks)


def test_tiktoken_tokenizer_counts_tokens() -> None:
    tokenizer = TiktokenTokenizer()
    tokens = tokenizer.encode("Retrieval-augmented generation grounds answers in documents.")
    assert 5 < len(tokens) < 20
    assert tokenizer.decode(tokens).startswith("Retrieval")
