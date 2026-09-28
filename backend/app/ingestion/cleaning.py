"""Normalise raw extracted text before chunking."""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_INLINE_SPACE = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
# A line that ends mid-sentence followed by a line starting in lower case: a soft wrap (PDFs).
_SOFT_WRAP = re.compile(r"(?<=[a-z0-9,;])\n(?=[a-z(])")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _INLINE_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _SOFT_WRAP.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
