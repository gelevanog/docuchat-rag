"""Small, dependency-free text helpers shared by the offline provider and query rewriting."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_LINE_MARKER_RE = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+|#{1,6}\s+)")
_EMPHASIS_RE = re.compile(r"(\*\*|__)(.+?)\1")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not now of off on once only
    or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would
    you your yours yourself yourselves i'm it's what's there's let's also get got tell
    please many much may might must shall us
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens (letters/digits, with simple apostrophe handling)."""
    return _WORD_RE.findall(text.lower())


def normalize_term(token: str) -> str:
    """A deliberately tiny stemmer: good enough to match `policy`/`policies`, `days`/`day`."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_terms(text: str) -> list[str]:
    """Normalized, stop-word-free terms in their original order."""
    return [normalize_term(tok) for tok in tokenize(text) if tok not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split prose into plain-text sentences; newlines are treated as hard boundaries.

    List markers, heading hashes and bold/underline emphasis markers are removed.
    """
    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = _EMPHASIS_RE.sub(r"\2", _LINE_MARKER_RE.sub("", raw_line.strip())).strip()
        if line:
            sentences.extend(part.strip() for part in _SENTENCE_RE.split(line) if part.strip())
    return sentences
