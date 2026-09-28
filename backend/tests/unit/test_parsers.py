from __future__ import annotations

import pytest

from app.ingestion.cleaning import clean_text
from app.ingestion.parsers import DocumentParseError, detect_file_type, parse_document
from tests.helpers import make_docx, make_pdf

MARKDOWN = b"""---
tags: [demo]
---
# Team Handbook

Intro paragraph.

## Leave

### Vacation
You get 25 days.

```python
# not a heading
```

## Expenses
Submit within 30 days.
"""


def test_detect_file_type() -> None:
    assert detect_file_type("Report.PDF") == "pdf"
    assert detect_file_type("notes.markdown") == "markdown"
    with pytest.raises(DocumentParseError, match="Unsupported"):
        detect_file_type("image.png")


def test_markdown_sections_carry_heading_paths() -> None:
    parsed = parse_document(MARKDOWN, "handbook.md")
    assert parsed.title == "Team Handbook"
    headings = [block.heading for block in parsed.blocks]
    assert headings == [None, "Leave > Vacation", "Expenses"]
    assert "# not a heading" in parsed.blocks[1].text
    assert "tags" not in parsed.blocks[0].text


def test_text_file_uses_filename_as_title() -> None:
    parsed = parse_document("Café menu\n\n\n\nOpen daily.".encode(), "cafe_menu-2026.txt")
    assert parsed.title == "cafe menu 2026"
    assert parsed.blocks[0].text == "Café menu\n\nOpen daily."


def test_pdf_pages_are_numbered() -> None:
    data = make_pdf(
        ["Page one talks about vacation.", "Page two talks about expenses."], title="Policy Pack"
    )
    parsed = parse_document(data, "policy.pdf")
    assert parsed.title == "Policy Pack"
    assert parsed.page_count == 2
    assert [block.page for block in parsed.blocks] == [1, 2]
    assert "expenses" in parsed.blocks[1].text


def test_pdf_without_text_is_rejected() -> None:
    with pytest.raises(DocumentParseError, match="no text layer"):
        parse_document(make_pdf([""]), "scan.pdf")


def test_corrupt_pdf_is_rejected() -> None:
    with pytest.raises(DocumentParseError):
        parse_document(b"%PDF-1.4 definitely not a pdf", "broken.pdf")


def test_docx_headings_become_sections() -> None:
    data = make_docx(
        "Employee Guide", {"Leave": ["25 days of vacation."], "Travel": ["Economy class."]}
    )
    parsed = parse_document(data, "guide.docx")
    assert parsed.title == "Employee Guide"
    assert [(b.heading, b.text) for b in parsed.blocks] == [
        ("Leave", "25 days of vacation."),
        ("Travel", "Economy class."),
    ]


def test_corrupt_docx_is_rejected() -> None:
    with pytest.raises(DocumentParseError, match="DOCX"):
        parse_document(b"not a zip", "broken.docx")


def test_empty_document_is_rejected() -> None:
    with pytest.raises(DocumentParseError, match="No extractable text"):
        parse_document(b"   \n  ", "empty.txt")


def test_clean_text_repairs_pdf_artifacts() -> None:
    raw = "Employ-\nees get   25\tdays\nof paid leave.\n\n\n\nNext para\x00graph."
    assert clean_text(raw) == "Employees get 25 days of paid leave.\n\nNext paragraph."
