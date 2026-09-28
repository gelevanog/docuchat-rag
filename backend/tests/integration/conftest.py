from __future__ import annotations

from typing import Any

import httpx
import pytest

HANDBOOK = b"""# Acme Handbook

## Vacation
Full-time employees receive 25 days of paid vacation per calendar year.
Up to 5 unused days may be carried over and expire on 31 March.

## Parental Leave
Birthing parents receive 20 weeks of fully paid leave.
Adoptive parents receive 10 weeks of fully paid leave.
"""

SECURITY = b"""# Security Policy

## Passwords
Passwords must be at least 14 characters long and stored in the password manager.

## Devices
A lost laptop must be reported to the IT helpdesk within one hour.
"""


async def upload(client: httpx.AsyncClient, name: str, content: bytes) -> dict[str, Any]:
    response = await client.post("/api/documents", files={"file": (name, content)})
    assert response.status_code == 202, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.fixture
async def handbook(client: httpx.AsyncClient) -> dict[str, Any]:
    return (await upload(client, "handbook.md", HANDBOOK))["document"]


@pytest.fixture
async def security(client: httpx.AsyncClient) -> dict[str, Any]:
    return (await upload(client, "security.md", SECURITY))["document"]
