"""Client-side pacing for rate-limited, OpenAI-compatible APIs (e.g. OpenRouter's free models).

`RequestPacer` spaces request starts at least `min_interval` seconds apart and retries
rate-limit (429), server (5xx) and connection errors (plus any error the caller marks as
safe to retry) with exponential backoff, honouring a `Retry-After` header when present.
One pacer is shared by every client that draws on the same quota, so chat, judge and
embedding calls queue behind each other.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import openai

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

RETRYABLE_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)
_MAX_RETRY_AFTER = 120.0


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return min(float(value), _MAX_RETRY_AFTER) if value is not None else None
    except ValueError:  # an HTTP date instead of seconds; fall back to our own backoff
        return None


class RequestPacer:
    def __init__(
        self,
        min_interval: float = 0.0,
        retries: int = 0,
        backoff: float = 5.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval = min_interval
        self._retries = retries
        self._backoff = backoff
        self._sleep = sleep
        self._clock = clock
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def _wait_for_slot(self) -> None:
        async with self._lock:
            delay = self._next_start - self._clock()
            if delay > 0:
                await self._sleep(delay)
            self._next_start = self._clock() + self._min_interval

    async def run(
        self,
        request: Callable[[], Awaitable[T]],
        retry_if: Callable[[Exception], bool] | None = None,
    ) -> T:
        """Run `request` (a fresh coroutine per attempt) once a slot is free, retrying
        transient failures, and errors matching `retry_if`, up to `retries` times."""
        attempt = 0
        while True:
            await self._wait_for_slot()
            try:
                return await request()
            except Exception as exc:
                transient = isinstance(exc, RETRYABLE_ERRORS) or (retry_if and retry_if(exc))
                if not transient or attempt >= self._retries:
                    raise
                wait = _retry_after(exc) or self._backoff * 2**attempt
                attempt += 1
                logger.warning(
                    "request_retry", error=type(exc).__name__, attempt=attempt, wait_s=wait
                )
                await self._sleep(wait)
