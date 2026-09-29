from __future__ import annotations

import asyncio

import httpx2
import openai
import pytest

from app.llm.pacing import RequestPacer


class FakeTime:
    """A clock that only moves when the pacer sleeps."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


def _rate_limited(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx2.Request("POST", "https://openrouter.test/api/v1/chat/completions")
    response = httpx2.Response(429, headers=headers, request=request)
    return openai.RateLimitError("rate-limited upstream", response=response, body=None)


class Flaky:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _pacer(
    time: FakeTime, min_interval: float = 0.0, retries: int = 0, backoff: float = 5.0
) -> RequestPacer:
    return RequestPacer(min_interval, retries, backoff, sleep=time.sleep, clock=time.clock)


async def test_requests_are_spaced_by_the_minimum_interval() -> None:
    time = FakeTime()
    pacer = _pacer(time, min_interval=3.0)
    for _ in range(3):
        assert await pacer.run(Flaky([])) == "ok"
    assert time.sleeps == [3.0, 3.0]  # the first request goes out immediately


async def test_concurrent_requests_queue_behind_each_other() -> None:
    time = FakeTime()
    pacer = _pacer(time, min_interval=2.0)
    results = await asyncio.gather(*(pacer.run(Flaky([])) for _ in range(3)))
    assert results == ["ok", "ok", "ok"]
    assert time.sleeps == [2.0, 2.0]


async def test_rate_limits_are_retried_with_exponential_backoff() -> None:
    time = FakeTime()
    request = Flaky([_rate_limited(), _rate_limited()])
    assert await _pacer(time, retries=3, backoff=5.0).run(request) == "ok"
    assert request.calls == 3
    assert time.sleeps == [5.0, 10.0]


async def test_retry_after_header_wins_over_the_backoff() -> None:
    time = FakeTime()
    request = Flaky([_rate_limited(retry_after="7")])
    assert await _pacer(time, retries=1, backoff=5.0).run(request) == "ok"
    assert time.sleeps == [7.0]


async def test_gives_up_after_the_configured_retries_and_does_not_retry_other_errors() -> None:
    time = FakeTime()
    exhausted = Flaky([_rate_limited(), _rate_limited()])
    with pytest.raises(openai.RateLimitError):
        await _pacer(time, retries=1).run(exhausted)
    assert exhausted.calls == 2

    bad_request = Flaky([ValueError("not transient")])
    with pytest.raises(ValueError, match="not transient"):
        await _pacer(time, retries=3).run(bad_request)
    assert bad_request.calls == 1


async def test_callers_can_mark_more_errors_as_retryable() -> None:
    time = FakeTime()
    request = Flaky([KeyError("stream error event")])
    retry_if = lambda exc: isinstance(exc, KeyError)  # noqa: E731
    assert await _pacer(time, retries=1, backoff=2.0).run(request, retry_if) == "ok"
    assert time.sleeps == [2.0]
