"""
Rate limiting - protects the LLM server and the budget from runaway scripts, loops or a leaked key.

Four limits (all in .env):
  RATE_LIMIT_PER_MINUTE          requests per caller per minute (sliding window)        -> 429
  MAX_CONCURRENT_PER_CALLER      requests one caller can have running at the same time  -> 429
  DAILY_TOKEN_BUDGET_PER_CALLER  LLM tokens per caller per day (0 = unlimited)           -> 429
  MAX_CONCURRENT_RUNS            agent runs in the whole service at once; extra requests
                                 wait up to 30s for a free slot, then get                -> 503
Every 429/503 carries a Retry-After header so well-behaved clients back off automatically.

NOTE: counters live in memory, per process. With several replicas, move them to Redis or enforce
the limits at your API gateway.
"""

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date

from app.observability.metrics import SECURITY_EVENTS
from app.security.errors import RateLimitError, ServerBusy

_QUEUE_WAIT_SECONDS = 30


class RateLimiter:
    def __init__(self, per_minute: int, max_concurrent_per_caller: int = 0,
                 daily_token_budget: int = 0, max_concurrent_runs: int = 0):
        self.per_minute = per_minute
        self.max_concurrent_per_caller = max_concurrent_per_caller
        self.daily_token_budget = daily_token_budget
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._running: dict[str, int] = defaultdict(int)
        self._tokens: dict[tuple[str, date], int] = defaultdict(int)
        self._slots = asyncio.Semaphore(max_concurrent_runs) if max_concurrent_runs > 0 else None

    # --- called for every authenticated request ------------------------------------
    def check(self, caller: str) -> None:
        if self.per_minute > 0:
            now = time.monotonic()
            hits = self._hits[caller]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                SECURITY_EVENTS.labels("rate_limited").inc()
                raise RateLimitError(retry_after=int(60 - (now - hits[0])) + 1)
            hits.append(now)

        if self.daily_token_budget > 0 and self._tokens[(caller, date.today())] >= self.daily_token_budget:
            SECURITY_EVENTS.labels("token_budget_exceeded").inc()
            raise RateLimitError("Daily token budget used up for this API key. It resets at midnight.",
                                 retry_after=_seconds_to_midnight())

    # --- wraps the agent run --------------------------------------------------------
    @asynccontextmanager
    async def run_slot(self, caller: str):
        if self.max_concurrent_per_caller > 0 and self._running[caller] >= self.max_concurrent_per_caller:
            SECURITY_EVENTS.labels("too_many_concurrent").inc()
            raise RateLimitError("Too many requests running at the same time for this API key.", retry_after=5)
        if self._slots:
            try:
                await asyncio.wait_for(self._slots.acquire(), _QUEUE_WAIT_SECONDS)
            except TimeoutError:
                SECURITY_EVENTS.labels("server_busy").inc()
                raise ServerBusy() from None
        self._running[caller] += 1
        try:
            yield
        finally:
            self._running[caller] -= 1
            if self._slots:
                self._slots.release()

    def add_tokens(self, caller: str, tokens: int) -> None:
        if self.daily_token_budget > 0:
            today = date.today()
            for key in [k for k in self._tokens if k[1] != today]:  # forget previous days
                del self._tokens[key]
            self._tokens[(caller, today)] += tokens


def _seconds_to_midnight() -> int:
    now = time.localtime()
    return max(1, 86400 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec))
