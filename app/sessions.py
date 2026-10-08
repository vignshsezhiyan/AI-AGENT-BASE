"""
Conversation memory.

A session = the message history of one conversation, owned by one caller.
Kept in memory with a TTL and a max count so it cannot grow forever.

To scale to several replicas, replace SessionStore with a Redis-backed one that has the same
methods (get_or_create / get / delete) - nothing else in the code needs to change.
"""

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.observability.metrics import ACTIVE_SESSIONS


@dataclass
class PendingApproval:
    id: str
    tool_calls: list[dict[str, Any]]   # the LLM's tool_calls waiting for a human decision
    needs_approval: list[str]          # names of the tools that triggered the approval


@dataclass
class Session:
    id: str
    owner: str
    messages: list[dict[str, Any]] = field(default_factory=list)  # OpenAI chat format, no system prompt
    pending: PendingApproval | None = None
    updated_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)       # one request per session at a time


class SessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[tuple[str, str], Session] = OrderedDict()

    def get_or_create(self, owner: str, session_id: str | None) -> Session:
        if session_id:
            existing = self.get(owner, session_id)
            if existing:
                return existing
        session = Session(id=session_id or uuid.uuid4().hex, owner=owner)
        self._sessions[(owner, session.id)] = session
        self._evict()
        return session

    def get(self, owner: str, session_id: str) -> Session | None:
        self._evict()
        session = self._sessions.get((owner, session_id))  # owner in the key = isolation between callers
        if session:
            session.updated_at = time.monotonic()
            self._sessions.move_to_end((owner, session_id))
        return session

    def delete(self, owner: str, session_id: str) -> bool:
        found = self._sessions.pop((owner, session_id), None) is not None
        ACTIVE_SESSIONS.set(len(self._sessions))
        return found

    def _evict(self) -> None:
        now = time.monotonic()
        # oldest-used first (OrderedDict order): drop expired, then enforce the cap
        while self._sessions:
            key, oldest = next(iter(self._sessions.items()))
            expired = now - oldest.updated_at > self.ttl
            if not (expired or len(self._sessions) > self.max_sessions) or oldest.lock.locked():
                break
            del self._sessions[key]
        ACTIVE_SESSIONS.set(len(self._sessions))
