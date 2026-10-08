"""
API-key authentication.

Clients send:   X-API-Key: <key>
Keys come from the API_KEYS env var as "caller:key" pairs. The caller name is used for
logs, rate limits and session ownership (caller A can never read caller B's session).

If API_KEYS is empty, auth is OFF - allowed only in dev (settings.py refuses it in production).
"""

import hmac
import logging

from fastapi import Request
from fastapi.security import APIKeyHeader

from app.observability.logging import caller_var
from app.observability.metrics import SECURITY_EVENTS
from app.security.errors import AuthError

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def resolve_caller(provided: str | None, key_map: dict[str, str]) -> str:
    if not key_map:
        return "anonymous"
    if provided:
        # compare_digest on every key -> no timing side channel revealing valid key prefixes
        for key, caller in key_map.items():
            if hmac.compare_digest(provided.encode(), key.encode()):
                return caller
    SECURITY_EVENTS.labels("auth_failed").inc()
    logger.warning("authentication failed", extra={"has_key": bool(provided)})
    raise AuthError()


async def require_caller(request: Request) -> str:
    """FastAPI dependency: returns the caller name or raises 401."""
    provided = await api_key_header(request)
    caller = resolve_caller(provided, request.app.state.api_key_map)
    caller_var.set(caller)
    request.state.caller = caller  # lets the HTTP middleware log who called
    request.app.state.rate_limiter.check(caller)
    return caller
