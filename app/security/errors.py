"""
Every error a client can see. Each maps to an HTTP status + a SAFE message.
Internal details (stack traces, upstream errors) go to logs only, never to the client.
"""


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    message = "Something went wrong. Quote the request_id when reporting this."

    headers: dict[str, str] = {}

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message


class AuthError(AppError):
    status_code, code, message = 401, "unauthorized", "Missing or invalid API key."


class RateLimitError(AppError):
    status_code, code, message = 429, "rate_limited", "Too many requests. Slow down and retry shortly."

    def __init__(self, message: str | None = None, retry_after: int = 60):
        super().__init__(message)
        self.headers = {"Retry-After": str(retry_after)}


class ServerBusy(AppError):
    status_code, code, message = 503, "busy", "The agent is at full capacity. Please retry in a minute."
    headers = {"Retry-After": "30"}


class InputRejected(AppError):
    status_code, code = 400, "input_rejected"


class NotFound(AppError):
    status_code, code = 404, "not_found"


class Conflict(AppError):
    status_code, code = 409, "conflict"


class UpstreamError(AppError):
    status_code, code, message = 502, "llm_unavailable", "The language model is unavailable. Please retry."


class AgentTimeout(AppError):
    status_code, code, message = 504, "timeout", "The agent took too long to answer. Please retry."
