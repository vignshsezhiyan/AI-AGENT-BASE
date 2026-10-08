"""
FastAPI application.  Run with:   uvicorn app.main:create_app --factory

Every request passes through `observe_request` which:
  - assigns a request_id (or reuses a valid incoming X-Request-ID) and returns it in the response
  - writes ONE access log line (method, route, status, duration, caller)
  - records HTTP metrics
  - adds basic security headers

Errors: AppError subclasses -> their status + safe message.  Anything unexpected -> generic 500.
The client always gets a request_id to quote; the full details are in the logs under that id.
"""

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import public, v1
from app.observability.logging import (caller_var, configure_logging, finish_request, request_id_var,
                                       session_id_var)
from app.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.runtime import Runtime
from app.security.errors import AppError
from app.security.rate_limit import RateLimiter
from app.sessions import SessionStore
from app.settings import Settings, get_settings

logger = logging.getLogger("app")
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format, log_dir=settings.log_dir or None,
                      request_files=settings.log_request_files, payloads=settings.log_request_payloads,
                      retention_days=settings.log_retention_days)
    runtime = runtime or Runtime(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(
        title="Agent API", version="1.0.0", lifespan=lifespan,
        docs_url=None if settings.app_env == "production" else "/docs",  # no public Swagger in prod
        redoc_url=None, openapi_url=None if settings.app_env == "production" else "/openapi.json",
    )
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.api_key_map = settings.api_key_map
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute, settings.max_concurrent_per_caller,
                                         settings.daily_token_budget_per_caller, settings.max_concurrent_runs)
    app.state.sessions = SessionStore(settings.session_ttl_seconds, settings.max_sessions)

    if not settings.api_key_map:
        logger.warning("API_KEYS is empty - authentication is DISABLED (dev only)")

    if settings.cors_origin_list:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                           allow_methods=["GET", "POST", "DELETE"], allow_headers=["X-API-Key", "Content-Type"])

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request_id_var.set(request_id)
        session_id_var.set("-")
        caller_var.set("-")
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            path = route.path if route else "_unmatched"  # template path keeps metric cardinality low
            caller_var.set(getattr(request.state, "caller", "-"))
            duration = time.perf_counter() - start
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(path).observe(duration)
            if path not in ("/health", "/ready", "/metrics"):
                logger.info("http request", extra={"event": "http_request", "method": request.method, "path": path, "status": status,
                                                   "duration_ms": round(duration * 1000)})
            finish_request(request_id)  # write this request's readable block to app.log + requests/
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError):
        if exc.status_code >= 500:
            logger.error("request failed", exc_info=exc.__cause__ or exc, extra={"error": exc.code})
        response = _error(exc.status_code, exc.code, exc.message)
        response.headers.update(exc.headers)  # e.g. Retry-After on 429 / 503
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        problems = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors())
        return _error(422, "invalid_request", problems[:1000])

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return _error(500, AppError.code, AppError.message)

    app.include_router(public)
    app.include_router(v1)
    return app


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message, "request_id": request_id_var.get()},
                        status_code=status)
