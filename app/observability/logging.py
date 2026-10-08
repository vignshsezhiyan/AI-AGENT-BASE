"""
Logging setup.

Two outputs:
  - stdout: one line per event, every line carries request_id / session_id / caller
            (JSON in production, for Loki/ELK/CloudWatch)
  - files:  human-readable, one block per request -> see app/observability/request_log.py

    logger.info("tool finished", extra={"tool": name, "duration_ms": 12})
"""

import json
import logging
import shutil
import sys
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.observability.request_log import RequestLogCollector

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
session_id_var: ContextVar[str] = ContextVar("session_id", default="-")
caller_var: ContextVar[str] = ContextVar("caller", default="-")

# Attributes every LogRecord has; anything else came from `extra=` and is logged as a field.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        record.caller = caller_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS:
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s%(extras)s")

    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in vars(record).items()
                  if k not in _STANDARD_ATTRS and k not in ("request_id", "session_id", "caller", "extras")}
        record.extras = f"  {extras}" if extras else ""
        return super().format(record)


class TraceFormatter(logging.Formatter):
    """Overall trace (logs/app.log): one readable line per event, every request interleaved in time order.

    2026-10-08 15:35:25.512  INFO     req=6fbcf363...  caller=dev  agent.tools     tool call  tool=add_numbers status=ok
    Filter one request:  grep "req=<request_id>" logs/app.log
    """

    def format(self, record: logging.LogRecord) -> str:
        when = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S") + f".{int(record.msecs):03d}"
        extras = "  ".join(f"{k}={v}" for k, v in vars(record).items()
                           if k not in _STANDARD_ATTRS and k not in ("request_id", "session_id", "caller", "extras", "event"))
        line = (f"{when}  {record.levelname:<7}  req={record.request_id:<32}  caller={record.caller:<10}  "
                f"{record.name:<15} {record.getMessage()}" + (f"  {extras}" if extras else ""))
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _delete_old_request_logs(directory: Path, retention_days: int) -> None:
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    for day_dir in directory.glob("????-??-??"):
        if day_dir.is_dir() and day_dir.name < cutoff:
            shutil.rmtree(day_dir, ignore_errors=True)


_collector: RequestLogCollector | None = None


def finish_request(request_id: str) -> None:
    """Write the readable block for a finished request (called by the HTTP middleware / CLI)."""
    if _collector:
        _collector.finish(request_id)


def configure_logging(level: str = "INFO", fmt: str = "json", *, log_dir: str | None = None,
                      request_files: bool = False, payloads: bool = False, retention_days: int = 14) -> None:
    """
    stdout                                 one line per event (JSON in production) - for log collectors
    <log_dir>/app.log                      OVERALL trace: every event of every request, one readable line each
                                           (rotated 20 MB x 5). grep "req=<request_id>" to follow one request.
    <log_dir>/requests/<day>/<id>.log      PER-REQUEST: one readable block (steps, thinking, LLM, tools, answer)

    payloads=True adds the user message, LLM thinking and tool arguments/results to the per-request
    blocks. app.log never contains payloads (it is the long-lived file - keep customer data out of it).
    """
    global _collector
    context = _ContextFilter()

    stdout = logging.StreamHandler(sys.stdout)
    stdout.addFilter(context)
    stdout.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    stdout.setLevel(level)
    handlers: list[logging.Handler] = [stdout]

    if _collector:
        _collector.close()
        _collector = None
    if log_dir:
        base = Path(log_dir)
        base.mkdir(parents=True, exist_ok=True)
        app_log = RotatingFileHandler(base / "app.log", maxBytes=20_000_000, backupCount=5, encoding="utf-8")
        app_log.addFilter(context)
        app_log.setFormatter(TraceFormatter())
        app_log.setLevel(level)  # same level as stdout -> payload (DEBUG) lines never reach app.log
        handlers.append(app_log)
        if request_files:
            _delete_old_request_logs(base / "requests", retention_days)
            _collector = RequestLogCollector(base / "requests")
            _collector.addFilter(context)
            handlers.append(_collector)

    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level)

    # "agent.trace" carries payloads (messages, thinking, tool arguments/results) at DEBUG level:
    # it reaches the per-request blocks, but never stdout or app.log (which use LOG_LEVEL).
    logging.getLogger("agent.trace").setLevel(logging.DEBUG if payloads else logging.INFO)
    # Step lines (iteration, llm call, tool call) are INFO; keep them even if LOG_LEVEL is WARNING
    # so the request blocks stay complete.
    for name in ("agent.run", "agent.tools", "app.llm", "app"):
        logging.getLogger(name).setLevel(min(logging.INFO, root.level))

    # We log our own access line per request (with request_id), so silence uvicorn's.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "mcp"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
