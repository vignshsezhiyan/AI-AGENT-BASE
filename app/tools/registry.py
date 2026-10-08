"""
ToolRegistry: the ONE place every tool call goes through.

Because everything funnels through execute(), every tool - Python or MCP - automatically gets:
  timeout, error capture, result truncation, audit log line and metrics.

Errors are returned to the LLM as text (not raised), so the model can recover:
retry with fixed arguments, try another tool, or explain the problem to the user.
"""

import asyncio
import fnmatch
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.observability.metrics import TOOL_CALLS, TOOL_LATENCY
from app.tools.base import Tool, ToolArgumentError, ToolError

logger = logging.getLogger("agent.tools")


@dataclass
class ToolResult:
    ok: bool
    content: str  # what the LLM sees


class ToolRegistry:
    def __init__(self, *, timeout_seconds: float, max_result_chars: int, approval_patterns: list[str]):
        self._tools: dict[str, Tool] = {}
        self.timeout_seconds = timeout_seconds
        self.max_result_chars = max_result_chars
        self.approval_patterns = approval_patterns

    # --- registration -------------------------------------------------------
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name '{tool.name}' ({tool.source} vs {self._tools[tool.name].source})")
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        for t in tools:
            self.register(t)

    # --- lookups ------------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "source": t.source, "requires_approval": self.requires_approval(t.name),
                 "description": t.description.splitlines()[0]} for t in self._tools.values()]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def requires_approval(self, name: str) -> bool:
        tool = self._tools.get(name)
        if tool is None:
            return False  # unknown tool -> execute() returns an error, nothing runs
        return tool.requires_approval or any(fnmatch.fnmatch(name, p) for p in self.approval_patterns)

    # --- execution ----------------------------------------------------------
    async def execute(self, name: str, raw_arguments: str | dict | None) -> ToolResult:
        start = time.perf_counter()
        status, result = "error", None
        try:
            tool = self._tools.get(name)
            if tool is None:
                result = ToolResult(False, f"Error: unknown tool '{name}'. Available tools: {', '.join(self.names())}")
                return result
            arguments = _parse_arguments(raw_arguments)
            async with asyncio.timeout(self.timeout_seconds):
                output = await tool.handler(arguments)
            status, result = "ok", ToolResult(True, self._truncate(_to_text(output)))
            return result
        except ToolArgumentError as exc:
            result = ToolResult(False, f"Error: invalid arguments for '{name}': {exc}")
            return result
        except ToolError as exc:
            logger.warning("tool returned error", extra={"tool": name, "error": str(exc)[:500]})
            result = ToolResult(False, f"Error from '{name}': {str(exc)[:2000]}")
            return result
        except TimeoutError:
            status = "timeout"
            result = ToolResult(False, f"Error: tool '{name}' timed out after {self.timeout_seconds}s")
            return result
        except Exception as exc:  # unexpected bug / remote failure - must not kill the agent run
            logger.exception("tool raised", extra={"tool": name})
            result = ToolResult(False, f"Error: tool '{name}' failed: {type(exc).__name__}: {str(exc)[:500]}")
            return result
        finally:
            duration = time.perf_counter() - start
            TOOL_CALLS.labels(name if name in self._tools else "_unknown", status).inc()
            TOOL_LATENCY.labels(name if name in self._tools else "_unknown").observe(duration)
            # Audit line: argument NAMES only - values may contain customer PII. Use DEBUG for values.
            logger.info("tool call", extra={"event": "tool_call",
                "tool": name, "status": status, "duration_ms": round(duration * 1000),
                "arg_keys": sorted(_safe_keys(raw_arguments)),
                "result_chars": len(result.content) if result else 0})

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_result_chars:
            return text
        return text[: self.max_result_chars] + f"\n...[truncated {len(text) - self.max_result_chars} chars]"


def _parse_arguments(raw: str | dict | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolArgumentError(f"arguments are not valid JSON ({exc.msg})") from None
    if not isinstance(parsed, dict):
        raise ToolArgumentError("arguments must be a JSON object")
    return parsed


def _safe_keys(raw: Any) -> list[str]:
    try:
        return list(_parse_arguments(raw).keys())
    except ToolArgumentError:
        return []


def _to_text(output: Any) -> str:
    if output is None:
        return "null"
    if isinstance(output, str):
        return output
    if hasattr(output, "model_dump"):
        output = output.model_dump(mode="json")
    return json.dumps(output, default=str, ensure_ascii=False)
