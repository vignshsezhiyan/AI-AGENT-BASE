"""
Generic MCP connector, driven entirely by agent/agent.yaml (no code per MCP server).

  transport: http  -> remote MCP over Streamable HTTP (e.g. Copilot MCP with a bearer token)
  transport: stdio -> launches a local process      (e.g. `npx mcp-remote <zoho-url>`)

Each MCP tool is exposed to the LLM as "<server>_<tool>" so names never collide.
Only tools matching include/exclude are exposed: give the agent the minimum it needs.
"""

import fnmatch
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from app.agent_config import MCPServer
from app.observability.metrics import MCP_SERVER_UP
from app.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)


class MCPToolError(ToolError):
    """The MCP server reported that the tool call failed."""


class MCPConnection:
    def __init__(self, config: MCPServer):
        self.config = config
        self.client: Client | None = None
        self._stack = AsyncExitStack()

    async def connect(self) -> list[Tool]:
        cfg = self.config
        if cfg.transport == "http":
            http = await self._stack.enter_async_context(httpx2.AsyncClient(
                headers=cfg.headers, timeout=httpx2.Timeout(cfg.timeout_seconds, read=300.0), follow_redirects=True))
            target = streamable_http_client(cfg.url, http_client=http)
        else:
            target = StdioServerParameters(command=cfg.command, args=cfg.args)

        self.client = await self._stack.enter_async_context(
            Client(target, read_timeout_seconds=cfg.timeout_seconds))
        listed = await self.client.list_tools()

        tools = [self._wrap(t) for t in listed.tools if self._allowed(t.name)]
        MCP_SERVER_UP.labels(cfg.name).set(1)
        logger.info("mcp server connected", extra={
            "server": cfg.name, "tools_available": len(listed.tools), "tools_exposed": len(tools)})
        return tools

    def _allowed(self, tool_name: str) -> bool:
        match = lambda patterns: any(fnmatch.fnmatch(tool_name, p) for p in patterns)  # noqa: E731
        return match(self.config.include) and not match(self.config.exclude)

    def _wrap(self, mcp_tool) -> Tool:
        original = mcp_tool.name
        annotations = mcp_tool.annotations
        destructive = bool(annotations and annotations.destructive_hint and not annotations.read_only_hint)

        async def handler(arguments: dict[str, Any]) -> Any:
            if self.client is None:
                raise MCPToolError(f"MCP server '{self.config.name}' is not connected")
            result = await self.client.call_tool(original, arguments)
            output = _result_to_python(result)
            if result.is_error:
                raise MCPToolError(str(output)[:2000])
            return output

        schema = mcp_tool.input_schema or {"type": "object", "properties": {}}
        return Tool(
            name=f"{self.config.name}_{original}"[:64],  # OpenAI-style APIs cap names at 64 chars
            description=mcp_tool.description or f"{original} ({self.config.name} MCP tool)",
            parameters=schema if isinstance(schema, dict) else json.loads(json.dumps(schema, default=str)),
            handler=handler,
            requires_approval=destructive,  # registry also applies the yaml approval patterns
            source=f"mcp:{self.config.name}",
        )

    async def close(self) -> None:
        try:
            await self._stack.aclose()
        except Exception:
            logger.exception("mcp server close failed", extra={"server": self.config.name})
        finally:
            self.client = None
            MCP_SERVER_UP.labels(self.config.name).set(0)


def _result_to_python(result) -> Any:
    """Prefer structured output; otherwise join the text blocks."""
    if result.structured_content is not None:
        return result.structured_content
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{getattr(block, 'type', 'content')} omitted]")
    return "\n".join(parts)
