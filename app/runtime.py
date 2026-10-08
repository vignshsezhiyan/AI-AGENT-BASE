"""
Startup / shutdown wiring. Builds the agent from:
  .env               -> LLM endpoint, secrets
  agent/agent.yaml   -> limits, approval policy, MCP servers
  agent/prompt.md    -> system prompt
  agent/tools.py     -> Python tools (every @tool function is picked up automatically)
"""

import importlib
import logging
from dataclasses import replace

from app.agent import Agent
from app.agent_config import AgentConfig, load_agent_config
from app.llm import OpenAILLM
from app.observability.metrics import AGENT_INFO, MCP_SERVER_UP
from app.security.guardrails import Redactor
from app.settings import Settings
from app.tools.base import collect_tools
from app.tools.mcp import MCPConnection
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config: AgentConfig | None = None
        self.agent: Agent | None = None
        self.judge: OpenAILLM | None = None
        self.mcp: list[MCPConnection] = []
        self.mcp_status: dict[str, str] = {}

    async def start(self) -> Agent:
        s = self.settings
        self.config = config = self.load_config()
        registry = ToolRegistry(timeout_seconds=config.limits.tool_timeout_seconds,
                                max_result_chars=config.limits.max_tool_result_chars,
                                approval_patterns=config.approval.tools)

        # 1. Python tools from agent/tools.py
        registry.register_many(collect_tools(importlib.import_module("agent.tools")))

        # 2. MCP tools from agent.yaml
        for server in config.mcp_servers:
            conn = MCPConnection(server)
            try:
                tools = await conn.connect()
                if not config.approval.destructive_mcp_tools:
                    tools = [replace(t, requires_approval=False) for t in tools]
                registry.register_many(tools)
                self.mcp.append(conn)
                self.mcp_status[server.name] = "up"
            except Exception:
                await conn.close()
                MCP_SERVER_UP.labels(server.name).set(0)
                self.mcp_status[server.name] = "down"
                if server.required:
                    raise
                logger.exception("optional mcp server failed - continuing without it",
                                 extra={"server": server.name})

        # 3. LLM + agent
        llm = OpenAILLM(api_key=s.llm_api_key.get_secret_value(), base_url=s.llm_base_url, model=s.llm_model,
                        temperature=config.llm.temperature, max_tokens=config.llm.max_tokens,
                        reasoning_effort=config.llm.reasoning_effort,
                        timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries)
        # The groundedness "judge": same endpoint, temperature 0 for consistent verdicts.
        judge = None
        if config.guardrails.groundedness != "off":
            judge = OpenAILLM(api_key=s.llm_api_key.get_secret_value(), base_url=s.llm_base_url, model=s.llm_model,
                              temperature=0.0, max_tokens=2048, reasoning_effort="low",
                              timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries)
        redactor = Redactor(extra_secrets=list(s.api_key_map) + [s.llm_api_key.get_secret_value()])
        self.agent = Agent(config, llm, registry, redactor, judge=judge)
        self.judge = judge

        AGENT_INFO.labels(config.name, s.llm_model).set(1)
        logger.info("agent ready", extra={"agent": config.name, "model": s.llm_model,
                                          "tools": registry.names(), "mcp": self.mcp_status})
        return self.agent

    def load_config(self) -> AgentConfig:
        return load_agent_config()

    async def stop(self) -> None:
        for conn in reversed(self.mcp):
            await conn.close()
        if self.agent:
            await self.agent.llm.close()
        if self.judge:
            await self.judge.close()
        self.agent = None
        logger.info("agent stopped")
