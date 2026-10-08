"""
Test helpers. FakeLLM replays scripted responses, so tests run offline, fast and deterministic.
"""

import json

import pytest

from app.agent import Agent
from app.agent_config import AgentConfig
from app.llm import LLMResponse
from app.runtime import Runtime
from app.security.groundedness import GroundednessChecker
from app.security.guardrails import Redactor
from app.sessions import Session
from app.settings import Settings
from app.tools.base import tool
from app.tools.registry import ToolRegistry


def tool_call(name: str, args: dict | str, call_id: str = "call_1") -> dict:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, prompt_tokens=10, completion_tokens=5)


def calls(*tool_calls: dict) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=list(tool_calls), prompt_tokens=10, completion_tokens=5)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[list[dict]] = []   # every `messages` list the agent sent

    async def run(self, messages, tools=None, response_format=None):
        self.requests.append([dict(m) for m in messages])
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        pass


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(requires_approval=True)
def delete_record(record_id: str) -> dict:
    """Delete a record."""
    return {"deleted": record_id}


def make_agent(llm: FakeLLM, judge: FakeLLM | None = None, guardrails: dict | None = None, **limits) -> Agent:
    config = AgentConfig(name="test", system_prompt="You are a test agent that helps the sales team with leads.",
                         limits=limits or {}, guardrails=guardrails or {})
    registry = ToolRegistry(timeout_seconds=config.limits.tool_timeout_seconds,
                            max_result_chars=config.limits.max_tool_result_chars, approval_patterns=[])
    registry.register(add.__agent_tool__)
    registry.register(delete_record.__agent_tool__)
    return Agent(config, llm, registry, Redactor(), judge=judge)


@pytest.fixture
def session() -> Session:
    return Session(id="s1", owner="tester")


def make_settings(**overrides) -> Settings:
    values = dict(llm_base_url="http://fake/v1", llm_model="fake-model", log_format="text", log_dir="",
                  api_keys="tests:test-key-0123456789abcdef", rate_limit_per_minute=1000)
    values.update(overrides)
    return Settings(_env_file=None, **values)


class FakeRuntime(Runtime):
    """Real startup (loads agent/), but with the LLM swapped for a FakeLLM."""

    def __init__(self, settings: Settings, llm: FakeLLM, judge: FakeLLM | None = None):
        super().__init__(settings)
        self.fake_llm = llm
        self.fake_judge = judge

    def load_config(self):
        config = super().load_config()
        config.mcp_servers = []  # tests never call real MCP servers (see test_mcp.py for a local fake)
        return config

    async def start(self):
        agent = await super().start()
        await agent.llm.close()
        if self.judge:
            await self.judge.close()
            self.judge = None
        agent.llm = self.fake_llm
        agent.grounding = GroundednessChecker(self.fake_judge)  # None = groundedness not checked
        return agent
