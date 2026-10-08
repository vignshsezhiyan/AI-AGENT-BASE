"""
Loads and validates agent/agent.yaml.

Any "${VAR_NAME}" inside the YAML is replaced with the environment variable,
so secrets (MCP tokens, URLs) stay in .env and never in the YAML file.
A typo in the YAML fails fast at startup with a clear message.
"""

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
_ENV_VAR = re.compile(r"\$\{([A-Z0-9_]+)\}")


class LLMParams(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"


class Limits(BaseModel):
    max_iterations: int = Field(8, ge=1, le=50)            # LLM <-> tool round trips per request
    request_timeout_seconds: float = Field(180, gt=0)      # whole request, all iterations
    tool_timeout_seconds: float = Field(30, gt=0)          # each single tool call
    max_tool_result_chars: int = Field(20_000, ge=500)     # big results are truncated
    max_history_messages: int = Field(40, ge=2)            # memory kept per session


class ApprovalPolicy(BaseModel):
    # Tool names (wildcards allowed, e.g. "zoho_delete_*") that need a human "yes".
    tools: list[str] = []
    # Also require approval when an MCP server marks a tool as destructive.
    destructive_mcp_tools: bool = True
    # What happens when a human REJECTS:
    #   continue = the LLM is told "rejected" and the conversation goes on (it can suggest alternatives)
    #   end_chat = nothing runs, no further LLM call, the conversation (session) is closed
    on_reject: Literal["continue", "end_chat"] = "continue"


class Guardrails(BaseModel):
    # User message looks like "ignore your instructions / reveal your prompt / you are now DAN":
    #   block = reject with HTTP 400, flag = allow but log + report it, off = no check
    prompt_injection: Literal["block", "flag", "off"] = "block"
    # Tool result (email body, CRM note, web page...) contains instructions aimed at the AI
    # ("indirect" injection): flag = wrap the result with a warning for the LLM + log it
    tool_result_injection: Literal["flag", "off"] = "flag"
    # Is every fact in the answer backed by tool results? Checked by a second LLM call (a "judge").
    #   flag = report grounded true/false in the response, revise = ask the LLM to fix it once, off = skip
    groundedness: Literal["off", "flag", "revise"] = "flag"
    # Personal data masked in final answers. Leave "email"/"phone" out if the agent's job needs them.
    mask_pii: list[Literal["card", "email", "phone"]] = ["card"]
    max_answer_chars: int = Field(8000, ge=200)


class MCPServer(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")  # becomes the tool-name prefix: <name>_<tool>
    transport: Literal["http", "stdio"]
    # http
    url: str | None = None
    headers: dict[str, str] = {}
    # stdio
    command: str | None = None
    args: list[str] = []
    # which tools to expose to the LLM (wildcards on the ORIGINAL tool name)
    include: list[str] = ["*"]
    exclude: list[str] = []
    required: bool = True  # False = start the agent even if this server is down
    timeout_seconds: float = 30

    @model_validator(mode="after")
    def _check_transport(self) -> "MCPServer":
        if self.transport == "http" and not self.url:
            raise ValueError(f"MCP server '{self.name}': 'url' is required for http transport")
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"MCP server '{self.name}': 'command' is required for stdio transport")
        return self


class AgentConfig(BaseModel):
    name: str
    description: str = ""
    llm: LLMParams = LLMParams()
    limits: Limits = Limits()
    approval: ApprovalPolicy = ApprovalPolicy()
    guardrails: Guardrails = Guardrails()
    mcp_servers: list[MCPServer] = []
    system_prompt: str = ""  # loaded from agent/prompt.md


def _expand_env(value):
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            if var not in os.environ:
                raise ValueError(f"agent.yaml references ${{{var}}} but it is not set in the environment")
            return os.environ[var]
        return _ENV_VAR.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_agent_config(agent_dir: Path = AGENT_DIR) -> AgentConfig:
    load_dotenv(override=False)  # make .env values visible to ${VAR} expansion
    raw =yaml.safe_load((agent_dir / "agent.yaml").read_text(encoding="utf-8")) or {}
    config = AgentConfig.model_validate(_expand_env(raw))
    config.system_prompt = (agent_dir / "prompt.md").read_text(encoding="utf-8").strip()
    return config
