"""
LLM client for any OpenAI-compatible endpoint (our vLLM gpt-oss server, OpenAI, Azure, Ollama...).

Same shape as the team's original OpenAILLM:  run(messages, tools, response_format)
returns content / tool_calls / reasoning.  Changes for production:

  1. AsyncOpenAI instead of OpenAI    -> a slow LLM call no longer freezes the whole server
  2. timeout + automatic retries      -> 429 / 5xx / network blips are retried with backoff
  3. token usage + latency recorded   -> visible in /metrics and logs
  4. params come from config          -> no hard-coded keys, URLs or temperatures
  5. errors wrapped in LLMError       -> API returns a clean 502 instead of a stack trace
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from app.observability.metrics import LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """The LLM endpoint failed after retries (down, timeout, bad request...)."""


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAILLM:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 16000,
        reasoning_effort: str | None = "low",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    async def run(self, messages: list[dict], tools: list[dict] | None = None,
                  response_format: dict | None = None) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if response_format:
            params["response_format"] = response_format

        start = time.perf_counter()
        try:
            resp = (await self.client.chat.completions.create(**params)).model_dump()
        except openai.APIError as exc:
            LLM_REQUESTS.labels(self.model_name, "error").inc()
            logger.error("llm call failed", extra={"model": self.model_name, "error": type(exc).__name__,
                                                   "detail": str(exc)[:500]})
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        finally:
            LLM_LATENCY.labels(self.model_name).observe(time.perf_counter() - start)

        if not resp.get("choices"):
            LLM_REQUESTS.labels(self.model_name, "error").inc()
            raise LLMError("LLM returned no choices")

        LLM_REQUESTS.labels(self.model_name, "ok").inc()
        choice = resp["choices"][0]
        msg = choice["message"]
        usage = resp.get("usage") or {}
        result = LLMResponse(
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls") or [],
            # vLLM exposes gpt-oss thinking as `reasoning` or `reasoning_content` depending on version
            reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
        )
        LLM_TOKENS.labels(self.model_name, "prompt").inc(result.prompt_tokens)
        LLM_TOKENS.labels(self.model_name, "completion").inc(result.completion_tokens)
        logger.info("llm call ok", extra={"event": "llm_call",
            "model": self.model_name, "duration_ms": round((time.perf_counter() - start) * 1000),
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "tool_calls": len(result.tool_calls), "finish_reason": result.finish_reason,
        })
        return result

    async def close(self) -> None:
        await self.client.close()
