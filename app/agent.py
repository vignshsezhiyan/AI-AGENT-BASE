"""
The agent loop.  This is the whole "brain" - read this file first.

    user message
         |
         v
    +---------+   tool_calls    +---------------+
    |   LLM   | --------------> | ToolRegistry  |   (timeouts, errors, audit, metrics)
    |         | <-------------- |  python / MCP |
    +---------+   tool results  +---------------+
         |   (repeat until the LLM answers without tools, max N iterations)
         v
    final answer

Human-in-the-loop: if the LLM wants a tool that needs approval, the loop PAUSES and returns
status="approval_required". The caller approves/rejects via the API and the loop RESUMES
exactly where it stopped.

Safety nets: whole-run timeout, max iterations, and rollback - if a run fails halfway,
the session history is restored so the conversation is never left half-written.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from app.agent_config import AgentConfig
from app.llm import LLMError, OpenAILLM
from app.observability.metrics import AGENT_ITERATIONS, AGENT_RUNS, APPROVALS, SECURITY_EVENTS, TOOL_CALLS
from app.security.errors import AgentTimeout, Conflict, InputRejected, NotFound, UpstreamError
from app.security.groundedness import GroundednessChecker
from app.security.guardrails import SAFETY_PREAMBLE, TOOL_RESULT_WARNING, OutputGuard, Redactor, detect_injection
from app.sessions import PendingApproval, Session
from app.tools.registry import ToolRegistry

logger = logging.getLogger("agent.run")
# Payload trace (user message, LLM output, tool arguments/results). DEBUG level -> only written to the
# per-request log files, never to stdout/app.log. Controlled by LOG_REQUEST_PAYLOADS.
trace = logging.getLogger("agent.trace")
_TRACE_CHARS = 4000  # cap per field so one huge tool result doesn't create a 10 MB log file


@dataclass
class AgentResult:
    status: Literal["completed", "approval_required", "max_iterations", "cancelled"]
    answer: str
    approval: PendingApproval | None = None
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # What the guardrails found/did, returned to the client, e.g.
    # {"grounded": false, "unsupported_claims": [...], "output_actions": ["pii_masked:card"]}
    checks: dict[str, Any] = field(default_factory=dict)


class Agent:
    def __init__(self, config: AgentConfig, llm: OpenAILLM, registry: ToolRegistry, redactor: Redactor,
                 judge: OpenAILLM | None = None):
        self.config = config
        self.limits = config.limits
        self.guards = config.guardrails
        self.llm = llm
        self.registry = registry
        self.redactor = redactor
        self.output_guard = OutputGuard(config.system_prompt, redactor, self.guards.mask_pii,
                                        self.guards.max_answer_chars)
        self.grounding = GroundednessChecker(judge if self.guards.groundedness != "off" else None)

    # ------------------------------------------------------------------ public API
    async def chat(self, session: Session, message: str) -> AgentResult:
        if session.pending:
            raise Conflict(f"Approval '{session.pending.id}' is pending. Approve or reject it first.")
        injection = self._check_user_input(message)

        async def run(result: AgentResult) -> AgentResult:
            if injection:
                result.checks["prompt_injection"] = injection
            _trim_history(session.messages, self.limits.max_history_messages - 1)
            logger.info("agent run started", extra={"event": "run_started", "history_messages": len(session.messages)})
            trace.debug("user message", extra={"event": "user_message", "user_message": self._clip(message)})
            session.messages.append({"role": "user", "content": message})
            return await self._loop(session, result)

        return await self._guarded(session, run)

    async def decide(self, session: Session, approval_id: str, approved: bool, reason: str = "") -> AgentResult:
        pending = session.pending
        if pending is None or pending.id != approval_id:
            raise NotFound("No such pending approval in this session.")

        async def run(result: AgentResult) -> AgentResult:
            session.pending = None
            APPROVALS.labels("approved" if approved else "rejected").inc()
            logger.info("approval decided", extra={"event": "approval_decided", "approval_id": approval_id, "approved": approved,
                                                   "tools": pending.needs_approval})
            if not approved and self.config.approval.on_reject == "end_chat":
                return self._end_chat(result, pending, reason)
            rejected = set() if approved else set(pending.needs_approval)
            await self._run_tools(session, pending.tool_calls, result, rejected, reason)
            return await self._loop(session, result)

        return await self._guarded(session, run)

    # ------------------------------------------------------------------ the loop
    async def _loop(self, session: Session, result: AgentResult) -> AgentResult:
        tools = self.registry.openai_schemas() or None

        for _ in range(self.limits.max_iterations):
            result.iterations += 1
            step = result.iterations
            logger.info(f"iteration {step} started", extra={"event": "iteration", "iteration": step,
                                                            "context_messages": len(session.messages) + 1})
            response = await self.llm.run(self._system_messages() + session.messages, tools=tools)
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens

            planned = [c.get("function", {}).get("name") for c in response.tool_calls]
            logger.info(f"iteration {step} llm decision", extra={"event": "llm_decision",
                "iteration": step, "decision": "call_tools" if planned else "final_answer", "tools": planned})
            trace.debug(f"iteration {step} llm output", extra={"event": "llm_output",
                "iteration": step, "reasoning": self._clip(response.reasoning), "content": self._clip(response.content),
                "tool_calls": [{"name": c.get("function", {}).get("name"),
                                "arguments": self._clip(c.get("function", {}).get("arguments"))}
                               for c in response.tool_calls]})

            # 1) No tool calls -> this is the final answer. Check it before anyone sees it.
            if not response.tool_calls:
                answer = (response.content or "").strip()
                if not answer:
                    logger.warning("llm returned empty answer", extra={"finish_reason": response.finish_reason})
                    answer = "Sorry, I could not produce an answer. Please rephrase or try again."
                answer = await self._check_answer(session, result, answer)
                session.messages.append({"role": "assistant", "content": answer})
                result.status, result.answer = "completed", answer
                trace.debug("final answer", extra={"event": "final_answer", "iteration": step, "answer": self._clip(answer)})
                return result

            # 2) The LLM wants tools. Record its request in the history first.
            calls = [_normalize_call(c) for c in response.tool_calls]
            session.messages.append({"role": "assistant", "content": response.content, "tool_calls": calls})

            # 3) Any risky tool? Pause and ask a human.
            gated = [c["function"]["name"] for c in calls if self.registry.requires_approval(c["function"]["name"])]
            if gated:
                session.pending = PendingApproval(id=uuid.uuid4().hex[:12], tool_calls=calls, needs_approval=gated)
                APPROVALS.labels("requested").inc()
                result.status, result.approval = "approval_required", session.pending
                result.answer = (response.content or "").strip() or f"Approval needed to run: {', '.join(gated)}."
                logger.info(f"iteration {step} paused for approval", extra={"event": "approval_paused",
                    "iteration": step, "approval_id": session.pending.id, "tools": gated})
                return result

            # 4) Run the tools and loop again so the LLM can read the results.
            await self._run_tools(session, calls, result)

        logger.warning("max iterations reached", extra={"iterations": result.iterations})
        result.status = "max_iterations"
        result.answer = "I could not finish this within the allowed number of steps. Please narrow down the request."
        session.messages.append({"role": "assistant", "content": result.answer})
        return result

    async def _run_tools(self, session: Session, calls: list[dict], result: AgentResult,
                         rejected: set[str] = frozenset(), reason: str = "") -> None:
        async def one(call: dict) -> str:
            name = call["function"]["name"]
            result.tools_used.append(name)
            if name in rejected:
                TOOL_CALLS.labels(name, "rejected").inc()
                return (f"The user REJECTED this action{': ' + reason if reason else ''}. "
                        "It was NOT executed. Do not retry it; tell the user and ask how to proceed.")
            outcome = await self.registry.execute(name, call["function"]["arguments"])
            content = self._check_tool_result(name, self.redactor.redact(outcome.content), result)
            trace.debug("tool result", extra={"event": "tool_result", "tool": name, "ok": outcome.ok,
                                              "arguments": self._clip(call["function"]["arguments"]),
                                              "result": self._clip(content)})
            return content

        # Independent tool calls from the same LLM turn run in parallel.
        contents = await asyncio.gather(*(one(c) for c in calls))
        for call, content in zip(calls, contents):
            session.messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})

    def _end_chat(self, result: AgentResult, pending: PendingApproval, reason: str) -> AgentResult:
        """Rejected with on_reject=end_chat: run NOTHING (not even the safe tools in the same batch),
        make no further LLM call, and tell the caller to close the session."""
        for name in pending.needs_approval:
            TOOL_CALLS.labels(name, "rejected").inc()
        result.status = "cancelled"
        result.answer = (f"Understood - I did not run {', '.join(pending.needs_approval)}. "
                         "Nothing was changed and this conversation is now closed. "
                         "Start a new chat if you need anything else.")
        logger.info("chat ended after rejection", extra={"event": "chat_ended", "approval_id": pending.id,
                                                         "tools": pending.needs_approval, "reason": reason})
        return result

    # ------------------------------------------------------------------ guardrails
    def _check_user_input(self, message: str) -> list[str]:
        """Prompt injection in the user's message: block (HTTP 400) or flag, per agent.yaml."""
        if self.guards.prompt_injection == "off":
            return []
        rules = detect_injection(message)
        if rules:
            SECURITY_EVENTS.labels("prompt_injection").inc()
            blocked = self.guards.prompt_injection == "block"
            self._guard_event("prompt_injection", f"{'BLOCKED' if blocked else 'flagged'} - {', '.join(rules)}",
                              detail=message, level=logging.WARNING)
            if blocked:
                raise InputRejected("Your message looks like an attempt to change the assistant's instructions, "
                                    "so it was not processed. Please rephrase your request.")
        return rules

    def _check_tool_result(self, tool: str, content: str, result: AgentResult) -> str:
        """Indirect injection: instructions hidden in data a tool returned (email body, CRM note, web page)."""
        if self.guards.tool_result_injection == "off":
            return content
        rules = detect_injection(content)
        if not rules:
            return content
        SECURITY_EVENTS.labels("tool_result_injection").inc()
        result.checks.setdefault("tool_result_injection", []).append({"tool": tool, "rules": rules})
        self._guard_event("tool_result_injection", f"flagged in {tool} - {', '.join(rules)}", level=logging.WARNING)
        return TOOL_RESULT_WARNING.format(rules=", ".join(rules)) + content

    async def _check_answer(self, session: Session, result: AgentResult, answer: str) -> str:
        """Groundedness (+ optional one-time revision), then the output guard."""
        if self.guards.groundedness != "off":
            question, evidence = _question_and_evidence(session.messages)
            verdict = await self.grounding.check(question, answer, evidence)

            if verdict.grounded is False and self.guards.groundedness == "revise":
                self._guard_event("groundedness", "NOT grounded - asking the agent to revise",
                                  detail=verdict.unsupported_claims, level=logging.WARNING)
                revised = await self._revise(session, result, answer, verdict.unsupported_claims)
                if revised:
                    answer = revised
                    result.checks["revised"] = True
                    verdict = await self.grounding.check(question, answer, evidence)

            result.checks["grounded"] = verdict.grounded
            if verdict.unsupported_claims:
                result.checks["unsupported_claims"] = verdict.unsupported_claims
            if verdict.note:
                result.checks["grounded_note"] = verdict.note
            outcome = {True: "grounded", False: "NOT grounded", None: f"not checked ({verdict.note})"}[verdict.grounded]
            self._guard_event("groundedness", outcome, detail=verdict.unsupported_claims or None,
                              level=logging.WARNING if verdict.grounded is False else logging.INFO)

        checked = self.output_guard.check(answer)
        if checked.actions:
            result.checks["output_actions"] = checked.actions
            self._guard_event("output", ", ".join(checked.actions), level=logging.WARNING)
        return checked.text

    async def _revise(self, session: Session, result: AgentResult, answer: str, claims: list[str]) -> str | None:
        """One extra LLM call (no tools) to rewrite the answer using only the evidence already gathered.
        The correction request is NOT stored in the session history."""
        feedback = ("[Automatic fact-check] Your answer contains claims that are not supported by the tool "
                    f"results: {json.dumps(claims, ensure_ascii=False)}. Rewrite the answer using ONLY facts from "
                    "the tool results above. If something is unknown, say so. Reply with the corrected answer only.")
        messages = self._system_messages() + session.messages + [
            {"role": "assistant", "content": answer}, {"role": "user", "content": feedback}]
        response = await self.llm.run(messages, tools=None)
        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens
        return (response.content or "").strip() or None

    def _guard_event(self, check: str, outcome: str, detail: Any = None, level: int = logging.INFO) -> None:
        logger.log(level, f"guardrail {check}: {outcome}", extra={"event": "guardrail", "check": check,
                                                                   "outcome": outcome})
        if detail:
            text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
            trace.debug(f"guardrail {check} detail", extra={"event": "guardrail_detail", "check": check,
                                                            "detail": self._clip(text)})

    # ------------------------------------------------------------------ safety wrapper
    async def _guarded(self, session: Session,
                       body: Callable[[AgentResult], Awaitable[AgentResult]]) -> AgentResult:
        """Timeout + rollback + metrics + one summary log line, around every run."""
        snapshot_len, snapshot_pending = len(session.messages), session.pending
        result = AgentResult(status="completed", answer="")
        start, outcome = time.perf_counter(), "error"
        try:
            async with asyncio.timeout(self.limits.request_timeout_seconds):
                await body(result)
            outcome = result.status
            return result
        except TimeoutError:
            outcome = "timeout"
            raise AgentTimeout() from None
        except LLMError as exc:
            raise UpstreamError() from exc
        finally:
            if outcome in ("error", "timeout"):  # also covers client disconnect (CancelledError)
                del session.messages[snapshot_len:]
                session.pending = snapshot_pending
            AGENT_RUNS.labels(outcome).inc()
            AGENT_ITERATIONS.observe(result.iterations)
            logger.info("agent run finished", extra={"event": "run_finished",
                "status": outcome, "iterations": result.iterations, "tools_used": result.tools_used,
                "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                "duration_ms": round((time.perf_counter() - start) * 1000)})

    def _clip(self, text: str | None) -> str | None:
        """Redact + shorten a payload before it is written to a log file."""
        if not text:
            return text
        text = self.redactor.redact(text)
        return text if len(text) <= _TRACE_CHARS else text[:_TRACE_CHARS] + f"...[+{len(text) - _TRACE_CHARS} chars]"

    def _system_messages(self) -> list[dict[str, Any]]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")
        return [{"role": "system",
                 "content": f"{SAFETY_PREAMBLE}\nCurrent date (UTC): {today}\n\n{self.config.system_prompt}"}]


def _question_and_evidence(messages: list[dict]) -> tuple[str, list[str]]:
    """The latest user message, and every tool result produced after it (this run's evidence)."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    question = messages[last_user]["content"] if last_user >= 0 else ""
    evidence = [m["content"] for m in messages[last_user + 1:] if m["role"] == "tool"]
    return question, evidence


def _normalize_call(call: dict) -> dict:
    """Keep only the fields the chat API accepts back, and make sure there is an id."""
    fn = call.get("function") or {}
    return {"id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments") or "{}"}}


def _trim_history(messages: list[dict], keep: int) -> None:
    """Drop the oldest turns so at most `keep` messages remain.
    Always cuts at a user message, so a tool call is never separated from its result."""
    if len(messages) <= keep:
        return
    for i in range(len(messages) - keep, len(messages)):
        if messages[i]["role"] == "user":
            del messages[:i]
            return
    del messages[:]  # a single turn longer than the limit - start fresh
