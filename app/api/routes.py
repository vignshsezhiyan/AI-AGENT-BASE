"""
HTTP endpoints.

  GET    /health                                     liveness  (is the process alive?)        - no auth
  GET    /ready                                      readiness (agent + MCP servers ready?)   - no auth
  GET    /metrics                                    Prometheus metrics                        - API key in production
  GET    /v1/agent                                   agent name + tools it can use
  POST   /v1/chat                                    send a message (new or existing session)
  POST   /v1/sessions/{sid}/approvals/{aid}          approve / reject a pending risky tool call
  DELETE /v1/sessions/{sid}                          forget a conversation
"""

import re

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.agent import Agent, AgentResult
from app.api.schemas import SESSION_ID, ApprovalDecision, ApprovalInfo, ChatRequest, ChatResponse
from app.observability.logging import request_id_var, session_id_var
from app.security.auth import api_key_header, require_caller, resolve_caller
from app.security.errors import Conflict, NotFound
from app.security.guardrails import validate_input
from app.sessions import Session

public = APIRouter()
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_caller)])


# ---------------------------------------------------------------- probes & metrics
@public.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}


@public.get("/ready", tags=["ops"])
async def ready(request: Request):
    runtime = request.app.state.runtime
    body = {"status": "ready" if runtime.agent else "starting", "mcp_servers": runtime.mcp_status}
    return JSONResponse(body, status_code=200 if runtime.agent else 503)


@public.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics(request: Request):
    settings = request.app.state.settings
    if not settings.metrics_enabled:
        return Response(status_code=404)
    if settings.app_env == "production":  # on a public host there is no ingress to hide it behind
        resolve_caller(await api_key_header(request), request.app.state.api_key_map)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------- agent API
def get_agent(request: Request) -> Agent:
    return request.app.state.runtime.agent


@v1.get("/agent", tags=["agent"])
async def agent_info(agent: Agent = Depends(get_agent)):
    return {"name": agent.config.name, "description": agent.config.description, "tools": agent.registry.describe()}


@v1.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(body: ChatRequest, request: Request, caller: str = Depends(require_caller),
               agent: Agent = Depends(get_agent)):
    message = validate_input(body.message, request.app.state.settings.max_input_chars)
    session = request.app.state.sessions.get_or_create(caller, body.session_id)
    limiter = request.app.state.rate_limiter
    async with limiter.run_slot(caller), _exclusive(session):
        result = await agent.chat(session, message)
    limiter.add_tokens(caller, result.prompt_tokens + result.completion_tokens)
    return _response(session, result, agent)


@v1.post("/sessions/{session_id}/approvals/{approval_id}", response_model=ChatResponse, tags=["agent"])
async def decide(session_id: str, approval_id: str, body: ApprovalDecision, request: Request,
                 caller: str = Depends(require_caller), agent: Agent = Depends(get_agent)):
    session = _get_session(request, caller, session_id)
    limiter = request.app.state.rate_limiter
    async with limiter.run_slot(caller), _exclusive(session):
        result = await agent.decide(session, approval_id, body.approve, body.reason)
    limiter.add_tokens(caller, result.prompt_tokens + result.completion_tokens)
    if result.status == "cancelled":  # rejected with on_reject=end_chat -> the conversation is over
        request.app.state.sessions.delete(caller, session_id)
    return _response(session, result, agent)


@v1.delete("/sessions/{session_id}", status_code=204, tags=["agent"])
async def delete_session(session_id: str, request: Request, caller: str = Depends(require_caller)):
    _get_session(request, caller, session_id)
    request.app.state.sessions.delete(caller, session_id)


# ---------------------------------------------------------------- helpers
def _get_session(request: Request, caller: str, session_id: str) -> Session:
    session = re.fullmatch(SESSION_ID, session_id) and request.app.state.sessions.get(caller, session_id)
    if not session:
        raise NotFound("Session not found or expired.")
    return session


class _exclusive:
    """One request at a time per session - two parallel messages would corrupt the history."""

    def __init__(self, session: Session):
        self.session = session

    async def __aenter__(self):
        if self.session.lock.locked():
            raise Conflict("This session is already processing a message. Wait for it to finish.")
        await self.session.lock.acquire()
        session_id_var.set(self.session.id)

    async def __aexit__(self, *exc):
        self.session.lock.release()


def _response(session: Session, result: AgentResult, agent: Agent) -> ChatResponse:
    approval = None
    if result.approval:
        approval = ApprovalInfo(
            approval_id=result.approval.id,
            tools=result.approval.needs_approval,
            arguments={c["function"]["name"]: agent.redactor.redact(c["function"]["arguments"])
                       for c in result.approval.tool_calls if c["function"]["name"] in result.approval.needs_approval},
        )
    return ChatResponse(request_id=request_id_var.get(), session_id=session.id, status=result.status,
                        answer=result.answer, session_closed=result.status == "cancelled", approval=approval, tools_used=result.tools_used,
                        usage={"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                               "iterations": result.iterations},
                        checks=result.checks)
