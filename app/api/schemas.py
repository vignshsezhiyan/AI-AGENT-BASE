"""Request / response models of the public API (also shown in the /docs Swagger UI)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SESSION_ID = r"^[A-Za-z0-9_\-]{1,64}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000, description="The user's message")
    session_id: str | None = Field(None, pattern=SESSION_ID,
                                   description="Omit to start a new conversation; send it back to continue one")


class ApprovalDecision(BaseModel):
    approve: bool
    reason: str = Field("", max_length=1000, description="Optional - passed to the agent when rejecting")


class ApprovalInfo(BaseModel):
    approval_id: str
    tools: list[str] = Field(description="Tools waiting for approval")
    arguments: dict[str, str] = Field(description="tool name -> raw JSON arguments the agent wants to use")


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    status: Literal["completed", "approval_required", "max_iterations", "cancelled"]
    answer: str
    session_closed: bool = Field(False, description="True when the conversation was ended (e.g. approval "
                                                    "rejected with on_reject=end_chat). Start a new session.")
    approval: ApprovalInfo | None = None
    tools_used: list[str] = []
    usage: dict[str, int] = {}
    checks: dict[str, Any] = Field({}, description="Guardrail results: prompt_injection, tool_result_injection, "
                                                    "grounded, unsupported_claims, revised, output_actions")


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
