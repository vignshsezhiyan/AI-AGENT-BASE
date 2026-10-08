"""
YOUR TOOLS GO HERE.

Every function decorated with @tool is given to the LLM automatically. Rules:
  1. The docstring is what the LLM reads to decide WHEN to use the tool - make it specific.
  2. Every parameter needs a type hint. Use Annotated[..., Field(description=...)] to explain it.
  3. Return a dict / list / str. Keep it small - only fields the LLM needs.
  4. For expected failures raise ToolError("clear message") - the LLM reads it and can react.
  5. Anything that CHANGES data (create/update/delete/send) -> @tool(requires_approval=True).
  6. Secrets: read them from env (os.environ["..."]); never hard-code, never return them.

The tools below are examples - delete them when you build a real agent.
"""

import json
import math
import os
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from app.tools.base import ToolError, tool


@tool
def get_current_time(
    timezone: Annotated[str, Field(description="IANA timezone, e.g. 'Asia/Kolkata' or 'Europe/London'")] = "UTC",
) -> dict:
    """Get the current date and time in a timezone. Use for any question about 'now', 'today' or deadlines."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        raise ToolError(f"Unknown timezone '{timezone}'. Use an IANA name like 'Asia/Kolkata'.") from None
    return {"timezone": timezone, "datetime": now.isoformat(timespec="seconds"), "weekday": now.strftime("%A")}


@tool(requires_approval=True)
async def create_support_ticket(
    title: Annotated[str, Field(min_length=5, max_length=120, description="One-line summary of the issue")],
    priority: Literal["low", "medium", "high"] = "medium",
) -> dict:
    """Create a support ticket. EXAMPLE ONLY - it does not create anything real.
    Shows how a data-changing tool requires human approval before it runs."""
    # Real version: call your ticketing API here, e.g. with httpx.AsyncClient(...)
    return {"ticket_id": "EXAMPLE-1001", "title": title, "priority": priority, "note": "example tool, nothing created"}


# --- Calculator tools ---------------------------------------------------------------------------
# LLMs are bad at exact arithmetic; tools make it exact. One operation per tool, so a question like
# "(12 + 8) * 3 ^ 2" makes the agent loop several times - handy for seeing each iteration in the logs.

Number = Annotated[float, Field(description="A number, e.g. 42 or 3.14")]


def _result(value: float) -> dict:
    if math.isnan(value) or math.isinf(value):
        raise ToolError("Result is not a finite number (too large or undefined).")
    return {"result": int(value) if value.is_integer() and abs(value) < 2**53 else value}


@tool
def add_numbers(a: Number, b: Number) -> dict:
    """Add two numbers: a + b."""
    return _result(a + b)


@tool
def subtract_numbers(a: Number, b: Number) -> dict:
    """Subtract two numbers: a - b."""
    return _result(a - b)


@tool
def multiply_numbers(a: Number, b: Number) -> dict:
    """Multiply two numbers: a * b."""
    return _result(a * b)


@tool
def divide_numbers(a: Number, b: Number) -> dict:
    """Divide two numbers: a / b."""
    if b == 0:
        raise ToolError("Cannot divide by zero.")
    return _result(a / b)


@tool
def power_of_number(
    base: Number,
    exponent: Annotated[float, Field(ge=-1000, le=1000, description="The exponent, between -1000 and 1000")],
) -> dict:
    """Raise a number to a power: base ^ exponent (e.g. 2 ^ 10 = 1024)."""
    if base == 0 and exponent < 0:
        raise ToolError("0 cannot be raised to a negative power.")
    try:
        value = math.pow(base, exponent)  # float math: huge results raise OverflowError instead of hanging
    except OverflowError:
        raise ToolError("Result is too large.") from None
    except ValueError:
        raise ToolError("Negative base with a fractional exponent has no real result.") from None
    return _result(value)


# --- Human-in-the-loop example: a tool with a REAL side effect -------------------------------------
# requires_approval=True -> the agent PAUSES before running it and the API returns
# status="approval_required" with the exact arguments. A human then approves or rejects:
#   approve -> the tool runs and the agent continues
#   reject  -> with approval.on_reject: end_chat (agent.yaml) nothing runs and the chat is closed

_FOLLOW_UPS_FILE = Path(os.getenv("FOLLOW_UPS_FILE", "data/follow_ups.jsonl"))
_follow_ups_lock = threading.Lock()


@tool(requires_approval=True)
def schedule_follow_up_call(
    contact_name: Annotated[str, Field(min_length=1, max_length=100, description="Contact's name, e.g. 'Emma'")],
    phone: Annotated[str, Field(pattern=r"^\+?[0-9 ()-]{7,20}$", description="Phone number from the contact record")],
    call_date: Annotated[date, Field(description="Date of the call, YYYY-MM-DD. Today or later.")],
    purpose: Annotated[str, Field(min_length=3, max_length=300, description="Why we are calling, one sentence")],
) -> dict:
    """Schedule a follow-up phone call with a contact (saved to the follow-up list the sales team works from).
    Get the phone number from the contact record first - never guess it."""
    if call_date < date.today():
        raise ToolError(f"call_date {call_date} is in the past. Use today ({date.today()}) or a later date.")

    record = {
        "follow_up_id": f"FU-{uuid.uuid4().hex[:8].upper()}",
        "contact_name": contact_name,
        "phone": phone,
        "call_date": call_date.isoformat(),
        "purpose": purpose,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _follow_ups_lock:  # parallel requests must not interleave lines
        _FOLLOW_UPS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _FOLLOW_UPS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return {"status": "scheduled", **record}
