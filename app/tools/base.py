"""
The @tool decorator: turns a normal Python function into an LLM tool.

    @tool
    async def get_customer(customer_id: str, include_orders: bool = False) -> dict:
        '''Fetch a customer from the CRM by ID.'''
        ...

From that we automatically get:
  - the tool name               -> function name
  - the description for the LLM -> docstring
  - the JSON schema for the LLM -> type hints (+ Field(description=...) if you want)
  - argument validation         -> bad arguments from the LLM are rejected *before*
                                   your function runs, and the error is sent back to the
                                   LLM so it can fix the call itself.
"""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import ConfigDict, ValidationError, create_model


class ToolError(Exception):
    """Raise this from a tool for EXPECTED failures (not found, invalid input...).
    The message goes back to the LLM and is logged as a warning without a stack trace."""


class ToolArgumentError(ToolError):
    """The LLM sent arguments that don't match the tool's schema."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]                       # JSON schema of the arguments
    handler: Callable[[dict[str, Any]], Awaitable[Any]]
    requires_approval: bool = False
    source: str = "python"                           # "python" or "mcp:<server>"

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def tool(fn: Callable | None = None, *, name: str | None = None, requires_approval: bool = False):
    """Decorator. Use as @tool or @tool(requires_approval=True)."""

    def wrap(func: Callable) -> Callable:
        func.__agent_tool__ = _build_tool(func, name or func.__name__, requires_approval)
        return func

    return wrap(fn) if fn is not None else wrap


def _build_tool(func: Callable, name: str, requires_approval: bool) -> Tool:
    description = inspect.getdoc(func)
    if not description:
        raise TypeError(f"Tool '{name}' needs a docstring - the LLM uses it to decide when to call the tool")

    fields: dict[str, Any] = {}
    for pname, param in inspect.signature(func).parameters.items():
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool '{name}': parameter '{pname}' needs a type hint")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (param.annotation, default)

    # extra="forbid": the LLM cannot smuggle in arguments the function doesn't declare
    args_model = create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)
    is_async = inspect.iscoroutinefunction(func)

    async def handler(arguments: dict[str, Any]) -> Any:
        try:
            validated = args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())
            raise ToolArgumentError(problems) from None
        kwargs = {k: getattr(validated, k) for k in fields}
        if is_async:
            return await func(**kwargs)
        return await asyncio.to_thread(func, **kwargs)  # sync tools don't block the event loop

    return Tool(name=name, description=description, parameters=_clean_schema(args_model.model_json_schema()),
                handler=handler, requires_approval=requires_approval)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's auto-generated 'title' keys - they waste tokens and confuse some models."""

    def walk(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, in_properties=(k in ("properties", "$defs")))
                    for k, v in node.items() if in_properties or k != "title"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def collect_tools(module) -> list[Tool]:
    """Find every @tool-decorated function in a module (e.g. agent/tools.py)."""
    return [obj.__agent_tool__ for obj in vars(module).values() if hasattr(obj, "__agent_tool__")]
