import asyncio
from typing import Annotated, Literal

import pytest
from pydantic import Field

from app.tools.base import tool
from app.tools.registry import ToolRegistry


@tool
async def search(query: Annotated[str, Field(description="What to look for")],
                 limit: int = 5, kind: Literal["lead", "contact"] = "lead") -> list:
    """Search CRM records."""
    return [query] * limit


@tool
async def slow() -> str:
    """Sleeps."""
    await asyncio.sleep(5)
    return "done"


def registry(**kw) -> ToolRegistry:
    r = ToolRegistry(timeout_seconds=kw.get("timeout", 1), max_result_chars=kw.get("max_chars", 1000),
                     approval_patterns=kw.get("patterns", []))
    r.register(search.__agent_tool__)
    r.register(slow.__agent_tool__)
    return r


def test_schema_is_generated_from_type_hints():
    params = search.__agent_tool__.parameters
    assert params["required"] == ["query"]
    assert params["properties"]["query"] == {"type": "string", "description": "What to look for"}
    assert params["properties"]["kind"]["enum"] == ["lead", "contact"]
    assert "title" not in params


def test_tool_requires_docstring_and_type_hints():
    with pytest.raises(TypeError):
        @tool
        def no_doc(x: int): ...
    with pytest.raises(TypeError):
        @tool
        def no_hint(x):
            """doc"""


async def test_extra_arguments_rejected():
    result = await registry().execute("search", '{"query": "x", "admin": true}')
    assert not result.ok and "admin" in result.content


async def test_invalid_json_arguments():
    result = await registry().execute("search", "{not json")
    assert not result.ok and "not valid JSON" in result.content


async def test_timeout():
    result = await registry(timeout=0.05).execute("slow", "{}")
    assert not result.ok and "timed out" in result.content


async def test_truncation():
    result = await registry(max_chars=500).execute("search", {"query": "x" * 100, "limit": 20})
    assert result.ok and "[truncated" in result.content


def test_approval_patterns():
    r = registry(patterns=["sea*"])
    assert r.requires_approval("search") and not r.requires_approval("slow")


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        registry().register(search.__agent_tool__)
