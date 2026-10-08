import sys
from pathlib import Path

import pytest

from app.agent_config import MCPServer
from app.tools.mcp import MCPConnection
from app.tools.registry import ToolRegistry

SERVER = Path(__file__).with_name("fake_mcp_server.py")


@pytest.fixture
async def registry():
    conn = MCPConnection(MCPServer(name="crm", transport="stdio", command=sys.executable,
                                   args=[str(SERVER)], exclude=["internal_*"]))
    reg = ToolRegistry(timeout_seconds=10, max_result_chars=5000, approval_patterns=[])
    reg.register_many(await conn.connect())
    yield reg
    await conn.close()


async def test_discovery_prefix_and_filters(registry):
    assert registry.names() == ["crm_broken", "crm_delete_lead", "crm_get_lead"]  # internal_admin excluded


async def test_destructive_tools_need_approval(registry):
    assert registry.requires_approval("crm_delete_lead")
    assert not registry.requires_approval("crm_get_lead")


async def test_call_and_error(registry):
    ok = await registry.execute("crm_get_lead", '{"lead_id": "L1"}')
    assert ok.ok and "Asha" in ok.content
    failed = await registry.execute("crm_broken", "{}")
    assert not failed.ok and "Error" in failed.content
