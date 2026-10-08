"""A tiny MCP server used by test_mcp.py (runs as a subprocess over stdio)."""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("fake-crm")


@server.tool()
def get_lead(lead_id: str) -> dict:
    """Get a lead by id."""
    return {"id": lead_id, "name": "Asha", "stage": "Qualified"}


@server.tool(annotations=ToolAnnotations(destructive_hint=True))
def delete_lead(lead_id: str) -> str:
    """Delete a lead."""
    return "deleted"


@server.tool()
def internal_admin() -> str:
    """Must be hidden by the exclude filter."""
    return "secret"


@server.tool()
def broken() -> str:
    """Always fails."""
    raise RuntimeError("CRM exploded")


if __name__ == "__main__":
    server.run("stdio")
