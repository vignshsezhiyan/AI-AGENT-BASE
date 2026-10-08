"""
Prometheus metrics, exposed at GET /metrics.

What you can answer on a Grafana dashboard with these:
  - Is the agent up, how busy is it, how slow is it?      -> HTTP_*
  - Is the LLM slow / failing / how many tokens are used? -> LLM_*
  - Which tools fail or time out?                         -> TOOL_*
  - Are runs hitting the iteration limit?                 -> AGENT_*
  - Is someone attacking / abusing it?                    -> SECURITY_EVENTS
"""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "agent_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "agent_http_request_duration_seconds", "HTTP request latency", ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300))

LLM_REQUESTS = Counter(
    "agent_llm_requests_total", "LLM calls", ["model", "status"])
LLM_LATENCY = Histogram(
    "agent_llm_request_duration_seconds", "LLM call latency", ["model"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120))
LLM_TOKENS = Counter(
    "agent_llm_tokens_total", "Tokens used", ["model", "type"])  # type = prompt | completion

TOOL_CALLS = Counter(
    "agent_tool_calls_total", "Tool calls", ["tool", "status"])  # ok | error | timeout | rejected
TOOL_LATENCY = Histogram(
    "agent_tool_duration_seconds", "Tool latency", ["tool"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))

AGENT_RUNS = Counter(
    "agent_runs_total", "Agent runs by outcome", ["status"])
AGENT_ITERATIONS = Histogram(
    "agent_run_iterations", "LLM round trips per run", buckets=(1, 2, 3, 4, 5, 6, 8, 10, 15, 20))
APPROVALS = Counter(
    "agent_approvals_total", "Human approval decisions", ["decision"])

SECURITY_EVENTS = Counter(
    "agent_security_events_total", "Auth failures, rate limits, blocked input, redactions", ["event"])

AGENT_INFO = Gauge("agent_info", "Static info about the running agent", ["agent", "model"])
MCP_SERVER_UP = Gauge("agent_mcp_server_up", "1 if the MCP server connected", ["server"])
ACTIVE_SESSIONS = Gauge("agent_active_sessions", "Sessions held in memory")
