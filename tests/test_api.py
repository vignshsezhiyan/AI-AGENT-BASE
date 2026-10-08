import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeLLM, FakeRuntime, answer, calls, make_settings, tool_call

KEY = {"X-API-Key": "test-key-0123456789abcdef"}


def client_with(*responses, **settings) -> tuple[TestClient, FakeLLM]:
    llm = FakeLLM(*responses)
    s = make_settings(**settings)
    return TestClient(create_app(s, FakeRuntime(s, llm))), llm


def test_health_and_ready():
    client, _ = client_with()
    with client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200


def test_auth_required():
    client, _ = client_with()
    with client:
        r = client.post("/v1/chat", json={"message": "hi"})
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"
        assert client.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_chat_and_session_continuation():
    client, llm = client_with(answer("first"), answer("second"))
    with client:
        r1 = client.post("/v1/chat", json={"message": "one"}, headers=KEY).json()
        assert r1["answer"] == "first" and r1["status"] == "completed" and r1["request_id"]
        r2 = client.post("/v1/chat", json={"message": "two", "session_id": r1["session_id"]}, headers=KEY).json()
        assert r2["answer"] == "second"
        assert [m["role"] for m in llm.requests[1]] == ["system", "user", "assistant", "user"]


def test_approval_over_http():
    client, _ = client_with(
        calls(tool_call("create_support_ticket", {"title": "Printer on fire", "priority": "high"})),
        answer("Ticket created."))
    with client:
        r = client.post("/v1/chat", json={"message": "open a ticket"}, headers=KEY).json()
        assert r["status"] == "approval_required"
        assert "Printer on fire" in r["approval"]["arguments"]["create_support_ticket"]
        url = f"/v1/sessions/{r['session_id']}/approvals/{r['approval']['approval_id']}"
        done = client.post(url, json={"approve": True}, headers=KEY).json()
        assert done["answer"] == "Ticket created."


def test_sessions_are_isolated_between_callers():
    client, _ = client_with(answer("hi"), api_keys="a:aaaaaaaaaaaaaaaaaaaa,b:bbbbbbbbbbbbbbbbbbbb")
    with client:
        sid = client.post("/v1/chat", json={"message": "x"}, headers={"X-API-Key": "a" * 20}).json()["session_id"]
        r = client.delete(f"/v1/sessions/{sid}", headers={"X-API-Key": "b" * 20})
        assert r.status_code == 404


def test_rate_limit():
    client, _ = client_with(answer("a"), answer("b"), rate_limit_per_minute=1)
    with client:
        assert client.post("/v1/chat", json={"message": "x"}, headers=KEY).status_code == 200
        assert client.post("/v1/chat", json={"message": "x"}, headers=KEY).status_code == 429


def test_input_too_long():
    client, _ = client_with(max_input_chars=10)
    with client:
        r = client.post("/v1/chat", json={"message": "x" * 50}, headers=KEY)
        assert r.status_code == 400 and r.json()["error"] == "input_rejected"


def test_llm_down_returns_clean_502():
    from app.llm import LLMError
    client, _ = client_with(LLMError("connection refused to 10.0.0.1"))
    with client:
        r = client.post("/v1/chat", json={"message": "x"}, headers=KEY)
        assert r.status_code == 502
        assert "10.0.0.1" not in r.text  # internal details never leak


def test_request_id_and_metrics():
    client, _ = client_with(answer("a"))
    with client:
        r = client.post("/v1/chat", json={"message": "x"}, headers={**KEY, "X-Request-ID": "trace-12345678"})
        assert r.headers["X-Request-ID"] == "trace-12345678" == r.json()["request_id"]
        metrics = client.get("/metrics").text
        assert "agent_llm_tokens_total" in metrics and "agent_runs_total" in metrics


def test_production_requires_api_keys():
    with pytest.raises(ValueError):
        make_settings(app_env="production", api_keys="")


def test_follow_up_tool_approve_writes_and_reject_closes_chat(tmp_path, monkeypatch):
    import agent.tools as agent_tools
    monkeypatch.setattr(agent_tools, "_FOLLOW_UPS_FILE", tmp_path / "follow_ups.jsonl")
    args = {"contact_name": "Emma", "phone": "9876543210", "call_date": "2099-01-15", "purpose": "Renewal discussion"}

    # approve -> tool runs, record saved, agent continues
    client, _ = client_with(calls(tool_call("schedule_follow_up_call", args)), answer("Scheduled for 15 Jan."))
    with client:
        r = client.post("/v1/chat", json={"message": "schedule a call with Emma"}, headers=KEY).json()
        assert r["status"] == "approval_required" and r["approval"]["tools"] == ["schedule_follow_up_call"]
        url = f"/v1/sessions/{r['session_id']}/approvals/{r['approval']['approval_id']}"
        done = client.post(url, json={"approve": True}, headers=KEY).json()
        assert done["status"] == "completed" and done["session_closed"] is False
    saved = (tmp_path / "follow_ups.jsonl").read_text()
    assert '"contact_name": "Emma"' in saved

    # reject -> nothing saved, chat ended, session gone
    (tmp_path / "follow_ups.jsonl").unlink()
    client, llm = client_with(calls(tool_call("schedule_follow_up_call", args)), answer("Hello!"))
    with client:
        r = client.post("/v1/chat", json={"message": "schedule a call with Emma"}, headers=KEY).json()
        url = f"/v1/sessions/{r['session_id']}/approvals/{r['approval']['approval_id']}"
        done = client.post(url, json={"approve": False, "reason": "not now"}, headers=KEY).json()
        assert done["status"] == "cancelled" and done["session_closed"] is True
        assert len(llm.requests) == 1  # LLM was not called again
        client.post("/v1/chat", json={"message": "hi", "session_id": r["session_id"]}, headers=KEY)
        assert [m["role"] for m in llm.requests[1]] == ["system", "user"]  # old conversation is gone
    assert not (tmp_path / "follow_ups.jsonl").exists()
