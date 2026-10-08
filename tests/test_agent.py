import pytest

from app.llm import LLMError
from app.security.errors import Conflict, UpstreamError
from tests.conftest import FakeLLM, answer, calls, make_agent, tool_call


async def test_plain_answer(session):
    agent = make_agent(FakeLLM(answer("Hello!")))
    result = await agent.chat(session, "hi")
    assert result.status == "completed" and result.answer == "Hello!"
    assert [m["role"] for m in session.messages] == ["user", "assistant"]


async def test_tool_call_then_answer(session):
    llm = FakeLLM(calls(tool_call("add", {"a": 2, "b": 3})), answer("It is 5."))
    result = await make_agent(llm).chat(session, "2+3?")
    assert result.answer == "It is 5."
    assert result.tools_used == ["add"]
    assert llm.requests[1][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}


async def test_bad_arguments_are_sent_back_to_llm(session):
    llm = FakeLLM(calls(tool_call("add", {"a": "two", "b": 3})), answer("sorry"))
    await make_agent(llm).chat(session, "add")
    tool_msg = llm.requests[1][-1]["content"]
    assert "invalid arguments" in tool_msg and "a:" in tool_msg


async def test_unknown_tool_does_not_crash(session):
    llm = FakeLLM(calls(tool_call("rm_rf", {})), answer("ok"))
    await make_agent(llm).chat(session, "x")
    assert "unknown tool" in llm.requests[1][-1]["content"]


async def test_approval_approve_flow(session):
    llm = FakeLLM(calls(tool_call("delete_record", {"record_id": "42"})), answer("Deleted 42."))
    agent = make_agent(llm)

    paused = await agent.chat(session, "delete 42")
    assert paused.status == "approval_required"
    assert paused.approval.needs_approval == ["delete_record"]
    assert len(llm.requests) == 1  # nothing executed yet

    with pytest.raises(Conflict):  # can't chat while an approval is pending
        await agent.chat(session, "hello?")

    done = await agent.decide(session, paused.approval.id, approved=True)
    assert done.answer == "Deleted 42."
    assert '"deleted": "42"' in llm.requests[1][-1]["content"]


async def test_approval_reject_flow(session):
    llm = FakeLLM(calls(tool_call("delete_record", {"record_id": "42"})), answer("OK, not deleted."))
    agent = make_agent(llm)
    paused = await agent.chat(session, "delete 42")
    await agent.decide(session, paused.approval.id, approved=False, reason="wrong record")
    assert "REJECTED" in llm.requests[1][-1]["content"] and "wrong record" in llm.requests[1][-1]["content"]


async def test_max_iterations(session):
    llm = FakeLLM(*[calls(tool_call("add", {"a": 1, "b": 1}, f"c{i}")) for i in range(3)])
    result = await make_agent(llm, max_iterations=3).chat(session, "loop forever")
    assert result.status == "max_iterations" and result.iterations == 3


async def test_llm_failure_rolls_back_history(session):
    llm = FakeLLM(calls(tool_call("add", {"a": 1, "b": 1})), LLMError("down"))
    with pytest.raises(UpstreamError):
        await make_agent(llm).chat(session, "x")
    assert session.messages == []  # nothing half-written


async def test_history_is_trimmed_at_turn_boundaries(session):
    llm = FakeLLM(*[answer(f"a{i}") for i in range(10)])
    agent = make_agent(llm, max_history_messages=4)
    for i in range(10):
        await agent.chat(session, f"q{i}")
    assert len(session.messages) <= 5
    assert session.messages[0]["role"] == "user"


async def test_secrets_are_redacted_from_answers(session, monkeypatch):
    monkeypatch.setenv("CRM_API_TOKEN", "supersecretvalue123")
    from app.security.guardrails import Redactor
    agent = make_agent(FakeLLM(answer("token is supersecretvalue123")))
    agent.redactor = Redactor()
    result = await agent.chat(session, "x")
    assert "supersecretvalue123" not in result.answer


async def test_reject_with_end_chat_runs_nothing_and_stops(session):
    llm = FakeLLM(calls(tool_call("delete_record", {"record_id": "42"}, "c1"), tool_call("add", {"a": 1, "b": 1}, "c2")))
    agent = make_agent(llm)
    agent.config.approval.on_reject = "end_chat"
    paused = await agent.chat(session, "delete 42")
    done = await agent.decide(session, paused.approval.id, approved=False)
    assert done.status == "cancelled" and "did not run delete_record" in done.answer
    assert len(llm.requests) == 1          # no LLM call after the rejection
    assert done.tools_used == []           # not even the safe tool in the same batch ran
