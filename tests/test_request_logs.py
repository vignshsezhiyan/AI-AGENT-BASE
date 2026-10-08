import logging

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeLLM, FakeRuntime, answer, calls, make_settings, tool_call

KEY = {"X-API-Key": "test-key-0123456789abcdef"}


def run_one_request(tmp_path, *responses, message="(2+3)*4?", **settings):
    llm = FakeLLM(*responses)
    s = make_settings(log_dir=str(tmp_path), log_request_files=True, **settings)
    try:
        with TestClient(create_app(s, FakeRuntime(s, llm))) as client:
            r = client.post("/v1/chat", json={"message": message}, headers={**KEY, "X-Request-ID": "req-test-0001"})
            assert r.status_code == 200
    finally:
        logging.getLogger().handlers.clear()  # release file handles for the next tests
    files = list((tmp_path / "requests").glob("*/req-test-0001.log"))
    assert len(files) == 1
    return (tmp_path / "app.log").read_text(), files[0].read_text()


def test_request_is_written_as_one_readable_block(tmp_path):
    app_log, request_file = run_one_request(
        tmp_path,
        calls(tool_call("add_numbers", {"a": 2, "b": 3})),
        calls(tool_call("multiply_numbers", {"a": 5, "b": 4})),
        answer("The answer is 20."),
        log_request_payloads=True)

    # app.log = overall trace: one line per event, tagged with the request id, no payloads
    trace_lines = [line for line in app_log.splitlines() if "req=req-test-0001" in line]
    for event in ("iteration 1 started", "llm decision", "tool call", "agent run finished", "http request"):
        assert any(event in line for line in trace_lines), event
    assert "(2+3)*4?" not in app_log and "STEP 1" not in app_log

    block = request_file
    expected_in_order = [
        "REQUEST  req-test-0001", "POST /v1/chat  ->  200", "caller   tests",
        "USER", "(2+3)*4?",
        "STEP 1", "LLM", "decided: call add_numbers", "TOOL      add_numbers  ->  OK",
        'args    {"a": 2, "b": 3}', 'result  {"result": 5}',
        "STEP 2", "decided: call multiply_numbers", 'result  {"result": 20}',
        "STEP 3", "decided: final answer",
        "ANSWER", "The answer is 20.",
        "SUMMARY  completed | 3 steps | 2 tool calls",
    ]
    position = 0
    for text in expected_in_order:
        found = block.find(text, position)
        assert found >= 0, f"missing or out of order: {text!r}\n{block}"
        position = found


def test_payloads_can_be_turned_off(tmp_path):
    _, block = run_one_request(
        tmp_path, calls(tool_call("add_numbers", {"a": 2, "b": 3})), answer("Five."),
        message="secret customer question", log_request_payloads=False)
    assert "STEP 1" in block and "add_numbers  ->  OK" in block and "SUMMARY" in block
    assert "secret customer question" not in block and "Five." not in block
