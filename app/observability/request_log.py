"""
Human-readable PER-REQUEST logs:   logs/requests/<YYYY-MM-DD>/<request_id>.log

While a request runs, its log records are collected in memory. When it finishes, they are written
as ONE block - user message, every step (thinking -> LLM -> tools), answer, summary:

    ==============================================================================
    REQUEST  demo-multistep-001
      when     2026-10-08 15:23:41
      call     POST /v1/chat  ->  200   (6.96s)
      caller   dev   |   session 31b1a1a67b41...
    ------------------------------------------------------------------------------
    USER
      Calculate ((15 + 9) * 3 - 12) / 4

    STEP 1                                                              +0.00s
      THINKING  We need to compute using tools stepwise.
      LLM       1.69s | tokens 807 in / 55 out | decided: call add_numbers
      TOOL      add_numbers  ->  OK (1 ms)
                  args    {"a": 15, "b": 9}
                  result  {"result": 24}
    ...
    ANSWER
      **Result:** 15.
    ------------------------------------------------------------------------------
    SUMMARY  completed | 4 steps | 3 tool calls | tokens 3600 in / 180 out | 4.20s
    ==============================================================================

The OVERALL trace (every event of every request, one line each) is logs/app.log - see logging.py.
"""

import logging
import textwrap
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

WIDTH = 78
_MAX_OPEN_REQUESTS = 500       # requests in flight we keep buffers for
_MAX_RECORDS_PER_REQUEST = 2000


class RequestLogCollector(logging.Handler):
    """Buffers records per request_id and writes one readable file per request when it finishes."""

    def __init__(self, request_dir: Path):
        super().__init__(logging.DEBUG)
        self.request_dir = request_dir
        self._buffers: OrderedDict[str, list[logging.LogRecord]] = OrderedDict()
        self._buffers_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        request_id = getattr(record, "request_id", "-")
        if request_id == "-":  # startup / shutdown lines belong to app.log only
            return
        with self._buffers_lock:
            records = self._buffers.setdefault(request_id, [])
            if len(records) < _MAX_RECORDS_PER_REQUEST:
                records.append(record)
            while len(self._buffers) > _MAX_OPEN_REQUESTS:  # a request that never finished
                self._buffers.popitem(last=False)

    def finish(self, request_id: str) -> None:
        """Call once when a request is done: renders and writes its block."""
        with self._buffers_lock:
            records = self._buffers.pop(request_id, None)
        if not records:
            return
        try:
            day = datetime.fromtimestamp(records[0].created).strftime("%Y-%m-%d")
            path = self.request_dir / day / f"{request_id}.log"  # request_id is validated in main.py
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(render_request(request_id, records) + "\n")
        except Exception:
            self.handleError(records[-1])


# ----------------------------------------------------------------------------- rendering
def render_request(request_id: str, records: list[logging.LogRecord]) -> str:
    start = records[0].created
    http = _find(records, "http_request")
    summary = _find(records, "run_finished")

    out = ["=" * WIDTH, f"REQUEST  {request_id}",
           f"  when     {datetime.fromtimestamp(start):%Y-%m-%d %H:%M:%S}"]
    if http:
        out.append(f"  call     {http.method} {http.path}  ->  {http.status}   ({http.duration_ms / 1000:.2f}s)")
    caller = next((r.caller for r in records if getattr(r, "caller", "-") != "-"), "-")
    session = next((r.session_id for r in records if getattr(r, "session_id", "-") != "-"), "-")
    out.append(f"  caller   {caller}   |   session {session}")
    out.append("-" * WIDTH)

    used_results: set[int] = set()
    for i, r in enumerate(records):
        event = getattr(r, "event", None)

        if event == "user_message":
            out += ["USER", _indent(r.user_message, 2), ""]

        elif event == "approval_decided":
            decision = "APPROVED" if r.approved else "REJECTED"
            out += [f"APPROVAL  {decision} by user: {', '.join(r.tools)}  (approval {r.approval_id})", ""]

        elif event == "iteration":
            if out[-1] != "":
                out.append("")
            label = f"STEP {r.iteration}"
            offset = f"+{r.created - start:.2f}s"
            out.append(label + offset.rjust(WIDTH - len(label)))

        elif event == "llm_decision":
            # THINKING first (what the model reasoned), then what it decided (+ timing/tokens if logged)
            output = _find(records[i + 1:], "llm_output")
            if output and output.iteration == r.iteration:
                # (some vLLM versions echo the final answer as "reasoning" - don't print it twice)
                if output.reasoning and output.reasoning.strip() != (output.content or "").strip():
                    out.append(_field("THINKING", output.reasoning))
                if output.content and output.tool_calls:  # text the model said alongside its tool calls
                    out.append(_field("SAYS", output.content))
            decided = "call " + ", ".join(r.tools) if r.decision == "call_tools" else "final answer"
            call = _last_before(records, i, "llm_call", stop_at="iteration")
            stats = (f"{call.duration_ms / 1000:.2f}s | tokens {call.prompt_tokens} in / {call.completion_tokens} out | "
                     if call else "")
            out.append(_field("LLM", f"{stats}decided: {decided}"))

        elif event == "tool_call":
            status = {"ok": "OK", "error": "ERROR", "timeout": "TIMEOUT"}.get(r.status, r.status.upper())
            out.append(_field("TOOL", f"{r.tool}  ->  {status} ({r.duration_ms} ms)"))
            j = _find_index(records, i + 1, lambda x: getattr(x, "event", None) == "tool_result"
                            and x.tool == r.tool, used_results)
            if j is not None:
                used_results.add(j)
                out.append(_sub("args", records[j].arguments))
                out.append(_sub("result", records[j].result))
            else:
                out.append(_sub("args", ", ".join(r.arg_keys) or "(none)"))

        elif event == "tool_result" and i not in used_results:  # e.g. a tool the user rejected
            out.append(_field("TOOL", f"{r.tool}  ->  NOT RUN"))
            out.append(_sub("result", r.result))

        elif event == "approval_paused":
            out.append(_field("PAUSED", f"waiting for human approval of {', '.join(r.tools)}"))
            out.append(_sub("id", r.approval_id))

        elif event == "chat_ended":
            out.append(_field("ENDED", "chat closed after rejection - nothing executed, no further LLM call"))

        elif event == "final_answer":
            out += ["", "ANSWER", _indent(r.answer, 2)]

        elif event == "guardrail":
            text = f"{r.check}: {r.outcome}"
            judge = _last_before(records, i, "llm_call", stop_at="llm_decision") if r.check == "groundedness" else None
            if judge:  # the judge LLM call made for this check
                text += f"  (judge {judge.duration_ms / 1000:.2f}s | tokens {judge.prompt_tokens} in / " \
                        f"{judge.completion_tokens} out)"
            out.append(_field("GUARD", text))

        elif event == "guardrail_detail":
            out.append(_sub("detail", r.detail))

        elif r.levelno >= logging.WARNING:
            out.append(_field(r.levelname, _message_with_extras(r)))
            if r.exc_info:
                out.append(_indent(logging.Formatter().formatException(r.exc_info), 12))

    out.append("-" * WIDTH)
    if summary:
        out.append(f"SUMMARY  {summary.status} | {_plural(summary.iterations, 'step')} | "
                   f"{_plural(len(summary.tools_used), 'tool call')} | "
                   f"tokens {summary.prompt_tokens} in / {summary.completion_tokens} out | "
                   f"{summary.duration_ms / 1000:.2f}s")
    elif http:
        out.append(f"SUMMARY  HTTP {http.status} | {http.duration_ms / 1000:.2f}s")
    out.append("=" * WIDTH)
    return "\n".join(out) + "\n"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _find(records, event):
    return next((r for r in records if getattr(r, "event", None) == event), None)


def _last_before(records, index, event, stop_at):
    for k in range(index - 1, -1, -1):
        kind = getattr(records[k], "event", None)
        if kind == event:
            return records[k]
        if kind == stop_at:
            return None
    return None


def _find_index(records, start, predicate, exclude):
    return next((k for k in range(start, len(records)) if k not in exclude and predicate(records[k])), None)


def _field(label: str, text: str) -> str:
    """'  LABEL     text' with continuation lines aligned under the text."""
    prefix = f"  {label:<9} "
    return _wrap(str(text), prefix, " " * len(prefix))


def _sub(label: str, text) -> str:
    prefix = f"{'':12}{label:<7} "
    return _wrap(str(text if text is not None else "-"), prefix, " " * len(prefix))


def _indent(text, spaces: int) -> str:
    pad = " " * spaces
    return _wrap(str(text if text is not None else ""), pad, pad)


def _wrap(text: str, first: str, rest: str) -> str:
    lines = []
    for n, paragraph in enumerate(text.splitlines() or [""]):
        bullet = len(paragraph) - len(paragraph.lstrip(" -*•0123456789.)"))  # "- item", "1. item"
        lines += textwrap.wrap(paragraph, WIDTH, initial_indent=first if n == 0 else rest,
                               subsequent_indent=rest + " " * bullet, break_long_words=True,
                               replace_whitespace=False) or [rest.rstrip()]
    return "\n".join(lines)


_STD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName", "request_id",
                                                "session_id", "caller", "event", "extras"}


def _message_with_extras(record: logging.LogRecord) -> str:
    extras = {k: v for k, v in vars(record).items() if k not in _STD}
    tail = "  " + "  ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
    return record.getMessage() + tail
