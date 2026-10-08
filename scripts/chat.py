"""
Talk to the agent in your terminal - no HTTP server needed. Great while building tools.

    python -m scripts.chat
"""

import asyncio
import json
import uuid

from app.observability.logging import configure_logging, finish_request, request_id_var
from app.runtime import Runtime
from app.sessions import Session
from app.settings import get_settings


async def main() -> None:
    settings = get_settings()
    configure_logging("WARNING", "text", log_dir=settings.log_dir or None,
                      request_files=settings.log_request_files, payloads=settings.log_request_payloads)
    runtime = Runtime(settings)
    agent = await runtime.start()
    session = Session(id="cli", owner="cli")
    print(f"Agent '{agent.config.name}' ready with tools: {', '.join(agent.registry.names())}")
    print("Type a message (Ctrl+C to quit).\n")
    try:
        while True:
            message = input("you > ").strip()
            if not message:
                continue
            request_id_var.set(f"cli-{uuid.uuid4().hex[:12]}")  # each message gets its own log file
            try:
                result = await agent.chat(session, message)
            finally:
                finish_request(request_id_var.get())
            while result.status == "approval_required":
                for call in result.approval.tool_calls:
                    print(f"  ! wants to run {call['function']['name']}({call['function']['arguments']})")
                ok = input("  approve? [y/N] ").strip().lower() == "y"
                request_id_var.set(f"cli-{uuid.uuid4().hex[:12]}")
                try:
                    result = await agent.decide(session, result.approval.id, ok)
                finally:
                    finish_request(request_id_var.get())
            print(f"agent > {result.answer}")
            if result.status == "cancelled":  # rejected with on_reject=end_chat
                session = Session(id=f"cli-{uuid.uuid4().hex[:8]}", owner="cli")
                print("        [conversation closed - new conversation started]")
            print(f"        [tools: {json.dumps(result.tools_used)}, iterations: {result.iterations}, "
                  f"tokens: {result.prompt_tokens}+{result.completion_tokens}, log: {request_id_var.get()}]\n")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
