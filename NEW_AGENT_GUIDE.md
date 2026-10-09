# Building a New Agent from This Template

This template is **one engine + one agent definition**. To make a new agent you copy the folder,
replace the agent definition, and leave the engine alone.

```
agent-template/
├── agent/        ← WHAT the agent is   (prompt, tools, config)       → you change this
├── app/          ← HOW agents run      (loop, API, security, logs)   → you don't touch this
├── tests/        ← offline tests (fake LLM)                          → update the agent-specific ones
├── evals/        ← live evaluation (real LLM + real tools)           → rewrite the dataset
└── Dockerfile, docker-compose.yml, render.yaml, .env                 → rename + add your secrets
```

---

## Part 1 — How it works (the flow)

### 1.1 Startup: how the agent is assembled

When the app starts (`uvicorn app.main:create_app --factory`), `app/runtime.py` builds the agent:

```
 .env  ──────────────►  Settings (app/settings.py)
                        LLM url/model/key, API keys, rate limits, log settings
                               │
 agent/agent.yaml  ──►  AgentConfig (app/agent_config.py)
                        ${VARS} replaced from .env, validated - a typo fails at startup
                               │
 agent/prompt.md   ──►  system prompt
                               │
 agent/tools.py    ──►  every @tool function is collected automatically
                               │
 mcp_servers:      ──►  connect to each MCP server, discover its tools,
                        name them <server>_<tool>, keep only include/exclude matches
                               │
                               ▼
                        ToolRegistry  (all tools, one place: validation, timeout, audit log)
                               │
                               ▼
                        Agent = LLM + registry + guardrails + limits
                               │
                               ▼
                        FastAPI routes ready  →  /health  /ready  /v1/chat ...
```

If an MCP server is down and `required: false`, the agent starts without those tools and
`/ready` shows it as `"down"`. If `required: true`, startup fails.

### 1.2 A request: what happens on `POST /v1/chat`

```
Client ── POST /v1/chat {"message": "...", "session_id": "..."}  + header X-API-Key
  │
  ├─ 1. middleware     request_id assigned (returned in X-Request-ID), timer started
  ├─ 2. auth           X-API-Key → caller name                        → 401 if wrong
  ├─ 3. rate limit     per caller: per-minute, concurrency, daily tokens → 429
  ├─ 4. input guard    empty / too long / invisible characters        → 400
  ├─ 5. injection      "ignore your instructions..." (regex)          → 400 (or flag)
  ├─ 6. session        load this caller's conversation history (or create a new one)
  │
  └─ 7. AGENT LOOP  (app/agent.py)   - up to max_iterations, within request_timeout_seconds
        ┌────────────────────────────────────────────────────────────────────┐
        │  LLM receives: safety rules + today's date + prompt.md             │
        │                + history + list of tools                           │
        │                                                                    │
        │  LLM decides:                                                      │
        │   a) answer in text ───────────────────────────► go to step 8      │
        │   b) call safe tools ──► registry runs them (parallel, timeout)    │
        │                          results checked for hidden injections     │
        │                          results added to history ──► loop again   │
        │   c) call an approval tool ──► PAUSE, return approval_required     │
        └────────────────────────────────────────────────────────────────────┘
  │
  ├─ 8. answer checks   groundedness (judge LLM: are facts backed by tool results?)
  │                     secrets redacted, system-prompt leak blocked, PII masked
  ├─ 9. response        {status, answer, session_id, tools_used, usage, checks}
  └─ 10. logging        one access line + logs/requests/<day>/<request_id>.log
                        (every step: thinking, tool calls, results, answer)

If anything fails halfway (LLM down, timeout), the session history is rolled back -
it's never left half-written. The client gets a safe error + request_id, never a stack trace.
```

### 1.3 Human approval (tools that change data)

```
POST /v1/chat  "Schedule a call with Emma tomorrow"
   └─ LLM: copilot_getContacts (safe, runs)  →  schedule_follow_up_call (needs approval)
   ◄─ {"status": "approval_required", "approval": {"approval_id": "...",
        "arguments": {"schedule_follow_up_call": "{...exact args...}"}}}

POST /v1/sessions/{session_id}/approvals/{approval_id}  {"approve": true}
   └─ tool runs → LLM continues → {"status": "completed", ...}

                                                       {"approve": false}
   └─ on_reject: end_chat → nothing runs, chat closed → {"status": "cancelled", "session_closed": true}
      on_reject: continue → LLM is told "rejected" and replies
```

The approval gate is enforced **in code**, so the LLM can't skip it, whatever the user says.

### 1.4 Continuing a conversation

The first response returns a `session_id`. Send it back with every next message to keep the
context. Same API key required (sessions are private per caller). Sessions live in memory:
they expire after `SESSION_TTL_SECONDS` and are lost on restart.

---

## Part 2 — Steps to create a new agent

Example: building **`sales-assist-agent`**.

### Step 1 — Copy the template (without local junk)

```bash
cd ~/My-projects/agents/agent-platform-single-agent-without-registry
rsync -a --exclude .venv --exclude .env --exclude logs --exclude data \
         --exclude .git --exclude evals/results --exclude __pycache__ --exclude .pytest_cache \
         agent-template/ sales-assist-agent/
cd sales-assist-agent
git init
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
```

### Step 2 — Write the job description: `agent/prompt.md`

This is the most important file. Replace every `<placeholder>`. It should say:

| Section | What to write | Example |
|---|---|---|
| Role | Who it is, for which company | "You are the sales assistant for NetXD." |
| Job | The one task it automates, for whom | "Help sales reps find leads and schedule follow-ups." |
| Scope | What it must **refuse** | "Only answer questions about leads, contacts and follow-ups. Politely decline anything else (poems, general knowledge, advice)." |
| Systems | Each tool and **when** to use it | "Use `zoho_searchLeads` to find a lead by name or email before any update." |
| Rules | Never guess; ask for missing info | "Never pick a date yourself - if the user didn't give one, ask." |
| Failures | What to do when a tool fails | "If a tool fails, say so. Never report success unless a tool confirmed it." |
| Style | Format of answers | "Short. Tables for lists. Say which system the data came from." |

> Lessons from our eval run on the template: without a **Scope** section the agent wrote poems;
> without "never pick a date" it invented dates; without "never report success unless a tool
> confirmed it" it claimed a failed action succeeded. Put these rules in every prompt.

Don't write "ask the user to confirm before changing data" - the approval step already does that.

### Step 3 — Write the tools: `agent/tools.py`

Delete the example tools (`get_current_time` can stay if useful) and add yours:

```python
from typing import Annotated, Literal
from pydantic import Field
from app.tools.base import ToolError, tool

@tool
async def find_lead(email: Annotated[str, Field(description="Lead's email address")]) -> dict:
    """Find a CRM lead by email. Use before any follow-up or update on a lead."""
    lead = await crm.search(email)
    if not lead:
        raise ToolError(f"No lead with email {email}")        # LLM reads this and reacts
    return {"id": lead.id, "name": lead.name, "stage": lead.stage}   # small: only what's needed

@tool(requires_approval=True)                                  # changes data → human must approve
async def update_lead_stage(lead_id: str, stage: Literal["new", "qualified", "won", "lost"]) -> dict:
    """Move a lead to a new pipeline stage."""
    ...
```

Rules:
1. **Docstring = when to use it.** The LLM picks tools by reading it. Be specific.
2. **Type hints on every parameter** (`Annotated[..., Field(description=...)]`). Arguments are
   validated before your code runs; use `Literal`, `pattern`, `ge/le` to restrict them.
3. **Return small dicts** with only the fields the LLM needs (big results are truncated).
4. **Expected failures → `raise ToolError("clear message")`.** Unexpected exceptions are caught too.
5. **Anything that creates/updates/deletes/sends → `requires_approval=True`.**
6. **Secrets from env** (`os.environ["CRM_TOKEN"]`), never hard-coded, never returned.

Tools that already exist as an MCP server don't need Python code - see Step 4.

### Step 4 — Configure: `agent/agent.yaml`

| Key | Change to |
|---|---|
| `name`, `description` | e.g. `sales_assist`, "Sales assistant for lead follow-ups" (shown in `/v1/agent`, logs, metrics) |
| `llm.temperature` | 0.1–0.3 for repetitive/deterministic jobs; 0.7 for conversational ones |
| `llm.max_tokens` | Prompt + this must fit the model context (gpt-oss-120b: 32k) |
| `limits.*` | `max_iterations` = most tool steps a normal task needs + a few; timeouts to fit your APIs |
| `approval.tools` | Wildcards for risky MCP tools, e.g. `["zoho_*update*", "zoho_*delete*"]` |
| `approval.on_reject` | `end_chat` (strict) or `continue` (agent suggests alternatives) |
| `guardrails.groundedness` | `revise` if wrong answers are costly (+1 LLM call); `flag` to only report |
| `guardrails.mask_pii` | Add `email` / `phone` only if the job doesn't need to show them |
| `mcp_servers` | Your servers. **Always** set `include:` to the minimum tools needed |

```yaml
mcp_servers:
  - name: zoho                                 # tools become zoho_<tool>
    transport: http                            # or stdio (command/args), e.g. npx mcp-remote
    url: ${ZOHO_MCP_URL}                       # value comes from .env
    headers:
      Authorization: "Bearer ${ZOHO_MCP_TOKEN}"
    include: ["searchLeads", "getLead", "updateLead"]
    timeout_seconds: 30
    required: true                             # false = agent may start without it
```

Remove the `copilot` server if the new agent doesn't use it.

### Step 5 — Secrets and settings: `.env` and `.env.example`

- `.env` (never committed): real values for `LLM_*`, `API_KEYS`, and every `${VAR}` used in agent.yaml.
- `.env.example` (committed): same keys, empty or dummy values, so the next person knows what's needed.
- Generate API keys: `python -c "import secrets;print(secrets.token_urlsafe(32))"` → `API_KEYS=web-ui:<key>`

### Step 6 — Rename the project

| File | Change |
|---|---|
| `pyproject.toml` | `name = "sales-assist-agent"` |
| `render.yaml` | `name: sales-assist-agent`; add each new secret as `- key: ZOHO_MCP_TOKEN` + `sync: false`; remove `COPILOT_*` if unused |
| `README.md` | Replace the intro with what this agent does |

### Step 7 — Container (only if needed)

- **`Dockerfile`**: if you only use HTTP MCP servers (no `npx`), delete the `apt-get install nodejs npm` line - smaller and safer image.
- **`docker-compose.yml`**: on a server behind Nginx, change `"8000:8000"` to `"127.0.0.1:8000:8000"`.
- If the agent writes files, write them under `data/` (mounted volume).

### Step 8 — Update the tests

These tests reference the template's example tools and **will fail** after you delete them:

| File | References | Do |
|---|---|---|
| `tests/test_calculator.py` | calculator tools | Delete, or replace with tests for your tools |
| `tests/test_api.py` | `create_support_ticket`, `schedule_follow_up_call` | Point the approval tests at one of **your** approval tools |
| `tests/test_request_logs.py` | `add_numbers` | Use any safe tool of yours |

All other tests cover the engine and should pass unchanged. Add one test file per tool
(`tests/test_<tool>.py`): happy path, bad arguments, and the `ToolError` path.

```bash
.venv/bin/python -m pytest -q        # must be all green
```

### Step 9 — Try it by hand

```bash
.venv/bin/python -m scripts.chat                                   # terminal chat, no server
.venv/bin/uvicorn app.main:create_app --factory --reload           # API → http://localhost:8000/docs
```

Read `logs/requests/<today>/<request_id>.log` to see exactly what the agent thought and which
tools it called.

### Step 10 — Rewrite the evaluation dataset

`evals/dataset.yaml` is written for the template's data (David, Emma, users 1 and 2). Replace the
cases with your agent's ground truth, keeping all 12 categories:

happy path · edge cases · ambiguous · missing info · wrong info · tool failures · API failures ·
auth failures · security attacks · prompt injection · out of scope · adversarial

Update the facts at the top of the judge prompt in `evals/run_evals.py` (`JUDGE_PROMPT`) to your
agent's tools and data. The API/auth/security cases (AF-, AU-, SA-, PI-01..04) mostly work
as they are - only change tool names.

```bash
.venv/bin/python evals/run_evals.py              # full run → evals/results/<time>/report.md
.venv/bin/python evals/run_evals.py -k HP-       # one category
.venv/bin/python evals/run_evals.py --repeats 3  # catch flaky behaviour
```

Ship when the happy path is 100% and every security / prompt-injection / tool-failure case passes.

### Step 11 — Deploy

**Render:** push to GitHub → Render → New + → Blueprint → pick the repo → fill the `sync: false`
secrets → deploy. Check `https://<app>.onrender.com/health` and `/ready`.

**EC2 + Docker + Nginx:**
```bash
git clone <repo> && cd sales-assist-agent
nano .env                                      # APP_ENV=production, LOG_FORMAT=json,
                                               # LOG_REQUEST_PAYLOADS=false, API_KEYS=...
mkdir -p logs data && sudo chown -R 10001:10001 logs data   # container runs as uid 10001
docker compose up -d --build
curl localhost:8000/health
```
Nginx: `proxy_pass http://127.0.0.1:8000;` with `proxy_read_timeout` **above**
`request_timeout_seconds` (e.g. 200s, otherwise long answers get a 504), then
`sudo certbot --nginx -d <domain>` for HTTPS.

---

## Part 3 — Quick reference

### What you change vs. what you leave alone

| Change per agent | Leave alone |
|---|---|
| `agent/prompt.md`, `agent/tools.py`, `agent/agent.yaml` | `app/**` (engine) |
| `.env`, `.env.example` | `tests/conftest.py`, engine tests |
| `pyproject.toml`, `render.yaml`, `README.md` | `scripts/chat.py` |
| `tests/test_calculator.py`, `test_api.py`, `test_request_logs.py` | `evals/run_evals.py` (except `JUDGE_PROMPT` facts) |
| `evals/dataset.yaml` | |
| `Dockerfile` (Node line), `docker-compose.yml` (port bind) | |

If you find yourself editing `app/`, the change probably belongs in the template itself so every
agent gets it - make it there and copy it across.

### Where to look when something is wrong

| Symptom | Look at |
|---|---|
| Agent picks the wrong tool / no tool | Tool docstrings, `prompt.md` "Systems" section |
| Tool arguments wrong | Type hints / `Field(description=...)` in `tools.py` |
| Agent invents data | Prompt rules + `guardrails.groundedness: revise` |
| 400 on a normal message | `guardrails.prompt_injection` regex false positive → `flag` instead of `block` |
| MCP tools missing | `/ready` (`"down"`?), `include:` list, `${VAR}` set in `.env` |
| 502 `llm_unavailable` | `LLM_BASE_URL` reachable? `LLM_API_KEY`? |
| 504 timeout | `limits.request_timeout_seconds`, slow tool, Nginx `proxy_read_timeout` |
| Any error | Take `request_id` from the response → `logs/requests/<day>/<request_id>.log` |

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process alive (no auth) |
| GET | `/ready` | Agent + MCP servers ready (no auth) |
| GET | `/metrics` | Prometheus (API key in production) |
| GET | `/v1/agent` | Agent name + tool list |
| POST | `/v1/chat` | Send a message (`session_id` optional) |
| POST | `/v1/sessions/{sid}/approvals/{aid}` | Approve / reject a pending tool call |
| DELETE | `/v1/sessions/{sid}` | Forget a conversation |
