# PawBox Retention Agent

A reason-aware customer retention demo built for Chatathon 2026 (Klaviyo track). The agent investigates at-risk PawBox subscribers with OpenAI tool calling, identifies the likely cause of churn, and recommends a next step instead of reflexively offering a discount.

> **Demo notice:** All customer, order, support, subscription, and engagement data in this project is simulated. No production systems are connected.

This README is written for whoever (or whatever) picks up the repo next. Read [Before you change anything](#before-you-change-anything) before editing — several non-obvious constraints will bite you otherwise.

---

## Quick start

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

Create a `.env` in the repo root:

```text
OPENAI_API_KEY=sk-your-key-here
```

`.env` is gitignored and `agent.py` loads it via `python-dotenv` at import time. **Never put a real key in the README, in a committed file, or in a shell command you paste into a log.** A key was committed to this repo once and had to be scrubbed.

If you prefer an environment variable over `.env`, the agent reads `OPENAI_API_KEY` from the process environment either way:

```bash
$env:OPENAI_API_KEY = "sk-your-key-here"
```

**Requirements:** Python 3.10+ (`agent.py` uses `str | None` in a runtime-evaluated signature, so 3.9 raises `TypeError` on import), plus the three packages in `requirements.txt` — `openai`, `streamlit`, `python-dotenv`.

---

## Commands

Every command below makes live OpenAI API calls except where noted. There is no offline/mock mode.

| Command | What it does |
| --- | --- |
| `streamlit run app.py` | Launch the demo UI. Pick a customer in the sidebar, click **Investigate Customer**. No API call until you click. |
| `python agent.py` | Run one investigation, defaults to `C001`. |
| `python agent.py C003` | Run one specific persona. |
| `python agent.py C001 C004` | Run several personas in sequence. |
| `python agent.py all` | Run all five personas, then print a one-line-per-persona summary. |
| `python eval.py` | Run all five personas once each and print a pass/fail table against expected decisions. |
| `python eval.py C002` | Eval a subset. Case IDs are case-insensitive. |
| `python eval.py -n 5` | Run each case 5 times and report a rate instead of a boolean. **This is 5× the API spend** — `-n 5` over all cases is 25 investigations. |
| `python eval.py -v` | Also dump each full agent response. Combines with the above. |

`eval.py` exit codes: `0` all clean, `1` at least one case failed or was flaky, `2` bad arguments or unknown case ID. Use it in a check gate; don't parse the table text.

---

## How it fits together

```text
.
├── agent.py            # SYSTEM_PROMPT, the tool-calling loop, output parsing. The core.
├── app.py              # Streamlit UI. Has its OWN copy of the loop — see gotchas.
├── tools.py            # Five read-only data tools + their OpenAI schemas.
├── eval.py             # Persona regression harness. Read its docstring before trusting it.
├── requirements.txt
└── data/
    └── personas.json   # The five simulated customers. Single source of fixture truth.
```

Flow: `run_agent(customer_id)` seeds a system + user message, then loops — call the model, execute any tool calls against `tools.TOOLS`, append results, repeat — until the model replies without tool calls or `MAX_TURNS` (12) is hit. The final markdown is parsed into structured fields.

### Programmatic entry point

```python
from agent import run_agent

result = run_agent("C002", on_event=None, max_turns=12)
```

Returns this contract — consume these keys rather than re-parsing the markdown yourself:

| Key | Type | Notes |
| --- | --- | --- |
| `customer_id` | `str` | Echoed back. |
| `final_response` | `str` | Full markdown analysis. `""` if the loop hit the turn cap. |
| `decision` | `"ACT" \| "WAIT" \| "ESCALATE" \| None` | `None` when unparseable or stopped early. |
| `confidence` | `"HIGH" \| "MEDIUM" \| "LOW" \| None` | Same. |
| `churn_cause` | `str \| None` | One-line diagnosis. |
| `tool_calls` | `list[dict]` | `{tool, input, result, error}` per call, in order. |
| `turns` | `int` | Model round-trips used. |
| `usage` | `dict` | Summed `prompt_tokens`, `completion_tokens`, `total_tokens`. |
| `stopped_early` | `bool` | `True` means the turn cap was hit — treat the run as failed. |

`run_agent` raises `RuntimeError` immediately if `OPENAI_API_KEY` is unset, before spending anything.

The optional `on_event(event_type, data)` callback streams progress. Event types: `thinking`, `tool_call`, `tool_result`, `response`, `error`. See `print_event` at the bottom of `agent.py` for a working consumer.

### Tools

All five take a single `customer_id` and return a JSON string (or a plain-text "not found" / "no tickets" message — callers must tolerate non-JSON):

`get_customer_summary`, `get_order_history`, `get_email_engagement`, `get_support_tickets`, `get_subscription_events`.

They are pure reads from `data/personas.json`, re-loaded from disk on every call. Adding a tool means adding both the function to `TOOLS` **and** its schema to `TOOL_DEFINITIONS` in `tools.py` — the registry and the schema list are separate and nothing checks they agree.

---

## The personas

| ID | Scenario | Expected decision |
| --- | --- | --- |
| `C001` | Product mismatch | `ACT` |
| `C002` | Price sensitive / discount bait | `ACT` (without offering a discount) |
| `C003` | Repeated service failure | `ESCALATE` |
| `C004` | Life circumstance change | `ESCALATE` |
| `C005` | False alarm | `WAIT` |

Expectations live in the `CASES` map in `eval.py`. Each record in `data/personas.json` needs: `name`, `pet_name`, `pet_type`, `pet_breed`, `pet_age_months`, `plan`, `monthly_price`, `member_since`, `status`, `churn_scenario`, `order_history`, `email_engagement`, `support_tickets`, `subscription_events`. Adding a persona means touching `personas.json`, `CASES` in `eval.py`, and `SCENARIO_LABELS` in `app.py`.

---

## Decision framework

The prompt forces one of three outcomes:

- **ACT** — evidence supports a specific first action; the agent drafts the actual customer message.
- **WAIT** — signals are ambiguous; the agent states what to monitor and for how long.
- **ESCALATE** — needs human judgment (sensitive life circumstance, repeated ops failure).

Discounting is deliberately constrained. All three must hold: clear price sensitivity, demonstrated product value, and a plausible path back to a sustainable full-price relationship. The prompt also separates price-*sensitive* from discount-*dependent* customers — the latter get a structural fix (smaller plan, longer cadence, honest value case), not another promo code. When price sensitivity is diagnosed and the agent declines to discount, it must say so explicitly in its rationale.

---

## Before you change anything

These are the things that have already caused problems here.

**1. `app.py` does not use `run_agent()`.** It imports only `SYSTEM_PROMPT` from `agent.py` and reimplements the loop inline, with its own hardcoded `gpt-4o`, `max_tokens`, decision parsing, and no turn cap. A fix to the loop, the model constant, or the parser in `agent.py` will not reach the UI. If you change one, check the other — or consolidate them, which is the real fix.

**2. The model is pinned in two places.** `MODEL` in `agent.py:15` and a string literal in the `client.chat.completions.create` call in `app.py`.

**3. Freeze `agent.py` across an eval run you intend to compare.** The system prompt was rewritten mid-suite once and silently mixed two populations into one results table. Eval numbers are only comparable within one version of the prompt.

**4. One eval run is one sample of a nondeterministic system.** `C005` has returned both `WAIT` and `ACT` on identical input. A single-run PASS is weak evidence; a single-run FAIL is worth re-running with `-n` before you act on it.

**5. The C002 discount check is partly self-fulfilling.** Critical Rule 1 in the prompt tells the model to state plainly that it is declining a discount — almost the exact string `DISCOUNT_DECLINE_PATTERNS` greps for. A PASS there mostly measures prompt compliance. The unscripted signal is the `CONTRADICTION` verdict: the draft message isn't mandated, so what the agent actually offers the customer still measures behavior. Read the draft before believing any prompt change "fixed" C002.

**6. Both C002 checks grep free-form prose.** A miss means "go read the response", not "the agent is wrong". The durable fix is a structured `discount_offered` field in the output contract, which requires changing the output format in `agent.py`.

The full version of points 3–6 lives in the `KNOWN LIMITATIONS` docstring at the top of `eval.py`. Read it before trusting a green table.

**7. Every run costs money and rate limits are real.** Roughly 15 concurrent calls has been enough to trigger `RateLimitError` when several people run the suite at once. `eval.py` retries transient rate-limit, connection, and timeout errors up to 3 times with a backoff, but it can't conjure budget. Don't run `agent.py` or `eval.py` casually to "check something" — read the code first, and coordinate if others are working the repo.

---

## Verifying a change

Static checks are free; agent runs are not. In order:

1. Read the diff and the two gotchas above that touch your files.
2. `python -m py_compile agent.py app.py tools.py eval.py` — catches syntax breaks with zero API calls and zero side effects. (Don't `import app` to check it: its Streamlit calls execute at module level.)
3. `python eval.py C00X` — the single case your change should affect.
4. `python eval.py` — the full suite, once you expect it to pass.
5. `python eval.py -n 5` — only when you need a rate rather than a verdict, and only with budget to spare.
