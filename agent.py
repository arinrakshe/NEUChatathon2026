"""Reason-Aware Retention Agent — core loop using OpenAI tool use."""

import json
import os
import re
from openai import OpenAI
from tools import TOOLS, TOOL_DEFINITIONS

try:  # optional — lets `python agent.py C001` work straight from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "gpt-4o"
MAX_TURNS = 12          # hard stop so a tool-call loop can't run away
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a retention agent for PawBox, a pet food and treats subscription service. Your job is to investigate customers who have been flagged as at-risk of churning, determine WHY they are at risk, and decide on the appropriate first action.

## Your investigation process
You have access to tools that let you pull customer data on demand. You decide what to look at and when you have enough evidence. Do not call every tool by default — investigate based on what you learn.

Start with get_customer_summary to understand who the customer is, then dig deeper based on what you find.

## Decision framework
After investigating, you MUST reach one of three decisions:

**ACT** — You have enough evidence to take a specific, helpful first action. Draft the actual message you would send.
**WAIT** — The signals are ambiguous. Specify what you'd monitor and for how long before re-evaluating.
**ESCALATE** — A human has to make a judgement call you should not make on the company's behalf: a sensitive life circumstance (bereavement, illness, financial hardship), or a repeated service failure that needs ops intervention or compensation.
ESCALATE is NOT the label for "this case needs a specialist." Booking a nutrition consult, a product-fit review, or a plan change is a first action you own — that is ACT, and the referral is the action. A product fit problem is not an escalation no matter how many formulas have failed, and neither is a case simply being important, frustrating, or close to cancellation.
Tiebreaker: if you can write the customer message yourself and sending it is the right next step, the decision is ACT — even when that message hands off to a specialist. Choose ESCALATE only when a human must decide something BEFORE any message goes out. If you have drafted a competent outward message and still labelled it ESCALATE, the label is wrong.

## Critical rules
1. NEVER default to offering a discount. Discounting a customer who has a product mismatch, service problem, or life change is tone-deaf and wasteful. Evidence of price sensitivity is NECESSARY but NOT SUFFICIENT to justify a discount — all three of these must hold: (a) clear evidence of price sensitivity, (b) the customer has shown they value the product, and (c) the discount plausibly leads to a sustainable full-price relationship rather than to another promo-to-promo cycle.
   In particular, distinguish price-SENSITIVE from discount-DEPENDENT. A customer who converts only during promotions — cancelling, skipping, or going dormant every time full price returns, and who engages with promo emails but ignores everything else — is discount-dependent. For them, condition (c) fails: another discount just re-confirms the pattern, trains them to wait for the next promo, and erodes margin with no path back to full price. The correct first action is to fix the price/value fit structurally — move them to a lower-priced or smaller plan, a longer delivery cadence, or a right-sized box they will actually pay full price for — or to make the value case honestly. Not another promo code, and not a bespoke "special rate."
   Whenever price sensitivity is part of the diagnosis and you decide NOT to discount, say so explicitly in your Rationale: state plainly that you are declining to offer a discount, and why.
2. Before deciding on any action, explicitly check: "Could this be a service failure? Could this be a product fit issue? Could there be a life circumstance I should respect?" Only after ruling those out should you consider price-related actions.
3. Read support ticket text carefully. Customers tell you what's wrong in their own words — that is your best evidence.
4. Be genuinely helpful and non-pushy. If a customer is going through something difficult, respect that. Sometimes the right business move is to show empathy, not push retention.
5. State your confidence level (HIGH / MEDIUM / LOW) and explain what evidence supports your decision.

## Output format
After your investigation, provide your final analysis in this structure:

**Churn Cause:** [one-line diagnosis]
**Confidence:** [HIGH / MEDIUM / LOW]
**Evidence Summary:** [bullet points of key evidence from your investigation]
**Decision:** [ACT / WAIT / ESCALATE]
**Rationale:** [why this decision and not the alternatives]
**Action:** [if ACT: the specific action. If WAIT: what to monitor. If ESCALATE: who and why.]
**Discount Offered:** [YES / NO — YES only if your action includes a price reduction, promo code, coupon, or bespoke rate on the CURRENT plan. Moving the customer to a genuinely different, lower-priced plan or cadence is NOT a discount; answer NO. Answer for every customer, including ones where price never came up.]

If your decision is ACT, also draft the customer message under:
**Draft Message:**
[The actual email/message you would send, in a warm but professional brand voice]
"""

DECISIONS = ("ACT", "WAIT", "ESCALATE")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
DISCOUNT_ANSWERS = ("YES", "NO")


def _extract_field(text: str, label: str) -> str | None:
    """Pull a '**Label:** value' field out of the agent's markdown analysis.

    Tolerates the variants the model actually produces: '**Label:** v',
    '**Label: v**', and a plain 'Label: v'.
    """
    pattern = rf"\*{{0,2}}{re.escape(label)}\s*:?\*{{0,2}}\s*:?\s*(.+)"
    for line in text.splitlines():
        match = re.match(pattern, line.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("*").strip() or None
    return None


def _extract_enum(text: str, label: str, allowed: tuple[str, ...]) -> str | None:
    """Extract a field whose value must be one of `allowed` (e.g. the decision)."""
    value = _extract_field(text, label)
    if value:
        upper = value.upper()
        for option in allowed:
            if option in upper:
                return option
    # Fall back to a document-wide scan — the model sometimes bolds the value
    # on its own line instead of inline after the label.
    for option in allowed:
        if re.search(rf"\*\*{label}\s*:?\*\*\s*\n+\s*\*{{0,2}}{option}", text, re.IGNORECASE):
            return option
    return None


def _extract_yes_no(text: str, label: str) -> str | None:
    """Extract a YES/NO field, anchoring on the leading token.

    Deliberately not _extract_enum: these answers carry a trailing explanation
    ("NO - a smaller plan is not a discount") that can contain the opposite
    word, and a substring scan would read that as YES.
    """
    value = _extract_field(text, label)
    if not value:
        return None
    upper = value.upper().lstrip("*[( ").strip()
    for option in DISCOUNT_ANSWERS:
        if upper.startswith(option):
            return option
    return None


def parse_analysis(final_text: str) -> dict:
    """Parse the agent's final markdown into the fields the UI renders.

    This is the output contract the Streamlit app consumes — the UI should read
    these keys rather than re-parsing the markdown itself.
    """
    return {
        "decision": _extract_enum(final_text, "Decision", DECISIONS),
        "confidence": _extract_enum(final_text, "Confidence", CONFIDENCES),
        "churn_cause": _extract_field(final_text, "Churn Cause"),
        "discount_offered": _extract_yes_no(final_text, "Discount Offered"),
    }


def run_agent(customer_id: str, on_event=None, max_turns: int = MAX_TURNS) -> dict:
    """Run the retention agent for a given customer.

    on_event: optional callback(event_type, data) for streaming UI updates.
    Event types: 'thinking', 'tool_call', 'tool_result', 'response', 'error'

    Returns the output contract described in parse_analysis(), plus the tool
    trace, turn count, and token usage.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env (and `pip install python-dotenv`) "
            "or export it in your shell before running the agent."
        )

    client = OpenAI()

    def emit(event_type, data):
        if on_event:
            on_event(event_type, data)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Investigate customer {customer_id} who has been flagged as at-risk of churning. Use the tools available to understand their situation, determine the likely cause, and decide on the appropriate action. Think step by step about what you're seeing."
        }
    ]

    tool_call_log = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    turns = 0

    # Agent loop — keep going until the model stops calling tools, or we hit the cap
    while turns < max_turns:
        turns += 1

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.usage:
            usage["prompt_tokens"] += response.usage.prompt_tokens
            usage["completion_tokens"] += response.usage.completion_tokens
            usage["total_tokens"] += response.usage.total_tokens

        message = response.choices[0].message

        # Intermediate reasoning only. On the final turn there are no tool
        # calls and the same string goes out as "response" below — emitting
        # both made `python agent.py C001` print the analysis twice.
        if message.content and message.tool_calls:
            emit("thinking", message.content)

        # If no tool calls, we're done
        if not message.tool_calls:
            final_text = message.content or ""
            emit("response", final_text)
            return {
                "customer_id": customer_id,
                "final_response": final_text,
                "tool_calls": tool_call_log,
                "turns": turns,
                "usage": usage,
                "stopped_early": False,
                **parse_analysis(final_text),
            }

        # Append the assistant message (with tool calls) to history
        messages.append(message.model_dump(exclude_none=True))

        # Execute each tool call. Every tool_call MUST get a matching tool
        # message back — a missing one makes the next request invalid.
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            error = None

            try:
                tool_input = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                tool_input = {}
                error = f"Could not parse arguments: {exc}"

            emit("tool_call", {"name": tool_name, "input": tool_input})

            if error:
                result = f"Error: {error}"
            else:
                fn = TOOLS.get(tool_name)
                if fn is None:
                    error = f"Unknown tool: {tool_name}"
                    result = f"Error: {error}. Available tools: {', '.join(TOOLS)}"
                else:
                    try:
                        result = fn(**tool_input)
                    except Exception as exc:  # feed the failure back so the model can recover
                        error = f"{type(exc).__name__}: {exc}"
                        result = f"Error calling {tool_name}: {error}"

            if error:
                emit("error", {"name": tool_name, "error": error})

            tool_call_log.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result,
                "error": error,
            })
            emit("tool_result", {"name": tool_name, "result": result, "error": error})

            # Append tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Hit the turn cap without the model settling on an answer
    emit("error", {"error": f"Stopped after {max_turns} turns without a final analysis."})
    return {
        "customer_id": customer_id,
        "final_response": "",
        "tool_calls": tool_call_log,
        "turns": turns,
        "usage": usage,
        "stopped_early": True,
        "decision": None,
        "confidence": None,
        "churn_cause": None,
        "discount_offered": None,
    }


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0].lower() == "all":
        customer_ids = ["C001", "C002", "C003", "C004", "C005"]
    else:
        customer_ids = args or ["C001"]

    def print_event(event_type, data):
        if event_type == "tool_call":
            print(f"\n[TOOL] Calling: {data['name']}({data['input']})")
        elif event_type == "tool_result":
            flag = "[ERR]" if data.get("error") else "[OK]"
            print(f"   {flag} Got result ({len(data['result'])} chars)")
        elif event_type == "thinking":
            print(f"\n[THINKING] {data[:200]}...")
        elif event_type == "error":
            print(f"\n[WARN] {data.get('error')}")
        elif event_type == "response":
            print(f"\n{'='*60}\nFINAL ANALYSIS\n{'='*60}\n{data}")

    summary = []
    for cid in customer_ids:
        print(f"\n\n{'#'*60}\n# {cid}\n{'#'*60}")
        result = run_agent(cid, on_event=print_event)
        summary.append(result)
        print(
            f"\n→ decision={result['decision']} confidence={result['confidence']} "
            f"turns={result['turns']} tools={len(result['tool_calls'])} "
            f"tokens={result['usage']['total_tokens']}"
        )

    if len(summary) > 1:
        print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
        for r in summary:
            print(f"{r['customer_id']}: {r['decision'] or 'NO DECISION':10} ({r['confidence'] or '—'}) — {r['churn_cause'] or ''}")
