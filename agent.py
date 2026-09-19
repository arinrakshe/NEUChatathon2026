"""Reason-Aware Retention Agent — core loop using OpenAI tool use."""

import json
from openai import OpenAI
from tools import TOOLS, TOOL_DEFINITIONS

SYSTEM_PROMPT = """You are a retention agent for PawBox, a pet food and treats subscription service. Your job is to investigate customers who have been flagged as at-risk of churning, determine WHY they are at risk, and decide on the appropriate first action.

## Your investigation process
You have access to tools that let you pull customer data on demand. You decide what to look at and when you have enough evidence. Do not call every tool by default — investigate based on what you learn.

Start with get_customer_summary to understand who the customer is, then dig deeper based on what you find.

## Decision framework
After investigating, you MUST reach one of three decisions:

**ACT** — You have enough evidence to take a specific, helpful first action. Draft the actual message you would send.
**WAIT** — The signals are ambiguous. Specify what you'd monitor and for how long before re-evaluating.
**ESCALATE** — The situation requires human judgment (e.g., a sensitive life circumstance, a repeated service failure that needs ops intervention).

## Critical rules
1. NEVER default to offering a discount. A discount is only appropriate when there is clear evidence of price sensitivity AND the customer has shown they value the product. Discounting a customer who has a product mismatch, service problem, or life change is tone-deaf and wasteful.
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

If your decision is ACT, also draft the customer message under:
**Draft Message:**
[The actual email/message you would send, in a warm but professional brand voice]
"""


def run_agent(customer_id: str, on_event=None):
    """Run the retention agent for a given customer.

    on_event: optional callback(event_type, data) for streaming UI updates.
    Event types: 'thinking', 'tool_call', 'tool_result', 'response'
    """
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

    # Agent loop — keep going until the model stops calling tools
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        message = response.choices[0].message

        # If there's text content, emit it
        if message.content:
            emit("thinking", message.content)

        # If no tool calls, we're done
        if not message.tool_calls:
            final_text = message.content or ""
            emit("response", final_text)
            return {
                "final_response": final_text,
                "tool_calls": tool_call_log,
            }

        # Append the assistant message (with tool calls) to history
        messages.append(message)

        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            emit("tool_call", {"name": tool_name, "input": tool_input})

            # Execute the tool
            fn = TOOLS.get(tool_name)
            if fn:
                result = fn(**tool_input)
            else:
                result = f"Unknown tool: {tool_name}"

            tool_call_log.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result,
            })
            emit("tool_result", {"name": tool_name, "result": result})

            # Append tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


if __name__ == "__main__":
    import sys
    cid = sys.argv[1] if len(sys.argv) > 1 else "C001"

    def print_event(event_type, data):
        if event_type == "tool_call":
            print(f"\n🔧 Calling: {data['name']}({data['input']})")
        elif event_type == "tool_result":
            print(f"   ✓ Got result ({len(data['result'])} chars)")
        elif event_type == "thinking":
            print(f"\n💭 {data[:200]}...")
        elif event_type == "response":
            print(f"\n{'='*60}\nFINAL ANALYSIS\n{'='*60}\n{data}")

    result = run_agent(cid, on_event=print_event)
