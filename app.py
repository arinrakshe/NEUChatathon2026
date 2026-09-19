"""Streamlit UI for the Reason-Aware Retention Agent."""

import streamlit as st
import json
from tools import TOOLS, TOOL_DEFINITIONS
from pathlib import Path

# Load persona data for the sidebar
DATA_PATH = Path(__file__).parent / "data" / "personas.json"
with open(DATA_PATH) as f:
    CUSTOMERS = json.load(f)["customers"]

SCENARIO_LABELS = {
    "product_mismatch": "Product Mismatch",
    "price_sensitive_no_discount": "Price Sensitive (Don't Discount)",
    "service_failure": "Service Failure",
    "life_change": "Life Circumstance Change",
    "false_alarm": "False Alarm",
}

DECISION_COLORS = {
    "ACT": "#2ecc71",
    "WAIT": "#f39c12",
    "ESCALATE": "#e74c3c",
}

st.set_page_config(page_title="PawBox Retention Agent", page_icon="🐾", layout="wide")

st.markdown("""
<div style='background-color: #1a1a2e; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px;'>
    <span style='color: #888; font-size: 0.8em;'>ALL DATA ON THIS PAGE IS <b>SIMULATED</b> FOR DEMONSTRATION PURPOSES</span>
</div>
""", unsafe_allow_html=True)

st.title("PawBox Retention Agent")
st.caption("Reason-aware churn investigation — spot the cause, not just the risk")

# Sidebar: customer picker
st.sidebar.header("Select Customer")
customer_options = {}
for cid, c in CUSTOMERS.items():
    label = f"{c['name']} — {c['pet_name']} the {c['pet_breed']}"
    customer_options[label] = cid

selected_label = st.sidebar.radio("Flagged at-risk customers:", list(customer_options.keys()))
selected_id = customer_options[selected_label]
selected = CUSTOMERS[selected_id]

# Sidebar: customer card
st.sidebar.divider()
st.sidebar.subheader(f"{selected['pet_name']}")
st.sidebar.markdown(f"""
**Owner:** {selected['name']}
**Pet:** {selected['pet_name']} the {selected['pet_breed']} ({selected['pet_age_months']} months old)
**Plan:** {selected['plan']} (${selected['monthly_price']}/mo)
**Member since:** {selected['member_since']}
**Orders:** {len(selected['order_history'])}
**Scenario:** `{SCENARIO_LABELS.get(selected['churn_scenario'], selected['churn_scenario'])}`
""")

st.sidebar.divider()
st.sidebar.markdown("*Built for Chatathon 2026 — Klaviyo Track*")
st.sidebar.markdown("*All data is simulated*")

# Main area
col1, col2 = st.columns([3, 2])

with col2:
    st.subheader("Tool Call Trace")
    trace_container = st.container(height=500)

with col1:
    st.subheader("Agent Investigation")

    if st.button("Investigate Customer", type="primary", use_container_width=True):
        import anthropic
        from agent import SYSTEM_PROMPT

        client = anthropic.Anthropic()

        messages = [
            {
                "role": "user",
                "content": f"Investigate customer {selected_id} who has been flagged as at-risk of churning. Use the tools available to understand their situation, determine the likely cause, and decide on the appropriate action. Think step by step about what you're seeing."
            }
        ]

        step_counter = 0
        thinking_placeholder = st.empty()
        result_placeholder = st.empty()

        with st.spinner("Agent is investigating..."):
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )

                assistant_content = response.content
                tool_use_blocks = []
                text_blocks = []

                for block in assistant_content:
                    if block.type == "text":
                        text_blocks.append(block.text)
                    elif block.type == "tool_use":
                        tool_use_blocks.append(block)

                # Show thinking
                if text_blocks:
                    thinking_text = "\n".join(text_blocks)
                    if tool_use_blocks:
                        # Still investigating — show as expandable
                        step_counter += 1
                        with thinking_placeholder.container():
                            with st.expander(f"Reasoning step {step_counter}", expanded=True):
                                st.markdown(thinking_text)
                    else:
                        # Final response
                        pass

                # If no tool calls, we're done
                if not tool_use_blocks:
                    final_text = "\n".join(text_blocks)

                    # Parse decision for color-coded card
                    decision = None
                    for d in ["ACT", "WAIT", "ESCALATE"]:
                        if f"**Decision:** {d}" in final_text or f"**Decision: {d}" in final_text:
                            decision = d
                            break

                    if decision:
                        color = DECISION_COLORS.get(decision, "#666")
                        result_placeholder.markdown(f"""
<div style='border-left: 4px solid {color}; padding: 16px; background-color: #0e1117; border-radius: 4px; margin-top: 16px;'>

{final_text}

</div>
""", unsafe_allow_html=True)
                    else:
                        result_placeholder.markdown(final_text)
                    break

                # Execute tool calls
                messages.append({"role": "assistant", "content": assistant_content})
                tool_results = []

                for tool_block in tool_use_blocks:
                    tool_name = tool_block.name
                    tool_input = tool_block.input

                    # Show in trace
                    with trace_container:
                        st.markdown(f"**{tool_name}**(`{tool_input.get('customer_id', '')}`)")

                    fn = TOOLS.get(tool_name)
                    if fn:
                        result = fn(**tool_input)
                    else:
                        result = f"Unknown tool: {tool_name}"

                    # Show result preview in trace
                    with trace_container:
                        preview = result[:200] + "..." if len(result) > 200 else result
                        with st.expander("Result", expanded=False):
                            st.code(result, language="json")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})
    else:
        st.info("Click **Investigate Customer** to start the agent's analysis.")
