# Retention Agent

A reason-aware customer retention demo built for Chatathon 2026. The agent investigates at-risk PawBox subscribers, identifies the likely cause of churn, and recommends an appropriate next step instead of automatically offering a discount.

> **Demo notice:** All customer, order, support, subscription, and engagement data in this project is simulated.

## What It Does

The agent uses OpenAI tool calling to investigate a selected customer. It can retrieve:

- Customer profile and subscription status
- Order history, including delivery problems
- Email open and click activity
- Support tickets and customer message threads
- Subscription lifecycle events

After gathering enough evidence, the agent produces:

- Churn cause
- Confidence level
- Evidence summary
- One of three decisions: `ACT`, `WAIT`, or `ESCALATE`
- A recommended action, including a draft customer message for `ACT`

The Streamlit interface displays the agent's reasoning steps and a live tool-call trace.

## Scenarios Included

The simulated dataset contains five at-risk customer scenarios:

| Customer | Scenario |
| --- | --- |
| `C001` | Product mismatch |
| `C002` | Price sensitivity without default discounting |
| `C003` | Repeated service failure |
| `C004` | Life circumstance change |
| `C005` | False alarm |

## Requirements

- Python 3.9 or newer
- An OpenAI API key
- Dependencies listed in `requirements.txt`

## Setup

Create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set your OpenAI API key for the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

For a persistent user-level environment variable, use Windows' environment variable settings or:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-api-key", "User")
```

Do not commit API keys to the repository.

## Run the Streamlit App

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, select a customer in the sidebar, and click **Investigate Customer**.

## Run the CLI Agent

Run an investigation for a specific customer ID:

```powershell
python agent.py C001
```

If no ID is supplied, the CLI defaults to `C001`:

```powershell
python agent.py
```

The CLI prints tool calls, tool results, intermediate reasoning, and the final analysis.

## Project Structure

```text
.
├── agent.py              # OpenAI tool-calling loop and retention prompt
├── app.py                # Streamlit interface and investigation trace
├── tools.py              # Simulated customer-data tools and tool definitions
├── requirements.txt      # Python dependencies
└── data/
    └── personas.json     # Simulated customer records
```

## Decision Framework

The agent is instructed to investigate the underlying reason for churn risk before choosing an action:

- **ACT**: Evidence supports a specific, helpful first action. The agent drafts the message to send.
- **WAIT**: Signals are ambiguous. The agent specifies what to monitor and when to reassess.
- **ESCALATE**: The situation needs human judgment, such as a sensitive life circumstance or repeated operational failure.

Discounts are not treated as a default retention tactic. They are appropriate only when the evidence indicates price sensitivity and the customer still values the product.

## Notes

- The application currently uses the `gpt-4o` model in `agent.py` and `app.py`.
- Tool data is loaded locally from `data/personas.json`; no production systems are connected.
- The OpenAI API is called only when an investigation is started.
