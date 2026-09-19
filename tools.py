"""Simulated data tools the retention agent can call."""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "personas.json"

def _load():
    with open(DATA_PATH) as f:
        return json.load(f)["customers"]

def get_customer_summary(customer_id: str) -> str:
    """Return basic profile + current subscription status for a customer."""
    customers = _load()
    c = customers.get(customer_id)
    if not c:
        return f"No customer found with ID {customer_id}"
    return json.dumps({
        "customer_id": customer_id,
        "name": c["name"],
        "pet_name": c["pet_name"],
        "pet_type": c["pet_type"],
        "pet_breed": c["pet_breed"],
        "pet_age_months": c["pet_age_months"],
        "plan": c["plan"],
        "monthly_price": c["monthly_price"],
        "member_since": c["member_since"],
        "status": c["status"],
        "total_orders": len(c["order_history"]),
    }, indent=2)


def get_order_history(customer_id: str) -> str:
    """Return full order history for a customer."""
    customers = _load()
    c = customers.get(customer_id)
    if not c:
        return f"No customer found with ID {customer_id}"
    return json.dumps(c["order_history"], indent=2)


def get_email_engagement(customer_id: str) -> str:
    """Return email open/click history for a customer."""
    customers = _load()
    c = customers.get(customer_id)
    if not c:
        return f"No customer found with ID {customer_id}"
    return json.dumps(c["email_engagement"], indent=2)


def get_support_tickets(customer_id: str) -> str:
    """Return all support tickets and message threads for a customer."""
    customers = _load()
    c = customers.get(customer_id)
    if not c:
        return f"No customer found with ID {customer_id}"
    if not c["support_tickets"]:
        return "No support tickets found for this customer."
    return json.dumps(c["support_tickets"], indent=2)


def get_subscription_events(customer_id: str) -> str:
    """Return subscription lifecycle events (subscribe, skip, pause, cancel, renewal dates)."""
    customers = _load()
    c = customers.get(customer_id)
    if not c:
        return f"No customer found with ID {customer_id}"
    return json.dumps(c["subscription_events"], indent=2)


# Registry for the agent to use
TOOLS = {
    "get_customer_summary": get_customer_summary,
    "get_order_history": get_order_history,
    "get_email_engagement": get_email_engagement,
    "get_support_tickets": get_support_tickets,
    "get_subscription_events": get_subscription_events,
}

TOOL_DEFINITIONS = [
    {
        "name": "get_customer_summary",
        "description": "Get basic profile and current subscription status for a customer. Returns name, pet info, plan, price, member since date, status, and total order count.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID (e.g. C001)"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_order_history",
        "description": "Get the full order history for a customer, including items, dates, delivery status, and any damage or delay notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID (e.g. C001)"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_email_engagement",
        "description": "Get email engagement history showing which emails (newsletters, promos) were opened and clicked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID (e.g. C001)"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_support_tickets",
        "description": "Get all support tickets and full message threads for a customer. Contains the actual customer and agent messages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID (e.g. C001)"}
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "get_subscription_events",
        "description": "Get subscription lifecycle events: subscribes, cancellations, resubscriptions, skips, pauses, and upcoming renewal dates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID (e.g. C001)"}
            },
            "required": ["customer_id"]
        }
    }
]
