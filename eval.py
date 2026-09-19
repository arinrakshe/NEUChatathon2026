"""Eval harness for the retention agent — runs the personas and checks decisions.

Usage:
    python eval.py              # run all cases
    python eval.py C002         # run one or more specific cases
    python eval.py -v           # also print each full agent response
"""

import re
import sys
import time

from agent import run_agent

# Expected decision per persona.
CASES = {
    "C001": "ACT",
    "C002": "ACT",
    "C003": "ESCALATE",
    "C004": "ESCALATE",
    "C005": "WAIT",
}

DECISIONS = ("ACT", "WAIT", "ESCALATE")

# C002 is the discount-bait persona: getting ACT right is not enough, the
# response should also say out loud that a discount is not the right move.
DISCOUNT_DECLINE_PATTERNS = [
    r"(?:not|avoid|never|rather than|instead of|without)\s+(?:\w+\s+){0,4}?discount",
    r"discount(?:ing|s)?\s+(?:\w+\s+){0,4}?(?:is|would be|isn't|is not|are not)\s+(?:\w+\s+){0,3}?"
    r"(?:not|inappropriate|wasteful|tone-deaf|premature|unnecessary|wrong)",
    r"no\s+discount",
    r"(?:decline|declining|refrain from|hold off on|resist)\s+(?:\w+\s+){0,4}?discount",
    r"discount\s+is\s+not\s+(?:appropriate|warranted|justified|needed)",
    # "another discount does not resolve ...", "a discount won't address ..."
    r"discount\w*\s+(?:\w+\s+){0,3}?(?:does not|doesn't|did not|will not|won't|would not|"
    r"wouldn't|fails to|only masks|merely)",
]


def extract_decision(text: str) -> str:
    """Pull the ACT/WAIT/ESCALATE decision out of the agent's final analysis."""
    # Preferred: the structured '**Decision:** ACT' line from the output format.
    m = re.search(r"\**Decision:?\**\s*:?\s*\[?\**\s*(ACT|WAIT|ESCALATE)", text, re.I)
    if m:
        return m.group(1).upper()
    # Fallback: last standalone keyword anywhere in the response.
    hits = re.findall(r"\b(ACT|WAIT|ESCALATE)\b", text)
    return hits[-1].upper() if hits else "UNPARSED"


def extract_reasoning(text: str) -> str:
    """One-line 'why' — prefer the churn diagnosis, fall back to the rationale."""
    for label in ("Churn Cause", "Rationale", "Decision"):
        m = re.search(rf"\**{label}:?\**\s*:?\s*(.+)", text, re.I)
        if m:
            line = m.group(1).strip().strip("[]*").strip()
            if line:
                return re.sub(r"\s+", " ", line)
    return re.sub(r"\s+", " ", text.strip())[:200] or "(empty response)"


def check_discount_decline(text: str):
    """Return (declined, matched_phrase) for explicit refusal-to-discount language."""
    flat = re.sub(r"\s+", " ", text)
    for pattern in DISCOUNT_DECLINE_PATTERNS:
        m = re.search(pattern, flat, re.I)
        if m:
            return True, m.group(0).strip()
    return False, ""


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def run_with_retry(cid: str, attempts: int = 3):
    """Run the agent, retrying transient rate-limit / connection errors."""
    for attempt in range(attempts):
        try:
            return run_agent(cid)
        except Exception as exc:
            transient = any(
                word in type(exc).__name__.lower() for word in ("ratelimit", "connection", "timeout")
            ) or "429" in str(exc)
            if not transient or attempt == attempts - 1:
                raise
            delay = 10 * (attempt + 1)
            print(f"  {type(exc).__name__} on {cid}, retrying in {delay}s ...", file=sys.stderr, flush=True)
            time.sleep(delay)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    selected = [c.upper() for c in args] or list(CASES)

    unknown = [c for c in selected if c not in CASES]
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(CASES)}")
        return 2

    rows = []
    discount_check = None

    for cid in selected:
        expected = CASES[cid]
        print(f"Running {cid} ...", file=sys.stderr, flush=True)
        try:
            result = run_with_retry(cid)
            response = result["final_response"]
            actual = extract_decision(response)
            reasoning = extract_reasoning(response)
        except Exception as exc:  # a crashed run is a failed case, not a dead harness
            response = ""
            actual = "ERROR"
            reasoning = f"{type(exc).__name__}: {exc}"

        rows.append((cid, expected, actual, actual == expected, reasoning))

        if cid == "C002":
            discount_check = check_discount_decline(response)

        if verbose and response:
            print(f"\n{'-' * 70}\n{cid} full response\n{'-' * 70}\n{response}\n")

    # Table
    headers = ("persona", "expected", "actual", "result", "reasoning")
    widths = (8, 9, 9, 6, 62)
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for cid, expected, actual, ok, reasoning in rows:
        cells = (cid, expected, actual, "PASS" if ok else "FAIL", truncate(reasoning, widths[4]))
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))

    passed = sum(1 for r in rows if r[3])
    print(f"\nDecisions: {passed}/{len(rows)} passed")

    # Separate C002 check: did it explicitly decline to discount?
    extra_failed = False
    if discount_check is not None:
        declined, phrase = discount_check
        if declined:
            print(f'C002 discount-decline check: PASS  (matched: "{phrase}")')
        else:
            print("C002 discount-decline check: FAIL  (no explicit refusal-to-discount language found)")
            extra_failed = True

    return 0 if passed == len(rows) and not extra_failed else 1


if __name__ == "__main__":
    sys.exit(main())
