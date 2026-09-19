"""Eval harness for the retention agent — runs the personas and checks decisions.

Usage:
    python eval.py              # run all cases once
    python eval.py C002         # run one or more specific cases
    python eval.py -n 5         # run each case 5 times, report a pass rate
    python eval.py -v           # also print each full agent response

KNOWN LIMITATIONS — read these before trusting a green table:

1. The C002 decline check is partly self-fulfilling. Critical Rule 1 in
   agent.py now instructs the model to "state plainly that you are declining
   to offer a discount", which is close to the exact string
   DISCOUNT_DECLINE_PATTERNS greps for. So a PASS there largely measures
   prompt compliance, not the agent's judgment. The unforced signal is the
   CONTRADICTION check: the draft message is not mandated the same way, so
   what the agent actually offers the customer is still worth measuring.

2. A single run is one sample of a nondeterministic system. C005 has returned
   both WAIT and ACT on identical input. Any case that has shown flakiness
   should be run with -n and read as a rate, not a boolean. A 1-run PASS is
   weak evidence; a 1-run FAIL is worth re-running before acting on.

3. Results are only comparable within one version of agent.py. The system
   prompt has already been rewritten mid-suite once, which silently mixed two
   populations into one table. Freeze agent.py before a run you intend to
   compare against anything.

4. Both C002 checks grep free-form prose, so unseen phrasing reads as a miss.
   A failing check means "go read the response", not "the agent is wrong". The
   structured alternative — a `discount_offered: false` field in the agent's
   output contract — would need a change to agent.py's output format, which is
   outside this harness's lane.
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

# Discount-shaped language in the *outward* message. Declining a discount in
# the rationale and then putting one in the draft is the failure this catches —
# and it's the half of C002 the prompt doesn't script, so it's the half that
# still measures behaviour rather than compliance.
DISCOUNT_OFFER_PATTERNS = [
    r"\b\d{1,2}\s*%",
    r"\$\s*\d+(?:\.\d{2})?\s*off\b",
    r"\bdiscount(?:ed|s)?\b",
    r"\bpromo(?:tion|tional)?\s*code\b",
    r"\bcoupon\b",
    r"\bspecial\s+(?:rate|price|pricing|offer|deal)\b",
    r"\breduced\s+(?:rate|price|pricing)\b",
    r"\b(?:take|get|save|knock)\s+\d+\s*%",
    r"\bmoney\s+off\b",
]

# If one of these sits just before a discount mention, the mention is the agent
# ruling a discount out rather than offering one.
NEGATION_CUES = (
    r"not\b|n't|without|\bno\b|never|rather than|instead of|avoid|stop|"
    r"free from|rely on|relying on|waiting for"
)


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


def extract_draft_message(text: str) -> str:
    """The outward-facing message only, or '' if the agent didn't draft one."""
    m = re.search(r"\**Draft Message:?\**\s*:?\s*", text, re.I)
    return text[m.end():].strip() if m else ""


def check_discount_decline(text: str):
    """Return (declined, matched_phrase) for explicit refusal-to-discount language."""
    flat = re.sub(r"\s+", " ", text)
    for pattern in DISCOUNT_DECLINE_PATTERNS:
        m = re.search(pattern, flat, re.I)
        if m:
            return True, m.group(0).strip()
    return False, ""


def find_discount_offer(draft: str):
    """Return (offered, phrase_with_context) for a live discount in the draft.

    Mentions preceded by a negation cue are the agent ruling a discount out
    ("...doesn't rely on waiting for occasional discounts"), not offering one,
    so they don't count.
    """
    flat = re.sub(r"\s+", " ", draft)
    for pattern in DISCOUNT_OFFER_PATTERNS:
        for m in re.finditer(pattern, flat, re.I):
            if re.search(NEGATION_CUES, flat[max(0, m.start() - 60):m.start()], re.I):
                continue
            lo, hi = max(0, m.start() - 35), min(len(flat), m.end() + 20)
            return True, f"...{flat[lo:hi].strip()}..."
    return False, ""


def check_c002(response: str):
    """Combine the two C002 checks into one verdict.

    PASS          — declined in the prose, nothing discount-shaped in the draft.
    CONTRADICTION — declined in the prose, but the draft discounts anyway.
    FAIL          — no explicit decline found at all.
    """
    declined, decline_phrase = check_discount_decline(response)
    offered, offer_phrase = find_discount_offer(extract_draft_message(response))

    if declined and offered:
        return "CONTRADICTION", f'declined "{decline_phrase}" but draft says {offer_phrase}'
    if declined:
        return "PASS", f'matched: "{decline_phrase}"'
    if offered:
        return "FAIL", f"no decline language, and draft discounts: {offer_phrase}"
    return "FAIL", "no explicit refusal-to-discount language found"


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


def parse_args(argv):
    """Return (cases, runs, verbose). Raises ValueError on bad input."""
    cases, runs, verbose = [], 1, False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-v", "--verbose"):
            verbose = True
        elif arg in ("-n", "--runs"):
            i += 1
            if i >= len(argv):
                raise ValueError("-n needs a number, e.g. -n 5")
            runs = int(argv[i])
        elif re.fullmatch(r"-n\d+", arg):
            runs = int(arg[2:])
        elif arg.startswith("-"):
            raise ValueError(f"unknown flag: {arg}")
        else:
            cases.append(arg.upper())
        i += 1

    if runs < 1:
        raise ValueError("-n must be at least 1")
    return cases, runs, verbose


def tally(decisions) -> str:
    """'WAIT x3, ACT x1' — most frequent first."""
    counts = {}
    for d in decisions:
        counts[d] = counts.get(d, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{d} x{n}" if n > 1 else d for d, n in ordered)


def run_case(cid: str, expected: str, runs: int, verbose: bool):
    """Run one persona `runs` times and fold the runs into a single row."""
    decisions, reasonings, c002_verdicts, errors = [], [], [], 0

    for attempt in range(runs):
        label = f"{cid} ({attempt + 1}/{runs})" if runs > 1 else cid
        print(f"Running {label} ...", file=sys.stderr, flush=True)
        try:
            response = run_with_retry(cid)["final_response"]
            decisions.append(extract_decision(response))
            reasonings.append(extract_reasoning(response))
        except Exception as exc:  # a crashed run is a failed run, not a dead harness
            response = ""
            errors += 1
            decisions.append("ERROR")
            reasonings.append(f"{type(exc).__name__}: {exc}")

        if cid == "C002" and response:
            c002_verdicts.append(check_c002(response))

        if verbose and response:
            header = f"{cid} run {attempt + 1}" if runs > 1 else cid
            print(f"\n{'-' * 70}\n{header} full response\n{'-' * 70}\n{response}\n")

    row = {
        "cid": cid,
        "expected": expected,
        "observed": tally(decisions),
        "passes": sum(1 for d in decisions if d == expected),
        "runs": runs,
        "errors": errors,
        # Show the first run that *didn't* match, so a mixed result explains its
        # failure rather than its success.
        "reasoning": next((r for d, r in zip(decisions, reasonings) if d != expected), reasonings[0]),
    }
    return row, c002_verdicts


def status_of(row) -> str:
    """PASS / FAIL(wrong) / FAIL(error) for one run; a rate for several."""
    passes, runs, errors = row["passes"], row["runs"], row["errors"]
    if runs == 1:
        if passes:
            return "PASS"
        return "FAIL(error)" if errors else "FAIL(wrong)"
    rate = f"{passes}/{runs}"
    if passes == runs:
        return rate
    if passes == 0:
        return f"{rate} {'ERR' if errors == runs else 'FAIL'}"
    return f"{rate} FLAKY"


def print_table(rows):
    headers = ("persona", "expected", "observed", "result", "reasoning")
    widths = (8, 9, 22, 12, 52)
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        cells = (
            row["cid"],
            row["expected"],
            truncate(row["observed"], widths[2]),
            status_of(row),
            truncate(row["reasoning"], widths[4]),
        )
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))


def report(rows, c002_verdicts, runs) -> int:
    total_runs = sum(r["runs"] for r in rows)
    total_passes = sum(r["passes"] for r in rows)
    clean = [r for r in rows if r["passes"] == r["runs"]]
    flaky = [r for r in rows if 0 < r["passes"] < r["runs"]]

    print(f"\nDecisions: {total_passes}/{total_runs} runs passed, "
          f"{len(clean)}/{len(rows)} personas clean")
    if flaky:
        print("FLAKY (nondeterministic on identical input): "
              + ", ".join(f"{r['cid']} {r['passes']}/{r['runs']}" for r in flaky))

    failed = len(clean) != len(rows)

    if c002_verdicts:
        verdicts = [v for v, _ in c002_verdicts]
        passed = verdicts.count("PASS")
        print(f"\nC002 discount check: {passed}/{len(verdicts)} passed")
        for i, (verdict, detail) in enumerate(c002_verdicts, 1):
            prefix = f"  run {i}: " if len(c002_verdicts) > 1 else "  "
            print(f"{prefix}{verdict}  ({detail})")
        if passed != len(verdicts):
            failed = True
        if "CONTRADICTION" in verdicts:
            print("  NOTE: CONTRADICTION means the rationale declined a discount but the draft\n"
                  "        message offers one anyway. The decline is prompt-scripted; the draft\n"
                  "        is not. Read the draft before trusting any prompt change that claims\n"
                  "        to have fixed C002.")

    if runs == 1:
        print("\nOne sample per case. See KNOWN LIMITATIONS at the top of this file —\n"
              "re-run anything surprising with -n before acting on it.")

    return 1 if failed else 0


def main() -> int:
    try:
        args, runs, verbose = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(f"{exc}\nUsage: python eval.py [CASE ...] [-n RUNS] [-v]")
        return 2

    selected = args or list(CASES)
    unknown = [c for c in selected if c not in CASES]
    if unknown:
        print(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(CASES)}")
        return 2

    rows, c002_verdicts = [], []
    for cid in selected:
        row, verdicts = run_case(cid, CASES[cid], runs, verbose)
        rows.append(row)
        c002_verdicts.extend(verdicts)

    print_table(rows)
    return report(rows, c002_verdicts, runs)


if __name__ == "__main__":
    sys.exit(main())
