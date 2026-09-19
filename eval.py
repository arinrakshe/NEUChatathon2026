"""Eval harness for the retention agent — runs the personas and checks decisions.

Usage:
    python eval.py              # run all cases once
    python eval.py C002         # run one or more specific cases
    python eval.py -n 5         # run each case 5 times, report a pass rate
    python eval.py -v           # also print each full agent response

KNOWN LIMITATIONS — read these before trusting a green table:

1. The C002 decline check is conditional, and still partly self-fulfilling.
   Critical Rule 1 in agent.py requires the decline sentence only "whenever
   price sensitivity is part of the diagnosis", so the check first looks for
   price language in the Churn Cause / Rationale (PRICE_IN_DIAGNOSIS_PATTERNS)
   and only demands a decline when it finds some. With no price in the
   diagnosis the verdict is N/A and the churn cause is printed for a human.

   N/A is not a pass, and on C002 it is a yellow flag rather than a neutral
   result. The persona's record still carries unambiguous price evidence — a
   2025-08-01 cancellation with reason "too_expensive", three resubscriptions
   landing within a day of a 20% / 30% / 25% promo, and email engagement that
   opens every promo and ignores every newsletter. A response that diagnoses
   only portion mismatch ("one box lasts 6+ weeks") has arguably missed the
   pattern C002 exists to test. The harness cannot distinguish that miss from a
   legitimately price-free diagnosis, which is exactly why the no-price branch
   prints evidence and defers instead of scoring. Do not read a table of N/As
   as a clean run.

   When it does grade, a PASS still largely measures prompt compliance: Rule 1
   scripts almost the exact string DISCOUNT_DECLINE_PATTERNS greps for. Two
   residual circularities. The price detector reads prose the same prompt
   shapes, and "promo" is one of its cues, so a rationale that declines
   "another promo code" reads as price-diagnosed and is then graded on the
   sentence it just supplied. And the gate can only ever loosen the check, so
   it must not be widened casually — the reason PRICE_IN_DIAGNOSIS_PATTERNS
   deliberately omits a bare \bdiscount\b.

   Because of all of the above, the decline check is ADVISORY: it prints its
   verdict and the text it matched or missed, but it does not set the exit
   code. It had already gone red on a correct response (limitation 4), and a
   check that reds on correct behaviour just teaches people to ignore red.

   The unforced signal remains the CONTRADICTION check, which the gate does not
   touch and which is the only branch that still hard-fails the run: the draft
   message is not mandated by the prompt, so what the agent actually offers the
   customer is the one thing here measuring behaviour rather than narration. A
   draft that discounts is graded regardless of the diagnosis — with no price
   in the diagnosis and a discount in the draft, that is FAIL, not N/A.

2. A single run is one sample of a nondeterministic system. C005 has returned
   both WAIT and ACT on identical input. Any case that has shown flakiness
   should be run with -n and read as a rate, not a boolean. A 1-run PASS is
   weak evidence; a 1-run FAIL is worth re-running before acting on.

3. Results are only comparable within one version of agent.py. The system
   prompt has already been rewritten mid-suite once, which silently mixed two
   populations into one table. Every run now prints a header naming the commit,
   whether the tree is dirty (and which result-affecting files are modified),
   and agent.MODEL, so a pasted table can identify its own agent version. A
   dirty tree means the commit does not pin the prompt: commit before a run you
   intend to compare against anything, and keep the header with the table.

4. Both C002 checks grep free-form prose, so unseen phrasing reads as a miss.
   A failing check means "go read the response", not "the agent is wrong".

   A structured `Discount Offered: YES/NO` field landed in agent.py at 3262d41.
   THIS HARNESS DOES NOT READ IT YET — check_c002 still greps prose. Whoever
   wires it up: assert on parse_analysis()["discount_offered"], which is
   YES / NO / None and is present on both the normal and the stopped_early
   return paths. Do not call the private _extract_field, and do not re-grep the
   markdown.

   Do not mistake the field for a fix. It removes the regex gap, but it is
   still the model narrating its own compliance in prose the same prompt
   shapes — it relocates the compliance problem rather than solving it. It also
   carries a definitional trap that has already bitten twice, once in the
   prompt and once in the parser: Rule 1's prescribed remedy for C002 is moving
   the customer to a cheaper plan or a longer cadence, which a model may
   truthfully describe as a discount. The field is therefore scoped — YES means
   a price cut, promo code, coupon, or bespoke rate on the CURRENT plan, and a
   plan or cadence change is NO. A YES/NO answer also tends to carry a trailing
   explanation containing the opposite word ("NO - a smaller plan is cheaper,
   yes, but not a discount"), which is why agent.py anchors on the leading
   token instead of substring-scanning; any check here must not undo that by
   scanning for "YES" anywhere in the value.

   When it is wired up, the contradiction check stays authoritative over the
   field: a response reporting NO whose draft discounts anyway is still
   CONTRADICTION. Self-report loses to what the customer would actually
   receive. The contradiction check remains the only branch measuring behaviour
   rather than narration.

   A worked example, reported from a 3-run C002 batch at ce0e94d (not verified
   in this file's own runs): all three responses had price in the diagnosis, so
   the gate in limitation 1 exempted none of them, and one still scored FAIL on
   the wording "Offering another discount would likely lead to a repeating
   cycle without addressing underlying issues." That is a decline. It misses
   because pattern 2 wants "is"/"would be" and gets "would likely lead",
   pattern 6 wants "does not"/"won't", and pattern 1's "without" branch needs a
   discount within four words. So at least one known FAIL is a regex gap, not
   an agent error, and the price gate does not rescue it.

   The fix was NOT another pattern. Rule 1 already scripts this wording, so
   each pattern added to match observed output moves the check further toward
   measuring itself; a green result bought that way means less, not more. That
   example is why the decline check is advisory and why it prints the Rationale
   on a FAIL: read it and overrule the regex, rather than teaching the regex
   the answer.
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime

import agent
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

# Section labels from agent.py's output format, used to slice one section out of
# a response without swallowing the next one.
SECTION_LABELS = (
    "Churn Cause", "Confidence", "Evidence Summary", "Decision",
    "Rationale", "Action", "Draft Message",
)

# Files whose contents change what a run means. Named in the header when dirty.
RESULT_AFFECTING_FILES = ("agent.py", "tools.py", "data/personas.json", "eval.py")

# C002 verdicts that are reported but do NOT set the exit code. The decline
# check greps prose whose wording Rule 1 already dictates, so it cannot tell a
# real narration gap from a decline phrased outside the patterns — and it has
# gone red on a correct response at least once. A check that reds on correct
# behaviour teaches people to ignore red, so it advises instead of failing.
# CONTRADICTION is deliberately absent: the draft is not prompt-scripted.
ADVISORY_C002_VERDICTS = ("PASS", "FAIL", "N/A")

# Price language in the *diagnosis*. Rule 1 requires the decline sentence only
# "whenever price sensitivity is part of the diagnosis", so the decline check is
# gated on this rather than run unconditionally.
#
# Deliberately narrow, and deliberately NOT a bare \bdiscount\b: the mandated
# decline sentence itself contains the word "discount", so matching that word
# would make every declining response look price-diagnosed by construction.
# False positives here are the safe direction — they only move a run into the
# graded branch, which demands more of the agent, not less.
PRICE_IN_DIAGNOSIS_PATTERNS = [
    r"\bcosts?\b|\bcostly\b",
    r"\bpric(?:e|es|ed|ing|ey)\b",
    r"\bexpensive\b",
    r"\bafford\w*\b",
    r"\bdiscount[-\s]depend\w*\b|\bdepend\w*\s+on\s+discounts?\b|\bpromo[-\s]to[-\s]promo\b",
    r"\bpromo\w*\b",
]

# C002 is the discount-bait persona: getting ACT right is not enough, the
# response should also say out loud that a discount is not the right move —
# but only when price is actually part of the diagnosis. See check_c002.
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


def extract_section(text: str, label: str) -> str:
    """The body of one '**Label:**' section, up to the next known label or the end."""
    m = re.search(rf"\**{label}:?\**\s*:?\s*", text, re.I)
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(
        r"\n[^\S\n]*\**\s*(?:" + "|".join(SECTION_LABELS) + r")\s*\**\s*:",
        rest,
        re.I,
    )
    body = rest[: nxt.start()] if nxt else rest
    return re.sub(r"\s+", " ", body.strip().strip("[]*").strip())


def check_price_in_diagnosis(text: str):
    """Return (present, phrase) for price language in the Churn Cause / Rationale.

    Gates the decline check: Rule 1 only asks for the decline sentence when
    price sensitivity is part of the diagnosis, so a response with no price
    dimension shouldn't be marked down for not declining one.

    Do not expect this gate to fire on C002. Its record is still saturated with
    price (see limitation 1), and every observed C002 response so far has put
    price or promo language in the diagnosis. The gate is here because the
    decline requirement is conditional in the prompt, not because C002 is
    expected to trip it — on C002 an N/A is a signal to go read the response.

    Scoped to the two diagnosis sections on purpose: price language in the
    Evidence Summary is the agent reciting the account history, not concluding
    anything about it.
    """
    diagnosis = " ".join(
        s for s in (extract_section(text, "Churn Cause"), extract_section(text, "Rationale")) if s
    )
    for pattern in PRICE_IN_DIAGNOSIS_PATTERNS:
        m = re.search(pattern, diagnosis, re.I)
        if m:
            lo, hi = max(0, m.start() - 30), min(len(diagnosis), m.end() + 30)
            return True, f"...{diagnosis[lo:hi].strip()}..."
    return False, ""


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
    """Combine the C002 checks into one verdict, conditional on the diagnosis.

    Only CONTRADICTION sets the exit code. Everything else here is ADVISORY —
    see ADVISORY_C002_VERDICTS and limitation 1.

    CONTRADICTION — the draft discounts after the prose declined. Checked first
                    and regardless of the diagnosis: whatever the agent
                    concluded, saying one thing and sending another is wrong.
                    The draft is not scripted by the prompt, so this is the one
                    branch measuring behaviour rather than narration. Hard fail.
    FAIL          — the draft discounts with no decline anywhere. Also checked
                    ahead of the price gate: Rule 1 says never DEFAULT to a
                    discount, so discounting a diagnosis that isn't even about
                    price is the C002 trap, not an N/A.
    N/A           — no price language in the diagnosis and nothing discount-
                    shaped in the draft. Rule 1 doesn't ask for a decline
                    sentence here, so there is nothing to grade. NOT a pass:
                    the churn cause is printed for a human to judge.
    PASS          — price is in the diagnosis, the prose declines, draft clean.
    FAIL          — price is in the diagnosis and the prose never declines. The
                    narration gap Rule 1 is about — but also what a decline the
                    patterns don't recognise looks like, which is why this one
                    prints the Rationale and doesn't fail the run.
    """
    priced, price_phrase = check_price_in_diagnosis(response)
    declined, decline_phrase = check_discount_decline(response)
    offered, offer_phrase = find_discount_offer(extract_draft_message(response))

    if offered:
        if declined:
            return "CONTRADICTION", f'declined "{decline_phrase}" but draft says {offer_phrase}'
        return "FAIL", f"no decline language, and draft discounts: {offer_phrase}"

    if not priced:
        cause = extract_section(response, "Churn Cause") or "(no Churn Cause line)"
        return "N/A", f"price not in diagnosis — judge by hand. Churn Cause: {cause}"

    if declined:
        return "PASS", f'price in diagnosis ("{price_phrase}"); matched: "{decline_phrase}"'
    # Print the prose the patterns failed on. A decline phrased outside
    # DISCOUNT_DECLINE_PATTERNS is indistinguishable from no decline at all, and
    # that has already happened once ("would likely lead", limitation 4), so the
    # text goes in the output where a human can overrule the regex.
    rationale = extract_section(response, "Rationale") or "(no Rationale line)"
    return "FAIL", (f'price in diagnosis ("{price_phrase}") but no pattern matched a '
                    f'refusal. Read the Rationale: {truncate(rationale, 240)}')


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def git_provenance():
    """Return (short_sha, tree_state) for the checkout this harness lives in."""
    repo = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(
            ("git",) + args, cwd=repo, capture_output=True, text=True, timeout=10
        )

    try:
        head = git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return "unknown", "not a git checkout"
        status = git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return "unknown", "git unavailable"

    sha = head.stdout.strip() or "unknown"
    if status.returncode != 0:
        return sha, "dirty state unknown"

    changed = {line[3:].strip().strip('"').replace("\\", "/")
               for line in status.stdout.splitlines() if line.strip()}
    if not changed:
        return sha, "clean"

    # Name the dirty files that actually change results — "dirty" alone doesn't
    # tell you whether the agent under test differs from the commit.
    relevant = sorted(f for f in RESULT_AFFECTING_FILES if f in changed)
    return sha, "dirty" + (f" — modified: {', '.join(relevant)}" if relevant else "")


def print_header(selected, runs):
    """Name the agent version this table came from, before the table exists.

    Results have been silently pooled across two versions of the system prompt
    once already. A table that can't identify its own agent isn't evidence.
    """
    sha, tree = git_provenance()
    model = getattr(agent, "MODEL", "unknown")
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"eval run {stamp}")
    print(f"  commit: {sha} ({tree})")
    print(f"  model:  {model}  (agent.MODEL)")
    print(f"  cases:  {', '.join(selected)}  x{runs} run{'s' if runs != 1 else ''}")


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
        skipped = verdicts.count("N/A")
        graded = len(verdicts) - skipped
        summary = f"{passed}/{graded} graded runs declined explicitly" if graded else "nothing graded"
        if skipped:
            summary += f", {skipped} N/A (price not in diagnosis)"
        print(f"\nC002 discount check: {summary}")
        print("  Decline check is ADVISORY — it does not set the exit code. Only "
              "CONTRADICTION does.")
        for i, (verdict, detail) in enumerate(c002_verdicts, 1):
            prefix = f"  run {i}: " if len(c002_verdicts) > 1 else "  "
            tag = " [advisory]" if verdict in ADVISORY_C002_VERDICTS else " [HARD FAIL]"
            print(f"{prefix}{verdict}{tag}  ({detail})")
        if any(v not in ADVISORY_C002_VERDICTS for v in verdicts):
            failed = True
        if skipped:
            print("  NOTE: N/A is not a pass — it means the decline sentence wasn't required,\n"
                  "        so nothing was graded. Read the churn cause above. C002's record does\n"
                  "        contain price evidence (a 'too_expensive' cancellation, three resubs on\n"
                  "        promo codes, promo-only email engagement), so a diagnosis that never\n"
                  "        registers price is more likely a miss than a clean no-price read. The\n"
                  "        harness can't tell those two apart; that judgement is yours.")
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

    print_header(selected, runs)

    rows, c002_verdicts = [], []
    for cid in selected:
        row, verdicts = run_case(cid, CASES[cid], runs, verbose)
        rows.append(row)
        c002_verdicts.extend(verdicts)

    print_table(rows)
    return report(rows, c002_verdicts, runs)


if __name__ == "__main__":
    sys.exit(main())
