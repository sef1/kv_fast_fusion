"""Count the outputs that are garbage, and say what BFF did to each of those requests.

F1 hides corruption: on GLM Magicoder-4096 (2026-11-02) a dedup run lost 8 F1 points while the
median per-prompt change was 0 — 13 of 121 outputs had collapsed into whitespace, ``]]]`` or
``texttexttext`` and the rest were fine. Fixed-output (THROUGHPUT_ONLY) runs have no F1 at all, and
had 10. A degenerate tail is cheap to detect: it compresses far better than code or prose, or it is
mostly punctuation.

With a decode log, each garbage request is joined with the scheduler's free-at-promotion trace:

* ``swapped`` — its own victim was swapped to a representative at promotion;
* ``owner`` — it owned a representative another request was swapped onto;
* ``recipient`` — it was promoted holding a block freed at an earlier promotion;
* ``none`` — BFF's free path never touched it.

Usage:
    python3 -m f1_benchmark.degenerate_outputs RESULTS_DIR [DECODE_LOG]
"""

import re
import sys
import zlib

from f1_benchmark.alias_attribution import load_ledger, load_outputs, uuid_of

TAIL_CHARS = 1500
MIN_CHARS = 300
MAX_ZLIB_RATIO = 0.12
MIN_ALNUM_SHARE = 0.2

_SWAP = re.compile(r"BFF swap \| req=(\S+) .* owner=(\S+)")
_REUSE = re.compile(r"BFF victim reuse \| req=(\S+) .* freed by (\S+)")


def is_degenerate(text: str) -> bool:
    """True if the last ``TAIL_CHARS`` are a repetition loop or mostly non-alphanumeric."""
    tail = (text or "")[-TAIL_CHARS:]
    if len(tail) < MIN_CHARS:
        return False
    raw = tail.encode()
    if len(zlib.compress(raw)) / len(raw) < MAX_ZLIB_RATIO:
        return True
    return sum(c.isalnum() for c in tail) / len(tail) < MIN_ALNUM_SHARE


def parse_trace(log_text: str) -> dict:
    """``{role: {uuid}}`` from the decode's ``BFF swap`` / ``BFF victim reuse`` lines."""
    roles = {"swapped": set(), "owner": set(), "recipient": set()}
    for m in _SWAP.finditer(log_text):
        roles["swapped"].add(uuid_of(m.group(1)))
        roles["owner"].add(uuid_of(m.group(2)))
    for m in _REUSE.finditer(log_text):
        roles["recipient"].add(uuid_of(m.group(1)))
    for ids in roles.values():
        ids.discard(None)
    return roles


def roles_of(key, roles: dict) -> list:
    got = [r for r in ("swapped", "owner", "recipient") if key in roles[r]]
    return got or ["none"]


def report(outputs, ledger=None, log_text=None) -> list:
    """Printable lines: the count, then one line per degenerate output."""
    ledger = ledger or {}
    bad = [(i, o) for i, o in enumerate(outputs) if is_degenerate(o.get("generated_text") or "")]
    lines = [f"degenerate outputs: {len(bad)}/{len(outputs)}"]
    roles = parse_trace(log_text) if log_text else None
    if roles is not None and not any(roles.values()):
        # No swap line at all: free-at-promotion was off, or the build predates the trace. Calling
        # every garbage request "none" would read as an exoneration the log cannot give.
        lines.append("  (no free-at-promotion trace in the decode log — attribution skipped)")
        roles = None
    tally: dict = {}
    for i, o in bad:
        key = uuid_of(o.get("response_id"))
        e = ledger.get(key) or {}
        line = (f"  #{i} {key or '?'} declined={e.get('declined', '-')}/{e.get('planned', '-')}"
                f"{' capped' if e.get('capped') else ''}")
        if roles is not None:
            rs = roles_of(key, roles)
            for r in rs:
                tally[r] = tally.get(r, 0) + 1
            line += " | " + ",".join(rs)
        lines.append(line)
    if roles is not None and bad:
        n_all = {r: len(ids) for r, ids in roles.items()}
        lines.insert(1, "  free-path roles of the garbage: "
                     + " ".join(f"{r}={tally.get(r, 0)}" for r in ("swapped", "owner", "recipient",
                                                                   "none"))
                     + f"  (run-wide: {n_all['swapped']} swapped, {n_all['owner']} owners, "
                       f"{n_all['recipient']} recipients)")
    return lines


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    outputs = load_outputs(argv[0])
    log_text = None
    if len(argv) > 1:
        try:
            with open(argv[1], errors="replace") as f:
                log_text = f.read()
        except OSError:
            log_text = None
    for line in report(outputs, load_ledger(argv[0]), log_text):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
