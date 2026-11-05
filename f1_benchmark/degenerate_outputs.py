"""Count the outputs that are garbage, and say what BFF did to each of those requests.

F1 hides corruption: on GLM Magicoder-4096 (2026-11-02) a dedup run lost 8 F1 points while the
median per-prompt change was 0 — 13 of 121 outputs had collapsed into whitespace, ``]]]`` or
``texttexttext`` and the rest were fine. Fixed-output (THROUGHPUT_ONLY) runs have no F1 at all, and
had 10. A degenerate tail is cheap to detect: it compresses far better than code or prose, or it is
mostly punctuation.

That misses garbage that varies. KV shifted by one block gives ``pythonpython`` or `` or or or)``, and
such answers can even recover into plausible code, so the tail looks fine: prompt 54 of the 121-prompt
set was wrong in every BFF run while this reported 0. :func:`is_garbled` reads the opening instead.

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
GARBLE_HEAD_CHARS = 80
GARBLE_SHORT_CHARS = 32
GARBLE_MIN_LETTER_SHARE = 0.5
_RESTART = re.compile(r"[^\n]```(?:python|html|sql|css|javascript|js|bash|json|yaml|xml|java|cpp|go|typescript|ts)?\n"
                      r"|[^\n]## Code: ")
_NORMAL_OPENING = ("```", "#", "<", "{", "[", "*", '"', "'", "-", "/", "(")

_SWAP = re.compile(r"BFF swap \| req=(\S+) .* owner=(\S+)")
_REUSE = re.compile(r"BFF victim reuse \| req=(\S+) .* freed by (\S+)")
# kv_fast_fusion_ascend/shared_write_check.py
_SHARED_WRITE = re.compile(r"BFF shared-block write \| req=(\S+) .*other holders: (.*)$", re.M)
_PREEMPT = re.compile(r"BFF preempt \| req=(\S+)")
_RECOMPUTE = re.compile(r"BFF recompute \| req=(\S+)")
_LANDED_SHARED = re.compile(r"BFF landed-into-shared \| req=(\S+) .*other holders: (.*)$", re.M)
_PULL_SHARED = re.compile(r"BFF pull-into-shared \| req=(\S+)")
_STALE_HASH = re.compile(r"BFF stale-hash block \| req=(\S+)")
# kv_fast_fusion/fast_fusion_block_pool.py (BFF_V2_ALIAS_TRACE)
_ALIAS_HIT = re.compile(r"BFF alias hit \| req=(\S+) g=\d+ pos=\d+ block=(\d+) via=(\S+)")
_REARRIVAL = re.compile(r"BFF rearrival \| req=(\S+)")
_SWAP_REP = re.compile(r"BFF swap \| req=\S+ .* rep=(\d+) owner=(\S+)")
# written/writer: a scheduled write into a shared block. landed_*: a landing copy into one.
# alias_hitter: a prefix hit through an alias or on a freed victim; hit_owner: owner of the hit rep.
EVENTS = ("written", "writer", "preempted", "recomputed", "landed_writer", "landed_holder",
          "pull_shared", "alias_hitter", "hit_owner", "rearrived", "stale_hash")


def is_degenerate(text: str) -> bool:
    """True if the last ``TAIL_CHARS`` are a repetition loop or mostly non-alphanumeric."""
    tail = (text or "")[-TAIL_CHARS:]
    if len(tail) < MIN_CHARS:
        return False
    raw = tail.encode()
    if len(zlib.compress(raw)) / len(raw) < MAX_ZLIB_RATIO:
        return True
    return sum(c.isalnum() for c in tail) / len(tail) < MIN_ALNUM_SHARE


def is_garbled(text: str) -> bool:
    """True if the answer is broken from its first token: an abnormal opening and bad text after it.

    An answer normally opens with a capital, a digit or markup. An abnormal opening alone is not
    enough, because a dropped first token leaves `` Code: models.py`` or ``python\\n## user.py`` with
    good text behind it. Those are told apart by what follows: a very short answer, stray punctuation
    first, a doubled word (``or or or``), a doubled fragment (``pythonpython``), a repeating fragment
    (``::::``, ``ZOZOZO``), or an opening that is mostly not letters. Thresholds come from every saved
    run: all 19 known-bad answers caught, none of 1,153 baseline answers flagged. It cannot see garbage
    that opens normally. Empty answers are failed requests rather than garbage, so they are not counted."""
    text = text or ""
    if not text.strip() or text[0].isupper() or text[0].isdigit() or text.startswith(_NORMAL_OPENING):
        return False
    if len(text.strip()) < GARBLE_SHORT_CHARS:
        return True
    head = text[:GARBLE_HEAD_CHARS]
    first = text.lstrip()[0]
    if not first.isalnum() and first not in "_`#":
        return True
    if re.search(r"(\S{1,3})\1{3,}", head):
        return True
    words = head.split()[:12]
    if any(a == b for a, b in zip(words, words[1:])):
        return True
    if re.match(r"\s*(\w{3,})\1", head):
        return True
    return sum(c.isalpha() for c in head) / len(head) < GARBLE_MIN_LETTER_SHARE


def has_restart(text: str) -> bool:
    """True if the answer was cut off and begins again, the mark of an eviction under the stock proxy.

    ``__tablename__ =```python``: a code fence or a ``## Code:`` heading glued onto unfinished text. Not
    garbage in the sense above, and it hits baseline too, but such answers score lower (F1 0.51 against
    0.64 on families at limit 24). Fixed-output runs also trip it on text forced past the answer's end."""
    return _RESTART.search(text or "") is not None


def is_garbage(text: str) -> bool:
    return is_degenerate(text) or is_garbled(text)


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


def parse_events(log_text: str) -> dict:
    """``{event: {uuid}}`` from the shared-write check and the alias trace (see ``EVENTS``).

    ``hit_owner`` is approximate: a rep id is mapped to every owner it was ever swapped onto, and
    block ids are reused over a run."""
    ev = {e: set() for e in EVENTS}
    for m in _SHARED_WRITE.finditer(log_text):
        ev["writer"].add(uuid_of(m.group(1)))
        for holder in m.group(2).split():
            ev["written"].add(uuid_of(holder.split("@")[0]))
    for m in _PREEMPT.finditer(log_text):
        ev["preempted"].add(uuid_of(m.group(1)))
    for m in _RECOMPUTE.finditer(log_text):
        ev["recomputed"].add(uuid_of(m.group(1)))
    for m in _LANDED_SHARED.finditer(log_text):
        ev["landed_writer"].add(uuid_of(m.group(1)))
        for holder in m.group(2).split():
            ev["landed_holder"].add(uuid_of(holder.split("@")[0]))
    for m in _PULL_SHARED.finditer(log_text):
        ev["pull_shared"].add(uuid_of(m.group(1)))
    rep_owner: dict = {}
    for m in _SWAP_REP.finditer(log_text):
        rep_owner.setdefault(m.group(1), set()).add(uuid_of(m.group(2)))
    for m in _ALIAS_HIT.finditer(log_text):
        ev["alias_hitter"].add(uuid_of(m.group(1)))
        if m.group(3) == "alias":
            ev["hit_owner"].update(rep_owner.get(m.group(2), ()))
    for m in _REARRIVAL.finditer(log_text):
        ev["rearrived"].add(uuid_of(m.group(1)))
    for m in _STALE_HASH.finditer(log_text):
        ev["stale_hash"].add(uuid_of(m.group(1)))
    for ids in ev.values():
        ids.discard(None)
    return ev


def roles_of(key, roles: dict) -> list:
    got = [r for r in ("swapped", "owner", "recipient") if key in roles[r]]
    return got or ["none"]


def report(outputs, ledger=None, log_text=None) -> list:
    """Printable lines: the count, then one line per degenerate output."""
    ledger = ledger or {}
    bad = [(i, o) for i, o in enumerate(outputs) if is_garbage(o.get("generated_text") or "")]
    # Only those the tail check would have missed, so an old run's report keeps its meaning.
    garbled = sum(not is_degenerate(o.get("generated_text") or "") for _i, o in bad)
    lines = [f"degenerate outputs: {len(bad)}/{len(outputs)}"
             + (f" ({garbled} garbled from the first token)" if garbled else "")]
    restarted = sum(has_restart(o.get("generated_text") or "") for o in outputs)
    if restarted:
        lines[0] += f" | restarted by an eviction: {restarted}"
    roles = parse_trace(log_text) if log_text else None
    if roles is not None and not any(roles.values()):
        # No swap line at all: free-at-promotion was off, or the build predates the trace. Calling
        # every garbage request "none" would read as an exoneration the log cannot give.
        lines.append("  (no free-at-promotion trace in the decode log — attribution skipped)")
        roles = None
    events = parse_events(log_text) if log_text else None
    if events is not None and not any(events.values()):
        events = None             # no shared-write check in this build, or nothing to say
    tally: dict = {}
    ev_tally: dict = {}
    for i, o in bad:
        key = uuid_of(o.get("response_id"))
        e = ledger.get(key) or {}
        line = (f"  #{i} {key or '?'} declined={e.get('declined', '-')}/{e.get('planned', '-')}"
                f"{' capped' if e.get('capped') else ''}"
                f"{' clipped=' + str(e['clipped']) if e.get('clipped') else ''}")
        if roles is not None:
            rs = roles_of(key, roles)
            for r in rs:
                tally[r] = tally.get(r, 0) + 1
            line += " | " + ",".join(rs)
        if events is not None:
            hit = [ev for ev in EVENTS if key in events[ev]]
            for ev in hit:
                ev_tally[ev] = ev_tally.get(ev, 0) + 1
            if hit:
                line += " | " + ",".join(hit)
        lines.append(line)
    clipped = {k for k, e in ledger.items() if e.get("clipped")}
    if clipped:
        # BFF_V2_CLIP_FRAC's verdict: the capped requests it kept partially, and how many broke.
        n_bad = sum(1 for _i, o in bad if uuid_of(o.get("response_id")) in clipped)
        lines.insert(1, f"  clipped requests (BFF_V2_CLIP_FRAC): {n_bad}/{len(clipped)} degenerate, "
                        f"{sum(ledger[k]['clipped'] for k in clipped)} aliases kept")
    if events is not None:
        lines.insert(1, "  shared-write/lifecycle events on the garbage: "
                     + " ".join(f"{ev}={ev_tally.get(ev, 0)}" for ev in EVENTS)
                     + "  (run-wide: " + ", ".join(f"{len(events[ev])} {ev}" for ev in EVENTS)
                     + ")")
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
