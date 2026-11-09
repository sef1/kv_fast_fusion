"""Answers that wrote a sibling prompt's content instead of their own.

The failure this catches: with a lot of a request's KV replaced from a near-identical sibling, the model
can answer the SIBLING's request under its own header. Prompt 18 of the families set asks for
``models.py`` and, with no cap, wrote the sibling ``forms.py`` code under a correct ``## models.py``.
Neither F1 nor the right-file check sees it: the header is right and the two files share most of their
characters (prompt 18 scored F1 above its baseline).

The rule, per answer, against the baseline runs of the same prompts:

* ``own``   — similarity to this prompt's own baseline answer (the best over the baseline repeats);
* ``other`` — similarity to the baseline answer of a SIBLING: a prompt within ``WINDOW`` places in arrival
  order (requests that can be in the decode together) whose text is near-identical (char 5-gram Jaccard at
  least ``PROMPT_SIM``);
* flagged when ``other > own + MARGIN``.

Similarity is the Jaccard of character 5-grams over the first ``HEAD`` characters of each answer.

A baseline run is judged against the OTHER baseline repeats, which gives the control: how often the model
does this on its own. It needs at least two baseline repeats, and a run with no baseline of the same
prompts gets no count. Fixed-length runs are not judged: text forced past the answer's end makes even two
baseline repeats look unlike each other, so the rule flags baselines there.

Validated on the client table runs (2026-11-08): it flags the 4 confirmed no-cap copies (prompt 18 in
both repeats, 62 and 150 once each) and one further no-cap answer with nothing replaced (prompt 150,
repeat 1). It flags 0 in clip 0.7, 1 of 438 in the baseline control, and 0 in the 121 and 200 sets.
The baseline flag is a known false one: prompt 205 asks to change an SVG given in the prompt, and its
answer repeats that SVG, which its sibling's answer also repeats. Leaving out text copied from the prompt
removes it, but adds false flags in clip 0.7 and the 121 set and shrinks the margin on real copies, so the
rule stays as it is. Read every flagged answer before quoting a number.

A check WITHOUT a baseline (an answer unusually close to a sibling's answer in the same run) was tried
and is not reliable: sibling answers in a normal run are often that close, so the copies do not stand out.
"""

import glob
import json
import os
import re

GRAM = 5
HEAD = 3000          # characters of each answer compared
MARGIN = 0.1         # how much closer to the sibling than to its own baseline
PROMPT_SIM = 0.6     # prompts at least this similar are siblings; unrelated template prompts sit below 0.5
WINDOW = 16          # siblings are at most this many places apart in arrival order
CODE_REQUEST = re.compile(r"## Code: ([^\s]+)")


class _Grams:
    """Character n-gram sets, cached per text."""

    def __init__(self):
        self._cache = {}

    def __call__(self, text: str, head: int | None = HEAD) -> frozenset:
        t = (text or "")[:head] if head else (text or "")
        g = self._cache.get(t)
        if g is None:
            g = self._cache[t] = frozenset(t[i:i + GRAM] for i in range(max(1, len(t) - GRAM + 1)))
        return g


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / max(1, len(a | b))


def siblings(prompts: list, window: int = WINDOW, min_sim: float = PROMPT_SIM, grams=None) -> list:
    """For each prompt, the indices of its siblings: near-identical prompts close in arrival order."""
    grams = grams or _Grams()
    out = [[] for _ in prompts]
    for i in range(len(prompts)):
        for j in range(i + 1, min(len(prompts), i + window + 1)):
            if jaccard(grams(prompts[i], None), grams(prompts[j], None)) >= min_sim:
                out[i].append(j)
                out[j].append(i)
    return out


def drifted(texts: list, refs: list, prompts: list, margin: float = MARGIN, window: int = WINDOW) -> list:
    """Answers closer to a sibling's reference answer than to their own.

    ``texts`` are one run's answers, ``refs`` a list of reference runs' answers (same prompts, same
    order). Returns ``[(i, j, own, other)]``: answer i looks like sibling j's."""
    grams = _Grams()
    sibs = siblings(prompts, window, grams=grams)
    hits = []
    for i, text in enumerate(texts):
        if not text or not sibs[i]:
            continue
        g = grams(text)
        own = max((jaccard(g, grams(r[i])) for r in refs if r[i]), default=0.0)
        other, j = max(((max(jaccard(g, grams(r[k])) for r in refs), k) for k in sibs[i]), default=(0.0, None))
        if j is not None and other > own + margin:
            hits.append((i, j, own, other))
    return hits


def _ledger(run_dir: str) -> dict:
    """``response_id -> {declined, planned, ...}`` from the BFF stats files, if any."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "results_latest", "bff_stats_*.json")):
        try:
            with open(path) as f:
                out.update(json.load(f).get("request_ledger") or {})
        except (OSError, ValueError):
            pass
    return out


def describe(run: dict, hit: tuple) -> str:
    """One readable line for a flagged answer, with how much of its KV BFF replaced when known."""
    i, j, own, other = hit
    outs = run.get("outputs") or []
    asked = lambda k: (CODE_REQUEST.findall(outs[k].get("prompt") or "") or ["?"])[-1] if k < len(outs) else "?"
    led = run.get("_ledger")
    if led is None and run.get("dir"):
        led = run["_ledger"] = _ledger(run["dir"])
    rec = (led or {}).get(outs[i].get("response_id")) if i < len(outs) else None
    kv = f", {rec.get('declined')}/{rec.get('planned')} blocks declined (before any clip)" if rec else ""
    return (f"{run['name']}: prompt {i} ({asked(i)}) reads like prompt {j}'s ({asked(j)}): "
            f"{own:.2f} to its own baseline, {other:.2f} to the sibling's{kv}")


def mark(runs: list, key=lambda r: (r.get("fingerprint"), r.get("fixed", False))) -> None:
    """Set ``sibling`` (a count, or None when it cannot be judged) and ``sibling_hits`` on each run.

    Runs are judged against the baseline runs with the same ``key`` (same prompts and mode), never
    against themselves."""
    for run in runs:
        run["sibling"], run["sibling_hits"] = None, []
        if run.get("fixed") or not run.get("outputs"):
            continue
        refs = [b for b in runs if b is not run and b.get("arm") == "baseline" and key(b) == key(run)
                and len(b.get("outputs") or []) == len(run["outputs"])]
        if not refs:
            continue
        texts = lambda r: [o.get("generated_text") or "" for o in r["outputs"]]
        hits = drifted(texts(run), [texts(b) for b in refs], [o.get("prompt") or "" for o in run["outputs"]])
        run["sibling"], run["sibling_hits"] = len(hits), hits
