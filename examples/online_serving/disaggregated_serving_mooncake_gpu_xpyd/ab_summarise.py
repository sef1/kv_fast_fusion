#!/usr/bin/env python3
"""Summarise an interleaved A/B and test each pair of arms for a real difference.

**The statistic is an exact permutation test on the difference of means.** With 3 runs per arm
there are only C(6,3)=20 ways to split the six numbers into two arms, so the null distribution can
be enumerated rather than approximated — no normality assumption, correct for unequal n.

This replaces an earlier rule ("the gap must exceed the larger arm's own spread") that was wrong
and said so out loud: it declared NOISE on data where all three legacy runs beat all three current
runs (exact p = 0.025). That rule can never let a high-variance arm win, and variance driven by an
arm's own *best* run is evidence for it, not against.

Two metrics are reported because they must agree: `request_throughput_rps` and
`total_output_tokens / elapsed_s`. They matched to 0.1% on the 2026-08-19 runs; a divergence means
a request-accounting problem, not a real effect.

The mechanism counters print beside them on purpose. If arms differ in throughput but NOT in wire
saving / aliases applied / recomputes, the difference is not coming from the dedup logic and
diffing connectors is the wrong next move.

Usage:  ./ab_summarise.py [result_dir]
"""

import glob
import json
import math
import os
import re
import sys
from itertools import combinations

ALPHA = 0.05


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    b = d.get("bff_v2") or {}
    dec = [v for k, v in (d.get("kv_transfer_failures") or {}).items() if "decode" in k]
    v = dec[0] if dec else {}
    ev = (d.get("evaluation") or {}).get("codebleu") or {}
    sat = [x for k, x in (d.get("kv_saturation") or {}).items() if "decode" in k]
    run = (sat[0].get("running") or {}).get("mean") if sat else None
    elapsed, out = d.get("elapsed_s"), d.get("total_output_tokens")
    return {
        "rps": d.get("request_throughput_rps"),
        "tok_s": (out / elapsed) if (out and elapsed) else None,
        "wire": b.get("wire_saving_pct"),
        "applied": b.get("aliases_applied"),
        "alias_fail": b.get("aliases_recomputed"),
        "recomp": v.get("recomputed_requests"),
        "ngram": ev.get("ngram_match"),
        "drun": run,
    }


def exact_p(a, b):
    """One-sided exact permutation p for mean(a) > mean(b).

    Enumerates every way to relabel the pooled values into groups of |a| and |b|, and returns the
    fraction whose mean difference is at least the observed one. With n=m=3 the smallest attainable
    p is 1/20 = 0.05 two-sided / 0.025 one-sided — worth knowing before reading a null result as
    'no effect': at three runs an arm can be genuinely better and still not clear a 0.01 bar."""
    obs = sum(a) / len(a) - sum(b) / len(b)
    pool = list(a) + list(b)
    n = len(a)
    total = hits = 0
    for idx in combinations(range(len(pool)), n):
        g1 = [pool[i] for i in idx]
        g2 = [pool[i] for i in range(len(pool)) if i not in idx]
        total += 1
        if sum(g1) / len(g1) - sum(g2) / len(g2) >= obs - 1e-12:
            hits += 1
    return hits / total, obs


def arm_of(tag):
    """Arm identity = the run tag with the repeat suffix stripped."""
    return re.sub(r"_r\d+$", "", tag)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "f1_results")
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "f1_*_r[0-9]*.json"))):
        tag = os.path.basename(f)[3:-5]
        try:
            runs.setdefault(arm_of(tag), []).append((tag, load(f)))
        except Exception as e:                      # a crashed run leaves a truncated file
            print(f"  ! skipping {os.path.basename(f)}: {e}")

    if not runs:
        print(f"No repeated runs in {d} (expects f1_*_r<N>.json — set RUN_REPEAT).")
        return 1

    series = {}
    for arm, rows in sorted(runs.items()):
        print(f"\n{arm}   (n={len(rows)})")
        for tag, r in sorted(rows):
            print(f"   {tag[-3:]:>3}  rps={r['rps']:.3f}  tok/s={r['tok_s'] or 0:6.0f}  "
                  f"wire={r['wire'] or 0:5.2f}%  applied={r['applied']}  "
                  f"fail={r['alias_fail']}  recomp={r['recomp']}  "
                  f"Drun={r['drun'] or 0:5.1f}  ngram={r['ngram'] or 0:.4f}")
        rps = [r["rps"] for _, r in rows if r["rps"] is not None]
        tps = [r["tok_s"] for _, r in rows if r["tok_s"] is not None]
        if rps:
            print(f"   mean rps={sum(rps) / len(rps):.3f}   mean tok/s={sum(tps) / len(tps):.0f}"
                  if tps else f"   mean rps={sum(rps) / len(rps):.3f}")
            series[arm] = (rps, tps)

    if len(series) < 2:
        print("\n(need two or more arms to compare)")
        return 0

    print(f"\n{'=' * 78}\nPairwise exact permutation tests (one-sided, alpha={ALPHA})")
    names = sorted(series, key=lambda a: -sum(series[a][0]) / len(series[a][0]))
    pairs = list(combinations(names, 2))
    if len(pairs) > 1:
        print(f"NOTE: {len(pairs)} comparisons — a single p<{ALPHA} among them is weaker than it "
              f"looks (Bonferroni bar is {ALPHA / len(pairs):.3f}).")

    for hi, lo in pairs:
        a, b = series[hi][0], series[lo][0]
        p, obs = exact_p(a, b)
        sep = min(a) > max(b)
        floor = 1 / math.comb(len(a) + len(b), len(a))
        print(f"\n  {hi}\n    vs {lo}")
        print(f"    mean gap = {obs:+.3f} rps ({obs / (sum(b) / len(b)) * 100:+.1f}%)   "
              f"exact p = {p:.3f}   separation = {sep}")
        if series[hi][1] and series[lo][1]:
            pt, ot = exact_p(series[hi][1], series[lo][1])
            print(f"    tok/s gap = {ot:+.0f} ({ot / (sum(series[lo][1]) / len(series[lo][1])) * 100:+.1f}%)"
                  f"   exact p = {pt:.3f}"
                  f"{'' if (p < ALPHA) == (pt < ALPHA) else '   <-- METRICS DISAGREE, investigate'}")
        at_floor = abs(p - floor) < 1e-9
        if p <= ALPHA and at_floor:
            # Every run of one arm beat every run of the other: the strongest ordering these sample
            # sizes can express, and also the smallest p they can produce. Real, but one more
            # repeat per arm is cheap and would take it well below alpha.
            print(f"    -> REAL, but p is the FLOOR for n={len(a)}+{len(b)} (cannot go lower "
                  f"without more repeats). Add one per arm to strengthen it.")
        elif p <= ALPHA:
            print("    -> REAL at this alpha.")
        elif at_floor:
            # Separation is already perfect and p STILL cannot clear alpha — the sample sizes make
            # significance unreachable, which is a different statement from "no effect" and must
            # not be reported as one.
            print(f"    -> INCONCLUSIVE: separation is already perfect, but the floor for "
                  f"n={len(a)}+{len(b)} is {floor:.3f} > alpha={ALPHA}. No amount of effect size "
                  f"can pass at these n; balance the arms or add repeats.")
        else:
            print("    -> not distinguishable from noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
