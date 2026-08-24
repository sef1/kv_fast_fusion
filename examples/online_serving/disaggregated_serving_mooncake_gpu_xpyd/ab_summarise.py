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


# Config fields that make two runs DIFFERENT EXPERIMENTS rather than repeats of one. Pooling across
# any of these is not averaging noise, it is averaging two answers. `max_tokens` is here because it
# is absent from RUN_TAG and on 2026-08-20 silently pooled a 1024-token sweep with an 8192-token one
# into a single arm, reporting a mean of 0.809 rps that described neither.
POOL_KEYS = ("max_tokens", "num_prompts", "max_concurrency", "request_rate", "burstiness",
             "min_tokens", "model", "dataset_path")

# The same rule applied to the run's own settings, which the benchmark never recorded and
# `collect_bff_stats.run_config` now does. Topology is first because it is the one that distorts
# hardest: req/s scales with the decode count, so pooling a 1P×1D arm with a 1P×2D arm reports a
# mean that describes neither and looks exactly like a regression. Runs written before run_config
# existed have none of these, compare equal on all of them, and still pool as they always did.
CONFIG_KEYS = ("num_decode", "num_prefill", "tp", "baseline",
               "bff_threshold", "bff_threshold_g", "bff_max_rel_err", "bff_scale_mode",
               "bff_sig_dim", "bff_group_size", "bff_pd_cross_index",
               "bff_v2_dedup", "bff_v2_resident", "bff_ff_groups",
               "prefill_gpu_util", "decode_gpu_util", "max_model_len")


def _run_cfg(d, key):
    """One run_config value, or None for a result file written before run_config existed."""
    e = (d.get("run_config") or {}).get(key)
    return e.get("value") if isinstance(e, dict) else None


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    if d.get("request_throughput_rps") is None:
        # A cell that died mid-run leaves the collector's log-derived sections but no benchmark
        # result. Naming it is the point: silently dropping it turns a 2-of-3 arm into something
        # that reads like a complete one.
        raise ValueError("no request_throughput_rps — incomplete run")
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
        "cfg": (tuple((k, (d.get("config") or {}).get(k)) for k in POOL_KEYS)
                + tuple((k, _run_cfg(d, k)) for k in CONFIG_KEYS)),
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
    """Arm identity = the run tag with the run-set and repeat suffixes stripped.

    Both are stripped so runs from DIFFERENT invocations pool into one arm. Replication across
    sweeps is the point: the 2026-08-19 legacy-vs-current effect looked significant at n=3+3 in two
    consecutive sweeps and evaporated at n=9+9 once a third was added (p 0.011 -> 0.083)."""
    return re.sub(r"(_s[0-9-]+)?(_r\d+)?$", "", tag)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "f1_results")
    runs = {}
    skipped = []
    for f in sorted(glob.glob(os.path.join(d, "f1_*_r[0-9]*.json"))):
        tag = os.path.basename(f)[3:-5]
        try:
            r = load(f)
        except Exception as e:                      # a crashed run leaves a truncated file
            skipped.append((os.path.basename(f), str(e)))
            continue
        # The arm is the tag AND the config: two runs that differ in an untagged knob are two
        # experiments, and pooling them reports a mean describing neither.
        runs.setdefault((arm_of(tag), r["cfg"]), []).append((tag, r))

    if skipped:
        for name, why in skipped:
            print(f"  ! incomplete cell, not counted: {name}  ({why})")

    # Same tag, different config: name the fields that differ, because the tag alone gives the user
    # no way to tell the two apart in the listing below.
    by_tag, differing = {}, {}
    for arm, cfg in runs:
        by_tag.setdefault(arm, []).append(cfg)
    for arm, cfgs in by_tag.items():
        if len(cfgs) > 1:
            differing[arm] = sorted(k for k in POOL_KEYS + CONFIG_KEYS
                                    if len({dict(c).get(k) for c in cfgs}) > 1)
            print(f"\n  ! {arm}\n    split into {len(cfgs)} arms — same run tag but different "
                  f"{', '.join(differing[arm])}. These are separate experiments; they are NOT "
                  f"pooled and must not be compared with each other.")

    if not runs:
        print(f"No repeated runs in {d} (expects f1_*_r<N>.json — set RUN_REPEAT).")
        return 1

    series, of_arm = {}, {}
    for (arm, cfg), rows in sorted(runs.items()):
        # Disambiguate with only the fields that actually differ, so the common case stays readable
        # and the split case names its cause.
        keys = differing.get(arm)
        label = arm if not keys else (
            arm + "  [" + " ".join(f"{k}={dict(cfg).get(k)}" for k in keys) + "]")
        print(f"\n{label}   (n={len(rows)})")
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
            series[label] = (rps, tps)
            of_arm[label] = (arm, cfg)

    if len(series) < 2:
        print("\n(need two or more arms to compare)")
        return 0

    print(f"\n{'=' * 78}\nPairwise exact permutation tests (one-sided, alpha={ALPHA})")
    names = sorted(series, key=lambda a: -sum(series[a][0]) / len(series[a][0]))
    # Only arms measured under the SAME run config are a treatment comparison. Across configs the
    # gap is dominated by the workload — max_tokens 1024 vs 8192 moved rps by 100% here, twenty
    # times any effect under test — so such a p-value describes the benchmark, not the connector.
    pairs, cross = [], 0
    for hi, lo in combinations(names, 2):
        if of_arm[hi][1] != of_arm[lo][1]:
            cross += 1
            continue
        pairs.append((hi, lo))
    if cross:
        print(f"({cross} pair(s) skipped: different run configs — those gaps measure the workload, "
              f"not the treatment.)")
    if not pairs:
        print("No comparable pairs: every arm here differs by run config rather than by treatment.")
        return 0
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
