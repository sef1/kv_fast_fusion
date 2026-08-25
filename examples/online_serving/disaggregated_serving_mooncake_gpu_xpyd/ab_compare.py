#!/usr/bin/env python3
"""Compare two runs, leading with what differs in their CONFIGURATION.

Every result file is a measurement of one configuration, and the gap between two of them is a
treatment effect only if nothing else changed. That condition is easy to violate and, until
`run_config` existed, impossible to check from the files — the benchmark recorded eight fields and
the rest survived only in the filename.

Three times on 2026-08-24 a configuration difference was read as a code regression:

  * `MAX_TOKENS` 8192 vs 1024 — "the code runs very very slow". Token throughput was HIGHER; the
    requests were simply four times longer.
  * `NUM_DECODE` 1 vs 2 — "without REL_ERROR the throughput slows down". 1374 vs 1312 tok/s per
    decode GPU: the same machine speed, twice the machines.
  * two `thr0.8` runs sharing a filename, the second silently overwriting the first.

So this tool prints the configuration diff FIRST and refuses to characterise the outcome gap as an
effect when an experiment-defining field moved. It reports; it does not conclude on your behalf.

Usage:  ./ab_compare.py A.json B.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_summarise import POOL_KEYS   # noqa: E402  (same directory, deliberately shared)

# Fields whose value makes two runs DIFFERENT EXPERIMENTS rather than repeats of one. POOL_KEYS
# covers the benchmark half; these are the run_config half the benchmark never recorded. Ordered
# by how badly each one distorts a naive comparison, so the loudest confounder is named first.
DECISIVE = (
    "num_decode", "num_prefill", "tp",
    "baseline", "bff_threshold", "bff_threshold_g", "bff_max_rel_err", "bff_scale_mode",
    "bff_sig_dim", "bff_group_size", "bff_pd_cross_index", "bff_v2_dedup", "bff_v2_resident",
    "bff_ff_groups", "prefill_gpu_util", "decode_gpu_util", "max_model_len", "min_tokens",
)


def cfg(d, key):
    e = (d.get("run_config") or {}).get(key)
    if isinstance(e, dict):
        v = e.get("value")
        return None if v is None else f"{v}{'' if e.get('explicit') else ' (default)'}"
    return None


def n_dec(d):
    e = (d.get("run_config") or {}).get("num_decode")
    try:
        return max(1, int(e["value"])) if isinstance(e, dict) else _from_logs(d)
    except (TypeError, ValueError, KeyError):
        return _from_logs(d)


def _from_logs(d):
    return max(1, len([k for k in (d.get("bff_sched") or {}) if "decode" in k]))


def outcomes(d):
    b, ev = d.get("bff_v2") or {}, (d.get("evaluation") or {}).get("codebleu") or {}
    el, out = d.get("elapsed_s"), d.get("total_output_tokens")
    toks = (out / el) if (out and el) else None
    sch = d.get("bff_sched") or {}
    D = ([v for k, v in sch.items() if "decode" in k] or [{}])[0]
    P = ([v for k, v in sch.items() if "prefill" in k] or [{}])[0]
    xf = d.get("kv_transfer_failures") or {}
    xD = ([v for k, v in xf.items() if "decode" in k] or [{}])[0]
    xP = ([v for k, v in xf.items() if "prefill" in k] or [{}])[0]
    wedge = (100.0 * xP["wedged_samples"] / xP["sched_samples"]) if xP.get("sched_samples") else None
    return {
        "req/s": d.get("request_throughput_rps"),
        "tok/s": toks,
        "tok/s per decode": (toks / n_dec(d)) if toks else None,
        "TTFT s": (d.get("ttft_ms") or {}).get("mean", 0) / 1000 or None,
        "wire saved %": b.get("wire_saving_pct"),
        "ngram": ev.get("ngram_match"),
        "D running": (D.get("running") or {}).get("mean"),
        "D preempt": D.get("preempt_cum"),
        "D recompute": xD.get("recomputed_requests"),
        "P usage %": (P.get("block_usage_pct") or {}).get("mean"),
        "P wedged %": wedge,
    }


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[-1])
        return 2
    A, B = (json.load(open(p)) for p in argv[1:3])
    na, nb = (os.path.basename(p)[3:-5] if os.path.basename(p).startswith("f1_")
              else os.path.basename(p) for p in argv[1:3])

    print(f"\n  A  {na}\n  B  {nb}\n")

    # ---- 1. configuration ----------------------------------------------------------------
    diffs = []
    for k in DECISIVE:
        va, vb = cfg(A, k), cfg(B, k)
        if va != vb and not (va is None and vb is None):
            diffs.append((k, va, vb))
    for k in POOL_KEYS:
        va, vb = (A.get("config") or {}).get(k), (B.get("config") or {}).get(k)
        if va != vb:
            diffs.append((k, va, vb))

    missing = [n for n, d in ((na, A), (nb, B)) if not d.get("run_config")]
    print("  " + "=" * 74)
    if diffs:
        print(f"  CONFIGURATION DIFFERS in {len(diffs)} field(s)")
        w = max(len(k) for k, _, _ in diffs)
        for k, va, vb in diffs:
            print(f"    {k:<{w}}   {va}   →   {vb}")
    else:
        print("  Configuration is identical in every recorded field.")
    if missing:
        # Absence is not sameness. A file predating run_config cannot be shown to match, and saying
        # "identical" about it would be the exact false reassurance this tool exists to prevent.
        print(f"\n    ! no run_config in: {', '.join(missing)} — predates config recording, so its"
              f"\n      settings CANNOT be compared. Re-run the collector on it, or treat any"
              f"\n      agreement above as unverified.")
    print("  " + "=" * 74)

    # ---- 2. outcomes ---------------------------------------------------------------------
    oa, ob = outcomes(A), outcomes(B)
    print(f"\n  {'':<18}{'A':>12}{'B':>12}{'B/A':>10}")
    for k in oa:
        a, b = oa[k], ob[k]
        if a is None and b is None:
            continue
        # Both cells in a row share one precision, chosen from the larger. Formatting them
        # independently printed "114" beside "15.580", which reads as two different units.
        nums = [v for v in (a, b) if isinstance(v, (int, float))]
        dp = 0 if (nums and max(abs(v) for v in nums) >= 100) else 3
        fmt = lambda v: f"{v:,.{dp}f}" if isinstance(v, (int, float)) else "—"   # noqa: E731
        ratio = f"{b / a:.2f}x" if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and a else "—"
        print(f"  {k:<18}{fmt(a):>12}{fmt(b):>12}{ratio:>10}")

    # ---- 3. verdict ----------------------------------------------------------------------
    print()
    if diffs:
        names = ", ".join(k for k, _, _ in diffs[:3]) + (" …" if len(diffs) > 3 else "")
        print(f"  DIFFERENT EXPERIMENTS ({names}).")
        print("  The gaps above measure the configuration change, not a treatment effect —")
        print("  and not a regression.")
        if any(k == "num_decode" for k, _, _ in diffs):
            print("  Decode count differs: read 'tok/s per decode', not 'req/s'. Throughput scales")
            print("  with the number of decode GPUs.")
        if any(k in ("max_tokens", "min_tokens", "num_prompts") for k, _, _ in diffs):
            print("  Workload length differs: req/s falls when requests get longer even though the")
            print("  engine is unchanged. Compare 'tok/s per decode'.")
    else:
        print("  Same configuration — the gaps above are a treatment effect or run-to-run noise.")
        print("  Two runs cannot separate those. Use ab_interleaved.sh + ab_summarise.py for a")
        print("  test with repeats; at n=1+1 no p-value is available.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
