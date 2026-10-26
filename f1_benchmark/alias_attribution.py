"""Attribute a BFF run's F1 change to the prompts dedup actually touched.

The run-level stats say how MANY requests had KV substituted (``request_decline_frac``), not WHICH.
A GLM run at BFF_MAX_REL_ERR=0.05 with the per-request cap off lost 6 F1 points against the
baseline, and a text-similarity proxy put the loss on unrelated prompts as much as on the
near-duplicates that must have been aliased — which would mean the damage crosses requests. This
replaces the proxy with the real join:

* the decode's ``bff_stats_*.json`` carries ``request_ledger`` = {ext_id: declined/planned/failed};
* f1_main saves each prompt's ``response_id`` (``chatcmpl-<X-Request-Id>``), and the P/D proxy sends
  the decode that same uuid, so the uuid inside both ids is the key.

Prompts are bucketed by the fraction of their KV that was substituted, and each bucket reports the
mean paired F1 difference against a baseline run of the same prompts, with a bootstrap CI. The
bucket to read first is ``0%``: a loss there is damage to requests that were never aliased.

Usage:
    python3 -m f1_benchmark.alias_attribution BFF_RESULTS_DIR BASELINE_RESULTS_DIR
            [--dataset-path P] [--hf-split S] [--input-key K] [--output-key K]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from f1_benchmark.f1_main import f1_score

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# (label, lower bound inclusive, upper bound exclusive) on the substituted fraction.
BUCKETS = (("0%", 0.0, 1e-9), ("(0,10%)", 1e-9, 0.10), ("[10,50%)", 0.10, 0.50),
           ("[50,100%]", 0.50, 1.01))


def uuid_of(rid):
    """The proxy's uuid inside a vLLM request id, or None."""
    m = _UUID.search(rid or "")
    return m.group(0).lower() if m else None


def load_outputs(results_dir):
    with open(os.path.join(results_dir, "raw_outputs.json")) as f:
        saved = json.load(f)
    return saved["outputs"] if isinstance(saved, dict) else saved


def load_ledger(results_dir):
    """{uuid: entry} over every decode rank's dump. Ranks make the same decisions (MLA latents
    are identical across TP ranks), so the first entry per request is kept, not summed."""
    ledger = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "bff_stats_*.json"))):
        with open(path) as f:
            stats = json.load(f)
        for ext_id, entry in (stats.get("request_ledger") or {}).items():
            key = uuid_of(ext_id)
            if key and key not in ledger:
                ledger[key] = entry
    return ledger


def bucket_of(entry):
    """Bucket label for a ledger entry; a capped request was read in full, so it counts as 0%.
    ``None`` means the request was planned but never aliased-and-recomputed."""
    if entry is None:
        return "0%"
    if entry.get("failed"):
        return "recomputed"
    planned = entry.get("planned") or 0
    frac = 0.0 if entry.get("capped") or planned <= 0 else entry.get("declined", 0) / planned
    for label, lo, hi in BUCKETS:
        if lo <= frac < hi:
            return label
    return BUCKETS[-1][0]


def join(bff_outputs, base_outputs, ledger, references):
    """Rows of (bucket, f1_bff, f1_base) for prompts present in both runs with a reference."""
    base = {o.get("prompt"): o.get("generated_text") or "" for o in base_outputs}
    rows, unmatched = [], 0
    for o in bff_outputs:
        prompt = o.get("prompt")
        ref = references.get(prompt)
        if ref is None or prompt not in base:
            continue
        key = uuid_of(o.get("response_id"))
        if key is None:
            unmatched += 1
            continue
        rows.append((bucket_of(ledger.get(key)),
                     f1_score(o.get("generated_text") or "", ref), f1_score(base[prompt], ref)))
    return rows, unmatched


def bootstrap_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bff_results", help="results dir of the BFF run (raw_outputs + bff_stats_*)")
    p.add_argument("baseline_results", help="results dir of the baseline run (raw_outputs)")
    p.add_argument("--dataset-path", default="ise-uiuc/Magicoder-Evol-Instruct-110K")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--input-key", default="instruction")
    p.add_argument("--output-key", default="response")
    args = p.parse_args()

    bff, base = load_outputs(args.bff_results), load_outputs(args.baseline_results)
    if not any(o.get("response_id") for o in bff):
        print("The BFF run's raw_outputs.json has no response_id — it predates the ledger. Re-run.")
        return 1
    ledger = load_ledger(args.bff_results)
    if not ledger:
        print("No request_ledger in the BFF run's bff_stats_*.json — it predates the ledger. Re-run.")
        return 1

    from datasets import load_dataset
    if args.dataset_path.endswith(".jsonl") or os.path.isfile(args.dataset_path):
        ds = load_dataset("json", data_files=args.dataset_path, split=args.hf_split)
    else:
        ds = load_dataset(args.dataset_path, split=args.hf_split)
    wanted = {o.get("prompt") for o in bff}
    references = {}
    for prompt, ref in zip(ds[args.input_key], ds[args.output_key]):
        if prompt in wanted:
            references[prompt] = ref

    rows, unmatched = join(bff, base, ledger, references)
    if not rows:
        print("No prompts joined between the two runs.")
        return 1
    n_led = sum(1 for o in bff if uuid_of(o.get("response_id")) in ledger)
    print(f"{len(rows)} prompts scored; {n_led} matched a ledger entry, {unmatched} had no "
          f"response_id. Ledger: {len(ledger)} request(s).")
    d_all = np.array([a - b for _, a, b in rows])
    lo, hi = bootstrap_ci(d_all)
    print(f"  all prompts : n={len(rows):4d}  BFF-baseline {d_all.mean()*100:+.2f} pts "
          f"[{lo*100:+.2f}, {hi*100:+.2f}]")
    for label in [b[0] for b in BUCKETS] + ["recomputed"]:
        d = np.array([a - b for bk, a, b in rows if bk == label])
        if len(d) == 0:
            continue
        lo, hi = bootstrap_ci(d) if len(d) > 1 else (d[0], d[0])
        print(f"  {label:<12}: n={len(d):4d}  BFF-baseline {d.mean()*100:+.2f} pts "
              f"[{lo*100:+.2f}, {hi*100:+.2f}]")
    print("  -> a loss in '0%' is damage to requests that were never aliased (cross-request).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
