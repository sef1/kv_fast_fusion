"""Compare saved benchmark runs: one table, and the best run of an arm.

Reads a sweep folder written by run_best_compare.sh: one sub-folder per run holding ``logs_latest``,
``results_latest`` and ``config.txt``. A run's arm and decode limit come from config.txt.

``prefill_k`` is the prompt tokens the prefill node really processed, integrated from its log. It is
the re-prefill measure that means the same for both connectors: mooncake logs "Delaying free" per
prefill and BFF logs "BFF recompute" per eviction, and those two counts are not comparable.

Usage:
    python3 -m f1_benchmark.compare_runs SWEEP_DIR              # the table and the arm means
    python3 -m f1_benchmark.compare_runs SWEEP_DIR --best baseline   # prints the best run's limit
"""

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys

from f1_benchmark import sibling_drift
from f1_benchmark.degenerate_outputs import has_restart, is_garbage

LOG_INTERVAL_S = 10          # vLLM's loggers.py reports throughput on this period
_ENGINE = re.compile(r"prompt throughput: ([\d.]+).*generation throughput: ([\d.]+).*"
                     r"Running: (\d+) reqs, Waiting: (\d+) reqs.*KV cache usage: ([\d.]+)%")


def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_engine_log(text: str) -> dict:
    """Prompt tokens processed and the queue averages, over samples with any request present."""
    rows = [tuple(map(float, m.groups())) for m in _ENGINE.finditer(text)]
    busy = [r for r in rows if r[2] or r[3]]
    n = len(busy) or 1
    return {"prompt_toks": sum(r[0] for r in rows) * LOG_INTERVAL_S,
            "running": sum(r[2] for r in busy) / n, "waiting": sum(r[3] for r in busy) / n,
            "kv": sum(r[4] for r in busy) / n, "max_running": max((r[2] for r in busy), default=0)}


def read_config(run_dir: str) -> dict:
    out = {}
    try:
        with open(os.path.join(run_dir, "config.txt")) as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k:
                    out[k] = v
    except OSError:
        pass
    return out


def _read(path: str) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def read_run(run_dir: str) -> dict:
    """Everything the table prints for one saved run."""
    logs, results = os.path.join(run_dir, "logs_latest"), os.path.join(run_dir, "results_latest")
    serving = "".join(_read(p) for p in glob.glob(os.path.join(logs, "*serving.txt")))
    prefill, decode = _read(os.path.join(logs, "prefill-0.txt")), _read(os.path.join(logs, "decode-0.txt"))
    try:
        with open(os.path.join(results, "raw_outputs.json")) as f:
            outs = json.load(f)
        outs = outs["outputs"] if isinstance(outs, dict) else outs
    except (OSError, ValueError):
        outs = []
    cfg = read_config(run_dir)
    p, d = parse_engine_log(prefill), parse_engine_log(decode)
    freed = re.findall(r"BFF promotion free \|.* freed=(\d+)", decode)
    build = re.search(r"build (\d{4}-\d\d-\d\d[a-z])", decode)
    return {
        "name": os.path.basename(run_dir.rstrip("/")), "arm": cfg.get("arm", "?"),
        "limit": cfg.get("decode_limit", "?"), "stage": cfg.get("stage", "sweep"), "n": len(outs),
        "fingerprint": hashlib.sha1("\n".join(o.get("prompt", "") for o in outs).encode()).hexdigest()[:8],
        "garbage": sum(is_garbage(o.get("generated_text") or "") for o in outs),
        "restarted": sum(has_restart(o.get("generated_text") or "") for o in outs),
        "tok_s": _num(r"throughput: ([\d.]+) output tok/s", serving),
        "req_s": _num(r"\| ([\d.]+) req/s", serving),
        "wall": _num(r"completed in ([\d.]+) s", serving),
        "ttft_s": (_num(r"TTFT\[mean=([\d.]+)", serving) or 0) / 1000.0,
        "tpot": _num(r"TPOT\[mean=[\d.]+ median=([\d.]+)", serving),
        "f1": _num(r"Mean F1 Score: ([\d.]+)", serving),
        "codebleu": _num(r"CodeBLEU Score: ([\d.]+)", serving),
        "ast": _num(r"AST Syntax Validity: ([\d.]+)", serving),
        "prefill_k": p["prompt_toks"] / 1000.0,
        "delaying": len(re.findall(r"Delaying free of", prefill)),
        "recompute": len(re.findall(r"BFF recompute \|", decode)),
        "running": d["running"], "max_running": d["max_running"], "waiting": d["waiting"], "kv": d["kv"],
        "freed": int(freed[-1]) if freed else 0, "build": build.group(1) if build else "-",
        # For the sibling-content check, which needs the baseline runs too (see read_sweep).
        "outputs": outs, "dir": run_dir, "fixed": cfg.get("THROUGHPUT_ONLY") == "1",
        "sibling": None, "sibling_hits": [],
    }


def read_sweep(sweep_dir: str) -> list:
    dirs = sorted(d for d in glob.glob(os.path.join(sweep_dir, "*")) if os.path.isfile(os.path.join(d, "DONE")))
    runs = [read_run(d) for d in dirs]
    sibling_drift.mark(runs)
    return runs


def best_of(runs: list, arm: str):
    """The arm's fastest SWEEP run with no garbage output; a broken run cannot win on speed.

    Confirmation runs are excluded so that they cannot change the limit they are confirming, which
    would also move a resumed sweep onto a different limit."""
    cand = [r for r in runs if r["arm"] == arm and r["stage"] == "sweep"
            and r["tok_s"] is not None and not r["garbage"]]
    return max(cand, key=lambda r: r["tok_s"]) if cand else None


def _f(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def _sib(r) -> str:
    return "-" if r.get("sibling") is None else str(r["sibling"])


def sibling_lines(runs: list) -> list:
    """Each answer the sibling-content check flagged, to be read by a person, and what the column means."""
    hits = [sibling_drift.describe(r, h) for r in runs for h in r.get("sibling_hits") or []]
    if not any(r.get("sibling") is not None for r in runs):
        return []
    lines = ["sib: answers closer to a sibling prompt's baseline answer than to their own (a baseline run is "
             "judged against the other baseline repeats; '-' = no baseline of the same prompts, or fixed length)."]
    return lines + [f"  {h}" for h in hits]


def table(runs: list) -> list:
    head = (f"{'run':28s} {'limit':>5s} {'tok/s':>6s} {'req/s':>5s} {'wall':>6s} {'TTFT':>5s} {'TPOT':>5s} "
            f"{'F1':>6s} {'CBLEU':>6s} {'AST':>5s} {'bad':>3s} {'rst':>3s} {'sib':>3s} {'prefill':>7s} {'evict':>5s} "
            f"{'run':>4s} {'max':>3s} {'wait':>4s} {'KV%':>4s} {'freed':>5s}  prompts/build")
    lines = [head]
    for r in runs:
        evict = r["recompute"] or (r["delaying"] - r["n"] if r["delaying"] else 0)
        lines.append(
            f"{r['name'][:28]:28s} {r['limit']:>5s} {_f(r['tok_s']):>6s} {_f(r['req_s'], '.2f'):>5s} "
            f"{_f(r['wall'], '.0f'):>6s} {_f(r['ttft_s'], '.0f'):>5s} {_f(r['tpot']):>5s} {_f(r['f1'], '.3f'):>6s} "
            f"{_f(r['codebleu'], '.2f'):>6s} {_f(r['ast']):>5s} {r['garbage']:>3d} {r['restarted']:>3d} {_sib(r):>3s} {r['prefill_k']:>6.0f}k "
            f"{evict:>5d} {r['running']:>4.1f} {r['max_running']:>3.0f} {r['waiting']:>4.1f} {r['kv']:>4.0f} "
            f"{r['freed']:>5d}  {r['fingerprint']} {r['build']}")
    return lines


def compare_arms(runs: list, base_limit: str, bff_limit: str, stage=None) -> list:
    """Baseline against BFF, each at its own limit: means, and how far apart they are against noise.

    ``stage="confirm"`` is the verdict: runs that did not take part in choosing the limits."""
    a = [r for r in runs if r["arm"] == "baseline" and r["limit"] == base_limit
         and (stage is None or r["stage"] == stage)]
    b = [r for r in runs if r["arm"] == "bff" and r["limit"] == bff_limit
         and (stage is None or r["stage"] == stage)]
    if not a or not b:
        return []
    lines = [f"baseline at limit {base_limit} ({len(a)} run(s)) vs BFF at limit {bff_limit} "
             f"({len(b)} run(s)){', ' + stage + ' runs only' if stage else ''}:"]
    if {r["fingerprint"] for r in a + b} != {a[0]["fingerprint"]}:
        lines.append("  WARNING: these runs did NOT use the same prompts, so they are not comparable.")
    for key, label, higher in (("tok_s", "throughput tok/s", True), ("req_s", "requests/s", True),
                               ("wall", "wall s", False), ("prefill_k", "prefill tokens k", False),
                               ("f1", "F1", True), ("codebleu", "CodeBLEU", True), ("ast", "AST %", True)):
        va, vb = [r[key] for r in a if r[key] is not None], [r[key] for r in b if r[key] is not None]
        if not va or not vb:
            continue
        ma, mb = statistics.mean(va), statistics.mean(vb)
        pct = 100.0 * (mb / ma - 1) if ma else 0.0
        better = (mb > ma) == higher
        lines.append(f"  {label:18s} {ma:9.2f} -> {mb:9.2f}  {pct:+6.1f}%  {'level' if abs(pct) < 0.05 else 'BFF ahead' if better else 'baseline ahead'}")
    lines.append(f"  garbage outputs    {sum(r['garbage'] for r in a)} vs {sum(r['garbage'] for r in b)}"
                 f" | BFF blocks freed per run: {statistics.mean(r['freed'] for r in b):.0f}")
    if any(r.get("sibling") is not None for r in b):
        sib = lambda rs: "-" if all(r.get("sibling") is None for r in rs) else str(sum(r.get("sibling") or 0 for r in rs))
        lines.append(f"  sibling content    {sib(a)} vs {sib(b)} (answers that read like a sibling prompt's)")
    lines.append("  Noise: identical BFF runs differ by about +-2.7% in wall time and +-1 point of CodeBLEU, "
                 "so a smaller gap from a few runs is not yet a result.")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir")
    ap.add_argument("--best", help="print the decode limit of this arm's best run, and nothing else")
    args = ap.parse_args(argv)
    runs = read_sweep(args.sweep_dir)
    if args.best:
        best = best_of(runs, args.best)
        if best is None:
            return 1
        print(best["limit"])
        return 0
    print("\n".join(table(runs)))
    sib = sibling_lines(runs)
    if sib:
        print("\n" + "\n".join(sib))
    best_a, best_b = best_of(runs, "baseline"), best_of(runs, "bff")
    if best_a is None or best_b is None:
        return 0
    print(f"\nBest limit from the sweep: baseline {best_a['limit']}, BFF {best_b['limit']}")
    confirm = compare_arms(runs, best_a["limit"], best_b["limit"], stage="confirm")
    if confirm:
        print("\nVERDICT (fresh runs that did not choose the limits)")
        print("\n".join(confirm))
    else:
        print("\nNo confirmation runs yet. These are the runs that chose the limits, so they flatter "
              "both arms:")
        print("\n".join(compare_arms(runs, best_a["limit"], best_b["limit"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
