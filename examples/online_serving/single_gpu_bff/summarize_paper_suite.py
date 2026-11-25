#!/usr/bin/env python3
"""Summarize a run_paper_suite.py results folder into plot-ready CSVs.

  <root>/summary_runs.csv        one row per run (F1, paired dF1 vs vanilla, throughput, latency, CR, ...)
  <root>/summary_repeats.csv     mean / std over repeats per arm
  <root>/summary_sweep.csv       Fig 2 table: arm x concurrency
  <root>/<group>/timeseries/<tag>_stock.csv   vLLM logger: t, prompt/gen tok/s, running, waiting, kv%
  <root>/<group>/timeseries/<tag>_bff.csv     BFF scheduler: t, running, waiting, free/total blocks, usage%, preempt

Paired dF1 = mean over prompts of (F1_run - F1_ref), ref = the vanilla run(s) on the same prompts
(repeats: mean of the vanilla repeats; sweep: vanilla at the same concurrency).
Usage: python3 summarize_paper_suite.py results/paper_20261125
"""
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

TS = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d)")
STOCK = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                   r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
BFF = re.compile(r"BFF sched \| step=(\d+) \| running=(\d+) \| waiting=(\d+) \| free_blocks=(\d+) / (\d+) \| "
                 r"block_usage=([\d.]+)% \| preempt\(cum\)=(\d+)")


def parse_tag(tag):
    m = re.match(r"(?:rep(\d+)_)?(?:sweep_|ablate_)?([a-z0-9]+)_c(\d+)$", tag)
    return (int(m.group(1)) if m and m.group(1) else 0, m.group(2) if m else tag, int(m.group(3)) if m else 0)


def paired(x, ref):
    d = [a - b for a, b in zip(x, ref)]
    if len(d) < 2:
        return float("nan"), float("nan")
    m = sum(d) / len(d)
    return m, math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1) / len(d))


def timeseries(sub, tag):
    log = os.path.join(sub, f"server_{tag}.log")
    if not os.path.exists(log):
        return {}
    out = os.path.join(sub, "timeseries"); os.makedirs(out, exist_ok=True)
    rows_s, rows_b, t0 = [], [], None
    for line in open(log, errors="replace"):
        ts = TS.search(line)
        if not ts:
            continue
        t = datetime.strptime("2026-" + ts.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if (m := STOCK.search(line)):
            t0 = t0 or t
            rows_s.append([t - t0, *m.groups()])
        elif (m := BFF.search(line)):
            t0 = t0 or t
            rows_b.append([t - t0, *m.groups()])
    stats = {}
    busy = [r for r in rows_s if int(r[3]) > 0]
    if busy:
        stats = dict(run_stock=sum(int(r[3]) for r in busy) / len(busy),
                     kv_stock=sum(float(r[5]) for r in busy) / len(busy))
    if rows_s:
        with open(os.path.join(out, f"{tag}_stock.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_s", "prompt_tok_s", "gen_tok_s", "running", "waiting", "kv_usage_pct"])
            w.writerows(rows_s)
    if rows_b:
        with open(os.path.join(out, f"{tag}_bff.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_s", "step", "running", "waiting", "free_blocks", "total_blocks",
                                           "block_usage_pct", "preempt_cum"])
            w.writerows(rows_b)
    return stats


def main(root):
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "*", "f1_*.json"))):
        sub = os.path.dirname(path); group = os.path.basename(sub)
        tag = os.path.basename(path)[3:-5]
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if "evaluation" not in d:
            continue
        rep, arm, con = parse_tag(tag)
        b = d.get("bff_single", {})
        runs.append(dict(group=group, tag=tag, arm=arm, con=con, rep=rep, d=d, b=b,
                         ts=timeseries(sub, tag)))

    # references for paired dF1
    ref = {}
    van_rep = [r for r in runs if r["group"] == "repeats" and r["arm"] == "vanilla"]
    if van_rep:
        ps = [r["d"]["evaluation"]["per_sample_f1"] for r in van_rep]
        ref[("repeats", 128)] = [sum(v) / len(v) for v in zip(*ps)]
    for r in runs:
        if r["arm"] == "vanilla" and r["group"] in ("conc_sweep",):
            ref[(r["group"], r["con"])] = r["d"]["evaluation"]["per_sample_f1"]
    if ("repeats", 128) in ref:
        ref.setdefault(("ablation", 128), ref[("repeats", 128)])

    cols = ["group", "tag", "arm", "concurrency", "rep", "completed", "mean_f1", "dF1_vs_vanilla", "dF1_se",
            "finish_length_pct", "output_tok_s", "req_s", "elapsed_s", "ttft_mean_ms", "ttft_p99_ms",
            "itl_mean_ms", "itl_median_ms", "itl_p99_ms", "tpot_mean_ms", "e2e_mean_ms",
            "running_mean", "running_max", "block_usage_mean_pct", "preempt_cum",
            "running_mean_stock", "kv_usage_mean_stock", "cr_fused_layers", "match_frac", "matched_tree", "matched_registry", "confirm_ok", "rejected"]
    rows = []
    for r in runs:
        d, b = r["d"], r["b"]
        x = d["evaluation"]["per_sample_f1"]
        rf = ref.get((r["group"], r["con"]))
        dm, dse = paired(x, rf) if rf is not None and r["arm"] != "vanilla" else (
            paired(x, rf) if rf is not None and r["group"] == "repeats" else (float("nan"), float("nan")))
        g = lambda k, s: (d.get(k) or {}).get(s)
        cr = b.get("online_cr") or b.get("fusion_layer_cr") or 1.0
        rows.append([r["group"], r["tag"], r["arm"], r["con"], r["rep"], d.get("completed"),
                     d["evaluation"]["mean_f1"], dm, dse, d.get("finish_length_pct"),
                     d.get("output_throughput_toks_s"), d.get("request_throughput_rps"), d.get("elapsed_s"),
                     g("ttft_ms", "mean"), g("ttft_ms", "p99"), g("itl_ms", "mean"), g("itl_ms", "median"),
                     g("itl_ms", "p99"), g("tpot_ms", "mean"), g("e2e_latency_ms", "mean"),
                     b.get("running_mean"), b.get("running_max"), b.get("block_usage_mean"),
                     b.get("preempt_cum"), r["ts"].get("run_stock"), r["ts"].get("kv_stock"), cr, b.get("online_match_frac"), b.get("online_matched_tree"),
                     b.get("online_matched_registry"), b.get("online_confirm_ok"), b.get("online_rejected")])
    with open(os.path.join(root, "summary_runs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)

    def agg(keyf, name):
        acc = defaultdict(list)
        for row in rows:
            k = keyf(row)
            if k is not None:
                acc[k].append(row)
        out = []
        for k, rs in sorted(acc.items()):
            def ms(i):
                v = [r[i] for r in rs if r[i] is not None and not (isinstance(r[i], float) and math.isnan(r[i]))]
                if not v:
                    return (None, None)
                m = sum(v) / len(v)
                return m, (math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1)) if len(v) > 1 else 0.0)
            f1, tok, df1, cr, run, itl, ttft = (ms(cols.index(c)) for c in
                ("mean_f1", "output_tok_s", "dF1_vs_vanilla", "cr_fused_layers", "running_mean_stock", "itl_p99_ms",
                 "ttft_mean_ms"))
            out.append([*k, len(rs), *f1, *df1, *tok, *cr, *run, *itl, *ttft])
        with open(os.path.join(root, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "concurrency", "n", "f1_mean", "f1_std", "dF1_mean", "dF1_std", "tok_s_mean",
                        "tok_s_std", "cr_mean", "cr_std", "running_mean", "running_std", "itl_p99_mean",
                        "itl_p99_std", "ttft_mean", "ttft_std"])
            w.writerows(out)
        return out

    agg(lambda r: (r[2], r[3]) if r[0] == "repeats" else None, "summary_repeats.csv")
    sweep = agg(lambda r: (r[2], r[3]) if r[0] == "conc_sweep" else None, "summary_sweep.csv")
    print(f"{len(rows)} runs -> {root}/summary_runs.csv, summary_repeats.csv, summary_sweep.csv")
    van = {k[1]: v for k, v in [((r[0], r[1]), r) for r in sweep] if k[0] == "vanilla"}
    for r in sweep:
        base = van.get(r[1])
        gain = (100 * (r[7] / base[7] - 1)) if base and base[7] else float("nan")
        print(f"  {r[0]:10} c{r[1]:<4} tok/s {r[7]:7.1f} ({gain:+5.1f}%)  F1 {r[3]:.4f}  dF1 {r[5] if r[5] is not None else float('nan'):+.4f}  CR {r[9]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/paper_20261125")
