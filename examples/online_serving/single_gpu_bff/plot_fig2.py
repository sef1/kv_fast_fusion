#!/usr/bin/env python3
"""Fig 2 (headline): throughput, concurrency actually reached, and quality vs offered concurrency.

Reads <root>/summary_runs.csv (from summarize_paper_suite.py), conc_sweep group only, and writes
<root>/fig2_throughput_vs_load.{pdf,png} plus fig2_data.csv (the plotted numbers, for the table view).

  (a) output throughput (tok/s)                       one line per arm
  (b) mean running requests (vLLM logger)              the mechanism: freed KV -> larger batch
  (c) paired dF1 vs vanilla at the same concurrency    +-1 s.e., zero line = vanilla

Usage: python3 plot_fig2.py results/paper_20261125
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (dataviz skill, light mode), validated: slots 1-4 adjacent CVD dE >= 9.1.
# Vanilla is the neutral reference line, not a categorical slot.
ARMS = [  # key, label, color, marker, linestyle
    ("vanilla",   "vLLM (vanilla)",            "#52514e", "o", "-"),
    ("fuse0",     "BFF layout only (FUSE=0)",  "#2a78d6", "s", ":"),
    ("fuse1",     "BFF nr_tree, per-step",     "#eb6834", "^", "--"),
    ("online085", "BFF online, reg. thr 0.85", "#1baf7a", "D", "-"),
    ("online080", "BFF online, reg. thr 0.80", "#eda100", "v", "-"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(root):
    rows = list(csv.DictReader(open(os.path.join(root, "summary_runs.csv"))))
    data = defaultdict(dict)
    for r in rows:
        if r["group"] != "conc_sweep":
            continue
        f = lambda k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
        data[r["arm"]][int(r["concurrency"])] = dict(
            tok=f("output_tok_s"), run=f("running_mean_stock"), df1=f("dF1_vs_vanilla"),
            se=f("dF1_se"), f1=f("mean_f1"), cr=f("cr_fused_layers"), itl99=f("itl_p99_ms"),
            ttft=f("ttft_mean_ms"))
    return data


def style(ax, ylabel, title):
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("offered concurrency (max in-flight requests)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def main(root):
    data = load(root)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    panels = [("tok", "output tokens / s", "(a) Throughput"),
              ("run", "mean running requests", "(b) Batch size reached"),
              ("df1", "ΔF1 vs vanilla (paired, ±1 s.e.)", "(c) Quality")]
    table = []
    for ax, (key, ylab, title) in zip(axes, panels):
        ends = []
        for ai, (arm, label, color, marker, ls) in enumerate(ARMS):
            pts = sorted(data.get(arm, {}).items())
            if not pts or (key == "df1" and arm == "vanilla"):
                continue
            xs = [c for c, _ in pts]
            ys = [v[key] for _, v in pts]
            if key == "df1":  # dodge so the error bars of different arms don't overprint
                xs = [c * (1 + 0.025 * (ai - 2.5)) for c in xs]
            ax.plot(xs, ys, color=color, lw=1.6, ls=ls, marker=marker, ms=5.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
            if key == "df1":
                ax.errorbar(xs, ys, yerr=[v["se"] for _, v in pts], fmt="none", ecolor=color,
                            elinewidth=1.0, capsize=2.5, zorder=2)
            if key == "tok":
                ends.append([ys[-1], ys[-1], xs[-1], label.replace("BFF ", "")])
        if ends:  # direct labels at line ends, nudged apart (contrast relief for aqua/yellow)
            lo, hi = ax.get_ylim()
            gap = 0.055 * (hi - lo)
            ends.sort(key=lambda e: e[0])
            for i in range(1, len(ends)):
                ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
            for y, ylab_pos, x, text in ends:
                ax.annotate(text, (x, y), xytext=(x * 1.04, ylab_pos), textcoords="data",
                            va="center", fontsize=7.5, color=INK2)
        if key == "df1":
            ax.axhline(0, color=INK2, lw=0.9, zorder=1)
        style(ax, ylab, title)
        ax.set_xticks(sorted({c for a in data.values() for c in a}))
    axes[0].margins(x=0.30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("Llama-3.1-8B-Instruct, 1× A100-40GB, Magicoder prompts 1.5k–6k tokens, ≤1024 output tokens",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(root, f"fig2_throughput_vs_load.{ext}"), dpi=200)
    with open(os.path.join(root, "fig2_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "concurrency", "tok_s", "gain_vs_vanilla_pct", "running_mean", "mean_f1", "dF1",
                    "dF1_se", "cr_fused_layers", "itl_p99_ms", "ttft_mean_ms"])
        van = data.get("vanilla", {})
        for arm, *_ in ARMS:
            for c, v in sorted(data.get(arm, {}).items()):
                g = 100 * (v["tok"] / van[c]["tok"] - 1) if c in van else float("nan")
                w.writerow([arm, c, f"{v['tok']:.1f}", f"{g:+.1f}", f"{v['run']:.1f}", f"{v['f1']:.4f}",
                            f"{v['df1']:+.4f}", f"{v['se']:.4f}", f"{v['cr']:.2f}", f"{v['itl99']:.0f}",
                            f"{v['ttft']:.0f}"])
    print("wrote", os.path.join(root, "fig2_throughput_vs_load.pdf"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/paper_20261125")
