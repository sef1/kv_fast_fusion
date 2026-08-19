#!/usr/bin/env python3
"""Parse the BFF P/D F1 result JSONs in this directory and plot every collected metric.

Each `f1_<tag>.json` (written by `f1_benchmark.f1_main` + the disagg shell's post-run merge)
carries: accuracy (mean_f1 + per_sample_f1), throughput (req/s, output tok/s, elapsed),
latency (TTFT/TPOT/ITL/E2E mean·median·p99), and the BFF telemetry merged in by the shell
(producer dedup overhead, compression factor per group, decode scheduler capacity, blocks
freed, redirects). The BFF *config* (merge / scale / repr / threshold / group-size / concurrency
/ topology) is encoded in the run label, so it's parsed from there; concurrency also comes from
the JSON `config` block when present (authoritative).

Usage:
    python plot_f1_results.py                 # scan this script's directory, write ./plots
    python plot_f1_results.py --dir <dir> --out <dir>
    python plot_f1_results.py --csv-only       # just write the summary CSV, no plots

Requires matplotlib + numpy:  pip install matplotlib numpy
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: write PNGs, never open a window
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Label grammar (set by the disagg shell's RUN_TAG):
#   BFF:     <merge>_<scale>_<repr>_thr<THR>_gs<GS>_con_<CONC>_<n>Px<m>D
#            merge ∈ {cc, nr_tree}; scale ∈ {raw, ratio, norm}; repr ∈ {full, proj, mean}
#   vanilla: vanilla[_con_<CONC>]_<n>Px<m>D
#   eb<N> = cross-batch encoded-batch window (ROUND 58); OPTIONAL — absent on pre-eb runs → 0.
_BFF_RE = re.compile(
    r"^(?P<merge>cc|nr_tree)_(?P<scale>raw|ratio|norm)_(?P<repr>full|proj|mean)"
    r"_thr(?P<thr>[\d.]+)_gs(?P<gs>\d+)(?:_eb(?P<eb>\d+))?_con_(?P<con>\d+)_(?P<np>\d+)Px(?P<nd>\d+)D$")
_VAN_RE = re.compile(r"^vanilla(?:_con_(?P<con>\d+))?_(?P<np>\d+)Px(?P<nd>\d+)D$")


def parse_label(label, fallback_con=None):
    """label → dict of parsed config (merge/scale/repr/thr/gs/eb/con/topology/system)."""
    m = _BFF_RE.match(label)
    if m:
        d = m.groupdict()
        return {
            "system": "bff", "merge": d["merge"], "scale": d["scale"], "repr": d["repr"],
            "thr": float(d["thr"]), "gs": int(d["gs"]),
            "eb": int(d["eb"]) if d["eb"] else 0,   # absent → within-batch only (0)
            "con": int(d["con"]), "topo": f"{d['np']}Px{d['nd']}D",
        }
    m = _VAN_RE.match(label)
    if m:
        d = m.groupdict()
        con = int(d["con"]) if d["con"] else (fallback_con or 0)
        return {
            "system": "vanilla", "merge": "-", "scale": "-", "repr": "-",
            "thr": float("nan"), "gs": 0, "eb": 0, "con": con, "topo": f"{d['np']}Px{d['nd']}D",
        }
    return {"system": "other", "merge": "-", "scale": "-", "repr": "-",
            "thr": float("nan"), "gs": 0, "eb": 0, "con": fallback_con or 0, "topo": "?"}


def _lat(d, key, stat):
    v = d.get(key)
    return v.get(stat) if isinstance(v, dict) else None


def load_records(directory):
    """Load every f1_*.json in `directory` into a flat list of metric records."""
    recs = []
    for path in sorted(glob.glob(os.path.join(directory, "f1_*.json"))):
        try:
            with open(path) as f:
                j = json.load(f)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue
        cfg = j.get("config", {})
        label = j.get("label") or os.path.basename(path)[3:].rsplit(".json", 1)[0]
        p = parse_label(label, fallback_con=cfg.get("max_concurrency"))
        con = cfg.get("max_concurrency") or p["con"]   # JSON config is authoritative
        comp = j.get("bff_compression", {}) or {}
        dec = (j.get("bff_sched", {}) or {}).get("decode1.log", {}) or {}
        freed = (j.get("bff_blocks_freed", {}) or {}).get("decode1.log", {}) or {}
        redir = (j.get("bff_redirects_applied", {}) or {}).get("decode1.log", {}) or {}
        rec = {
            "file": os.path.basename(path), "label": label, **p, "con": con,
            # accuracy
            "mean_f1": j.get("mean_f1"),
            "finish_len_pct": j.get("finish_length_pct"),
            "per_sample_f1": j.get("per_sample_f1") or [],
            # throughput
            "req_s": j.get("request_throughput_rps"),
            "out_tok_s": j.get("output_throughput_toks_s"),
            "elapsed_s": j.get("elapsed_s"),
            "completed": j.get("completed"),
            "total_out_tok": j.get("total_output_tokens"),
            # latency (median + p99)
            "ttft_med": _lat(j, "ttft_ms", "median"), "ttft_p99": _lat(j, "ttft_ms", "p99"),
            "tpot_med": _lat(j, "tpot_ms", "median"), "tpot_p99": _lat(j, "tpot_ms", "p99"),
            "itl_med": _lat(j, "itl_ms", "median"), "itl_p99": _lat(j, "itl_ms", "p99"),
            "e2e_med": _lat(j, "e2e_latency_ms", "median"), "e2e_p99": _lat(j, "e2e_latency_ms", "p99"),
            # BFF telemetry
            "comp_factor": comp.get("avg_factor"),
            "comp_per_group": {int(k): v for k, v in (comp.get("per_group") or {}).items()},
            "overhead_ms": (j.get("bff_overhead", {}) or {}).get("producer_avg_group_dedup_ms"),
            "dec_running_mean": (dec.get("running") or {}).get("mean"),
            "dec_running_max": (dec.get("running") or {}).get("max"),
            "dec_usage_max": (dec.get("block_usage_pct") or {}).get("max"),
            "dec_usage_mean": (dec.get("block_usage_pct") or {}).get("mean"),
            "dec_free_min": (dec.get("free_blocks") or {}).get("min"),
            "dec_preempt": dec.get("preempt_cum"),
            "net_blocks_freed": freed.get("net_blocks_freed"),
            "redirects_applied": redir.get("redirects_applied"),
        }
        # compact tag for plot axes
        if p["system"] == "bff":
            rec["tag"] = (f"{p['merge']}/{p['scale']}/{p['repr']} thr{p['thr']} gs{p['gs']} "
                          f"eb{p['eb']} c{con} {p['topo']}")
            rec["sys_key"] = f"{p['merge']}/{p['scale']}/{p['repr']}"
        elif p["system"] == "vanilla":
            rec["tag"] = f"vanilla c{con} {p['topo']}"
            rec["sys_key"] = "vanilla"
        else:
            rec["tag"] = label
            rec["sys_key"] = "other"
        recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------
_CSV_COLS = ["file", "system", "merge", "scale", "repr", "thr", "gs", "eb", "con", "topo",
             "mean_f1", "finish_len_pct", "req_s", "out_tok_s", "elapsed_s", "total_out_tok",
             "ttft_med", "ttft_p99", "tpot_med", "tpot_p99", "itl_med", "e2e_med",
             "comp_factor", "overhead_ms", "dec_running_mean", "dec_running_max",
             "dec_usage_max", "dec_preempt", "net_blocks_freed", "redirects_applied"]


def write_csv(recs, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x["system"], -(x["mean_f1"] or 0))):
            w.writerow(r)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _color_map(recs):
    keys = sorted({r["sys_key"] for r in recs})
    cmap = plt.get_cmap("tab10" if len(keys) <= 10 else "tab20")
    return {k: cmap(i % cmap.N) for i, k in enumerate(keys)}


def _barh(ax, recs, value_fn, title, xlabel, colors):
    rs = [r for r in recs if value_fn(r) is not None]
    rs = sorted(rs, key=lambda r: value_fn(r))
    y = np.arange(len(rs))
    ax.barh(y, [value_fn(r) for r in rs], color=[colors[r["sys_key"]] for r in rs])
    ax.set_yticks(y)
    ax.set_yticklabels([r["tag"] for r in rs], fontsize=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def plot_tradeoff(recs, out):
    """The headline: accuracy vs throughput, vanilla highlighted."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = _color_map(recs)
    for r in recs:
        if r["req_s"] is None or r["mean_f1"] is None:
            continue
        van = r["system"] == "vanilla"
        ax.scatter(r["req_s"], r["mean_f1"], s=160 if van else 90,
                   marker="*" if van else "o", color=colors[r["sys_key"]],
                   edgecolor="black", linewidth=0.6, zorder=3)
        note = (f"van c{r['con']}" if van
                else f"gs{r['gs']} thr{r['thr']} eb{r['eb']} c{r['con']}")
        ax.annotate(note, (r["req_s"], r["mean_f1"]), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in colors.items()]
    ax.legend(handles=handles, fontsize=7, title="merge/scale/repr", loc="best")
    ax.set_xlabel("request throughput (req/s)")
    ax.set_ylabel("mean F1")
    ax.set_title("Accuracy vs throughput  (★ = vanilla baseline)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_headline_bars(recs, out):
    """One barh panel per headline metric — the at-a-glance per-run overview."""
    colors = _color_map(recs)
    panels = [
        ("mean_f1", "mean F1", "F1"),
        ("req_s", "throughput", "req/s"),
        ("out_tok_s", "output throughput", "tok/s"),
        ("ttft_med", "TTFT (median)", "ms"),
        ("tpot_med", "TPOT (median)", "ms"),
        ("comp_factor", "KV compression factor (× smaller)", "factor"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, (key, title, xl) in zip(axes.flat, panels):
        _barh(ax, recs, lambda r, k=key: r[k], title, xl, colors)
    fig.suptitle("Per-run headline metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_latency(recs, out):
    """Latency breakdown: large-scale (TTFT/E2E) and small-scale (TPOT/ITL) per run."""
    rs = sorted([r for r in recs if r["ttft_med"] is not None],
                key=lambda r: r["e2e_med"] or 0)
    y = np.arange(len(rs)); h = 0.4
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, max(6, 0.4 * len(rs))))
    a1.barh(y - h / 2, [r["ttft_med"] or 0 for r in rs], h, label="TTFT median", color="#4C72B0")
    a1.barh(y + h / 2, [r["e2e_med"] or 0 for r in rs], h, label="E2E median", color="#C44E52")
    a1.set_title("TTFT / E2E latency (median, ms)"); a1.set_xlabel("ms"); a1.legend(fontsize=8)
    a2.barh(y - h / 2, [r["tpot_med"] or 0 for r in rs], h, label="TPOT median", color="#55A868")
    a2.barh(y + h / 2, [r["itl_med"] or 0 for r in rs], h, label="ITL median", color="#8172B3")
    a2.set_title("TPOT / ITL latency (median, ms)"); a2.set_xlabel("ms"); a2.legend(fontsize=8)
    for a in (a1, a2):
        a.set_yticks(y); a.set_yticklabels([r["tag"] for r in rs], fontsize=6); a.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_compression_heatmap(recs, out):
    """Per-fusion-group compression factor across BFF runs (rows=run, cols=group)."""
    rs = [r for r in recs if r["comp_per_group"]]
    if not rs:
        return
    groups = sorted({g for r in rs for g in r["comp_per_group"]})
    M = np.full((len(rs), len(groups)), np.nan)
    for i, r in enumerate(rs):
        for j, g in enumerate(groups):
            if g in r["comp_per_group"]:
                M[i, j] = r["comp_per_group"][g]
    fig, ax = plt.subplots(figsize=(1.2 * len(groups) + 4, 0.4 * len(rs) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"g{g}" for g in groups])
    ax.set_yticks(range(len(rs))); ax.set_yticklabels([r["tag"] for r in rs], fontsize=6)
    ax.set_title("BFF compression factor per fusion group (× smaller)")
    ax.set_xlabel("fusion group (0=warmup excluded)")
    for i in range(len(rs)):
        for j in range(len(groups)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmean(M) else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="factor"); fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_decode_capacity(recs, out):
    """Decode-instance capacity signals — the mechanism BFF is meant to relieve."""
    colors = _color_map(recs)
    panels = [
        ("dec_running_max", "decode running batch (max)", "seqs"),
        ("dec_usage_max", "decode KV usage (max)", "%"),
        ("dec_preempt", "decode preemptions (cum)", "count"),
        ("net_blocks_freed", "blocks freed by fusion (net)", "blocks"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (key, title, xl) in zip(axes.flat, panels):
        _barh(ax, recs, lambda r, k=key: r[k], title, xl, colors)
    fig.suptitle("Decode-instance capacity / fusion effect", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_f1_distribution(recs, out):
    """Per-prompt F1 distribution (box) per run — shows spread, not just the mean."""
    rs = sorted([r for r in recs if r["per_sample_f1"]], key=lambda r: np.mean(r["per_sample_f1"]))
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(rs))))
    ax.boxplot([r["per_sample_f1"] for r in rs], vert=False, showmeans=True,
               flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_yticklabels([r["tag"] for r in rs], fontsize=6)
    ax.set_xlabel("per-prompt F1"); ax.set_title("F1 distribution per run (box = quartiles, ▲ = mean)")
    ax.grid(axis="x", alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_eb_paper(recs, out):
    """Paper-style ablation: compression factor, F1, and throughput vs encoded-batch size (the
    cross-batch window). One line per config group (merge/scale/repr/thr/gs/con/topology fixed,
    eb varied); the vanilla baseline is drawn as a dashed reference on the F1/throughput panels."""
    bff = [r for r in recs if r["system"] == "bff"]
    groups = defaultdict(list)
    for r in bff:
        groups[(r["merge"], r["scale"], r["repr"], r["thr"], r["gs"], r["con"], r["topo"])].append(r)
    groups = {k: sorted(v, key=lambda r: r["eb"]) for k, v in groups.items()
              if len({r["eb"] for r in v}) >= 2}
    if not groups:
        print("  (skip eb paper figure — need ≥2 encoded-batch sizes for some config)")
        return
    van = [r for r in recs if r["system"] == "vanilla"]
    van_f1 = np.mean([r["mean_f1"] for r in van if r["mean_f1"] is not None]) if van else None
    van_rps = np.mean([r["req_s"] for r in van if r["req_s"] is not None]) if van else None

    panels = [("comp_factor", "KV compression factor (× smaller)", None),  # vanilla has no fusion
              ("mean_f1", "mean F1", van_f1),
              ("req_s", "throughput (req/s)", van_rps)]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.125))
    for ax, (key, ylabel, baseline) in zip(axes, panels):
        for ci, (gkey, rs) in enumerate(sorted(groups.items())):
            xs = [r["eb"] for r in rs]
            ys = [r[key] for r in rs]
            lbl = f"{gkey[0]}/{gkey[1]}/{gkey[2]} thr{gkey[3]} gs{gkey[4]} c{gkey[5]} {gkey[6]}"
            ax.plot(xs, ys, "o-", color=cmap(ci % 10), label=lbl, linewidth=1.8, markersize=6)
        if baseline is not None:
            ax.axhline(baseline, ls="--", color="black", alpha=0.6, label="vanilla baseline")
        ax.set_xlabel("encoded batch size (cross-batch window)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("symlog", base=2)
        all_eb = sorted({r["eb"] for rs in groups.values() for r in rs})
        ax.set_xticks(all_eb); ax.set_xticklabels(all_eb)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title("Compression vs encoded batch size")
    axes[1].set_title("Accuracy vs encoded batch size")
    axes[2].set_title("Throughput vs encoded batch size")
    axes[1].legend(fontsize=12, loc="best")   # one shared legend (lines are identical across panels)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def plot_sweeps(recs, out_dir):
    """Auto 1-D sweeps: for each of threshold / group-size / encoded-batch / concurrency, plot F1
    and req/s vs that knob, drawing a line per group of runs that hold all OTHER knobs fixed."""
    bff = [r for r in recs if r["system"] == "bff"]
    dims = {"thr": "threshold", "gs": "group size", "eb": "encoded batch size", "con": "concurrency"}
    for dim, dim_label in dims.items():
        others = [d for d in ("thr", "gs", "eb", "con") if d != dim]
        groups = defaultdict(list)
        for r in bff:
            key = (r["merge"], r["scale"], r["repr"], r["topo"]) + tuple(r[o] for o in others)
            groups[key].append(r)
        groups = {k: v for k, v in groups.items() if len({r[dim] for r in v}) >= 2}
        if not groups:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
        for key, rs in sorted(groups.items()):
            rs = sorted(rs, key=lambda r: r[dim])
            xs = [r[dim] for r in rs]
            lbl = f"{key[0]}/{key[1]}/{key[2]} {key[3]} " + \
                  " ".join(f"{o}{key[4 + i]}" for i, o in enumerate(others))
            a1.plot(xs, [r["mean_f1"] for r in rs], "o-", label=lbl)
            a2.plot(xs, [r["req_s"] for r in rs], "o-", label=lbl)
        a1.set_xlabel(dim_label); a1.set_ylabel("mean F1"); a1.set_title(f"F1 vs {dim_label}")
        a2.set_xlabel(dim_label); a2.set_ylabel("req/s"); a2.set_title(f"throughput vs {dim_label}")
        for a in (a1, a2):
            a.grid(alpha=0.3); a.legend(fontsize=6)
        fig.tight_layout()
        path = os.path.join(out_dir, f"sweep_{dim}.png")
        fig.savefig(path, dpi=130); plt.close(fig)
        print(f"  wrote {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=here, help="directory of f1_*.json (default: this script's dir)")
    ap.add_argument("--out", default=None, help="output dir for plots/CSV (default: <dir>/plots)")
    ap.add_argument("--csv-only", action="store_true", help="write the summary CSV only, no plots")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    recs = load_records(args.dir)
    print(f"Loaded {len(recs)} runs from {args.dir}")
    if not recs:
        return
    write_csv(recs, os.path.join(out_dir, "summary.csv"))

    # quick stdout table
    print(f"\n{'tag':<48}{'F1':>7}{'req/s':>8}{'tok/s':>8}{'TTFTmed':>9}{'comp×':>7}")
    for r in sorted(recs, key=lambda x: -(x["mean_f1"] or 0)):
        print(f"{r['tag']:<48}{(r['mean_f1'] or 0):>7.3f}{(r['req_s'] or 0):>8.2f}"
              f"{(r['out_tok_s'] or 0):>8.0f}{(r['ttft_med'] or 0):>9.0f}"
              f"{(r['comp_factor'] or 0):>7.2f}")

    if args.csv_only:
        return
    if plt is None:
        print("\nmatplotlib not installed → CSV only. Install with: pip install matplotlib")
        return

    print("\nPlots:")
    plot_tradeoff(recs, os.path.join(out_dir, "accuracy_vs_throughput.pdf"))
    plot_headline_bars(recs, os.path.join(out_dir, "headline_metrics.pdf"))
    plot_latency(recs, os.path.join(out_dir, "latency_breakdown.pdf"))
    plot_compression_heatmap(recs, os.path.join(out_dir, "bff_compression_per_group.pdf"))
    plot_decode_capacity(recs, os.path.join(out_dir, "decode_capacity.pdf"))
    plot_f1_distribution(recs, os.path.join(out_dir, "f1_distribution.pdf"))
    plot_eb_paper(recs, os.path.join(out_dir, "eb_compression_f1_throughput.pdf"))
    plot_sweeps(recs, out_dir)
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
