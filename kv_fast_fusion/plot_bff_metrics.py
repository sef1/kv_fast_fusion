#!/usr/bin/env python3
"""Plot BFF runtime metrics from a server log (terminal plots).

Parses the instrumentation lines emitted by the fast-fusion patches:
  - "BFF sched | step=… | running=… | waiting=… | free_blocks=… / … | block_usage=…%"
  - "BFF phase ms/step (avg/100): {…}"
  - "BFF KV sizing | num_gpu_blocks=… | groups=… | EFFECTIVE_CONCURRENCY=…x | …"

and renders terminal plots of running/waiting batch, KV block usage, and the per-step
phase breakdown over time. Uses termplotlib+gnuplot if available (per the project tip),
otherwise a dependency-free ASCII fallback.

Usage:
    python -m kv_fast_fusion.plot_bff_metrics server.log
    tail -f server.log | python -m kv_fast_fusion.plot_bff_metrics -      # stream / stdin
"""

import ast
import re
import shutil
import sys

_SCHED_RE = re.compile(
    r"BFF sched \| step=(\d+) \| running=(\d+) \| waiting=(\d+) \| "
    r"free_blocks=(\d+) / (\d+) \| block_usage=([\d.]+)%"
)
_PHASE_RE = re.compile(r"BFF phase ms/step \(avg/(\d+)\): (\{.*\})")
_SIZING_RE = re.compile(r"BFF KV sizing \|.*")

_HAVE_GNUPLOT = shutil.which("gnuplot") is not None
try:
    import termplotlib as _tpl  # noqa: F401
    _HAVE_TPL = True
except Exception:
    _HAVE_TPL = False


def parse(lines):
    sched = []  # (step, running, waiting, free, total, usage)
    phases = []  # (idx, {phase: ms})
    sizing = None
    for ln in lines:
        m = _SCHED_RE.search(ln)
        if m:
            sched.append((int(m[1]), int(m[2]), int(m[3]), int(m[4]),
                          int(m[5]), float(m[6])))
            continue
        m = _PHASE_RE.search(ln)
        if m:
            try:
                phases.append((len(phases), ast.literal_eval(m[2])))
            except Exception:
                pass
            continue
        if sizing is None:
            m = _SIZING_RE.search(ln)
            if m:
                sizing = m.group(0).split("BFF KV sizing | ", 1)[-1].strip()
    return sched, phases, sizing


def _ascii_line(xs, series, title, height=12, width=90):
    """Dependency-free multi-series line plot. series: list of (label, ys)."""
    print(f"\n=== {title} ===")
    if not xs:
        print("  (no data)")
        return
    allv = [v for _, ys in series for v in ys]
    lo, hi = min(allv), max(allv)
    if hi == lo:
        hi = lo + 1.0
    marks = "*o+x#@"
    n = len(xs)
    step = max(1, n // width)
    cols = list(range(0, n, step))
    grid = [[" "] * len(cols) for _ in range(height)]
    for si, (_, ys) in enumerate(series):
        ch = marks[si % len(marks)]
        for ci, i in enumerate(cols):
            y = ys[i]
            row = int((hi - y) / (hi - lo) * (height - 1))
            row = min(max(row, 0), height - 1)
            if grid[row][ci] == " ":
                grid[row][ci] = ch
    for r, row in enumerate(grid):
        yval = hi - (hi - lo) * r / (height - 1)
        print(f"{yval:10.1f} |{''.join(row)}")
    print(f"{'':10} +{'-' * len(cols)}")
    print(f"{'':10}  step {xs[cols[0]]} … {xs[cols[-1]]}")
    legend = "  ".join(f"{marks[i % len(marks)]}={lbl}"
                       for i, (lbl, _) in enumerate(series))
    print(f"{'':10}  {legend}")


def _tpl_line(xs, series, title):
    import termplotlib as tpl
    fig = tpl.figure()
    for lbl, ys in series:
        fig.plot(xs, ys, label=lbl, width=90, height=18)
    print(f"\n=== {title} ===")
    fig.show()


def plot(sched, phases, sizing):
    if sizing:
        print(f"\nKV sizing: {sizing}")

    line = _tpl_line if (_HAVE_TPL and _HAVE_GNUPLOT) else _ascii_line

    if sched:
        steps = [s[0] for s in sched]
        line(steps,
             [("running", [s[1] for s in sched]),
              ("waiting", [s[2] for s in sched])],
             "Batch (running / waiting) vs step")
        line(steps,
             [("block_usage%", [s[5] for s in sched])],
             "KV block usage % vs step")

    if phases:
        names = sorted({k for _, d in phases for k in d})
        idx = [p[0] for p in phases]
        line(idx,
             [(nm, [d.get(nm, 0.0) for _, d in phases]) for nm in names],
             "Phase host ms/step vs window")
        last = phases[-1][1]
        print("\nLatest phase ms/step:")
        for nm in names:
            bar = "#" * int(min(last.get(nm, 0.0), 60))
            print(f"  {nm:12} {last.get(nm, 0.0):7.2f} |{bar}")

    if not sched and not phases:
        print("No BFF metric lines found. Pipe a server log containing "
              "'BFF sched' / 'BFF phase' lines.")


def main(argv):
    if not _HAVE_GNUPLOT:
        print("(note: gnuplot not found → ASCII fallback. "
              "`sudo apt-get install -y gnuplot` for termplotlib plots.)",
              file=sys.stderr)
    src = argv[1] if len(argv) > 1 else "-"
    if src == "-":
        lines = sys.stdin
        plot(*parse(lines))
    else:
        with open(src, "r", errors="ignore") as f:
            plot(*parse(f))


if __name__ == "__main__":
    main(sys.argv)
