"""The table for the client, from the runs ``run_client_table.sh`` saved.

One row per scenario, mode and setting: means over the repeats, with the range. Every label comes from
what the run's ``config.txt`` says, never from its folder name, and a cell whose runs do not match each
other is called out. Throughput is compared on tok/s, because req/s and wall time also move with answer
length. A gain counts only when every repeat of the arm beats every repeat of the baseline.

Fixed-length runs force text past the answer's natural end, so they say nothing about quality: those columns
stay empty there, and only the first-token check is counted, since the tail and restart checks fire on the
forced text.

Usage:
    python3 -m f1_benchmark.client_table TABLE_DIR      # prints it, writes client_table.md and client_runs.csv
"""

import csv
import glob
import json
import os
import re
import statistics as st
import sys

from f1_benchmark import sibling_drift
from f1_benchmark.compare_runs import _read, read_config, read_run
from f1_benchmark.degenerate_outputs import is_garbled

CODE_REQUEST = re.compile(r"## Code: ([^\s]+)")
NOTES = (
    "Each row uses the same prompts in the same order as its baseline. Hardware: one prefill and one decode "
    "node, TP8 each, GLM-5.1 W4A8, vLLM 0.18.0.",
    "`families` is selected in BFF's favour: real, unmodified Magicoder prompts, but only those with a "
    "near-identical partner, sent family by family (one software project, a different file each). Results there "
    "describe workloads whose concurrent requests share long context, not typical traffic. `121` and `200` "
    "use dataset order.",
    "Fixed-length runs give the same number of tokens per request, so they show throughput alone. They do not "
    "measure quality.",
    "`Sibling` counts answers that read like a near-identical sibling prompt's answer instead of their own (closer "
    "to the sibling's baseline answer than to their own). F1 and the right-file check cannot see this. A baseline "
    "row is each baseline repeat judged against the others: the rate at which the model does it on its own.",
)


def setting_of(cfg: dict) -> str:
    """What this run really was, from its configuration."""
    if cfg.get("arm") != "bff":
        return "baseline"
    if cfg.get("BFF_V2_DEDUP") == "0":
        return "dedup off"
    label = "no cap" if cfg.get("BFF_V2_MAX_REQ_DECLINE") == "1" else f"clip {cfg.get('BFF_V2_CLIP_FRAC', '?')}"
    label += ", no freeing" if cfg.get("BFF_V2_FREE_AT_PROMOTION") == "0" else ""
    # The layout is part of the setting: edge 20 (the default) and edge 10 + split 3 are different
    # experiments and must not be averaged into one row.
    edge, split = cfg.get("BFF_EDGE_LAYERS", "20"), cfg.get("BFF_EDGE_SPLIT", "1")
    if edge != "20" or split != "1":
        label += f", edge {edge}" + (f" split {split}" if split != "1" else "")
    if cfg.get("BFF_EDGE_AT_END") == "1":
        label += ", unshared at end"
    return label


def extras(run_dir: str, cfg: dict) -> dict:
    """The fields the sweep table does not read."""
    out = {"fixed": cfg.get("THROUGHPUT_ONLY") == "1", "fix": cfg.get("PROXY_CONTINUE_ANSWER", "0") == "1"}
    try:
        with open(os.path.join(run_dir, "results_latest", "benchmark_results.json")) as f:
            b = json.load(f)
    except (OSError, ValueError):
        b = {}
    completed = b.get("completed") or 0
    out["completed"] = completed
    out["mean_tokens"] = b["total_output_tokens"] / completed if completed and b.get("total_output_tokens") else None
    out["capped"] = b.get("finish_length_pct")
    for key, name in (("ttft_ms", "ttft"), ("tpot_ms", "tpot"), ("e2e_latency_ms", "e2e")):
        d = b.get(key) or {}
        out[name + "_p50"], out[name + "_p99"] = d.get("median"), d.get("p99")
    try:
        with open(os.path.join(run_dir, "results_latest", "raw_outputs.json")) as f:
            outs = json.load(f)
        outs = outs["outputs"] if isinstance(outs, dict) else outs
    except (OSError, ValueError):
        outs = []
    texts = [o.get("generated_text") or "" for o in outs]
    out["garbled"] = sum(is_garbled(t) for t in texts)
    asked = [(CODE_REQUEST.findall(o.get("prompt") or "") or [None])[-1] for o in outs]
    named = [(a, t) for a, t in zip(asked, texts) if a]
    out["right_file"] = (sum(a in t[:300] for a, t in named), len(named))
    logs = os.path.join(run_dir, "logs_latest")
    decode, proxy = _read(os.path.join(logs, "decode-0.txt")), _read(os.path.join(logs, "proxy.txt"))
    console = _read(os.path.join(run_dir, "console.txt"))
    out["refused"] = len(re.findall(r"continuation refused", proxy))
    continued = len(re.findall(r"BFF proxy continued answer", proxy))
    delaying = len(re.findall(r"Delaying free of", _read(os.path.join(logs, "prefill-0.txt"))))
    out["evictions"] = max(continued, len(re.findall(r"BFF recompute \|", decode)),
                           delaying - len(outs) if delaying else 0)
    saving = re.findall(r"bff v2 wire saving: ([\d.]+)%", console)
    layout = re.search(r"BFF KV sizing .*?groups=\d+ \[([\d, ]+)\]", decode)
    if saving and layout:
        sizes = [int(x) for x in layout.group(1).split(",")]
        # The saving is a share of the SIGNED blocks: every group but group 0 (the edges). Weighting
        # by the last group alone was right only for the two-group layout [40, 39]; on edge 10 +
        # split 3 ([20, 20, 20, 19]) it showed 1.9% where 6.0% of all KV was not pulled.
        out["kv_not_pulled"] = float(saving[-1]) * sum(sizes[1:]) / sum(sizes)
    else:
        out["kv_not_pulled"] = 0.0
    return out


def read_table(table_dir: str) -> list:
    runs = []
    for done in sorted(glob.glob(os.path.join(table_dir, "*", "*", "DONE"))):
        run_dir = os.path.dirname(done)
        cfg = read_config(run_dir)
        run = read_run(run_dir)
        run.update(extras(run_dir, cfg))
        run.update(cell=os.path.basename(os.path.dirname(run_dir)), setting=setting_of(cfg),
                   folder=os.path.basename(run_dir), scenario=cfg.get("scenario", "?"))
        runs.append(run)
    sibling_drift.mark(runs, key=lambda r: (r["cell"], r["fingerprint"], r["fixed"]))
    return runs


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return st.mean(vals) if vals else None


def _fmt(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


def _verdict(rows, base) -> str:
    if not base or rows[0]["setting"] == "baseline":
        return ""
    a, b = [r["tok_s"] for r in rows if r["tok_s"]], [r["tok_s"] for r in base if r["tok_s"]]
    if not a or not b:
        return ""
    pct = 100.0 * (st.mean(a) / st.mean(b) - 1)
    if len(a) < 2 or len(b) < 2:
        return f"{pct:+.1f}% (1 run: unconfirmed)"
    if min(a) > max(b):
        return f"{pct:+.1f}% (every repeat ahead)"
    if max(a) < min(b):
        return f"{pct:+.1f}% (every repeat behind)"
    return f"{pct:+.1f}% (repeats overlap: no clear difference)"


def problems(rows) -> list:
    """Anything that would make a row's mean misleading."""
    out = []
    for key, what in (("fingerprint", "prompts"), ("limit", "decode limit"), ("build", "build"), ("fix", "restart fix")):
        seen = {str(r.get(key)) for r in rows}
        if len(seen) > 1:
            out.append(f"mixed {what}: {sorted(seen)}")
    n = rows[0]["n"]
    short = [r["folder"] for r in rows if r["completed"] < n]
    if short:
        out.append(f"requests lost in {short}")
    if any(r["refused"] for r in rows):
        out.append("some continuations were refused")
    return out


def unlike(inside: list) -> list:
    """Why a cell's rows cannot be compared with each other, even if each row is consistent."""
    out = []
    for key, what in (("fixed", "fixed-length and normal runs"), ("fix", "restart fix on and off"),
                      ("fingerprint", "different prompts"), ("limit", "different decode limits")):
        seen = {str(r.get(key)) for r in inside}
        if len(seen) > 1:
            out.append(f"compares {what}: {sorted(seen)}")
    return out


def table(runs: list) -> list:
    head = ("| Scenario | Mode | Setting | Runs | Fix | tok/s (range) | vs baseline | Running | Evictions | KV not pulled | "
            "Freed | F1 | CodeBLEU | Valid syntax | Bad | Restarted | Sibling | Right file | Tokens | Capped | TTFT p50 s | "
            "TPOT p50 ms | TPOT p99 ms | E2E p99 s |")
    lines = [head, "|" + "---|" * 3 + "---:|" * 2 + "---|" + "---|" + "---:|" * 17]
    warnings = []
    order = {"baseline": 0}
    for cell in sorted({r["cell"] for r in runs}):
        inside = [r for r in runs if r["cell"] == cell]
        settings = sorted({r["setting"] for r in inside}, key=lambda s: (order.get(s, 1), s))
        base = [r for r in inside if r["setting"] == "baseline"]
        warnings += [f"{cell}: {p}" for p in unlike(inside)]
        for setting in settings:
            rows = [r for r in inside if r["setting"] == setting]
            for p in problems(rows):
                warnings.append(f"{cell} / {setting}: {p}")
            fixed = rows[0]["fixed"]
            tok = [r["tok_s"] for r in rows if r["tok_s"]]
            rng = f"{st.mean(tok):.1f} ({min(tok):.1f}-{max(tok):.1f})" if tok else "-"
            bad = sum(r["garbled"] if fixed else r["garbage"] for r in rows)
            right = sum(r["right_file"][0] for r in rows), sum(r["right_file"][1] for r in rows)
            ms = lambda k: _fmt(_mean(rows, k), ".0f")
            secs = lambda k: _fmt(_mean(rows, k) / 1000.0 if _mean(rows, k) is not None else None, ".0f")
            q = (lambda k, spec=".3f": "-" if fixed else _fmt(_mean(rows, k), spec))
            judged = [r["sibling"] for r in rows if r.get("sibling") is not None]
            sib = str(sum(judged)) if judged and not fixed else "-"
            lines.append(
                f"| {rows[0]['scenario']} | {'fixed' if fixed else 'normal'} | {setting} | {len(rows)} | "
                f"{'on' if rows[0]['fix'] else 'off'} | {rng} | {_verdict(rows, base) or '-'} | {_fmt(_mean(rows, 'running'), '.2f')} | "
                f"{_fmt(_mean(rows, 'evictions'), '.0f')} | {_fmt(_mean(rows, 'kv_not_pulled'))}% | {_fmt(_mean(rows, 'freed'), '.0f')} | "
                f"{q('f1')} | {q('codebleu', '.2f')} | {q('ast', '.1f')}{'' if fixed else '%'} | {bad} | "
                f"{'-' if fixed else sum(r['restarted'] for r in rows)} | {sib} | {'-' if fixed or not right[1] else f'{right[0]}/{right[1]}'} | "
                f"{_fmt(_mean(rows, 'mean_tokens'), '.0f')} | {_fmt(_mean(rows, 'capped'), '.0f')}% | {secs('ttft_p50')} | "
                f"{ms('tpot_p50')} | {ms('tpot_p99')} | {secs('e2e_p99')} |")
    return lines, warnings


CSV_FIELDS = ("cell", "folder", "setting", "scenario", "fixed", "fix", "limit", "build", "fingerprint", "n", "completed", "tok_s",
              "req_s", "wall", "running", "max_running", "waiting", "kv", "evictions", "prefill_k", "kv_not_pulled", "freed",
              "f1", "codebleu", "ast", "garbage", "garbled", "restarted", "sibling", "right_file", "refused", "mean_tokens", "capped",
              "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p50", "e2e_p99")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    runs = read_table(argv[0])
    if not runs:
        print(f"No finished runs under {argv[0]}")
        return 1
    lines, warnings = table(runs)
    text = ["# BFF against baseline", ""] + lines + [""]
    if warnings:
        text += ["**Check before using:**", ""] + [f"- {w}" for w in warnings] + [""]
    hits = [f"{r['cell']}/{sibling_drift.describe(r, h)}" for r in runs for h in r.get("sibling_hits") or []]
    if hits:
        text += ["**Answers that read like a sibling's (read them before quoting quality):**", ""] + \
                [f"- {h}" for h in hits] + [""]
    text += [f"- {n}" for n in NOTES]
    print("\n".join(text))
    with open(os.path.join(argv[0], "client_table.md"), "w") as f:
        f.write("\n".join(text) + "\n")
    with open(os.path.join(argv[0], "client_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)
    print(f"\nSaved {argv[0]}/client_table.md and client_runs.csv ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
