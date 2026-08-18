#!/usr/bin/env python3
"""Paired per-sample quality comparison of two benchmark runs → is the BFF delta real or noise?

Usage:
    python bff_f1_paired.py <baseline_results.json> <bff_results.json>

Why paired. At n=200 the UNPAIRED test cannot resolve the effect we care about: per-sample F1 has
sd ~0.124 against a mean of ~0.09, because a handful of prompts score 0.5-0.9 while the median is
0.05 (in the con128 run the top 5 of 200 samples carried 22% of all F1 mass). That yields z=1.14 for
a 14% relative difference — indistinguishable from zero, needing ~1200 prompts per arm to resolve.
But both runs answer the SAME prompts, and prompt difficulty is the dominant variance term. Pairing
on prompt index cancels it, so the paired test can resolve at n=200 what the unpaired one cannot.

Reports the paired mean delta with a CI, a paired t, and a Wilcoxon signed-rank (which does not
assume normality — with this distribution shape, it is the one to believe). Also breaks the delta
down by finish_reason and by input length, because WHERE the loss sits names the cause:
  - concentrated in requests that hit the token cap → a termination/ramble effect
  - uniform across all prompts                     → systematic KV error
  - a few large negative outliers                  → a small number of badly-fused blocks

Exit status is 0 always; this is a report, not a gate.
"""
import json
import math
import sys


def _load(path):
    with open(path) as f:
        d = json.load(f)
    ev = d.get("evaluation") or d
    xs = ev.get("per_sample_f1")
    if not xs:
        raise SystemExit(f"{path}: no per_sample_f1 array (the summary alone is not enough — the "
                         f"paired test needs the per-sample values)")
    return d, [float(v) for v in xs]


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))


def _norm_sf(z):
    """P(Z > z) for the standard normal, via erfc — avoids a scipy dependency."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _wilcoxon(deltas):
    """Signed-rank statistic with a normal approximation and average ranks for ties. Returns
    (n_nonzero, two-sided p). Robust to the heavy tail that makes the mean untrustworthy here."""
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    if n < 10:
        return n, float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0                  # average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return n, float("nan")
    z = (w_plus - mu) / sigma
    return n, 2.0 * _norm_sf(abs(z))


def _bucket_report(title, groups):
    """groups: {label: [deltas]} — print mean delta and n per bucket, worst first."""
    if not groups:
        return
    print(f"\n{title}")
    for label, ds in sorted(groups.items(), key=lambda kv: _mean(kv[1]) if kv[1] else 0.0):
        if ds:
            print(f"  {label:<28s} n={len(ds):<5d} mean delta={_mean(ds):+.4f}")


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    base_doc, base = _load(argv[1])
    bff_doc, bff = _load(argv[2])
    if len(base) != len(bff):
        raise SystemExit(f"arrays differ in length ({len(base)} vs {len(bff)}) — the runs did not "
                         f"answer the same prompt set, so they cannot be paired")
    n = len(base)
    deltas = [b - a for a, b in zip(base, bff)]     # negative = BFF worse

    print(f"n = {n} paired samples")
    print(f"baseline mean F1 = {_mean(base):.4f}   BFF mean F1 = {_mean(bff):.4f}")

    md, sdd = _mean(deltas), _sd(deltas)
    se = sdd / math.sqrt(n) if n else 0.0
    print(f"\nPAIRED delta (BFF - baseline): mean = {md:+.4f}  sd = {sdd:.4f}  se = {se:.4f}")
    if se > 0:
        t = md / se
        print(f"  95% CI = [{md - 1.96 * se:+.4f}, {md + 1.96 * se:+.4f}]")
        print(f"  paired t = {t:+.2f}   p ~= {2.0 * _norm_sf(abs(t)):.4f}")
    nzw, pw = _wilcoxon(deltas)
    print(f"  Wilcoxon signed-rank: n_nonzero = {nzw}  p ~= {pw:.4f}"
          if not math.isnan(pw) else f"  Wilcoxon: too few nonzero deltas ({nzw})")

    # Unpaired comparison for contrast — this is the test that could NOT resolve the effect.
    se_unp = math.sqrt(_sd(base) ** 2 / n + _sd(bff) ** 2 / n)
    if se_unp > 0:
        print(f"\n  (unpaired z for the same difference = {md / se_unp:+.2f} — shown for contrast; "
              f"pairing is what buys the power here)")

    worse = sum(1 for d in deltas if d < 0)
    better = sum(1 for d in deltas if d > 0)
    print(f"\nper-prompt direction: BFF worse on {worse}, better on {better}, tied on "
          f"{n - worse - better}")
    ranked = sorted(range(n), key=lambda i: deltas[i])
    print("  largest regressions (idx, baseline -> bff):")
    for i in ranked[:5]:
        print(f"    #{i:<4d} {base[i]:.3f} -> {bff[i]:.3f}   ({deltas[i]:+.3f})")

    # Where does the loss sit? Both cuts are optional — they need fields the harness may not emit.
    for doc, name in ((bff_doc, "BFF"), (base_doc, "baseline")):
        reasons = doc.get("finish_reasons") or doc.get("per_sample_finish_reason")
        if reasons and len(reasons) == n:
            groups = {}
            for r, d in zip(reasons, deltas):
                groups.setdefault(str(r), []).append(d)
            _bucket_report(f"delta split by {name} finish_reason "
                           f"(concentration in 'length' => a termination effect):", groups)
            break
    else:
        print("\n  (no per-sample finish_reason in either file — add it to the harness to test "
              "whether the loss concentrates in requests that hit the token cap)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
