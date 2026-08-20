"""Merge a run's BFF metrics into its benchmark result JSON and print one screen summary.

Extracted verbatim from the inline collector in
``examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh`` so the
Mooncake launch script can reuse it instead of copying 130 lines of heredoc. Both transports emit
the same shapes, so one collector serves both.

Everything here runs POST-run, so it has zero impact on throughput / elapsed time.

Sources:
  * producer overhead + compression — the per-process ``bff_stats_*.json`` the producers dump into
    ``stats_dir`` (always-current cumulative totals; no log flood, and no throttled-line scrape that
    a prefill-only producer would miss);
  * scheduler / consumer metrics — scraped from the server logs:
        "BFF sched | ... free_blocks=F / T | block_usage=U% | running=.. | preempt(cum)=.."
        "Block merging freed N blocks"            (D net free)
        "redirects_applied=N | reps_unresolved=M"

Merged under ``bff_overhead`` / ``bff_compression`` / ``bff_sched`` / ``bff_blocks_freed`` /
``bff_redirects_applied``.

Usage:  python3 -m kv_fast_fusion.tools.collect_bff_stats RESULT_JSON STATS_DIR LOG [LOG ...]
"""

import glob
import json
import os
import re
import sys

sched_pat = re.compile(r"BFF sched \| step=\d+ \| running=(\d+) \| waiting=(\d+) \| "
                       r"free_blocks=(\d+) / (\d+) \| block_usage=([\d.]+)% \| preempt\(cum\)=(\d+)")
freed_pat = re.compile(r"Block merging freed (-?\d+) blocks")
redir_pat = re.compile(r"redirects_applied=(\d+) \| reps_unresolved=(\d+)")
# vLLM's OWN engine line, emitted by vanilla and BFF alike — the only saturation signal that is
# directly comparable between the two (the BFF sched line does not exist in a vanilla run).
engine_pat = re.compile(r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
                        r"GPU KV cache usage: ([\d.]+)%")
# vLLM's LOCAL prefix-cache hit rate (the leading ", " excludes "External prefix cache hit rate",
# which under P/D is just the connector reporting itself). On a PREFILL log this is an independent,
# BFF-free measure of how much genuinely repeated content the workload contains — the denominator
# any compression claim has to be read against. CodeFeedback measured 1.3%/1.9%.
prefix_pat = re.compile(r", Prefix cache hit rate: ([\d.]+)%")
# KV transfer failures and what they cost. Under Mooncake's PULL model the producer writes straight
# into the decode's registered KV cache, so a transfer can only proceed as fast as D frees blocks —
# and at saturation it starts failing (measured: 0 failures below ~85% decode KV usage, 8-26/min at
# 99.9%). Each failure re-prefills that request ON THE DECODE GPU, which is throughput subtracted
# from the very number a BFF-vs-vanilla comparison reads. NCCL has no equivalent because it pushes
# into a recv buffer that spills to pinned host memory, so it never appears there.
pull_fail_pat = re.compile(r"pull FAILED for \S+ \((\d+) blocks invalid\)")
# vLLM's OWN scheduler line, so it is present in a vanilla run too — that is what makes the arms
# comparable. It is also the authoritative recompute cost.
recompute_pat = re.compile(r"Recovered from KV load failure: (\d+) request\(s\) rescheduled "
                           r"\((\d+) tokens affected\)")
send_timeout_pat = re.compile(r"timed out after \d+ seconds without being sent")
# A producer holding a full KV cache with nothing running is wedged behind KV that decode has not
# pulled: it cannot start new prefills. On a decode this combination cannot persist.
WEDGE_PCT = 95.0
# What counts as "the cache is full". Below this the run is ramping or draining, and whole-run means
# over those phases say nothing about behaviour at saturation — which is the only regime where
# compression can buy anything.
SATURATED_PCT = 95.0


def stats(xs):
    return None if not xs else {"min": min(xs), "mean": sum(xs) / len(xs),
                                "max": max(xs), "last": xs[-1]}


def _is_wedged(v: dict) -> bool:
    """A wedge is a SUSTAINED state, not a moment. Every instance is briefly full-and-idle while it
    drains at the end of a run, so require the condition to hold over a real share of the samples."""
    n = v.get("sched_samples") or 0
    return bool(n) and v.get("wedged_samples", 0) >= max(5, 0.05 * n)


def _decode_log(per_log: dict):
    """The decode instance's entry — the one whose saturation the summary should quote.

    Prefill logs also appear in these maps, and a wedged producer sits at 100% cache with nothing
    running, so picking arbitrarily can report `running mean=0` as if the decode were idle."""
    if not per_log:
        return None
    for lg, v in per_log.items():
        if "decode" in os.path.basename(lg).lower():
            return v
    # No naming convention to lean on: the decode is the instance that actually runs requests.
    return max(per_log.values(), key=lambda v: (v.get("running") or {}).get("max", 0))


def _interp(q: float, known: list) -> float | None:
    """Value at quantile `q`, linearly interpolated between the audit's reported quantiles.

    The audit stores four points, and the quantile of interest (0.5^(1/pool)) almost never lands on
    one of them. Snapping to the next reported point instead would move the bar by a full decile at
    small pool sizes, which is the range where the verdict is closest to the line."""
    if not known:
        return None
    if q <= known[0][0]:
        return known[0][1]
    for (lo_q, lo_v), (hi_q, hi_v) in zip(known, known[1:]):
        if q <= hi_q:
            span = hi_q - lo_q
            return lo_v if span <= 0 else lo_v + (q - lo_q) / span * (hi_v - lo_v)
    return known[-1][1]


def error_budget(accepted: dict, threshold: float | None) -> str | None:
    """What the configured threshold ALLOWS, and how much of the result survives a strict one.

    A cosine threshold is not a similarity preference — at matched norms it is an error budget:
    ``rel_err = sqrt(2 - 2*cos)``, so 0.75 admits substitutions that differ by 71% in relative L2,
    and only cos > 0.98 keeps the error under 20%. Pairing that with the share of merges above 0.95
    says, without a sweep, how much of the reported compression would survive tightening the bar.
    On the 2048-request run 1.3% of merges cleared 0.95, i.e. 21.3% of cache saved became 0.3%."""
    n = sum(accepted.values())
    if not n:
        return None
    hi = sum(c for label, c in accepted.items() if float(label.split("-")[0]) >= 0.95)
    out = f"{hi} of {n} merges ({100.0 * hi / n:.1f}%) above cos 0.95"
    if threshold is not None:
        out = (f"threshold {threshold:g} admits rel_err up to "
               f"{(max(0.0, 2 - 2 * threshold)) ** 0.5:.3f} | " + out)
    return out


def null_model_verdict(accepted: dict, floor: dict, pool: int) -> str | None:
    """Is the accepted-cosine distribution better than picking the most similar of `pool` UNRELATED
    blocks?

    A redirect keeps the best candidate above the threshold, so even with no real redundancy the
    accepted cosines land in the upper tail of the random-pair distribution simply by being a max
    over `pool` draws. The null model is that order statistic: the median of the best of N draws
    sits at the F⁻¹(0.5^(1/N)) quantile of the random-pair distribution. Beating it means genuine
    near-duplicates were found; matching it means the merges are substitutions of interchangeable
    blocks, which may still be worth doing — but is not deduplication and will not scale with pool
    size the way deduplication would.

    `accepted` is the binned histogram, `floor` the random-pair quantiles from the audit."""
    n = sum(accepted.values())
    if not n or not floor or pool < 2:
        return None
    # Median of the accepted distribution, taken at bin granularity (lower edge of the bin holding
    # the middle sample) so it is directly comparable to the quantiles below.
    seen, acc_p50 = 0, None
    for label, count in accepted.items():
        seen += count
        if seen >= n / 2:
            acc_p50 = float(label.split("-")[0])
            break
    q = 0.5 ** (1.0 / pool)                       # quantile the best-of-pool median sits at
    known = [(lvl, floor.get(k)) for lvl, k in
             ((0.50, "p50"), (0.90, "p90"), (0.99, "p99"), (1.0, "max"))
             if floor.get(k) is not None]
    null_p50 = _interp(q, known)
    if acc_p50 is None or null_p50 is None:
        return None
    verdict = ("consistent with substitution of interchangeable blocks, NOT duplicate detection"
               if acc_p50 <= null_p50 + 1e-9 else "above the noise model — real near-duplicates")
    return (f"accepted p50~{acc_p50:.2f} vs best-of-{pool} null ~{null_p50:.2f} -> {verdict}")


def collect(result_file: str, stats_dir: str, logs: list[str]) -> dict:
    # Producer fuse stats (overhead + compression) from the dumped per-process JSON files.
    ov_per, cm_per, cfg_per, v2_per = {}, {}, {}, {}
    for sf in sorted(glob.glob(os.path.join(stats_dir, "bff_stats_*.json"))):
        try:
            with open(sf) as f:
                s = json.load(f)
        except Exception:
            continue
        name = os.path.basename(sf)
        if s.get("bff_version") == 2:
            # v2 dumps from the DECODE (it is the side that decides), and reports blocks it never
            # requested. There is no producer-claim/landed gap to reconcile here: a block that is
            # not requested is not transferred, so the wire saving is the compression.
            v2_per[name] = s
            continue
        if s.get("steps"):
            ov_per[name] = {"avg_group_dedup_ms": s["overhead_avg_group_dedup_ms"],
                            "groups": s["steps"],
                            # Queue drain is what the step waited on, not what fusion cost; the
                            # percentiles separate a genuinely slow path from a cold-start outlier
                            # dominating a short run.
                            "avg_queue_drain_ms": s.get("overhead_avg_queue_drain_ms", 0.0),
                            "pct": s.get("overhead_ms_pct") or {}}
        if s.get("total_blocks"):
            cm_per[name] = {"avg_factor": s["compression_avg_factor"],
                            "total_blocks": s["total_blocks"], "freed": s["freed"],
                            "per_group": {int(gi): r for gi, r
                                          in s.get("compression_per_group", {}).items()}}
        # Fusion configuration + cross-request index state. Which backend ran, how much of the
        # compression came from EARLIER requests vs the current batch, how big the index grew, and
        # where the accepted-merge cosines sit (the last one says whether BFF_THRESHOLD is the
        # quality lever or the merges are already near-identical).
        cfg_per[name] = {k: s[k] for k in (
            "cross_index", "ff_groups", "encoded_batch_size", "cross_batch_redirects",
            "within_batch_redirects", "registry_blocks", "lsh_entries", "lsh_accept_cos",
            "lsh_reject_cos", "min_cos_for_budget",
            "lsh_owners", "lsh_evicted", "layers_fused", "layers_total",
            "thresholds_per_group", "threshold", "audit_random_pair_cos",
            "lsh_accept_rel_err")
            if k in s}

    sched_per, freed_per, redir_per, sat_per, redundancy_per = {}, {}, {}, {}, {}
    xfer_per = {}
    for lg in logs:
        runs, waits, frees, usages, total, preempt_last = [], [], [], [], None, None
        freed_sum = freed_cnt = redir_app = redir_unres = redir_cnt = 0
        sat_run, sat_wait, eng_n, prefix_hits = [], [], 0, []
        fail_cnt = fail_blocks = recov_reqs = recov_tokens = send_timeouts = wedged = 0
        try:
            with open(lg) as f:
                for line in f:
                    e = engine_pat.search(line)
                    if e:
                        eng_n += 1
                        if float(e.group(3)) >= SATURATED_PCT:
                            sat_run.append(int(e.group(1)))
                            sat_wait.append(int(e.group(2)))
                    p = prefix_pat.search(line)
                    if p:
                        prefix_hits.append(float(p.group(1)))
                    s = sched_pat.search(line)
                    if s:
                        runs.append(int(s.group(1)))
                        waits.append(int(s.group(2)))
                        frees.append(int(s.group(3)))
                        total = int(s.group(4))
                        usages.append(float(s.group(5)))
                        preempt_last = int(s.group(6))
                        if float(s.group(5)) >= WEDGE_PCT and int(s.group(1)) == 0:
                            wedged += 1
                    pf = pull_fail_pat.search(line)
                    if pf:
                        fail_cnt += 1
                        fail_blocks += int(pf.group(1))
                    rc = recompute_pat.search(line)
                    if rc:
                        recov_reqs += int(rc.group(1))
                        recov_tokens += int(rc.group(2))
                    if send_timeout_pat.search(line):
                        send_timeouts += 1
                    fr = freed_pat.search(line)
                    if fr:
                        freed_sum += int(fr.group(1))
                        freed_cnt += 1
                    rd = redir_pat.search(line)
                    if rd:
                        redir_app += int(rd.group(1))
                        redir_unres += int(rd.group(2))
                        redir_cnt += 1
        except FileNotFoundError:
            continue
        if frees:
            sched_per[lg] = {"total_blocks": total, "free_blocks": stats(frees),
                             "block_usage_pct": stats(usages), "running": stats(runs),
                             "waiting": stats(waits), "preempt_cum": preempt_last}
        if freed_cnt:
            freed_per[lg] = {"net_blocks_freed": freed_sum, "merge_events": freed_cnt}
        if redir_cnt:
            redir_per[lg] = {"redirects_applied": redir_app, "reps_unresolved": redir_unres,
                             "apply_calls": redir_cnt}
        if sat_run:
            sat_per[lg] = {"samples": len(sat_run), "of_samples": eng_n,
                           "running": stats(sat_run), "waiting": stats(sat_wait)}
        if prefix_hits and max(prefix_hits) > 0:
            redundancy_per[lg] = {"prefix_cache_hit_pct": stats(prefix_hits)}
        if fail_cnt or recov_reqs or send_timeouts or wedged:
            xfer_per[lg] = {"failed_pulls": fail_cnt, "invalid_blocks": fail_blocks,
                            "recomputed_requests": recov_reqs,
                            "recomputed_tokens": recov_tokens,
                            "producer_send_timeouts": send_timeouts,
                            "wedged_samples": wedged, "sched_samples": len(usages)}

    try:
        with open(result_file) as f:
            data = json.load(f)
    except Exception:
        data = {}

    if ov_per:
        avg = sum(v["avg_group_dedup_ms"] for v in ov_per.values()) / len(ov_per)
        drain = sum(v.get("avg_queue_drain_ms", 0.0) for v in ov_per.values()) / len(ov_per)
        steps = sum(v["groups"] for v in ov_per.values())
        p50s = [v["pct"]["p50"] for v in ov_per.values() if v.get("pct")]
        p99s = [v["pct"]["p99"] for v in ov_per.values() if v.get("pct")]
        data["bff_overhead"] = {"producer_avg_group_dedup_ms": avg,
                                "producer_avg_queue_drain_ms": drain, "per_prefill": ov_per}
        print(f"  bff overhead: producer avg group dedup {avg:.3f} ms"
              + (f" (p50 {min(p50s):.3f} p99 {max(p99s):.3f})" if p50s and p99s else "")
              + f" | queue drain {drain:.3f} ms | steps={steps}")
        if steps < 20:
            print(f"    ! only {steps} fusion steps sampled — the mean is cold-start dominated")
    elif not v2_per:
        print("  bff overhead: no bff_stats_*.json with steps>0 "
              "(BFF_PD_FUSE off, or producer ran no fusion groups)")

    if v2_per:
        planned = sum(s.get("blocks_planned", 0) for s in v2_per.values())
        dropped = sum(s.get("blocks_not_requested", 0) for s in v2_per.values())
        resident = sum(s.get("blocks_not_requested_resident", 0) for s in v2_per.values())
        same = sum(s.get("blocks_not_requested_same_pull", 0) for s in v2_per.values())
        applied = sum(s.get("aliases_applied", 0) for s in v2_per.values())
        recomp = sum(s.get("aliases_recomputed", 0) for s in v2_per.values())
        sigfail = sum(s.get("signature_phase_failed", 0) for s in v2_per.values())
        pct = 100.0 * dropped / planned if planned else 0.0
        data["bff_v2"] = {"blocks_planned": planned, "blocks_not_requested": dropped,
                          "wire_saving_pct": pct, "from_resident": resident,
                          "from_same_pull": same, "aliases_applied": applied,
                          "aliases_recomputed": recomp, "signature_phase_failed": sigfail,
                          "per_decode": v2_per}
        landed = 100.0 * applied / max(1, applied + recomp)
        data["bff_v2"]["pct_landed"] = landed
        # "0.0% saving" means two completely different things: asked and found nothing, or never
        # asked at all. The first Ascend run was the second for a whole benchmark and looked
        # identical to the first. Say which.
        # A connector predating this counter reports nothing rather than zero, and the two must not
        # print alike: "exchanges=0" claims the mechanism never ran, while an ABSENT counter says
        # only that this build cannot tell. mooncake_connector_ff_v2_legacy.py — kept verbatim as a
        # measurement baseline, so it cannot gain the counter — is the second case, and printing it
        # as the first made a legacy run that deduplicated 5.9% of the wire look stone dead.
        reporting = [s for s in v2_per.values() if "exchanges" in s]
        exchanges = sum(s.get("exchanges", 0) for s in reporting)
        skips: dict = {}
        for s in v2_per.values():
            for k, v in (s.get("exchange_skip_reasons") or {}).items():
                skips[k] = skips.get(k, 0) + v
        if reporting:
            data["bff_v2"]["exchanges"] = exchanges
            data["bff_v2"]["skip_reasons"] = skips
        if reporting and exchanges == 0 and any(skips.values()):
            data["bff_v2"]["inert"] = True
            print("  ! bff v2 INERT: not one signature exchange was attempted, so no block could "
                  "ever be deduplicated. Reasons: "
                  + " ".join(f"{k}={v}" for k, v in sorted(skips.items(), key=lambda kv: -kv[1])
                             if v))
        print(f"  bff v2 (DECODE DECIDES): {dropped} of {planned} blocks never requested "
              f"= {pct:.1f}% of the wire | resident={resident} same-pull={same} "
              f"| exchanges={exchanges if reporting else 'n/a (not reported by this connector)'}")
        if reporting and exchanges and any(skips.values()):
            print("    exchanges skipped: "
                  + " ".join(f"{k}={v}" for k, v in sorted(skips.items(), key=lambda kv: -kv[1])
                             if v))
        print(f"    aliases applied={applied} ({landed:.1f}% landed) | recompute (never-written "
              f"block, see the connector docstring)={recomp} | signature phase unavailable="
              f"{sigfail} pulls")
        # WHY an alias failed, not just how many. Reaching here with owner_never_batched dominant
        # means the maps are staged before the transfer completes (the first v2 run: 26,509 of
        # 26,531); rep_not_resident dominant means representatives are being freed inside the
        # decide->apply window and want pinning.
        causes: dict = {}
        for s in v2_per.values():
            for k, v in (s.get("alias_failure_reasons") or {}).items():
                causes[k] = causes.get(k, 0) + v
        if any(causes.values()):
            data["bff_v2"]["failure_causes"] = causes
            print("    why they failed: "
                  + " ".join(f"{k}={v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])
                             if v))
        for name, s in v2_per.items():
            idx = s.get("dedup_index_blocks") or {}
            if idx:
                print(f"    dedup index [{name}]: "
                      + " ".join(f"g{g}={n}" for g, n in sorted(idx.items())))
            # Per group: a group saving a lot while its accepted cosines hug the bar is matching
            # noise, not finding duplicates — that is the threshold signal, and it is per group
            # because each group's blocks have their own similarity floor.
            per_g = s.get("wire_saving_per_group") or {}
            thr_g = s.get("thresholds_per_group") or {}
            if per_g:
                print("    wire saving per group: " + " ".join(
                    f"g{g}={v['pct']:.0f}%({v['not_requested']}/{v['planned']}"
                    f"@thr{thr_g.get(g, '?')})" for g, v in sorted(per_g.items())))
            # What the norm bought over a cosine-only bar. Cosine is scale-free, so these are pairs
            # that looked aligned and would still have been bad substitutions.
            rej = s.get("rejected_by_rel_err") or {}
            if any(rej.values()):
                kept = sum(v["not_requested"] for v in per_g.values()) or 1
                n = sum(rej.values())
                print(f"    rejected by rel_err<={s.get('max_rel_err')}: {n} "
                      f"({100.0 * n / (n + kept):.0f}% of cosine-passing candidates) | "
                      + " ".join(f"g{g}={v}" for g, v in sorted(rej.items())))
            for label, key in (("accepted-cosine", "lsh_accept_cos"),
                               ("substitution rel-err", "lsh_accept_rel_err"),
                               ("rejected-cosine", "lsh_reject_cos")):
                for gi, bins in sorted((s.get(key) or {}).items()):
                    nz = " ".join(f"{k}:{v}" for k, v in bins.items() if v)
                    if nz:
                        print(f"    {label} g{gi}: {nz}")
            # Split the rejections at the cosine below which NO norm ratio can meet the budget.
            # Below it a rejection is permanently unreachable; above it, only the norm ratio lost it.
            # Without this line the histogram cannot answer "would rescaling the rep have helped?".
            floor = s.get("min_cos_for_budget")
            rej_hist = s.get("lsh_reject_cos") or {}
            if floor and rej_hist:
                lo = hi = 0
                for bins in rej_hist.values():
                    for label, n in bins.items():
                        # Bin labels are "<lo>-<hi>"; a bin whose TOP is at or below the floor is
                        # wholly unreachable. Bins straddling it count as reachable, which
                        # over-states the recoverable share rather than inventing precision.
                        top = float(label.split("-")[1])
                        (lo, hi) = (lo + n, hi) if top <= floor else (lo, hi + n)
                if lo + hi:
                    print(f"    of those rejections: {hi} sit above cos>={floor:.3f} (recoverable "
                          f"only by a better-matched rep) and {lo} below it "
                          f"({100.0 * lo / (lo + hi):.0f}%, unreachable at any norm ratio)")

    if cm_per:
        # Compression FACTOR = total/(total-freed) = how many× smaller the KV cache gets. Overall is
        # block-weighted across producers (ΣB / Σ(B-freed)); per-group is the mean factor across them.
        B = sum(v["total_blocks"] for v in cm_per.values())
        F = sum(v["freed"] for v in cm_per.values())
        avg_factor = B / max(1, B - F)
        gids = sorted({gi for v in cm_per.values() for gi in v["per_group"]})
        per_group = {gi: sum(v["per_group"][gi] for v in cm_per.values() if gi in v["per_group"])
                         / sum(1 for v in cm_per.values() if gi in v["per_group"]) for gi in gids}
        data["bff_compression"] = {"avg_factor": avg_factor, "per_group": per_group,
                                   "per_prefill": cm_per}
        print(f"  bff compression (x smaller): avg_factor={avg_factor:.4f} | per_group "
              + " ".join(f"g{gi}={r:.4f}" for gi, r in per_group.items()))
    elif not v2_per:
        print("  bff compression: no bff_stats_*.json with total_blocks>0")

    # The headline. What the producer CLAIMS it merged and what the decode instance actually freed
    # are different numbers, and only the second one is compression: a redirect naming a rep that D
    # has already freed resolves to nothing and the owner keeps its own block. Reporting the claim
    # produced a "73.1x" on a configuration whose arithmetic ceiling was 14.3%.
    freed_D = sum(v["net_blocks_freed"] for v in freed_per.values())
    cache = sum(v["total_blocks"] for v in sched_per.values() if v.get("total_blocks"))
    if freed_D and cache:
        pct = 100.0 * freed_D / cache
        claimed = sum(v["freed"] for v in cm_per.values())
        landed = (100.0 * sum(v["redirects_applied"] for v in redir_per.values())
                  / max(1, sum(v["redirects_applied"] + v["reps_unresolved"]
                               for v in redir_per.values()))) if redir_per else None
        fused = sum(s.get("layers_fused", 0) for s in cfg_per.values() if s)
        tot_l = max((s.get("layers_total", 0) for s in cfg_per.values() if s), default=0)
        ceiling = 100.0 * fused / tot_l / max(1, len(cfg_per)) if tot_l else None
        data["bff_decode_compression"] = {
            "blocks_freed_on_decode": freed_D, "decode_cache_blocks": cache,
            "pct_of_cache": pct, "avg_factor": cache / max(1, cache - freed_D),
            "producer_claimed_blocks": claimed, "pct_landed": landed,
            "ceiling_pct_of_layers": ceiling}
        print(f"  bff compression (DECODE-APPLIED): {freed_D} blocks freed = {pct:.1f}% of cache "
              f"({cache / max(1, cache - freed_D):.3f}x)"
              + (f" | producer claimed {claimed}, {landed:.0f}% landed" if landed is not None else "")
              + (f" | ceiling {ceiling:.1f}% of layers" if ceiling else ""))

    if any(cfg_per.values()):
        data["bff_fusion_cfg"] = cfg_per
        for name, c in cfg_per.items():
            if not c:
                continue
            cross, within = c.get("cross_batch_redirects", 0), c.get("within_batch_redirects", 0)
            pool = c.get("lsh_entries") or c.get("registry_blocks") or {}
            print(f"  bff fusion [{name}]: index={c.get('cross_index', '?')} "
                  f"groups={c.get('ff_groups') or 'all'} | redirects cross={cross} within={within} "
                  f"| pool={{{', '.join(f'g{g}={n}' for g, n in sorted(pool.items()))}}}")
            if c.get("lsh_evicted"):
                print(f"    lsh reps retired after D freed them: {c['lsh_evicted']} "
                      f"| live owners {c.get('lsh_owners') or {}}")
            hist = c.get("lsh_accept_cos") or {}
            floors = c.get("audit_random_pair_cos") or {}
            pools = c.get("lsh_entries") or {}
            for gi, bins in sorted(hist.items()):
                nz = " ".join(f"{k}:{v}" for k, v in bins.items() if v)
                if nz:
                    print(f"    accepted-cosine g{gi}: {nz}")
                v = null_model_verdict(bins, floors.get(gi) or {}, int(pools.get(gi) or 0))
                if v:
                    print(f"      null model g{gi}: {v}")
                thr = (c.get("thresholds_per_group") or {}).get(gi, c.get("threshold"))
                b = error_budget(bins, thr)
                if b:
                    print(f"      error budget g{gi}: {b}")
            # The substitution error the decode actually inherits, which cosine alone cannot show.
            for gi, bins in sorted((c.get("lsh_accept_rel_err") or {}).items()):
                nz = " ".join(f"{k}:{v}" for k, v in bins.items() if v)
                if nz:
                    print(f"    substitution rel-err g{gi}: {nz}")
            # The similarity FLOOR. If p50 here is above the threshold in use, that group merges
            # essentially every pair and its compression factor is degeneracy, not redundancy.
            for gi, q in sorted((c.get("audit_random_pair_cos") or {}).items()):
                thr = (c.get("thresholds_per_group") or {}).get(gi)
                print(f"    random-pair cosine g{gi}: p50={q['p50']:.3f} p90={q['p90']:.3f} "
                      f"p99={q['p99']:.3f} max={q['max']:.3f} (n={q['n']}"
                      + (f", threshold {thr}" if thr else "") + ")")

    if xfer_per:
        # Read this BEFORE any throughput comparison: recomputed prefill runs on the DECODE GPU, so
        # an arm with more failures is handicapped for reasons that have nothing to do with fusion.
        data["kv_transfer_failures"] = xfer_per
        for lg, v in xfer_per.items():
            if v["failed_pulls"] or v["recomputed_requests"]:
                print(f"  kv transfer failures [{lg}]: {v['failed_pulls']} failed pulls | "
                      f"{v['recomputed_tokens']:,} tokens re-prefilled on D "
                      f"({v['recomputed_requests']} requests)"
                      + (f" | {v['producer_send_timeouts']} producer send-timeouts"
                         if v["producer_send_timeouts"] else ""))
            # Sustained only: every instance passes through "full and idle" briefly while draining
            # at the end of a run, and that is not a wedge.
            if _is_wedged(v):
                pct = 100.0 * v["wedged_samples"] / max(1, v["sched_samples"])
                print(f"  ! producer wedged [{lg}]: {v['wedged_samples']} samples "
                      f"({pct:.0f}% of the run) with the KV cache >={WEDGE_PCT:.0f}% full and "
                      "nothing running (completed KV waiting to be pulled; new prefills "
                      "cannot start)")

    if redundancy_per:
        # Read every compression number against this. If the workload repeats almost nothing, a
        # large factor is substitution of interchangeable blocks, not deduplication.
        data["workload_redundancy"] = redundancy_per
        for lg, v in redundancy_per.items():
            h = v["prefix_cache_hit_pct"]
            print(f"  workload redundancy [{lg}]: exact prefix-cache hit rate "
                  f"max={h['max']:.1f}% mean={h['mean']:.1f}%"
                  + ("  — near-zero: there is little real duplication to find here"
                     if h["max"] < 5.0 else ""))

    if sat_per:
        data["kv_saturation"] = sat_per
        for lg, v in sat_per.items():
            r = v["running"]
            print(f"  kv saturation [{lg}]: {v['samples']}/{v['of_samples']} samples at "
                  f">={SATURATED_PCT:.0f}% cache | running mean={r['mean']:.0f} max={r['max']} "
                  f"| waiting mean={v['waiting']['mean']:.0f}")

    if sched_per:
        data["bff_sched"] = sched_per
        for lg, v in sched_per.items():
            fb, us = v["free_blocks"], v["block_usage_pct"]
            print(f"  bff sched [{lg}]: free_blocks min={fb['min']} mean={fb['mean']:.0f} "
                  f"last={fb['last']} / {v['total_blocks']} | block_usage% max={us['max']:.1f} "
                  f"mean={us['mean']:.1f} | running mean={v['running']['mean']:.0f} "
                  f"max={v['running']['max']} | preempt(cum)={v['preempt_cum']}")
    if freed_per:
        data["bff_blocks_freed"] = freed_per
        for lg, v in freed_per.items():
            print(f"  bff blocks freed [{lg}]: net={v['net_blocks_freed']} "
                  f"over {v['merge_events']} merge events")
    if redir_per:
        data["bff_redirects_applied"] = redir_per
        for lg, v in redir_per.items():
            print(f"  bff redirects applied [{lg}]: {v['redirects_applied']} "
                  f"(reps_unresolved={v['reps_unresolved']})")

    # v2_per belongs here: it comes from bff_stats_*.json rather than the logs, so a run whose logs
    # parsed to nothing would PRINT its v2 stats and then drop them on the floor unwritten.
    if (ov_per or cm_per or sched_per or freed_per or redir_per or sat_per
            or redundancy_per or xfer_per or v2_per or any(cfg_per.values())):
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  → merged into {result_file}")

    return data


def summarize(data: dict) -> None:
    """One consolidated screen summary (accuracy + throughput + latency + BFF)."""
    def _lat(d):
        return (f"mean={d['mean']:.1f} med={d['median']:.1f} p99={d['p99']:.1f}"
                if isinstance(d, dict) else "n/a")
    f1 = data.get("mean_f1")
    rps = data.get("request_throughput_rps")
    otps = data.get("output_throughput_toks_s")
    el = data.get("elapsed_s")
    print(f"\n===== SUMMARY [{data.get('label', '')}] =====")
    print(f"  accuracy: F1={f1:.4f}" if isinstance(f1, (int, float)) else "  accuracy: F1=n/a")
    print(f"  throughput: {rps:.2f} req/s" if isinstance(rps, (int, float))
          else "  throughput: n/a", end="")
    print((f" | {otps:.1f} output tok/s" if isinstance(otps, (int, float)) else "")
          + (f" | elapsed {el:.1f}s" if isinstance(el, (int, float)) else ""))
    print(f"  latency ms: TTFT[{_lat(data.get('ttft_ms'))}] "
          f"TPOT[{_lat(data.get('tpot_ms'))}] ITL[{_lat(data.get('itl_ms'))}]")
    xf = data.get("kv_transfer_failures") or {}
    tot_fail = sum(v["failed_pulls"] for v in xf.values())
    tot_tok = sum(v["recomputed_tokens"] for v in xf.values())
    if tot_fail or tot_tok:
        print(f"  kv transfer failures: {tot_fail} failed pulls | {tot_tok:,} tokens re-prefilled "
              "on the decode GPU (throughput lost to transport, not to fusion)")
    if any(_is_wedged(v) for v in xf.values()):
        print("  ! a producer wedged with a full KV cache and nothing running — see the per-log line")
    # Quote the DECODE instance: a wedged prefill also sits at 100% cache with nothing running, and
    # picking arbitrarily would report `running mean=0` as though the decode were idle.
    v = _decode_log(data.get("kv_saturation") or {})
    if v:
        print(f"  at saturation: running mean={v['running']['mean']:.0f} max={v['running']['max']} "
              f"over {v['samples']}/{v['of_samples']} samples with the KV cache "
              f">={SATURATED_PCT:.0f}% full")
    # Headline is what the DECODE instance actually freed. The producer's own factor counts
    # redirects it merely sent, including those naming reps D had already freed, so it can exceed
    # what fusion is even capable of; it is reported beside it as a claim, never as the result.
    # v2 headline: blocks never requested. Unlike v1's claim there is no gap to reconcile — an
    # unrequested block is an untransferred block — so this number cannot be inflated.
    v2 = data.get("bff_v2")
    dc = data.get("bff_decode_compression")
    if v2 and v2.get("inert"):
        print("  ! bff v2 did not run: no signature exchange was ever attempted "
              + " ".join(f"{k}={n}" for k, n in (v2.get("skip_reasons") or {}).items() if n))
    elif v2:
        print(f"  bff v2: {v2['wire_saving_pct']:.1f}% of the wire not requested "
              f"({v2['blocks_not_requested']} of {v2['blocks_planned']} blocks; "
              f"resident={v2['from_resident']} same-pull={v2['from_same_pull']})")
        if v2["aliases_recomputed"]:
            causes = v2.get("failure_causes") or {}
            top = max(causes.items(), key=lambda kv: kv[1])[0] if any(causes.values()) else "?"
            print(f"    ! {v2['aliases_recomputed']} alias(es) could not be applied and forced a "
                  f"local recompute (mostly {top}) — every one of those blocks was never written")
        if v2["signature_phase_failed"]:
            print(f"    ! signature phase unavailable on {v2['signature_phase_failed']} pull(s) "
                  "— those requests were pulled in full")
    elif dc:
        print(f"  bff: compression {dc['avg_factor']:.3f}x smaller "
              f"({dc['blocks_freed_on_decode']} blocks = {dc['pct_of_cache']:.1f}% of cache"
              + (f", ceiling {dc['ceiling_pct_of_layers']:.1f}%"
                 if dc.get("ceiling_pct_of_layers") else "") + ")"
              + (f" | {dc['pct_landed']:.0f}% of redirects landed"
                 if dc.get("pct_landed") is not None else ""))
    elif "bff_compression" in data:
        print(f"  bff: producer CLAIMED {data['bff_compression']['avg_factor']:.3f}x "
              "(no decode-side merge logs to confirm it)")
    if "bff_overhead" in data:
        print(f"  bff overhead: {data['bff_overhead']['producer_avg_group_dedup_ms']:.3f} ms/group"
              + (f" | queue drain {data['bff_overhead']['producer_avg_queue_drain_ms']:.3f} ms"
                 if data["bff_overhead"].get("producer_avg_queue_drain_ms") else ""))
    print("=" * (len(data.get("label", "")) + 18))


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    summarize(collect(argv[1], argv[2], argv[3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
