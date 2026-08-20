"""Unit tests for the post-run collector (`kv_fast_fusion.tools.collect_bff_stats`).

The collector is where a run stops being logs and becomes a claim, so the thing worth pinning is
that it cannot restate a misleading one. Two guards in particular:

* the headline is what the DECODE instance actually freed, never what the producer claimed it merged
  (those differed by 3x on the run this was written for, and the claim exceeded the arithmetic
  ceiling of the configuration by 5x);
* the null-model verdict, which says whether the accepted merges beat "pick the most similar of N
  unrelated blocks" — the distinction between finding duplicates and substituting interchangeable
  blocks, which no compression factor can express.

Pure post-processing: no GPU, no logs, no vLLM.
"""

import json

import pytest

from kv_fast_fusion.tools import collect_bff_stats as cbs

# The measured group-1 similarity floor: unrelated blocks already sit at cosine 0.845, so a redirect
# taking the best of N candidates lands high on merit of N alone.
FLOOR = {"p50": 0.845, "p90": 0.909, "p99": 0.933, "max": 0.942, "n": 3279}
SUBSTITUTION = {"0.75-0.80": 28, "0.80-0.85": 109, "0.85-0.90": 188,
                "0.90-0.95": 1004, "0.95-0.98": 0, "0.98-1.00": 0}
DUPLICATES = {"0.75-0.80": 2, "0.80-0.85": 3, "0.85-0.90": 5,
              "0.90-0.95": 40, "0.95-0.98": 300, "0.98-1.00": 900}


# =====================================================================================
# null_model_verdict
# =====================================================================================
def test_merges_at_the_noise_floor_are_called_substitution():
    v = cbs.null_model_verdict(SUBSTITUTION, FLOOR, pool=18)
    assert v and "NOT duplicate detection" in v
    assert "best-of-18" in v


def test_merges_above_the_noise_floor_are_called_duplicates():
    v = cbs.null_model_verdict(DUPLICATES, FLOOR, pool=18)
    assert v and "real near-duplicates" in v


def test_a_bigger_pool_raises_the_bar_it_has_to_clear():
    """The order statistic is the whole point: growing the candidate pool raises the accepted cosine
    on its own, so the same histogram must not start looking like a better result."""
    small = cbs.null_model_verdict(DUPLICATES, FLOOR, pool=2)
    big = cbs.null_model_verdict(DUPLICATES, FLOOR, pool=1000)
    assert "null ~0.88" in small and "null ~0.94" in big


def test_verdict_is_withheld_without_the_audit():
    assert cbs.null_model_verdict(SUBSTITUTION, {}, pool=18) is None
    assert cbs.null_model_verdict({}, FLOOR, pool=18) is None
    assert cbs.null_model_verdict(SUBSTITUTION, FLOOR, pool=1) is None, "no max over one candidate"


# =====================================================================================
# collect: the headline
# =====================================================================================
def _run(tmp_path, stats, log_lines):
    (tmp_path / "bff_stats_1.json").write_text(json.dumps(stats))
    log = tmp_path / "decode.log"
    log.write_text("\n".join(log_lines))
    result = tmp_path / "r.json"
    result.write_text(json.dumps({"label": "t"}))
    return cbs.collect(str(result), str(tmp_path), [str(log)])


def test_headline_is_what_decode_freed_not_what_the_producer_claimed(tmp_path):
    data = _run(
        tmp_path,
        {"steps": 10, "overhead_avg_group_dedup_ms": 1.0, "total_blocks": 1355, "freed": 1329,
         "compression_avg_factor": 52.1, "compression_per_group": {"1": 52.1},
         "cross_index": "lsh", "ff_groups": [1], "layers_fused": 4, "layers_total": 28,
         "lsh_entries": {"1": 18}},
        ["Block merging freed 100 blocks",
         ("BFF sched | step=1 | running=5 | waiting=0 | free_blocks=900 / 1000 | "
          "block_usage=10.0% | preempt(cum)=0"),
         "BFF P/D apply | redirects_applied=100 | reps_unresolved=300"])

    dc = data["bff_decode_compression"]
    assert dc["blocks_freed_on_decode"] == 100
    assert dc["pct_of_cache"] == 10.0
    assert dc["producer_claimed_blocks"] == 1329, "the claim is kept, but only as a diagnostic"
    assert dc["pct_landed"] == 25.0, "100 of 400 redirects resolved"
    assert dc["ceiling_pct_of_layers"] == pytest.approx(4 / 28 * 100)
    assert dc["avg_factor"] < 1.2, "nothing like the producer's 52x"


def test_saturation_is_measured_from_vllms_own_line(tmp_path):
    """It has to work for a VANILLA run too — the BFF sched line does not exist there, so a
    like-for-like saturation comparison can only come from vLLM's engine log."""
    data = _run(tmp_path, {"steps": 0}, [
        "Running: 10 reqs, Waiting: 400 reqs, GPU KV cache usage: 12.0%",   # ramping
        "Running: 300 reqs, Waiting: 100 reqs, GPU KV cache usage: 99.9%",
        "Running: 280 reqs, Waiting: 0 reqs, GPU KV cache usage: 97.0%",
        "Running: 4 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.0%",       # draining
    ])
    sat = next(iter(data["kv_saturation"].values()))
    assert sat["samples"] == 2 and sat["of_samples"] == 4
    assert sat["running"]["max"] == 300
    assert sat["running"]["mean"] == 290, "ramp and drain excluded, not averaged in"


# =====================================================================================
# error_budget
#
# The threshold reads like a similarity preference and is really an error budget. These pin the
# translation, because it is the number that explained a real F1 regression: at 0.75 the majority of
# accepted merges differed from what they replaced by more than half their own magnitude.
# =====================================================================================
def test_threshold_is_reported_as_the_error_budget_it_is():
    assert "rel_err up to 0.707" in cbs.error_budget(SUBSTITUTION, 0.75), (
        "cos 0.75 permits a 71% relative substitution error")
    assert cbs.error_budget(SUBSTITUTION, 0.98).startswith(
        "threshold 0.98 admits rel_err up to 0.200")


def test_error_budget_reports_how_much_survives_a_strict_bar():
    """Answers 'what would tightening the threshold cost me' without running the sweep."""
    assert "0 of 1329 merges (0.0%) above cos 0.95" in cbs.error_budget(SUBSTITUTION, 0.75), (
        "none of these are near-duplicates, so a strict bar would keep nothing")
    assert "1200 of 1250 merges (96.0%) above cos 0.95" in cbs.error_budget(DUPLICATES, 0.75)


def test_error_budget_without_a_recorded_threshold_still_reports_the_share():
    b = cbs.error_budget(SUBSTITUTION, None)
    assert b and "rel_err" not in b and "above cos 0.95" in b
    assert cbs.error_budget({}, 0.75) is None


# =====================================================================================
# KV transfer failures + the producer-wedge alarm
#
# Under Mooncake's pull model the producer writes into the decode's live KV cache, so at saturation
# transfers start failing and each failure re-prefills that request ON THE DECODE GPU. That is
# throughput taken from the arm that suffers it, for reasons unrelated to fusion — so it has to be
# visible before any BFF-vs-vanilla throughput number is compared.
# =====================================================================================
def _sched(step, running, usage, free=100, total=1000):
    return (f"BFF sched | step={step} | running={running} | waiting=0 | "
            f"free_blocks={free} / {total} | block_usage={usage}% | preempt(cum)=0")


def test_failed_pulls_and_their_recompute_cost_are_counted(tmp_path):
    data = _run(tmp_path, {"steps": 0}, [
        "BFF Mooncake: pull FAILED for chatcmpl-aaa-1 (133 blocks invalid) — releasing it",
        "BFF Mooncake: pull FAILED for chatcmpl-bbb-2 (91 blocks invalid) — releasing it",
        "Recovered from KV load failure: 2 request(s) rescheduled (16788 tokens affected)",
        "Request chatcmpl-ccc-3 timed out after 480 seconds without being sent. Freeing its blocks.",
    ])
    v = next(iter(data["kv_transfer_failures"].values()))
    assert v["failed_pulls"] == 2 and v["invalid_blocks"] == 133 + 91
    assert v["recomputed_requests"] == 2 and v["recomputed_tokens"] == 16788
    assert v["producer_send_timeouts"] == 1


def test_recompute_cost_is_counted_in_a_vanilla_run_too(tmp_path):
    """The scheduler line is vLLM's own, so it appears without the BFF connector. That is what makes
    the two arms comparable — a vanilla run with the same failures must not look clean."""
    data = _run(tmp_path, {"steps": 0},
                ["Recovered from KV load failure: 6 request(s) rescheduled (16788 tokens affected)"])
    v = next(iter(data["kv_transfer_failures"].values()))
    assert v["failed_pulls"] == 0, "no BFF connector line in a vanilla log"
    assert v["recomputed_requests"] == 6 and v["recomputed_tokens"] == 16788


def test_a_sustained_wedge_is_flagged(tmp_path):
    """A producer at 99% cache with nothing running is holding KV decode has not pulled."""
    data = _run(tmp_path, {"steps": 0},
                [_sched(i, 0, 99.2, free=157, total=18617) for i in range(20)])
    v = next(iter(data["kv_transfer_failures"].values()))
    assert v["wedged_samples"] == 20 and cbs._is_wedged(v)


def test_a_brief_full_and_idle_moment_is_not_a_wedge(tmp_path):
    """Every instance passes through full-and-idle while draining; that is not a wedge, and the
    decode in a real run tripped this 4 times in 1000 samples."""
    lines = [_sched(i, 12, 99.9) for i in range(96)] + [_sched(i, 0, 99.9) for i in range(4)]
    data = _run(tmp_path, {"steps": 0}, lines)
    v = next(iter(data["kv_transfer_failures"].values()))
    assert v["wedged_samples"] == 4
    assert not cbs._is_wedged(v), "4 of 100 samples is a transient, not a stall"


def test_summary_quotes_the_decode_not_a_wedged_prefill(tmp_path, capsys):
    """A wedged prefill also sits at 100% cache with nothing running, so picking arbitrarily would
    report `running mean=0` as though the decode were idle."""
    sat = {"prefill2.log": {"samples": 7, "of_samples": 42,
                            "running": {"mean": 0, "max": 0}, "waiting": {"mean": 1}},
           "decode1.log": {"samples": 39, "of_samples": 94,
                           "running": {"mean": 9, "max": 48}, "waiting": {"mean": 138}}}
    cbs.summarize({"label": "t", "kv_saturation": sat})
    assert "running mean=9 max=48" in capsys.readouterr().out


def test_decode_log_falls_back_to_whoever_ran_requests():
    """No 'decode' in the name: the decode is the instance that actually runs requests."""
    per = {"a.log": {"running": {"max": 0}}, "b.log": {"running": {"max": 48}}}
    assert cbs._decode_log(per) is per["b.log"]
    assert cbs._decode_log({}) is None


def _v2_stats(**over):
    """Minimum bff_stats shape the v2 branch needs, plus whatever the case under test varies."""
    base = {"bff_version": 2, "blocks_planned": 100, "blocks_not_requested": 6,
            "blocks_not_requested_resident": 4, "blocks_not_requested_same_pull": 2,
            "wire_saving_pct": 6.0, "aliases_applied": 5, "aliases_recomputed": 1}
    base.update(over)
    return base


def test_a_connector_that_cannot_count_exchanges_is_not_reported_as_zero(tmp_path, capsys):
    """`exchanges=0` claims the mechanism never ran; an ABSENT counter says only that this build
    cannot tell. mooncake_connector_ff_v2_legacy.py is kept verbatim as a measurement baseline so it
    can never gain the counter, and conflating the two made a legacy run that deduplicated 5.9% of
    the wire read as stone dead."""
    data = _run(tmp_path, _v2_stats(), [])
    assert "exchanges" not in data["bff_v2"], "absent must not be persisted as a measured zero"
    assert "inert" not in data["bff_v2"]
    out = capsys.readouterr().out
    assert "exchanges=n/a" in out
    assert "INERT" not in out


def test_a_genuine_zero_with_skips_is_still_called_inert(tmp_path, capsys):
    """The counter's whole purpose: a connector that asked nothing looks identical to one that
    asked and found nothing. Reporting it must survive the fix above."""
    data = _run(tmp_path, _v2_stats(exchanges=0, exchange_skip_reasons={"no_peer": 7}), [])
    assert data["bff_v2"]["inert"] is True
    assert data["bff_v2"]["exchanges"] == 0
    assert "INERT" in capsys.readouterr().out


def test_v2_stats_survive_logs_that_parsed_to_nothing(tmp_path):
    """v2 stats come from bff_stats_*.json, not the logs, so the write must not be gated on the
    logs alone — that combination printed the numbers and then dropped them unwritten."""
    data = _run(tmp_path, _v2_stats(exchanges=3), [])
    assert data["bff_v2"]["exchanges"] == 3
    assert json.loads((tmp_path / "r.json").read_text())["bff_v2"]["blocks_planned"] == 100
