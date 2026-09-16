"""The per-step profiler (Update 28): does it produce the ONE number it exists for?

BFF lost 75.5 s of 1286 s at 4096 while every connector counter summed to 16.7 s, so ~59 s lives in
work no BFF code performs — the runner doing per-request block-table and slot-mapping work once per
KV-cache group, seven times where the baseline does it once. The profiler's job is to turn that into
`(calls_per_step - 1) x ms_per_call`, which is the excess over a single-group runner WITHOUT needing an
instrumented baseline run. These tests pin that arithmetic, and pin that the profiler is inert when it
is not asked for.

CPU only: the accumulator is plain arithmetic and the installer is monkeypatching, so neither needs an
NPU, vllm_ascend, or a model.
"""

import kv_fast_fusion_ascend.step_profile as sp


def _profile(groups=7, steps=100, ms_per_call=0.26):
    """`steps` steps of a `groups`-group runner, each group costing `ms_per_call` in every phase."""
    p = sp.StepProfile()
    for _ in range(steps):
        p.last_num_reqs = 200
        for phase in sp.PER_GROUP_PHASES:
            for _g in range(groups):
                p.note(phase, ms_per_call)
        p.note("step_total", 80.0)
        p.note_step()
    return p


def test_calls_per_step_recovers_the_group_count_without_being_told_it():
    """The whole reason the INNER per-group methods are wrapped rather than MultiGroupBlockTable's:
    the group multiplier falls out of the call count, so the report is right whatever BFF_GROUP_SIZE
    or BFF_BALANCED_GROUPS is set to."""
    p = _profile(groups=7)
    assert p.calls_per_step("bt_commit") == 7.0
    assert round(p.per_call("bt_commit"), 6) == 0.26
    assert round(p.per_step("bt_commit"), 6) == round(7 * 0.26, 6)


def test_group_excess_is_the_cost_over_a_SINGLE_group_runner():
    """ms/call is one group's cost — what the one-group baseline also pays — so the excess is the
    other six groups, across all three per-group phases. 7 groups x 3 phases x 0.26 ms, minus the one
    group of each phase the baseline would pay anyway."""
    p = _profile(groups=7, ms_per_call=0.26)
    assert round(p.group_excess_ms(), 4) == round(3 * 6 * 0.26, 4)
    # A single-group runner has nothing extra to pay, by construction.
    assert _profile(groups=1).group_excess_ms() == 0.0
    # And the excess is linear in the group count: 2 balanced groups cost a seventh of 7's excess.
    assert round(_profile(groups=2).group_excess_ms(), 4) == round(3 * 1 * 0.26, 4)


def test_a_fresh_profile_divides_by_zero_steps_without_exploding():
    """The first report fires on step 1, and a render on an idle engine must not raise — a profiler
    that can throw inside execute_model is worse than no profiler."""
    p = sp.StepProfile()
    assert p.per_step("bt_commit") == 0.0 and p.per_call("bt_commit") == 0.0
    assert p.calls_per_step("bt_commit") == 0.0 and p.mean_reqs() == 0.0
    assert p.group_excess_ms() == 0.0
    assert "steps=0" in p.render()          # renders, does not divide by zero
    assert p.snapshot()["steps"] == 0


def test_the_report_cadence_widens_so_a_long_run_does_not_drown_the_log():
    p = sp.StepProfile()
    due = [i for i in range(1, 1001) if p.note_step()]
    assert due == [1, 10, 100, 1000], "1, 10, 100, 1000 — the connector's own tally cadence"


def test_mean_num_reqs_is_carried_because_the_cost_scales_with_concurrency():
    """The residual is ~0 at Running ~115 and ~8-11 % at ~200, so a profile line without the request
    count cannot be placed on that curve and says nothing."""
    p = _profile(steps=10)
    assert p.mean_reqs() == 200.0
    assert p.snapshot()["mean_num_reqs"] == 200.0
    assert "mean_reqs=200.0" in p.render()


def test_the_headline_number_and_its_share_of_the_step_are_both_in_the_line():
    p = _profile(groups=7, ms_per_call=0.26)          # excess 4.68 ms of an 80 ms step
    line = p.render()
    assert "GROUP EXCESS" in line and "4.68 ms/step" in line
    assert "5.8% of the step" in line, "a ms figure with no share of the step is not actionable"


def test_install_is_a_complete_no_op_when_the_flag_is_off(monkeypatch):
    """A diagnostic must cost a normal run nothing at all — not a wrapper, not an attribute."""
    class _Fake:
        def commit_block_table(self, num_reqs):
            return num_reqs

    before = _Fake.commit_block_table
    monkeypatch.setattr(sp, "STEP_PROFILE", False)
    sp.install()
    assert _Fake.commit_block_table is before


def test_wrap_times_the_call_forwards_it_and_refuses_to_nest_on_a_second_install():
    """Double-wrapping is the failure that would silently double every number in the report; these
    installers run from module import, and an import can happen twice."""
    calls = []

    class _Fake:
        def commit_block_table(self, num_reqs):
            calls.append(num_reqs)
            return "kv"

    sp.PROFILE.__init__()          # a clean accumulator for this test
    assert sp._wrap(_Fake, "commit_block_table", "bt_commit", on_call=sp._note_num_reqs) is True
    assert sp._wrap(_Fake, "commit_block_table", "bt_commit") is False, "wrap-once"

    assert _Fake().commit_block_table(37) == "kv", "the return value must pass through"
    assert calls == [37], "and the original must actually run, once"
    assert sp.PROFILE.phases["bt_commit"][0] == 1
    assert sp.PROFILE.last_num_reqs == 37, "num_reqs is read off the call, not out of the runner"


def test_a_raising_target_is_still_timed_and_still_raises():
    """The timer is in a finally, so a failing step is accounted rather than losing the sample — and
    the profiler must not swallow the error it was standing next to."""
    import pytest

    class _Fake:
        def boom(self):
            raise RuntimeError("step failed")

    sp.PROFILE.__init__()
    sp._wrap(_Fake, "boom", "bt_commit")
    with pytest.raises(RuntimeError, match="step failed"):
        _Fake().boom()
    assert sp.PROFILE.phases["bt_commit"][0] == 1


# =====================================================================================
# mem_probe (Update 33, relocated by Update 34). vLLM reports `Available KV cache memory` 34.66 GiB
# for the baseline against 33.78 GiB for BFF — a flat 0.88 GiB, IDENTICAL at 2, 4 and 7 groups AND
# with BFF_PD_FUSE=0, so it is not the group split and not this patch. Being fixed, it costs 2.5 % of
# KV at GPU_MEM_UTIL=1.0 and 9.8 % at 0.5, which is what flipped the memory-bound experiment. These
# pin that the probe is inert by default, cannot throw during init, and — the Update-34 lesson —
# always SAYS which of those it is doing.
# =====================================================================================
def _fake_npu(allocated=2, reserved=3, free=None, total=None):
    import types
    gib = 1024 ** 3
    ns = types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: allocated * gib,
        memory_reserved=lambda: reserved * gib,
    )
    if free is not None:
        ns.mem_get_info = lambda: (free * gib, total * gib)
    return ns


def test_mem_probe_is_inert_when_the_flag_is_off(monkeypatch):
    monkeypatch.setattr(sp, "MEM_PROBE", False)
    sp._ANNOUNCED.clear()
    assert sp.mem_probe("patch:before") is None


def test_mem_probe_never_raises_without_an_npu(monkeypatch):
    """It brackets engine init on a box that may not have torch.npu at all; a probe that raises
    there would take down serving for a diagnostic."""
    monkeypatch.setattr(sp, "MEM_PROBE", True)
    sp._ANNOUNCED.clear()
    assert sp.mem_probe("patch:before") is None      # no torch.npu on this box -> None, no raise


def test_mem_probe_reports_reserved_as_well_as_allocated(monkeypatch):
    """determine_available_memory is charged for RESERVED memory — the caching allocator returns
    blocks to its pool, not to the driver — so a probe watching only `allocated` would miss exactly
    the overhead being hunted."""
    import torch
    fake = _fake_npu()
    monkeypatch.setattr(sp, "MEM_PROBE", True)
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    sp._MEM_LAST.clear()
    sp._ANNOUNCED.clear()
    first = sp.mem_probe("patch:before")
    assert first == {"allocated": 2.0, "reserved": 3.0}, "no mem_get_info on this fake -> no free/total"
    # A second probe must report a delta against the first, not restate absolutes.
    fake.memory_allocated = lambda: 3 * 1024 ** 3
    second = sp.mem_probe("patch:after")
    assert second["allocated"] == 3.0 and sp._MEM_LAST["last"] == second


def test_mem_probe_reports_device_wide_free_because_non_torch_memory_is_the_suspect(monkeypatch):
    """vLLM's `non_torch_increase` — one of the three terms the 0.88 GiB must land in — is computed
    from mem_get_info, so it counts memory held outside torch, including by ANOTHER PROCESS on the
    device. A torch-only reading is blind to exactly that term."""
    import torch
    monkeypatch.setattr(sp, "MEM_PROBE", True)
    monkeypatch.setattr(torch, "npu", _fake_npu(free=10, total=64), raising=False)
    sp._MEM_LAST.clear()
    sp._ANNOUNCED.clear()
    got = sp.mem_probe("worker:before-profile")
    assert got["free"] == 10.0 and got["total"] == 64.0


# --- the Update-34 lesson: a diagnostic that cannot say whether it ran is not a diagnostic ---------
def test_announce_once_says_it_once_per_key(monkeypatch):
    sp._ANNOUNCED.clear()
    assert sp.announce_once("k", "info", "hello") is True
    assert sp.announce_once("k", "info", "hello") is False, "once, or it is just log spam"
    assert sp.announce_once("other", "info", "hello") is True, "keys are independent"


def test_an_unset_flag_and_a_working_probe_never_look_the_same(monkeypatch):
    """Four box runs were lost to this: BFF_MEM_PROBE/BFF_STEP_PROFILE not reaching the engine
    produced a log identical to a probe that ran fine — empty. The OFF state must be stated."""
    monkeypatch.setattr(sp, "MEM_PROBE", False)
    sp._ANNOUNCED.clear()
    sp.mem_probe("patch:before")
    assert "mem_probe:off" in sp._ANNOUNCED
    sp.mem_probe("patch:after")
    assert list(sp._ANNOUNCED) == ["mem_probe:off"], "said once, not once per probe point"

    monkeypatch.setattr(sp, "STEP_PROFILE", False)
    sp._ANNOUNCED.clear()
    sp.install()
    assert "step_profile:off" in sp._ANNOUNCED


def test_unavailable_early_does_not_silence_the_later_ON(monkeypatch):
    """patch:before runs before the device exists, so the first probe legitimately reports nothing.
    That must not consume the announcement the working probes owe us."""
    import torch
    monkeypatch.setattr(sp, "MEM_PROBE", True)
    monkeypatch.delattr(torch, "npu", raising=False)
    sp._MEM_LAST.clear()
    sp._ANNOUNCED.clear()
    sp.mem_probe("patch:before")
    assert "mem_probe:unavailable" in sp._ANNOUNCED and "mem_probe:on" not in sp._ANNOUNCED

    monkeypatch.setattr(torch, "npu", _fake_npu(), raising=False)
    sp.mem_probe("worker:before-profile")
    assert "mem_probe:on" in sp._ANNOUNCED


# --- install_mem_probe: bracket the profiling, NOT the connector -----------------------------------
def _fake_worker_module(monkeypatch, calls):
    """A stand-in for vllm_ascend.worker.worker, which does not exist off the Ascend stack."""
    import sys
    import types

    class NPUWorker:
        def determine_available_memory(self):
            calls.append("profiled")
            return 8_650_000_000

    mod = types.ModuleType("vllm_ascend.worker.worker")
    mod.NPUWorker = NPUWorker
    pkg_w = sys.modules.get("vllm_ascend.worker") or types.ModuleType("vllm_ascend.worker")
    pkg = sys.modules.get("vllm_ascend") or types.ModuleType("vllm_ascend")
    monkeypatch.setitem(sys.modules, "vllm_ascend", pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", pkg_w)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", mod)
    return NPUWorker


def test_install_mem_probe_is_a_complete_no_op_when_the_flag_is_off(monkeypatch):
    calls: list = []
    W = _fake_worker_module(monkeypatch, calls)
    before = W.determine_available_memory
    monkeypatch.setattr(sp, "MEM_PROBE", False)
    sp.install_mem_probe()
    assert W.determine_available_memory is before


def test_install_mem_probe_brackets_profiling_and_wraps_once(monkeypatch):
    """The connector is built in initialize_from_config, AFTER determine_available_memory sizes the
    pool (vllm_ascend/worker/worker.py:516 vs :327) — so this call, not the connector's __init__, is
    the only window that can hold the fixed overhead."""
    calls: list = []
    W = _fake_worker_module(monkeypatch, calls)
    monkeypatch.setattr(sp, "MEM_PROBE", True)
    monkeypatch.setattr(sp, "mem_probe", lambda label: calls.append(label))

    sp.install_mem_probe()
    wrapped = W.determine_available_memory
    sp.install_mem_probe()
    assert W.determine_available_memory is wrapped, "wrap-once, or every probe point doubles"

    assert W().determine_available_memory() == 8_650_000_000, "the return value must pass through"
    assert calls == ["worker:before-profile", "profiled", "worker:after-profile"]
