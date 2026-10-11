"""Attribute the per-step cost BFF's group split imposes on the RUNNER, not on BFF's own code.

Why this exists (Update 28). At ``FIXED_OUTPUT_LEN=4096`` BFF lost 75.5 s of 1286 s to the baseline,
and **every BFF counter in the run sums to ~16.7 s** (``plan`` 15.0, ``apply`` 0.56, ``hook`` 0.71,
``sig_decode`` 0.40). ~59 s is somewhere no connector counter can see. Its shape is the clue: the
per-step penalty is ~0 at Running ~115 and ~8-11 % at Running ~200-230 — i.e. it scales with
**requests x groups**, and vLLM-ascend does exactly that work three times per step over BFF's 7 groups
where the baseline has 1:

* ``BlockTable.commit_block_table``  — an H2D copy of ``num_reqs x max_blocks_per_req x int32``,
* ``BlockTable.compute_slot_mapping`` — a numpy gather over every scheduled token,
* ``BlockTable.commit_slot_mapping``  — the slot-mapping H2D,

each driven once per group by ``MultiGroupBlockTable``, plus ``_build_attention_metadata``'s per-group
loop (a copy + a builder call per group).

**The attribution trick, which needs no instrumented baseline run:** these are the INNER, per-group
methods, so ``calls/step`` recovers the group multiplier on its own and ``ms/call`` is the cost of a
SINGLE group — which is what the one-group baseline pays. BFF's excess is therefore
``(calls_per_step - 1) x ms_per_call``, computed here as ``group_excess_ms``. If that lands near the
3-8 ms/step the logs imply at Running ~200, the group split IS the residual and the layout is worth
changing; if it is well under 1 ms, the cost is in the kernel rather than in Python and the chase
stops. Either way it is a decisive fork, which is the only reason to add a profiler at all.

Default OFF (``BFF_STEP_PROFILE=1`` to install). When off, nothing is wrapped and no attribute on any
vLLM class is touched — so a normal run cannot pay for a diagnostic it is not using. Pure Python and
import-light: no torch, no device, no vllm at module scope, so the accumulator is unit-testable on CPU.
"""

from __future__ import annotations

import os
import time

from vllm.logger import init_logger

# Named under ``vllm.`` on purpose: vLLM's logging config attaches a handler to that namespace only, so
# ``init_logger(__name__)`` (``kv_fast_fusion_ascend.step_profile``) had every line of this module
# discarded — the profiler timed correctly and printed nothing, which read as "inert" for five runs.
logger = init_logger("vllm.bff_step_profile")

STEP_PROFILE = os.environ.get("BFF_STEP_PROFILE", "0") == "1"

# Attribute the FIXED device memory BFF costs before the KV pool is sized (Update 33). vLLM's own
# line reports `Available KV cache memory` 34.66 GiB for the baseline against 33.78 GiB for BFF — a
# flat **0.88 GiB**, IDENTICAL at 2, 4 and 7 groups, so it is not the group split, the block tables or
# the per-group attention metadata (all of which scale with group count). It comes off the KV pool
# because it is allocated before `determine_available_memory` runs, and being fixed it hurts in
# proportion to how small the pool is: −2.5 % of KV at GPU_MEM_UTIL=1.0 and **−9.8 % at 0.5**, which
# is why halving memory flipped the memory-bound experiment against BFF.
MEM_PROBE = os.environ.get("BFF_MEM_PROBE", "0") == "1"
_MEM_LAST: dict = {}

# Which "this diagnostic is/is not running" lines have already been said. Four box runs in this
# project have been lost to a diagnostic that engaged silently and failed silently: an env var that
# never reached the engine, a module that was never committed, and a probe that ran before the device
# existed all produce the SAME empty log as success does. So every diagnostic here states its state
# once, unconditionally, and the states are keyed separately — a probe that is unavailable early and
# working later must be able to say both.
_ANNOUNCED: dict = {}

# Marks a wrapped function so a second install() cannot nest timers and double-count. Same idiom (and
# the same reason) as fast_fusion_ascend_patch's _WRAP_SENTINEL: these installers run from module
# import, and an import can happen twice.
_WRAP_SENTINEL = "_bff_step_profiled"

# The phases whose per-group cost is what the group split multiplies. Kept as data so the report and
# the excess arithmetic cannot drift apart from what is actually wrapped.
PER_GROUP_PHASES = ("bt_commit", "slot_map", "slot_commit")

# The scheduler half of the step and what is inside it. Reported with their call counts, because a
# cost that is paid once per running request reads very differently from one paid once per step.
SCHED_PHASES = ("sched", "sched_update", "alloc", "computed", "free", "conn_meta")

# After the engagement proof (1, 10, 100, 1000) report on a fixed stride, so a 15k-step run yields
# several comparable windows instead of one.
REPORT_STRIDE = 2000


class StepProfile:
    """Calls and milliseconds per phase, plus the step count they are divided by.

    Counters are plain ints/floats mutated with ``+=``; they are only ever touched from the forward
    thread, and even a stray read from elsewhere costs at worst a slightly stale number in a log line.
    No lock, because a profiler that contends with the thing it measures is measuring itself."""

    def __init__(self) -> None:
        self.steps = 0
        self.reqs_total = 0          # running sum of num_reqs, for the mean
        self.last_num_reqs = 0
        self.phases: dict[str, list] = {}     # name -> [calls, ms]
        self._next_report = 1
        self._mark = (0, 0, {})      # (steps, reqs_total, phases) as of the last render()

    # -- recording -----------------------------------------------------------------------
    def note(self, phase: str, ms: float) -> None:
        slot = self.phases.get(phase)
        if slot is None:
            self.phases[phase] = [1, ms]
        else:
            slot[0] += 1
            slot[1] += ms

    def note_step(self) -> bool:
        """Close one decode step. Returns True when this step should be reported.

        The cadence widens (1, 10, 100, 1000) so the first step proves the profiler engaged, then
        holds at every ``REPORT_STRIDE`` steps so a long run yields several windows to compare."""
        self.steps += 1
        self.reqs_total += self.last_num_reqs
        if self.steps < self._next_report:
            return False
        if self._next_report < 1000:
            self._next_report *= 10
        else:
            self._next_report += REPORT_STRIDE
        return True

    def window(self) -> "StepProfile":
        """The steps since the last ``render()``, as a profile of their own.

        A cumulative mean is diluted by the idle steps before load arrives (the first 1,000 steps of
        a run average ~2 requests), so each report describes only its own stretch."""
        steps0, reqs0, phases0 = self._mark
        w = StepProfile()
        w.steps = self.steps - steps0
        w.reqs_total = self.reqs_total - reqs0
        for name, (calls, ms) in self.phases.items():
            c0, m0 = phases0.get(name, (0, 0.0))
            w.phases[name] = [calls - c0, ms - m0]
        return w

    # -- reading -------------------------------------------------------------------------
    def per_step(self, phase: str) -> float:
        return self.phases.get(phase, (0, 0.0))[1] / self.steps if self.steps else 0.0

    def calls_per_step(self, phase: str) -> float:
        return self.phases.get(phase, (0, 0.0))[0] / self.steps if self.steps else 0.0

    def per_call(self, phase: str) -> float:
        calls, ms = self.phases.get(phase, (0, 0.0))
        return ms / calls if calls else 0.0

    def mean_reqs(self) -> float:
        return self.reqs_total / self.steps if self.steps else 0.0

    def group_excess_ms(self) -> float:
        """What BFF pays per step that a single-group runner would not.

        ``per_call`` is one group's cost — the baseline's whole cost for that phase — so the excess is
        the other ``calls_per_step - 1`` groups. This is the number the whole module exists to print."""
        return sum(max(0.0, self.calls_per_step(p) - 1.0) * self.per_call(p)
                   for p in PER_GROUP_PHASES)

    def snapshot(self) -> dict:
        out = {"steps": self.steps, "mean_num_reqs": round(self.mean_reqs(), 1),
               "group_excess_ms_per_step": round(self.group_excess_ms(), 3)}
        for phase in self.phases:
            out[f"{phase}_ms_per_step"] = round(self.per_step(phase), 3)
            out[f"{phase}_ms_per_call"] = round(self.per_call(phase), 4)
            out[f"{phase}_calls_per_step"] = round(self.calls_per_step(phase), 2)
        return out

    def render(self) -> str:
        """One line for the window since the previous ``render()``; advances the window."""
        w = self.window()
        self._mark = (self.steps, self.reqs_total, {k: tuple(v) for k, v in self.phases.items()})
        total = w.per_step("step_total")
        pct = (100.0 * w.group_excess_ms() / total) if total else 0.0
        parts = [(f"BFF step profile | steps={self.steps} window={w.steps} "
                  f"mean_reqs={w.mean_reqs():.1f} | engine_step {w.per_step('engine_step'):.2f} ms | "
                  f"step_total {total:.2f} ms sample {w.per_step('sample'):.2f} | prepare_inputs {w.per_step('prepare_inputs'):.2f} "
                  f"attn_meta {w.per_step('attn_meta'):.2f}")]
        for phase in SCHED_PHASES:
            parts.append(f"{phase} {w.per_step(phase):.2f} ms/step over "
                         f"{w.calls_per_step(phase):.1f} calls")
        for phase in PER_GROUP_PHASES:
            parts.append(f"{phase} {w.per_step(phase):.2f} ms/step over "
                         f"{w.calls_per_step(phase):.1f} calls ({w.per_call(phase):.3f} ms/call)")
        parts.append(f"GROUP EXCESS (calls-1)xms/call = {w.group_excess_ms():.2f} ms/step "
                     f"({pct:.1f}% of the step)")
        return " | ".join(parts)


PROFILE = StepProfile()


# Function-level attribution for the scheduler (2026-10-11). At matched load BFF's schedule() costs
# 4.96 ms/step against the baseline's 0.62, and update_from_output 2.03 against 0.36, with nothing in
# BFF's scheduler-side code that obviously explains it. Rather than guess phase by phase, profile the
# two calls with cProfile for a window of loaded steps and print the table once. Off unless
# BFF_SCHED_CPROFILE=1 (and BFF_STEP_PROFILE=1, which installs the wrappers this rides on).
SCHED_CPROFILE = os.environ.get("BFF_SCHED_CPROFILE", "0") == "1"
SCHED_CPROFILE_START = int(os.environ.get("BFF_SCHED_CPROFILE_START", "4000"))
SCHED_CPROFILE_STEPS = int(os.environ.get("BFF_SCHED_CPROFILE_STEPS", "2000"))


class SchedCProfile:
    """cProfile over the wrapped scheduler calls, for steps in ``[start, start + steps)``.

    cProfile inflates what it measures, so the window is short and its throughput is not to be read;
    the table's value is the ranking, and ``cumtime / steps`` as a per-step upper bound."""

    def __init__(self, start: int, steps: int, label: str = "sched") -> None:
        self.label = label
        self.start = start
        self.end = start + steps
        self.calls = 0
        self.reported = False
        self._active = False
        self._prof = None

    def maybe_enable(self, step: int) -> bool:
        """Start profiling this call if ``step`` is in the window. True when it did."""
        if self._active or self.reported or not (self.start <= step < self.end):
            return False
        if self._prof is None:
            import cProfile
            self._prof = cProfile.Profile()
        self._active = True
        self.calls += 1
        self._prof.enable()
        return True

    def disable(self) -> None:
        self._prof.disable()
        self._active = False

    def due(self, step: int) -> bool:
        return not self.reported and self._prof is not None and step >= self.end

    def report(self) -> str:
        """The table, once: by cumulative time (who owns the cost), then by self time (who burns it)."""
        import io
        import pstats
        self.reported = True
        out = io.StringIO()
        what = "scheduler" if self.label == "sched" else self.label
        out.write(f"BFF {self.label} cprofile | steps {self.start}-{self.end} | {self.calls} {what} "
                  f"calls profiled | per-step cost of a row = cumtime / {self.end - self.start}\n")
        stats = pstats.Stats(self._prof, stream=out).strip_dirs()
        stats.sort_stats("cumulative").print_stats(40)
        stats.sort_stats("tottime").print_stats(25)
        return out.getvalue()


SCHED_CPROF = SchedCProfile(SCHED_CPROFILE_START, SCHED_CPROFILE_STEPS)

# The same window over the RUNNER (execute_model + sample_tokens). With dedup off the two-group layout
# is still ~2 ms/step slower than one group and most of it sits inside execute_model, where the phase
# timers cannot tell host work from waiting on the device. In the table, host work shows as Python
# rows and device waits as time in a builtin (synchronize / .cpu() / tolist), so the two arms can be
# compared row by row. Its own Profile object: the runner may not share a thread with the scheduler.
RUNNER_CPROFILE = os.environ.get("BFF_RUNNER_CPROFILE", "0") == "1"
RUNNER_CPROF = SchedCProfile(SCHED_CPROFILE_START, SCHED_CPROFILE_STEPS, label="runner")


def announce_once(key: str, level: str, msg: str, *args) -> bool:
    """Say a diagnostic's state exactly once. Returns True when it printed.

    The point is that an unset flag and a working probe must never look the same in a log. See
    ``_ANNOUNCED``."""
    if _ANNOUNCED.get(key):
        return False
    _ANNOUNCED[key] = True
    getattr(logger, level)(msg, *args)
    return True


def mem_probe(label: str) -> dict | None:
    """Log device memory at ``label`` and the delta since the previous probe, or do nothing.

    Reports allocated, reserved AND the device-wide free/total. Allocated alone is not enough: the
    caching allocator hands memory back to its pool but not to the driver, and
    ``determine_available_memory`` is charged for what is RESERVED. Free/total is not optional either
    — vLLM's ``non_torch_increase`` is computed from ``mem_get_info``, so it counts memory held
    OUTSIDE torch, including by another process on the same device, and that is the only term a
    torch-only reading cannot see.

    Returns the reading (for tests) or None when the flag is off or there is no NPU. Every failure is
    swallowed: a memory probe that can raise during engine init is worse than no probe."""
    if not MEM_PROBE:
        announce_once("mem_probe:off", "info",
                      "BFF mem probe: OFF (BFF_MEM_PROBE unset) — device-memory attribution disabled.")
        return None
    try:
        import torch
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            announce_once("mem_probe:unavailable", "warning",
                          "BFF mem probe: UNAVAILABLE at %s — torch.npu is absent or the device is "
                          "not initialised yet, so this probe point reports nothing.", label)
            return None
        announce_once("mem_probe:on", "info", "BFF mem probe: ON (BFF_MEM_PROBE=1).")
        gib = 1024 ** 3
        cur = {"allocated": npu.memory_allocated() / gib, "reserved": npu.memory_reserved() / gib}
        # Device-wide, so it catches non-torch and other-process memory. Not every torch_npu build
        # exposes it; its absence must cost the two fields, never the probe.
        try:
            free, total = npu.mem_get_info()
            cur["free"] = free / gib
            cur["total"] = total / gib
        except Exception:  # noqa: BLE001, S110 - optional; its absence costs two fields, not the probe
            pass
        prev = _MEM_LAST.get("last")
        tail = ""
        if "free" in cur:
            tail = f" | device free={cur['free']:.3f}/{cur['total']:.3f} GiB"
            if prev is not None and "free" in prev:
                tail += f" ({cur['free'] - prev['free']:+.3f})"
        if prev is None:
            logger.info("BFF mem probe | %-26s allocated=%.3f GiB reserved=%.3f GiB%s",
                        label, cur["allocated"], cur["reserved"], tail)
        else:
            logger.info("BFF mem probe | %-26s allocated=%.3f GiB (%+.3f) reserved=%.3f GiB (%+.3f)%s",
                        label, cur["allocated"], cur["allocated"] - prev["allocated"],
                        cur["reserved"], cur["reserved"] - prev["reserved"], tail)
        _MEM_LAST["last"] = cur
        return cur
    except Exception as e:  # noqa: BLE001 - a diagnostic must never break engine init
        logger.warning("BFF mem probe (%s) unavailable: %s", label, e)
        return None


def install_mem_probe() -> None:
    """Bracket ``determine_available_memory`` — the only window that can hold the fixed overhead.

    Update 34 settled the ordering from source (``vllm_ascend/worker/worker.py``): ``init_device``
    (:309), ``load_model`` (:429), **``determine_available_memory``** (:327), and only THEN
    ``initialize_from_config`` (:514) → ``ensure_kv_transfer_initialized`` (:516). The KV connector is
    built after the pool is sized, so it cannot be the 0.88 GiB and bracketing its ``__init__`` (as
    Update 33 did) measures something downstream of the number. The profile run is where the arms
    diverge, so that is what gets bracketed — and it is installed above the ``BFF_PD_FUSE`` gate so it
    reaches the baseline arm too, because the comparison is the whole point."""
    if not MEM_PROBE:
        return
    try:
        from vllm_ascend.worker.worker import NPUWorker
    except Exception as e:  # noqa: BLE001 - a diagnostic must never break serving
        logger.warning("BFF mem probe: could not reach NPUWorker (%s); memory profiling stays "
                       "unbracketed.", e)
        return
    orig = getattr(NPUWorker, "determine_available_memory", None)
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return

    def _probed(*args, _orig=orig, **kwargs):
        mem_probe("worker:before-profile")
        try:
            return _orig(*args, **kwargs)
        finally:
            mem_probe("worker:after-profile")

    setattr(_probed, _WRAP_SENTINEL, True)
    NPUWorker.determine_available_memory = _probed
    logger.info("BFF mem probe: bracketing NPUWorker.determine_available_memory.")


def _wrap(cls, name: str, phase: str, on_call=None, cprofile=None) -> bool:
    """Time ``cls.name`` into ``phase``. Returns True if it wrapped something new.

    ``on_call`` gets the call's args before the inner function runs, which is how ``num_reqs`` is
    captured — it arrives as ``commit_block_table``'s own argument, so nothing has to reach into the
    runner's state to find it."""
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return False

    if cprofile is True:          # the scheduler window, by its original spelling
        cprofile = "sched"

    def _timed(*args, _orig=orig, _phase=phase, _on=on_call, _which=cprofile, **kwargs):
        if _on is not None:
            _on(*args, **kwargs)
        # Looked up per call, not bound at wrap time, so a test (or a reload) can swap the window.
        cprof = SCHED_CPROF if _which == "sched" else RUNNER_CPROF if _which == "runner" else None
        profiling = cprof is not None and cprof.maybe_enable(PROFILE.steps)
        t0 = time.perf_counter()
        try:
            return _orig(*args, **kwargs)
        finally:
            PROFILE.note(_phase, (time.perf_counter() - t0) * 1e3)
            if profiling:
                cprof.disable()
            elif cprof is not None and cprof.due(PROFILE.steps):
                logger.info("%s", cprof.report())

    setattr(_timed, _WRAP_SENTINEL, True)
    setattr(cls, name, _timed)
    return True


def _note_num_reqs(_self, num_reqs, *_a, **_k) -> None:
    PROFILE.last_num_reqs = int(num_reqs)


_CONNECTOR_WRAPPED = [False]


def _note_connector(scheduler, *_a, **_k) -> None:
    """Time the live connector's ``build_connector_meta`` as ``conn_meta``.

    Done from inside the first ``schedule()`` because the connector class is only known once the
    scheduler has built it. One attempt, whatever the outcome."""
    if _CONNECTOR_WRAPPED[0]:
        return
    _CONNECTOR_WRAPPED[0] = True
    try:
        connector = getattr(scheduler, "connector", None)
        if connector is not None and _wrap(type(connector), "build_connector_meta", "conn_meta"):
            logger.info("BFF step profile: timing %s.build_connector_meta as conn_meta.",
                        type(connector).__name__)
    except Exception as e:  # noqa: BLE001 - a diagnostic must never break scheduling
        logger.warning("BFF step profile: could not wrap the connector (%s).", e)


def _wrap_step(cls, name: str) -> bool:
    """``execute_model`` — the denominator. Times the whole step AND closes it, so every other phase
    is reported per step of the same run rather than against a step count guessed elsewhere."""
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return False

    def _timed(*args, _orig=orig, **kwargs):
        cprof = RUNNER_CPROF if RUNNER_CPROFILE else None
        profiling = cprof is not None and cprof.maybe_enable(PROFILE.steps)
        t0 = time.perf_counter()
        try:
            return _orig(*args, **kwargs)
        finally:
            PROFILE.note("step_total", (time.perf_counter() - t0) * 1e3)
            if profiling:
                cprof.disable()
            elif cprof is not None and cprof.due(PROFILE.steps):
                logger.info("%s", cprof.report())
            if PROFILE.note_step():
                logger.info("%s", PROFILE.render())

    setattr(_timed, _WRAP_SENTINEL, True)
    setattr(cls, name, _timed)
    return True


def install() -> None:
    """Wrap the runner and scheduler call sites, or do nothing at all when the flag is off.

    Every target is wrapped independently inside its own try/except: this is a diagnostic, and a
    vLLM-ascend version that renamed one method must cost that one phase, never the run."""
    if not STEP_PROFILE:
        announce_once("step_profile:off", "info",
                      "BFF step profile: OFF (BFF_STEP_PROFILE unset) — nothing wrapped.")
        return
    wrapped = []
    try:
        from vllm_ascend.worker.block_table import BlockTable
        # The INNER, per-group methods on purpose — see the module docstring: calls/step then recovers
        # the group count and ms/call is the single-group cost the baseline also pays.
        if _wrap(BlockTable, "commit_block_table", "bt_commit", on_call=_note_num_reqs):
            wrapped.append("bt_commit")
        if _wrap(BlockTable, "compute_slot_mapping", "slot_map"):
            wrapped.append("slot_map")
        if _wrap(BlockTable, "commit_slot_mapping", "slot_commit"):
            wrapped.append("slot_commit")
    except Exception as e:  # noqa: BLE001 - a diagnostic must never break serving
        logger.warning("BFF step profile: could not wrap the block table (%s).", e)
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        if _wrap(NPUModelRunner, "_prepare_inputs", "prepare_inputs"):
            wrapped.append("prepare_inputs")
        if _wrap(NPUModelRunner, "_build_attention_metadata", "attn_meta"):
            wrapped.append("attn_meta")
        if _wrap_step(NPUModelRunner, "execute_model"):
            wrapped.append("step_total")
        # With async scheduling the device wait can land here instead of in execute_model, so the
        # runner's share of the step is only comparable across arms as step_total + sample.
        if _wrap(NPUModelRunner, "sample_tokens", "sample",
                 cprofile="runner" if RUNNER_CPROFILE else None):
            wrapped.append("sample")
    except Exception as e:  # noqa: BLE001 - as above
        logger.warning("BFF step profile: could not wrap the model runner (%s).", e)
    # The scheduler half of the step. Block allocation runs once per KV-cache group for every running
    # request, so it is the other place requests x groups work can hide, and nothing above sees it.
    # Only where the class DEFINES the method (same rule as the patch's update_from_output wrapper), so
    # no MRO times a call twice. This runs before the BFF patch wraps update_from_output, so BFF's own
    # merge hook stays outside the timed region.
    for mod, cls_name in (("vllm.v1.core.sched.scheduler", "Scheduler"),
                          ("vllm_ascend.core.recompute_scheduler", "RecomputeScheduler")):
        try:
            import importlib
            cls = getattr(importlib.import_module(mod), cls_name)
            for name, phase, on_call in (("schedule", "sched", _note_connector),
                                         ("update_from_output", "sched_update", None)):
                if name in cls.__dict__ and _wrap(cls, name, phase, on_call=on_call,
                                                  cprofile="sched" if SCHED_CPROFILE else None):
                    wrapped.append(f"{cls_name}.{phase}")
        except Exception as e:  # noqa: BLE001 - as above
            logger.warning("BFF step profile: could not wrap %s (%s).", cls_name, e)
    # What is inside the scheduler's time. This installs before the BFF patch captures
    # get_computed_blocks, so BFF's own wrapper stays outside `computed`; `free` contains BFF's
    # patched free_blocks and with it the NPU free-flush, which a stock arm does not pay.
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        for name, phase in (("allocate_slots", "alloc"), ("get_computed_blocks", "computed"),
                            ("free", "free")):
            if _wrap(KVCacheManager, name, phase):
                wrapped.append(phase)
    except Exception as e:  # noqa: BLE001 - as above
        logger.warning("BFF step profile: could not wrap the KV cache manager (%s).", e)
    # The whole engine step, so time that no phase above sees is still counted. step_fn is bound in
    # EngineCore.__init__, which runs after this.
    try:
        from vllm.v1.engine.core import EngineCore
        for name in ("step", "step_with_batch_queue"):
            if name in EngineCore.__dict__ and _wrap(EngineCore, name, "engine_step"):
                wrapped.append(f"engine_step({name})")
    except Exception as e:  # noqa: BLE001 - as above
        logger.warning("BFF step profile: could not wrap the engine core (%s).", e)
    if SCHED_CPROFILE:
        announce_once("sched_cprofile:on", "info",
                      "BFF sched cprofile: ON, profiling schedule/update_from_output for steps "
                      "%d-%d; the table prints once when the window closes.",
                      SCHED_CPROF.start, SCHED_CPROF.end)
    if RUNNER_CPROFILE:
        announce_once("runner_cprofile:on", "info",
                      "BFF runner cprofile: ON, profiling execute_model/sample_tokens for steps "
                      "%d-%d; the table prints once when the window closes.",
                      RUNNER_CPROF.start, RUNNER_CPROF.end)
    if wrapped:
        logger.info("BFF step profile ON (BFF_STEP_PROFILE=1): timing %s. Read GROUP EXCESS at "
                    "Running ~200 — the cost scales with requests x groups, so the drain tail is "
                    "already at parity and says nothing.", ", ".join(wrapped))
    else:
        logger.warning("BFF step profile requested but nothing was wrapped — the report would be "
                       "empty. Check that this process runs the Ascend model runner.")
