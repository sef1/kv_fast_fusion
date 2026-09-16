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

logger = init_logger(__name__)

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

        The cadence widens (1, 10, 100, 1000, ...) so the first step proves the profiler engaged and a
        long run does not drown the decode log — the same shape the connector's own tallies use."""
        self.steps += 1
        self.reqs_total += self.last_num_reqs
        if self.steps < self._next_report:
            return False
        self._next_report *= 10
        return True

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
        total = self.per_step("step_total")
        pct = (100.0 * self.group_excess_ms() / total) if total else 0.0
        parts = [(f"BFF step profile | steps={self.steps} mean_reqs={self.mean_reqs():.1f} | "
                  f"step_total {total:.2f} ms | prepare_inputs {self.per_step('prepare_inputs'):.2f} "
                  f"attn_meta {self.per_step('attn_meta'):.2f}")]
        for phase in PER_GROUP_PHASES:
            parts.append(f"{phase} {self.per_step(phase):.2f} ms/step over "
                         f"{self.calls_per_step(phase):.1f} calls ({self.per_call(phase):.3f} ms/call)")
        parts.append(f"GROUP EXCESS (calls-1)xms/call = {self.group_excess_ms():.2f} ms/step "
                     f"({pct:.1f}% of the step)")
        return " | ".join(parts)


PROFILE = StepProfile()


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


def _wrap(cls, name: str, phase: str, on_call=None) -> bool:
    """Time ``cls.name`` into ``phase``. Returns True if it wrapped something new.

    ``on_call`` gets the call's args before the inner function runs, which is how ``num_reqs`` is
    captured — it arrives as ``commit_block_table``'s own argument, so nothing has to reach into the
    runner's state to find it."""
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return False

    def _timed(*args, _orig=orig, _phase=phase, _on=on_call, **kwargs):
        if _on is not None:
            _on(*args, **kwargs)
        t0 = time.perf_counter()
        try:
            return _orig(*args, **kwargs)
        finally:
            PROFILE.note(_phase, (time.perf_counter() - t0) * 1e3)

    setattr(_timed, _WRAP_SENTINEL, True)
    setattr(cls, name, _timed)
    return True


def _note_num_reqs(_self, num_reqs, *_a, **_k) -> None:
    PROFILE.last_num_reqs = int(num_reqs)


def _wrap_step(cls, name: str) -> bool:
    """``execute_model`` — the denominator. Times the whole step AND closes it, so every other phase
    is reported per step of the same run rather than against a step count guessed elsewhere."""
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return False

    def _timed(*args, _orig=orig, **kwargs):
        t0 = time.perf_counter()
        try:
            return _orig(*args, **kwargs)
        finally:
            PROFILE.note("step_total", (time.perf_counter() - t0) * 1e3)
            if PROFILE.note_step():
                logger.info("%s", PROFILE.render())

    setattr(_timed, _WRAP_SENTINEL, True)
    setattr(cls, name, _timed)
    return True


def install() -> None:
    """Wrap the five call sites, or do nothing at all when the flag is off.

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
    except Exception as e:  # noqa: BLE001 - as above
        logger.warning("BFF step profile: could not wrap the model runner (%s).", e)
    if wrapped:
        logger.info("BFF step profile ON (BFF_STEP_PROFILE=1): timing %s. Read GROUP EXCESS at "
                    "Running ~200 — the cost scales with requests x groups, so the drain tail is "
                    "already at parity and says nothing.", ", ".join(wrapped))
    else:
        logger.warning("BFF step profile requested but nothing was wrapped — the report would be "
                       "empty. Check that this process runs the Ascend model runner.")
