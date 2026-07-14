"""Ascend/NPU BFF fast-fusion runtime patch (raw mode, connector-level P/D fusion).

The NPU analogue of :func:`kv_fast_fusion.fast_fusion_pd_patch.apply_fast_fusion_pd_patch`. It applies
only what the Mooncake raw path needs on the decode node, and deliberately omits the CUDA/NCCL-only
bits (P2pNccl engine/connector, ratio-mode FlashAttention/Triton kernel).

Two categories:

  * **Reused, device-agnostic** (imported unchanged from ``kv_fast_fusion``): the EngineCore KV-cache
    group split, the scheduler block-merge handler ``_handle_block_merging_with_counts``, and the
    block-pool ``free_blocks`` / ``_maybe_evict_cached_block`` / ``get_computed_blocks`` patches.
  * **Ascend-specific** (defined here): the ``NPUModelRunner.__init__`` lean init that publishes
    ``_ACTIVE_RUNNER``, and a **wrapper** around the scheduler's ``update_from_output`` — vllm_ascend
    ships ``RecomputeScheduler``/``AsyncRecomputeScheduler`` that OVERRIDE ``update_from_output``, so
    the CUDA path's wholesale replacement of ``Scheduler.update_from_output`` would be shadowed (or
    would clobber the recompute logic). We wrap instead: run the BFF block-merge hook, then call the
    original.
"""

import os

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

logger = init_logger("vllm.fast_fusion_ascend_patch")

# Per-step block-merge diagnostics (ready/pending/dropped, running vs waiting) are noisy once we free
# pre-RUNNING; gate them behind BFF_PD_DEBUG. Errors/warnings and "Block merging freed X" stay on.
_BFF_DEBUG = os.environ.get("BFF_PD_DEBUG", "0") == "1"

# Marks a wrapped update_from_output / __init__ so re-applying the patch is a no-op.
_WRAP_SENTINEL = "_bff_ascend_wrapped"


def _bff_apply_block_merge(scheduler, model_runner_output) -> None:
    """Run BFF's consumer-side block-merge for this step, then leave the rest of the scheduler's
    update_from_output untouched. Ported from
    ``kv_fast_fusion.fast_fusion_scheduler.update_from_output`` (the merge-apply prologue), so the
    NCCL scheduler patch is left untouched. Consume-once semantics on the ``_ACTIVE_RUNNER`` channel
    make this safe even if two wrapped classes in an MRO both ran it (they don't today)."""
    block_merge_mapping = getattr(model_runner_output, "_updated_block_tables", None)
    if not block_merge_mapping:
        # TP=1: the connector wrote the redirect map straight onto the runner. Read it off
        # _ACTIVE_RUNNER (worker+scheduler share the process) and CONSUME it so it applies once.
        try:
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            _runner = getattr(_bp, "_ACTIVE_RUNNER", None)
            if _runner is not None:
                block_merge_mapping = getattr(_runner, "_updated_block_tables", None)
                if block_merge_mapping:
                    _runner._updated_block_tables = None
                    if _BFF_DEBUG:
                        logger.info("BFF Ascend: block-merge via runner | reqs=%d",
                                    len(block_merge_mapping))
        except Exception as e:
            logger.warning("BFF Ascend runner block-merge fallback failed: %s", e)
    kv_connector_output = getattr(model_runner_output, "kv_connector_output", None)
    if not block_merge_mapping and kv_connector_output is not None:
        # TP>1: the map rides the connector stats carrier (BFFMergeStats) — separate process.
        try:
            _stats = getattr(kv_connector_output, "kv_connector_stats", None)
            _data = getattr(_stats, "data", None)
            if _data and _data.get("bff_merges"):
                block_merge_mapping = _data["bff_merges"]
                logger.info("BFF Ascend: block-merge via connector stats (TP>1) | reqs=%d",
                            len(block_merge_mapping))
        except Exception as e:
            logger.warning("BFF Ascend connector-stats block-merge fallback failed: %s", e)
    # Persist-and-retry: the worker stages the redirect map at the step the owner's KV recv
    # completes, but a just-loaded P/D consumer request is usually not yet applicable in the scheduler
    # (still receiving, or WAITING behind the max_num_seqs cap) — so the consume-once map would be lost
    # (→ "freed 0 blocks"). Accumulate maps and apply each once its request is SAFE to merge, retrying
    # the rest and dropping finished ones. A request is safe once its KV load has finished
    # (not WAITING_FOR_REMOTE_KVS) and it has computed tokens — this frees post-load WAITING consumers
    # too (they pin the receive-buffer backlog), not just RUNNING ones. The shared handler re-checks.
    pending = getattr(scheduler, "_bff_pending_merges", None)
    if pending is None:
        pending = {}
        scheduler._bff_pending_merges = pending
    if block_merge_mapping:
        for rid, groups in block_merge_mapping.items():
            pending.setdefault(rid, {}).update(groups)
    if not pending:
        return
    ready: dict[str, dict[int, list[int]]] = {}
    keep: dict[str, dict[int, list[int]]] = {}
    n_dropped = 0
    for rid, groups in pending.items():
        req = scheduler.requests.get(rid)
        if req is None:
            n_dropped += 1                       # finished / evicted → drop
        elif (req.status != RequestStatus.WAITING_FOR_REMOTE_KVS
              and req.num_computed_tokens > 0):
            ready[rid] = groups                  # post-KV-load (RUNNING or WAITING) → safe to merge
        else:
            keep[rid] = groups                   # still receiving KV → retry next step
    scheduler._bff_pending_merges = keep
    # Diagnostic (BFF_PD_DEBUG only): ready/pending/dropped + running vs waiting (block-bound check).
    if _BFF_DEBUG:
        n_wait = len(getattr(scheduler, "waiting", ())) + len(getattr(scheduler, "skipped_waiting", ()))
        logger.info("BFF Ascend: block-merge partition | ready=%d pending=%d dropped=%d | "
                    "running=%d waiting=%d", len(ready), len(keep), n_dropped,
                    len(getattr(scheduler, "running", ())), n_wait)
    if ready:
        try:
            scheduler._handle_block_merging_with_counts(ready)
        except Exception as e:
            logger.error("BFF Ascend block merging failed — skipping this step: %s", e,
                         exc_info=True)


def _wrap_scheduler_update_from_output(cls) -> bool:
    """Wrap ``cls.update_from_output`` (only if the class DEFINES it in its own __dict__) so the BFF
    merge hook runs before the original. Returns True if it wrapped. Idempotent."""
    orig = cls.__dict__.get("update_from_output")
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return False

    def wrapped(self, scheduler_output, model_runner_output, _orig=orig):
        _bff_apply_block_merge(self, model_runner_output)
        return _orig(self, scheduler_output, model_runner_output)

    setattr(wrapped, _WRAP_SENTINEL, True)
    cls.update_from_output = wrapped
    logger.info("BFF Ascend: wrapped update_from_output on %s.", cls.__name__)
    return True


def _patch_npu_model_runner() -> None:
    """Patch ``NPUModelRunner.__init__`` to publish ``_ACTIVE_RUNNER`` and the merge-channel state
    (raw mode: no norm buffers). Mirrors the raw branch of the CUDA ``_pd_patched_runner_init``."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    orig_init = NPUModelRunner.__init__
    if getattr(orig_init, _WRAP_SENTINEL, False):
        return

    def _init(self, *args, _orig=orig_init, **kwargs):
        # Publish + seed the merge-channel state BEFORE the heavy init (mirrors the CUDA patch).
        self.fused_requests = {}
        self._updated_block_tables = None
        self.norms_k_buf = None            # raw mode reads no per-block norms
        self.norms_v_buf = None
        import kv_fast_fusion.fast_fusion_block_pool as _ffbp
        _ffbp._ACTIVE_RUNNER = self
        _orig(self, *args, **kwargs)
        logger.info("BFF Ascend: lean NPUModelRunner init (raw, _ACTIVE_RUNNER published).")

    setattr(_init, _WRAP_SENTINEL, True)
    NPUModelRunner.__init__ = _init
    logger.info("BFF Ascend: patched NPUModelRunner.__init__.")


def apply_fast_fusion_ascend_patch() -> None:
    """Apply the Ascend/NPU BFF patch (see module docstring). Raw mode only.

    GATED on ``BFF_PD_FUSE==1``. When BFF is off (stock ``layerwise``/``mooncakev1`` baselines that
    still launch via ``kv_fast_fusion.fast_fusion_main``), this is a NO-OP: it must NOT split the KV
    cache into warmup+fusion groups, because a multi-group config crashes the scheduler's
    ``_connector_finished`` (``assert len(kv_cache_groups)==1``) whenever the top connector isn't taken
    as ``SupportsHMA`` — and reshaping the layout for a stock run is wrong regardless. The FF connector
    name is still registered (harmless; it self-disables fusion when ``BFF_PD_FUSE!=1``)."""
    # Register the connector name unconditionally so it always resolves if selected.
    try:
        from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
            register_mooncake_layerwise_ff,
        )
        register_mooncake_layerwise_ff()
    except Exception as e:  # pragma: no cover
        logger.warning("BFF Ascend: MooncakeLayerwiseConnectorFF registration skipped: %s", e)

    if os.environ.get("BFF_PD_FUSE", "0") != "1":
        logger.info("BFF Ascend: BFF_PD_FUSE!=1 → stock (no KV-cache group split, no patches).")
        return

    scale_mode = os.environ.get("BFF_SCALE_MODE", "raw").lower()
    if scale_mode != "raw":
        logger.warning("BFF Ascend: BFF_SCALE_MODE=%s requested but only 'raw' is supported on NPU; "
                       "proceeding without the ratio kernel.", scale_mode)

    # --- 1. KV-cache group split (EngineCore, device-agnostic) ---
    from vllm.v1.engine.core import EngineCore
    from kv_fast_fusion.fast_fusion_core import _initialize_kv_caches
    EngineCore._initialize_kv_caches = _initialize_kv_caches

    # --- 2. Block-pool + KV-cache-manager patches (device-agnostic) ---
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    import kv_fast_fusion.fast_fusion_block_pool as _ffbp
    BlockPool.free_blocks = _ffbp.patched_free_blocks
    if _ffbp._ORIG_MAYBE_EVICT is None:
        _ffbp._ORIG_MAYBE_EVICT = BlockPool._maybe_evict_cached_block
        BlockPool._maybe_evict_cached_block = _ffbp.patched_maybe_evict_cached_block
    if _ffbp._ORIG_GET_COMPUTED is None:
        _ffbp._ORIG_GET_COMPUTED = KVCacheManager.get_computed_blocks
        KVCacheManager.get_computed_blocks = _ffbp.patched_get_computed_blocks

    # --- 3. Block-merge handler on the base Scheduler (inherited by the ascend schedulers) ---
    from vllm.v1.core.sched.scheduler import Scheduler
    from kv_fast_fusion.fast_fusion_scheduler import _handle_block_merging_with_counts
    Scheduler._handle_block_merging_with_counts = _handle_block_merging_with_counts

    # --- 4. NPUModelRunner lean init ---
    try:
        _patch_npu_model_runner()
    except Exception as e:  # pragma: no cover - only reachable off the Ascend stack
        logger.warning("BFF Ascend: could not patch NPUModelRunner (%s); "
                       "consumer block-sharing will be inert.", e)

    # --- 5. Wrap update_from_output on every scheduler class that defines it (adapt, not replace) ---
    # Candidates: base Scheduler + the vllm_ascend recompute schedulers. Only those that define the
    # method in their own __dict__ are wrapped; AsyncScheduler / AsyncRecomputeScheduler inherit it
    # (verified) so they are covered transitively — and since RecomputeScheduler.update_from_output
    # does not call super(), no MRO produces a double-apply.
    candidates = [Scheduler]
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        candidates.append(AsyncScheduler)
    except Exception:
        pass
    try:
        from vllm_ascend.core.recompute_scheduler import (
            AsyncRecomputeScheduler,
            RecomputeScheduler,
        )
        candidates += [RecomputeScheduler, AsyncRecomputeScheduler]
    except Exception as e:
        logger.info("BFF Ascend: recompute schedulers not importable (%s); wrapping base only.", e)
    for cls in candidates:
        try:
            _wrap_scheduler_update_from_output(cls)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("BFF Ascend: failed to wrap %s.update_from_output: %s",
                           getattr(cls, "__name__", cls), e)

    logger.info("Fast fusion Ascend patch applied (mode=raw, BFF_PD_FUSE=1).")
