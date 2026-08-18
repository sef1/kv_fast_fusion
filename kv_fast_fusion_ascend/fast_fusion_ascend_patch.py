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

# Stage 1a (skip-transfer): a bounded ring of recently-promoted requests' OWN per-group physical
# block ids, keyed by _ext_hash(ext). When a later owner's rep has already FINISHED (rep-gone), its
# live block table is gone — but this ring still holds the rep's OLD block ids, so we can look up
# the physical block and ask the pool whether its KV is still intact (ref_cnt / cached) → split
# rep-gone into revivable vs truly-gone. This MEASURES the revival opportunity before we build the
# (cheap re-ref vs costly pin-at-arrival) mechanism. Read-only w.r.t. apply behavior. Scheduler
# thread only (single-writer at TP=1) so a plain OrderedDict needs no lock.
from collections import OrderedDict, defaultdict

_REP_HISTORY: "OrderedDict[int, list[list[int]]]" = OrderedDict()
_REP_HISTORY_CAP = int(os.environ.get("BFF_FF_REP_HISTORY_CAP", "4096"))


def _rep_history_record(ext_hash: int, blocks_by_group: list[list[int]]) -> None:
    """Record a promoted request's own per-group block ids, evicting the oldest past the cap."""
    _REP_HISTORY[ext_hash] = blocks_by_group
    _REP_HISTORY.move_to_end(ext_hash)
    while len(_REP_HISTORY) > _REP_HISTORY_CAP:
        _REP_HISTORY.popitem(last=False)


_PROMOTED_SEEN: "OrderedDict[str, None]" = OrderedDict()
_PROMOTED_SEEN_CAP = int(os.environ.get("BFF_FF_PROMOTED_SEEN_CAP", "4096"))


def _note_promoted(ext: str) -> None:
    """Remember that ``ext`` has been through the promotion hook, so a redirect map that shows up
    afterwards can be recognized as LATE rather than sitting in ``pending`` unexplained."""
    _PROMOTED_SEEN[ext] = None
    _PROMOTED_SEEN.move_to_end(ext)
    while len(_PROMOTED_SEEN) > _PROMOTED_SEEN_CAP:
        _PROMOTED_SEEN.popitem(last=False)


def _bff_sweep_late_maps(scheduler) -> None:
    """Count and discard redirect maps that arrived after their owner was already promoted.

    These CANNOT be applied, and the reason is structural rather than a missing guard. The promotion
    window is the only safe one in this mode: a newly scheduled request ships its FULL block table to
    the worker, so rewriting ``req_to_blocks`` before its first schedule propagates for free. Once a
    request has been scheduled, the scheduler sends only NEWLY allocated blocks
    (``_make_cached_request_data`` → ``req_to_new_blocks[req_id].get_block_ids()``), so a later
    rewrite never reaches the worker's device block table — it would keep reading the owner's
    original block while this frees it. That is a use-after-free, not a missed optimization, which is
    why the legacy worker-side path coupled every free to a successful device write.

    So the only correct handling is to drop them and make them visible: ``promo_rows_late`` is the
    metric to watch when the async fusion worker is enabled, since deferring the fusion check is what
    could push maps past their owner's promotion.
    """
    from kv_fast_fusion import fast_fusion_block_pool as _bp
    src = getattr(_bp, "_FF_PENDING_SOURCE", None)
    if src is None or not _PROMOTED_SEEN:
        return
    stats = getattr(src, "promo_stats", None)
    with src.lock:
        late = [e for e in src.pending if e in _PROMOTED_SEEN]
        n_rows = sum(len(rows) for e in late for rows in src.pending[e].values())
        for e in late:
            src.pending.pop(e, None)
    if late and stats is not None:
        stats["promo_rows_late"] = stats.get("promo_rows_late", 0) + n_rows
        stats["promo_maps_late"] = stats.get("promo_maps_late", 0) + len(late)
        logger.warning("BFF Ascend: %d redirect map(s) (%d rows) arrived after promotion and were "
                       "dropped — fusion is lagging its owner's KV transfer.", len(late), n_rows)


def _rep_gone_bucket(block_pool, ext_hash: int, gi: int, rep_slot: int) -> str:
    """Classify a rep-gone row by whether its old physical block still holds valid KV. Returns one of:
    nohist_missing | nohist_gi_oob | nohist_slot_oob | nohist_badblock | truly_gone |
    revive_live (ref_cnt>0) | revive_cached (freed but hash kept).

    The four nohist_* buckets were one "no_history" label until the con128 run made it the LARGEST
    loss category (768 of 1574 rep-gone rows). They mean completely different things and have
    different fixes, so lumping them hid the actual bug:
      nohist_missing  — the rep never promoted here (or was evicted from _REP_HISTORY): a genuine
                        rep-LIFETIME miss, which content-hash naming (V2) would fix.
      nohist_gi_oob   — the rep's recorded history has fewer groups than the row references.
      nohist_slot_oob — the rep's block list for this group is SHORTER than rep_slot: a P/D
                        block-table SHAPE mismatch. V2 does not fix this; the redirect itself is
                        describing a block layout the decode side never had.
    With only 200 prompts against a 4096-entry cap, eviction cannot explain 768 rows — so if this
    split comes back dominated by slot_oob, the shape mismatch is the real target."""
    old = _REP_HISTORY.get(ext_hash)
    if old is None:
        return "nohist_missing"
    if gi >= len(old):
        return "nohist_gi_oob"
    if not (0 <= rep_slot < len(old[gi])):
        return "nohist_slot_oob"
    bid = old[gi][rep_slot]
    try:
        blk = block_pool.blocks[bid]
    except (IndexError, AttributeError, TypeError):
        return "nohist_badblock"
    if blk.ref_cnt > 0:
        return "revive_live"        # something still references it → KV intact
    if blk.block_hash is not None:
        return "revive_cached"      # freed but not yet reused (hash kept) → optimistically revivable
    return "truly_gone"             # ref_cnt 0 and hash dropped → reused/overwritten


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


def _ext_of(rid: str) -> str:
    """External (cross-P/D stable) id of a request id. Uses vllm_ascend's canonical helper on the
    NPU stack; falls back to stripping the per-server random ``-<8hex>`` suffix (the same transform)
    off-NPU so the promotion apply is unit-testable."""
    try:
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
            get_external_request_id,
        )
        return get_external_request_id(rid)
    except ImportError:
        return rid.rsplit("-", 1)[0]


def _bff_promotion_apply(scheduler, request) -> None:
    """Apply this request's pending fusion redirect at PROMOTION time — the instant it leaves
    WAITING_FOR_REMOTE_KVS, before its first schedule.

    Why here: the audit at con512 proved the worker-side apply window is structurally dead — a
    redirect applied at the owner's recv-completion step fails its device block-table write because
    the owner joins ``input_batch`` only at its NEXT schedule (owner_not_written ≈ applied), so no
    blocks were actually freed. At promotion the scheduler still owns the block table: rewriting
    ``req_to_blocks`` NOW means the worker receives the rewritten table as the request's FULL initial
    block ids (a newly scheduled request ships complete ids, not a delta) — no device write, no
    overlay, no write-success coupling. This is also the provably pre-decode window the original
    apply-timing rule wanted.

    The pending map is the consumer connector's redirect-recv thread, published via
    ``fast_fusion_block_pool._FF_PENDING_SOURCE`` (same EngineCore process at TP=1 — the
    ``_ACTIVE_RUNNER`` pattern in reverse). Reps are resolved from the scheduler's OWN state,
    restricted to load-complete requests: a still-loading rep's blocks exist but its KV content is
    incomplete, and repointing an owner at half-arrived KV is silent corruption."""
    from kv_fast_fusion import fast_fusion_block_pool as _bp
    src = getattr(_bp, "_FF_PENDING_SOURCE", None)
    if src is None:
        return
    # Only plain promotions: a preempted-resume (num_preemptions > 0) re-enters with partial state
    # and its redirect was computed for a lifetime that no longer exists.
    if request.status != RequestStatus.WAITING:
        return
    ext = _ext_of(request.request_id)
    with src.lock:
        groups_rows = src.pending.pop(ext, None)
    # Mark AFTER the pop: anything that lands for this ext from here on is provably late (see
    # _bff_sweep_late_maps for why late maps cannot be applied).
    _note_promoted(ext)
    stats = getattr(src, "promo_stats", None)
    if not groups_rows:
        if stats is not None:
            stats["promo_no_rows"] += 1
        return
    from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
        _ext_hash,
        resolve_redirect_rows,
    )
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers

    def _blocks_of(rid):
        return [[b.block_id for b in m.req_to_blocks.get(rid, [])] for m in managers]

    ext2blocks = {ext: _blocks_of(request.request_id)}
    hash2ext = {_ext_hash(ext): ext}
    # Stage 1a: record THIS request's own blocks so it is a resolvable rep for future owners even
    # after it finishes (see _rep_gone_bucket). Cheap; scheduler-thread only.
    _rep_history_record(_ext_hash(ext), ext2blocks[ext])
    block_pool = getattr(getattr(scheduler.kv_cache_manager, "coordinator", None),
                         "block_pool", None)
    # Fetch block tables ONLY for the reps these rows actually reference (a promotion can touch a
    # handful of reps; materializing all ~max_num_seqs requests × groups every promotion would not).
    needed = {int(h) for rows in groups_rows.values() for (_o, h, _s) in rows}
    loading_hashes: set[int] = set()
    for rid2, req2 in scheduler.requests.items():
        ext2 = _ext_of(rid2)
        h2 = _ext_hash(ext2)
        if h2 not in needed or ext2 in ext2blocks:
            continue
        if req2.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            loading_hashes.add(h2)         # rep present but KV incomplete → not a valid target (yet)
            continue
        ext2blocks[ext2] = _blocks_of(rid2)
        hash2ext[h2] = ext2
    merged: dict[int, list[int]] = {}
    n_applied = n_unresolved = 0
    for gi, rows in groups_rows.items():
        new_blocks, na, nu, _nom = resolve_redirect_rows(ext2blocks, hash2ext, ext, int(gi), rows)
        n_applied += na
        n_unresolved += nu
        if new_blocks is not None:
            merged[int(gi)] = new_blocks
    # Attribute every unresolved row to exactly one of the two possible causes — the fixes differ
    # completely (rep finished → rep-lifetime work; rep still loading → defer/retry ordering work).
    # For rep-gone, further split (Stage 1a) by whether the rep's old physical block is still
    # revivable — this decides the Stage 1b mechanism (cheap re-ref vs costly pin-at-arrival).
    n_rep_loading = n_rep_gone = 0
    rev = defaultdict(int)      # bucket name -> count; see _rep_gone_bucket for the taxonomy
    for gi, rows in groups_rows.items():
        for (_o, h, rep_slot) in rows:
            if int(h) not in hash2ext:
                if int(h) in loading_hashes:
                    n_rep_loading += 1
                else:
                    n_rep_gone += 1
                    if block_pool is not None:
                        rev[_rep_gone_bucket(block_pool, int(h), int(gi), int(rep_slot))] += 1
    if n_rep_loading + n_rep_gone != n_unresolved:
        # A third cause would mean resolve_redirect_rows rejects rows for a reason this
        # classification doesn't model — surface it rather than letting it hide in either bucket.
        logger.warning("BFF Ascend promotion: unresolved split mismatch for %s: loading=%d gone=%d "
                       "!= unresolved=%d", request.request_id, n_rep_loading, n_rep_gone,
                       n_unresolved)
    if merged:
        # Existing machinery: rewrites req_to_blocks, touches reps, frees orphans, marks
        # num_cached_block. Its _BFF_FREE_PRERUNNING gates pass exactly here (status WAITING,
        # num_computed_tokens set and blocks cached by _update_waiting_for_remote_kv).
        scheduler._handle_block_merging_with_counts({request.request_id: merged})
    if stats is not None:
        stats["promo_applied"] += n_applied
        stats["promo_unresolved"] += n_unresolved
        stats["promo_unres_rep_loading"] = stats.get("promo_unres_rep_loading", 0) + n_rep_loading
        stats["promo_unres_rep_gone"] = stats.get("promo_unres_rep_gone", 0) + n_rep_gone
        # Stage 1a: revivability split of rep-gone (their sum == n_rep_gone). Names the fix:
        # revive_live+revive_cached large → content-hash naming (V2) recovers them;
        # nohist_slot_oob large → a P/D block-table shape mismatch, which V2 does NOT fix.
        for bucket, n in rev.items():
            key = "repgone_" + bucket
            stats[key] = stats.get(key, 0) + n
        # Keep the pre-split key populated so older dashboards//harness readers don't silently
        # report zero for what is currently the largest loss category.
        stats["repgone_no_history"] = stats.get("repgone_no_history", 0) + sum(
            n for b, n in rev.items() if b.startswith("nohist_"))
        if merged:
            stats["promo_merge_calls"] += 1


def _wrap_scheduler_promotion(scheduler_cls) -> bool:
    """Wrap ``_try_promote_blocked_waiting_request`` so a successful promotion immediately applies
    the request's pending fusion redirect (see :func:`_bff_promotion_apply`). Promotion and first
    schedule happen in the SAME ``schedule()`` iteration, so this is the only pre-first-schedule
    hook. Idempotent."""
    orig = scheduler_cls.__dict__.get("_try_promote_blocked_waiting_request")
    if orig is None:
        orig = scheduler_cls._try_promote_blocked_waiting_request
    if getattr(orig, _WRAP_SENTINEL, False):
        return False

    def _promote(self, request, _orig=orig):
        promoted = _orig(self, request)
        if promoted:
            try:
                _bff_promotion_apply(self, request)
            except Exception as e:  # pragma: no cover - never break scheduling
                logger.warning("BFF Ascend promotion apply failed for %s: %s",
                               request.request_id, e, exc_info=True)
        return promoted

    setattr(_promote, _WRAP_SENTINEL, True)
    scheduler_cls._try_promote_blocked_waiting_request = _promote
    return True


def _wrap_scheduler_update_from_output(cls) -> bool:
    """Wrap ``cls.update_from_output`` (only if the class DEFINES it in its own __dict__) so the BFF
    merge hook runs before the original. Returns True if it wrapped. Idempotent."""
    orig = cls.__dict__.get("update_from_output")
    if orig is None or getattr(orig, _WRAP_SENTINEL, False):
        return False

    def wrapped(self, scheduler_output, model_runner_output, _orig=orig):
        _bff_apply_block_merge(self, model_runner_output)
        try:
            _bff_sweep_late_maps(self)
        except Exception as e:      # pragma: no cover - diagnostics must never break the step
            logger.warning("BFF Ascend late-map sweep failed: %s", e)
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

    # v2 of the same transport: P ships per-block signatures and the DECODE decides which blocks are
    # worth sending, so a deduplicated block is never written. Registered alongside v1 rather than
    # replacing it, so the two are one --kv-transfer-config apart in an A/B.
    try:
        from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff_v2 import (
            register_mooncake_layerwise_ff_v2,
        )
        register_mooncake_layerwise_ff_v2()
    except Exception as e:  # pragma: no cover
        logger.warning("BFF Ascend: MooncakeLayerwiseConnectorFFv2 registration skipped: %s", e)

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
    # Tolerate an already-hashed block in cache_full_blocks: BFF_ALIAS_FUSED shares a rep block
    # across requests, so a later request can prefix-hit a block that already carries its
    # original owner's hash → stock's `assert blk.block_hash is None` kills EngineCore (seen on
    # NPU via _update_waiting_for_remote_kv → cache_blocks). Wrapper delegates to this original.
    if _ffbp._ORIG_CACHE_FULL is None:
        _ffbp._ORIG_CACHE_FULL = BlockPool.cache_full_blocks
        BlockPool.cache_full_blocks = _ffbp.patched_cache_full_blocks
    if _ffbp._ORIG_GET_COMPUTED is None:
        _ffbp._ORIG_GET_COMPUTED = KVCacheManager.get_computed_blocks
        KVCacheManager.get_computed_blocks = _ffbp.patched_get_computed_blocks

    # --- 3. Block-merge handler on the base Scheduler (inherited by the ascend schedulers) ---
    from vllm.v1.core.sched.scheduler import Scheduler
    from kv_fast_fusion.fast_fusion_scheduler import (
        _handle_block_merging_with_counts,
        _update_requests_with_invalid_blocks,
    )
    Scheduler._handle_block_merging_with_counts = _handle_block_merging_with_counts
    # KV-load-failure recovery must understand the BFF multi-group layout, exactly as on CUDA.
    # Stock unpacks a 1-tuple from get_block_ids ("TODO: add support for hybrid memory
    # allocator"); with BFF's 7 groups that raises `ValueError: too many values to unpack` and
    # kills EngineCore the first time a connector reports a failed KV load. v2 reports those
    # routinely (a declined block that never gets an alias applied is a recompute), so on the
    # NPU this fires as soon as dedup does any work at all.
    # Only the base class is patched: vllm_ascend's RecomputeScheduler/AsyncRecomputeScheduler
    # call _handle_invalid_blocks but do not define _update_requests_with_invalid_blocks, so
    # they inherit this replacement.
    Scheduler._update_requests_with_invalid_blocks = _update_requests_with_invalid_blocks
    # Promotion-time redirect apply: rewrite the owner's req_to_blocks the instant it leaves
    # WAITING_FOR_REMOTE_KVS (before first schedule). Consumes the pending map the FF consumer
    # connector publishes via _FF_PENDING_SOURCE; a no-op on the producer / when unpublished.
    if _wrap_scheduler_promotion(Scheduler):
        logger.info("BFF Ascend: promotion-time redirect apply installed on Scheduler.")

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
