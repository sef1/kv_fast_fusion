"""Fast-Fusion P/D disaggregation patch (ROUND 27).

A SEPARATE, leaner monkey-patch from the single-instance `apply_fast_fusion_graph_patch`.
P/D runs in `BFF_SCALE_MODE=raw` with connector-level fusion (`BFF_PD_FUSE=1`), so the
single-instance attention kernel, per-block norm buffers, and post-forward LSH/cc/tree dedup
never run — this patch deliberately omits them. It applies only what P/D needs:

  * the warmup+fusion KV-cache group split (P and D must match for block-id alignment),
  * a lean GPUModelRunner init that publishes `_ACTIVE_RUNNER` + the merge-channel state,
  * the scheduler block-merge channel (D-side block freeing via `_updated_block_tables`),
  * dedup-before-decrement `free_blocks`,
  * `P2pNcclEngine.free_recv_tensor` (frees recv buffers at consume time — bounds the pinned
    pool; defined here as a monkey-patch so stock vLLM is untouched),
  * registration of the group-aware `P2pNcclConnectorFF`.

Toggle in `kv_fast_fusion/__init__.py` against `apply_fast_fusion_graph_patch`.
"""

import os

from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.engine.core import EngineCore

from kv_fast_fusion.fast_fusion_core import _initialize_kv_caches
from kv_fast_fusion.fast_fusion_block_pool import patched_free_blocks
from kv_fast_fusion.fast_fusion_scheduler import (
    _handle_block_merging_with_counts,
    _update_requests_with_invalid_blocks,
    update_from_output,
)

logger = init_logger("vllm.fast_fusion_pd_patch")

# ROUND 48: `ratio` scale mode for P/D. In raw (default) the lean patch omits the norm/kernel
# infra. In ratio, the producer ships per-redirect K/V norm ratios and D re-enables the minimal
# subset (norm buffers + slots + BFF Triton kernel) so the shared rep block is scaled per request.
_PD_SCALE_MODE = os.environ.get("BFF_SCALE_MODE", "raw").lower()


def _selected_connector_names() -> set:
    """Connector names this process was actually launched with, from ``--kv-transfer-config``.

    Read from ``sys.argv`` because the patch runs at import, long before a ``VllmConfig`` exists.
    Includes connectors nested under MultiConnector's ``kv_connector_extra_config.connectors``.
    Returns an empty set when the flag is absent or unparseable — callers must treat that as
    "unknown", never as "not selected"."""
    import json
    import sys

    argv = sys.argv
    raw = None
    for i, a in enumerate(argv):
        if a == "--kv-transfer-config" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif a.startswith("--kv-transfer-config="):
            raw = a.split("=", 1)[1]
    if not raw:
        return set()
    try:
        cfg = json.loads(raw)
    except Exception:
        return set()

    names = set()

    def _walk(node):
        if not isinstance(node, dict):
            return
        name = node.get("kv_connector")
        if isinstance(name, str):
            names.add(name)
        extra = node.get("kv_connector_extra_config")
        if isinstance(extra, dict):
            for child in extra.get("connectors") or ():
                _walk(child)

    _walk(cfg)
    return names


def _registration_failed(name: str, exc: Exception) -> None:
    """A connector failed to register. Fatal iff this run actually selected it.

    Registration lives in try/except so a box without mooncake can still run the NCCL path. But
    swallowing the failure for the connector the run REQUIRES turns an ImportError into
    ``Unsupported connector type`` raised ~400 lines later inside pydantic validation, naming
    neither the module nor the real cause. That cost two full debugging cycles on 2026-08-19, and
    it is the same silent-degradation class as `no_kv_tensors` and `owner_never_batched`.

    When the selected set cannot be determined (no --kv-transfer-config on this process, e.g. the
    API server front-end) we warn rather than raise: refusing to start a process that never needed
    the connector would be worse than the bug being fixed."""
    selected = _selected_connector_names()
    if name in selected:
        raise RuntimeError(
            f"{name} is selected by --kv-transfer-config but failed to register: {exc!r}. "
            "Refusing to continue — serving would fail later with an unrelated "
            "'Unsupported connector type' error. Fix the import above."
        ) from exc
    logger.warning("Fast fusion P/D patch: %s registration skipped (not selected by this "
                   "process): %s", name, exc)


def _free_recv_tensor(self, tensor_id: str):
    """Release a recv'd tensor as soon as the connector has consumed it.

    For PUT/PUT_ASYNC each tensor_id is received exactly once, so the buffer can be released
    at consume time instead of waiting for `get_finished` (request completion). A spilled
    (pool) tensor frees its pinned block; a GPU-resident tensor already had its `buffer_size`
    decremented by `recv_tensor`, so popping it just drops the last ref. Bounds pinned-pool /
    recv_store residency to in-flight, not-yet-consumed tensors."""
    with self.recv_store_cv:
        tensor = self.recv_store.pop(tensor_id, None)
    if isinstance(tensor, tuple):
        addr, _, _ = tensor
        self.pool.free(addr)


def apply_fast_fusion_pd_patch():
    """Apply the P/D disaggregation patch (see module docstring)."""

    # --- 1. KV-cache group split (warmup + fusion groups), shared with single instance ---
    EngineCore._initialize_kv_caches = _initialize_kv_caches

    # --- 2. Lean GPUModelRunner init: publish runner + merge-channel state only ---
    original_init = GPUModelRunner.__init__

    def _pd_patched_runner_init(self, *args, **kwargs):
        # No sample_tokens override: the connector writes the redirect map onto this runner and
        # the patched scheduler `update_from_output` reads it directly off `_ACTIVE_RUNNER`
        # (same process at TP=1) — the async sample_tokens output wrapper drops attached attrs.
        self.fused_requests = {}
        self._updated_block_tables = None
        # raw mode reads no per-block norms; the connector only needs block tables.
        self.norms_k_buf = None
        self.norms_v_buf = None
        # Publish this runner so the connector (block-table writes / group map) and the
        # scheduler-side free path can reach it (same EngineCore process at TP=1).
        import kv_fast_fusion.fast_fusion_block_pool as _ffbp
        _ffbp._ACTIVE_RUNNER = self

        original_init(self, *args, **kwargs)

        # ratio mode: allocate the slot-indexed per-(layer, slot, block) K/V norm buffers the
        # BFF Triton kernel reads (ported from the single-instance graph patch). The connector
        # populates runner.fused_requests from the shipped ratios; _fill_norm_buffers slot-fills.
        if _PD_SCALE_MODE == "ratio":
            import torch
            vcfg = self.vllm_config
            num_layers = vcfg.model_config.get_num_layers(vcfg.parallel_config)
            max_reqs = vcfg.scheduler_config.max_num_seqs
            max_blocks_per_req = max(1, vcfg.model_config.max_model_len // vcfg.cache_config.block_size)
            num_slots = max_reqs
            self.norms_k_buf = torch.ones(
                num_layers, num_slots + 1, max_blocks_per_req,
                dtype=torch.bfloat16, device=self.device)
            self.norms_v_buf = torch.ones(
                num_layers, num_slots + 1, max_blocks_per_req,
                dtype=torch.bfloat16, device=self.device)
            self._fused_slot = {}                          # req_id → slot in [1, num_slots]
            self._fused_filled = {}                        # req_id → layers written (see refill)
            self._free_slots = list(range(1, num_slots + 1))
            self._seq_to_slot = torch.zeros(max_reqs, dtype=torch.int32, device=self.device)
            self._seq_to_slot_cpu = torch.zeros(max_reqs, dtype=torch.int32, device="cpu")
            # The kernel is bound only for layers in [warmup, max_layer). A FUSION layer outside
            # that window would be substituted UNSCALED — raw-mode error re-entering through the
            # back door, and invisible. It cannot happen today: fast_fusion_core splits
            # `warmup = layers[0:2] + layers[-2:]`, `fused = layers[2:-2]`, exactly this window. The
            # two sides derive the bound independently, so _pd_refill_grown checks the agreement
            # against the layers that actually carry scales rather than trusting this comment.
            self._ff_warmup_layers = 2
            self._ff_max_layer_idx = num_layers - 2
            logger.info("Fast fusion P/D ratio: norm buffers [%d layers, %d slots, %d blocks/req].",
                        num_layers, max_reqs, max_blocks_per_req)
        logger.info("Fast fusion P/D patch: lean runner init (mode=%s, connector-level fusion).",
                    _PD_SCALE_MODE)

    GPUModelRunner.__init__ = _pd_patched_runner_init

    # --- 3. Merge channel (D-side block freeing) ---
    Scheduler._handle_block_merging_with_counts = _handle_block_merging_with_counts
    Scheduler.update_from_output = update_from_output
    # KV-load-failure recovery must understand the BFF multi-group layout. Stock unpacks a 1-tuple
    # from get_block_ids ("TODO: add support for hybrid memory allocator") and raises ValueError
    # with BFF's groups, killing EngineCore the first time a KV pull fails. See the docstring.
    Scheduler._update_requests_with_invalid_blocks = _update_requests_with_invalid_blocks

    # --- 4. Dedup-before-decrement free (LSH evict is guarded → no-op here) ---
    BlockPool.free_blocks = patched_free_blocks

    # ROUND 39: wrap _maybe_evict_cached_block so a recycled/evicted block drops any
    # lever-3 fusion aliases pointing at it (staleness guard). Idempotent.
    import kv_fast_fusion.fast_fusion_block_pool as _ffbp2
    if _ffbp2._ORIG_MAYBE_EVICT is None:
        _ffbp2._ORIG_MAYBE_EVICT = BlockPool._maybe_evict_cached_block
        BlockPool._maybe_evict_cached_block = _ffbp2.patched_maybe_evict_cached_block

    # Tolerate an already-hashed block in cache_full_blocks. Lever 3 (add_block_alias) shares a
    # representative block across requests, so a later request can prefix-hit a block that already
    # carries its original owner's hash and stock's `assert blk.block_hash is None` kills
    # EngineCore. Observed on NPU (via _update_waiting_for_remote_kv → cache_blocks); the same
    # latent bug exists here, so keep both backends patched identically.
    if _ffbp2._ORIG_CACHE_FULL is None:
        _ffbp2._ORIG_CACHE_FULL = BlockPool.cache_full_blocks
        BlockPool.cache_full_blocks = _ffbp2.patched_cache_full_blocks

    # ROUND 40: wrap get_computed_blocks to record per-group resume recovery (BFF_HIT_DEBUG).
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    if _ffbp2._ORIG_GET_COMPUTED is None:
        _ffbp2._ORIG_GET_COMPUTED = KVCacheManager.get_computed_blocks
        KVCacheManager.get_computed_blocks = _ffbp2.patched_get_computed_blocks

    # --- 5. P/D-only: pool lifecycle + connector ---
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
            P2pNcclEngine,
        )
        P2pNcclEngine.free_recv_tensor = _free_recv_tensor
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("Fast fusion P/D patch: could not patch free_recv_tensor: %s", e)

    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        if "P2pNcclConnectorFF" not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                "P2pNcclConnectorFF",
                "kv_fast_fusion.connectors.p2p_nccl_connector_ff",
                "P2pNcclConnectorFF",
            )
            logger.info("Fast fusion P/D patch: registered P2pNcclConnectorFF.")
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("Fast fusion P/D patch: connector registration skipped: %s", e)

    # Mooncake transport for the same GPU P/D setup (RDMA pull instead of NCCL push). Registered
    # by module path, so this does NOT import the mooncake package — a box without it only fails
    # when a run actually selects the connector.
    try:
        from kv_fast_fusion.connectors.mooncake_connector_ff import (
            register_mooncake_connector_ff,
        )
        register_mooncake_connector_ff()
    except Exception as e:  # pragma: no cover - optional dependency
        _registration_failed("MooncakeConnectorFF", e)

    # v2 of the same transport: the producer ships per-block signatures and the DECODE decides what
    # not to pull, so a deduplicated block is never transferred at all. Registered alongside v1
    # rather than replacing it, so the two are one --kv-transfer-config apart in an A/B.
    try:
        from kv_fast_fusion.connectors.mooncake_connector_ff_v2 import (
            register_mooncake_connector_ff_v2,
        )
        register_mooncake_connector_ff_v2()
    except Exception as e:  # pragma: no cover - optional dependency
        _registration_failed("MooncakeConnectorFFv2", e)

    # The PRE-EXTRACTION v2 (773 lines, commit 6122e3126), kept as a measurement baseline so the
    # shared-core extraction can finally be A/B'd against what it replaced. Registered beside the
    # current one, never instead of it. Delete both this block and the module once the question is
    # settled — see the module docstring.
    try:
        from kv_fast_fusion.connectors.mooncake_connector_ff_v2_legacy import (
            register_mooncake_connector_ff_v2_legacy,
        )
        register_mooncake_connector_ff_v2_legacy()
    except Exception as e:  # pragma: no cover - optional dependency
        _registration_failed("MooncakeConnectorFFv2Legacy", e)

    # --- 6. ratio mode: re-enable the minimal norm-scaling kernel infra on D ---
    if _PD_SCALE_MODE == "ratio":
        _apply_pd_ratio_kernel_infra()

    logger.info("Fast fusion P/D patch applied (mode=%s).", _PD_SCALE_MODE)


def _apply_pd_ratio_kernel_infra() -> None:
    """ratio-only: bind the BFF Triton-kernel attention path + a lean attention-metadata wrapper
    that slot-fills the connector-supplied norms and attaches them to each fusion group's metadata.
    Reuses the single-instance `patched_forward` (kernel routing) and `_fill_norm_buffers` verbatim;
    the heavy `_build_attention_metadata` is WRAPPED (not reimplemented), attaching norms per
    fusion layer afterward."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from kv_fast_fusion.fast_fusion_flash import patched_forward
    from kv_fast_fusion.legacy.kv_fast_fusion_graph_runner import _fill_norm_buffers

    # Route fusion-layer attention through the BFF kernel (gates internally on
    # BFF_SCALE_MODE!=raw + has_fused_reqs + norms_k_buf_full; raw layers fall back to flash).
    FlashAttentionImpl.forward = patched_forward
    # Slot-fill from self.fused_requests + build the per-step seq→slot map (reused verbatim).
    GPUModelRunner._fill_norm_buffers = _fill_norm_buffers

    _orig_build_meta = GPUModelRunner._build_attention_metadata

    def _pd_refill_grown(runner) -> None:
        """Re-write the norm rows of any request whose scales GREW after its slot was filled.

        `_fill_norm_buffers` is write-once per request (`if req_id in self._fused_slot: continue`) —
        correct for the single-instance path, where a request's whole ratio map is known before it
        is ever batched. In P/D it is not: `AliasApplier` applies group by group, and the
        `rep_not_resident` retry deliberately carries a group over to a later step. A request whose
        group 1 alias lands at step N and group 3 alias at step N+2 would keep the slot it got at
        step N and group 3's scale would never reach the buffer — an UNSCALED substitution, the one
        error ratio mode exists to remove, showing up only as unexplained accuracy loss.

        Cheap because it fires only on the growth: `_fused_filled` records how many layers were
        written, so the steady state is one dict lookup per fused request."""
        filled = runner._fused_filled
        dt = runner.norms_k_buf.dtype
        B = runner.norms_k_buf.shape[2]
        for rid, slot in runner._fused_slot.items():
            per_layer = runner.fused_requests.get(rid)
            if per_layer is None or filled.get(rid) == len(per_layer):
                continue
            for layer_name, (nk, nv) in per_layer.items():
                try:
                    li = int(layer_name.split('.')[2])
                except Exception:
                    continue
                # A scale for a layer the kernel is not bound to is a substitution that silently
                # stays unscaled. Say so once — it means the group split and the kernel window have
                # drifted apart, and no accuracy number would name the cause.
                if not (runner._ff_warmup_layers <= li < runner._ff_max_layer_idx):
                    if not getattr(runner, "_ff_window_warned", False):
                        runner._ff_window_warned = True
                        logger.warning(
                            "BFF P/D ratio: layer %s (index %d) carries a scale but is outside the "
                            "kernel window [%d, %d) — that alias is applied UNSCALED. The fusion "
                            "group split and the kernel window no longer agree.",
                            layer_name, li, runner._ff_warmup_layers, runner._ff_max_layer_idx)
                    continue
                n = min(nk.shape[0], B)
                runner.norms_k_buf[li, slot, :n] = nk[:n].to(dt)
                runner.norms_v_buf[li, slot, :n] = nv[:n].to(dt)
            filled[rid] = len(per_layer)
            # Once per process: the single line that says the scales reached the GPU buffer the
            # kernel reads. Everything upstream can succeed while this never happens (the connector
            # downgraded to raw, the norms were never shipped, the runner has no buffers), and the
            # only other symptom is an accuracy number that quietly looks like raw mode.
            if not getattr(runner, "_ff_ratio_logged", False):
                runner._ff_ratio_logged = True
                logger.info(
                    "BFF P/D ratio: first scales in the norm buffers | req=%s slot=%d layers=%d "
                    "| k range [%.3f, %.3f]", rid, slot, len(per_layer),
                    float(runner.norms_k_buf[:, slot, :].min()),
                    float(runner.norms_k_buf[:, slot, :].max()))

    def _pd_build_attention_metadata(self, *args, **kwargs):
        out = _orig_build_meta(self, *args, **kwargs)
        if getattr(self, "norms_k_buf", None) is None:
            return out
        try:
            attn_metadata, _spec = out
            # Free slots of finished requests (gone from runner.requests) → reset their norm rows.
            live = set(getattr(self, "requests", {}).keys())
            for rid in list(self._fused_slot.keys()):
                if rid not in live:
                    slot = self._fused_slot.pop(rid)
                    self._free_slots.append(slot)
                    self.norms_k_buf[:, slot, :] = 1.0
                    self.norms_v_buf[:, slot, :] = 1.0
                    self.fused_requests.pop(rid, None)
                    self._fused_filled.pop(rid, None)
            # Slot-fill (write-once) + build this step's seq→slot map, matching single-instance.
            req_ids = self.input_batch.req_ids
            fused_reqs = [r for r in req_ids if r in self.fused_requests]
            _pd_refill_grown(self)
            self._fill_norm_buffers(req_ids, fused_reqs)
            # `_fill_norm_buffers` skips a request when `_free_slots` is empty, and a request with
            # no slot maps to seq_to_slot 0 — the all-ones sentinel row — so its aliases are applied
            # UNSCALED. There is one slot per max_num_seqs and the decode has been seen running 149
            # of 150, so this is not a hypothetical margin. It is also completely silent: the run
            # still reports ratio mode, still pays for the kernel, and quietly delivers raw for
            # whichever requests lost the race. Count it, and name it once.
            missing = [r for r in fused_reqs if r not in self._fused_slot]
            if missing:
                self._ff_slots_missed = getattr(self, "_ff_slots_missed", 0) + len(missing)
                if not getattr(self, "_ff_slots_warned", False):
                    self._ff_slots_warned = True
                    logger.warning(
                        "BFF P/D ratio: %d fused request(s) got no norm slot (%d of %d in use) — "
                        "their aliases are applied UNSCALED. Raise --max-num-seqs or the slot "
                        "count; every later occurrence is counted, not logged.",
                        len(missing), len(self._fused_slot), self.norms_k_buf.shape[1] - 1)
            # Attach the full slot-indexed buffers + seq→slot to every fusion-layer metadata so
            # patched_forward selects [layer_idx]. has_fused_reqs gates the kernel vs flash path
            # (forced True during cudagraph capture so the captured graph records the kernel).
            for_capture = bool(kwargs.get("for_cudagraph_capture", False))
            has = for_capture or bool(fused_reqs)
            warmup = self._ff_warmup_layers
            max_layer = self._ff_max_layer_idx
            md_dicts = ([attn_metadata] if isinstance(attn_metadata, dict)
                        else attn_metadata if isinstance(attn_metadata, list) else [])
            for md in md_dicts:
                if not isinstance(md, dict):
                    continue
                for layer_name, meta_obj in md.items():
                    try:
                        li = int(layer_name.split('.')[2])
                    except Exception:
                        continue
                    if meta_obj is not None and warmup <= li < max_layer:
                        meta_obj.norms_k_buf_full = self.norms_k_buf
                        meta_obj.norms_v_buf_full = self.norms_v_buf
                        meta_obj.bff_seq_to_slot = self._seq_to_slot
                        meta_obj.has_fused_reqs = has
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("BFF P/D ratio: build-meta attach failed: %s", e)
        return out

    GPUModelRunner._build_attention_metadata = _pd_build_attention_metadata
    logger.info("Fast fusion P/D ratio: bound BFF kernel + norm-attach metadata wrapper.")
