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

from types import MethodType

from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.engine.core import EngineCore

from kv_fast_fusion.fast_fusion_core import _initialize_kv_caches
from kv_fast_fusion.fast_fusion_block_pool import patched_free_blocks
from kv_fast_fusion.fast_fusion_scheduler import (
    _handle_block_merging_with_counts,
    update_from_output,
)

logger = init_logger("vllm.fast_fusion_pd_patch")


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
        # Bind the thin sample_tokens wrapper that propagates _updated_block_tables.
        self.sample_tokens = MethodType(_pd_sample_tokens, self)
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
        logger.info("Fast fusion P/D patch: lean runner init (raw, connector-level fusion).")

    GPUModelRunner.__init__ = _pd_patched_runner_init

    # --- 3. Merge channel (D-side block freeing) ---
    Scheduler._handle_block_merging_with_counts = _handle_block_merging_with_counts
    Scheduler.update_from_output = update_from_output

    # --- 4. Dedup-before-decrement free (LSH evict is guarded → no-op here) ---
    BlockPool.free_blocks = patched_free_blocks

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
                "kv_fast_fusion.p2p_nccl_connector_ff",
                "P2pNcclConnectorFF",
            )
            logger.info("Fast fusion P/D patch: registered P2pNcclConnectorFF.")
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("Fast fusion P/D patch: connector registration skipped: %s", e)

    logger.info("Fast fusion P/D patch applied.")


# Captured once; the thin wrapper delegates to the stock sampler then attaches the
# worker→scheduler block-merge channel. Defined at import (before __init__ patches it).
_ORIGINAL_SAMPLE_TOKENS = GPUModelRunner.sample_tokens


def _pd_sample_tokens(self, grammar_output):
    """Stock sample_tokens + propagate `_updated_block_tables` to the output so the patched
    scheduler `update_from_output` can free the redundant D-side blocks (connector fusion)."""
    output = _ORIGINAL_SAMPLE_TOKENS(self, grammar_output)
    updated = getattr(self, "_updated_block_tables", None)
    if updated and output is not None:
        try:
            output._updated_block_tables = updated
        except Exception:
            pass
    self._updated_block_tables = None
    return output
