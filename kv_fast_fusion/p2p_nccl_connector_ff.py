"""Fast-Fusion-aware P2P NCCL connector for P/D disaggregation.

The stock ``P2pNcclConnector`` is **single-group**: it captures only ``block_ids[0]``
(KV-cache group 0) in the connector metadata and applies that one block table to *every*
attention layer when it saves (producer) / injects (consumer) KV. That assumption breaks
under BFF, which splits the model into a warmup group (first/last 2 layers, sliding window)
plus ``BFF_GROUP_SIZE``-packed fusion groups — each layer lives in its own KV-cache group
with its **own** block table. Indexing a fusion-layer's paged KV with the warmup group's
block ids sends/receives the wrong physical blocks ("kv_cache does not match").

``P2pNcclConnectorFF`` fixes exactly that: it carries the **full per-group** block-id list
in the metadata and, on save/load, indexes each layer by the block table of the group that
layer belongs to. Everything else (the NCCL engine, handshake, finished-tracking) is
inherited unchanged.

Milestone 1 (P-side fusion): fusion still runs on the producer post-forward, *after* the KV
is already saved with pre-fusion block tables, so it only frees the producer's blocks and
does not affect the transfer. Use ``BFF_SCALE_MODE=raw`` so the transferred KV is byte-exact
(no per-block scales to ship). Decode-side fusion is a separate phase.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

_UNSET = object()   # sentinel for lazily-resolved cached values (e.g. the TP process group)

# Connector-level P/D fusion (plan ROUND 21): compute BFF dedup on the producer as KV streams
# through save_kv_layer (overhead overlaps the NCCL transfer), ship a per-request redirect map,
# and have the consumer share physical blocks on D (freeing the redundant copies). raw only.
_BFF_PD_FUSE = os.environ.get("BFF_PD_FUSE", "0") == "1"
_PD_REDIR_TAG = "#__bff_redir__#"          # side-channel tensor id suffix for the redirect map
_PD_LOG_EVERY = int(os.environ.get("BFF_PD_LOG_EVERY", "200"))
# Producer-side fuse summary (overhead + compression) cadence — SEPARATE from the decode consume
# log above. The producer is prefill-only, so it accrues far fewer group-completions than the
# decode accrues load calls; gating its summary at 200 means a short/prefill-bound run never logs
# it. Keep this low (and emit once at step 1) so the cumulative summary is always captured.
_PD_FUSE_LOG_EVERY = int(os.environ.get("BFF_PD_FUSE_LOG_EVERY", "50"))
# Producer dumps its cumulative fuse overhead + compression to a per-process JSON file here (always
# current — no log flooding, no scrape of throttled log lines). The shell reads bff_stats_*.json
# after the run. Off the decode path; the dump is a ~nanosecond dict build + a small atomic write.
_PD_STATS_DIR = os.environ.get("BFF_PD_STATS_DIR", ".")
# Verbose consumer trace: log every recv'd tensor_id so the LAST line before a hang is the
# unsent id `recv_tensor` (no-timeout) is blocked on. See plan ROUND 29.
_PD_DEBUG = os.environ.get("BFF_PD_DEBUG", "0") == "1"
# ROUND 50: per-(request, layer) hot-path send/recv tracing — SEPARATE from _PD_DEBUG. These
# synchronous INFO lines fire inside the transfer loop (28 layers × N reqs × every step), so
# enabling them under _PD_DEBUG throttles P enough to collapse prefill batches to ~1 req/step →
# no co-prefill peers → fusion silently stops. Gate them on their own flag so BFF_PD_DEBUG=1 keeps
# the cheap summary logs WITHOUT killing the batching fusion depends on.
_PD_TRACE = os.environ.get("BFF_PD_TRACE", "0") == "1"
# Connector within-batch clustering: nr_tree (butterfly, full precision) or cc. See ROUND 32.
_PD_MERGE = os.environ.get("BFF_PD_MERGE", "nr_tree")
# Per-layer block representation for the clustering similarity (producer-only). See ROUND 34.
#   full → exact cosine over the full flattened block-K; proj → JL projection; mean → head_dim mean.
_PD_REPR = os.environ.get("BFF_PD_REPR", "full")
_PD_PROJ_DIM = int(os.environ.get("BFF_PD_PROJ_DIM", "512"))
# ROUND 48: `ratio` scale mode for P/D. When BFF_SCALE_MODE=ratio the producer ALSO computes
# per-(redirect, layer) K/V norm ratios ‖owner‖/‖rep‖ and ships them as a float side-tensor
# (`_PD_RATIO_TAG`), co-located with the redirect map; D writes them into its norm buffers and the
# BFF Triton kernel scales the shared rep block per request. raw (default) ships no ratios.
_PD_SCALE_MODE = os.environ.get("BFF_SCALE_MODE", "raw").lower()
_PD_RATIO = _PD_SCALE_MODE == "ratio"
_PD_RATIO_TAG = "#__bff_ratio__#"          # side-channel suffix for the per-redirect K/V ratios
# ROUND 58: cross-batch fusion. When >0, the producer keeps a rolling registry of the last
# N REQUESTS' rep blocks per fusion group and matches each new prefill batch against it (not just
# the within-step batch), so a current block can redirect to a rep from an EARLIER batch (still
# resident + decoding on D). 0 = disabled → within-batch-only (today's behavior). The window should
# approximate the decode-resident request count; bigger N only raises reps_unresolved. proj repr
# recommended to bound registry memory (≈ N · blocks/req · G·D_repr floats per group).
_PD_ENCODED_BATCH = int(os.environ.get("BFF_PD_ENCODED_BATCH_SIZE", "0"))


def _pd_key(request_id: str) -> str:
    """Stable cross-P/D key for a request: vLLM v1 `InputProcessor.assign_request_id` sets
    `request_id = f"{external_req_id}-{random_uuid():.8}"` on EACH server, so the full id carries
    a per-server random 8-hex suffix that differs between P and D. The proxy gives both the same
    `external_req_id` (the X-Request-Id), so strip the trailing `-<random8>` (hex, no internal
    `-`) to recover it. Tensor ids / rep-hashes MUST use this, not the full request_id."""
    return request_id.rsplit("-", 1)[0]


def _rid_hash(request_id: str) -> int:
    """Process-stable positive int64 hash of a request id (Python's hash() is salted per
    process, so it can't be shared across P and D — use blake2b). Callers pass _pd_key(...)."""
    h = hashlib.blake2b(request_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFFFFFFFFFF


@dataclass
class ReqMetaFF:
    request_id: str
    # One block-id tensor PER KV-cache group (group 0 = warmup, 1..N = fusion).
    block_ids: list[torch.Tensor]
    num_tokens: int

    @staticmethod
    def make_meta(
        request_id: str,
        token_ids: list[int],
        block_ids_per_group: list[list[int]],
        block_size: int,
    ) -> "ReqMetaFF":
        return ReqMetaFF(
            request_id=request_id,
            block_ids=[torch.tensor(b) for b in block_ids_per_group],
            num_tokens=len(token_ids),
        )


class BFFMergeStats(KVConnectorStats):
    """ROUND 52: carries the D-side block-merge map worker→scheduler under TP>1.

    At TP>1 the worker and scheduler are SEPARATE processes, so the in-process
    `_ACTIVE_RUNNER._updated_block_tables` channel used at TP=1 isn't visible across them. This
    rides `KVConnectorOutput.kv_connector_stats` (the only serializable connector→scheduler slot):
    `data["bff_merges"] = {req_id: {group_idx: [block_ids]}}`. Every TP rank produces the identical
    (all-reduced) map, so the cross-rank `aggregate()` just keeps the accumulator's (rank-0's) copy."""

    def reset(self):
        self.data = {}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        # Ranks are coherent → identical maps; keep the accumulator's, never concatenate
        # (double-applying would corrupt ref counts).
        if not self.data.get("bff_merges") and getattr(other, "data", None):
            self.data = other.data
        return self

    def reduce(self) -> dict:
        return {"bff_merge_reqs": len(self.data.get("bff_merges", {}) or {})}

    def is_empty(self) -> bool:
        return not self.data.get("bff_merges")


@dataclass
class P2pNcclConnectorMetadataFF(KVConnectorMetadata):
    requests: list[ReqMetaFF]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[list[int]],  # per-group
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMetaFF.make_meta(request_id, token_ids, block_ids, block_size)
        )


class P2pNcclConnectorFF(P2pNcclConnector, SupportsHMA):
    """Group-aware P2P NCCL connector (see module docstring).

    Declares SupportsHMA because BFF uses a hybrid multi-group KV layout (warmup sliding-window
    group + fusion full-attention groups). Without it, the stock scheduler's `_connector_finished`
    asserts `len(kv_cache_groups) == 1` and crashes; SupportsHMA routes it to the per-group
    `request_finished_all_groups` instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._layer_group: dict[str, int] | None = None
        self._group_layers: dict[int, set[str]] = {}     # gi -> set of layer_names in group
        self._warned_layers: set[str] = set()
        # P/D connector-fusion state (producer side); only used when _BFF_PD_FUSE.
        self._pd_fuse = _BFF_PD_FUSE
        self._pd_buf: dict[int, dict] = {}               # gi -> partial group buffer this step
        self._pd_sent: set[int] = set()                  # fusion groups whose map was sent this step
        self._pd_cur_meta_id: int | None = None          # detects step boundary (reset buffers)
        self._pd_ms = 0.0                                 # accumulated fusion time (ms)
        self._pd_steps = 0
        # Cumulative compression accounting per fusion group (gi → totals over the run):
        # ratio = redirected(=freed) blocks / total fusable blocks. Logged periodically so the
        # shell can scrape the run-wide compression into the results JSON.
        self._pd_blk_total: dict[int, int] = {}
        self._pd_redir_total: dict[int, int] = {}
        # ROUND 58: cross-batch rolling registry, per fusion group. Each entry is a dict with the
        # registered rep blocks' raw concat vectors (this rank's head shard), their FULL squared
        # concat norm, stable (rep_hash, rep_slot), and (ratio) per-layer K/V norms; plus LRU
        # bookkeeping to evict whole oldest requests past _PD_ENCODED_BATCH. None until first use.
        self._pd_registry: dict[int, dict] = {}
        self._pd_cross_redir_total = 0   # redirects to a PREVIOUS-batch rep (the cross-batch lift)
        self._pd_within_redir_total = 0  # redirects to a same-batch rep (original behavior)
        # Consumer-side diagnostics (ROUND 27): is start_load_kv actually consuming? Logged
        # every _PD_LOG_EVERY load calls — reveals empty-metadata / id-mismatch / back-pressure.
        self._pd_load_calls = 0
        self._pd_load_reqs = 0
        self._pd_recv_layers = 0
        self._pd_freed_layers = 0
        self._pd_waiting = None   # tensor_id currently blocked on in recv_tensor (hang trace)
        self._pd_jl = None        # lazy fixed-seed JL matrix for BFF_PD_REPR=proj (producer only)
        self._pd_tp = _UNSET      # lazy TP process group (None at TP=1); see _pd_tp_group (ROUND 52)
        self._pd_pending_merges = None   # this-step block-merge map; emitted via stats under TP>1
        if self._pd_fuse:
            logger.info("P2pNcclConnectorFF: BFF_PD_FUSE enabled (connector-level fusion).")
        # One-time identity log: role/rank/port — catches a peer-address/port mismatch.
        try:
            logger.info(
                "P2pNcclConnectorFF init | is_producer=%s | rank=%s | "
                "engine=%s | pd_fuse=%s | pd_debug=%s",
                getattr(self, "is_producer", "?"), getattr(self, "_rank", "?"),
                self.p2p_nccl_engine is not None, self._pd_fuse, _PD_DEBUG)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # layer_name -> kv-cache group index (from the live BFF group layout)
    # ------------------------------------------------------------------
    def _build_group_maps(self) -> dict[str, int]:
        m: dict[str, int] = {}
        try:
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            runner = getattr(_bp, "_ACTIVE_RUNNER", None)
            if runner is not None:
                for gi, g in enumerate(runner.kv_cache_config.kv_cache_groups):
                    self._group_layers[gi] = set(g.layer_names)
                    for ln in g.layer_names:
                        m[ln] = gi
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("FF connector: could not build layer→group map: %s", e)
        return m

    def _pd_tp_group(self):
        """ROUND 52: the tensor-parallel torch.distributed process group when TP>1, else None.

        Under TP>1 each rank holds only a head SHARD of K/V, so the producer's per-shard cosine /
        norms are partial; the clustering + ratio norms all-reduce over THIS group to reconstruct
        the full-vector statistics (identical decision on every rank → coherent block table).
        Returns None at TP=1 (or if distributed isn't initialized) → the original single-GPU path.
        Cached (resolved once)."""
        if self._pd_tp is _UNSET:
            grp = None
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    from vllm.distributed.parallel_state import get_tp_group
                    tp = get_tp_group()
                    if tp.world_size > 1:
                        grp = tp.device_group
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("FF connector: TP group lookup failed (assuming TP=1): %s", e)
            self._pd_tp = grp
            if grp is not None:
                logger.info("P2pNcclConnectorFF: TP>1 detected → all-reduced fusion decision.")
        return self._pd_tp

    def get_kv_connector_stats(self):
        """ROUND 52: under TP>1, emit this step's D-side block-merge map via the connector stats
        carrier so it reaches the (separate-process) scheduler; the worker runs this after the
        forward, when `_pd_pending_merges` holds the map set in `_pd_consumer_apply`. At TP=1 the
        in-process `_ACTIVE_RUNNER` channel is used instead, so fall through to the base (None)."""
        merges = self._pd_pending_merges
        self._pd_pending_merges = None
        if merges and self._pd_tp_group() is not None:
            return BFFMergeStats(data={"bff_merges": merges})
        return super().get_kv_connector_stats()

    def _remote_addr_or_none(self, request_id: str, is_prefill: bool) -> str | None:
        """Resolve the peer NCCL address from the request id. The disagg PROXY injects the
        ``___{prefill,decode}_addr_HOST:PORT`` suffix; a request hitting P/D directly (no proxy)
        lacks it. Return None + warn-once instead of letting ValueError kill the EngineCore — so
        a misrouted request becomes a transfer no-op and a standalone P run can still measure
        the connector fusion overhead."""
        try:
            ip, port = self.parse_request_id(request_id, is_prefill)
        except ValueError:
            if "__noaddr__" not in self._warned_layers:
                self._warned_layers.add("__noaddr__")
                logger.warning(
                    "FF connector: request id %r has no peer address (not routed via the "
                    "disagg proxy) — skipping KV transfer. Point the client at the proxy for "
                    "actual P/D, or ignore for a standalone overhead run.", request_id)
            return None
        return ip + ":" + str(port + self._rank)

    def _group_of(self, layer_name: str) -> int:
        m = self._layer_group
        if m is None:
            m = self._build_group_maps()
            self._layer_group = m
        gi = m.get(layer_name)
        if gi is None:
            # Robustness: keys may carry/drop a trailing ".attn"; try both forms.
            gi = m.get(layer_name + ".attn")
            if gi is None and layer_name.endswith(".attn"):
                gi = m.get(layer_name[: -len(".attn")])
        if gi is None:
            if layer_name not in self._warned_layers:
                self._warned_layers.add(layer_name)
                logger.warning(
                    "FF connector: layer %s not found in any KV-cache group; "
                    "falling back to group 0 (block ids may be wrong).", layer_name)
            return 0
        return gi

    # ------------------------------------------------------------------
    # Worker-side: group-aware save (producer) / load (consumer)
    # ------------------------------------------------------------------
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self.is_producer:
            return
        assert self.p2p_nccl_engine is not None
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        def inject_kv_into_layer(layer, kv_cache, block_ids, request_id):
            if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
                num_block = kv_cache.shape[0]
                self.check_tensors_except_dim(layer, kv_cache, 0)
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    layer[block_ids[:num_block], ...] = kv_cache
                    logger.warning("🚧kv_cache does not match, block_ids:%d, "
                                   "num_block:%d, request_id:%s",
                                   len(block_ids), num_block, request_id)
            elif layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                self.check_tensors_except_dim(layer, kv_cache, 1)
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache
                    logger.warning("🚧kv_cache does not match, block_ids:%d, "
                                   "num_block:%d, request_id:%s",
                                   len(block_ids), num_block, request_id)

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadataFF)
        if metadata is None:
            return

        self._pd_load_calls += 1
        self._pd_load_reqs += len(metadata.requests)
        if _PD_DEBUG or self._pd_load_calls <= 5:
            logger.info("BFF P/D consume ENTER | call=%d | reqs=%d | layers=%d",
                        self._pd_load_calls, len(metadata.requests),
                        len(forward_context.no_compile_layers))
        for request in metadata.requests:
            request_id = request.request_id
            remote_address = self._remote_addr_or_none(request_id, False)
            if remote_address is None:
                continue
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache = getattr(layer, "kv_cache", None)
                if kv_cache is None:
                    continue
                layer = kv_cache[forward_context.virtual_engine]
                tid = _pd_key(request.request_id) + "#" + layer_name
                # Record what we're about to (possibly indefinitely) block on. The LAST value
                # logged before a hang is the tensor_id the producer never sent.
                self._pd_waiting = tid
                if _PD_TRACE:
                    logger.info("BFF P/D consume: recv KV %s", tid)
                kv_cache = self.p2p_nccl_engine.recv_tensor(tid, remote_address)
                self._pd_waiting = None
                if kv_cache is None:
                    logger.warning("🚧kv_cache is None, %s", request.request_id)
                    continue
                self._pd_recv_layers += 1
                # group-aware: index this layer by ITS group's block table
                block_ids = request.block_ids[self._group_of(layer_name)]
                inject_kv_into_layer(layer, kv_cache, block_ids, request.request_id)
                # KV is now in D's GPU cache → release the recv buffer immediately (frees the
                # pinned-pool block if it spilled) instead of waiting for request completion.
                self.p2p_nccl_engine.free_recv_tensor(tid)
                self._pd_freed_layers += 1

        if self._pd_load_calls % _PD_LOG_EVERY == 0:
            try:
                eng = self.p2p_nccl_engine
                pool_blocks = len(getattr(eng.pool, "allocated_blocks", {}))
                logger.info(
                    "BFF P/D consume | calls=%d | reqs(cum)=%d | recv_layers(cum)=%d | "
                    "freed_layers(cum)=%d | pool_allocated=%d | buffer_size=%d",
                    self._pd_load_calls, self._pd_load_reqs, self._pd_recv_layers,
                    self._pd_freed_layers, pool_blocks, getattr(eng, "buffer_size", -1))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF P/D consume log failed: %s", e)

        # Connector-level fusion: receive the producer's per-group redirect maps and apply
        # them on D — point owner block-table entries at the representative's D-physical block
        # and free the redundant copies via the existing BFF merge channel.
        if self._pd_fuse:
            self._pd_consumer_apply(metadata)

    def _pd_consumer_apply(self, metadata) -> None:
        """Apply the producer's redirect maps on D (share blocks + free redundant copies).

        UNVALIDATED on a live P/D topology — fully guarded: any structural mismatch logs and
        leaves D correct (per-request copies, no sharing) instead of crashing the load."""
        self._pd_pending_merges = None   # reset per step (no stale map under TP>1 stats path)
        try:
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            runner = getattr(_bp, "_ACTIVE_RUNNER", None)
            if runner is None:
                return
            # rep-request resolution + D block ids per request. A redirect's representative may
            # have loaded in an EARLIER step, so resolve against ALL running requests (runner
            # state), not just this step's metadata. Key the hash on the STABLE id (rep_hash was
            # computed on P over the stable id).
            hash2rid: dict[int, str] = {}
            rid2blocks: dict[str, Any] = {}
            for rid_r, st in getattr(runner, "requests", {}).items():
                bids = getattr(st, "block_ids", None)
                if bids is not None:
                    hash2rid[_rid_hash(_pd_key(rid_r))] = rid_r
                    rid2blocks[rid_r] = bids
            # Overlay this step's metadata (authoritative per-group tensors for loading reqs).
            for r in metadata.requests:
                hash2rid[_rid_hash(_pd_key(r.request_id))] = r.request_id
                rid2blocks[r.request_id] = r.block_ids
            fusion_groups = [gi for gi in self._group_layers if gi > 0]
            updated: dict[str, dict[int, list[int]]] = {}
            n_applied = 0
            n_unresolved = 0
            if _PD_DEBUG:
                logger.info("BFF P/D apply ENTER | reqs=%d | running=%d | fusion_groups=%s",
                            len(metadata.requests), len(rid2blocks), fusion_groups)

            for request in metadata.requests:
                rid = request.request_id
                remote_address = self._remote_addr_or_none(rid, False)
                if remote_address is None:
                    continue
                for gi in fusion_groups:
                    map_tid = _pd_key(rid) + _PD_REDIR_TAG + str(gi)
                    self._pd_waiting = map_tid
                    if _PD_TRACE:
                        logger.info("BFF P/D apply: recv map %s", map_tid)
                    payload = self.p2p_nccl_engine.recv_tensor(map_tid, remote_address)
                    self._pd_waiting = None
                    # Map ids never match the engine's get_finished cleanup pattern, so release
                    # the recv buffer here (frees the pinned block if it spilled + the dict entry).
                    self.p2p_nccl_engine.free_recv_tensor(map_tid)
                    # ratio mode: the producer always co-sends a [num_rows, G, 2] K/V ratio
                    # side-tensor (same row order as the map) — recv it here so the blocking
                    # recv never deadlocks, even when payload is a sentinel.
                    rmat = None
                    ratio_layers = None
                    if _PD_RATIO:
                        ratio_tid = _pd_key(rid) + _PD_RATIO_TAG + str(gi)
                        self._pd_waiting = ratio_tid
                        rmat = self.p2p_nccl_engine.recv_tensor(ratio_tid, remote_address)
                        self._pd_waiting = None
                        self.p2p_nccl_engine.free_recv_tensor(ratio_tid)
                        ratio_layers = sorted(
                            self._group_layers[gi], key=lambda ln: int(ln.split('.')[2]))
                    if payload is None or payload.numel() == 0:
                        continue
                    owner_blocks = list(request.block_ids[gi].tolist())
                    nb = len(owner_blocks)
                    changed = False
                    for r_i, (owner_slot, rep_hash, rep_slot) in enumerate(payload.tolist()):
                        if owner_slot < 0:        # sentinel row → nothing to free for this group
                            continue
                        rep_rid = hash2rid.get(int(rep_hash))
                        if rep_rid is None or rep_rid not in rid2blocks:
                            n_unresolved += 1     # rep not (yet) resident on D → can't share
                            continue
                        rep_grp = rid2blocks[rep_rid][gi]
                        if not (0 <= rep_slot < len(rep_grp)
                                and 0 <= owner_slot < len(owner_blocks)):
                            n_unresolved += 1
                            continue
                        owner_blocks[owner_slot] = int(rep_grp[rep_slot])
                        changed = True
                        n_applied += 1
                        # ratio: stash this owner block's per-layer K/V scale (‖own‖/‖rep‖) into
                        # runner.fused_requests so _fill_norm_buffers slot-fills it for the kernel.
                        if _PD_RATIO and rmat is not None and r_i < rmat.shape[0]:
                            self._pd_store_ratio(
                                runner, rid, ratio_layers, owner_slot, nb, rmat[r_i])
                    if changed:
                        updated.setdefault(rid, {})[gi] = owner_blocks
                        self._pd_write_runner_block_table(runner, rid, gi, owner_blocks)

            if updated:
                # Stage for D's scheduler to free the orphaned blocks + fix ref-counts
                # (reuses the BFF merge channel: _updated_block_tables → update_from_output).
                runner._updated_block_tables = updated
                # TP>1: scheduler is a different process → also emit the map via the connector
                # stats carrier (get_kv_connector_stats). ROUND 52.
                self._pd_pending_merges = updated
            if n_applied or n_unresolved or _PD_DEBUG:
                logger.info("BFF P/D apply | redirects_applied=%d | reps_unresolved=%d",
                            n_applied, n_unresolved)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("BFF P/D consumer apply failed: %s", e)

    @staticmethod
    def _pd_store_ratio(runner, rid, layers, owner_slot, nb, rrow) -> None:
        """ratio mode (ROUND 48): record this owner block's per-layer K/V scale ‖own‖/‖rep‖ into
        runner.fused_requests[rid][layer_name] = (nk_vec, nv_vec) ([nb], default 1.0). The single-
        instance `_fill_norm_buffers` then slot-fills these into norms_*_buf and the BFF Triton
        kernel scales the shared rep block per request. `rrow` is the redirect's [G, 2] ratio row
        (col g ↔ layers[g] in sorted absolute-index order; [:,0]=K, [:,1]=V), on D's device."""
        fr = runner.fused_requests.setdefault(rid, {})
        for g_i, ln in enumerate(layers):
            entry = fr.get(ln)
            if entry is None:
                nk = torch.ones(nb, dtype=torch.float32, device=rrow.device)
                nv = torch.ones(nb, dtype=torch.float32, device=rrow.device)
                fr[ln] = (nk, nv)
            else:
                nk, nv = entry
            if 0 <= owner_slot < nk.shape[0]:
                nk[owner_slot] = rrow[g_i, 0]
                nv[owner_slot] = rrow[g_i, 1]

    @staticmethod
    def _pd_write_runner_block_table(runner, rid, gi, new_blocks) -> None:
        """Write the redirected per-group block table into the runner's worker-side mirror so
        the forward reads the shared blocks. Guarded; mirrors _update_block_tables_after_compression."""
        ridx = runner.input_batch.req_id_to_index.get(rid)
        if ridx is None:
            return
        bt_obj = runner.input_batch.block_table.block_tables[gi]
        n = min(len(new_blocks), int(bt_obj.num_blocks_per_row[ridx]))
        row = new_blocks[:n]
        bt_obj.block_table.np[ridx, :n] = row
        bt_obj.block_table.gpu[ridx, :n] = torch.tensor(
            row, device=bt_obj.block_table.gpu.device,
            dtype=bt_obj.block_table.gpu.dtype)
        st = runner.requests.get(rid)
        if st is not None and gi < len(st.block_ids):
            st.block_ids[gi][:n] = row

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self.is_producer:
            return
        assert self.p2p_nccl_engine is not None

        def extract_kv_from_layer(layer, block_ids):
            if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
                return layer[block_ids, ...]
            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]
            return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadataFF)
        gi = self._group_of(layer_name)
        if _PD_TRACE:
            logger.info("BFF P/D save ENTER | layer=%s | gi=%d | reqs=%d",
                        layer_name, gi, len(connector_metadata.requests))
        for request in connector_metadata.requests:
            request_id = request.request_id
            remote_address = self._remote_addr_or_none(request_id, True)
            if remote_address is None:
                continue
            block_ids = request.block_ids[gi]            # this layer's group
            kv_cache = extract_kv_from_layer(kv_layer, block_ids)
            tid = _pd_key(request_id) + "#" + layer_name
            if _PD_TRACE:
                logger.info("BFF P/D send KV %s -> %s | shape=%s",
                            tid, remote_address, tuple(kv_cache.shape))
            self.p2p_nccl_engine.send_tensor(tid, kv_cache, remote_address)

        # Connector-level fusion: accumulate this fusion group's per-layer K reps; when the
        # group's last layer is seen, cluster (concat cosine) and ship the per-request redirect
        # map. KV data is still sent per-layer above (no wire-dedup yet); D applies the map to
        # share physical blocks and free the redundant copies.
        if self._pd_fuse and gi > 0:
            self._pd_producer_accumulate(gi, layer_name, kv_layer, connector_metadata)

    # ------------------------------------------------------------------
    # Producer P/D fusion: buffer a group's layers, decide at completion, ship redirect map.
    # NOTE: the producer math reuses the tested pd_fuse core; the NCCL send pairing and the
    # consumer apply (below) require validation on the live P/D topology.
    # ------------------------------------------------------------------
    def _pd_block_repr(self, kv_layer, idx):
        """Per-layer block representation [N, D_repr] (float32) for the clustering similarity,
        selected by BFF_PD_REPR. K-only (kv_layer[0]); the concatenation cosine over the G group
        layers is applied by the clustering. `full` = exact (whole block), `mean` = head_dim mean,
        `proj` = fixed-seed JL projection (cosine-preserving, cheaper). Producer-only."""
        blk = kv_layer[0, idx].float()                     # [N, block_sz, kv_heads, head_dim]
        N = idx.shape[0]
        if _PD_REPR == "mean":
            head_dim = blk.shape[-1]
            return blk.reshape(N, -1, head_dim).mean(dim=1)
        full = blk.reshape(N, -1)                           # [N, D_full_layer]
        if _PD_REPR == "proj":
            if self._pd_jl is None:
                g = torch.Generator(device=full.device)
                g.manual_seed(1234)
                self._pd_jl = torch.randn(
                    full.shape[1], _PD_PROJ_DIM,
                    generator=g, device=full.device, dtype=torch.float32)
            return full @ self._pd_jl
        return full                                         # full

    def _pd_producer_accumulate(self, gi, layer_name, kv_layer, meta):
        # New step → fresh metadata object → reset all partial group buffers + sent-tracking.
        if id(meta) != self._pd_cur_meta_id:
            self._pd_cur_meta_id = id(meta)
            self._pd_buf.clear()
            self._pd_sent.clear()

        group_layer_set = self._group_layers.get(gi)
        if not group_layer_set:
            return
        buf = self._pd_buf.get(gi)
        if buf is None:
            # Build the flat block structure ONCE for this group/step (same across its layers):
            # one entry per real (>0) block, tagged with owner request + slot.
            flat_bids, flat_req_local, flat_slot = [], [], []
            req_ids: list[str] = []
            for ri, request in enumerate(meta.requests):
                req_ids.append(request.request_id)
                bids = request.block_ids[gi].tolist()
                for slot, bid in enumerate(bids):
                    if bid > 0:                              # skip null block 0
                        flat_bids.append(bid)
                        flat_req_local.append(ri)
                        flat_slot.append(slot)
            buf = {
                "seen": set(),
                "k_layers": [],
                "flat_bids": flat_bids,
                "flat_req_local": flat_req_local,
                "flat_slot": flat_slot,
                "req_ids": req_ids,
                # ratio mode only: layer_name -> [N] per-flat-block K / V norms.
                "k_norms": {},
                "v_norms": {},
            }
            self._pd_buf[gi] = buf

        if buf["flat_bids"]:
            idx = torch.as_tensor(buf["flat_bids"], device=kv_layer.device, dtype=torch.long)
            # Per-layer block repr (full|proj|mean) → [N, D_repr] for the concat-cosine.
            buf["k_layers"].append(self._pd_block_repr(kv_layer, idx))
            if _PD_RATIO:
                # Per-flat-block K and V L2 norms for this layer (FlashAttention layout
                # kv_layer[0]=K, [1]=V). Shipped as ‖owner‖/‖rep‖ ratios per redirect.
                N = idx.shape[0]
                ksq = kv_layer[0, idx].float().reshape(N, -1).pow(2).sum(dim=1)
                vsq = kv_layer[1, idx].float().reshape(N, -1).pow(2).sum(dim=1)
                tp_group = self._pd_tp_group()
                if tp_group is not None:
                    # TP>1: this rank holds a head shard → all-reduce the SQUARED norms to get the
                    # full-vector norm, so the shipped ratio is ‖own‖_full/‖rep‖_full (each rank
                    # then applies the same global scalar to its shard). ROUND 52.
                    import torch.distributed as dist
                    dist.all_reduce(ksq, op=dist.ReduceOp.SUM, group=tp_group)
                    dist.all_reduce(vsq, op=dist.ReduceOp.SUM, group=tp_group)
                buf["k_norms"][layer_name] = ksq.sqrt()
                buf["v_norms"][layer_name] = vsq.sqrt()
        buf["seen"].add(layer_name)

        # Count-based completion (robust to layer-name `.attn` variance): this layer was routed to
        # gi by _group_of, so it's a member; complete once we've seen all of the group's layers.
        if len(buf["seen"]) < len(group_layer_set):
            return  # group not complete yet
        # --- group complete: cluster (within-batch + cross-batch registry) + ship redirect map ---
        try:
            from kv_fast_fusion.kv_fast_fusion_graph_runner import THRESHOLD
            dev = kv_layer.device
            req_ids = buf["req_ids"]
            # ratio mode: layers in sorted absolute-index order — the P↔D column ordering
            # invariant for the [num_rows, G, 2] ratio side-tensor (col g ↔ this layer).
            ratio_layers = (
                sorted(group_layer_set, key=lambda ln: int(ln.split('.')[2]))
                if _PD_RATIO else [])
            Gn = len(ratio_layers)
            tp_group = self._pd_tp_group()

            t0 = time.perf_counter()
            # Unified rows: cross-batch (registry) matches first, then within-batch clustering on the
            # remainder; new reps registered. send_rows[owner_ri] = [(owner_slot, rep_hash, rep_slot,
            # own_flat, rep_kind, rep_ref), ...]. registry disabled (_PD_ENCODED_BATCH<=0) → within-only.
            send_rows, n_cross, n_within = self._pd_build_send_rows(
                gi, buf, dev, tp_group, THRESHOLD, ratio_layers, req_ids)
            self._pd_ms += (time.perf_counter() - t0) * 1000.0
            self._pd_steps += 1

            n_redir = n_cross + n_within
            # Cumulative compression accounting (all steps), per fusion group.
            self._pd_blk_total[gi] = self._pd_blk_total.get(gi, 0) + len(buf["flat_bids"])
            self._pd_redir_total[gi] = self._pd_redir_total.get(gi, 0) + n_redir
            self._pd_cross_redir_total += n_cross
            self._pd_within_redir_total += n_within
            if n_redir or _PD_DEBUG:
                logger.info(
                    "BFF P/D fuse group gi=%d | merge=%s | repr=%s | reqs=%d | blocks=%d | "
                    "redirects=%d (cross=%d within=%d) | reg_blocks=%d",
                    gi, _PD_MERGE, _PD_REPR, len(req_ids), len(buf["flat_bids"]),
                    n_redir, n_cross, n_within, self._pd_registry_size(gi))

            # Ship a redirect-map tensor per request for this group, co-located with the group's
            # last KV layer (just sent above). ALWAYS send (one per request per group) so the
            # consumer's blocking recv_tensor never deadlocks; a 1-row SENTINEL [[-1,-1,-1]] means
            # "nothing to free → continue as usual". GPU + non-empty (NCCL can't ship 0 elements).
            for ri, request_id in enumerate(req_ids):
                rows = send_rows.get(ri, [])
                remote_address = self._remote_addr_or_none(request_id, True)
                if remote_address is None:
                    continue
                if rows:
                    # [num_rows, 3] int64: (owner_slot, rep_request_hash, rep_slot). rep_hash is the
                    # STABLE-key hash (resolves on D whose full ids differ by random8) — for both a
                    # within-batch rep and a registry (earlier-batch) rep, identically.
                    data = [[r[0], r[1], r[2]] for r in rows]
                    payload = torch.tensor(data, dtype=torch.int64, device=dev)
                else:
                    payload = torch.tensor([[-1, -1, -1]], dtype=torch.int64, device=dev)
                self.p2p_nccl_engine.send_tensor(
                    _pd_key(request_id) + _PD_REDIR_TAG + str(gi), payload, remote_address)
                if _PD_RATIO:
                    # Per-redirect, per-layer ‖owner‖/‖rep‖ K/V ratios, SAME row order as the int map;
                    # [:, g, 0]=K, [:, g, 1]=V. Rep norms come from the current buffer (within-batch
                    # rep) or the registry (cross-batch rep). Sentinel (1.0) when no redirects.
                    rmat = (self._pd_ratio_rows(gi, buf, rows, ratio_layers, dev) if rows
                            else torch.ones((1, Gn, 2), dtype=torch.float32, device=dev))
                    self.p2p_nccl_engine.send_tensor(
                        _pd_key(request_id) + _PD_RATIO_TAG + str(gi), rmat, remote_address)
            self._pd_sent.add(gi)

            # Persist the cumulative overhead + compression to a per-process JSON file (always the
            # latest totals — the shell reads it post-run, no log scrape). Cheap: a dict build + a
            # small atomic write, gated to a low cadence so it stays off the hot path.
            if self._pd_steps and (self._pd_steps % _PD_FUSE_LOG_EVERY == 0
                                    or self._pd_steps == 1):
                self._pd_dump_fuse_stats()
        except Exception as e:  # pragma: no cover - defensive (do not break the transfer)
            logger.warning("BFF P/D producer fusion failed (group %d): %s", gi, e)
        finally:
            self._pd_buf.pop(gi, None)

    # ------------------------------------------------------------------
    # ROUND 58: cross-batch fusion — rolling per-group rep registry.
    # ------------------------------------------------------------------
    def _pd_registry_size(self, gi) -> int:
        reg = self._pd_registry.get(gi)
        return 0 if reg is None or reg["vecs"] is None else int(reg["vecs"].shape[0])

    def _pd_build_send_rows(self, gi, buf, dev, tp_group, threshold, ratio_layers, req_ids):
        """Build the unified per-owner redirect rows for this group and update the registry.

        Returns ``(send_rows, n_cross, n_within)`` where
        ``send_rows[owner_ri] = [(owner_slot, rep_hash, rep_slot, own_flat, rep_kind, rep_ref), ...]``
        with ``rep_kind`` ∈ {"cur" (within-batch rep, ``rep_ref``=current flat idx),
        "reg" (registry/earlier-batch rep, ``rep_ref``=registry row)}. When ``_PD_ENCODED_BATCH<=0``
        the registry is skipped → within-batch-only (identical to the pre-ROUND-58 path)."""
        from kv_fast_fusion.pd_fuse import (
            concat_cosine_cc_labels, concat_cosine_nr_tree_labels,
            build_group_redirect, concat_cosine_cross_match)
        send_rows: dict[int, list] = {}
        n_cross = n_within = 0
        if not buf["flat_bids"]:
            return send_rows, n_cross, n_within
        flat_req_local = buf["flat_req_local"]
        flat_slot = buf["flat_slot"]
        N = len(flat_req_local)
        dev0 = buf["k_layers"][0].device

        def _cluster(k_layers, req_of_block):
            if tp_group is not None:
                # TP>1: only CC exposes the raw Gram/sq for the cross-rank all-reduce (nr_tree
                # normalizes before similarity). ROUND 52.
                return concat_cosine_cc_labels(k_layers, req_of_block, threshold, tp_group=tp_group)
            cluster = (concat_cosine_nr_tree_labels if _PD_MERGE == "nr_tree"
                       else concat_cosine_cc_labels)
            return cluster(k_layers, req_of_block, threshold)

        # ---- registry disabled → original within-batch-only path ----
        if _PD_ENCODED_BATCH <= 0:
            labels = _cluster(buf["k_layers"], torch.as_tensor(flat_req_local, device=dev0))
            _, redirects = build_group_redirect(labels, flat_req_local, flat_slot)
            for owner_ri, rws in redirects.items():
                for (slot, rep_local, rep_slot, rep_flat, own_flat) in rws:
                    send_rows.setdefault(owner_ri, []).append(
                        (slot, _rid_hash(_pd_key(req_ids[rep_local])), rep_slot,
                         own_flat, "cur", rep_flat, None))
                    n_within += 1
            return send_rows, n_cross, n_within

        # ---- cross-batch (registry enabled) ----
        reg = self._pd_registry.get(gi)
        reg_vecs = reg["vecs"] if reg else None
        reg_sq = reg["sq"] if reg else None
        best_idx, _score, cur_sq, cur_concat = concat_cosine_cross_match(
            buf["k_layers"], reg_vecs, reg_sq, threshold, tp_group=tp_group)
        # forbid a self-merge (a registered rep from the SAME request, e.g. chunked re-register).
        if reg is not None and bool((best_idx >= 0).any()):
            own_hash = torch.tensor(
                [_rid_hash(_pd_key(req_ids[r])) for r in flat_req_local],
                dtype=torch.long, device=best_idx.device)
            self_hit = (best_idx >= 0) & (reg["hash"][best_idx.clamp(min=0)] == own_hash)
            best_idx = torch.where(self_hit, torch.full_like(best_idx, -1), best_idx)
        best_list = best_idx.tolist()

        # Phase 1: cross-batch matches → redirect to the registry rep (already resident on D).
        # Resolve EVERYTHING the rep contributes to VALUES now (hash, slot, and — for ratio — its
        # per-layer K/V norms), because _pd_register_reps below mutates/re-indexes the registry in
        # this same call; a stored row index would be stale at serialize time (ROUND 59 bug A).
        ratio_reg = _PD_RATIO and reg is not None and reg.get("knorm") is not None
        matched = [False] * N
        for i, ridx in enumerate(best_list):
            if ridx < 0:
                continue
            rep_norms = ((reg["knorm"][ridx].clone(), reg["vnorm"][ridx].clone())
                         if ratio_reg else None)
            send_rows.setdefault(flat_req_local[i], []).append(
                (flat_slot[i], int(reg["hash"][ridx].item()), int(reg["slot"][ridx].item()),
                 i, "reg", ridx, rep_norms))
            matched[i] = True
            n_cross += 1

        # Phase 2: within-batch clustering on the UNMATCHED current blocks (subset → map back).
        unmatched = [i for i in range(N) if not matched[i]]
        reps_to_register = []
        if unmatched:
            sub_k = [Kg[unmatched] for Kg in buf["k_layers"]]
            sub_req = [flat_req_local[i] for i in unmatched]
            sub_slot = [flat_slot[i] for i in unmatched]
            labels = _cluster(sub_k, torch.as_tensor(sub_req, device=dev0))
            _, redirects = build_group_redirect(labels, sub_req, sub_slot)
            for owner_ri, rws in redirects.items():
                for (slot, rep_local, rep_slot, rep_flat_sub, own_flat_sub) in rws:
                    send_rows.setdefault(owner_ri, []).append(
                        (slot, _rid_hash(_pd_key(req_ids[rep_local])), rep_slot,
                         unmatched[own_flat_sub], "cur", unmatched[rep_flat_sub], None))
                    n_within += 1
            labels_l = labels.tolist()
            reps_to_register = [unmatched[i] for i in range(len(labels_l)) if labels_l[i] == i]

        self._pd_register_reps(gi, buf, reps_to_register, cur_concat, cur_sq, ratio_layers, req_ids)
        return send_rows, n_cross, n_within

    def _pd_register_reps(self, gi, buf, rep_flats, cur_concat, cur_sq, ratio_layers, req_ids):
        """Append this step's new rep blocks to the group registry, then LRU-evict to the window.
        Only reps for requests with a remote address (i.e. actually loaded on a D) are registered."""
        if not rep_flats:
            return
        flat_req_local = buf["flat_req_local"]
        flat_slot = buf["flat_slot"]
        dev = cur_concat.device
        reg = self._pd_registry.get(gi)
        if reg is None:
            reg = {"vecs": None, "sq": None, "hash": None, "slot": None, "seq": None,
                   "knorm": None, "vnorm": None, "key2seq": {}, "next_seq": 0}
            self._pd_registry[gi] = reg
        v, sq, hsh, slt, seq, kn, vn = [], [], [], [], [], [], []
        for f in rep_flats:
            request_id = req_ids[flat_req_local[f]]
            if self._remote_addr_or_none(request_id, True) is None:
                continue
            key = _pd_key(request_id)
            s = reg["key2seq"].get(key)
            if s is None:
                s = reg["next_seq"]; reg["key2seq"][key] = s; reg["next_seq"] = s + 1
            v.append(cur_concat[f]); sq.append(cur_sq[f])
            hsh.append(_rid_hash(key)); slt.append(flat_slot[f]); seq.append(s)
            if _PD_RATIO:
                kn.append(torch.stack([buf["k_norms"][ln][f] for ln in ratio_layers]))
                vn.append(torch.stack([buf["v_norms"][ln][f] for ln in ratio_layers]))
        if not v:
            return

        def _cat(old, new):
            return new if old is None else torch.cat([old, new])
        reg["vecs"] = _cat(reg["vecs"], torch.stack(v))
        reg["sq"] = _cat(reg["sq"], torch.stack(sq))
        reg["hash"] = _cat(reg["hash"], torch.tensor(hsh, dtype=torch.long, device=dev))
        reg["slot"] = _cat(reg["slot"], torch.tensor(slt, dtype=torch.long, device=dev))
        reg["seq"] = _cat(reg["seq"], torch.tensor(seq, dtype=torch.long, device=dev))
        if _PD_RATIO:
            reg["knorm"] = _cat(reg["knorm"], torch.stack(kn))
            reg["vnorm"] = _cat(reg["vnorm"], torch.stack(vn))
        self._pd_evict_registry(gi)

    def _pd_evict_registry(self, gi):
        """Drop rows from requests older than the last _PD_ENCODED_BATCH distinct requests. Seq ids
        are dense + monotonic, so keeping seq >= next_seq - N keeps exactly the last N requests."""
        reg = self._pd_registry.get(gi)
        if reg is None or reg["seq"] is None:
            return
        keep_from = reg["next_seq"] - _PD_ENCODED_BATCH
        if keep_from <= 0:
            return
        keep = reg["seq"] >= keep_from
        if bool(keep.all()):
            return
        idx = keep.nonzero(as_tuple=True)[0]
        for k in ("vecs", "sq", "hash", "slot", "seq", "knorm", "vnorm"):
            if reg[k] is not None:
                reg[k] = reg[k][idx]
        reg["key2seq"] = {k: s for k, s in reg["key2seq"].items() if s >= keep_from}

    def _pd_ratio_rows(self, gi, buf, rows, ratio_layers, dev):
        """[num_rows, G, 2] ‖own‖/‖rep‖ K/V ratios for `rows`; rep norms from the current buffer
        (within-batch "cur" rep) or the rep's per-layer norms CAPTURED at match time ("reg" rep —
        the registry may have been re-indexed since, so never index it here; ROUND 59). Few rows →
        a plain loop is fine."""
        rmat = torch.ones((len(rows), len(ratio_layers), 2), dtype=torch.float32, device=dev)
        for r_i, (_oslot, _rhash, _rslot, own_flat, rep_kind, rep_ref, rep_norms) in enumerate(rows):
            for g, ln in enumerate(ratio_layers):
                nk = buf["k_norms"][ln]; nv = buf["v_norms"][ln]
                if rep_kind == "cur":
                    rep_k = nk[rep_ref]; rep_v = nv[rep_ref]
                elif rep_norms is not None:
                    rep_k = rep_norms[0][g]; rep_v = rep_norms[1][g]
                else:
                    continue   # cross-batch rep without captured norms → leave ratio 1.0
                rmat[r_i, g, 0] = nk[own_flat] / rep_k.clamp(min=1e-6)
                rmat[r_i, g, 1] = nv[own_flat] / rep_v.clamp(min=1e-6)
        return rmat

    def _pd_dump_fuse_stats(self) -> None:
        """Write this producer's cumulative fuse overhead + compression to a per-process JSON file
        (``bff_stats_<pid>.json`` in ``BFF_PD_STATS_DIR``). The shell reads + merges these after the
        run — replaces the old periodic-log + scrape (which a prefill-only producer rarely emitted).
        Compression FACTOR = total / (total - freed) — how many× smaller the KV cache gets from
        fusion (>1; 2.0 = half the blocks); block-weighted overall + per-group."""
        try:
            def _factor(b, r):
                return b / max(1, b - r)
            tot_b = sum(self._pd_blk_total.values())
            tot_r = sum(self._pd_redir_total.values())
            stats = {
                "pid": os.getpid(),
                "is_producer": bool(getattr(self, "is_producer", False)),
                "steps": self._pd_steps,
                "overhead_avg_group_dedup_ms": (self._pd_ms / self._pd_steps
                                                if self._pd_steps else 0.0),
                "total_blocks": tot_b,
                "freed": tot_r,
                "compression_avg_factor": _factor(tot_b, tot_r),
                "compression_per_group": {
                    str(gi): _factor(self._pd_blk_total[gi], self._pd_redir_total[gi])
                    for gi in sorted(self._pd_blk_total)},
                # ROUND 58: cross-batch lift — how many redirects came from the rolling registry
                # (earlier batches) vs the within-step batch, and the current registry size.
                "encoded_batch_size": _PD_ENCODED_BATCH,
                "cross_batch_redirects": self._pd_cross_redir_total,
                "within_batch_redirects": self._pd_within_redir_total,
                "registry_blocks": {str(gi): self._pd_registry_size(gi)
                                    for gi in sorted(self._pd_registry)},
            }
            path = os.path.join(_PD_STATS_DIR, f"bff_stats_{os.getpid()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(stats, f)
            os.replace(tmp, path)   # atomic — the reader never sees a half-written file
        except Exception as e:  # pragma: no cover - defensive (must never break the transfer)
            logger.warning("BFF P/D: could not dump fuse stats: %s", e)

    def wait_for_save(self):
        # Safety net: before blocking on the send queue, ensure EVERY fusion group sent a map this
        # step (sentinel for any that didn't complete), so the consumer's per-(request,group)
        # blocking recv_tensor can never deadlock on a missing id.
        if self.is_producer and self._pd_fuse:
            try:
                self._pd_flush_unsent()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF P/D flush_unsent failed: %s", e)
        super().wait_for_save()

    def _pd_flush_unsent(self) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, P2pNcclConnectorMetadataFF) or not meta.requests:
            return
        pending = [gi for gi in self._group_layers if gi > 0 and gi not in self._pd_sent]
        if not pending:
            return
        dev = torch.device("cuda")
        try:
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            runner = getattr(_bp, "_ACTIVE_RUNNER", None)
            if runner is not None and runner.kv_caches:
                dev = runner.kv_caches[0].device
        except Exception:
            pass
        sentinel = torch.tensor([[-1, -1, -1]], dtype=torch.int64, device=dev)
        for gi in pending:
            # ratio mode: the consumer recvs a [rows, G, 2] ratio side-tensor per (request, group)
            # right after the int map (blocking, no timeout) — so a pending group MUST also get a
            # ratio sentinel here, else D deadlocks on the ratio recv (ROUND 59 bug B).
            gn = len(self._group_layers.get(gi, ()))
            ratio_sentinel = (torch.ones((1, gn, 2), dtype=torch.float32, device=dev)
                              if _PD_RATIO else None)
            for request in meta.requests:
                remote_address = self._remote_addr_or_none(request.request_id, True)
                if remote_address is None:
                    continue
                self.p2p_nccl_engine.send_tensor(
                    _pd_key(request.request_id) + _PD_REDIR_TAG + str(gi),
                    sentinel, remote_address)
                if _PD_RATIO:
                    self.p2p_nccl_engine.send_tensor(
                        _pd_key(request.request_id) + _PD_RATIO_TAG + str(gi),
                        ratio_sentinel, remote_address)
            self._pd_sent.add(gi)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Multi-group-safe replacement for `request_finished` (SupportsHMA path). P2pNccl frees
        synchronously — `wait_for_save` already blocked until the KV was sent — so the per-group
        `block_ids` aren't needed here; mirror the parent's `request_finished` body."""
        self.chunked_prefill.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------
    # Scheduler-side: carry ALL groups' block ids (parent kept only [0])
    # ------------------------------------------------------------------
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                blocks.get_block_ids(),  # full per-group list (parent took [0])
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = P2pNcclConnectorMetadataFF()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                if num_tokens < len(new_req.prompt_token_ids or []):
                    # chunked prefill: stash the full per-group block ids
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids, new_req.prompt_token_ids)
                    continue
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=new_req.block_ids,
                    block_size=self._block_size,
                )
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=new_req.block_ids,
                    block_size=self._block_size,
                )
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                # A cached producer req we never stashed = a post-prefill decode step (e.g. the
                # 1-token decode before finish): nothing to transfer-accumulate → skip.
                if req_id not in self.chunked_prefill:
                    continue
                prev = self.chunked_prefill[req_id][0]          # per-group accumulated
                if new_block_ids is None:
                    # No new blocks allocated this chunk (fits in already-allocated / sliding-
                    # window blocks) — carry the accumulated per-group ids forward unchanged.
                    block_ids = prev
                elif resumed_from_preemption:
                    block_ids = list(new_block_ids)             # per-group
                else:
                    block_ids = [prev[g] + list(new_block_ids[g]) for g in range(len(new_block_ids))]
                prompt_token_ids = self.chunked_prefill[req_id][1]
                assert prompt_token_ids is not None
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                    continue
                meta.add_request(
                    request_id=req_id,
                    token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): resumed requests are the first N in scheduled_cached_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                if new_block_ids is None:
                    continue                                    # nothing new to load this step
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]
                meta.add_request(
                    request_id=req_id,
                    token_ids=token_ids,
                    block_ids=new_block_ids,                    # per-group
                    block_size=self._block_size,
                )

        self._requests_need_load.clear()
        return meta
