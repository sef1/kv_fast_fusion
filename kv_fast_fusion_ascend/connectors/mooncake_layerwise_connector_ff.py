"""Fast-Fusion-aware wrapper for the Ascend ``MooncakeLayerwiseConnector`` (BFF P/D milestone 1).

This is the Ascend/NPU analogue of :mod:`kv_fast_fusion.connectors.p2p_nccl_connector_ff`. The NCCL
connector is deliberately left **untouched**; this file is fully self-contained and reuses only the
already-standalone pieces of BFF:

  * the pure clustering in :mod:`kv_fast_fusion.pd_fuse` (device-generic; runs on NPU tensors), and
  * the consumer block-merge channel (``_ACTIVE_RUNNER._updated_block_tables`` → the patched
    scheduler ``_handle_block_merging_with_counts``).

Everything transport-specific is re-implemented here for the Mooncake RDMA push-write transport,
whose control plane is a ZMQ side-channel (there is no NCCL ``recv_tensor`` to intercept).

Milestone 1 (raw, byte-exact): the producer still pushes every block over RDMA; after a fusion
group's ``BFF_GROUP_SIZE`` layers have streamed through ``save_kv_layer``, the producer clusters the
group's per-block K (concat cosine), builds a per-request redirect map (owner-slot → representative
request's block-slot), and ships it to the decode node over a **dedicated FF ZMQ channel**. The
decode node applies the map post-transfer: it repoints owner block-table slots at the representative's
physical block and frees the redundant copies (BFF merge channel). No wire-dedup and no per-block
scales are shipped (``BFF_SCALE_MODE=raw`` only — ratio mode needs a CUDA Triton kernel).

The module top level imports nothing Ascend/NPU-specific, so the pure fusion glue
(:class:`MooncakeFFProducer`, :func:`resolve_redirect_rows`) is importable and unit-testable on any
box. The connector subclass + ZMQ wiring are defined only when ``vllm_ascend`` is importable.
"""

import hashlib
import json
import os
import queue
import struct
import threading
import time
from typing import Any

import torch

from vllm.logger import init_logger

from kv_fast_fusion.constants import THRESHOLD
from kv_fast_fusion.pd_fuse import (
    build_group_redirect,
    concat_cosine_cc_labels,
    concat_cosine_cross_match,
    concat_cosine_nr_tree_labels,
)

logger = init_logger("vllm." + __name__)

# --- fusion config (self-contained; mirrors the env knobs used by the NCCL connector) ---
_PD_MERGE = os.environ.get("BFF_PD_MERGE", "nr_tree")
_PD_REPR = os.environ.get("BFF_PD_REPR", "full")
_PD_PROJ_DIM = int(os.environ.get("BFF_PD_PROJ_DIM", "512"))
_PD_SCALE_MODE = os.environ.get("BFF_SCALE_MODE", "raw").lower()
_PD_DEBUG = os.environ.get("BFF_PD_DEBUG", "0") == "1"
# Cross-batch encoded registry window: keep the last N distinct requests' rep blocks per fusion
# group and match this step's blocks against them. 0 = disabled (within-batch fusion only).
_PD_ENCODED_BATCH = int(os.environ.get("BFF_PD_ENCODED_BATCH_SIZE", "0"))
# Dedicated FF ZMQ control channel: D binds (its side-channel base + this offset); P sends redirect
# maps there. Kept separate from the connector's own handshake port so the stock recv thread is
# untouched. TP=1 assumed for M1 (the new setup runs tp_size=1 on both P and D).
_FF_PORT_OFFSET = int(os.environ.get("BFF_MOONCAKE_FF_PORT_OFFSET", "20000"))

# Fusion-stats dump: the producer drops a per-process ``bff_stats_<pid>.json`` into this dir so the
# benchmark's collect_bff_stats() can report real compression (mirrors the NCCL connector). The shell
# exports BFF_PD_STATS_DIR only for the kv_producer role, so "not None" also gates on producer-only.
_PD_STATS_DIR = os.environ.get("BFF_PD_STATS_DIR")
_PD_STATS_EVERY = int(os.environ.get("BFF_PD_STATS_EVERY", "50"))

_FF_REDIRECT_MSG = b"bff_redirect_msg"


def _ext_hash(external_id: str) -> int:
    """Process-stable int64 hash of a request's *external* id (shared across P and D). Uses sha256
    to match the connector's own ``string_to_int64_hash``; keyed on the external id so P and D
    (whose full request ids differ by a 9-char suffix) agree."""
    return struct.unpack("<q", hashlib.sha256(external_id.encode("utf-8")).digest()[:8])[0] & 0x7FFFFFFFFFFFFFFF


def _external_id(request_id: str) -> str:
    """Strip the 9-char EngineCore suffix (vLLM PR #27987) to recover the proxy-assigned external id.
    Mirrors the connector's ``get_external_request_id`` without importing the NPU-only module."""
    return request_id[:-9]


def _block_repr(k_cache: torch.Tensor, idx: torch.Tensor, jl_holder: list) -> torch.Tensor:
    """Per-layer block representation ``[N, D_repr]`` (float32) for the clustering similarity.

    ``k_cache`` is one layer's paged K tensor ``[num_blocks, block_size, kv_heads, head_dim]`` (the
    Mooncake list layout ``kv_layer[0]``); ``idx`` selects the flat blocks. ``full`` = exact whole
    block, ``mean`` = head_dim mean, ``proj`` = fixed-seed JL projection. ``jl_holder`` is a 1-elem
    list caching the lazily-built projection matrix."""
    blk = k_cache[idx].float()                         # [N, block_size, kv_heads, head_dim]
    n = idx.shape[0]
    if _PD_REPR == "mean":
        head_dim = blk.shape[-1]
        return blk.reshape(n, -1, head_dim).mean(dim=1)
    full = blk.reshape(n, -1)
    if _PD_REPR == "proj":
        if jl_holder[0] is None:
            g = torch.Generator(device=full.device)
            g.manual_seed(1234)
            jl_holder[0] = torch.randn(
                full.shape[1], _PD_PROJ_DIM, generator=g, device=full.device, dtype=torch.float32)
        return full @ jl_holder[0]
    return full


class MooncakeFFProducer:
    """Transport-agnostic producer-side fusion accumulator (raw mode, within-batch).

    Fed one attention layer at a time via :meth:`on_layer`. It buffers each fusion group's per-block
    K representation across the group's layers; when the group's last layer arrives it clusters
    (concat cosine) and returns the per-request redirect rows to ship. Pure torch + pd_fuse — no NPU,
    no ZMQ — so it is unit-testable with synthetic tensors.
    """

    def __init__(self) -> None:
        self._buf: dict[int, dict] = {}       # gi -> partial group buffer for the current step
        self._cur_step_id: int | None = None  # detects step boundary (fresh metadata object)
        self._jl = [None]                     # lazy JL matrix for BFF_PD_REPR=proj
        # cumulative compression accounting (per fusion group)
        self.blk_total: dict[int, int] = {}
        self.redir_total: dict[int, int] = {}
        self.cross_redir_total = 0            # redirects to a PREVIOUS-batch rep (cross-batch lift)
        self.within_redir_total = 0           # redirects to a same-batch rep (legacy behavior)
        self.steps = 0                        # scheduler steps seen (approximate; via metadata id())
        self.group_completions = 0            # group finishes seen (true denominator for overhead avg)
        self.dedup_ms = 0.0                   # cumulative clustering time (ms), for overhead avg
        # Cross-batch encoded registry: gi -> rolling window of the last _PD_ENCODED_BATCH distinct
        # requests' representative block K-reps (raw mode). Empty/unused when _PD_ENCODED_BATCH <= 0.
        self._registry: dict[int, dict] = {}

    def reset_step(self, step_id: int) -> None:
        if step_id != self._cur_step_id:
            self._cur_step_id = step_id
            self._buf.clear()
            self.steps += 1

    def on_layer(
        self,
        gi: int,
        layer_name: str,
        k_cache: torch.Tensor,
        group_layer_names: set[str],
        requests: list[tuple],
        tp_group=None,
    ) -> dict[str, list[tuple[int, int, int]]] | None:
        """Accumulate one layer of fusion group ``gi``. ``requests`` is the ordered list of
        ``(external_id, local_block_ids_for_gi[, has_remote])`` for this step's batch (``has_remote``
        defaults to True when omitted; it gates cross-batch registration to decode-bound requests).
        Returns ``None`` until the group completes, then a dict
        ``{owner_external_id: [(owner_slot, rep_hash, rep_slot), ...]}``.
        """
        buf = self._buf.get(gi)
        if buf is None:
            flat_bids: list[int] = []
            flat_req_local: list[int] = []
            flat_slot: list[int] = []
            ext_ids: list[str] = []
            ext_has_remote: list[bool] = []
            for ri, req in enumerate(requests):
                ext_id, bids = req[0], req[1]
                ext_ids.append(ext_id)
                ext_has_remote.append(bool(req[2]) if len(req) > 2 else True)
                for slot, bid in enumerate(bids):
                    if bid > 0:                        # skip the null block 0
                        flat_bids.append(bid)
                        flat_req_local.append(ri)
                        flat_slot.append(slot)
            # Build the block-index tensor once — flat_bids is constant across the group's layers,
            # so this avoids G-1 redundant host->device copies per group per step.
            idx = (torch.as_tensor(flat_bids, device=k_cache.device, dtype=torch.long)
                   if flat_bids else None)
            buf = {
                "seen": set(),
                "k_layers": [],
                "flat_bids": flat_bids,
                "flat_req_local": flat_req_local,
                "flat_slot": flat_slot,
                "ext_ids": ext_ids,
                "ext_has_remote": ext_has_remote,
                "idx": idx,
            }
            self._buf[gi] = buf

        if buf["idx"] is not None:
            buf["k_layers"].append(_block_repr(k_cache, buf["idx"], self._jl))
        buf["seen"].add(layer_name)

        if len(buf["seen"]) < len(group_layer_names):
            return None                                # group not complete yet
        # --- group complete: cluster + build redirect rows ---
        try:
            return self._build_send_rows(gi, buf, tp_group)
        finally:
            self._buf.pop(gi, None)

    def _build_send_rows(self, gi, buf, tp_group) -> dict[str, list[tuple[int, int, int]]]:
        """Cluster the completed group's blocks and build ``{owner_ext: [(owner_slot, rep_hash,
        rep_slot), ...]}``. When ``_PD_ENCODED_BATCH>0`` also matches against the rolling cross-batch
        registry (earlier requests' reps) before within-batch clustering the remainder, then registers
        the new reps. When disabled it is the original within-batch-only path."""
        self.group_completions += 1
        send_rows: dict[str, list[tuple[int, int, int]]] = {}
        flat_req_local = buf["flat_req_local"]
        flat_slot = buf["flat_slot"]
        ext_ids = buf["ext_ids"]
        n_cross = 0
        n_within = 0
        if buf["flat_bids"] and buf["k_layers"]:
            t0 = time.perf_counter()
            dev0 = buf["k_layers"][0].device

            def _cluster(k_layers, req_of_block):
                # TP>1: only CC exposes the raw Gram/sq for the cross-rank all-reduce.
                if tp_group is not None:
                    return concat_cosine_cc_labels(k_layers, req_of_block, THRESHOLD,
                                                   tp_group=tp_group)
                fn = (concat_cosine_nr_tree_labels if _PD_MERGE == "nr_tree"
                      else concat_cosine_cc_labels)
                return fn(k_layers, req_of_block, THRESHOLD)

            def _emit_within(redirects):
                nonlocal n_within
                for owner_ri, rws in redirects.items():
                    owner_ext = ext_ids[owner_ri]
                    for (slot, rep_local, rep_slot, _rep_flat, _own_flat) in rws:
                        send_rows.setdefault(owner_ext, []).append(
                            (int(slot), _ext_hash(ext_ids[rep_local]), int(rep_slot)))
                        n_within += 1

            N = len(flat_req_local)
            if _PD_ENCODED_BATCH <= 0:
                # ---- within-batch only (legacy path, unchanged behavior) ----
                labels = _cluster(buf["k_layers"], torch.as_tensor(flat_req_local, device=dev0))
                _, redirects = build_group_redirect(labels, flat_req_local, flat_slot)
                _emit_within(redirects)
            else:
                # ---- cross-batch (registry enabled) ----
                reg = self._registry.get(gi)
                reg_vecs = reg["vecs"] if reg else None
                reg_sq = reg["sq"] if reg else None
                best_idx, _score, cur_sq, cur_concat = concat_cosine_cross_match(
                    buf["k_layers"], reg_vecs, reg_sq, THRESHOLD, tp_group=tp_group)
                # Forbid a self-merge (a registered rep from the SAME request, e.g. chunked prefill).
                if reg is not None and bool((best_idx >= 0).any()):
                    own_hash = torch.tensor(
                        [_ext_hash(ext_ids[r]) for r in flat_req_local],
                        dtype=torch.long, device=best_idx.device)
                    self_hit = (best_idx >= 0) & (reg["hash"][best_idx.clamp(min=0)] == own_hash)
                    best_idx = torch.where(self_hit, torch.full_like(best_idx, -1), best_idx)
                best_list = best_idx.tolist()
                # Phase 1: cross-batch matches → redirect to the registry rep (already resident on D).
                # Resolve hash/slot NOW — _register_reps below re-indexes the registry in this call.
                matched = [False] * N
                for i, ridx in enumerate(best_list):
                    if ridx < 0:
                        continue
                    owner_ext = ext_ids[flat_req_local[i]]
                    send_rows.setdefault(owner_ext, []).append(
                        (int(flat_slot[i]), int(reg["hash"][ridx].item()),
                         int(reg["slot"][ridx].item())))
                    matched[i] = True
                    n_cross += 1
                # Phase 2: within-batch clustering on the UNMATCHED remainder (subset → map back).
                unmatched = [i for i in range(N) if not matched[i]]
                reps_to_register: list[int] = []
                if unmatched:
                    sub_k = [Kg[unmatched] for Kg in buf["k_layers"]]
                    sub_req = [flat_req_local[i] for i in unmatched]
                    sub_slot = [flat_slot[i] for i in unmatched]
                    labels = _cluster(sub_k, torch.as_tensor(sub_req, device=dev0))
                    _, redirects = build_group_redirect(labels, sub_req, sub_slot)
                    _emit_within(redirects)
                    labels_l = labels.tolist()
                    reps_to_register = [unmatched[i] for i in range(len(labels_l))
                                        if labels_l[i] == i]
                self._register_reps(gi, buf, reps_to_register, cur_concat, cur_sq)
            self.dedup_ms += (time.perf_counter() - t0) * 1e3
        n_redir = n_cross + n_within
        self.blk_total[gi] = self.blk_total.get(gi, 0) + len(buf["flat_bids"])
        self.redir_total[gi] = self.redir_total.get(gi, 0) + n_redir
        self.cross_redir_total += n_cross
        self.within_redir_total += n_within
        if n_redir or _PD_DEBUG:
            logger.info("BFF Mooncake fuse group gi=%d | merge=%s | repr=%s | reqs=%d | blocks=%d | "
                        "redirects=%d (cross=%d within=%d) | reg=%d", gi, _PD_MERGE, _PD_REPR,
                        len(ext_ids), len(buf["flat_bids"]), n_redir, n_cross, n_within,
                        self._registry_size(gi))
        return send_rows

    # -- cross-batch encoded registry (raw mode; NCCL analogue) -----------------------------
    def _registry_size(self, gi) -> int:
        reg = self._registry.get(gi)
        return 0 if reg is None or reg["vecs"] is None else int(reg["vecs"].shape[0])

    def _register_reps(self, gi, buf, rep_flats, cur_concat, cur_sq) -> None:
        """Append this step's new representative blocks to the group registry, then FIFO-evict to the
        window. Only reps for requests bound to a decode target (``has_remote``) are registered, so a
        future owner can always resolve them on D."""
        if not rep_flats:
            return
        flat_req_local = buf["flat_req_local"]
        flat_slot = buf["flat_slot"]
        ext_ids = buf["ext_ids"]
        ext_has_remote = buf["ext_has_remote"]
        dev = cur_concat.device
        reg = self._registry.get(gi)
        if reg is None:
            reg = {"vecs": None, "sq": None, "hash": None, "slot": None, "seq": None,
                   "key2seq": {}, "next_seq": 0}
            self._registry[gi] = reg
        v, sq, hsh, slt, seq = [], [], [], [], []
        for f in rep_flats:
            ri = flat_req_local[f]
            if not ext_has_remote[ri]:
                continue
            ext = ext_ids[ri]
            s = reg["key2seq"].get(ext)
            if s is None:
                s = reg["next_seq"]; reg["key2seq"][ext] = s; reg["next_seq"] = s + 1
            v.append(cur_concat[f]); sq.append(cur_sq[f])
            hsh.append(_ext_hash(ext)); slt.append(flat_slot[f]); seq.append(s)
        if not v:
            return

        def _cat(old, new):
            return new if old is None else torch.cat([old, new])
        reg["vecs"] = _cat(reg["vecs"], torch.stack(v))
        reg["sq"] = _cat(reg["sq"], torch.stack(sq))
        reg["hash"] = _cat(reg["hash"], torch.tensor(hsh, dtype=torch.long, device=dev))
        reg["slot"] = _cat(reg["slot"], torch.tensor(slt, dtype=torch.long, device=dev))
        reg["seq"] = _cat(reg["seq"], torch.tensor(seq, dtype=torch.long, device=dev))
        self._evict_registry(gi)

    def _evict_registry(self, gi) -> None:
        """Drop rows from requests older than the last ``_PD_ENCODED_BATCH`` distinct requests. Seq
        ids are dense + monotonic, so keeping ``seq >= next_seq - N`` keeps exactly the last N."""
        reg = self._registry.get(gi)
        if reg is None or reg["seq"] is None:
            return
        keep_from = reg["next_seq"] - _PD_ENCODED_BATCH
        if keep_from <= 0:
            return
        keep = reg["seq"] >= keep_from
        if bool(keep.all()):
            return
        idx = keep.nonzero(as_tuple=True)[0]
        for k in ("vecs", "sq", "hash", "slot", "seq"):
            if reg[k] is not None:
                reg[k] = reg[k][idx]
        reg["key2seq"] = {k: s for k, s in reg["key2seq"].items() if s >= keep_from}

    def dump_stats(self, stats_dir: str) -> None:
        """Write this producer's cumulative fusion compression to ``bff_stats_<pid>.json`` in
        ``stats_dir``. The benchmark's collect_bff_stats() merges these after the run. Ascend analogue
        of the NCCL connector's ``_pd_dump_fuse_stats``. Compression FACTOR = total / (total - freed):
        how many x smaller the KV cache gets from fusion (>1; 2.0 = half the blocks)."""
        try:
            def _factor(b, r):
                return b / max(1, b - r)
            tot_b = sum(self.blk_total.values())
            tot_r = sum(self.redir_total.values())
            stats = {
                "pid": os.getpid(),
                "steps": self.steps,
                # dedup_ms accumulates once per group-completion, so the true per-group average
                # divides by group_completions (not steps, which id()-collisions undercount).
                "overhead_avg_group_dedup_ms": (self.dedup_ms / self.group_completions
                                                if self.group_completions else 0.0),
                "total_blocks": tot_b,
                "freed": tot_r,
                "compression_avg_factor": _factor(tot_b, tot_r),
                "compression_per_group": {
                    str(gi): _factor(self.blk_total[gi], self.redir_total.get(gi, 0))
                    for gi in sorted(self.blk_total)},
                # Cross-batch lift: how many redirects came from the rolling registry (earlier
                # batches) vs the within-step batch, plus the window size and current registry size.
                "encoded_batch_size": _PD_ENCODED_BATCH,
                "cross_batch_redirects": self.cross_redir_total,
                "within_batch_redirects": self.within_redir_total,
                "registry_blocks": {str(gi): self._registry_size(gi)
                                    for gi in sorted(self._registry)},
            }
            path = os.path.join(stats_dir, f"bff_stats_{os.getpid()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(stats, f)
            os.replace(tmp, path)   # atomic — the reader never sees a half-written file
        except Exception as e:  # pragma: no cover - defensive (must never break the transfer)
            logger.warning("BFF Mooncake: could not dump fuse stats: %s", e)


def resolve_redirect_rows(
    ext2blocks: dict[str, list[list[int]]],
    hash2ext: dict[int, str],
    owner_ext_id: str,
    gi: int,
    rows: list[tuple[int, int, int]],
) -> tuple[list[int] | None, int, int, int]:
    """Consumer-side: turn shipped redirect ``rows`` into the owner's new (deduped) block table.

    ``ext2blocks`` maps external id → per-group D-physical block ids (from the decode runner, plus
    this step's load metadata); ``hash2ext`` maps ``_ext_hash`` → external id. Returns
    ``(new_owner_blocks, n_applied, n_unresolved, n_owner_missing)``; ``new_owner_blocks`` is ``None``
    when nothing changed. ``n_owner_missing`` counts rows we couldn't even attempt because the OWNER
    itself was not resolvable (distinct from ``n_unresolved`` = the REP not resolvable) — kept
    separate so the apply log can tell owner-residency problems from rep-residency problems. Port of
    the resolve loop in the NCCL connector's ``_pd_consumer_apply`` (raw mode)."""
    owner_groups = ext2blocks.get(owner_ext_id)
    if owner_groups is None or gi >= len(owner_groups):
        return None, 0, 0, len(rows)                   # owner not resident → owner-miss, not rep-miss
    owner_blocks = list(owner_groups[gi])
    n_applied = n_unresolved = 0
    changed = False
    for owner_slot, rep_hash, rep_slot in rows:
        rep_ext = hash2ext.get(int(rep_hash))
        if rep_ext is None or rep_ext not in ext2blocks:
            n_unresolved += 1                          # rep not (yet) resident on D → can't share
            continue
        rep_groups = ext2blocks[rep_ext]
        if gi >= len(rep_groups):
            n_unresolved += 1
            continue
        rep_grp = rep_groups[gi]
        if not (0 <= rep_slot < len(rep_grp) and 0 <= owner_slot < len(owner_blocks)):
            n_unresolved += 1
            continue
        owner_blocks[owner_slot] = int(rep_grp[rep_slot])
        changed = True
        n_applied += 1
    return (owner_blocks if changed else None), n_applied, n_unresolved, 0


# ---------------------------------------------------------------------------------------------
# Ascend/NPU-only section: the connector subclass + ZMQ side-channel + block-table rewrite.
# Guarded so the pure glue above stays importable on non-Ascend boxes (for unit tests).
# ---------------------------------------------------------------------------------------------
try:
    import msgspec
    import zmq
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
        SupportsHMA,
    )
    from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        MooncakeLayerwiseConnector,
        MooncakeLayerwiseConnectorMetadata,
        MooncakeLayerwiseConnectorWorker,
        get_external_request_id,
        zmq_ctx,
    )

    _ASCEND_AVAILABLE = True
except Exception as _imp_err:  # pragma: no cover - only importable on the Ascend/NPU stack
    _ASCEND_AVAILABLE = False
    logger.debug("MooncakeLayerwiseConnectorFF: Ascend stack unavailable (%s); pure glue only.",
                 _imp_err)


if _ASCEND_AVAILABLE:

    class _FFRedirectRecvThread(threading.Thread):
        """Decode-side listener for the dedicated FF redirect channel. Binds a ``zmq.PULL`` socket and
        records each arrived redirect map as ``{external_id: {gi: rows}}`` for the connector to apply
        at ``get_finished`` (post-transfer). Fire-and-forget: no ACK is sent, so the producer never
        blocks — a dropped map just means that request keeps per-request copies on D (less compression,
        never incorrect). Kept separate from the stock ``KVCacheRecvingLayerThread`` (untouched)."""

        def __init__(self, host: str, port: int):
            super().__init__(daemon=True, name="BFF-FFRedirectRecvThread")
            self._host = host
            self._port = port
            self.lock = threading.Lock()
            self.pending: dict[str, dict[int, list]] = {}
            self._decoder = msgspec.msgpack.Decoder(type=tuple)

        def drain(self) -> dict[str, dict[int, list]]:
            with self.lock:
                out, self.pending = self.pending, {}
            return out

        def run(self):
            path = make_zmq_path("tcp", self._host, self._port)
            logger.info("BFF FF redirect listener (PULL) on %s", path)
            ctx = zmq.Context()
            sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.PULL, bind=True)
            try:
                while True:
                    try:
                        msg = self._decoder.decode(sock.recv())
                        if msg and msg[0] == _FF_REDIRECT_MSG:
                            # (tag, external_id, gi, rows) — rows: list[[owner_slot, rep_hash, rep_slot]]
                            _tag, ext_id, gi, rows = msg
                            with self.lock:
                                self.pending.setdefault(ext_id, {})[int(gi)] = rows
                    except Exception as e:  # pragma: no cover - defensive (never kill the listener)
                        logger.warning("BFF FF redirect listener error: %s", e)
            finally:
                ctx.destroy(linger=0)


    class _FFRedirectSendThread(threading.Thread):
        """Producer-side fire-and-forget sender for the FF redirect channel. The ``save_kv_layer``
        hook only enqueues ``(host, port, ext_id, gi, rows)``; this daemon thread owns persistent
        ``zmq.PUSH`` sockets keyed by ``(host, port)`` and sends off the prefill hot path. No ACK —
        a dropped map costs compression, not correctness (the consumer apply is fully guarded)."""

        def __init__(self):
            super().__init__(daemon=True, name="BFF-FFRedirectSendThread")
            self._q: "queue.Queue" = queue.Queue()
            self._ctx = None
            self._socks: dict[tuple, Any] = {}
            self._encoder = msgspec.msgpack.Encoder()

        def submit(self, host, port, ext_id, gi, rows) -> None:
            if host is None or port is None:
                return
            self._q.put((host, int(port), ext_id, int(gi), rows))   # non-blocking (unbounded queue)

        def _sock_for(self, host, port):
            key = (host, port)
            s = self._socks.get(key)
            if s is None:
                path = make_zmq_path("tcp", host, port)
                s = make_zmq_socket(ctx=self._ctx, path=path, socket_type=zmq.PUSH, bind=False)
                s.setsockopt(zmq.LINGER, 0)
                s.setsockopt(zmq.SNDTIMEO, 2000)   # bound the bg thread if a peer is (briefly) absent
                self._socks[key] = s
            return s

        def run(self):
            self._ctx = zmq.Context()
            try:
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    host, port, ext_id, gi, rows = item
                    try:
                        payload = self._encoder.encode((_FF_REDIRECT_MSG, ext_id, gi, rows))
                        self._sock_for(host, port).send(payload)
                    except Exception as e:  # pragma: no cover - drop on timeout/no-peer (best-effort)
                        logger.warning("BFF Mooncake ship redirect dropped (%s:%d): %s", host, port, e)
            finally:
                if self._ctx is not None:
                    self._ctx.destroy(linger=0)


    class MooncakeLayerwiseConnectorFF(MooncakeLayerwiseConnector, SupportsHMA):
        """Group-aware, fusion-adding subclass of the Ascend layerwise connector (see module doc)."""

        def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole",
                     kv_cache_config: "KVCacheConfig | None" = None):
            super().__init__(vllm_config, role, kv_cache_config)
            self._ff_enabled = os.environ.get("BFF_PD_FUSE", "0") == "1" and _PD_SCALE_MODE == "raw"
            self._ff_producer = MooncakeFFProducer() if self._ff_enabled else None
            self._ff_group_layers: dict[int, set[str]] | None = None
            self._ff_fusion_groups: set[int] | None = None
            self._ff_recv_thread: _FFRedirectRecvThread | None = None
            self._ff_send_thread: _FFRedirectSendThread | None = None
            if self._ff_enabled and self.connector_worker is not None:
                self._ff_install_worker_hooks()
            if self._ff_enabled:
                logger.info("MooncakeLayerwiseConnectorFF: fusion enabled (raw, role=%s).", role)

        # -- worker (producer + consumer) integration --------------------------------------
        def _ff_install_worker_hooks(self) -> None:
            """Wrap the inner worker's ``save_kv_layer`` (producer accumulate/ship) without editing
            the vendored connector. The consumer apply is driven from ``get_finished`` below."""
            worker = self.connector_worker
            is_producer = self.vllm_config.kv_transfer_config.is_kv_producer
            is_consumer = self.vllm_config.kv_transfer_config.is_kv_consumer

            if is_producer:
                self._ff_send_thread = _FFRedirectSendThread()
                self._ff_send_thread.start()
                orig_save = worker.save_kv_layer

                def _wrapped_save(layer_name, kv_layer, attn_metadata, connector_metadata, **kw):
                    # Resolve the layer name the SAME way the connector does (empty → index_to_name),
                    # but BEFORE orig_save runs, since orig_save increments worker.current_layer.
                    resolved = layer_name
                    if resolved == "" and worker.current_layer < worker.total_layers:
                        names = worker.index_to_name.get(worker.current_layer)
                        if names:
                            resolved = names[0]
                    orig_save(layer_name, kv_layer, attn_metadata, connector_metadata, **kw)
                    try:
                        if resolved:
                            self._ff_producer_accumulate(
                                worker, resolved, kv_layer, connector_metadata)
                    except Exception as e:  # pragma: no cover - never break the transfer
                        logger.warning("BFF Mooncake producer fusion failed: %s", e)

                worker.save_kv_layer = _wrapped_save

            if is_consumer:
                host = worker.side_channel_host
                port = worker.side_channel_port + _FF_PORT_OFFSET + worker.tp_rank
                self._ff_recv_thread = _FFRedirectRecvThread(host, port)
                self._ff_recv_thread.start()

        def _ff_build_group_layers(self, worker) -> None:
            """Map fusion group index → layer names + the set of fusion groups (full-attention,
            gi>0), from the worker's registered ``layer_metadata`` + kv-cache specs."""
            group_layers: dict[int, set[str]] = {}
            for ln, lm in worker.layer_metadata.items():
                group_layers.setdefault(lm.tensor_group_idx[0], set()).add(ln)
            fusion_groups = set()
            for gi in group_layers:
                if gi <= 0 or gi >= len(worker.kv_cache_specs):
                    continue
                spec = worker.kv_cache_specs[gi]
                if isinstance(spec, FullAttentionSpec) and worker.kernel_block_size_scale[gi] == 1:
                    fusion_groups.add(gi)
            self._ff_group_layers = group_layers
            self._ff_fusion_groups = fusion_groups
            logger.info("BFF Mooncake: fusion groups=%s (of %d groups)",
                        sorted(fusion_groups), len(group_layers))

        def _ff_producer_accumulate(self, worker, layer_name, kv_layer, connector_metadata) -> None:
            if not connector_metadata.requests:
                return
            if self._ff_group_layers is None:
                self._ff_build_group_layers(worker)
            gi = worker.layer_metadata[layer_name].tensor_group_idx[0]
            if gi not in self._ff_fusion_groups:
                return
            self._ff_producer.reset_step(id(connector_metadata))
            requests = [
                (get_external_request_id(rid), list(rm.local_block_ids[gi]),
                 rm.remote_host is not None and rm.remote_port is not None)
                for rid, rm in connector_metadata.requests.items()
                if gi < len(rm.local_block_ids)
            ]
            if not requests:
                return
            send_rows = self._ff_producer.on_layer(
                gi, layer_name, kv_layer[0], self._ff_group_layers[gi], requests)
            if send_rows is None:
                return
            for rid, rm in connector_metadata.requests.items():
                ext_id = get_external_request_id(rid)
                rows = send_rows.get(ext_id)
                if not rows:
                    continue
                self._ff_ship_redirect(rm.remote_host, rm.remote_port, ext_id, gi, rows)
            # Periodically dump fusion stats so the benchmark's collect_bff_stats() can report real
            # compression (mirrors the NCCL cadence: step 1, then every _PD_STATS_EVERY steps).
            steps = self._ff_producer.steps
            if _PD_STATS_DIR and steps and (steps % _PD_STATS_EVERY == 0 or steps == 1):
                self._ff_producer.dump_stats(_PD_STATS_DIR)

        def _ff_ship_redirect(self, host, base_port, ext_id, gi, rows) -> None:
            """Enqueue one request's redirect rows for group ``gi`` to the background sender (the
            decode node's FF PULL channel). Non-blocking: NOTHING here runs on the prefill hot path
            beyond a queue append. Rows are normalized to plain ints for msgpack."""
            if host is None or base_port is None or self._ff_send_thread is None:
                return
            data = [[int(o), int(h), int(s)] for (o, h, s) in rows]
            self._ff_send_thread.submit(host, base_port + _FF_PORT_OFFSET, ext_id, gi, data)

        # -- consumer apply -----------------------------------------------------------------
        def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
            sending, recving = super().get_finished(finished_req_ids)
            if self._ff_enabled and self._ff_recv_thread is not None:
                try:
                    # `recving` = request ids whose KV *just* fully landed this step. Applying a
                    # redirect only for those requests guarantees we repoint+free BEFORE the owner
                    # decodes — mirroring the NCCL connector's load-time apply. Draining and applying
                    # for already-decoding requests frees in-use blocks → pool aliasing → global KV
                    # corruption (every request garbage → no EOS → runs to max_tokens).
                    self._ff_apply_pending(recving)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("BFF Mooncake consumer apply failed: %s", e)
            return sending, recving

        def _ff_apply_pending(self, recving: set[str]) -> None:
            """Apply redirect maps for the requests whose KV *just* completed this step (``recving``),
            then stage the freed/redirected block tables for the scheduler via the BFF merge channel.

            Timing is the correctness gate (see ``get_finished``): a redirect is applied ONLY at the
            step its owner's recv completes — before the owner decodes — so repoint+free is safe.
            Three cases for a pending owner:
              * owner in ``recving``  → apply now (pre-decode window);
              * owner not yet resident → arrived early, re-queue for the step it lands;
              * owner resident but not in ``recving`` → its window already passed (it is decoding) →
                DROP. Applying now would free in-use blocks; a dropped map costs compression, not
                correctness. Dropping also bounds the pending map."""
            pending = self._ff_recv_thread.drain()
            if not pending:
                return
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            runner = getattr(_bp, "_ACTIVE_RUNNER", None)
            if runner is None:
                logger.warning("BFF Mooncake: _ACTIVE_RUNNER unset on D; redirect maps dropped.")
                return
            # external id -> per-group D block ids, + hash -> external id (from resident requests).
            # This is the RESIDENCY set (requests already in runner state): it decides defer-vs-drop
            # below and must NOT include still-loading requests.
            ext2blocks: dict[str, list] = {}
            hash2ext: dict[int, str] = {}
            rid_by_ext: dict[str, str] = {}
            for rid, st in getattr(runner, "requests", {}).items():
                bids = getattr(st, "block_ids", None)
                if bids is None:
                    continue
                ext = get_external_request_id(rid)
                ext2blocks[ext] = bids
                hash2ext[_ext_hash(ext)] = ext
                rid_by_ext[ext] = rid
            # RESOLUTION set = residency set overlaid with THIS step's load metadata. A request whose
            # KV just landed (`recving`) is in the connector's load metadata but not yet in
            # runner.requests; without this overlay resolve_redirect_rows can't find the owner (or a
            # co-loading rep) → every row counts as unresolved. Metadata is still bound here — the
            # runner clears it only after get_finished. Mirrors the NCCL _pd_consumer_apply overlay.
            res_ext2blocks = dict(ext2blocks)
            res_hash2ext = dict(hash2ext)
            meta = self._connector_metadata
            for rid, rm in getattr(meta, "requests", {}).items():
                lb = getattr(rm, "local_block_ids", None)
                if lb is None:
                    continue
                ext = get_external_request_id(rid)
                res_ext2blocks[ext] = lb
                res_hash2ext[_ext_hash(ext)] = ext
                rid_by_ext.setdefault(ext, rid)
            just_recv_ext = {get_external_request_id(rid) for rid in recving}
            updated: dict[str, dict[int, list[int]]] = {}
            n_applied = n_unresolved = n_owner_missing = n_deferred = n_dropped = 0
            leftover: dict[str, dict[int, list]] = {}
            for ext_id, groups in pending.items():
                if ext_id in just_recv_ext:
                    # Owner's KV just landed and it has not decoded yet → safe to repoint + free.
                    for gi, rows in groups.items():
                        new_blocks, na, nu, nom = resolve_redirect_rows(
                            res_ext2blocks, res_hash2ext, ext_id, gi, rows)
                        n_applied += na
                        n_unresolved += nu
                        n_owner_missing += nom
                        if new_blocks is not None:
                            rid = rid_by_ext.get(ext_id)
                            if rid is not None:
                                updated.setdefault(rid, {})[gi] = new_blocks
                                self._ff_write_runner_block_table(runner, rid, gi, new_blocks)
                elif ext_id not in ext2blocks:
                    # Redirect arrived before the owner's KV → keep for the step its recv completes.
                    leftover[ext_id] = groups
                    n_deferred += 1
                else:
                    # Owner resident but past its recv-complete window (already decoding) → unsafe to
                    # apply; drop to avoid freeing in-use blocks (and to bound the pending map).
                    n_dropped += 1
            if leftover:
                with self._ff_recv_thread.lock:
                    for ext_id, groups in leftover.items():
                        self._ff_recv_thread.pending.setdefault(ext_id, {}).update(groups)
            if updated:
                runner._updated_block_tables = updated
            if n_applied or n_unresolved or n_owner_missing or n_dropped or _PD_DEBUG:
                logger.info("BFF Mooncake apply | redirects_applied=%d | owner_unresident=%d | "
                            "reps_unresolved=%d | owners_deferred=%d | owners_dropped_post_decode=%d",
                            n_applied, n_owner_missing, n_unresolved, n_deferred, n_dropped)

        @staticmethod
        def _ff_write_runner_block_table(runner, rid, gi, new_blocks) -> None:
            """Write the redirected per-group block table into the runner's worker-side mirror so the
            forward reads the shared blocks. Guarded; ports the NCCL connector's
            ``_pd_write_runner_block_table`` (NPU tensors)."""
            ridx = runner.input_batch.req_id_to_index.get(rid)
            if ridx is None:
                return
            # NPU `MultiGroupBlockTable.__getitem__(gi)` → the per-group BlockTable (valid on GPU too).
            bt_obj = runner.input_batch.block_table[gi]
            n = min(len(new_blocks), int(bt_obj.num_blocks_per_row[ridx]))
            row = new_blocks[:n]
            bt_obj.block_table.np[ridx, :n] = row
            bt_obj.block_table.gpu[ridx, :n] = torch.tensor(
                row, device=bt_obj.block_table.gpu.device, dtype=bt_obj.block_table.gpu.dtype)
            st = runner.requests.get(rid)
            if st is not None and gi < len(st.block_ids):
                st.block_ids[gi][:n] = row


    def register_mooncake_layerwise_ff() -> None:
        """Register ``MooncakeLayerwiseConnectorFF`` with the KV connector factory. Call from the
        Ascend init (alongside ``register_connector``); then set
        ``connectors[0].kv_connector = "MooncakeLayerwiseConnectorFF"`` in KV_TRANSFER_CONFIG."""
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        if "MooncakeLayerwiseConnectorFF" in KVConnectorFactory._registry:
            KVConnectorFactory._registry.pop("MooncakeLayerwiseConnectorFF")
        KVConnectorFactory.register_connector(
            "MooncakeLayerwiseConnectorFF",
            "kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff",
            "MooncakeLayerwiseConnectorFF",
        )
        logger.info("Registered MooncakeLayerwiseConnectorFF.")
