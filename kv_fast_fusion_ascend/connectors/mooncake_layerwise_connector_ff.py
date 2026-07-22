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
import itertools
import json
import os
import queue
import struct
import threading
import time
from collections import Counter, OrderedDict
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
# Cross-request index backend: "lsh" (SimHash banded index — O(N) probe, large capacity) or "matrix"
# (the legacy dense concat_cosine_cross_match over a bounded FIFO window). Within-batch stays cc.
_PD_CROSS_INDEX = os.environ.get("BFF_PD_CROSS_INDEX", "lsh").lower()
# Within-batch cc fusion (merges duplicate blocks across DIFFERENT requests in the same step). This is
# the O(N²) concat_cosine_cc/nr_tree pass — the main per-group overhead. 1 = on (default, current
# behavior); 0 = off (only cross-request matching runs; unmatched blocks still register into the index).
_PD_INTRA_REQ_FF = os.environ.get("BFF_PD_INTRA_REQ_FF", "1") == "1"
# Rep-lifetime safety (con512 merge corruption fix). The consumer apply resolves the redirect REP from a
# residency set overlaid with accumulated load-metadata (`_ff_load_blocks`). A rep resolved from that
# overlay can be STALE (its request since preempted/finished → blocks recycled) → the owner is repointed to
# another request's live KV → garbage → rambling (seen at con512, not con256; churn-gated). With
# BFF_FF_REP_SAFE=1 the REP is resolved ONLY from live `runner.requests` — a stale rep then counts as
# `unresolved`, so the owner safely keeps its own block (compression lost on that row, never wrong).
# Owners still use the overlay (a just-landed owner isn't in runner.requests yet). Default OFF for A/B.
_FF_REP_SAFE = os.environ.get("BFF_FF_REP_SAFE", "0") == "1"
# Diagnostic: count applied redirects whose REP was resolved from the (stale-prone) load-metadata overlay
# vs live runner state, surfaced in the apply log + stats. Cheap; helps confirm the mechanism at con512.
_FF_AUDIT = os.environ.get("BFF_FF_AUDIT", "0") == "1"


def _parse_groups(raw: str | None):
    """Parse BFF_FF_GROUPS ("1,2,3") into a set of group indices, or None for "all eligible".
    Unset/empty/whitespace → None. Ignores blanks so "1, 2," is accepted; a value that parses to
    nothing (e.g. ",,") also yields None rather than silently disabling fusion entirely."""
    if not raw or not raw.strip():
        return None
    out = {int(p) for p in raw.split(",") if p.strip()}
    return out or None


# Restrict fusion to specific KV-cache group indices (comma list, e.g. "1,2,3"). Unset = all
# eligible groups. Compression is very unevenly distributed across depth — see the rationale at
# the filter site in _ff_build_group_layers — so this trades a little potential for a lot of
# producer overhead. Applies to the PRODUCER only; the consumer applies whatever it receives.
_FF_GROUPS = _parse_groups(os.environ.get("BFF_FF_GROUPS"))
# SimHash LSH config (ported from kv_fast_fusion/legacy): NUM_LSH_TABLES banded sub-hashes of
# LSH_BITS_PER_TABLE bits each, from sign() of a fixed-seed random-hyperplane projection.
_LSH_TABLES = int(os.environ.get("BFF_LSH_TABLES", "16"))
# 16 tables x 20 bits: SimHash P(collision)=1-(1-(1-θ/π)^B)^T keeps ~87% recall at cos 0.95 while
# random-pair collisions drop ~1000x vs the old 16/10 (1.5e-2 → 1.5e-5) — near-duplicates still
# merge, dissimilar blocks (the con512 corruption source) no longer do. Retune analysis in the plan.
_LSH_BITS = int(os.environ.get("BFF_LSH_BITS_PER_TABLE", "20"))
# Max rep entries kept per fusion group in the LSH index before LRU-evicting the oldest half.
_LSH_MAX_ENTRIES = int(os.environ.get("BFF_LSH_MAX_ENTRIES", "50000"))
# Cap on candidates verified per block. Candidates/block grows as TABLES*MAX_ENTRIES/2**BITS, so a
# low BITS against a large index makes the verify dominate. >0 keeps only the top-K candidates by
# table-collision count (multi-probe ranking); 0 = uncapped (verify every bucket candidate).
_LSH_MAX_CAND = int(os.environ.get("BFF_LSH_MAX_CANDIDATES", "0"))
_LSH_POWERS = (2 ** torch.arange(_LSH_BITS, dtype=torch.int64)).tolist()
# Dedicated FF ZMQ control channel: D binds (its side-channel base + this offset); P sends redirect
# maps there. Kept separate from the connector's own handshake port so the stock recv thread is
# untouched. TP=1 assumed for M1 (the new setup runs tp_size=1 on both P and D).
_FF_PORT_OFFSET = int(os.environ.get("BFF_MOONCAKE_FF_PORT_OFFSET", "20000"))

# Fusion-stats dump dir, now exported for BOTH roles. Each side drops a per-process file the
# benchmark's collect_bff_stats() merges: the producer's ``bff_stats_<pid>.json`` (blocks seen +
# redirects SHIPPED — an upper bound), the consumer worker's ``bff_apply_stats_<pid>.json`` (how many
# landed, and why the rest didn't), and the decode scheduler's ``bff_decode_stats_<pid>.json`` (the
# real block-pool delta — see fast_fusion_scheduler). Only the producer runs save_kv_layer, so the
# decode side never writes a bff_stats_ file even though the dir is now set there too.
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


def _block_repr(caches, idx: torch.Tensor, jl_holder: list) -> torch.Tensor:
    """Per-layer block representation ``[N, D_repr]`` (float32) for the clustering similarity.

    ``caches`` is one layer's paged cache tensor(s) selected by ``idx``. For standard attention it is a
    single K tensor ``[num_blocks, block_size, kv_heads, head_dim]`` (``kv_layer[0]``); for MLA it is
    the pair ``[compressed/nope, rope]`` (each ``[num_blocks, block_size, 1, dim]``), whose per-block
    reprs are concatenated because they share one physical block. A bare tensor is accepted too.
    ``full`` = exact whole block, ``mean`` = per-token-feature mean, ``proj`` = fixed-seed JL
    projection. ``jl_holder`` caches the lazily-built projection matrix (sized to the concatenated
    width on first call)."""
    if not isinstance(caches, (list, tuple)):
        caches = (caches,)
    n = idx.shape[0]
    if _PD_REPR == "mean":
        parts = [c[idx].float().reshape(n, -1, c.shape[-1]).mean(dim=1) for c in caches]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    parts = [c[idx].float().reshape(n, -1) for c in caches]
    full = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    if _PD_REPR == "proj":
        if jl_holder[0] is None:
            g = torch.Generator(device=full.device)
            g.manual_seed(1234)
            jl_holder[0] = torch.randn(
                full.shape[1], _PD_PROJ_DIM, generator=g, device=full.device, dtype=torch.float32)
        return full @ jl_holder[0]
    return full


def _lsh_get_proj(jl_holder: list, d: int, device) -> torch.Tensor:
    """Fixed-seed random-hyperplane projection ``[d, _LSH_TABLES*_LSH_BITS]`` for SimHash, cached in
    ``jl_holder`` per feature width (deterministic across restarts). Ported from the legacy
    ``_get_simhash_matrix``."""
    m = jl_holder[0]
    if m is None or m.shape[0] != d:
        g = torch.Generator(device="cpu")
        g.manual_seed(20240517)
        m = torch.randn(d, _LSH_TABLES * _LSH_BITS, generator=g, dtype=torch.float32).to(device)
        jl_holder[0] = m
    return m


def _lsh_sub_hashes(vecs_norm: torch.Tensor, proj: torch.Tensor) -> list:
    """Per-row list of ``_LSH_TABLES`` banded sub-hashes (each an int in ``[0, 2**_LSH_BITS)``) from
    the sign bits of ``vecs_norm @ proj``. ``vecs_norm`` is ``[M, d]``; returns a length-M list of
    length-``_LSH_TABLES`` int lists. Mirrors legacy ``_lsh_fingerprint`` (batched).

    Packs on ``vecs_norm``'s device and only then copies: the host only ever needs the ``[M, T]``
    bucket ids, so transferring the unpacked ``[M, T*B]`` bits would move _LSH_BITS x more bytes
    across a device sync that stalls the forward pass."""
    bits = (vecs_norm.float() @ proj > 0).to(torch.int64)                # [M, T*B] (on device)
    powers = torch.tensor(_LSH_POWERS, dtype=torch.int64, device=bits.device)
    packed = (bits.view(-1, _LSH_TABLES, _LSH_BITS) * powers).sum(dim=2)  # [M, T] (on device)
    return packed.cpu().tolist()


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
        # Cross-batch encoded registry (MATRIX backend only): gi -> rolling window of the last
        # _PD_ENCODED_BATCH distinct requests' representative block K-reps (raw mode). Empty/unused
        # under the "lsh" backend, whose pool is bounded by BFF_LSH_MAX_ENTRIES instead.
        self._registry: dict[int, dict] = {}
        # SimHash LSH cross-request index (BFF_PD_CROSS_INDEX="lsh"): gi -> dict with banded bucket
        # tables, a rep-vector store (verify), and per-entry (ext_hash, slot, req_ext). Large-capacity
        # alternative to the bounded _registry, O(N) probe.
        self._lsh: dict[int, dict] = {}
        self._lsh_proj = [None]                # lazy fixed-seed SimHash projection (per feature width)

    def reset_step(self, step_id: int) -> None:
        if step_id != self._cur_step_id:
            self._cur_step_id = step_id
            self._buf.clear()
            self.steps += 1

    def on_layer(
        self,
        gi: int,
        layer_name: str,
        caches,
        group_layer_names: set[str],
        requests: list[tuple],
        tp_group=None,
    ) -> dict[str, list[tuple[int, int, int]]] | None:
        """Accumulate one layer of fusion group ``gi``. ``caches`` is the layer's paged cache tensor —
        a single K tensor for standard attention, or the ``[nope, rope]`` pair for MLA (a bare tensor
        is also accepted). ``requests`` is the ordered list of
        ``(external_id, local_block_ids_for_gi[, has_remote])`` for this step's batch (``has_remote``
        defaults to True when omitted; it gates cross-batch registration to decode-bound requests).
        Returns ``None`` until the group completes, then a dict
        ``{owner_external_id: [(owner_slot, rep_hash, rep_slot), ...]}``.
        """
        if not isinstance(caches, (list, tuple)):
            caches = (caches,)
        dev = caches[0].device
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
            idx = (torch.as_tensor(flat_bids, device=dev, dtype=torch.long)
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
            buf["k_layers"].append(_block_repr(caches, buf["idx"], self._jl))
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
        rep_slot), ...]}``. The cross-request phase (matching against earlier requests' reps before
        within-batch clustering the remainder, then registering the new reps) runs when the "lsh"
        backend is active (tp=1) OR ``_PD_ENCODED_BATCH>0`` for the "matrix" backend; ``_PD_ENCODED_BATCH``
        is the matrix FIFO-window size only. When neither applies it is the within-batch-only path."""
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
            # Cross-request backend + gate. "lsh" (tp=1) runs independently of _PD_ENCODED_BATCH
            # (its pool is bounded by BFF_LSH_MAX_ENTRIES, not the matrix FIFO window). The "matrix"
            # backend (and lsh's tp>1 fallback) is gated on _PD_ENCODED_BATCH > 0, its window size.
            use_lsh = _PD_CROSS_INDEX == "lsh" and tp_group is None
            run_cross = use_lsh or _PD_ENCODED_BATCH > 0
            if not run_cross:
                # ---- within-batch only (legacy path, unchanged behavior) ----
                if _PD_INTRA_REQ_FF:
                    labels = _cluster(buf["k_layers"], torch.as_tensor(flat_req_local, device=dev0))
                    _, redirects = build_group_redirect(labels, flat_req_local, flat_slot)
                    _emit_within(redirects)
                # else: no cross backend and within-batch cc disabled → emit no redirects.
            else:
                # ---- cross-request phase 1: match this step's blocks against earlier requests' reps.
                # "lsh" (default, tp=1): O(N) SimHash bucket probe over a large-capacity index.
                # "matrix" (or any tp>1): the dense concat_cosine_cross_match over the FIFO window.
                cur_concat = cur_sq = cur_cpu = sub_hashes = None
                if use_lsh:
                    cur_concat = torch.cat([Kg.float() for Kg in buf["k_layers"]], dim=1)  # [N, G*D]
                    cur_norm = cur_concat / cur_concat.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    proj = _lsh_get_proj(self._lsh_proj, cur_concat.shape[1], cur_concat.device)
                    sub_hashes = _lsh_sub_hashes(cur_norm, proj)
                    # ONE host copy of the rep vectors per group, reused by probe AND register (both
                    # verify/store against the host-side index). Each .cpu() drains the device
                    # pipeline, so a second copy of the same tensor costs a full sync for nothing.
                    # _lsh_sub_hashes just synced, so this transfer lands on an already-idle device.
                    cur_cpu = cur_norm.detach().cpu().float()
                    matched, hits = self._lsh_probe(
                        gi, cur_cpu, sub_hashes, ext_ids, flat_req_local)
                    for (i, rep_hash, rep_slot) in hits:
                        owner_ext = ext_ids[flat_req_local[i]]
                        send_rows.setdefault(owner_ext, []).append(
                            (int(flat_slot[i]), int(rep_hash), int(rep_slot)))
                        n_cross += 1
                else:
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
                    matched = [False] * N
                    for i, ridx in enumerate(best_idx.tolist()):
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
                if unmatched and _PD_INTRA_REQ_FF:
                    sub_k = [Kg[unmatched] for Kg in buf["k_layers"]]
                    sub_req = [flat_req_local[i] for i in unmatched]
                    sub_slot = [flat_slot[i] for i in unmatched]
                    labels = _cluster(sub_k, torch.as_tensor(sub_req, device=dev0))
                    _, redirects = build_group_redirect(labels, sub_req, sub_slot)
                    _emit_within(redirects)
                    labels_l = labels.tolist()
                    reps_to_register = [unmatched[i] for i in range(len(labels_l))
                                        if labels_l[i] == i]
                elif unmatched:
                    # Within-batch cc disabled: skip the O(N²) clustering and register every unmatched
                    # block as its own rep, so the cross-request index still grows (no within redirects).
                    reps_to_register = list(unmatched)
                # Register this step's new reps into the chosen cross-request index.
                if use_lsh:
                    self._lsh_register(gi, cur_cpu, sub_hashes, ext_ids, flat_req_local,
                                       flat_slot, reps_to_register, buf["ext_has_remote"])
                else:
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

    # -- SimHash LSH cross-request index (BFF_PD_CROSS_INDEX="lsh") -------------------------
    def _lsh_index(self, gi) -> dict:
        """Per-group index. Rep vectors live in ONE contiguous ``mat`` [cap, d] (grown by doubling)
        rather than per-entry tensors, so the probe's verify is a single ``index_select`` + mv instead
        of a per-block ``torch.stack`` over dict values (measured ~5-7x faster, and storing a copy
        avoids pinning each step's whole [N, d] buffer alive via an index view)."""
        idx = self._lsh.get(gi)
        if idx is None:
            idx = {
                "tables": [dict() for _ in range(_LSH_TABLES)],  # bucket_hash -> [row]
                "meta": {},                                      # row -> (ext_hash, slot, req_ext)
                "owner": {},                                     # row -> sub_hashes (for eviction)
                "lru": OrderedDict(),                            # row -> None (oldest first)
                "mat": None,                                     # [cap, d] cpu f32 rep vectors
                "n_rows": 0,
            }
            self._lsh[gi] = idx
        return idx

    def _lsh_size(self, gi) -> int:
        idx = self._lsh.get(gi)
        return 0 if idx is None else idx["n_rows"]

    @staticmethod
    def _lsh_ensure_cap(idx, need: int, d: int) -> None:
        """Grow ``mat`` by doubling until it holds ``need`` rows (lazy: never allocates the full
        _LSH_MAX_ENTRIES x d up front)."""
        mat = idx["mat"]
        if mat is None:
            idx["mat"] = torch.empty(max(256, need), d, dtype=torch.float32)
            return
        if mat.shape[0] >= need:
            return
        cap = mat.shape[0]
        while cap < need:
            cap *= 2
        new = torch.empty(cap, d, dtype=torch.float32)
        new[:idx["n_rows"]] = mat[:idx["n_rows"]]
        idx["mat"] = new

    def _lsh_probe(self, gi, cur_cpu, sub_hashes, ext_ids, flat_req_local):
        """Probe the group's LSH index for each current block; return (matched[list[bool]],
        hits[list[(i, rep_hash, rep_slot)]]). A hit requires a bucket-candidate from a DIFFERENT
        request whose exact cosine with the current block is >= THRESHOLD (best wins).

        ``cur_cpu`` is the caller's single HOST copy of the normalized reps ``[N, d]`` (the index it
        verifies against lives on the host)."""
        idx = self._lsh.get(gi)
        n = cur_cpu.shape[0]
        matched = [False] * n
        hits: list[tuple[int, int, int]] = []
        if idx is None or not idx["n_rows"]:
            return matched, hits
        tables, meta, lru = idx["tables"], idx["meta"], idx["lru"]
        mat = idx["mat"]
        for i in range(n):
            owner_ext = ext_ids[flat_req_local[i]]
            if _LSH_MAX_CAND > 0:
                # Multi-probe ranking: prefer candidates colliding in the MOST tables (more likely
                # similar), then keep at most _LSH_MAX_CAND of them.
                counts: Counter = Counter()
                for t, h in enumerate(sub_hashes[i]):
                    counts.update(tables[t].get(h, ()))
                cand_rows = [r for r, _ in counts.most_common()
                             if meta[r][2] != owner_ext][:_LSH_MAX_CAND]
            else:
                cand: set = set()
                for t, h in enumerate(sub_hashes[i]):
                    cand.update(tables[t].get(h, ()))
                # Rows ARE the index into mat, so no id→row lookup — and the buckets only ever hold
                # live rows (evict+compact run together), so no liveness check either.
                cand_rows = [r for r in cand if meta[r][2] != owner_ext]
            if not cand_rows:
                continue
            rows = torch.tensor(cand_rows, dtype=torch.long)
            sims = mat.index_select(0, rows) @ cur_cpu[i]          # [C]
            best_val, best_j = sims.max(dim=0)
            if best_val.item() > THRESHOLD:
                row = cand_rows[int(best_j.item())]
                rep_hash, rep_slot, _ = meta[row]
                hits.append((i, rep_hash, rep_slot))
                matched[i] = True
                lru.move_to_end(row)                              # LRU touch
        return matched, hits

    def _lsh_register(self, gi, cur_cpu, sub_hashes, ext_ids, flat_req_local, flat_slot,
                      rep_flats, ext_has_remote) -> None:
        """Insert this step's unmatched reps (decode-bound only) into the LSH index; LRU-evict the
        oldest half when over the per-group cap. ``cur_cpu`` is the caller's single HOST copy of the
        normalized reps ``[N, d]`` (shared with :meth:`_lsh_probe` — one sync per group, not two)."""
        if not rep_flats:
            return
        idx = self._lsh_index(gi)
        if idx["n_rows"] >= _LSH_MAX_ENTRIES:
            for row in list(itertools.islice(idx["lru"].keys(), max(1, _LSH_MAX_ENTRIES // 2))):
                self._lsh_evict(idx, row)
            self._lsh_compact(idx)
        # Bind AFTER compaction — it rebinds idx["meta"]/["owner"]/["lru"]/["mat"] to new objects.
        add = [f for f in rep_flats if ext_has_remote[flat_req_local[f]]]
        if not add:
            return
        self._lsh_ensure_cap(idx, idx["n_rows"] + len(add), cur_cpu.shape[1])
        tables, meta, owner, lru = idx["tables"], idx["meta"], idx["owner"], idx["lru"]
        mat = idx["mat"]
        for f in add:
            ext = ext_ids[flat_req_local[f]]
            row = idx["n_rows"]
            idx["n_rows"] += 1
            mat[row] = cur_cpu[f]                    # copy (not a view into this step's buffer)
            meta[row] = (_ext_hash(ext), int(flat_slot[f]), ext)
            sh = sub_hashes[f]
            owner[row] = sh
            lru[row] = None
            for t, h in enumerate(sh):
                tables[t].setdefault(h, []).append(row)

    @staticmethod
    def _lsh_evict(idx, row) -> None:
        """Drop one row's metadata + bucket memberships. The ``mat`` row itself is reclaimed by the
        _lsh_compact that always follows a batch evict."""
        sh = idx["owner"].pop(row, None)
        idx["meta"].pop(row, None)
        idx["lru"].pop(row, None)
        if sh is not None:
            for t, h in enumerate(sh):
                bucket = idx["tables"][t].get(h)
                if bucket:
                    try:
                        bucket.remove(row)
                    except ValueError:
                        pass
                    if not bucket:
                        del idx["tables"][t][h]

    @staticmethod
    def _lsh_compact(idx) -> None:
        """Renumber the surviving rows to 0..n-1 after a batch evict: gather them in ``mat`` and
        rebuild meta/owner/lru/tables under the new row ids (LRU order preserved)."""
        survivors = [r for r in range(idx["n_rows"]) if r in idx["meta"]]
        if not survivors:
            idx["mat"] = None
            idx["meta"] = {}
            idx["owner"] = {}
            idx["lru"] = OrderedDict()
            idx["tables"] = [dict() for _ in range(_LSH_TABLES)]
            idx["n_rows"] = 0
            return
        remap = {old: new for new, old in enumerate(survivors)}
        idx["mat"] = idx["mat"].index_select(
            0, torch.tensor(survivors, dtype=torch.long)).contiguous()
        idx["meta"] = {remap[r]: v for r, v in idx["meta"].items()}
        owner = {remap[r]: v for r, v in idx["owner"].items()}
        idx["owner"] = owner
        idx["lru"] = OrderedDict((remap[r], None) for r in idx["lru"])
        tables = [dict() for _ in range(_LSH_TABLES)]
        for r, sh in owner.items():
            for t, h in enumerate(sh):
                tables[t].setdefault(h, []).append(r)
        idx["tables"] = tables
        idx["n_rows"] = len(survivors)

    def dump_stats(self, stats_dir: str) -> None:
        """Write this producer's cumulative fusion counters to ``bff_stats_<pid>.json`` in
        ``stats_dir``. The benchmark's collect_bff_stats() merges these after the run.

        NOTE these are PRODUCER-side counters: ``redirects_emitted`` is the number of redirect ROWS
        SHIPPED, which is an UPPER BOUND on blocks actually freed — a row frees a block only if the
        decode side resolves it (rep still resident, owner resident, merge not dropped). Hence
        ``compression_potential_factor``, not a measurement. The REAL number comes from the decode
        side's block-pool delta (``bff_decode_stats_<pid>.json``, see fast_fusion_scheduler)."""
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
                "redirects_emitted": tot_r,
                "compression_potential_factor": _factor(tot_b, tot_r),
                "compression_potential_per_group": {
                    str(gi): _factor(self.blk_total[gi], self.redir_total.get(gi, 0))
                    for gi in sorted(self.blk_total)},
                # Cross-batch lift: how many redirects came from the rolling registry (earlier
                # batches) vs the within-step batch, plus the window size and current registry size.
                "encoded_batch_size": _PD_ENCODED_BATCH,
                "cross_index": _PD_CROSS_INDEX,
                "intra_req_ff": _PD_INTRA_REQ_FF,
                "rep_safe": _FF_REP_SAFE,
                "cross_batch_redirects": self.cross_redir_total,
                "within_batch_redirects": self.within_redir_total,
                "registry_blocks": {str(gi): self._registry_size(gi)
                                    for gi in sorted(self._registry)},
                "lsh_index_blocks": {str(gi): self._lsh_size(gi) for gi in sorted(self._lsh)},
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
    rep_ext2blocks: dict[str, list[list[int]]] | None = None,
    rep_hash2ext: dict[int, str] | None = None,
) -> tuple[list[int] | None, int, int, int]:
    """Consumer-side: turn shipped redirect ``rows`` into the owner's new (deduped) block table.

    ``ext2blocks`` maps external id → per-group D-physical block ids (from the decode runner, plus
    this step's load metadata); ``hash2ext`` maps ``_ext_hash`` → external id. Returns
    ``(new_owner_blocks, n_applied, n_unresolved, n_owner_missing)``; ``new_owner_blocks`` is ``None``
    when nothing changed. ``n_owner_missing`` counts rows we couldn't even attempt because the OWNER
    itself was not resolvable (distinct from ``n_unresolved`` = the REP not resolvable) — kept
    separate so the apply log can tell owner-residency problems from rep-residency problems. Port of
    the resolve loop in the NCCL connector's ``_pd_consumer_apply`` (raw mode).

    ``rep_ext2blocks``/``rep_hash2ext`` (optional): a SEPARATE, usually smaller, set to resolve the
    REP from — used by the rep-lifetime fix (BFF_FF_REP_SAFE) to restrict reps to live runner state so a
    stale overlay rep counts as ``unresolved`` (owner keeps its own block) instead of a wrong repoint.
    Defaults to the owner set (``ext2blocks``/``hash2ext``) → original behavior."""
    if rep_ext2blocks is None:
        rep_ext2blocks = ext2blocks
    if rep_hash2ext is None:
        rep_hash2ext = hash2ext
    owner_groups = ext2blocks.get(owner_ext_id)
    if owner_groups is None or gi >= len(owner_groups):
        return None, 0, 0, len(rows)                   # owner not resident → owner-miss, not rep-miss
    owner_blocks = list(owner_groups[gi])
    n_applied = n_unresolved = 0
    changed = False
    for owner_slot, rep_hash, rep_slot in rows:
        rep_ext = rep_hash2ext.get(int(rep_hash))
        if rep_ext is None or rep_ext not in rep_ext2blocks:
            n_unresolved += 1                          # rep not (yet) resident on D → can't share
            continue
        rep_groups = rep_ext2blocks[rep_ext]
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


def _ff_write_runner_block_table(runner, rid, gi, new_blocks) -> bool:
    """Write the redirected per-group block table into the runner's worker-side mirror so the
    forward reads the shared blocks. Ports the NCCL connector's ``_pd_write_runner_block_table``.

    Returns True iff the device table was actually rewritten. The caller MUST couple the block
    free to this return: a rid that is not in this step's ``input_batch`` early-returns here without
    rewriting, and freeing its blocks anyway leaves the request pointing at freed-then-reallocated
    KV (aliasing → ramble → F1 and throughput both collapse). Free only when this returns True.

    Module-level (like ``resolve_redirect_rows``) so the coupling is unit-testable off-NPU with a
    fake runner — this is the exact invariant the con512 corruption violated."""
    ridx = runner.input_batch.req_id_to_index.get(rid)
    if ridx is None:
        return False
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
    return True


def _classify_owner_miss(ext_id: str, ever_snapshotted) -> str:
    """Diagnostic (BFF_FF_AUDIT): why an owner whose recv just completed is unresolvable at apply.

    ``owner_unresident`` fires when the owner is in ``just_recv_ext`` yet absent from the resolution
    set (runner.requests ∪ the ``_ff_load_blocks`` overlay). Exactly two causes:
      * "never_snap"  — its load metadata never put block ids into the overlay (a snapshot keying/
                        timing gap); fix = snapshot more durably.
      * "pruned"      — it WAS snapshotted, then removed before its recv landed (resident-delete or
                        the stale-entry prune fired too early); fix = tighten the prune guard.
    ``ever_snapshotted`` is the cumulative set of exts seen by ``_ff_snapshot_load_meta``."""
    return "pruned" if ext_id in ever_snapshotted else "never_snap"


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
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        MLAAttentionSpec,
    )

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
            self._ff_mla_groups: set[int] | None = None
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
                # Persistent load-metadata block tables (ext_id -> (rid, local_block_ids)). The
                # consumer load metadata is transient (scheduler clears _reqs_need_recv each step) but
                # a request's KV recv completes a LATER step; accumulate here so the owner is still
                # resolvable when its recv finally lands. Pruned as requests become resident/finish.
                self._ff_load_blocks: dict[str, tuple[str, list]] = {}
                # Diagnostic (BFF_FF_AUDIT only): cumulative set of exts ever put into the overlay by
                # _ff_snapshot_load_meta, so an owner-miss can be split into never-snapshotted vs
                # snapshotted-then-pruned. Unbounded, so only populated under _FF_AUDIT.
                self._ff_ever_snapshotted: set[str] = set()
                # Cumulative apply outcomes. The producer only knows how many redirects it SHIPPED;
                # these say how many actually landed and, when they didn't, why — which is what
                # explains the emitted→freed gap in the benchmark report.
                self._ff_apply_totals: dict[str, int] = {
                    "applied": 0, "reps_unresolved": 0, "owner_unresident": 0,
                    "owners_deferred": 0, "owners_dropped_post_decode": 0, "apply_calls": 0,
                    # Resolved rows whose device write failed (owner not in input_batch) so the free
                    # was withheld — the part of `applied` that produced NO actual free.
                    "owner_not_written": 0,
                    "load_blocks_pruned": 0,
                    # BFF_FF_AUDIT split of owner_unresident (see _classify_owner_miss).
                    "ownmiss_never_snap": 0, "ownmiss_pruned": 0}

        def _ff_build_group_layers(self, worker) -> None:
            """Map fusion group index → layer names + the set of fusion groups (full-attention,
            gi>0), from the worker's registered ``layer_metadata`` + kv-cache specs."""
            group_layers: dict[int, set[str]] = {}
            for ln, lm in worker.layer_metadata.items():
                group_layers.setdefault(lm.tensor_group_idx[0], set()).add(ln)
            fusion_groups = set()
            mla_groups = set()
            for gi in group_layers:
                if gi <= 0 or gi >= len(worker.kv_cache_specs):
                    continue
                spec = worker.kv_cache_specs[gi]
                # MLAAttentionSpec subclasses FullAttentionSpec, so MLA groups pass this gate too; we
                # additionally flag them so on_layer clusters on the full latent+rope key (not just
                # the compressed latent kv_layer[0]).
                if isinstance(spec, FullAttentionSpec) and worker.kernel_block_size_scale[gi] == 1:
                    fusion_groups.add(gi)
                    if isinstance(spec, MLAAttentionSpec):
                        mla_groups.add(gi)
            # BFF_FF_GROUPS restricts fusion to the groups that actually pay. Measured at con512:
            # groups 1-3 produce 90.9% of all redirects while holding 23% of the LSH index; groups
            # 4-6 produce 9.1% while holding 77% (deep layers are content-specific, so almost
            # nothing matches, so nearly every block registers as a new rep — which then has to be
            # probed against). Excluding a group skips its clustering/hash/probe/register entirely;
            # its blocks still transfer normally, so the only cost is the lost (negligible)
            # compression. Empty/unset = all eligible groups (previous behavior).
            selected = _FF_GROUPS
            if selected is not None:
                skipped = sorted(fusion_groups - selected)
                fusion_groups = fusion_groups & selected
                if skipped:
                    logger.info("BFF Mooncake: BFF_FF_GROUPS excludes fusion groups %s", skipped)
            self._ff_group_layers = group_layers
            self._ff_fusion_groups = fusion_groups
            self._ff_mla_groups = mla_groups
            logger.info("BFF Mooncake: fusion groups=%s (mla=%s) (of %d groups)",
                        sorted(fusion_groups), sorted(mla_groups), len(group_layers))

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
            # Standard attention clusters on K (kv_layer[0]) only. MLA's key is split across two
            # latent tensors (kv_layer[0]=compressed/nope, kv_layer[1]=rope) that share one physical
            # block, so both must be compared or the redirect aliases the un-compared rope cache.
            if gi in self._ff_mla_groups and len(kv_layer) > 1:
                caches = [kv_layer[0], kv_layer[1]]
            else:
                caches = [kv_layer[0]]
            send_rows = self._ff_producer.on_layer(
                gi, layer_name, caches, self._ff_group_layers[gi], requests)
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
                    # Snapshot this step's (transient) load metadata EVERY step — the owner's recv
                    # completes a later step, by which point its metadata has been cleared.
                    self._ff_snapshot_load_meta(finished_req_ids)
                    # `recving` = request ids whose KV *just* fully landed this step. Applying a
                    # redirect only for those requests guarantees we repoint+free BEFORE the owner
                    # decodes — mirroring the NCCL connector's load-time apply. Draining and applying
                    # for already-decoding requests frees in-use blocks → pool aliasing → global KV
                    # corruption (every request garbage → no EOS → runs to max_tokens).
                    self._ff_apply_pending(recving)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("BFF Mooncake consumer apply failed: %s", e)
            return sending, recving

        def _ff_snapshot_load_meta(self, finished_req_ids: set[str]) -> None:
            """Accumulate this step's consumer load-metadata block tables into ``_ff_load_blocks``.
            The scheduler clears its recv list each step, so a request's D-side ``local_block_ids`` is
            only in the metadata the step it is scheduled — but its recv completes later. Persisting
            them here keeps the owner resolvable at recv-completion. Pruned on finish (and, in
            ``_ff_apply_pending``, once the request becomes resident in ``runner.requests``)."""
            meta = self._connector_metadata
            for rid, rm in getattr(meta, "requests", {}).items():
                lb = getattr(rm, "local_block_ids", None)
                if lb:
                    ext = get_external_request_id(rid)
                    self._ff_load_blocks[ext] = (rid, lb)
                    if _FF_AUDIT:
                        # Record that this ext DID enter the overlay, so a later owner-miss can be
                        # attributed to a prune rather than a missing snapshot (see _classify_owner_miss).
                        self._ff_ever_snapshotted.add(ext)
            for fid in finished_req_ids:
                self._ff_load_blocks.pop(get_external_request_id(fid), None)

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
            # RESOLUTION set = residency set overlaid with the accumulated load-metadata block tables
            # (`_ff_load_blocks`, filled every step by _ff_snapshot_load_meta). A request whose KV just
            # landed (`recving`) is not yet in runner.requests and its load metadata was cleared a
            # step or more ago; without this overlay resolve_redirect_rows can't find the owner (or a
            # not-yet-decoding rep) → every row counts as owner-missing. Skip ext already resident —
            # the runner's live table wins — and prune those from the store to keep it bounded.
            res_ext2blocks = dict(ext2blocks)
            res_hash2ext = dict(hash2ext)
            for ext, (rid, lb) in list(self._ff_load_blocks.items()):
                if ext in ext2blocks:
                    del self._ff_load_blocks[ext]   # now resident → covered by runner.requests
                    continue
                res_ext2blocks[ext] = lb
                res_hash2ext[_ext_hash(ext)] = ext
                rid_by_ext.setdefault(ext, rid)
            # Prune entries that are neither resident nor still expected: the request left
            # runner.requests (finished/preempted) AND no redirect is pending for it, so nothing will
            # ever resolve against it. Without this the overlay grows unboundedly and keeps handing
            # out block tables for requests whose blocks have been recycled. The `ext not in pending`
            # guard is what makes this safe — a queued redirect keeps its owner resolvable.
            n_load_pruned = 0
            for ext in [e for e, (rid, _lb) in self._ff_load_blocks.items()
                        if rid not in getattr(runner, "requests", {}) and e not in pending]:
                del self._ff_load_blocks[ext]
                res_ext2blocks.pop(ext, None)
                res_hash2ext.pop(_ext_hash(ext), None)
                n_load_pruned += 1
            if n_load_pruned and _FF_AUDIT:
                logger.warning("BFF: pruned %d stale _ff_load_blocks entries (remaining=%d)",
                               n_load_pruned, len(self._ff_load_blocks))
            just_recv_ext = {get_external_request_id(rid) for rid in recving}
            updated: dict[str, dict[int, list[int]]] = {}
            n_applied = n_unresolved = n_owner_missing = n_deferred = n_dropped = 0
            # Rows that RESOLVED but whose device write failed (owner not in this step's input_batch),
            # so the free was correctly withheld. Invisible in `applied` (incremented by
            # resolve_redirect_rows before the write is attempted), so without this a withheld free
            # would look like successful compression. On a healthy run this is > 0.
            n_owner_not_written = 0
            n_ownmiss_never_snap = n_ownmiss_pruned = 0
            leftover: dict[str, dict[int, list]] = {}
            # Rep-lifetime fix: resolve the REP only from LIVE runner state (ext2blocks/hash2ext), not the
            # load-metadata overlay (res_*), so a stale/recycled rep counts as unresolved (owner keeps its
            # own block) instead of being repointed to another request's live KV. Owners still use res_*
            # (a just-landed owner isn't in runner.requests yet). Off → original behavior (rep from res_*).
            rep_e2b = ext2blocks if _FF_REP_SAFE else None
            rep_h2e = hash2ext if _FF_REP_SAFE else None
            for ext_id, groups in pending.items():
                if ext_id in just_recv_ext:
                    # Owner's KV just landed and it has not decoded yet → safe to repoint + free.
                    # Diagnostic: an owner in just_recv but not in the resolution set will owner-miss
                    # on every group below; classify WHY once, before the per-group loop.
                    if _FF_AUDIT and ext_id not in res_ext2blocks:
                        cause = _classify_owner_miss(ext_id, self._ff_ever_snapshotted)
                        if cause == "pruned":
                            n_ownmiss_pruned += 1
                        else:
                            n_ownmiss_never_snap += 1
                        if (n_ownmiss_pruned + n_ownmiss_never_snap) <= 8:
                            logger.warning("BFF owner-miss | ext=%s | cause=%s | in_runner=%s",
                                           ext_id, cause, ext_id in ext2blocks)
                    for gi, rows in groups.items():
                        new_blocks, na, nu, nom = resolve_redirect_rows(
                            res_ext2blocks, res_hash2ext, ext_id, gi, rows,
                            rep_ext2blocks=rep_e2b, rep_hash2ext=rep_h2e)
                        n_applied += na
                        n_unresolved += nu
                        n_owner_missing += nom
                        if new_blocks is not None:
                            rid = rid_by_ext.get(ext_id)
                            # Free ONLY if the device block table was actually rewritten. An owner
                            # whose recv just completed is often not yet in this step's input_batch
                            # (recving ≠ running): the write early-returns, but freeing its blocks
                            # anyway leaves it pointing at freed-then-reallocated KV → aliasing →
                            # ramble → F1 and throughput both drop. Coupling free to write-success
                            # is the correctness invariant. A not-written owner is dropped, not
                            # re-queued: the apply window is "recv just completed" (just_recv_ext),
                            # which will not recur for it, so retry is unsafe anyway — this row
                            # costs compression, never correctness.
                            if rid is not None and _ff_write_runner_block_table(
                                    runner, rid, gi, new_blocks):
                                updated.setdefault(rid, {})[gi] = new_blocks
                            else:
                                n_owner_not_written += 1
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
            if n_applied or n_unresolved or n_owner_missing or n_dropped or _PD_DEBUG or _FF_AUDIT:
                logger.info("BFF Mooncake apply | redirects_applied=%d | owner_unresident=%d | "
                            "reps_unresolved=%d | owners_deferred=%d | owners_dropped_post_decode=%d | "
                            "owner_not_written=%d | load_pruned=%d | ownmiss_never_snap=%d | "
                            "ownmiss_pruned=%d | rep_safe=%d",
                            n_applied, n_owner_missing, n_unresolved, n_deferred,
                            n_dropped, n_owner_not_written, n_load_pruned,
                            n_ownmiss_never_snap, n_ownmiss_pruned, int(_FF_REP_SAFE))
            self._ff_record_apply(n_applied, n_unresolved, n_owner_missing, n_deferred, n_dropped,
                                  n_owner_not_written, n_load_pruned,
                                  n_ownmiss_never_snap, n_ownmiss_pruned)

        def _ff_record_apply(self, n_applied, n_unresolved, n_owner_missing, n_deferred,
                             n_dropped, n_owner_not_written=0, n_load_pruned=0,
                             n_ownmiss_never_snap=0, n_ownmiss_pruned=0) -> None:
            """Accumulate this step's apply outcomes and periodically dump them, so the benchmark can
            attribute the gap between redirects shipped and blocks actually freed."""
            t = self._ff_apply_totals
            t["applied"] += n_applied
            t["reps_unresolved"] += n_unresolved
            t["owner_unresident"] += n_owner_missing
            t["owners_deferred"] += n_deferred
            t["owners_dropped_post_decode"] += n_dropped
            t["owner_not_written"] += n_owner_not_written
            t["load_blocks_pruned"] += n_load_pruned
            t["ownmiss_never_snap"] += n_ownmiss_never_snap
            t["ownmiss_pruned"] += n_ownmiss_pruned
            t["apply_calls"] += 1
            if not _PD_STATS_DIR:
                return
            if t["apply_calls"] != 1 and t["apply_calls"] % _PD_STATS_EVERY:
                return
            try:
                path = os.path.join(_PD_STATS_DIR, f"bff_apply_stats_{os.getpid()}.json")
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"pid": os.getpid(), **t}, f)
                os.replace(tmp, path)
            except Exception as e:  # pragma: no cover - defensive (must never break the transfer)
                logger.warning("BFF Mooncake: could not dump apply stats: %s", e)


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
