"""Fast-Fusion-aware Mooncake connector for GPU/CUDA P/D disaggregation.

This is the Mooncake counterpart of :mod:`kv_fast_fusion.connectors.p2p_nccl_connector_ff`
for the GPU setup driven by ``examples/online_serving/.../disagg_bff_p2p_nccl_xpyd.sh``:
same BFF semantics, different transport. It subclasses vLLM's mainline
``MooncakeConnector`` (RDMA **pull**: D reads P's KV out of registered memory via the
Mooncake Transfer Engine) and adds the two things BFF needs.

**1. Per-KV-cache-group block tables.** Stock Mooncake is single-group: the decode ships
ONE flat ``local_block_ids`` list and ``_build_transfer_params`` applies it to *every*
registered layer base address. BFF splits the model into a warmup group (first/last 2
layers, sliding window) plus ``BFF_GROUP_SIZE``-packed fusion groups, each with its **own**
block table — so one flat list transfers the wrong physical blocks for every fusion layer.
Here the block ids are per-group end to end (``get_unhashed_block_ids_all_groups`` on D,
``request_finished_all_groups`` on P), and each registered base address — which is an
*allocation* shared by one layer from every group, not a single layer — is filled with the
blocks of all the groups living in it (``base_addr_groups``).

**2. Connector-level fusion + redirect propagation.** Identical algorithm to the NCCL
connector: as KV streams through ``save_kv_layer`` the producer accumulates each fusion
group's per-layer K representations, and on the group's last layer clusters them
(within-batch ``cc``/``nr_tree`` plus the optional cross-batch registry) into per-request
redirect rows ``(owner_slot, rep_transfer_hash, rep_slot)``. D applies them by pointing the
owner's block-table entries at the representative's physical blocks and freeing the
redundant copies through the existing BFF merge channel.

Two deliberate differences from the NCCL connector, both forced by pull-vs-push:

* **Delivery.** NCCL pushes the map as a side-channel tensor per (request, group) and must
  therefore always send a ``[[-1,-1,-1]]`` sentinel, because the consumer's ``recv_tensor``
  blocks. Here the rows ride the **existing P→D control plane**: they are attached to the
  ``MooncakeXferResponse`` that already acknowledges each completed transfer (see
  ``_FFResponseEncoder``). Delivery is therefore inherently synchronized with the KV
  arrival, needs no extra port or thread, and a missing map is simply "no sharing" — so
  there are no sentinels and no way to deadlock.
* **Apply timing.** NCCL applies inside ``start_load_kv`` of the step that loads the KV.
  Under pull, the recv completes *between* steps; the request is not in the worker's
  ``input_batch`` until the scheduler runs it. So rows are held and applied at the top of
  the first ``start_load_kv`` in which the owner is actually batched — i.e. still strictly
  **before** its first decode forward reads the block table (applying later corrupts KV).

``BFF_SCALE_MODE=raw`` only: the transferred KV must be byte-exact (no per-block scales to
ship). ``ratio`` falls back to ``raw`` here with a warning.

Everything below the ``_MOONCAKE_AVAILABLE`` gate needs the ``mooncake`` package; the pure
producer/consumer logic above it is device- and transport-free so it imports (and unit
tests) on any box.
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger

from kv_fast_fusion import pd_lsh

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger("vllm.mooncake_connector_ff")

_UNSET = object()   # sentinel for lazily-resolved cached values (e.g. the TP process group)

# ---------------------------------------------------------------------------------------
# Env knobs — deliberately the SAME names as the NCCL connector so one launch script can
# drive either transport and the stats scraper needs no change.
# ---------------------------------------------------------------------------------------
_BFF_PD_FUSE = os.environ.get("BFF_PD_FUSE", "0") == "1"
# Within-batch clustering: nr_tree (butterfly, full precision) or cc.
_PD_MERGE = os.environ.get("BFF_PD_MERGE", "nr_tree")
# Per-layer block representation for the clustering similarity (producer-only):
#   full → exact cosine over the flattened block-K; proj → JL projection; mean → head_dim mean.
_PD_REPR = os.environ.get("BFF_PD_REPR", "full")
_PD_PROJ_DIM = int(os.environ.get("BFF_PD_PROJ_DIM", "512"))
# Cross-batch fusion window, in REQUESTS. >0 keeps a rolling registry of the last N requests'
# rep blocks per fusion group so a current block can redirect to a rep from an earlier batch
# (still resident + decoding on D). 0 = within-batch only.
_PD_ENCODED_BATCH = int(os.environ.get("BFF_PD_ENCODED_BATCH_SIZE", "0"))
# Cross-request index backend: "matrix" (the dense concat_cosine_cross_match over the bounded FIFO
# window set by BFF_PD_ENCODED_BATCH_SIZE) or "lsh" (SimHash banded index — O(N) probe over a much
# larger pool, bounded by BFF_LSH_MAX_ENTRIES instead of a request window). Within-batch clustering
# is unaffected either way. Defaults to matrix here so existing measurements stay reproducible;
# kv_fast_fusion_ascend defaults to lsh.
_PD_CROSS_INDEX = os.environ.get("BFF_PD_CROSS_INDEX", "matrix").lower()
# Per-group merge thresholds, overriding the global BFF_THRESHOLD. See _parse_thresholds.
_THRESHOLD_G: dict[int, float] = {}
# Similarity audit: sample RANDOM cross-request block pairs for the first _PD_AUDIT_STEPS fusion
# steps of each group and report their cosine quantiles. This measures the group's similarity floor,
# which is what a threshold has to clear to mean anything. Producer-only, off by default, and it
# stops after the sampled steps so a long run pays nothing.
_PD_AUDIT = os.environ.get("BFF_PD_AUDIT", "0") == "1"
_PD_AUDIT_STEPS = int(os.environ.get("BFF_PD_AUDIT_STEPS", "8"))
_PD_AUDIT_PAIRS = int(os.environ.get("BFF_PD_AUDIT_PAIRS", "512"))
# raw is the only transfer-safe mode for P/D (no KV mutation, no scales to ship). The NCCL
# connector additionally supports `ratio` via a second side-channel tensor; here the rows ride
# the transfer ACK, which carries no float payload — so ratio degrades to raw (warned once).
_PD_SCALE_MODE = os.environ.get("BFF_SCALE_MODE", "raw").lower()
# Cheap summary logs (per-step fuse/apply lines).
_PD_DEBUG = os.environ.get("BFF_PD_DEBUG", "0") == "1"
# Producer stats-dump cadence + directory (`bff_stats_<pid>.json`, read by the shell post-run).
_PD_FUSE_LOG_EVERY = int(os.environ.get("BFF_PD_FUSE_LOG_EVERY", "50"))
# Wall-clock backstop for the dump. A step-count-only cadence silently freezes the file when the
# step RATE drops: BFF_FF_GROUPS=1 accrues one step per forward pass instead of one per group, so a
# run that used to reach step 1450 stops at ~25 and the JSON keeps reporting the step-1 snapshot —
# i.e. the cold-start step, alone, presented as the run average.
_PD_STATS_DUMP_SEC = float(os.environ.get("BFF_PD_STATS_DUMP_SEC", "10"))
_PD_STATS_DIR = os.environ.get("BFF_PD_STATS_DIR", ".")
# How many steps a consumer holds an arrived redirect map waiting for its owner to be batched.
# The owner normally lands in input_batch the step after its recv completes; a handful of steps
# of slack covers scheduler back-pressure without letting stale maps accumulate.
_FF_APPLY_MAX_AGE = int(os.environ.get("BFF_FF_APPLY_MAX_AGE", "16"))
# Cap on the producer's undelivered redirect maps (one per in-flight request), so aborted requests
# whose ACK never comes cannot grow the map without bound over a long run.
_FF_ROWS_MAX_PENDING = int(os.environ.get("BFF_FF_ROWS_MAX_PENDING", "4096"))
# How long D keeps re-advertising a finished transfer id to producers, and the hard cap on that
# list. D pulls continuously at saturation, so a few seconds covers every actively-serving producer
# several times over; a producer that pulls nothing in that window is fusing nothing either.
_FF_DONE_TID_TTL = float(os.environ.get("BFF_FF_DONE_TID_TTL", "10"))
_FF_DONE_TID_MAX = int(os.environ.get("BFF_FF_DONE_TID_MAX", "4096"))


def _parse_thresholds(raw: str | None) -> dict[int, float]:
    """Parse ``BFF_THRESHOLD_G`` ("1:0.97,2:0.90") into ``{group: threshold}``.

    One bar for every group is a guess, and the wrong one: a group's blocks have a natural
    similarity FLOOR (early layers share a large common component, so arbitrary blocks already sit
    near cosine 0.9), and a threshold under that floor merges everything indiscriminately. Groups
    not listed fall back to ``BFF_THRESHOLD``. Run with ``BFF_PD_AUDIT=1`` to measure the floor
    instead of guessing it."""
    out: dict[int, float] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gi, thr = part.split(":")
            out[int(gi)] = float(thr)
        except ValueError:
            logger.warning("BFF: ignoring malformed BFF_THRESHOLD_G entry %r", part)
    return out


def _global_threshold() -> float | None:
    """``BFF_THRESHOLD`` as the run actually resolved it, recorded in the stats so the collector can
    state the substitution-error budget it implies (see ``pd_lsh.rel_err``)."""
    try:
        from kv_fast_fusion.constants import THRESHOLD
        return float(THRESHOLD)
    except Exception:  # pragma: no cover - constants unavailable in some import paths
        return None


def _pcts(samples) -> dict:
    """p50/p90/p99/max over a sample (empty dict when there are none)."""
    xs = sorted(samples)
    if not xs:
        return {}
    def _q(q):
        return xs[min(len(xs) - 1, int(len(xs) * q))]
    return {"p50": _q(0.5), "p90": _q(0.9), "p99": _q(0.99), "max": xs[-1], "n": len(xs)}


def _parse_groups(raw: str | None):
    """Parse ``BFF_FF_GROUPS`` ("1,2,3") into a set of KV-cache group indices, or None for "all
    eligible". Unset/empty/whitespace → None. Blanks are ignored so "1, 2," is accepted; a value that
    parses to nothing (e.g. ",,") also yields None rather than silently disabling fusion entirely.

    ``"none"``/``"off"`` → an EMPTY set, which is deliberately different from None: it keeps the
    KV-cache group split and every other BFF patch in place while selecting no fusion groups, so
    nothing is ever clustered or redirected. That is the control arm separating the cost of the group
    split from the cost of fusion — without it, comparing against vanilla moves both variables at
    once and no throughput delta can be attributed to either.

    Mirrors ``_parse_groups`` in kv_fast_fusion_ascend/connectors/mooncake_layerwise_connector_ff.py.
    """
    if not raw or not raw.strip():
        return None
    if raw.strip().lower() in ("none", "off"):
        return frozenset()
    out = {int(p) for p in raw.split(",") if p.strip()}
    return out or None


# Restrict fusion to specific KV-cache group indices (comma list, e.g. "1,2,3"); unset = all
# eligible. Compression is very unevenly distributed across depth — a 2026-08-13 GPU run measured
# g1=171.6x g2=1.9x g3=4.8x but g4=1.01x g5=1.45x g6=1.01x, i.e. the deep groups pay full per-layer
# repr + clustering + registry cost for essentially nothing (the Ascend port records the same shape:
# groups 1-3 produce 90.9% of redirects while holding 23% of the index). Excluding a group skips its
# clustering/hash/probe/register entirely; its blocks still transfer normally, so the only cost is
# the forgone compression. PRODUCER only — the consumer applies whatever it receives.
_FF_GROUPS = _parse_groups(os.environ.get("BFF_FF_GROUPS"))
_THRESHOLD_G = _parse_thresholds(os.environ.get("BFF_THRESHOLD_G"))


def _tid_hash(transfer_id: str) -> int:
    """Process-stable positive int64 hash of a Mooncake ``transfer_id``.

    The transfer_id is the router-assigned key that P and D *both* see for the same request
    (their internal ``request_id``s differ — vLLM appends a per-server random suffix), so it is
    the natural P/D-stable identity here; the NCCL connector has to reconstruct the equivalent by
    stripping that suffix (``_pd_key``). Python's ``hash()`` is salted per process and cannot be
    shared across P and D, hence blake2b."""
    h = hashlib.blake2b(transfer_id.encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------------------
# Consumer-side pure glue (module-level so it is unit-testable with a fake runner, off-GPU).
# ---------------------------------------------------------------------------------------
def resolve_redirect_rows(
    rid2blocks: dict[str, Any],
    hash2rid: dict[int, str],
    owner_rid: str,
    gi: int,
    rows: list[tuple[int, int, int]],
) -> tuple[list[int] | None, int, int]:
    """Turn shipped redirect ``rows`` into the owner's new (deduped) per-group block table.

    ``rid2blocks`` maps decode request id → per-group D-physical block ids; ``hash2rid`` maps
    ``_tid_hash(transfer_id)`` → decode request id. Returns ``(new_owner_blocks, n_applied,
    n_unresolved)`` with ``new_owner_blocks is None`` when nothing changed. A rep that is not (yet)
    resident on D counts as unresolved and the owner simply keeps its own block — less compression,
    never incorrect. Port of the resolve loop in the NCCL connector's ``_pd_consumer_apply``."""
    owner_groups = rid2blocks.get(owner_rid)
    if owner_groups is None or gi >= len(owner_groups):
        return None, 0, len(rows)
    owner_blocks = [int(b) for b in owner_groups[gi]]
    n_applied = n_unresolved = 0
    changed = False
    for owner_slot, rep_hash, rep_slot in rows:
        if owner_slot < 0:
            continue
        rep_rid = hash2rid.get(int(rep_hash))
        rep_groups = rid2blocks.get(rep_rid) if rep_rid is not None else None
        if rep_groups is None or gi >= len(rep_groups):
            n_unresolved += 1
            continue
        rep_grp = rep_groups[gi]
        if not (0 <= rep_slot < len(rep_grp) and 0 <= owner_slot < len(owner_blocks)):
            n_unresolved += 1
            continue
        owner_blocks[owner_slot] = int(rep_grp[rep_slot])
        changed = True
        n_applied += 1
    return (owner_blocks if changed else None), n_applied, n_unresolved


def write_runner_block_table(runner, rid: str, gi: int, new_blocks: list[int]) -> bool:
    """Write the redirected per-group block table into the runner's worker-side mirror so the
    forward reads the shared blocks. Ports the NCCL connector's ``_pd_write_runner_block_table``.

    Returns True iff the device table was actually rewritten. Callers MUST couple the block free to
    this return: a rid absent from this step's ``input_batch`` early-returns without rewriting, and
    freeing its blocks anyway would leave the request pointing at freed-then-reallocated KV."""
    ridx = runner.input_batch.req_id_to_index.get(rid)
    if ridx is None:
        return False
    bt_obj = runner.input_batch.block_table.block_tables[gi]
    n = min(len(new_blocks), int(bt_obj.num_blocks_per_row[ridx]))
    if n <= 0:
        return False
    row = new_blocks[:n]
    bt_obj.block_table.np[ridx, :n] = row
    bt_obj.block_table.gpu[ridx, :n] = torch.tensor(
        row, device=bt_obj.block_table.gpu.device, dtype=bt_obj.block_table.gpu.dtype)
    st = runner.requests.get(rid)
    if st is not None and gi < len(st.block_ids):
        st.block_ids[gi][:n] = row
    return True


class BFFMergeStats(KVConnectorStats):
    """Carries the D-side block-merge map worker→scheduler under TP>1.

    At TP>1 the worker and scheduler are SEPARATE processes, so the in-process
    ``_ACTIVE_RUNNER._updated_block_tables`` channel used at TP=1 isn't visible across them. This
    rides ``KVConnectorOutput.kv_connector_stats`` (the only serializable connector→scheduler slot):
    ``data["bff_merges"] = {req_id: {group_idx: [block_ids]}}``. Every TP rank produces the identical
    (all-reduced) map, so the cross-rank ``aggregate()`` keeps the accumulator's copy."""

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


# ---------------------------------------------------------------------------------------
# Producer-side fusion engine. Transport-free by construction: it is handed the step's
# (transfer_id, per-group block ids) list and the layer tensors, and RETURNS redirect rows —
# the connector decides how to ship them. Direct port of the `_pd_*` producer half of
# p2p_nccl_connector_ff.P2pNcclConnectorFF.
# ---------------------------------------------------------------------------------------
class FFProducerFusion:
    """Accumulates a fusion group's per-layer K reps and, at group completion, clusters them
    into per-request redirect rows.

    Usage per producer forward step: call :meth:`on_layer` once per attention layer, in the order
    vLLM calls ``save_kv_layer``. It returns ``None`` until a group's last layer is seen, then
    ``{transfer_id: [(owner_slot, rep_transfer_hash, rep_slot), ...]}`` for that group."""

    def __init__(self, group_layers: dict[int, set[str]] | None = None):
        self.group_layers: dict[int, set[str]] = group_layers or {}
        self.tp_group = None                 # torch.distributed group when TP>1, else None
        self._buf: dict[int, dict] = {}      # gi -> partial group buffer for this step
        self._cur_step_key: int | None = None
        self._jl = None                      # lazy fixed-seed JL matrix for BFF_PD_REPR=proj
        # Cross-batch rolling registry, per fusion group. Each entry holds the registered rep
        # blocks' raw concat vectors (this rank's head shard), their FULL squared concat norm,
        # stable (rep_hash, rep_slot), plus LRU bookkeeping to evict whole oldest requests.
        self._registry: dict[int, dict] = {}
        # LSH cross-request backend state (BFF_PD_CROSS_INDEX=lsh): one index per fusion group,
        # plus the lazily-built SimHash projection (shared across groups, keyed by feature width).
        self._lsh: dict[int, "pd_lsh.LshIndex"] = {}
        self._lsh_proj: list = [None]
        # Stats (mirrors the NCCL connector's bff_stats_<pid>.json schema exactly).
        self.ms = 0.0                        # fusion's own cost, AFTER the queue drain below
        self.drain_ms = 0.0                  # queued GPU work the fusion step had to wait out
        self.step_ms: deque = deque(maxlen=4096)   # per-step self.ms samples, for percentiles
        self.steps = 0
        self._last_dump = 0.0
        self.blk_total: dict[int, int] = {}
        self.redir_total: dict[int, int] = {}
        self.cross_redir_total = 0
        self.within_redir_total = 0
        self.lsh_evicted = 0                 # reps dropped because D freed their KV
        self.audit_steps: dict[int, int] = {}      # gi -> steps sampled (BFF_PD_AUDIT)
        self.audit_cos: dict[int, list] = {}       # gi -> random cross-request pair cosines

    # -- representation -------------------------------------------------------------
    def block_repr(self, kv_layer, idx, is_mla):
        """Per-layer block representation ``[N, D_repr]`` (float32) for the clustering similarity,
        selected by ``BFF_PD_REPR``. K-only; the concatenation cosine over the G group layers is
        applied by the clustering. ``full`` = exact (whole block), ``mean`` = head_dim mean,
        ``proj`` = fixed-seed JL projection (cosine-preserving, cheaper).

        MLA has no separate K/V cache (single latent, dim 0 = num_blocks) vs. FlashAttention's
        stacked ``[2, num_blocks, ...]`` layout (dim 0 = K/V selector, ``kv_layer[0]`` = K), hence
        the ``is_mla`` discriminant threaded in by the caller."""
        blk = (kv_layer[idx] if is_mla else kv_layer[0, idx]).float()
        N = idx.shape[0]
        if _PD_REPR == "mean":
            head_dim = blk.shape[-1]
            return blk.reshape(N, -1, head_dim).mean(dim=1)
        full = blk.reshape(N, -1)
        if _PD_REPR == "proj":
            if self._jl is None:
                g = torch.Generator(device=full.device)
                g.manual_seed(1234)
                self._jl = torch.randn(
                    full.shape[1], _PD_PROJ_DIM,
                    generator=g, device=full.device, dtype=torch.float32)
            return full @ self._jl
        return full

    # -- per-layer accumulation ------------------------------------------------------
    def on_layer(self, gi, layer_name, kv_layer, reqs, step_key, is_mla):
        """Accumulate one layer; return this group's redirect rows once the group is complete.

        ``reqs`` is ``[(transfer_id, per_group_block_ids), ...]`` for the requests being prefilled
        this step. Returns ``{transfer_id: rows}`` (possibly empty) on completion, else ``None``."""
        if step_key != self._cur_step_key:      # new step → drop any partial groups
            self._cur_step_key = step_key
            self._buf.clear()

        group_layer_set = self.group_layers.get(gi)
        if not group_layer_set:
            return None

        buf = self._buf.get(gi)
        if buf is None:
            # Build the flat block structure ONCE for this group/step (identical across its
            # layers): one entry per real (>0) block, tagged with owner request + slot.
            flat_bids, flat_req_local, flat_slot = [], [], []
            tids: list[str] = []
            for ri, (tid, blocks_per_group) in enumerate(reqs):
                tids.append(tid)
                bids = blocks_per_group[gi] if gi < len(blocks_per_group) else []
                for slot, bid in enumerate(bids):
                    if bid > 0:                  # skip null block 0
                        flat_bids.append(int(bid))
                        flat_req_local.append(ri)
                        flat_slot.append(slot)
            buf = {"seen": set(), "k_layers": [], "flat_bids": flat_bids,
                   "flat_req_local": flat_req_local, "flat_slot": flat_slot, "tids": tids}
            self._buf[gi] = buf

        if buf["flat_bids"]:
            idx = torch.as_tensor(buf["flat_bids"], device=kv_layer.device, dtype=torch.long)
            buf["k_layers"].append(self.block_repr(kv_layer, idx, is_mla))
        buf["seen"].add(layer_name)

        # Count-based completion (robust to layer-name `.attn` variance): this layer was routed to
        # gi by the caller, so it's a member; complete once all of the group's layers are seen.
        if len(buf["seen"]) < len(group_layer_set):
            return None

        try:
            from kv_fast_fusion.constants import THRESHOLD
            # Drain FIRST, and charge it separately. Fusion has to read decisions on the host, so
            # some device->host copy is unavoidable — but a `.cpu()` inside the timed region bills
            # fusion for every kernel already queued ahead of it, which is a measure of concurrency,
            # not of fusion. Draining here is free (the copies below sit on the same stream and
            # would have waited anyway) and splits the two apart.
            t0 = time.perf_counter()
            self._drain(buf)
            t1 = time.perf_counter()
            if _PD_AUDIT:
                self._audit_pairs(gi, buf)
            send_rows, n_cross, n_within = self._build_send_rows(
                gi, buf, _THRESHOLD_G.get(gi, THRESHOLD))
            t2 = time.perf_counter()
            self.drain_ms += (t1 - t0) * 1000.0
            self.ms += (t2 - t1) * 1000.0
            self.step_ms.append((t2 - t1) * 1000.0)
            self.steps += 1

            n_redir = n_cross + n_within
            self.blk_total[gi] = self.blk_total.get(gi, 0) + len(buf["flat_bids"])
            self.redir_total[gi] = self.redir_total.get(gi, 0) + n_redir
            self.cross_redir_total += n_cross
            self.within_redir_total += n_within
            if n_redir or _PD_DEBUG:
                logger.info(
                    "BFF P/D fuse group gi=%d | merge=%s | repr=%s | reqs=%d | blocks=%d | "
                    "redirects=%d (cross=%d within=%d) | reg_blocks=%d",
                    gi, _PD_MERGE, _PD_REPR, len(buf["tids"]), len(buf["flat_bids"]),
                    n_redir, n_cross, n_within, self.registry_size(gi))

            out: dict[str, list] = {}
            for ri, tid in enumerate(buf["tids"]):
                rows = send_rows.get(ri)
                if rows:
                    out[tid] = [[int(r[0]), int(r[1]), int(r[2])] for r in rows]
            return out
        except Exception as e:  # pragma: no cover - defensive (must never break the transfer)
            logger.warning("BFF P/D producer fusion failed (group %d): %s", gi, e)
            return {}
        finally:
            self._buf.pop(gi, None)

    def _audit_pairs(self, gi, buf) -> None:
        """Sample cosines between RANDOM cross-request block pairs in this group.

        A merge threshold only means something relative to how similar two *unrelated* blocks in
        that group already are. If the p50 of this sample sits at 0.9, then a 0.75 bar accepts
        essentially every pair and the resulting "compression" is measuring degeneracy, not
        redundancy — which is exactly what group 1 (the first attention layers, whose keys share a
        large common component) was found doing at 73×.

        Same-request pairs are excluded: they are trivially similar and never merge candidates."""
        if self.audit_steps.get(gi, 0) >= _PD_AUDIT_STEPS or len(buf["flat_bids"]) < 2:
            return
        self.audit_steps[gi] = self.audit_steps.get(gi, 0) + 1
        try:
            cur = torch.cat([Kg.float() for Kg in buf["k_layers"]], dim=1)
            cur = cur / cur.norm(dim=1, keepdim=True).clamp(min=1e-6)
            n = cur.shape[0]
            g = torch.Generator().manual_seed(20240517 + gi)
            i = torch.randint(0, n, (_PD_AUDIT_PAIRS,), generator=g)
            j = torch.randint(0, n, (_PD_AUDIT_PAIRS,), generator=g)
            req = torch.as_tensor(buf["flat_req_local"])
            keep = (i != j) & (req[i] != req[j])      # distinct blocks, distinct requests
            if not bool(keep.any()):
                return
            i, j = i[keep].to(cur.device), j[keep].to(cur.device)
            cos = (cur.index_select(0, i) * cur.index_select(0, j)).sum(dim=1)
            self.audit_cos.setdefault(gi, []).extend(cos.detach().cpu().tolist())
        except Exception as e:  # pragma: no cover - diagnostics must never break a transfer
            logger.warning("BFF P/D similarity audit failed (group %d): %s", gi, e)

    @staticmethod
    def _drain(buf) -> None:
        """Wait out the GPU work already in flight, so it is not billed to fusion.

        Costs nothing net: every backend copies a decision to the host inside ``_build_send_rows``
        (LSH the reps + bucket ids, matrix the ``best_idx``/``labels`` lists), and those copies sit
        on the same stream behind the same queue."""
        layers = buf.get("k_layers")
        if layers and layers[0].device.type == "cuda":
            torch.cuda.current_stream(layers[0].device).synchronize()

    # -- clustering ------------------------------------------------------------------
    def registry_size(self, gi) -> int:
        reg = self._registry.get(gi)
        return 0 if reg is None or reg["vecs"] is None else int(reg["vecs"].shape[0])

    def _build_send_rows(self, gi, buf, threshold):
        """Build the per-owner redirect rows for this group and update the registry.

        Returns ``(send_rows, n_cross, n_within)`` where
        ``send_rows[owner_ri] = [(owner_slot, rep_transfer_hash, rep_slot), ...]``. When
        ``BFF_PD_ENCODED_BATCH_SIZE <= 0`` the registry is skipped → within-batch only."""
        from kv_fast_fusion.pd_fuse import (
            build_group_redirect, concat_cosine_cc_labels, concat_cosine_cross_match,
            concat_cosine_nr_tree_labels)
        send_rows: dict[int, list] = {}
        n_cross = n_within = 0
        if not buf["flat_bids"]:
            return send_rows, n_cross, n_within
        flat_req_local = buf["flat_req_local"]
        flat_slot = buf["flat_slot"]
        tids = buf["tids"]
        N = len(flat_req_local)
        dev0 = buf["k_layers"][0].device
        tp_group = self.tp_group

        def _cluster(k_layers, req_of_block):
            if tp_group is not None:
                # TP>1: each rank holds only a head SHARD, so per-shard cosines are partial. Only
                # CC exposes the raw Gram/sq for the cross-rank all-reduce (nr_tree normalizes
                # before the similarity) → every rank reaches the identical decision.
                return concat_cosine_cc_labels(k_layers, req_of_block, threshold, tp_group=tp_group)
            cluster = (concat_cosine_nr_tree_labels if _PD_MERGE == "nr_tree"
                       else concat_cosine_cc_labels)
            return cluster(k_layers, req_of_block, threshold)

        def _emit_within(unmatched=None):
            """Within-batch clustering over `unmatched` (or all blocks when None) → redirect rows.
            Returns the flat indices that are cluster representatives, for registration."""
            nonlocal n_within
            if unmatched is None:
                labels = _cluster(buf["k_layers"], torch.as_tensor(flat_req_local, device=dev0))
                _, redirects = build_group_redirect(labels, flat_req_local, flat_slot)
                for owner_ri, rws in redirects.items():
                    for (slot, rep_local, rep_slot, _rep_flat, _own_flat) in rws:
                        send_rows.setdefault(owner_ri, []).append(
                            (slot, _tid_hash(tids[rep_local]), rep_slot))
                        n_within += 1
                labels_l = labels.tolist()
                return [i for i in range(len(labels_l)) if labels_l[i] == i]
            if not unmatched:
                return []
            sub_k = [Kg[unmatched] for Kg in buf["k_layers"]]
            sub_req = [flat_req_local[i] for i in unmatched]
            sub_slot = [flat_slot[i] for i in unmatched]
            labels = _cluster(sub_k, torch.as_tensor(sub_req, device=dev0))
            _, redirects = build_group_redirect(labels, sub_req, sub_slot)
            for owner_ri, rws in redirects.items():
                for (slot, rep_local, rep_slot, _rep_flat_sub, _own_flat_sub) in rws:
                    send_rows.setdefault(owner_ri, []).append(
                        (slot, _tid_hash(tids[rep_local]), rep_slot))
                    n_within += 1
            labels_l = labels.tolist()
            return [unmatched[i] for i in range(len(labels_l)) if labels_l[i] == i]

        # ---- cross-request backend selection ----
        # "lsh" runs independently of _PD_ENCODED_BATCH (its pool is bounded by BFF_LSH_MAX_ENTRIES,
        # not the matrix FIFO window) and only at TP=1 — the index is a single host-side structure
        # with no cross-rank coherence, so under TP>1 fall back to matrix.
        use_lsh = _PD_CROSS_INDEX == "lsh" and tp_group is None
        if use_lsh:
            matched, n_cross = self._lsh_phase1(gi, buf, tids, send_rows, threshold)
            reps = _emit_within([i for i in range(N) if not matched[i]])
            self._lsh_register(gi, buf, tids, reps)
            return send_rows, n_cross, n_within

        # ---- registry disabled → within-batch-only path ----
        if _PD_ENCODED_BATCH <= 0:
            _emit_within()
            return send_rows, n_cross, n_within

        # ---- cross-batch (matrix registry enabled) ----
        reg = self._registry.get(gi)
        best_idx, _score, cur_sq, cur_concat = concat_cosine_cross_match(
            buf["k_layers"], reg["vecs"] if reg else None, reg["sq"] if reg else None,
            threshold, tp_group=tp_group)
        # Forbid a self-merge (a registered rep from the SAME request, e.g. chunked re-register).
        if reg is not None and bool((best_idx >= 0).any()):
            own_hash = torch.tensor([_tid_hash(tids[r]) for r in flat_req_local],
                                    dtype=torch.long, device=best_idx.device)
            self_hit = (best_idx >= 0) & (reg["hash"][best_idx.clamp(min=0)] == own_hash)
            best_idx = torch.where(self_hit, torch.full_like(best_idx, -1), best_idx)
        best_list = best_idx.tolist()

        # Phase 1: cross-batch matches → redirect to the registry rep (already resident on D).
        # Resolve the rep's hash/slot to VALUES now: `_register_reps` below re-indexes the registry
        # in this same call, so a stored row index would be stale by serialize time.
        matched = [False] * N
        for i, ridx in enumerate(best_list):
            if ridx < 0:
                continue
            send_rows.setdefault(flat_req_local[i], []).append(
                (flat_slot[i], int(reg["hash"][ridx].item()), int(reg["slot"][ridx].item())))
            matched[i] = True
            n_cross += 1

        # Phase 2: within-batch clustering on the UNMATCHED current blocks (subset → map back).
        reps_to_register = _emit_within([i for i in range(N) if not matched[i]])

        self._register_reps(gi, buf, reps_to_register, cur_concat, cur_sq)
        return send_rows, n_cross, n_within

    # ------------------------------------------------------------------
    # LSH cross-request backend (BFF_PD_CROSS_INDEX=lsh). See kv_fast_fusion/pd_lsh.py.
    # ------------------------------------------------------------------
    def _lsh_prepare(self, buf):
        """Normalized concat reps + their bucket ids for this group/step.

        Takes ONE host copy of the reps and caches it on the buffer, because probe and register both
        verify/store against the same matrix — and each ``.cpu()`` drains the device pipeline, so a
        second copy of the same tensor costs a full sync for nothing.

        The pre-normalisation norms ride along: cosine alone cannot tell a substitution that
        preserves magnitude from one that does not, and the difference is exactly the error the
        decode inherits (see ``pd_lsh.rel_err``)."""
        cached = buf.get("lsh")
        if cached is not None:
            return cached
        cur = torch.cat([Kg.float() for Kg in buf["k_layers"]], dim=1)      # [N, G*D]
        norms = cur.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cur = cur / norms
        proj = pd_lsh.get_proj(self._lsh_proj, cur.shape[1], cur.device)
        hashes = pd_lsh.sub_hashes_device(cur, proj).cpu().tolist()
        cur_cpu = cur.detach().cpu().float()   # index lives on the host; one transfer, shared below
        cached = (cur_cpu, hashes, norms.detach().cpu().flatten().tolist())
        buf["lsh"] = cached
        return cached

    def _lsh_phase1(self, gi, buf, tids, send_rows, threshold):
        """Cross-request phase 1 via the SimHash index: probe, emit a redirect per verified hit."""
        cur_cpu, hashes, norms = self._lsh_prepare(buf)
        flat_req_local, flat_slot = buf["flat_req_local"], buf["flat_slot"]
        owners = [tids[r] for r in flat_req_local]
        index = self._lsh.get(gi)
        if index is None:
            return [False] * len(flat_req_local), 0
        matched, hits = index.probe(cur_cpu, hashes, owners, threshold, norms)
        for (i, rep_key, rep_slot) in hits:
            send_rows.setdefault(flat_req_local[i], []).append(
                (flat_slot[i], int(rep_key), int(rep_slot)))
        return matched, len(hits)

    def evict_owners(self, tids) -> int:
        """Drop every representative owned by ``tids`` from every group's index.

        Called when D reports those transfer ids finished. A rep is only useful while its KV is
        still resident on the decode instance: once freed, a redirect naming it resolves to nothing
        and the owner just keeps its own block — pure overhead, and it starves registration, since
        blocks that "matched" a dead rep are never offered as new reps themselves.

        The matrix backend needs no equivalent: its registry is already bounded to the last
        ``BFF_PD_ENCODED_BATCH_SIZE`` requests, which at saturation are all still resident."""
        if not tids or not self._lsh:
            return 0
        dropped = 0
        for ix in self._lsh.values():
            dropped += ix.evict_owners(tids)
        self.lsh_evicted += dropped
        return dropped

    def _lsh_register(self, gi, buf, tids, rep_flats):
        """Insert this step's unmatched representatives into the group's index."""
        if not rep_flats:
            return
        cur_cpu, hashes, norms = self._lsh_prepare(buf)
        flat_req_local, flat_slot = buf["flat_req_local"], buf["flat_slot"]
        index = self._lsh.get(gi)
        if index is None:
            index = self._lsh[gi] = pd_lsh.LshIndex()
        index.register(cur_cpu, hashes, [
            (f, _tid_hash(tids[flat_req_local[f]]), flat_slot[f], tids[flat_req_local[f]])
            for f in rep_flats], norms)

    def _register_reps(self, gi, buf, rep_flats, cur_concat, cur_sq):
        """Append this step's new rep blocks to the group registry, then LRU-evict to the window."""
        if not rep_flats:
            return
        flat_req_local = buf["flat_req_local"]
        flat_slot = buf["flat_slot"]
        tids = buf["tids"]
        dev = cur_concat.device
        reg = self._registry.get(gi)
        if reg is None:
            reg = {"vecs": None, "sq": None, "hash": None, "slot": None, "seq": None,
                   "key2seq": {}, "next_seq": 0}
            self._registry[gi] = reg
        v, sq, hsh, slt, seq = [], [], [], [], []
        for f in rep_flats:
            key = tids[flat_req_local[f]]
            s = reg["key2seq"].get(key)
            if s is None:
                s = reg["next_seq"]
                reg["key2seq"][key] = s
                reg["next_seq"] = s + 1
            v.append(cur_concat[f])
            sq.append(cur_sq[f])
            hsh.append(_tid_hash(key))
            slt.append(flat_slot[f])
            seq.append(s)
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

    def _evict_registry(self, gi):
        """Drop rows from requests older than the last ``BFF_PD_ENCODED_BATCH_SIZE`` distinct
        requests. Seq ids are dense + monotonic, so keeping ``seq >= next_seq - N`` keeps exactly
        the last N requests."""
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

    # -- stats -----------------------------------------------------------------------
    def _layers_fused(self) -> int:
        """Attention layers fusion actually runs on — group 0 (the warmup/sliding-window group) is
        never fused, and BFF_FF_GROUPS may exclude more."""
        gl = self.group_layers or {}
        return sum(len(v) for gi, v in gl.items()
                   if gi > 0 and (_FF_GROUPS is None or gi in _FF_GROUPS))

    def stats_dict(self, extra: dict | None = None) -> dict:
        """Cumulative fuse overhead + compression, in the SAME schema the NCCL connector dumps —
        so the launch script's ``bff_stats_*.json`` merge step works unchanged.

        Compression FACTOR = total / (total - freed): how many× smaller the KV cache gets from
        fusion (>1; 2.0 = half the blocks); block-weighted overall and per-group."""
        def _factor(b, r):
            return b / max(1, b - r)
        tot_b = sum(self.blk_total.values())
        tot_r = sum(self.redir_total.values())
        out = {
            "pid": os.getpid(),
            "is_producer": True,
            "steps": self.steps,
            # NOTE: this is fusion's OWN cost. Before the drain split it also absorbed the queued
            # GPU work the step happened to land behind, which made it scale with concurrency
            # rather than with fusion; numbers from before that change are not comparable.
            "overhead_avg_group_dedup_ms": (self.ms / self.steps if self.steps else 0.0),
            # What that metric used to hide. Big here means the prefill GPU is deeply queued (high
            # concurrency), not that fusion is slow.
            "overhead_avg_queue_drain_ms": (self.drain_ms / self.steps if self.steps else 0.0),
            # Percentiles over the retained window: the mean alone cannot separate "every step is
            # slow" from "one cold-start step dominates a short run".
            "overhead_ms_pct": _pcts(self.step_ms),
            "total_blocks": tot_b,
            "freed": tot_r,
            "compression_avg_factor": _factor(tot_b, tot_r),
            "compression_per_group": {str(gi): _factor(self.blk_total[gi], self.redir_total[gi])
                                      for gi in sorted(self.blk_total)},
            # Only meaningful for the matrix backend: _build_send_rows returns inside the `use_lsh`
            # branch before ever reading it, so reporting a number here under LSH invites tuning a
            # knob that governs nothing (the LSH pool is bounded by BFF_LSH_MAX_* — and in practice
            # by the match rate, since a matched block never becomes a rep).
            "encoded_batch_size": (None if _PD_CROSS_INDEX == "lsh" else _PD_ENCODED_BATCH),
            "cross_batch_redirects": self.cross_redir_total,
            "within_batch_redirects": self.within_redir_total,
            "registry_blocks": {str(gi): self.registry_size(gi) for gi in sorted(self._registry)},
            # Cross-request backend + its pool size, and (LSH only) where the accepted-merge mass
            # sits. The histogram answers, from one run, whether BFF_THRESHOLD is the quality lever
            # (mass bunched near it) or the merges are already near-identical (mass > 0.95).
            "cross_index": _PD_CROSS_INDEX,
            "ff_groups": (None if _FF_GROUPS is None else sorted(_FF_GROUPS)),
            # The hard ceiling on whole-cache savings: fusion can only ever dedup the layers it
            # actually runs on. A reported factor above total/(total - fused_share) is impossible
            # and means the metric, not the cache, is being measured.
            "layers_fused": self._layers_fused(),
            "layers_total": sum(len(v) for v in (self.group_layers or {}).values()),
            "threshold": _global_threshold(),
            "thresholds_per_group": {str(g): t for g, t in sorted(_THRESHOLD_G.items())},
            "lsh_entries": {str(gi): ix.size() for gi, ix in sorted(self._lsh.items())},
            "lsh_owners": {str(gi): ix.n_owners() for gi, ix in sorted(self._lsh.items())},
            "lsh_evicted": self.lsh_evicted,
            # Similarity FLOOR per group (BFF_PD_AUDIT=1): quantiles of random cross-request pair
            # cosines. A threshold below the p50 here accepts essentially everything.
            "audit_random_pair_cos": {
                str(gi): _pcts(xs) for gi, xs in sorted(self.audit_cos.items()) if xs},
            "lsh_accept_cos": {
                str(gi): dict(zip(pd_lsh.ACCEPT_COS_LABELS, ix.accept_cos))
                for gi, ix in sorted(self._lsh.items()) if any(ix.accept_cos)},
            # What a merge actually costs the decode: ||k_owner - k_rep|| / ||k_owner||. Cosine
            # cannot see a magnitude mismatch; this can, and it is the axis a threshold sweep should
            # be plotted against rather than the compression factor.
            "lsh_accept_rel_err": {
                str(gi): dict(zip(pd_lsh.REL_ERR_LABELS, ix.accept_rel_err))
                for gi, ix in sorted(self._lsh.items()) if any(ix.accept_rel_err)},
        }
        if extra:
            out.update(extra)
        return out

    def should_dump(self) -> bool:
        """True when it is time to refresh ``bff_stats_<pid>.json``.

        Step count OR wall clock: the step rate depends on how many groups fusion is running
        (``BFF_FF_GROUPS=1`` is 6× slower per forward pass than all-groups), so a count-only rule
        leaves the file pinned at the first snapshot for entire runs."""
        if not self.steps:
            return False
        now = time.monotonic()
        if (self.steps <= 3 or self.steps % _PD_FUSE_LOG_EVERY == 0
                or now - self._last_dump >= _PD_STATS_DUMP_SEC):
            self._last_dump = now
            return True
        return False

    def dump_stats(self, stats_dir: str = _PD_STATS_DIR, extra: dict | None = None) -> None:
        """Atomically write :meth:`stats_dict` to ``<stats_dir>/bff_stats_<pid>.json``."""
        try:
            path = os.path.join(stats_dir, f"bff_stats_{os.getpid()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.stats_dict(extra), f)
            os.replace(tmp, path)   # atomic — the reader never sees a half-written file
        except Exception as e:  # pragma: no cover - defensive (must never break the transfer)
            logger.warning("BFF P/D: could not dump fuse stats: %s", e)


# ---------------------------------------------------------------------------------------
# Mooncake-gated section: the connector subclass, its scheduler/worker, and the wire structs.
# `mooncake` (the Transfer Engine python package) is imported transitively by vLLM's mooncake
# connector module, so this whole half is skipped on a box without it — the pure logic above
# stays importable (and unit-testable) regardless.
# ---------------------------------------------------------------------------------------
try:
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnector,
        MooncakeConnectorMetadata,
        MooncakeConnectorScheduler,
        MooncakeConnectorWorker,
        MooncakeXferMetadata,
        MooncakeXferResponse,
        MooncakeXferResponseStatus,
        group_concurrent_contiguous,
    )
    _MOONCAKE_AVAILABLE = True
except Exception as _e:  # pragma: no cover - optional dependency
    logger.info("MooncakeConnectorFF: mooncake stack unavailable (%s); "
                "only the pure fusion/apply glue is importable.", _e)
    _MOONCAKE_AVAILABLE = False


if _MOONCAKE_AVAILABLE:

    ReqId = str
    TransferId = str

    class MooncakeXferMetadataFF(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
    ):
        """D→P pull request. Identical to the stock ``MooncakeXferMetadata`` except that
        ``req_blocks`` carries **per-KV-cache-group** block ids (``list[list[int]]``) instead of one
        flat list — the whole point of the FF connector. Only the *decode's* annotation changes at
        encode time (msgspec serializes the object it is given), so D needs no override at all: its
        ``PullReqMeta.local_block_ids`` is already per-group, and P decodes with this struct.

        ``done_tids`` adds the reverse direction: the transfer ids D has finished with, so P can drop
        those blocks as merge representatives. Without it P keeps offering reps whose KV D freed long
        ago — measured at 63% of all redirects, decaying 56%→7% of them resolving over one run."""

        remote_hostname: str
        remote_port: int
        remote_tp_size: int
        remote_tp_rank: int
        req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
        kv_caches_base_addr: list[int]
        done_tids: list[TransferId] | None = None
        # v2 (decode-decides) phase 1: ask for per-block signatures and write nothing. The struct is
        # shared rather than subclassed so one decoder serves both versions; with omit_defaults a v1
        # peer never puts these on the wire. See connectors/mooncake_connector_ff_v2.py.
        want_signatures: bool = False

    class MooncakeXferResponseFF(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
    ):
        """P→D transfer acknowledgement, extended with the BFF redirect maps.

        ``ff_redirects[d_req_id][group_idx] = [[owner_slot, rep_transfer_hash, rep_slot], ...]``.
        Riding the ACK means the map arrives exactly when (and only when) the KV it describes has
        landed, with no extra socket, port or thread — and a lost/absent map degrades to "no
        sharing" rather than a stall, because nothing ever waits on it."""

        status: MooncakeXferResponseStatus
        ok_reqs: list[ReqId] | None = None
        err_reqs: list[ReqId] | None = None
        err_msg: str | None = None
        ff_redirects: dict[ReqId, dict[int, list[list[int]]]] | None = None
        # v2 phase-1 reply: {decode_req_id: {group_idx: SignatureCodec payload}}. Carries no
        # decision — the decode makes it — and no KV: nothing is written during phase 1.
        signatures: dict[ReqId, dict[int, dict]] | None = None

    class _FFResponseEncoder:
        """msgspec encoder shim carrying the BFF side-payloads in BOTH directions.

        * P→D, on ``MooncakeXferResponse``: the producer's pending redirect rows.
        * D→P, on ``MooncakeXferMetadata`` (the pull request): the transfer ids D has finished with.

        The base builds and sends these in half a dozen places. Rather than copy that vendored
        control-plane logic (which upstream keeps changing), intercept the one line they all share —
        ``self._encoder.encode(obj)`` — and upgrade the struct on the way out. Everything else in the
        base is untouched.

        ``done_tids`` is attached to EVERY pull request within its TTL rather than being consumed by
        the first one, because a decode instance pulls from several producers and this shim cannot
        see which one a given message is bound for. Eviction is idempotent, so over-sending costs a
        few bytes; under-sending would silently leave dead reps in a producer's index."""

        def __init__(self, worker):
            self._worker = worker
            self._enc = msgspec.msgpack.Encoder()

        def encode(self, obj):
            if isinstance(obj, MooncakeXferResponse) and obj.ok_reqs:
                rows = self._worker.pop_ff_rows(obj.ok_reqs)
                if rows:
                    obj = MooncakeXferResponseFF(
                        status=obj.status, ok_reqs=obj.ok_reqs, err_reqs=obj.err_reqs,
                        err_msg=obj.err_msg, ff_redirects=rows)
            elif isinstance(obj, MooncakeXferMetadata):
                done = self._worker.peek_done_tids()
                if done:
                    obj = MooncakeXferMetadataFF(
                        remote_hostname=obj.remote_hostname, remote_port=obj.remote_port,
                        remote_tp_size=obj.remote_tp_size, remote_tp_rank=obj.remote_tp_rank,
                        req_blocks=obj.req_blocks,
                        kv_caches_base_addr=obj.kv_caches_base_addr, done_tids=done)
            return self._enc.encode(obj)

    class _FFMetadataDecoder:
        """Mirror of :class:`_FFResponseEncoder` on the producer's receive path: decode the FF pull
        request and hand any ``done_tids`` to the worker, without copying ``_sender_worker``."""

        def __init__(self, worker):
            self._worker = worker
            self._dec = msgspec.msgpack.Decoder(MooncakeXferMetadataFF)

        def decode(self, buf):
            meta = self._dec.decode(buf)
            if meta.done_tids:
                self._worker.note_done_tids(meta.done_tids)
            return meta

    # -----------------------------------------------------------------------------------
    # Scheduler side: carry ALL groups' block ids (stock kept a single flat list).
    # -----------------------------------------------------------------------------------
    class MooncakeConnectorMetadataFF(MooncakeConnectorMetadata):
        """Adds ``fuse_reqs``: the requests whose prefill COMPLETES this step, as
        ``(request_id, transfer_id, per_group_block_ids)``.

        The producer's normal metadata (``reqs_to_send``) is useless for fusion: under the pull
        model it only gets block ids at ``request_finished``, long after the forward that wrote the
        KV. ``fuse_reqs`` is a separate, fusion-only channel built from the scheduler output, so the
        transfer path is completely unaffected."""

        def __init__(self):
            super().__init__()
            self.fuse_reqs: list[tuple[str, str, list[list[int]]]] = []

    class MooncakeConnectorSchedulerFF(MooncakeConnectorScheduler):
        """Per-group block ids + the fusion-only ``fuse_reqs`` channel."""

        def __init__(self, vllm_config, engine_id):
            super().__init__(vllm_config, engine_id)
            # req_id -> (accumulated per-group block ids, prompt_token_ids, transfer_id) for
            # prefills that span multiple scheduler steps. Fusion needs the FULL prompt's blocks
            # (all of them written) before it can read K, so chunks accumulate here and only the
            # final chunk emits a fuse_req.
            self._ff_chunked: dict[str, tuple[list[list[int]], list[int], str]] = {}
            # req_id -> transfer_id, for the producer's fusion bookkeeping. The transfer id is the
            # P/D-stable name of a request, but it lives on `Request.kv_transfer_params`, which
            # `NewRequestData` does NOT carry — so `_collect_fuse_reqs` cannot read it off the
            # scheduler output and has to look it up here instead. (Reading it off NewRequestData
            # was the bug that silently disabled fusion entirely.)
            self._ff_tid_of: dict[str, str] = {}

        def update_state_after_alloc(
            self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
        ):
            params = request.kv_transfer_params
            # Producer side: remember this request's transfer id while we still have the real
            # Request object. Called in the same scheduler step that emits it in
            # scheduled_new_reqs, so `_collect_fuse_reqs` always finds it.
            if params and params.get("do_remote_decode") and params.get("transfer_id"):
                self._ff_tid_of[request.request_id] = params["transfer_id"]
            if not params or not params.get("do_remote_prefill"):
                return super().update_state_after_alloc(request, blocks, num_external_tokens)
            # Consumer remote-prefill branch, per-group. Mirrors the stock body but keeps every
            # group's unhashed blocks instead of asserting a single group.
            assert not self.is_kv_producer
            if all(p in params
                   for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")):
                local_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else [])
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning("Got invalid KVTransferParams: %s. This request will not "
                               "utilize KVTransfer", params)
            params["do_remote_prefill"] = False   # only trigger 1 KV transfer per request

        def request_finished_all_groups(
            self, request: "Request", block_ids: tuple[list[int], ...]
        ) -> tuple[bool, dict[str, Any] | None]:
            """SupportsHMA replacement for ``request_finished``: hand the producer's send meta the
            per-group block ids so ``_build_transfer_params`` can index each layer by its group."""
            params = request.kv_transfer_params
            if not params or not params.get("transfer_id"):
                return False, None
            groups = [list(g) for g in block_ids]
            if params.get("do_remote_prefill"):
                # Aborted before it was ever scheduled — queue an empty recv so the worker
                # notifies P and its blocks aren't stranded there. (Stock behavior.)
                assert not self.is_kv_producer
                self._reqs_need_recv[request.request_id] = (request, [])
                params["do_remote_prefill"] = False
                return False, None
            if not params.get("do_remote_decode"):
                return False, None
            assert not self.is_kv_consumer
            from vllm.v1.request import RequestStatus
            if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
                self._reqs_not_processed.add(params["transfer_id"])
                self._ff_chunked.pop(request.request_id, None)
                self._ff_tid_of.pop(request.request_id, None)
                return False, None
            delay_free_blocks = any(len(g) > 0 for g in groups)
            if delay_free_blocks:
                self._reqs_need_send[request.request_id] = (request, groups)
            self._ff_chunked.pop(request.request_id, None)
            self._ff_tid_of.pop(request.request_id, None)
            return delay_free_blocks, None

        def build_connector_meta(
            self, scheduler_output: "SchedulerOutput"
        ) -> MooncakeConnectorMetadataFF:
            meta = MooncakeConnectorMetadataFF()

            # ---- stock transfer plumbing (unchanged, just onto the FF metadata) ----
            if not self.is_kv_producer:
                for req_id, (req, block_ids) in self._reqs_need_recv.items():
                    assert req.kv_transfer_params is not None
                    meta.add_new_req(request_id=req_id, local_block_ids=block_ids,
                                     kv_transfer_params=req.kv_transfer_params)
                self._reqs_need_recv.clear()

            if not self.is_kv_consumer:
                for req_id, (req, block_ids) in self._reqs_need_send.items():
                    assert req.kv_transfer_params is not None
                    meta.add_new_req(request_id=req_id, local_block_ids=block_ids,
                                     kv_transfer_params=req.kv_transfer_params,
                                     load_remote_cache=False)
                self._reqs_need_send.clear()
                meta.reqs_not_processed = self._reqs_not_processed
                self._reqs_not_processed = set()
                self._collect_fuse_reqs(scheduler_output, meta)

            return meta

        def _collect_fuse_reqs(self, scheduler_output, meta) -> None:
            """Fusion-only: record the per-group block ids of every prefill COMPLETING this step.

            A request whose prompt spans several scheduler steps has blocks that are not all written
            yet, so reading K from them would cluster on garbage; such requests accumulate in
            ``_ff_chunked`` and only emit once the last chunk lands. (With the benchmark's
            ``max_num_batched_tokens == max_model_len`` every prompt completes in one step, so this
            is the common path.)"""
            if not _BFF_PD_FUSE:
                return
            try:
                for new_req in scheduler_output.scheduled_new_reqs:
                    # NOTE: NewRequestData carries no kv_transfer_params — the transfer id must come
                    # from the map built in update_state_after_alloc.
                    tid = self._ff_tid_of.get(new_req.req_id)
                    if not tid:
                        continue
                    prompt = list(new_req.prompt_token_ids or [])
                    n = (scheduler_output.num_scheduled_tokens[new_req.req_id]
                         + new_req.num_computed_tokens)
                    groups = [list(g) for g in new_req.block_ids]
                    if n < len(prompt):
                        self._ff_chunked[new_req.req_id] = (groups, prompt, tid)
                        continue
                    meta.fuse_reqs.append((new_req.req_id, tid, groups))

                cached = scheduler_output.scheduled_cached_reqs
                for i, req_id in enumerate(cached.req_ids):
                    prev = self._ff_chunked.get(req_id)
                    if prev is None:
                        continue        # not a multi-step prefill → nothing to accumulate
                    groups, prompt, tid = prev
                    new_block_ids = cached.new_block_ids[i]
                    if new_block_ids is None:
                        blocks = groups                                  # no new blocks this chunk
                    elif req_id in cached.resumed_req_ids:
                        blocks = [list(g) for g in new_block_ids]        # restart after preemption
                    else:
                        blocks = [groups[g] + list(new_block_ids[g])
                                  for g in range(len(new_block_ids))]
                    n = scheduler_output.num_scheduled_tokens[req_id] + cached.num_computed_tokens[i]
                    if n < len(prompt):
                        self._ff_chunked[req_id] = (blocks, prompt, tid)
                        continue
                    self._ff_chunked.pop(req_id, None)
                    meta.fuse_reqs.append((req_id, tid, blocks))
            except Exception as e:  # pragma: no cover - defensive (never break scheduling)
                logger.warning("BFF Mooncake: could not collect fuse reqs: %s", e)

    # -----------------------------------------------------------------------------------
    # Worker side: group-aware transfer params + redirect rows on the transfer ACK.
    # -----------------------------------------------------------------------------------
    class MooncakeConnectorWorkerFF(MooncakeConnectorWorker):
        """Group-aware pull + the redirect side-payload.

        The transfer itself is untouched except for ``_build_transfer_params``, which is the single
        place the stock worker collapses all layers onto one block list."""

        def __init__(self, vllm_config, engine_id):
            super().__init__(vllm_config, engine_id)
            # Both peers run the FF connector, so both sides speak the FF structs.
            self._xfer_meta_decoder = _FFMetadataDecoder(self)
            self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponseFF)
            self._encoder = _FFResponseEncoder(self)
            # base-address index -> KV-cache group index (parallel to self.kv_caches_base_addr).
            # base-address index -> the KV-cache groups sharing that allocation (see
            # _build_base_addr_groups; one allocation holds a layer from EVERY group).
            self.base_addr_groups: list[list[int]] = []
            self._layer_group: dict[str, int] = {}
            self._group_layers: dict[int, set[str]] = {}
            self._warned_layers: set[str] = set()
            # Producer: transfer_id -> {group_idx: rows} awaiting delivery on the transfer ACK.
            self._ff_rows: dict[str, dict[int, list]] = {}
            self._ff_dreq2tid: dict[str, str] = {}
            # Consumer: decode request_id -> {group_idx: rows} that arrived with the ACK, plus the
            # transfer-hash → decode-request-id map used to resolve reps. Written from the receiver
            # thread, read from the forward thread → guarded.
            self._ff_lock = threading.Lock()
            self._ff_recv_rows: dict[str, dict[int, list]] = {}
            self._ff_tid2rid: dict[int, str] = {}
            # Consumer liveness feedback: decode request id -> its transfer id, and the ids finished
            # recently (tid -> when, insertion-ordered so expiry pops from the front).
            self._ff_rid2tid: dict[str, str] = {}
            self._ff_done_pending: OrderedDict[str, float] = OrderedDict()
            # Producer: transfer ids D reports finished, written by the receiver thread and drained
            # by the forward thread.
            self._ff_done_from_d: set[str] = set()
            # Blocks whose pull failed; drained by get_block_ids_with_load_errors so the scheduler
            # can recompute (or fail) those requests instead of waiting on KV that never arrives.
            self._ff_failed_blocks: set[int] = set()

        # -- group layout -------------------------------------------------------------
        def build_group_maps(self, kv_caches: dict[str, torch.Tensor] | None = None) -> None:
            """Populate layer→group / group→layers from the live BFF KV-cache group layout."""
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                if runner is None:
                    return
                for gi, g in enumerate(runner.kv_cache_config.kv_cache_groups):
                    self._group_layers[gi] = set(g.layer_names)
                    for ln in g.layer_names:
                        self._layer_group[ln] = gi
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("FF Mooncake: could not build layer→group map: %s", e)

        def group_of(self, layer_name: str) -> int:
            gi = self._layer_group.get(layer_name)
            if gi is None:
                # Robustness: keys may carry/drop a trailing ".attn"; try both forms.
                gi = self._layer_group.get(layer_name + ".attn")
                if gi is None and layer_name.endswith(".attn"):
                    gi = self._layer_group.get(layer_name[: -len(".attn")])
            if gi is None:
                if layer_name not in self._warned_layers:
                    self._warned_layers.add(layer_name)
                    logger.warning(
                        "FF Mooncake: layer %s not found in any KV-cache group; falling back to "
                        "group 0 (block ids may be wrong).", layer_name)
                return 0
            return gi

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
            super().register_kv_caches(kv_caches)
            self.build_group_maps(kv_caches)
            self.base_addr_groups = self._build_base_addr_groups()

        def _build_base_addr_groups(self) -> list[list[int]]:
            """Map each registered base address → the KV-cache groups that live inside it.

            A base address is an ALLOCATION, not a layer. Under the hybrid allocator vLLM packs one
            layer from *every* group into each tensor (kv_cache_utils.py, `_get_kv_cache_config_
            uniform_page_size`): "As layers of different groups have different block table, they
            will use different parts of the shared Tensor." For BFF's 7 groups × 4 layers that is 4
            tensors shared by 7 layers each, ×2 for split K/V = 8 base addresses.

            So there is no single group per address — tagging one (as this did before) meant the
            transfer copied group 0's blocks into every allocation and never shipped the other
            groups' KV at all, which corrupted 6 of 7 groups' layers on D. Instead return, per
            address, the FULL set of groups sharing it; their block ids are disjoint parts of the
            same block axis, so transferring each group's blocks into the same allocation is exactly
            right.

            Derived from `kv_cache_config.kv_cache_tensors` (authoritative) rather than inferred by
            walking `kv_caches`, and index-aligned with the base class's `kv_caches_base_addr`."""
            n_addr = len(self.kv_caches_base_addr)
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                tensors = list(runner.kv_cache_config.kv_cache_tensors)
                n_groups = len(runner.kv_cache_config.kv_cache_groups)
            except Exception as e:
                raise RuntimeError(
                    "FF Mooncake: cannot read kv_cache_config.kv_cache_tensors to map base "
                    f"addresses to KV-cache groups ({e}). Serving now would transfer every layer "
                    "with the wrong block table and silently corrupt the decode's KV. This "
                    "connector requires the BFF stack — launch via `python -m "
                    "kv_fast_fusion.fast_fusion_main serve`, or use the stock MooncakeConnector."
                ) from e

            per_tensor = [sorted({self.group_of(ln) for ln in t.shared_by}) for t in tensors]
            # split_k_and_v yields TWO base addresses per tensor (K then V), in tensor order — the
            # same order the base class appends them in register_kv_caches.
            groups = ([g for gs in per_tensor for g in (gs, list(gs))]
                      if self.kv_topo.split_k_and_v else per_tensor)

            seen_groups = {g for gs in groups for g in gs}
            if len(groups) != n_addr or len(seen_groups) != n_groups:
                # Refuse to serve. Degrading to single-group indexing is what produced a
                # wrong-but-running server (F1 0.69 -> 0.28) instead of an obvious failure.
                raise RuntimeError(
                    f"FF Mooncake: base-address/group mapping is inconsistent — {len(groups)} "
                    f"mapped vs {n_addr} registered base addrs, covering {len(seen_groups)} of "
                    f"{n_groups} KV-cache groups. Refusing to serve: transferring KV under a wrong "
                    "block table corrupts the decode silently.")
            logger.info(
                "FF Mooncake: registered %d base addrs across %d KV-cache groups "
                "(groups per addr: %s).", n_addr, n_groups,
                "/".join(str(len(gs)) for gs in groups))
            return groups

        # -- redirect rows (producer) ---------------------------------------------------
        def stash_ff_rows(self, transfer_id: str, gi: int, rows: list) -> None:
            self._ff_rows.setdefault(transfer_id, {})[gi] = rows
            # Rows are normally popped by the ACK that carries them, but a request that is aborted
            # (or whose decode never pulls) leaves its map behind forever. Bound the map by dropping
            # the oldest entries — dicts keep insertion order, and an evicted map only costs
            # compression for that one request.
            while len(self._ff_rows) > _FF_ROWS_MAX_PENDING:
                self._ff_rows.pop(next(iter(self._ff_rows)))

        def pop_ff_rows(self, d_req_ids: list[str]) -> dict[str, dict[int, list]]:
            """Rows for the requests this ACK covers, keyed by DECODE request id (what the ACK
            speaks). Popped, so each map ships exactly once."""
            out: dict[str, dict[int, list]] = {}
            for d_req_id in d_req_ids:
                tid = self._ff_dreq2tid.get(d_req_id)
                if tid is None:
                    continue
                rows = self._ff_rows.pop(tid, None)
                if rows:
                    out[d_req_id] = rows
            return out

        # -- redirect rows (consumer) ---------------------------------------------------
        def process_pulling_result(self, response, pull_metas):
            """Stock completion accounting, plus redirect maps, plus **failed-pull recovery**.

            The stock implementation logs ``err_reqs`` and returns. That is a hang: a request whose
            pull failed is never added to ``finished_recving_reqs``, so it sits in
            ``WAITING_FOR_REMOTE_KVS`` forever, holding a scheduler slot — and once enough of them
            accumulate the engine goes to ``Running: 0`` and simply stops. Observed on 2026-08-13:
            81 transfers failed with -1 in an early burst and the run wedged with 86 waiting.

            Instead, route a failed pull into vLLM's real KV-load-failure path: report the blocks
            that were never written via ``get_block_ids_with_load_errors`` AND mark the request done
            recving, so the scheduler's ``_handle_invalid_blocks`` either recomputes it locally
            (``kv_load_failure_policy=recompute``) or fails it outright. Either beats a hang, and
            neither runs the request on KV that never arrived."""
            rows = getattr(response, "ff_redirects", None)
            if rows:
                with self._ff_lock:
                    for d_req_id, by_group in rows.items():
                        dst = self._ff_recv_rows.setdefault(d_req_id, {})
                        for gi, r in by_group.items():
                            dst[int(gi)] = r

            for d_req_id in (response.err_reqs or []):
                pull_meta = pull_metas.get(d_req_id)
                if pull_meta is None:
                    continue
                # Every block this request expected to receive is now untrustworthy.
                bad: set[int] = set()
                for group in pull_meta.local_block_ids:
                    # Per-group under FF; tolerate a flat list in case a stock peer answered.
                    if isinstance(group, (list, tuple)):
                        bad.update(int(b) for b in group)
                    else:
                        bad.add(int(group))
                with self._ff_lock:
                    self._ff_failed_blocks |= bad
                    self._ff_recv_rows.pop(d_req_id, None)
                # Release it from WAITING_FOR_REMOTE_KVS; the scheduler decides recompute vs fail.
                pull_meta.pull_tasks_count = 0
                self.finished_recving_reqs.add(pull_meta.d_req_id)
                logger.warning(
                    "BFF Mooncake: pull FAILED for %s (%d blocks invalid) — releasing it for "
                    "local recompute instead of stranding it in WAITING_FOR_REMOTE_KVS.",
                    d_req_id, len(bad))

            super().process_pulling_result(response, pull_metas)

        def take_block_ids_with_load_errors(self) -> set[int]:
            with self._ff_lock:
                out, self._ff_failed_blocks = self._ff_failed_blocks, set()
                return out

        def note_pull_ids(self, reqs_to_recv) -> None:
            """Remember transfer_hash → decode request id so a redirect's representative (which the
            producer names by its transfer id) resolves to a D-side request."""
            with self._ff_lock:
                for pull_metas in reqs_to_recv.values():
                    for req_id, pull_meta in pull_metas.items():
                        self._ff_tid2rid[_tid_hash(pull_meta.transfer_id)] = req_id
                        self._ff_rid2tid[req_id] = pull_meta.transfer_id

        def drain_ff_rows(self) -> tuple[dict[str, dict[int, list]], dict[int, str]]:
            with self._ff_lock:
                out, self._ff_recv_rows = self._ff_recv_rows, {}
                return out, dict(self._ff_tid2rid)

        def forget_ff_ids(self, req_ids) -> None:
            """Drop resolution entries for requests that have left the engine (bounds the map), and
            queue their transfer ids for the producers.

            This is the moment a representative stops existing: D has freed the blocks, so any
            producer still offering them would emit redirects that resolve to nothing."""
            if not req_ids:
                return
            now = time.monotonic()
            with self._ff_lock:
                stale = [h for h, r in self._ff_tid2rid.items() if r in req_ids]
                for h in stale:
                    del self._ff_tid2rid[h]
                for r in req_ids:
                    self._ff_recv_rows.pop(r, None)
                    tid = self._ff_rid2tid.pop(r, None)
                    if tid is not None:
                        self._ff_done_pending[tid] = now

        def peek_done_tids(self) -> list[str] | None:
            """Transfer ids to advertise on the next pull request; expired entries are dropped here.

            Not a pop: see :class:`_FFResponseEncoder` for why every producer has to see them."""
            with self._ff_lock:
                if not self._ff_done_pending:
                    return None
                cutoff = time.monotonic() - _FF_DONE_TID_TTL
                while self._ff_done_pending:
                    tid, at = next(iter(self._ff_done_pending.items()))
                    if at >= cutoff and len(self._ff_done_pending) <= _FF_DONE_TID_MAX:
                        break
                    self._ff_done_pending.pop(tid)
                return list(self._ff_done_pending) or None

        def note_done_tids(self, tids) -> None:
            """Producer side: record ids D has finished with, for the forward thread to evict.

            Bounded, because nothing guarantees a drain: with ``BFF_FF_GROUPS=none`` every group
            returns before the drain point, so this would otherwise grow for the whole run. Dropping
            ids only costs a stale rep, and that arm holds no reps at all."""
            with self._ff_lock:
                if len(self._ff_done_from_d) > _FF_DONE_TID_MAX:
                    self._ff_done_from_d.clear()
                self._ff_done_from_d.update(tids)

        def take_done_tids(self) -> set[str]:
            with self._ff_lock:
                out, self._ff_done_from_d = self._ff_done_from_d, set()
                return out

        # -- the one transfer change: index every layer by ITS group's block table -------
        async def _build_transfer_params(self, ready_reqs, agent_meta):
            """Group-aware rewrite of the stock method.

            Stock zips ONE flat ``(local, remote)`` block-id pairing across every base address,
            which is only correct with a single KV-cache group. Under BFF each group has its own
            block table, and each allocation is shared by one layer from *every* group (see
            ``_build_base_addr_groups``) — so the pairing is computed per group and each allocation
            gets the pairings of ALL the groups that live in it. Their block ids are disjoint parts
            of the same block axis, so this fills each shared tensor exactly once, completely."""
            src_ptrs: list[int] = []
            dst_ptrs: list[int] = []
            lengths: list[int] = []
            err_reqs: list[ReqId] = []
            local_base_addr = self.kv_caches_base_addr
            remote_base_addr = agent_meta.kv_caches_base_addr
            block_len = self.block_len
            remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"
            addr_groups = self.base_addr_groups or [[0]] * len(local_base_addr)
            all_groups = sorted({g for gs in addr_groups for g in gs})

            for d_req_id, send_meta in ready_reqs:
                _, remote_groups = agent_meta.req_blocks[d_req_id]
                local_groups = send_meta.local_block_ids
                # Remember the pairing so the ACK can carry this request's redirect map.
                self._ff_dreq2tid[d_req_id] = send_meta.transfer_id

                # Pair up block ids ONCE per group, then reuse for every allocation holding it.
                pairs: dict[int, tuple[list[list[int]], list[list[int]]]] = {}
                failed = False
                n_blocks = 0
                for gi in all_groups:
                    remote_ids = remote_groups[gi] if gi < len(remote_groups) else []
                    local_ids = local_groups[gi] if gi < len(local_groups) else []
                    if not remote_ids:
                        continue
                    if len(local_ids) < len(remote_ids):
                        logger.error(
                            "req %s group %d: local blocks(%d) less than remote blocks(%d)!",
                            d_req_id, gi, len(local_ids), len(remote_ids))
                        failed = True
                        break
                    if len(local_ids) > len(remote_ids):
                        # Partial prefix-cache hit on D: it only wants the uncomputed tail.
                        local_ids = local_ids[-len(remote_ids):]
                    # v2 sentinels: D marks a block it decided NOT to pull with -1 rather than
                    # removing it, because the pairing above is POSITIONAL — a shortened list would
                    # silently pair D's survivors with P's last k blocks and write the wrong KV.
                    # Dropping the positions here (after the tail alignment, which must see the
                    # original length) keeps every surviving pair exactly as it was. No-op for v1,
                    # which never sends a negative id.
                    if any(b < 0 for b in remote_ids):
                        kept = [i for i, b in enumerate(remote_ids) if b >= 0]
                        remote_ids = [remote_ids[i] for i in kept]
                        local_ids = [local_ids[i] for i in kept]
                        if not remote_ids:
                            continue
                    pairs[gi] = group_concurrent_contiguous(local_ids, remote_ids)
                    n_blocks += len(remote_ids)
                if failed:
                    err_reqs.append(d_req_id)
                    continue
                if not pairs:
                    continue

                for addr_i, (local_layer_addr, remote_layer_addr) in enumerate(
                        zip(local_base_addr, remote_base_addr)):
                    for gi in (addr_groups[addr_i] if addr_i < len(addr_groups) else [0]):
                        pair = pairs.get(gi)
                        if pair is None:
                            continue
                        for group_local_block_id, group_remote_block_id in zip(*pair):
                            src_ptrs.append(local_layer_addr + group_local_block_id[0] * block_len)
                            dst_ptrs.append(remote_layer_addr + group_remote_block_id[0] * block_len)
                            lengths.append(block_len * len(group_local_block_id))

                logger.debug("Sending kv_caches for request %s (%d blocks over %d groups) to %s",
                             d_req_id, n_blocks, len(pairs), remote_session)

            return src_ptrs, dst_ptrs, lengths, err_reqs

        def start_load_kv(self, metadata):
            if not self.is_kv_producer and metadata.reqs_to_recv:
                self.note_pull_ids(metadata.reqs_to_recv)
            super().start_load_kv(metadata)

    # -----------------------------------------------------------------------------------
    # The connector.
    # -----------------------------------------------------------------------------------
    class MooncakeConnectorFF(MooncakeConnector, SupportsHMA):
        """Group-aware, fusion-adding Mooncake connector for GPU P/D (see module docstring).

        Declares ``SupportsHMA`` because BFF uses a hybrid multi-group KV layout (warmup
        sliding-window group + fusion full-attention groups). Without it the scheduler's
        ``_connector_finished`` asserts ``len(kv_cache_groups) == 1`` and crashes; SupportsHMA routes
        it to the per-group ``request_finished_all_groups`` instead."""

        # Subclass hook: v2 swaps in a worker that serves signatures and plans the pull, without
        # restating this __init__ (which deliberately skips MooncakeConnector's — see below).
        _WORKER_CLS = MooncakeConnectorWorkerFF

        # Whether BFF_SCALE_MODE other than "raw" reaches the kernel on this connector. False here
        # because v1 has nowhere to put the per-block scales: its redirect map rides the transfer
        # ACK, which carries no float payload. v2 sets it True — the producer ships exact per-block
        # K/V norms inside the signature payload and the scale is computed entirely on the decode,
        # so no new wire channel is involved. Kept as a class flag rather than an isinstance check
        # so a downgrade is a property of the connector, stated where the connector is defined.
        _SUPPORTS_SCALE_MODES = False

        def __init__(self, vllm_config, role, kv_cache_config=None):
            # NOTE: deliberately skips MooncakeConnector.__init__ and initializes its base
            # directly. The stock __init__ constructs the STOCK worker, which spins up a Transfer
            # Engine and — on a producer — binds the bootstrap server port. Letting it run and then
            # replacing the worker would leave a second engine alive and make the FF worker's
            # bootstrap server collide with the abandoned one on that same port.
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVConnectorBase_V1,
                KVConnectorRole,
            )
            KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

            kv_cfg = vllm_config.kv_transfer_config
            assert kv_cfg is not None
            assert kv_cfg.engine_id is not None, (
                "MooncakeConnectorFF needs an explicit `engine_id` in --kv-transfer-config: the "
                "proxy names the prefill by it when it tells the decode where to pull from, and "
                "the default is a fresh uuid per process.")
            self.engine_id = kv_cfg.engine_id
            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler = MooncakeConnectorSchedulerFF(
                    vllm_config, self.engine_id)
                self.connector_worker = None
            else:
                self.connector_scheduler = None
                self.connector_worker = self._WORKER_CLS(vllm_config, self.engine_id)

            self.is_producer = kv_cfg.kv_role == "kv_producer"
            self._ff_fuse = _BFF_PD_FUSE
            if self._ff_fuse and _PD_SCALE_MODE != "raw" and not self._SUPPORTS_SCALE_MODES:
                logger.warning(
                    "%s: BFF_SCALE_MODE=%s is not supported by this connector (v1's redirect maps "
                    "ride the transfer ACK, which carries no float payload) — running as 'raw'. "
                    "Use the v2 connector, or NCCL, for ratio mode.",
                    type(self).__name__, _PD_SCALE_MODE)
            self._fusion = FFProducerFusion() if self._ff_fuse else None
            self._ff_tp = _UNSET
            self._ff_step = 0
            # Consumer: rows waiting for their owner to be batched → (rows_by_group, first_seen_step).
            self._ff_pending: dict[str, tuple[dict[int, list], int]] = {}
            self._ff_pending_merges = None
            self._ff_applied = 0
            self._ff_unresolved = 0
            self._ff_groups_logged = False
            if self._ff_fuse:
                logger.info(
                    "MooncakeConnectorFF: BFF_PD_FUSE enabled | producer=%s | merge=%s | repr=%s "
                    "| encoded_batch=%d", self.is_producer, _PD_MERGE, _PD_REPR,
                    _PD_ENCODED_BATCH)

        @classmethod
        def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
            """Unlike stock Mooncake, this connector does real per-layer work in ``save_kv_layer``
            (the fusion accumulation). Python between graph pieces cannot run inside a full CUDA
            graph — it would simply be skipped on replay and fusion would silently stop — so demand
            PIECEWISE whenever fusion is on."""
            return _BFF_PD_FUSE

        # -- scheduler side -------------------------------------------------------------
        def request_finished_all_groups(
            self, request: "Request", block_ids: tuple[list[int], ...]
        ) -> tuple[bool, dict[str, Any] | None]:
            assert self.connector_scheduler is not None
            return self.connector_scheduler.request_finished_all_groups(request, block_ids)

        # -- worker side ----------------------------------------------------------------
        def _tp_group(self):
            """The tensor-parallel process group when TP>1, else None.

            Under TP>1 each rank holds only a head SHARD of K, so per-shard cosines are partial; the
            clustering all-reduces over this group to reconstruct the full-vector statistics, so
            every rank reaches the identical decision and the block tables stay coherent."""
            if self._ff_tp is _UNSET:
                grp = None
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        from vllm.distributed.parallel_state import get_tp_group
                        tp = get_tp_group()
                        if tp.world_size > 1:
                            grp = tp.device_group
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("FF Mooncake: TP group lookup failed (assuming TP=1): %s", e)
                self._ff_tp = grp
                if grp is not None:
                    logger.info("MooncakeConnectorFF: TP>1 → all-reduced fusion decision.")
            return self._ff_tp

        def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
            super().start_load_kv(forward_context, **kwargs)
            self._ff_step += 1
            if self._ff_fuse and not self.is_producer:
                self._ff_consumer_apply()

        def _ff_log_group_filter(self, worker) -> None:
            """One-shot log of which fusion groups BFF_FF_GROUPS selected / skipped."""
            if self._ff_groups_logged or _FF_GROUPS is None:
                return
            self._ff_groups_logged = True
            eligible = {gi for gi in (worker._group_layers or {}) if gi > 0}
            if not eligible:
                self._ff_groups_logged = False     # map not built yet; log on a later layer
                return
            selected = sorted(eligible & set(_FF_GROUPS))
            skipped = sorted(eligible - set(_FF_GROUPS))
            if not selected:
                logger.info(
                    "BFF Mooncake: BFF_FF_GROUPS selects NO fusion groups — group split and patches "
                    "stay active, no clustering or redirects (control arm).")
            else:
                logger.info("BFF Mooncake: BFF_FF_GROUPS selects fusion groups %s, excludes %s",
                            selected, skipped)

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
            """Producer fusion hook. The base is a no-op (Mooncake transfers whole requests by RDMA
            pull, not per layer), so this adds work without displacing any — the fusion cost
            overlaps the rest of the prefill step exactly as it does on the NCCL connector."""
            if not (self._ff_fuse and self.is_producer):
                return
            worker = self.connector_worker
            if worker is None:
                return
            try:
                meta = self._get_connector_metadata()
                fuse_reqs = getattr(meta, "fuse_reqs", None)
                if not fuse_reqs:
                    return
                gi = worker.group_of(layer_name)
                if gi <= 0:                       # group 0 is the warmup group — never fused
                    return
                if _FF_GROUPS is not None and gi not in _FF_GROUPS:
                    # BFF_FF_GROUPS excluded this group. Returning HERE (before on_layer) is what
                    # makes the knob pay: no block repr, no buffering, no clustering, no register.
                    # The group's KV still transfers normally.
                    self._ff_log_group_filter(worker)
                    return
                fusion = self._fusion
                if not fusion.group_layers:
                    fusion.group_layers = worker._group_layers
                    fusion.tp_group = self._tp_group()
                    self._ff_log_group_filter(worker)
                # Retire reps whose KV D has freed, BEFORE this step probes against them.
                fusion.evict_owners(worker.take_done_tids())
                from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
                is_mla = (kv_layer.ndim == 3
                          or isinstance(attn_metadata, MLACommonMetadata)
                          or kv_layer.shape[1] == 2)
                rows = fusion.on_layer(
                    gi, layer_name, kv_layer,
                    [(tid, groups) for (_rid, tid, groups) in fuse_reqs],
                    id(meta), is_mla)
                if rows is None:
                    return                        # group not complete yet
                for tid, r in rows.items():
                    worker.stash_ff_rows(tid, gi, r)
                if fusion.should_dump():
                    fusion.dump_stats(_PD_STATS_DIR)
            except Exception as e:  # pragma: no cover - defensive (never break the transfer)
                logger.warning("BFF Mooncake save_kv_layer fusion failed (%s): %s", layer_name, e)

        def get_finished(self, finished_req_ids: set[str]):
            out = super().get_finished(finished_req_ids)
            if self._ff_fuse and not self.is_producer and finished_req_ids:
                try:
                    self.connector_worker.forget_ff_ids(finished_req_ids)
                    for rid in finished_req_ids:
                        self._ff_pending.pop(rid, None)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("BFF Mooncake: ff id cleanup failed: %s", e)
            return out

        def get_block_ids_with_load_errors(self) -> set[int]:
            """Blocks whose remote pull failed this step (see the worker's
            ``process_pulling_result``). vLLM turns these into ``invalid_block_ids`` →
            ``Scheduler._handle_invalid_blocks``, which recomputes or fails the owning requests.
            Without this a failed transfer is a silent, permanent stall."""
            worker = self.connector_worker
            if worker is None:
                return set()
            try:
                return worker.take_block_ids_with_load_errors()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF Mooncake: could not collect load errors: %s", e)
                return set()

        def get_kv_connector_stats(self):
            """Under TP>1 the scheduler is a separate process, so this step's D-side block-merge map
            rides the connector-stats carrier. At TP=1 the in-process ``_ACTIVE_RUNNER`` channel is
            used instead → fall through to the base."""
            merges = self._ff_pending_merges
            self._ff_pending_merges = None
            if merges and self._tp_group() is not None:
                return BFFMergeStats(data={"bff_merges": merges})
            return super().get_kv_connector_stats()

        # -- consumer apply -------------------------------------------------------------
        def _ff_consumer_apply(self) -> None:
            """Apply arrived redirect maps for owners that are batched THIS step.

            Under the pull model a request's recv completes between steps, so it is not yet in
            ``input_batch`` when its map arrives; maps are therefore held until the step the owner is
            actually scheduled — which is still strictly before its first decode forward reads the
            block table. Fully guarded: any structural mismatch leaves D correct (per-request copies,
            no sharing) rather than crashing the load."""
            self._ff_pending_merges = None       # reset per step (no stale map on the TP>1 path)
            try:
                worker = self.connector_worker
                arrived, hash2rid_all = worker.drain_ff_rows()
                for rid, by_group in arrived.items():
                    prev = self._ff_pending.get(rid)
                    if prev is None:
                        self._ff_pending[rid] = (dict(by_group), self._ff_step)
                    else:
                        prev[0].update(by_group)
                if not self._ff_pending:
                    return

                from kv_fast_fusion import fast_fusion_block_pool as _bp
                runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                if runner is None:
                    return
                # Resolve reps only against requests BATCHED this step — the same residency test the
                # owner has to pass. `runner.requests` is not equivalent: a preempted request keeps
                # its entry there with stale `block_ids` while the scheduler has already freed and
                # reallocated those blocks, so resolving against it would point a live owner at
                # another request's KV. Costs some compression, and the run that motivated this had
                # 326 preemptions.
                batched = getattr(getattr(runner, "input_batch", None), "req_id_to_index", None)
                rid2blocks: dict[str, Any] = {}
                for rid_r in (batched or {}):
                    st = getattr(runner, "requests", {}).get(rid_r)
                    bids = getattr(st, "block_ids", None) if st is not None else None
                    if bids is not None:
                        rid2blocks[rid_r] = bids
                hash2rid = {h: r for h, r in hash2rid_all.items() if r in rid2blocks}

                updated: dict[str, dict[int, list[int]]] = {}
                n_applied = n_unresolved = 0
                done: list[str] = []
                for rid, (by_group, first_step) in self._ff_pending.items():
                    if rid not in runner.input_batch.req_id_to_index:
                        if self._ff_step - first_step > _FF_APPLY_MAX_AGE:
                            done.append(rid)     # owner never got batched → drop the map
                        continue
                    for gi, rows in by_group.items():
                        new_blocks, na, nu = resolve_redirect_rows(
                            rid2blocks, hash2rid, rid, int(gi), rows)
                        n_applied += na
                        n_unresolved += nu
                        # Free ONLY when the device table was really rewritten: otherwise the
                        # request would point at freed-then-reallocated KV (aliasing → corruption).
                        if new_blocks and write_runner_block_table(runner, rid, int(gi), new_blocks):
                            updated.setdefault(rid, {})[int(gi)] = new_blocks
                    done.append(rid)
                for rid in done:
                    self._ff_pending.pop(rid, None)

                if updated:
                    # Stage for D's scheduler to free the orphaned blocks + fix ref counts (reuses
                    # the BFF merge channel: _updated_block_tables → update_from_output).
                    runner._updated_block_tables = updated
                    self._ff_pending_merges = updated
                if n_applied or n_unresolved or _PD_DEBUG:
                    self._ff_applied += n_applied
                    self._ff_unresolved += n_unresolved
                    logger.info("BFF P/D apply | redirects_applied=%d | reps_unresolved=%d",
                                n_applied, n_unresolved)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF Mooncake consumer apply failed: %s", e)


def register_mooncake_connector_ff() -> None:
    """Register ``MooncakeConnectorFF`` with vLLM's connector factory (idempotent).

    Registration is by module path + class name and does NOT import this module's Mooncake half, so
    it is safe to call on a box without the ``mooncake`` package — the failure then surfaces at
    connector construction, where it belongs."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    if "MooncakeConnectorFF" in KVConnectorFactory._registry:
        return
    KVConnectorFactory.register_connector(
        "MooncakeConnectorFF",
        "kv_fast_fusion.connectors.mooncake_connector_ff",
        "MooncakeConnectorFF",
    )
    logger.info("Fast fusion P/D patch: registered MooncakeConnectorFF.")
