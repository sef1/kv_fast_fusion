"""BFF-aware Ascend Mooncake connector — the NON-layerwise (pull) transport.

Every NPU BFF run so far has gone over ``MooncakeLayerwiseConnector``, where P **pushes** KV one
layer at a time. The GPU results we compare against were measured on the mainline
``MooncakeConnector``, where D **pulls** whole requests over RDMA. The two are not comparable, so
this file exists to put the same transport under BFF on both devices.

It is phase A of two, and it deliberately contains **no deduplication**. Despite the ``FF`` in the
name it does not fuse anything — that is the layerwise v1 trick, and it has no counterpart on a pull
transport. All this class does is teach the vendored ``MooncakeConnector`` to speak BFF's multi-group
KV layout. The dedup layer goes on top in ``mooncake_connector_ff_v2.py``, once this one has a clean
measurement; the three previous NPU runs each died on a layer *beneath* the dedup logic, which is
exactly why the transport gets proven on its own first.

**What has to change, and why.** The vendored pull connector is single-group to the bone: ``ReqMeta``
carries a flat ``list[int]`` (:82), ``request_finished`` takes a flat ``block_ids`` (:848), and
``_transfer_kv_cache`` (:469) applies that one list to *every* registered base address. BFF splits
the model into a warmup group plus ``BFF_GROUP_SIZE``-packed fusion groups — seven for the default
config — each with its **own** block table. One flat list therefore transfers the wrong physical
blocks for six of the seven groups. It does not raise; it serves plausible garbage.

The sibling ``MooncakeLayerwiseConnector`` shipped with ``SupportsHMA`` and ``list[list[int]]``
already (:633, :97), which is the whole reason layerwise came first.

**The base-address hazard.** A registered base address is an *allocation*, not a layer. On GPU,
vLLM's uniform-page-size packing puts one layer from every group into each tensor, and the first
version of the GPU connector tagged a single group per address — silently corrupting six of seven
groups on D (see ``kv_fast_fusion/connectors/mooncake_connector_ff.py:1217``). Ascend's
``register_kv_caches`` (:1220) instead appends one address per ``(layer, K/V)`` by iterating
``kv_caches.values()``, so *usually* each address belongs to exactly one group. Both layouts are
handled below by keying on the address value itself, and a group that ends up unreachable is a hard
refusal rather than a fallback — a wrong map here is the most expensive failure mode this project
has, because nothing downstream can detect it.

Everything above the ``_ASCEND_AVAILABLE`` gate is pure and imports on any box, so the mapping and
alignment logic is unit-testable without an NPU.
"""

import os
import threading
import time
from collections import OrderedDict

from vllm.logger import init_logger

logger = init_logger("vllm.mooncake_connector_ff_ascend")

# Registered connector name, selected by `BASELINE=bff_pull` in run_benchmarks.sh.
CONNECTOR_NAME = "MooncakeConnectorFF"

# --- phase B (fusion) gates -----------------------------------------------------------------
# Three runtime-separable layers, so a bad run is bisected without a rebuild:
#   BFF_FF_SHIP=0            → accumulate + cluster on P, ship nothing. D is untouched.
#   BFF_FF_SHIP=1 APPLY=0    → redirects ride to D and are counted, never applied.
#   BFF_FF_SHIP=1 APPLY=1    → full fusion.
# The first two MUST leave decode output bit-identical to phase A; that invariance is the gate.
_FF_SHIP = os.environ.get("BFF_FF_SHIP", "1") == "1"
_FF_APPLY = os.environ.get("BFF_FF_APPLY", "1") == "1"
_FF_FUSE = os.environ.get("BFF_PD_FUSE", "0") == "1"


def _parse_ff_groups(raw):
    """``BFF_FF_GROUPS="1,2,3"`` → the set of fusion groups to run, or None for "every eligible".

    Shared spelling with the layerwise connector so one A/B knob means the same thing on both
    transports. Group 0 is the warmup group and is never eligible regardless."""
    if raw is None or not raw.strip():
        return None
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out or None


_FF_GROUPS = _parse_ff_groups(os.environ.get("BFF_FF_GROUPS"))

# Where MooncakeFFProducer writes `bff_stats_<pid>.json`, which the benchmark's collect_bff_stats
# globs for compression and overhead. Same env var as the layerwise connector so one harness reads
# both transports. Unset → the producer's maybe_dump_stats is a no-op and the harness reports
# "bff stats: none found", which is exactly what happened before this was wired.
_PD_STATS_DIR = os.environ.get("BFF_PD_STATS_DIR")


class KVGroupLayoutError(RuntimeError):
    """The BFF group layout could not be mapped onto the registered KV allocations.

    Always fatal. Serving anyway would transfer every layer with the wrong block table, and the
    result is not an error but wrong KV — the decode rambles, quality collapses, and every stat
    still reads as healthy."""


# =================================================================================================
# pure helpers (no NPU, no vllm_ascend)
# =================================================================================================
def build_layer_group_map(kv_cache_groups) -> dict[str, int]:
    """``layer_name -> kv-cache group index``, from the live BFF group layout."""
    out: dict[str, int] = {}
    for gi, g in enumerate(kv_cache_groups):
        for ln in g.layer_names:
            out[ln] = gi
    return out


def group_of(layer_group: dict[str, int], layer_name: str) -> int | None:
    """Group of a layer, tolerating the ``.attn`` suffix mismatch between the runner's KV-cache
    config and the connector's layer names. Returns None when genuinely unknown — callers must
    treat that as fatal rather than defaulting to group 0."""
    gi = layer_group.get(layer_name)
    if gi is None:
        gi = layer_group.get(layer_name + ".attn")
    if gi is None and layer_name.endswith(".attn"):
        gi = layer_group.get(layer_name[: -len(".attn")])
    return gi


def build_base_addr_groups(base_addrs, layer_names, layer_group, caches_per_layer,
                           n_groups) -> list[list[int]]:
    """Map every registered base address to the KV-cache groups whose blocks live inside it.

    ``base_addrs`` is index-aligned with the base class's ``kv_caches_base_addr``, which appends
    ``caches_per_layer`` entries per layer in ``kv_caches`` iteration order — so address ``j``
    belongs to layer ``layer_names[j // caches_per_layer]``.

    Two allocation layouts both come out right here:

    * one allocation per layer (the Ascend norm) — each address gets exactly one group;
    * one allocation shared by a layer from several groups (the GPU norm) — those layers report the
      **same** address, so keying on the address value unions their groups. Their block ids are
      disjoint parts of one block axis, so writing each group's blocks into the shared allocation is
      precisely correct.

    Raises :class:`KVGroupLayoutError` unless every group is reachable through some address. That
    check is the point of the function: a group that no address carries is a group whose KV is never
    transferred, and nothing downstream would notice."""
    if caches_per_layer <= 0:
        raise KVGroupLayoutError(f"caches_per_layer must be positive, got {caches_per_layer}")
    expected = len(layer_names) * caches_per_layer
    if len(base_addrs) != expected:
        raise KVGroupLayoutError(
            f"{len(base_addrs)} base addresses for {len(layer_names)} layers x "
            f"{caches_per_layer} caches (expected {expected}) — the base class's registration "
            "order is not what this connector assumes")

    per_addr: dict[int, set[int]] = {}
    for j, addr in enumerate(base_addrs):
        ln = layer_names[j // caches_per_layer]
        gi = group_of(layer_group, ln)
        if gi is None:
            raise KVGroupLayoutError(
                f"layer {ln!r} belongs to no KV-cache group; refusing to guess (group 0 would "
                "transfer the wrong blocks for it)")
        per_addr.setdefault(addr, set()).add(gi)

    out = [sorted(per_addr[a]) for a in base_addrs]
    seen = {g for gs in out for g in gs}
    missing = set(range(n_groups)) - seen
    if missing:
        raise KVGroupLayoutError(
            f"KV-cache groups {sorted(missing)} are not reachable through any registered base "
            f"address (found {sorted(seen)} of {n_groups}). Their KV would never be transferred.")
    return out


def descriptor_coverage(grouped, keep, addr_groups):
    """Which ``(group, local block)`` pairs the transfer's descriptor loop actually writes.

    Returns ``(covered, descriptors)`` — the set of pairs a segment was emitted for, and how many
    segments that took. Mirrors the emission in ``_transfer_kv_cache`` exactly, and exists to be
    compared against the blocks that were PLANNED: a block that receives no descriptor is never
    written, so the decode reads whatever the block's previous tenant left there.

    That is not hypothetical. Verification found blocks holding content matching no row of their
    request in any group — ~0.19% of transferred blocks, one per affected request, clustered under
    allocation churn — and a never-written block recycled from a finished request looks exactly like
    that. The existing per-request check only asserts every GROUP emitted something, so one missing
    block inside a group that emitted others passes it.

    Two silent-skip paths in the emission make this worth proving rather than assuming:
    ``if gi >= len(grouped): continue`` drops a whole group when ``addr_groups`` outruns the aligned
    list, and ``zip(grouped_remote, grouped_local)`` truncates to the shorter side if the two
    run-lists ever disagree. Neither raises, and neither is visible in the transferred bytes.

    Pure: takes the same three structures the loop does and touches no device, no engine and no
    addresses, so the arithmetic that decides "every block got written" is testable on CPU."""
    covered: set = set()
    descriptors = 0
    for k in keep:
        for gi in addr_groups[k]:
            if gi >= len(grouped):
                continue
            grouped_remote, grouped_local = grouped[gi]
            for _remote_run, local_run in zip(grouped_remote, grouped_local):
                descriptors += 1
                for b in local_run:
                    covered.add((int(gi), int(b)))
    return covered, descriptors


def block_segment(base_local, base_remote, local_id, remote_id, block_len, inner_block_len,
                  inner_offset=0):
    """The ``(src, dst, length)`` for ONE block, using the run's own arithmetic.

    ``_transfer_kv_cache`` coalesces consecutive blocks into a single segment addressed by the run's
    FIRST id, so the wire carries one descriptor for many blocks. Replaying a single block needs the
    same arithmetic applied to that block's own id, and it must agree with the run at ``j == 0`` or
    the replay would target a different address than the transfer did — which would make the replay
    answer a question nobody asked.

    ``src`` is the LOCAL destination and ``dst`` the REMOTE source: ``batch_transfer_sync_read``
    takes ``(buffers, peer_buffer_addresses)``, and the vendored names read backwards."""
    return (base_local + local_id * block_len + inner_offset * inner_block_len,
            base_remote + remote_id * inner_block_len,
            inner_block_len)


def chunk_segments(src, dst, lengths, max_n):
    """Split a transfer's three parallel lists into batches of at most ``max_n`` segments.

    ``max_n <= 0`` means one chunk holding everything — today's exact behaviour, so the knob's
    default changes nothing.

    Why bound it at all: verification found ~0.19% of transferred blocks never written, with this
    connector's descriptor list audited complete, so the write was lost inside
    ``batch_transfer_sync_read``. A segment count beyond what the engine will carry in one call is
    the obvious candidate, and issuing several bounded calls both tests that and fixes it if true.

    A strict partition, and the tests say so, because an off-by-one here would silently drop the
    tail — which is precisely the failure being chased, and it would look identical to it."""
    n = len(src)
    if max_n is None or max_n <= 0 or n <= max_n:
        return [(src, dst, lengths)] if n else []
    return [(src[i:i + max_n], dst[i:i + max_n], lengths[i:i + max_n])
            for i in range(0, n, max_n)]


class RecvThreadTimer:
    """Where the serial recv thread's wall clock goes, per request and in aggregate.

    The recv thread handles one request at a time — the vendored ``run`` pops and transfers
    synchronously, and its ``ThreadPoolExecutor(max_workers=32)`` is never used — so its **duty
    cycle** (busy ms over wall ms) is the whole story of whether it is a bottleneck. The stock
    connector's own per-request log puts that at 0.9% over a 523 s run at con512: 4.6 s of transfer
    across 512 requests, median 7.6 ms. That is the number BFF has to be read against, and BFF's
    override of ``_transfer_kv_cache`` dropped the log that produces it.

    Phases are free-form so the two connectors can report different ones (v1 has no signature
    exchange), and unknown phases simply appear in the summary. ``elapsed_ms`` is passed in rather
    than measured here so this stays pure and testable off-device.

    Note the phases need not sum to ``elapsed_ms``: they are the parts worth naming, not a partition.
    The summary reports both, so unattributed time is visible rather than silently redistributed."""

    __slots__ = ("busy_ms", "phase_ms", "requests", "started")

    def __init__(self, clock=None):
        self.requests = 0
        self.busy_ms = 0.0
        self.phase_ms: dict[str, float] = {}
        self.started = (clock or time.perf_counter)()

    def note(self, elapsed_ms: float, phases=None) -> None:
        self.requests += 1
        self.busy_ms += float(elapsed_ms)
        for name, ms in (phases or {}).items():
            if ms:
                self.phase_ms[name] = self.phase_ms.get(name, 0.0) + float(ms)

    def note_phase(self, name: str, elapsed_ms: float) -> None:
        """Recv-thread time that belongs to no single request — v2's per-BATCH signature exchange.

        Counts toward busy time (it blocks the thread exactly as a transfer does) but not toward
        the request count, or the ms/request figure would be divided by the wrong denominator."""
        if elapsed_ms:
            self.busy_ms += float(elapsed_ms)
            self.phase_ms[name] = self.phase_ms.get(name, 0.0) + float(elapsed_ms)

    def duty_cycle(self, now: float) -> float:
        """Busy fraction of wall clock since construction. 0.0 before any time has passed."""
        wall_ms = (now - self.started) * 1e3
        return self.busy_ms / wall_ms if wall_ms > 0 else 0.0

    def summary(self, now: float) -> str:
        """One line: how much of the recv thread's life was spent working, and on what."""
        parts = " ".join(f"{n} {ms / 1e3:.1f}s" for n, ms in sorted(
            self.phase_ms.items(), key=lambda kv: -kv[1]))
        mean = self.busy_ms / self.requests if self.requests else 0.0
        return (f"{self.requests} request(s), busy {self.busy_ms / 1e3:.1f}s of "
                f"{(now - self.started):.1f}s wall = {self.duty_cycle(now) * 100:.1f}% duty cycle "
                f"({mean:.1f} ms/request)" + (f" | {parts}" if parts else ""))


def planned_blocks(grouped):
    """``{(group, local block)}`` the transfer intends to write — the set coverage must equal."""
    out: set = set()
    for gi, (_remote, local_runs) in enumerate(grouped):
        for run in local_runs:
            for b in run:
                out.add((int(gi), int(b)))
    return out


def align_per_group(local_groups, remote_groups) -> list[tuple[list[int], list[int]]]:
    """Tail-align D's and P's block lists, per group.

    D may hold fewer blocks than P sent (a prefix cache hit covers the head), and the stock
    connector handles that by keeping P's *last* ``len(local)`` ids. That has to happen per group,
    because each group has its own list and they need not shorten by the same amount.

    Returns one ``(remote, local)`` pair per group, in group order; a group with nothing to pull
    yields two empty lists rather than being dropped, so the caller can still index by group."""
    out: list[tuple[list[int], list[int]]] = []
    n = max(len(local_groups), len(remote_groups))
    for gi in range(n):
        local = list(local_groups[gi]) if gi < len(local_groups) else []
        remote = list(remote_groups[gi]) if gi < len(remote_groups) else []
        if not local:
            out.append(([], []))
            continue
        if len(local) > len(remote):
            raise KVGroupLayoutError(
                f"group {gi}: decode wants {len(local)} blocks but the prefill only offered "
                f"{len(remote)}")
        if len(local) < len(remote):
            remote = remote[-len(local):]
        out.append((remote, local))
    return out


def flatten_group_lists(groups) -> list[int]:
    """Every block id across all groups — used only for counting and for the empty-transfer test."""
    return [int(b) for g in groups for b in g]


def transfer_indices(local_base, remote_base) -> list[int]:
    """Indices into the base-address lists that carry DISTINCT work, first occurrence wins.

    Under BFF's shared-tensor layout ``build_base_addr_groups`` degenerates: all seven layers
    sharing an allocation report the same address, so every address maps to the UNION of every
    group, and the 56-entry address list holds only 8 distinct regions. Left alone,
    ``_transfer_kv_cache`` then emits each needed segment 7 (duplicate addresses) x 7 (groups) = 49
    times instead of 7 — the same bytes to the same place, so the data is right, but it is 7x the
    RDMA traffic and it is what drives the decode into saturation.

    Keyed on the PAIR, never on the local address alone. Pair-keying is what stays correct if the
    two ends ever disagree about layout: a per-layer P (28 distinct regions) against a shared-layout
    D would pair one local region against seven *different* remote regions, and dropping six of them
    would silently lose five sevenths of the model's KV. With pair-keying that case is a no-op and
    every pair survives; only genuine (local, remote) repeats are dropped."""
    seen: set = set()
    out: list[int] = []
    for k, pair in enumerate(zip(local_base, remote_base)):
        if pair in seen:
            continue
        seen.add(pair)
        out.append(k)
    return out


class FFRowStash:
    """Producer-side handoff of redirect rows from the WORKER to the SCHEDULER.

    Fusion rows are produced in ``save_kv_layer`` (worker, mid-forward) but must leave the node in
    ``request_finished_all_groups`` (scheduler, at request completion). Under a PULL transport those
    are the only two points that exist: the worker never talks to D — D initiates every exchange —
    and the scheduler is the only place with a P→D message. So the two halves are joined here.

    This is the same in-process channel as ``_ACTIVE_RUNNER`` and carries the same TP=1 restriction,
    which this connector already requires elsewhere. Bounded, because a request whose rows are never
    collected (aborted before finishing) would otherwise leak them forever."""

    def __init__(self, cap: int = 4096):
        self.lock = threading.Lock()
        self.rows: "OrderedDict[str, dict[int, list]]" = OrderedDict()
        self.cap = cap
        self.dropped = 0

    def add(self, ext_id: str, gi: int, rows) -> None:
        if not rows:
            return
        with self.lock:
            per_group = self.rows.get(ext_id)
            if per_group is None:
                per_group = {}
                self.rows[ext_id] = per_group
            per_group[int(gi)] = [[int(o), int(h), int(s)] for (o, h, s) in rows]
            self.rows.move_to_end(ext_id)
            while len(self.rows) > self.cap:
                self.rows.popitem(last=False)
                self.dropped += 1

    def take(self, ext_id: str) -> "dict[int, list] | None":
        """Pop this request's rows. Consume-once: a second call returns None, so a retried
        ``request_finished`` cannot ship the same map twice."""
        with self.lock:
            return self.rows.pop(ext_id, None)


class FFPendingSource:
    """Consumer-side sink the scheduler's promotion hook drains.

    ``_bff_promotion_apply`` (fast_fusion_ascend_patch.py:226) consumes ``_FF_PENDING_SOURCE``
    expecting exactly ``.lock`` + ``.pending`` as ``{external_id: {gi: rows}}``, and optionally
    ``.promo_stats``. On the layerwise transport that object is a live ZMQ recv thread; on a pull
    transport the redirects arrive inside ``kv_transfer_params`` instead, so no thread is needed and
    this plain holder satisfies the same contract."""

    def __init__(self, cap: int = 4096):
        self.lock = threading.Lock()
        # Ordered + capped: the promotion hook removes an entry only when its owner is promoted, and
        # `_bff_sweep_late_maps` only removes entries whose owner ALREADY was. A request aborted
        # between allocation and promotion is in neither set, so its rows would sit here for the
        # life of the process. Same reasoning (and cap) as FFRowStash on the producer side.
        self.pending: "OrderedDict[str, dict[int, list]]" = OrderedDict()
        self.cap = cap
        self.dropped = 0
        self.promo_stats = {
            "promo_applied": 0, "promo_unresolved": 0, "promo_no_rows": 0,
            "promo_merge_calls": 0, "promo_pending_dropped": 0,
            "promo_unres_rep_loading": 0, "promo_unres_rep_gone": 0,
            "promo_rows_late": 0, "promo_maps_late": 0,
        }

    def offer(self, ext_id: str, groups_rows: dict) -> None:
        """Add this request's per-group rows to the pending map.

        MERGES rather than replaces. The promotion hook pops the whole ``{gi: rows}`` dict at once,
        so an offer that overwrote would silently discard any group already waiting there — and the
        producer emits one map per fusion group, not one per request."""
        if not groups_rows:
            return
        with self.lock:
            per_group = self.pending.setdefault(ext_id, {})
            for gi, rows in groups_rows.items():
                per_group[int(gi)] = rows
            self.pending.move_to_end(ext_id)
            while len(self.pending) > self.cap:
                self.pending.popitem(last=False)
                self.dropped += 1

    def drain(self) -> dict:
        with self.lock:
            out, self.pending = self.pending, {}
        return out


def normalize_ff_redirects(raw) -> "dict[int, list] | None":
    """Coerce the ``ff_redirects`` field back from its JSON round trip.

    ``kv_transfer_params`` crosses the proxy as JSON, which stringifies dict keys — so the group
    index arrives as ``"3"``, not ``3``, and indexing by int would silently find nothing. Returns
    None when the field is absent or unusable, which is the "no fusion for this request" case and
    must stay non-fatal: a dropped redirect costs compression, never correctness."""
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[int, list] = {}
    for gi, rows in raw.items():
        try:
            key = int(gi)
        except (TypeError, ValueError):
            continue
        if not rows:
            continue
        clean = [[int(r[0]), int(r[1]), int(r[2])] for r in rows if len(r) >= 3]
        if clean:
            out[key] = clean
    return out or None


def resolve_kv_cache_groups(kv_cache_config, runner):
    """The BFF group layout, from whichever source actually has it.

    Two sources, in this order:

    * ``kv_cache_config`` — what ``KVConnectorFactory.create_connector`` passes every REGISTERED
      connector, and what ``NPUWorker.initialize_from_config`` hands to
      ``ensure_kv_transfer_initialized`` *before* ``initialize_kv_cache``. Always the live
      post-split layout, and independent of import order.
    * ``runner.kv_cache_config`` — the original path, via the ``_ACTIVE_RUNNER`` the patched
      ``NPUModelRunner.__init__`` publishes. Only populated when the BFF patch was imported before
      the runner was constructed in this process, which is NOT the case when the connector module
      itself is what first drags in ``kv_fast_fusion`` (the factory loads it lazily, during
      ``ensure_kv_transfer_initialized``, long after the runner exists). That is the
      ``no active BFF runner`` failure this ordering fixes.

    Raises :class:`KVGroupLayoutError` when neither yields groups. Returning a default here is not
    an option: every downstream index is derived from this list, and a wrong one transfers real KV
    against the wrong block table."""
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not groups:
        groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None)
    if not groups:
        raise KVGroupLayoutError(
            "the KV-cache group layout is unreadable: the connector was given no kv_cache_config "
            "and there is no active BFF runner. Serving now would transfer every layer with the "
            "wrong block table and silently corrupt the decode's KV. Launch via "
            "`python -m kv_fast_fusion.fast_fusion_main serve`, or use the stock "
            "MooncakeConnectorV1.")
    return groups


def dedup_registration_regions(ptrs, sizes) -> tuple[list[int], list[int]]:
    """Drop repeated base pointers from a transfer-engine registration list, first occurrence wins.

    BFF's group split makes vLLM emit a SHARED-tensor layout: ``get_kv_cache_config_from_groups``
    sizes the pool by ``max_layers_per_group`` (``BFF_GROUP_SIZE``) and hands one ``KVCacheTensor``
    to one layer of *every* group, so a 28-layer model with 7 groups has 4 allocations, not 28. The
    NPU allocator honours that ``shared_by`` list, so the 7 layers sharing an allocation all report
    the SAME ``data_ptr()``.

    The vendored pull connector's ``register_kv_caches`` then appends one entry per ``(layer, K/V)``
    unconditionally, i.e. 56 pointers of which only 8 are distinct, and Mooncake refuses the 2nd
    registration of a region it already holds: *"Transfer Engine does not support overlapped memory
    region"*. The sibling LAYERWISE connector — the transport BFF already runs on NPU — guards the
    same append with ``if data_ptr() not in ptrs``, which is precisely why it survives the split and
    this one does not. This helper is that guard, applied from the outside.

    Only the REGISTRATION list is shortened. ``kv_caches_base_addr`` keeps all of its entries: it is
    indexed by layer in ``_transfer_kv_cache`` and shipped to the peer in the handshake metadata.
    Registering a region once and then reading at offsets inside it is exactly what
    ``batch_transfer_sync_read`` does, so per-layer registration was never required.
    """
    seen: set[int] = set()
    out_ptrs: list[int] = []
    out_sizes: list[int] = []
    for ptr, size in zip(ptrs, sizes):
        if ptr in seen:
            continue
        seen.add(ptr)
        out_ptrs.append(ptr)
        out_sizes.append(size)
    return out_ptrs, out_sizes


# =================================================================================================
# Ascend/NPU-only section. Guarded so the pure helpers above stay importable for unit tests.
# =================================================================================================
try:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
        SupportsHMA,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm_ascend import envs as ascend_envs
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
        KVCacheRecvingThread,
        MooncakeConnector,
        MooncakeConnectorMetadata,
        MooncakeConnectorScheduler,
        MooncakeConnectorWorker,
        ReqMeta,
        group_concurrent_contiguous,
    )
    from vllm_ascend.utils import enable_custom_op

    _ASCEND_AVAILABLE = True
except Exception as _e:  # pragma: no cover - optional dependency
    logger.info("MooncakeConnectorFF: Ascend stack unavailable (%s); only the pure group-layout "
                "glue is importable.", _e)
    _ASCEND_AVAILABLE = False


if _ASCEND_AVAILABLE:

    def _active_runner():
        from kv_fast_fusion import fast_fusion_block_pool as _bp
        return getattr(_bp, "_ACTIVE_RUNNER", None)

    def _ext_of(rid: str) -> str:
        """External (P/D-stable) request id. vLLM appends a per-server random 9-char suffix, so the
        raw request_id does NOT match across the two engines; every fusion key must be this."""
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
            get_external_request_id,
        )
        return get_external_request_id(rid)

    # Producer worker → producer scheduler (rows), and the consumer's sink for the promotion hook.
    # Module-level singletons because the two halves are different objects in the same process.
    _FF_ROWS = FFRowStash()
    _FF_SOURCE = None

    def _ff_pending_source():
        """The consumer sink, published to ``_FF_PENDING_SOURCE`` on first use.

        Created lazily so a producer-only process never publishes one: the promotion hook treats a
        published source as "this node consumes redirects", and on P that would make every promotion
        look for maps that will never exist."""
        global _FF_SOURCE
        if _FF_SOURCE is None:
            _FF_SOURCE = FFPendingSource()
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            _bp._FF_PENDING_SOURCE = _FF_SOURCE
            logger.info("MooncakeConnectorFF: promotion-time redirect apply ON "
                        "(pending source published).")
        return _FF_SOURCE

    class MooncakeConnectorMetadataFF(MooncakeConnectorMetadata):
        """Same container; ``local_block_ids`` / ``remote_block_ids`` are now per group.

        Adds ``fuse_reqs``: the requests whose prefill COMPLETES this step, as
        ``(request_id, external_id, per_group_block_ids)``. The producer's normal metadata is
        useless for fusion under a pull transport — it only learns block ids at
        ``request_finished``, long after the forward that wrote the KV. ``fuse_reqs`` is a
        separate, fusion-only channel built from the scheduler output, so the transfer path is
        completely unaffected. (Same design as the GPU pull connector, which hit this first.)"""

        def __init__(self):
            super().__init__()
            self.fuse_reqs: list[tuple] = []

        def add_new_req(self, request_id, local_block_ids, num_external_tokens,
                        kv_transfer_params):
            remote = kv_transfer_params["remote_block_ids"]
            # A producer running the stock connector (or a pre-BFF one) still sends a flat list.
            # Promote it rather than mis-indexing it, and say so once — mixing the two sides is a
            # deployment mistake, not something to paper over silently.
            if remote and not isinstance(remote[0], (list, tuple)):
                logger.warning_once(
                    "MooncakeConnectorFF: prefill sent a FLAT remote_block_ids list — it is not "
                    "running the BFF connector. Treating it as a single group; this is only "
                    "correct if the model really has one KV-cache group.")
                remote = [list(remote)]
            self.requests[request_id] = ReqMeta(
                local_block_ids=[list(g) for g in local_block_ids],
                num_external_tokens=num_external_tokens,
                remote_block_ids=[list(g) for g in remote],
                remote_engine_id=kv_transfer_params["remote_engine_id"],
                remote_request_id=kv_transfer_params["remote_request_id"],
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"],
                remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
                remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
                remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
                remote_multi_nodes_meta_mapping=kv_transfer_params.get(
                    "remote_multi_nodes_meta_mapping", {}),
                num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
            )

    class MooncakeConnectorSchedulerFF(MooncakeConnectorScheduler):
        """Per-group block ids on both ends of the P/D handshake, plus the fusion channel."""

        def __init__(self, vllm_config, engine_id):
            super().__init__(vllm_config, engine_id)
            # req_id -> (accumulated per-group block ids, prompt_token_ids) for prefills spanning
            # several scheduler steps. Fusion must read K from a FULLY written prompt, so chunks
            # accumulate here and only emit once the last one lands.
            self._ff_chunked: dict[str, tuple] = {}

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            params = request.kv_transfer_params
            if params is not None and (params.get("do_remote_prefill", False)
                                       or params.get("do_remote_decode", False)):
                self._reqs_in_batch.add(request.request_id)
            # Consumer side: lift this request's redirect map off the params the producer attached
            # and hand it to the promotion hook. Done HERE because it is the last point that sees
            # kv_transfer_params, and it is strictly before the request can leave
            # WAITING_FOR_REMOTE_KVS — i.e. inside the pre-decode window the apply requires.
            #
            # POP, not get. This method is called on EVERY allocation for a request, not just the
            # first, so a non-destructive read re-offers the same rows after the promotion hook has
            # already consumed and applied them. The sweep then sees an already-promoted ext and
            # discards them as "arrived after promotion", which is how a run ended up dropping 853
            # rows while applying only 371 blocks — a 1:1 apply/late pair per request. Consuming the
            # field at the source mirrors FFRowStash.take() on the producer, and mutating params is
            # what the vendored code itself does (`params["do_remote_prefill"] = False` below).
            if params is not None and _FF_APPLY:
                rows = normalize_ff_redirects(params.pop("ff_redirects", None))
                if rows:
                    _ff_pending_source().offer(_ext_of(request.request_id), rows)
            if params is None or not params.get("do_remote_prefill"):
                return
            if not params.get("remote_block_ids"):
                assert num_external_tokens == 0
                params["do_remote_prefill"] = False
                return
            if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port",
                                         "remote_request_id")):
                # The only change from stock: ALL groups, not group 0. Stock's
                # get_unhashed_block_ids() returns the first group's list, which under BFF is the
                # warmup group — four layers of twenty-eight.
                local_block_ids = (blocks.get_unhashed_block_ids_all_groups()
                                   if num_external_tokens > 0 else [])
                self._reqs_need_recv[request.request_id] = (
                    request, local_block_ids, num_external_tokens)
            else:
                logger.warning("Got invalid KVTransferParams: %s. This request will not utilize "
                               "KVTransfer", params)
            params["do_remote_prefill"] = False

        def build_connector_meta(self, scheduler_output):
            meta = MooncakeConnectorMetadataFF()
            for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    num_external_tokens=num_external_tokens,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()
            meta.requests_to_send = self._reqs_need_send
            self._reqs_need_send = {}
            meta.reqs_in_batch = self._reqs_in_batch
            self._reqs_in_batch = set()
            self._collect_fuse_reqs(scheduler_output, meta)
            return meta

        def _collect_fuse_reqs(self, scheduler_output, meta) -> None:
            """Fusion-only: record the per-group block ids of every prefill COMPLETING this step.

            A prompt spanning several scheduler steps has blocks that are not all written yet, so
            clustering on them would cluster on garbage. Those accumulate in ``_ff_chunked`` and emit
            only once the last chunk lands. (With ``max_num_batched_tokens == max_model_len`` every
            prompt completes in one step, so the first loop is the common path.)

            Touches only ``scheduler_output``, which is vLLM's — nothing Ascend-specific — and is
            wrapped whole: a fusion bookkeeping error must never break scheduling."""
            if not _FF_FUSE:
                return
            try:
                for new_req in scheduler_output.scheduled_new_reqs:
                    prompt = list(new_req.prompt_token_ids or [])
                    n = (scheduler_output.num_scheduled_tokens[new_req.req_id]
                         + new_req.num_computed_tokens)
                    groups = [list(g) for g in new_req.block_ids]
                    if n < len(prompt):
                        self._ff_chunked[new_req.req_id] = (groups, prompt)
                        continue
                    meta.fuse_reqs.append((new_req.req_id, _ext_of(new_req.req_id), groups))

                cached = scheduler_output.scheduled_cached_reqs
                for i, req_id in enumerate(cached.req_ids):
                    prev = self._ff_chunked.get(req_id)
                    if prev is None:
                        continue            # not a multi-step prefill → nothing to accumulate
                    groups, prompt = prev
                    new_block_ids = cached.new_block_ids[i]
                    if new_block_ids is None:
                        blocks = groups                                   # no new blocks this chunk
                    elif req_id in cached.resumed_req_ids:
                        blocks = [list(g) for g in new_block_ids]         # restart after preemption
                    else:
                        blocks = [groups[g] + list(new_block_ids[g])
                                  for g in range(len(new_block_ids))]
                    n = scheduler_output.num_scheduled_tokens[req_id] + cached.num_computed_tokens[i]
                    if n < len(prompt):
                        self._ff_chunked[req_id] = (blocks, prompt)
                        continue
                    self._ff_chunked.pop(req_id, None)
                    meta.fuse_reqs.append((req_id, _ext_of(req_id), blocks))
            except Exception as e:  # pragma: no cover - defensive (never break scheduling)
                logger.warning("MooncakeConnectorFF: could not collect fuse reqs: %s", e)

        def request_finished_all_groups(self, request, block_ids):
            """``SupportsHMA`` replacement for ``request_finished``.

            Same decision as stock — hand the KV over and delay the free — but the ids that go into
            ``kv_transfer_params`` are per group, so D can address each group's blocks. They cross
            the proxy as JSON, where a list of lists survives unchanged."""
            groups = [list(g) for g in block_ids]
            params = request.kv_transfer_params
            from vllm.v1.request import RequestStatus
            if (params is None or not params.get("do_remote_decode")
                    or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
                return False, None

            delay_free_blocks = any(len(g) > 0 for g in groups)
            if delay_free_blocks:
                import time
                self._reqs_need_send[request.request_id] = time.time()

            import math
            num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
            # Fusion redirects ride the params dict that already flows P→D. Under a pull transport
            # this is the ONLY P→D message: the worker never addresses D (D initiates every
            # exchange, and P's listener is a ROUTER that learns no address), so a push channel would
            # need address discovery this transport does not provide. The timing is also exactly
            # right — these params are produced after the whole prompt is written and consumed
            # before D allocates, i.e. strictly inside the pre-decode window the apply requires.
            ff_redirects = None
            if _FF_SHIP:
                ff_redirects = _FF_ROWS.take(_ext_of(request.request_id))
            return delay_free_blocks, dict(
                do_remote_prefill=True,
                do_remote_decode=False,
                ff_redirects=ff_redirects or {},
                remote_block_ids=groups,
                remote_engine_id=self.engine_id,
                remote_request_id=request.request_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_pcp_size=self.pcp_size,
                remote_dcp_size=self.dcp_size,
                remote_ptp_size=self.tp_size,
                last_token_id=request.output_token_ids[-1],
                remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
                num_prompt_blocks=num_prompt_blocks,
            )

    class KVCacheRecvingThreadFF(KVCacheRecvingThread):
        """The pull itself, with the block axis generalised to N groups.

        ``base_addr_groups`` is injected after construction by the worker's ``register_kv_caches``,
        because it can only be derived once the caches are registered."""

        base_addr_groups: list[list[int]] | None = None
        _logged_amplification: bool = False

        def _align_and_group(self, req_meta, local_groups, remote_groups, tp_num_need_pulls):
            """Per group: tail-align, then coalesce contiguous runs.

            Both are per group because the groups have independent block tables — one group's run of
            consecutive ids says nothing about another's.

            Factored out as a seam for v2, which drops the blocks the decode can satisfy locally in
            between the two steps. That position is not negotiable: after the tail-align so the
            pair's indices already correspond, and before the coalesce so a dropped block breaks a
            contiguous run instead of being silently absorbed into one."""
            aligned = align_per_group(local_groups, remote_groups)
            grouped: list[tuple[list, list]] = []
            for remote_ids, local_ids in aligned:
                if not local_ids:
                    grouped.append(([], []))
                elif tp_num_need_pulls == 1:
                    gr, gl = group_concurrent_contiguous(remote_ids, local_ids)
                    grouped.append((gr, gl))
                else:
                    grouped.append(([[b] for b in remote_ids], [[b] for b in local_ids]))
            return grouped

        _AUDIT_DESCRIPTORS = os.environ.get("BFF_AUDIT_DESCRIPTORS", "0") == "1"
        _descriptor_gaps = 0
        # Set by v2 when BFF_V2_VERIFY_TRANSFER is on. Off, the emission loop does no extra work at
        # all — this is a diagnostic, and it allocates one dict entry per block per address.
        capture_descriptors: bool = False
        # None, not {}: a shared mutable class default would be one dict across every recv thread,
        # and this is overwritten per request anyway. Readers use `or {}`.
        last_descriptors = None
        last_session_id = None
        last_segment_count = 0
        # Segments per batch_transfer_sync_read call. 0 = unlimited, i.e. exactly what this
        # connector has always done. See chunk_segments for why bounding it is worth a knob.
        _MAX_XFER_SEGMENTS = int(os.environ.get("BFF_MAX_XFER_SEGMENTS", "0"))
        # Submit through batch_transfer_async_read and poll get_batch_transfer_status instead of the
        # sync call. Off by default, so the transport is byte-for-byte unchanged unless asked.
        #
        # Why it exists: the sync call returns 0 for every batch and silently fails to write ~1 block
        # per request under load — proven by replaying the identical descriptor and getting exactly
        # the producer's KV the second time. The async pair is the only API here that reports
        # completion at all, so it is both the candidate fix and, if it still loses writes, the
        # reproduction to send upstream with a status code attached.
        #
        # It hung the FIRST time it ran, taking the whole EngineCore process with it, so it is now
        # written to fail in seconds instead: see _one_batch.
        _XFER_ASYNC = os.environ.get("BFF_XFER_ASYNC", "0") == "1"
        _XFER_ASYNC_TIMEOUT = float(os.environ.get("BFF_XFER_ASYNC_TIMEOUT", "30"))
        # The FIRST batch gets a much shorter leash than the rest. Nothing in vllm-ascend calls the
        # async pair, so its status convention is unverified; if this code has it backwards, every
        # batch polls to its deadline and a 30 s one stalls the run. Two seconds proves the guess.
        _XFER_ASYNC_FIRST_TIMEOUT = float(os.environ.get("BFF_XFER_ASYNC_FIRST_TIMEOUT", "2"))
        # Between polls. Without it the loop is a tight Python loop that holds the GIL and starves
        # the engine thread even when the convention IS right — which is half of why the run died.
        _XFER_ASYNC_POLL_S = float(os.environ.get("BFF_XFER_ASYNC_POLL_S", "0.001"))
        _logged_async = False
        _logged_ret = False
        # Flipped when the async path proves itself unusable; sends every later batch to the sync
        # call rather than repeating a failure that costs seconds each time.
        _async_disabled = False
        # One INFO per request, matching what the vendored _transfer_kv_cache emits and this
        # override dropped. Off by default because it is a log record on a hot serial thread; the
        # duty-cycle summary below is always on and is the figure that decides anything.
        _RECV_TIMING = os.environ.get("BFF_RECV_TIMING", "0") == "1"
        _next_timing_report = 1

        @property
        def _recv_timer(self) -> RecvThreadTimer:
            """Per-INSTANCE, created on first use.

            Not a class attribute: the duty cycle is a fraction of one thread's wall clock, and
            sharing one accumulator across recv threads would divide the sum of their busy time by a
            single thread's lifetime — a duty cycle over 100% and no way to tell which thread."""
            t = self.__dict__.get("_recv_timer_obj")
            if t is None:
                t = self.__dict__["_recv_timer_obj"] = RecvThreadTimer()
            return t

        def _one_batch(self, session_id, src, dst, lengths):
            """One batch, through whichever API is selected. Returns the engine's status.

            The async path submits and then polls ``get_batch_transfer_status``, which is the whole
            point: the sync call reports 0 and drops writes, so a status that can actually say "not
            done" is the only way to see the loss from inside the transport.

            **It is written defensively because its first run killed a benchmark.** Three things are
            unknown and none of them may cost more than seconds:

            * The status convention. Mooncake's own sync path is submit-then-poll-to-COMPLETED, so 0
              most likely means "all complete" — the opposite of what this loop assumes. If the guess
              is wrong every batch polls to its deadline, which is exactly what a 30 s timeout turns
              into a dead run. Hence the short first-batch leash and the permanent disable.
            * Whether the poll releases the GIL. If it does not, a tight loop freezes the whole
              process; the sleep bounds the damage either way.
            * Batch-id lifetime. ``free_batch_id`` is not exposed in the binding, so ids may leak
              toward the engine's "Exceed the limitation of capacity" ceiling. Nothing else in
              vllm-ascend calls this pair, so there is no prior art to copy."""
            if not self._XFER_ASYNC or KVCacheRecvingThreadFF._async_disabled:
                return self.engine.batch_transfer_sync_read(session_id, src, dst, lengths)
            handle = self.engine.batch_transfer_async_read(session_id, src, dst, lengths)
            if handle < 0:
                return handle
            first = not KVCacheRecvingThreadFF._logged_async
            if first:
                KVCacheRecvingThreadFF._logged_async = True
                logger.info("MooncakeConnectorFF: BFF_XFER_ASYNC=1 — submitting through "
                            "batch_transfer_async_read, handle %r for %d segment(s).",
                            handle, len(src))
            budget = self._XFER_ASYNC_FIRST_TIMEOUT if first else self._XFER_ASYNC_TIMEOUT
            t0 = time.perf_counter()
            deadline = t0 + budget
            polls = 0
            while True:
                status = self.engine.get_batch_transfer_status([handle])
                polls += 1
                if status != 0:
                    # Non-zero is either completion or failure depending on the engine's
                    # convention, which is undocumented in the binding. Return it and let the
                    # caller's `< 0` check apply — the same contract the sync path has.
                    if first:
                        logger.info(
                            "MooncakeConnectorFF: async status for the first batch was %r after "
                            "%d poll(s) in %.1f ms — that is the convention this path assumes "
                            "(non-zero = settled).", status, polls,
                            (time.perf_counter() - t0) * 1e3)
                    return min(0, status)
                if time.perf_counter() >= deadline:
                    if first:
                        # A first batch that never settles means the guess above is wrong, and every
                        # later batch would burn the same budget. Stop asking.
                        KVCacheRecvingThreadFF._async_disabled = True
                        logger.error(
                            "MooncakeConnectorFF: the first async transfer (%r, %d segment(s)) "
                            "returned status 0 on all %d poll(s) for %.1fs. Either 0 means "
                            "COMPLETED here — the opposite of what this path assumes — or the "
                            "batch never settles. DISABLING BFF_XFER_ASYNC for this process and "
                            "falling back to batch_transfer_sync_read.",
                            handle, len(src), polls, budget)
                    else:
                        logger.error(
                            "MooncakeConnectorFF: async transfer %r did not report completion "
                            "within %.1fs for %d segment(s); treating as done and continuing.",
                            handle, budget, len(src))
                    return 0
                # Never a tight loop: this thread holds the GIL between polls, and the engine core
                # it would starve is the one generating tokens.
                time.sleep(self._XFER_ASYNC_POLL_S)

        def _issue_transfer(self, req_meta, session_id, src_list, dst_list, length_list):
            """Run the batched read, in bounded chunks, and hold the engine to its return value.

            The vendored call discards ``ret`` unless it is negative. That matters here: verification
            found ~0.19% of transferred blocks never written while this connector's descriptor list
            audited COMPLETE, so a write was lost inside the engine and nothing reported it. If
            ``ret`` ever carries a completed-segment count, checking it against the segments we
            handed over catches that on every request in production, with none of the signature
            machinery. We do not know the convention, so the first value is logged rather than
            assumed, and only a NEGATIVE ret is still treated as fatal — the vendored contract."""
            for src, dst, lengths in chunk_segments(
                    src_list, dst_list, length_list, self._MAX_XFER_SEGMENTS):
                ret = self._one_batch(session_id, src, dst, lengths)
                if ret < 0:
                    raise RuntimeError(
                        f"KV transfer failed for request {req_meta['remote_request_id']} "
                        f"(ret={ret}, {len(src)} segments)")
                if not KVCacheRecvingThreadFF._logged_ret:
                    KVCacheRecvingThreadFF._logged_ret = True
                    logger.info(
                        "MooncakeConnectorFF: batch_transfer_sync_read returned %r for %d "
                        "segment(s). If that is a completed-segment count rather than a status "
                        "code, a silently dropped write is detectable here on every request.",
                        ret, len(src))
                elif ret and ret != len(src):
                    # Only fires if ret turns out to be a count AND it is short: the engine accepted
                    # fewer segments than we gave it, which is a lost KV write, not a slow one.
                    logger.error(
                        "MooncakeConnectorFF: request %s handed %d segment(s) to "
                        "batch_transfer_sync_read and it returned %r — if that is a count, %d "
                        "block write(s) were dropped and the decode will read stale KV.",
                        req_meta.get("remote_request_id"), len(src), ret, len(src) - ret)

        def _audit_descriptor_coverage(self, req_meta, grouped, keep, addr_groups, n_emitted):
            """Did every block the transfer planned to write actually get a descriptor?

            Off by default (``BFF_AUDIT_DESCRIPTORS=1``): it is O(blocks) per request on the recv
            thread, and the question it answers is a one-time one — whether the segment list this
            connector builds is complete.

            Why it matters. v2's transfer verification found blocks on the decode holding content
            that matches no row of their request in any group, at ~0.19% of transferred blocks, one
            per affected request. A block that never receives a descriptor is never written and
            still holds its previous tenant's KV, which is indistinguishable from that. This check
            decides where to look: if coverage is always complete, our segment list is right and the
            lost write happened inside ``batch_transfer_sync_read`` — whose return value is a bare
            int this code only tests for ``< 0``, and which serves the stock connector too.

            Never raises. It reports; the transfer is already correct or already not."""
            if not self._AUDIT_DESCRIPTORS:
                return
            try:
                covered, expected_n = descriptor_coverage(grouped, keep, addr_groups)
                missing = planned_blocks(grouped) - covered
            except Exception as e:  # noqa: BLE001 - an audit must never break a transfer
                logger.warning("MooncakeConnectorFF: descriptor audit failed (%s).", e)
                return
            if not missing:
                return
            KVCacheRecvingThreadFF._descriptor_gaps += len(missing)
            logger.error(
                "MooncakeConnectorFF: request %s planned %d block(s) that received NO transfer "
                "descriptor — e.g. group %s block %s. They are never written, so the decode reads "
                "whatever their previous tenant left. (%d segments emitted, %d expected; %d gap(s) "
                "this process.)",
                req_meta.get("remote_request_id"), len(missing), *min(missing),
                n_emitted, expected_n, KVCacheRecvingThreadFF._descriptor_gaps)

        def _after_transfer(self, req_meta) -> None:
            """Called once this request's KV has actually landed. No-op in v1; v2 releases the
            request's aliases and residency here."""

        def _transfer_kv_cache(self, req_meta):
            t_req = time.perf_counter()
            phases: dict[str, float] = {}
            # Cleared here, not left from the last request: this method has early returns (a full
            # prefix-cache hit, a fully-deduped request) that never reach the emission loop, and a
            # stale count would attribute the PREVIOUS request's segments to this one.
            self.last_segment_count = 0
            try:
                self._transfer_kv_cache_timed(req_meta, phases)
            finally:
                self._note_recv_timing(req_meta, (time.perf_counter() - t_req) * 1e3, phases)

        def _note_recv_timing(self, req_meta, elapsed_ms, phases) -> None:
            """Per-request timing, then the duty cycle on a widening cadence.

            The vendored ``_transfer_kv_cache`` logs one INFO per request and the stock connector's
            512 of them are what put its recv thread at a 0.9% duty cycle; this override dropped
            that log, leaving BFF's serial thread unmeasured. Restored here, plus the aggregate the
            per-request lines only imply — and the aggregate is the one that decides anything, so it
            is always on while the per-request line is behind BFF_RECV_TIMING=1."""
            timer = self._recv_timer
            timer.note(elapsed_ms, phases)
            now = time.perf_counter()
            if self._RECV_TIMING:
                parts = " ".join(f"{n} {ms:.1f}" for n, ms in phases.items())
                logger.info(
                    "MooncakeConnectorFF: KV cache transfer for request %s took %.2f ms "
                    "(%d segments) [ms: %s]", req_meta.get("remote_request_id"), elapsed_ms,
                    self.last_segment_count, parts)
            if timer.requests >= self._next_timing_report:
                self._next_timing_report *= 10
                logger.info("MooncakeConnectorFF recv thread: %s", timer.summary(now))

        def _transfer_kv_cache_timed(self, req_meta, phases):
            local_groups = req_meta["local_block_ids"]
            remote_groups = req_meta["remote_block_ids"]
            if not flatten_group_lists(local_groups):
                return                       # full prefix-cache hit: nothing to pull

            if self.base_addr_groups is None:
                raise KVGroupLayoutError(
                    "base_addr_groups was never built — this connector cannot transfer without "
                    "knowing which KV-cache group each allocation holds")

            remote_engine_id = req_meta["remote_engine_id"]
            remote_host = req_meta["remote_host"]
            remote_handshake_port = req_meta["remote_handshake_port"]
            offset = req_meta["offset"]
            tp_num_need_pulls = req_meta["tp_num_need_pulls"]

            if (remote_engine_id not in self.kv_caches_base_addr
                    or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]):
                _t = time.perf_counter()
                self._get_remote_metadata(remote_host, remote_handshake_port)
                phases["meta"] = (time.perf_counter() - _t) * 1e3

            _t = time.perf_counter()
            grouped = self._align_and_group(
                req_meta, local_groups, remote_groups, tp_num_need_pulls)
            phases["plan"] = (time.perf_counter() - _t) * 1e3

            prefill_pp_rank = offset // tp_num_need_pulls
            inner_offset = offset % tp_num_need_pulls

            remote_base = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
            first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
            if (self.vllm_config.speculative_config is not None
                    and prefill_pp_rank == self._prefill_pp_size - 1):
                end_layer_index = end_layer_index + 1
            num_cache_per_layer = len(next(iter(self.kv_caches.values())))
            lo = first_layer_index * num_cache_per_layer
            hi = end_layer_index * num_cache_per_layer
            local_base = self.kv_caches_base_addr[self.local_engine_id][
                self.local_handshake_port][lo:hi]
            # Slice the group map identically, or address k would be attributed to another layer.
            addr_groups = self.base_addr_groups[lo:hi]

            # zip(local_base, remote_base) pairs index k on both engines and assumes index k names
            # the SAME logical layer on each — which holds only because P and D iterate `kv_caches`
            # in the same order. Note remote_base is deliberately NOT sliced by [lo:hi] while
            # local_base is (stock behaviour); the two coincide only at PP=1. A length mismatch
            # means the pairing has shifted and every subsequent offset is against the wrong layer.
            if len(remote_base) != len(local_base):
                raise KVGroupLayoutError(
                    f"prefill offered {len(remote_base)} base addresses but this decode is using "
                    f"{len(local_base)}; index k no longer names the same layer on both sides, so "
                    "every transfer below would target the wrong layer. Refusing rather than "
                    "serving against a shifted map.")

            remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
            session_id = f"{remote_host}:{remote_transfer_port}"

            src_list, dst_list, length_list = [], [], []
            block_length = len(self.block_len)
            # Duplicate (local, remote) address pairs are the shared-tensor layout showing through;
            # processing them again re-sends bytes already in flight. `k` is kept from the ORIGINAL
            # enumeration because block_len is selected by `k % block_length` (K/V alternation for
            # MLA) — renumbering the survivors would pick the wrong cache's block length.
            keep = transfer_indices(local_base, remote_base)
            if not self._logged_amplification:
                self._logged_amplification = True
                logger.info("MooncakeConnectorFF: %d of %d base-address pairs carry distinct work "
                            "(%.1fx transfer amplification avoided).", len(keep), len(local_base),
                            len(local_base) / len(keep) if keep else 1.0)
            _t = time.perf_counter()
            groups_covered: set = set()
            per_block: dict = {}
            for k in keep:
                src_layer_base_addr = local_base[k]
                dst_layer_base_addr = remote_base[k]
                block_len = self.block_len[k % block_length]
                inner_block_len = block_len // tp_num_need_pulls
                for gi in addr_groups[k]:
                    if gi >= len(grouped):
                        continue
                    grouped_remote, grouped_local = grouped[gi]
                    for remote_block_id, local_block_id in zip(grouped_remote, grouped_local):
                        src = (src_layer_base_addr + local_block_id[0] * block_len
                               + inner_offset * inner_block_len)
                        dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(inner_block_len * len(local_block_id))
                        groups_covered.add(gi)
                        if self.capture_descriptors:
                            # One entry per BLOCK per address, so a single block can be replayed on
                            # its own. A block spans every kept address (one per layer slot and
                            # cache), so its list holds all of them — replaying one address would
                            # restore only a slice of the block.
                            for rb, lb in zip(remote_block_id, local_block_id):
                                per_block.setdefault((int(gi), int(lb)), []).append(block_segment(
                                    src_layer_base_addr, dst_layer_base_addr, lb, rb,
                                    block_len, inner_block_len, inner_offset))

            # Per-request coverage check. `build_base_addr_groups` proves at startup that every
            # group is reachable through SOME address; this proves that for THIS request every group
            # that actually had blocks to pull emitted segments. The gap between the two is a group
            # whose KV silently stays whatever was in the block — the decode then attends over stale
            # KV for those layers and the only symptom is the output text. Costs a set of small ints
            # per request.
            if self.capture_descriptors:
                self.last_descriptors = per_block
                self.last_session_id = session_id
            self.last_segment_count = len(src_list)
            phases["descriptors"] = (time.perf_counter() - _t) * 1e3
            self._audit_descriptor_coverage(req_meta, grouped, keep, addr_groups, len(src_list))

            wanted = {gi for gi, (_r, local_ids) in enumerate(grouped) if local_ids}
            if wanted - groups_covered:
                raise KVGroupLayoutError(
                    f"request {req_meta['remote_request_id']}: KV-cache groups "
                    f"{sorted(wanted - groups_covered)} have blocks to pull but no registered "
                    f"allocation carried them (covered {sorted(groups_covered)}). Their layers "
                    "would decode against stale KV with no other symptom than wrong output.")

            if not src_list:
                # Nothing to read. Under v2 this is the fully-deduped case — every block was
                # satisfied locally — and its aliases are ready NOW, so the landed-hook still has to
                # fire. Returning without it would strand the request's aliases until they expired
                # and send every one of its blocks to recompute.
                _t = time.perf_counter()
                self._after_transfer(req_meta)
                phases["after"] = (time.perf_counter() - _t) * 1e3
                return

            _t = time.perf_counter()
            self._issue_transfer(req_meta, session_id, src_list, dst_list, length_list)
            phases["xfer"] = (time.perf_counter() - _t) * 1e3

            _t = time.perf_counter()
            self._reformat_after_pull(grouped, offset, tp_num_need_pulls)
            phases["reformat"] = (time.perf_counter() - _t) * 1e3

            _t = time.perf_counter()
            self._after_transfer(req_meta)
            phases["after"] = (time.perf_counter() - _t) * 1e3

        def _reformat_after_pull(self, grouped, offset, tp_num_need_pulls):
            """Stock's post-pull NZ/cat reformat, unchanged except for the block list it is given.

            The reformat is per PHYSICAL block and does not care which group a block came from, so
            the union of every group's coalesced runs is the right argument. Keeping the branch
            identical to the base class (fused op vs. original path, and the same
            ``is_kv_transfer_end`` gate) matters: this runs on the last pull of a multi-rank
            transfer only, and doing it twice or never both corrupt the cache."""
            is_kv_transfer_end = (offset == tp_num_need_pulls * self._prefill_pp_size - 1)
            need_cat_cache = tp_num_need_pulls > 1 and is_kv_transfer_end
            need_nz_cache = get_ascend_config().enable_kv_nz and is_kv_transfer_end
            if not (need_nz_cache or need_cat_cache):
                return
            all_local = [run for _gr, gl in grouped for run in gl]
            if not all_local:
                return
            use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    self.reformat_kv_cache_with_fused_op(all_local, tp_num_need_pulls)
                if need_nz_cache:
                    self.reformat_kv_cache(all_local, tp_num_need_pulls, False, need_nz_cache)
            else:
                self.reformat_kv_cache(all_local, tp_num_need_pulls, need_cat_cache, need_nz_cache)

    class MooncakeConnectorWorkerFF(MooncakeConnectorWorker):
        """Builds the group maps and hands them to the recv thread."""

        _RECV_THREAD_CLS = KVCacheRecvingThreadFF

        def __init__(self, vllm_config, engine_id, kv_cache_config=None):
            # Keep the post-split layout handed to us by the factory. Reading it here rather than
            # from _ACTIVE_RUNNER is what makes this connector independent of whether the BFF patch
            # happened to be imported before NPUModelRunner was constructed in THIS process — see
            # _build_base_addr_groups.
            self._kv_cache_config = kv_cache_config
            self._layer_group: dict[str, int] = {}
            self._n_groups = 0
            super().__init__(vllm_config, engine_id)

        def register_kv_caches(self, kv_caches):
            # The base class constructs and STARTS the recv thread inside this call, so the class
            # swap has to be in place before it runs.
            _orig_cls = None
            import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector as _mc
            if self.kv_role != "kv_producer":
                _orig_cls = _mc.KVCacheRecvingThread
                _mc.KVCacheRecvingThread = self._RECV_THREAD_CLS

            # BFF's shared-tensor layout makes several layers report the same data_ptr(), and the
            # base registers one entry per (layer, K/V) without dedup → Mooncake rejects the region
            # as overlapped. Wrap the singleton's register_buffer for the duration of the super()
            # call; see dedup_registration_regions for why only the registration list shrinks.
            from vllm_ascend.distributed.kv_transfer.utils import (
                mooncake_transfer_engine as _mte,
            )
            _orig_register = _mte.global_te.register_buffer

            def _dedup_register(ptrs, sizes, _orig=_orig_register):
                kept_ptrs, kept_sizes = dedup_registration_regions(ptrs, sizes)
                logger.info(
                    "MooncakeConnectorFF: registering %d of %d KV regions (%d dropped as duplicates "
                    "of BFF's shared-tensor layout), %.2f GiB total.",
                    len(kept_ptrs), len(ptrs), len(ptrs) - len(kept_ptrs),
                    sum(kept_sizes) / (1024 ** 3))
                return _orig(kept_ptrs, kept_sizes)

            _mte.global_te.register_buffer = _dedup_register
            try:
                super().register_kv_caches(kv_caches)
            finally:
                _mte.global_te.register_buffer = _orig_register
                if _orig_cls is not None:
                    _mc.KVCacheRecvingThread = _orig_cls

            self._layer_names = list(kv_caches.keys())
            self._caches_per_layer = len(next(iter(kv_caches.values())))
            self.base_addr_groups = self._build_base_addr_groups()
            if self.kv_role != "kv_producer" and self.kv_recv_thread is not None:
                self.kv_recv_thread.base_addr_groups = self.base_addr_groups
            logger.info("MooncakeConnectorFF: mapped %d base addresses over %d KV-cache groups "
                        "(%d layers x %d caches).", len(self.base_addr_groups),
                        len({g for gs in self.base_addr_groups for g in gs}),
                        len(self._layer_names), self._caches_per_layer)

        def _build_base_addr_groups(self) -> list[list[int]]:
            groups = resolve_kv_cache_groups(self._kv_cache_config, _active_runner())
            layer_group = build_layer_group_map(groups)
            # Kept for the fusion hook, which needs layer_name -> group index on every
            # save_kv_layer. It is derived here anyway; discarding it would mean rebuilding the
            # same map from the same source a second time.
            self._layer_group = layer_group
            self._n_groups = len(groups)
            # The base class builds its address list as a LOCAL and only ever keeps it on the
            # handshake metadata (mooncake_connector.py:1222,1238) — the worker has no
            # `kv_caches_base_addr` attribute of its own. (`self.kv_caches_base_addr` does exist on
            # the RECV THREAD, but there it is a dict keyed by engine id, not this flat list.)
            return build_base_addr_groups(
                self.xfer_handshake_metadata.kv_caches_base_addr, self._layer_names, layer_group,
                self._caches_per_layer, len(groups))

        def start_load_kv(self, metadata):
            for meta in metadata.requests.values():
                if meta.remote_pcp_size * meta.remote_dcp_size > 1:
                    raise KVGroupLayoutError(
                        "prefill context/decode parallelism (pcp*dcp > 1) splits block ids in a way "
                        "this connector does not yet generalise per KV-cache group. Refusing rather "
                        "than splitting the wrong axis. Run with pcp=dcp=1, or use the layerwise "
                        "connector.")
            return super().start_load_kv(metadata)

    class MooncakeConnectorFF(MooncakeConnector, SupportsHMA):
        """The Ascend pull connector, taught BFF's multi-group KV layout.

        ``SupportsHMA`` is what routes vLLM to ``request_finished_all_groups``; without it the
        scheduler's ``_connector_finished`` asserts ``len(kv_cache_groups) == 1`` and dies on the
        first finished request."""

        # Named so a subclass can substitute its own halves. Constructing the classes by name here
        # instead would make an override silently ineffective — v2 would run v1's worker, with no
        # dedup engine and no signature server, and the only symptom would be a benchmark that shows
        # no improvement.
        _WORKER_CLS = MooncakeConnectorWorkerFF
        _SCHEDULER_CLS = MooncakeConnectorSchedulerFF

        def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole",
                     kv_cache_config: "KVCacheConfig | None" = None):
            assert vllm_config.kv_transfer_config is not None
            self.engine_id = vllm_config.kv_transfer_config.engine_id
            self._connector_metadata = MooncakeConnectorMetadataFF()
            self._kv_cache_config = kv_cache_config
            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler = self._SCHEDULER_CLS(
                    vllm_config, str(self.engine_id))
                self.connector_worker = None
            else:
                self.connector_scheduler = None
                # The worker is where the group layout is read; hand it the config the factory gave
                # us so it never has to depend on _ACTIVE_RUNNER being published in this process.
                self.connector_worker = self._WORKER_CLS(
                    vllm_config, str(self.engine_id), kv_cache_config)

            # Fusion accumulator: producer-side worker role only. MooncakeFFProducer is documented
            # transport-agnostic ("pure torch + pd_fuse — no NPU, no ZMQ"), so the layerwise
            # connector's engine is reused verbatim rather than reimplemented for this transport.
            self._ff_producer = None
            self._ff_mla = False
            self._ff_warned_unmapped = False
            self._ff_groups_done = 0
            if (_FF_FUSE and role != KVConnectorRole.SCHEDULER
                    and vllm_config.kv_transfer_config.is_kv_producer):
                from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
                    MooncakeFFProducer,
                )
                self._ff_producer = MooncakeFFProducer()
                self._ff_mla = bool(getattr(vllm_config.model_config, "use_mla", False))
                logger.info("MooncakeConnectorFF: producer fusion enabled (ship=%s, groups=%s).",
                            _FF_SHIP, "all" if _FF_GROUPS is None else sorted(_FF_GROUPS))
            if os.environ.get("BFF_PD_FUSE", "0") != "1":
                logger.warning(
                    "MooncakeConnectorFF selected with BFF_PD_FUSE!=1. This connector requires the "
                    "BFF multi-group KV layout; with the split off it is strictly worse than the "
                    "stock MooncakeConnectorV1.")

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            """Producer fusion hook — the one place this connector sees KV as it is written.

            The vendored pull connector's ``save_kv_layer`` is a no-op ``pass``
            (mooncake_connector.py:874-878), so this adds work where there was none rather than
            wrapping anything. vLLM drives it through the ``@maybe_transfer_kv_layer`` decorator on
            ``unified_attention``; it fires only because ``requires_piecewise_for_cudagraph`` keeps
            us out of a full graph, which would capture this call away and leave fusion silently
            inert.

            Everything is wrapped: fusion is an optimisation, and no failure in it may break a
            transfer that is otherwise correct."""
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            if self._ff_producer is None:
                return
            try:
                # Timed OUTSIDE the accumulate so `forward_ms` captures everything fusion adds to
                # the prefill thread — the per-layer accounting and the stash write too, not just
                # the clustering. That total is the number an overhead claim has to defend.
                _t0 = time.perf_counter()
                self._ff_accumulate(layer_name, kv_layer)
                self._ff_producer.note_forward((time.perf_counter() - _t0) * 1e3)
                # Dumped from HERE, never from inside _ff_accumulate. That function returns early
                # for the warmup group and for every layer that does not complete a group, so it is
                # reached on roughly one layer in 28 — and a backstop gated behind the conditions it
                # exists to survive is not a backstop. The layerwise connector learned this by
                # freezing a prefill node's ledger at its step-1 snapshot for an entire run.
                self._ff_producer.maybe_dump_stats(_PD_STATS_DIR)
            except Exception as e:  # pragma: no cover - never break the transfer
                logger.warning("MooncakeConnectorFF: producer fusion failed on %s: %s",
                               layer_name, e)

        def _ff_accumulate(self, layer_name, kv_layer) -> None:
            meta = self._connector_metadata
            fuse_reqs = getattr(meta, "fuse_reqs", None)
            if not fuse_reqs:
                return                       # no prefill completed this step → nothing to cluster
            worker = self.connector_worker
            gi = group_of(worker._layer_group, layer_name)
            # Group 0 is the warmup group and is deliberately never fused: it holds the first and
            # last two layers, whose KV the decode is most sensitive to. An unknown layer is skipped
            # rather than guessed — unlike the transfer path this is not fatal, because skipping only
            # forfeits compression.
            #
            # But say so LOUDLY once. on_layer detects a completed group by COUNT
            # (`len(seen) >= len(group_layer_names)`), so a single unresolvable layer means its
            # group never reaches its count and never clusters — fusion goes inert for that group
            # with no error anywhere. A silently inert producer is what makes a benchmark read as
            # "BFF doesn't help" instead of "BFF never ran".
            if gi is None:
                if not self._ff_warned_unmapped:
                    self._ff_warned_unmapped = True
                    logger.error(
                        "MooncakeConnectorFF: layer %r maps to no KV-cache group, so its group can "
                        "never complete and FUSION IS INERT for it. The connector's layer names "
                        "(%d known) disagree with the forward context's.",
                        layer_name, len(worker._layer_group))
                return

            # Wire-bytes denominator, tallied for EVERY group including the warmup one and BEFORE
            # the fusion filter below. Compression has to be quoted against all the blocks that
            # actually crossed the wire, not just the ones fusion was allowed to look at — measuring
            # it against the fusion groups alone would flatter the ratio by ~1/7 here.
            n_block_layers = sum(len(groups[gi])
                                 for (_rid, _ext, groups) in fuse_reqs if gi < len(groups))
            self._ff_producer.note_transferred(n_block_layers)

            if gi <= 0:
                return
            if _FF_GROUPS is not None and gi not in _FF_GROUPS:
                return

            self._ff_producer.reset_step(id(meta))
            requests = [(ext_id, list(groups[gi]))
                        for (_rid, ext_id, groups) in fuse_reqs if gi < len(groups)]
            if not requests:
                return

            # Standard attention clusters on K only. MLA splits the key across two latent tensors
            # sharing one physical block, so both must be compared or the redirect would alias a
            # rope cache that was never looked at.
            caches = ([kv_layer[0], kv_layer[1]]
                      if self._ff_mla and len(kv_layer) > 1 else [kv_layer[0]])
            group_layers = {ln for ln, g in worker._layer_group.items() if g == gi}
            result = self._ff_producer.on_layer(
                gi, layer_name, caches, group_layers, requests)
            if not result:
                return                       # group not complete yet, or nothing worth redirecting
            n_rows = 0
            for ext_id, rows in result.items():
                _FF_ROWS.add(ext_id, gi, rows)
                n_rows += len(rows)
            # First completion is the proof that the whole chain is live — the hook fires, the group
            # reaches its count, and clustering produced something. Without this the only evidence
            # fusion ran at all is a downstream F1 change, which is exactly the ambiguity that cost
            # this project three debugging cycles.
            self._ff_groups_done += 1
            if self._ff_groups_done == 1:
                logger.info("MooncakeConnectorFF: first fusion group completed (group=%d, "
                            "%d requests, %d redirect rows). ship=%s", gi, len(result), n_rows,
                            _FF_SHIP)

        @classmethod
        def requires_piecewise_for_cudagraph(cls, extra_config: dict) -> bool:
            """Refuse a FULL cudagraph while the BFF group split is on.

            vLLM applies this in `VllmConfig.__post_init__` (config/vllm.py:935-957): if the selected
            connector says yes and `cudagraph_mode.has_full_cudagraphs()`, it downgrades the mode to
            PIECEWISE. `FULL_DECODE_ONLY` counts as full (`max_cudagraph_mode() == FULL`), which is
            the mode vllm-ascend defaults to — so without this hook the decode node replays a
            captured graph for every decode step.

            That is not safe under the split. `AscendAttentionBackendImpl.update_graph_params`
            (vllm_ascend/attention/attention_v1.py:404-447) re-reads only `seq_lens` from the live
            metadata on each replay; `query`, `key_cache`, `value_cache`, `block_table` and `output`
            all come from the tuple frozen at capture time. That holds for vLLM's single-group
            contract, where those are persistent buffers updated in place — but BFF turns a dense
            model into a seven-group hybrid with seven separate block tables, and the surrounding
            code is visibly order-sensitive across groups (the Qwen3-next linear_attn/self_attn
            ordering hack at :461-470). The observed symptom is garbage from the FIRST decoded token
            with a verifiably correct KV transfer, clean at one group and broken at seven.

            The GPU sibling demands PIECEWISE for a different reason — real Python work in
            save_kv_layer that a full graph would skip — but the conclusion is the same, and phase B
            will need it here for that reason too.

            Gated on BFF_PD_FUSE so a stock single-group run keeps its full graphs."""
            return os.environ.get("BFF_PD_FUSE", "0") == "1"

        def request_finished_all_groups(self, request, block_ids):
            assert self.connector_scheduler is not None
            return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def register_mooncake_connector_ff() -> None:
        """Register ``MooncakeConnectorFF`` (idempotent)."""
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        if CONNECTOR_NAME in KVConnectorFactory._registry:
            return
        KVConnectorFactory.register_connector(
            CONNECTOR_NAME,
            "kv_fast_fusion_ascend.connectors.mooncake_connector_ff",
            "MooncakeConnectorFF",
        )
        logger.info("BFF Ascend: registered %s (non-layerwise pull transport).", CONNECTOR_NAME)

else:  # pragma: no cover - exercised only off the Ascend stack

    def register_mooncake_connector_ff() -> None:
        logger.warning("MooncakeConnectorFF not registered: the Ascend stack is unavailable.")
