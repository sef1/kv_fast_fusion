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

from vllm.logger import init_logger

logger = init_logger("vllm.mooncake_connector_ff_ascend")

# Registered connector name, selected by `BASELINE=bff_pull` in run_benchmarks.sh.
CONNECTOR_NAME = "MooncakeConnectorFF"


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

    class MooncakeConnectorMetadataFF(MooncakeConnectorMetadata):
        """Same container; ``local_block_ids`` / ``remote_block_ids`` are now per group."""

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
        """Per-group block ids on both ends of the P/D handshake."""

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            params = request.kv_transfer_params
            if params is not None and (params.get("do_remote_prefill", False)
                                       or params.get("do_remote_decode", False)):
                self._reqs_in_batch.add(request.request_id)
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
            return meta

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
            return delay_free_blocks, dict(
                do_remote_prefill=True,
                do_remote_decode=False,
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

        def _transfer_kv_cache(self, req_meta):
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
                self._get_remote_metadata(remote_host, remote_handshake_port)

            # Per group: tail-align, then coalesce contiguous runs. Both are per group because the
            # groups have independent block tables — one group's run of consecutive ids says
            # nothing about another's.
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

            remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
            session_id = f"{remote_host}:{remote_transfer_port}"

            src_list, dst_list, length_list = [], [], []
            block_length = len(self.block_len)
            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(local_base, remote_base)):
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

            if not src_list:
                return
            ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
            if ret < 0:
                raise RuntimeError(
                    f"KV transfer failed for request {req_meta['remote_request_id']} "
                    f"(ret={ret}, {len(src_list)} segments)")

            self._reformat_after_pull(grouped, offset, tp_num_need_pulls)

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

        def register_kv_caches(self, kv_caches):
            # The base class constructs and STARTS the recv thread inside this call, so the class
            # swap has to be in place before it runs.
            _orig_cls = None
            import vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector as _mc
            if self.kv_role != "kv_producer":
                _orig_cls = _mc.KVCacheRecvingThread
                _mc.KVCacheRecvingThread = self._RECV_THREAD_CLS
            try:
                super().register_kv_caches(kv_caches)
            finally:
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
            runner = _active_runner()
            if runner is None:
                raise KVGroupLayoutError(
                    "no active BFF runner, so the KV-cache group layout is unreadable. Serving now "
                    "would transfer every layer with the wrong block table and silently corrupt "
                    "the decode's KV. Launch via `python -m kv_fast_fusion.fast_fusion_main serve`, "
                    "or use the stock MooncakeConnectorV1.")
            groups = runner.kv_cache_config.kv_cache_groups
            layer_group = build_layer_group_map(groups)
            return build_base_addr_groups(
                self.kv_caches_base_addr, self._layer_names, layer_group,
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

        def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole",
                     kv_cache_config: "KVCacheConfig | None" = None):
            assert vllm_config.kv_transfer_config is not None
            self.engine_id = vllm_config.kv_transfer_config.engine_id
            self._connector_metadata = MooncakeConnectorMetadataFF()
            if role == KVConnectorRole.SCHEDULER:
                self.connector_scheduler = MooncakeConnectorSchedulerFF(
                    vllm_config, str(self.engine_id))
                self.connector_worker = None
            else:
                self.connector_scheduler = None
                self.connector_worker = MooncakeConnectorWorkerFF(
                    vllm_config, str(self.engine_id))
            if os.environ.get("BFF_PD_FUSE", "0") != "1":
                logger.warning(
                    "MooncakeConnectorFF selected with BFF_PD_FUSE!=1. This connector requires the "
                    "BFF multi-group KV layout; with the split off it is strictly worse than the "
                    "stock MooncakeConnectorV1.")

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
