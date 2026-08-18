"""Mooncake BFF v2 on GPU — the transport half. The decision logic lives in
:mod:`kv_fast_fusion.pd_dedup_v2`, shared with the Ascend connectors.

What this file owns is everything specific to vLLM's GPU Mooncake connector, whose transfer is a
decode-initiated **pull** (physically an RDMA write from P into D's registered memory):

* the two-phase exchange — D's first request carries ``want_signatures=True``, P replies with
  signatures and writes nothing, D then sends the real request with the deduplicated block list;
* the sentinel list riding ``MooncakeXferMetadataFF.req_blocks``, which
  ``mooncake_connector_ff._build_transfer_params`` filters before pairing (the pairing is
  POSITIONAL, so a shortened list would write the wrong KV — see ``pd_dedup_v2.SENTINEL``);
* ``process_pulling_result`` as the "KV has landed" signal that releases a request's aliases.

The producer has no forward-path hook at all. Signatures are computed on demand from
``device_kv_caches`` (populated once in ``register_kv_caches``), so v2 needs no ``save_kv_layer``
work, no chunked-prefill accumulation, and no PIECEWISE cudagraph constraint — v1 spent 10.8 ms per
group there and had to force PIECEWISE.

Registered as ``MooncakeConnectorFFv2`` alongside v1 so both can be run against the same benchmark.
Everything hard-won about the HMA/group-aware transfer path is inherited from v1 unchanged: this
subclasses its worker and connector rather than restating them.
"""

import asyncio
import os
from typing import TYPE_CHECKING, Any

import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

from kv_fast_fusion import pd_dedup_v2, pd_lsh
from kv_fast_fusion.connectors import mooncake_connector_ff as v1
from kv_fast_fusion.pd_dedup_v2 import (
    SENTINEL,
    AliasApplier,
    DedupEngine,
    DedupStats,
    SignatureCodec,
    signature_matrix,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = init_logger("vllm.mooncake_connector_ff_v2")

# The signature phase is best-effort: a producer that cannot answer within these seconds gets pulled
# in full. Both are far below the transfer timeouts — a slow exchange must never become a stall.
_SIG_PHASE_TIMEOUT = float(os.environ.get("BFF_V2_SIG_TIMEOUT", "10"))
_SIG_READY_TIMEOUT = float(os.environ.get("BFF_V2_READY_TIMEOUT", "10"))

# Re-exported constants/helpers. Deliberately NOT the master switch: an alias of a mutable flag
# reads like the switch but is a dead copy, and monkeypatching it silently does nothing. The switch
# lives at ``pd_dedup_v2.V2_ENABLED`` and only there.
_SENTINEL = SENTINEL
_signature_matrix = signature_matrix

__all__ = ["DedupStats", "MooncakeConnectorFFv2", "MooncakeConnectorWorkerFFv2", "SignatureCodec",
           "register_mooncake_connector_ff_v2"]


if v1._MOONCAKE_AVAILABLE:      # the same gate v1 uses

    import vllm.envs as envs
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeXferResponseStatus,
    )

    class MooncakeConnectorWorkerFFv2(v1.MooncakeConnectorWorkerFF):
        """Adds signature service on the producer and dedup planning on the decode.

        Inherits v1's ``_build_transfer_params`` unchanged — the group-aware pairing is orthogonal
        to who decides, and it is the part that took the longest to get right."""

        def __init__(self, vllm_config, engine_id):
            super().__init__(vllm_config, engine_id)
            self._jl: list = [None]
            self._engine = DedupEngine(lock=self._ff_lock)

        @property
        def _dedup(self) -> DedupStats:
            return self._engine.stats

        # -- producer: answer with signatures, never with decisions ---------------------
        def signatures_for(self, block_ids_by_group: dict[int, list[int]]) -> dict:
            """Compute this producer's signatures for the requested blocks, per fusion group.

            Reads straight out of ``device_kv_caches``, so it costs the producer nothing on the
            forward path — the reason v2 needs no ``save_kv_layer`` hook and no PIECEWISE
            cudagraph constraint."""
            out: dict[int, dict] = {}
            caches = getattr(self, "device_kv_caches", None) or {}
            for gi, block_ids in block_ids_by_group.items():
                layers = [caches[ln] for ln in sorted(self._group_layers.get(gi, ()))
                          if ln in caches]
                if not layers or not block_ids:
                    continue
                is_mla = layers[0].ndim == 3
                sig, norms = signature_matrix(layers, block_ids, is_mla, self._jl)
                if sig is None:
                    continue
                proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
                hashes = pd_lsh.sub_hashes_device(sig, proj).cpu().tolist()
                out[gi] = SignatureCodec.encode(sig, norms, hashes)
            return out

        async def send_kv_to_decode(self, identity, sock, meta):
            """Phase 1 answers with signatures and writes nothing; phase 2 is the stock transfer."""
            if getattr(meta, "want_signatures", False):
                return await self._send_signatures(identity, sock, meta)
            return await super().send_kv_to_decode(identity, sock, meta)

        async def _send_signatures(self, identity, sock, meta) -> None:
            """Reply with per-block signatures for exactly the blocks phase 2 would write.

            The rows are computed over ``local_ids[-len(remote_ids):]`` — the same tail alignment
            ``_build_transfer_params`` uses — so signature row *i* is the decode's slot *i*."""
            out: dict[str, dict[int, dict]] = {}
            marked = []
            try:
                for d_req_id, (transfer_id, remote_groups) in meta.req_blocks.items():
                    send_meta = self.reqs_need_send.get(transfer_id)
                    if send_meta is None:
                        continue
                    try:
                        await asyncio.wait_for(send_meta.ready.wait(), _SIG_READY_TIMEOUT)
                    except (TimeoutError, asyncio.TimeoutError):
                        continue          # not prefilled yet → D pulls this one whole
                    if transfer_id not in self.reqs_need_send:
                        continue          # expired between the wait and here
                    # Hold off the abort-timeout sweep for the duration: it only reaps entries with
                    # sending == 0, and one run showed that sweep firing 16 times.
                    send_meta.sending += 1
                    marked.append(send_meta)
                    local_groups = send_meta.local_block_ids
                    per_group: dict[int, list[int]] = {}
                    for gi, remote_ids in enumerate(remote_groups):
                        if gi <= 0 or not remote_ids:
                            continue      # group 0 is the warmup group — never fused
                        local_ids = local_groups[gi] if gi < len(local_groups) else []
                        if len(local_ids) < len(remote_ids):
                            continue      # P has fewer blocks than D wants; phase 2 will error
                        per_group[gi] = list(local_ids[-len(remote_ids):])
                    sigs = self.signatures_for(per_group)
                    if sigs:
                        out[d_req_id] = sigs
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF v2: signature phase failed (%s) — D will pull in full.", e)
                out = {}
            finally:
                for sm in marked:
                    sm.sending -= 1
            response = v1.MooncakeXferResponseFF(
                status=MooncakeXferResponseStatus.FINISH, signatures=out or None)
            await sock.send_multipart((identity, self._encoder.encode(response)))

        # -- decode: plan the pull ------------------------------------------------------
        def plan_pull(self, req_blocks: dict, signatures: dict, threshold=None) -> dict:
            """Replace the blocks this decode can satisfy locally with the sentinel."""
            return self._engine.plan(req_blocks, signatures, threshold)

        def take_pending_alias(self, req_id: str):
            return self._engine.pending_alias(req_id)

        def drain_pending_alias(self) -> dict:
            return self._engine.drain_ready()

        def note_resident(self, group: int, sigs, hashes, norms, block_ids, owner="") -> None:
            self._engine.note_resident(group, sigs, hashes, norms, block_ids, owner)

        def is_resident(self, group: int, block_id: int) -> bool:
            return self._engine.is_resident(group, block_id)

        def _on_blocks_freed(self, freed_ids) -> None:
            self._engine.on_blocks_freed(freed_ids)

        # -- decode: the two-phase pull --------------------------------------------------
        async def receive_kv_from_single_worker(self, worker_addr, pull_metas):
            """Ask for signatures, decide, then request only what is left.

            Replaces the stock body (which builds one metadata and sends it) rather than wrapping
            it, because the deduplicated block list has to go into the message it builds. The
            transfer half below is the stock loop verbatim."""
            req_ids = set(pull_metas)
            base = {rid: [list(g) for g in pm.local_block_ids] for rid, pm in pull_metas.items()}
            planned = base

            # Read through the module, never a from-import copy: the switch must have exactly one
            # home, or turning it off in one place silently leaves the other on.
            if pd_dedup_v2.V2_ENABLED:
                signatures = await self._request_signatures(worker_addr, pull_metas, base)
                if signatures:
                    try:
                        planned = self.plan_pull(base, signatures)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning("BFF v2: planning failed (%s) — pulling in full.", e)
                        self._forget_pending(req_ids)
                        planned = base

            metadata = v1.MooncakeXferMetadataFF(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks={rid: (pm.transfer_id, planned[rid])
                            for rid, pm in pull_metas.items()},
                kv_caches_base_addr=self.kv_caches_base_addr,
            )
            try:
                with make_zmq_socket(
                    self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
                ) as sock:
                    sock.setsockopt(
                        zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000)
                    await sock.send(self._encoder.encode(metadata))
                    while True:
                        ret_msg = await sock.recv()
                        response = self._xfer_resp_decoder.decode(ret_msg)
                        if response.status == MooncakeXferResponseStatus.ERROR:
                            logger.error("Error transferring kvcache for %s: %s",
                                         req_ids, response.err_msg)
                            self._forget_pending(req_ids)
                            return
                        self.process_pulling_result(response, pull_metas)
                        if response.status == MooncakeXferResponseStatus.FINISH:
                            break
            except zmq.ContextTerminated:
                self._forget_pending(req_ids)
                logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
            except Exception as e:
                self._forget_pending(req_ids)
                logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)

        async def _request_signatures(self, worker_addr, pull_metas, base) -> dict:
            """Phase 1, on its own short-lived socket.

            Best-effort by construction: on any failure or timeout this returns nothing and the
            caller pulls the full block list, which is exactly what vanilla would have sent. A slow
            producer must cost compression, never a stall — hence the separate socket, so a late
            reply can never be mistaken for a phase-2 response."""
            meta = v1.MooncakeXferMetadataFF(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks={rid: (pm.transfer_id, base[rid]) for rid, pm in pull_metas.items()},
                kv_caches_base_addr=self.kv_caches_base_addr,
                want_signatures=True,
            )
            try:
                with make_zmq_socket(
                    self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
                ) as sock:
                    sock.setsockopt(zmq.RCVTIMEO, int(_SIG_PHASE_TIMEOUT * 1000))
                    await sock.send(self._encoder.encode(meta))
                    raw = await asyncio.wait_for(sock.recv(), _SIG_PHASE_TIMEOUT)
                    return self._xfer_resp_decoder.decode(raw).signatures or {}
            except Exception as e:
                self._engine.stats.sig_phase_failed += 1
                logger.warning("BFF v2: signature phase unavailable (%s) — pulling in full.", e)
                return {}

        def _forget_pending(self, req_ids) -> None:
            self._engine.forget(req_ids)

        def process_pulling_result(self, response, pull_metas):
            """v1's failed-pull recovery, plus the "KV has landed" signal.

            Only here do two things become true: this request's aliases may be applied, and the
            blocks it did pull may serve as representatives for later ones. Releasing any earlier is
            the bug that made the first v2 run apply 22 of 26,531 aliases."""
            for rid in (response.err_reqs or []):
                self._forget_pending([rid])
            super().process_pulling_result(response, pull_metas)
            for rid in (response.ok_reqs or []):
                self._engine.release(rid)

        def note_failed_blocks(self, block_ids) -> None:
            """Route blocks that were never written into vLLM's KV-load-failure path."""
            if not block_ids:
                return
            with self._ff_lock:
                self._ff_failed_blocks |= {int(b) for b in block_ids}

    class MooncakeConnectorFFv2(v1.MooncakeConnectorFF):
        """v1's connector with the merge decision moved to the decode.

        The producer-side fusion engine, the redirect-row wire format and the whole
        resolve/hold/expire path on the consumer are simply not used: v2 never emits a redirect that
        might not resolve, so there is nothing to resolve, hold, or expire."""

        _WORKER_CLS = MooncakeConnectorWorkerFFv2

        @classmethod
        def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
            """v1 demanded PIECEWISE because it ran real Python per layer inside ``save_kv_layer``.
            v2 does no forward-path work at all, so a full cudagraph is fine again."""
            return False

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
            """No-op: signatures are computed on demand from the registered KV cache."""

        def _applier(self) -> AliasApplier:
            a = getattr(self, "_ff_applier", None)
            if a is None:
                worker = self.connector_worker
                a = self._ff_applier = AliasApplier(
                    worker._engine, v1.write_runner_block_table, worker.note_failed_blocks)
            return a

        def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
            # Deliberately skips v1's start_load_kv body (which drives the producer fusion engine
            # and the redirect apply) and calls the transport base directly.
            v1.MooncakeConnector.start_load_kv(self, forward_context, **kwargs)
            self._ff_step += 1
            if self.is_producer:
                return
            self._ff_consumer_apply()
            stats = self.connector_worker._engine.stats
            if stats.should_dump(self._ff_step):
                stats.dump()

        def get_kv_connector_stats(self):
            merges = self._ff_pending_merges
            self._ff_pending_merges = None
            if merges and self._tp_group() is not None:
                return v1.BFFMergeStats(data={"bff_merges": merges})
            return v1.MooncakeConnector.get_kv_connector_stats(self)

        def _ff_consumer_apply(self) -> None:
            """Apply landed aliases; see :class:`~kv_fast_fusion.pd_dedup_v2.AliasApplier`."""
            self._ff_pending_merges = None
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                applier = self._applier()
                before = applier._engine.stats.applied
                applier.apply(getattr(_bp, "_ACTIVE_RUNNER", None))
                self._ff_pending_merges = applier.pending_merges
                n = applier._engine.stats.applied - before
                if n:
                    self._ff_applied += n
                    logger.info("BFF v2 apply | aliases_applied=%d | recompute(cum)=%d",
                                n, applier._engine.stats.recomputed)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF v2 consumer apply failed: %s", e)


def register_mooncake_connector_ff_v2() -> None:
    """Register ``MooncakeConnectorFFv2`` (idempotent), by path + name so importing this module's
    Mooncake half is not required on a box without the package."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    if "MooncakeConnectorFFv2" in KVConnectorFactory._registry:
        return
    KVConnectorFactory.register_connector(
        "MooncakeConnectorFFv2",
        "kv_fast_fusion.connectors.mooncake_connector_ff_v2",
        "MooncakeConnectorFFv2",
    )
    logger.info("Fast fusion P/D patch: registered MooncakeConnectorFFv2.")
