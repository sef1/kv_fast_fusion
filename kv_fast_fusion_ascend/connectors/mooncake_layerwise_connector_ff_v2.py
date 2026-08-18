"""BFF v2 for the Ascend ``MooncakeLayerwiseConnector`` — the decode decides what not to receive.

The decision logic is the shared, transport-free core in :mod:`kv_fast_fusion.pd_dedup_v2`, the same
one the GPU Mooncake connector uses. This file owns only what the NPU layerwise transport does
differently, and it differs in exactly one interesting way:

**The exchange inverts; the decision does not.** On GPU the decode initiates (it pulls) and the
decode decides. Here the producer initiates — it *pushes* each layer as soon as it is computed — so
the producer has to **ask**. D still decides, because D is the side that knows what is resident;
P simply requests permission before writing.

The transport makes that surprisingly cheap to wire in, because D already publishes its per-group
block ids to P as ``kv_transfer_params["remote_block_ids"]``, and P pairs them against its own
**positionally** (``group_concurrent_contiguous(remote_block_ids, local_block_ids)``). So "do not
send me this block" is the same :data:`~kv_fast_fusion.pd_dedup_v2.SENTINEL` trick as on GPU, and
for the same non-negotiable reason: a *shortened* list would pair the surviving blocks against the
wrong source and silently write the wrong KV. :func:`filter_sentinels` is the one function that must
never be got wrong.

**Where the exchange runs.** On the sender thread (``KVCacheSendingLayerThread._transfer_kv_cache``),
never on the forward path. The round trip delays that request's transfer by one RTT; putting it in
``save_kv_layer`` would have put it in the model's critical path, which is the one thing layerwise
exists to avoid. One exchange covers every request in a send task for that fusion group, because the
send task already batches them.

**Its own channel, always.** REQ/REP on our own port with our own tag, mirroring v1's
``_FFRedirectSendThread``/``_FFRedirectRecvThread``. v1's module docstring records why: an attempt
to ride the base connector's control plane broke the decode node outright when a vendored thread
body grew a parameter, and every done-signal then raised inside a blanket ``except``. That lesson is
unchanged — the base's messages, signatures and thread bodies are not a stable interface.

**Two safety properties.**

1. The exchange is best-effort. No reply, a timeout, a dead peer ⇒ P writes the group whole, exactly
   as stock would. A slow decode costs compression, never a stall.
2. Because the sentinel and the alias are now *separate messages*, they can disagree — and the
   disagreement is safe in the direction that matters. If P never gets the reply it writes
   everything while D still aliases: a written block is replaced by a similar one, which is the
   intended semantic, merely with wasted bandwidth. The dangerous direction — a block D declined
   that is therefore never written, whose alias then fails to apply — is routed to
   ``_handle_invalid_blocks`` for local recompute by the shared
   :class:`~kv_fast_fusion.pd_dedup_v2.AliasApplier`, exactly as on GPU.

The module top level imports nothing Ascend-specific, so :func:`filter_sentinels` and the message
codec stay importable and unit-testable on any box. v1 is untouched; this registers as
``MooncakeLayerwiseConnectorFFv2`` beside it.
"""

import os
import threading
from typing import Any

import torch
from vllm.logger import init_logger

from kv_fast_fusion import pd_dedup_v2, pd_lsh
from kv_fast_fusion.pd_dedup_v2 import (
    SENTINEL,
    AliasApplier,
    DedupEngine,
    DedupStats,
    KVLayoutError,
    SignatureCodec,
    signature_matrix,
)

logger = init_logger("vllm." + __name__)

# SENTINEL is re-exported: it is the value D writes into remote_block_ids and the value
# filter_sentinels strips, so tests and any future transport should name it from here.
__all__ = ["SENTINEL", "MooncakeLayerwiseConnectorFFv2", "filter_sentinels",
           "register_mooncake_layerwise_ff_v2", "signatures_for_group"]

# Port for the v2 signature exchange, offset from the base connector's side channel so it cannot
# collide with either the base or v1's redirect channel.
FF_V2_PORT_OFFSET = int(os.environ.get("BFF_MOONCAKE_FF_V2_PORT_OFFSET", "21000"))
# Seconds P waits for D's answer before writing the group whole.
SIG_EXCHANGE_TIMEOUT = float(os.environ.get("BFF_V2_SIG_TIMEOUT", "2"))
# Which layers of a fusion group feed the signature.
#   "first" — only the group's first layer, which is all that exists when layerwise wants to send
#             it. Preserves the compute/transfer overlap.
#   "group" — every layer of the group, matching the GPU signature exactly, at the cost of holding
#             the group's transfer until its last layer is written.
# Whether one layer discriminates as well as the concat is an open question the accepted-cosine and
# rel_err histograms answer directly; "first" is the default because it keeps the overlap.
SIG_LAYERS = os.environ.get("BFF_SIG_LAYERS", "first")
# Message tags for our own channel.
MSG_SIG_REQUEST = b"bff_v2_sig_req"
MSG_SIG_REPLY = b"bff_v2_sig_rep"


def filter_sentinels(remote_ids, local_ids):
    """Drop the positions D declined, from BOTH sides of the pairing.

    The producer zips its own block ids against the decode's positionally, so a declined block has
    to be removed from **both** lists at the same index. Returning a shorter ``remote_ids`` alone —
    or filtering after the zip — pairs every subsequent survivor with the wrong source block and
    writes the wrong KV with no error anywhere. This function is the whole reason the sentinel is a
    placeholder rather than a deletion.

    Returns ``(remote_kept, local_kept)``; both are returned unchanged when nothing was declined,
    so a non-v2 request costs one ``any()`` scan."""
    if not remote_ids or not any(b < 0 for b in remote_ids):
        return remote_ids, local_ids
    kept = [i for i, b in enumerate(remote_ids) if b >= 0]
    n_local = len(local_ids)
    return ([remote_ids[i] for i in kept],
            [local_ids[i] for i in kept if i < n_local])


def signatures_for_group(kv_caches: dict, layer_names, block_ids, is_mla, jl_holder,
                         num_blocks=None):
    """Producer-side signature payload for one group's blocks.

    ``layer_names`` is however much of the group has been written when the transport wants to send
    it — one layer under ``BFF_SIG_LAYERS=first``, all of them under ``group``. Returns ``None``
    when there is nothing to describe; raises :class:`~kv_fast_fusion.pd_dedup_v2.KVLayoutError` if
    the cache cannot be indexed by connector block ids, which the caller must count rather than
    swallow."""
    layers = [kv_caches[ln] for ln in layer_names if ln in kv_caches]
    if not layers or not block_ids:
        return None
    sig, norms = signature_matrix(layers, block_ids, is_mla, jl_holder, num_blocks=num_blocks)
    if sig is None:
        return None
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    hashes = pd_lsh.sub_hashes_device(sig, proj).cpu().tolist()
    return SignatureCodec.encode(sig, norms, hashes)


# ---------------------------------------------------------------------------------------------
# Ascend/NPU-only section. Guarded so the pure helpers above stay importable for unit tests.
# ---------------------------------------------------------------------------------------------
try:
    import msgspec
    import zmq
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole, SupportsHMA
    from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        MooncakeLayerwiseConnector,
        get_external_request_id,
    )

    _ASCEND_AVAILABLE = True
except Exception as _e:  # pragma: no cover - optional dependency
    logger.info("MooncakeLayerwiseConnectorFFv2: Ascend stack unavailable (%s); "
                "only the pure dedup glue is importable.", _e)
    _ASCEND_AVAILABLE = False


if _ASCEND_AVAILABLE:

    class _SigReplyServer(threading.Thread):
        """Decode side: answer P's "which of these do you want?" with a sentinel block list.

        A REP socket of our own — own port, own tag — never a rider on the base's control plane.
        Every failure path answers with the request unchanged, so a decode that cannot decide
        degrades to stock behaviour rather than stalling a producer."""

        def __init__(self, host: str, port: int, engine: DedupEngine):
            super().__init__(daemon=True, name="BFF-v2-SigReplyServer")
            self._host, self._port = host, port
            self._engine = engine
            self._dec = msgspec.msgpack.Decoder()
            self._enc = msgspec.msgpack.Encoder()

        def run(self):
            path = make_zmq_path("tcp", self._host, self._port)
            logger.info("BFF v2 signature server (REP) on %s", path)
            ctx = zmq.Context()
            sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.REP, bind=True)
            try:
                while True:
                    try:
                        msg = self._dec.decode(sock.recv())
                        reply = self._handle(msg)
                    except Exception as e:  # pragma: no cover - never kill the listener
                        logger.warning("BFF v2 signature server error: %s", e)
                        reply = (MSG_SIG_REPLY, {})
                    try:
                        sock.send(self._enc.encode(reply))
                    except Exception as e:  # pragma: no cover
                        logger.warning("BFF v2 signature server reply failed: %s", e)
            finally:
                ctx.destroy(linger=0)

        def _handle(self, msg):
            """``(tag, gi, {ext_id: block_ids}, {ext_id: signature payload})`` →
            ``(tag, {ext_id: sentinel block ids})``."""
            if not msg or msg[0] != MSG_SIG_REQUEST:
                return (MSG_SIG_REPLY, {})
            _tag, gi, req_blocks, sigs = msg
            gi = int(gi)
            # The engine speaks per-group lists; this transport asks about one group at a time.
            wrapped = {rid: [[] for _ in range(gi + 1)] for rid in req_blocks}
            for rid, ids in req_blocks.items():
                wrapped[rid][gi] = [int(b) for b in ids]
            planned = self._engine.plan(wrapped, {rid: {gi: p} for rid, p in sigs.items()})
            return (MSG_SIG_REPLY, {rid: planned[rid][gi] for rid in req_blocks})

    class _SigRequestClient:
        """Producer side: one REQ socket per decode peer, used from the sender thread.

        Synchronous by nature — P must know the answer before it writes — but it runs on the
        transfer thread, so the round trip delays one request's KV rather than the model."""

        def __init__(self):
            self._lock = threading.Lock()
            self._ctx = None
            self._socks: dict[tuple, Any] = {}
            self._enc = msgspec.msgpack.Encoder()
            self._dec = msgspec.msgpack.Decoder()

        def ask(self, host, port, gi, req_blocks, sigs) -> dict:
            """Return ``{ext_id: sentinel block ids}``, or ``{}`` on any failure."""
            if host is None or port is None or not req_blocks:
                return {}
            with self._lock:
                try:
                    sock = self._sock_for(host, int(port))
                    sock.send(self._enc.encode((MSG_SIG_REQUEST, int(gi), req_blocks, sigs)))
                    msg = self._dec.decode(sock.recv())
                    if msg and msg[0] == MSG_SIG_REPLY:
                        return msg[1] or {}
                    return {}
                except Exception as e:
                    # A REQ socket that timed out is stuck in the wrong state; drop it so the next
                    # exchange starts clean, and write the group whole this time.
                    logger.warning("BFF v2: signature exchange with %s:%s failed (%s) — "
                                   "sending the group in full.", host, port, e)
                    self._drop(host, port)
                    return {}

        def _sock_for(self, host, port):
            key = (host, port)
            s = self._socks.get(key)
            if s is None:
                if self._ctx is None:
                    self._ctx = zmq.Context()
                path = make_zmq_path("tcp", host, port)
                s = make_zmq_socket(ctx=self._ctx, path=path, socket_type=zmq.REQ, bind=False)
                s.setsockopt(zmq.LINGER, 0)
                s.setsockopt(zmq.SNDTIMEO, int(SIG_EXCHANGE_TIMEOUT * 1000))
                s.setsockopt(zmq.RCVTIMEO, int(SIG_EXCHANGE_TIMEOUT * 1000))
                self._socks[key] = s
            return s

        def _drop(self, host, port):
            s = self._socks.pop((host, int(port)), None)
            if s is not None:
                try:
                    s.close(linger=0)
                except Exception:  # pragma: no cover
                    pass

    class MooncakeLayerwiseConnectorFFv2(MooncakeLayerwiseConnector, SupportsHMA):
        """Layerwise connector where the DECODE decides which blocks are worth sending.

        Subclasses the vendored connector and installs two hooks on the worker, both off the
        forward path. v1 is untouched and can be run against this in the same script."""

        def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole",
                     kv_cache_config: "KVCacheConfig | None" = None):
            super().__init__(vllm_config, role, kv_cache_config)
            self._v2_enabled = pd_dedup_v2.V2_ENABLED
            self._jl: list = [None]
            self._engine: DedupEngine | None = None
            self._applier: AliasApplier | None = None
            self._client: _SigRequestClient | None = None
            self._server: _SigReplyServer | None = None
            self._skip: dict[tuple, tuple] = {}     # (ext_id, gi) -> (remote_kept, local_kept)
            self._asked: set[tuple] = set()         # (send-task key, gi) already exchanged
            self._ff_step = 0
            # The producer has no DedupEngine (it does not decide), but it still has to be able to
            # say why it never asked — see _exchange_for and _warn_if_inert.
            self._stats = DedupStats()
            self._kv_caches: dict = {}
            self._send_tasks = 0
            self._inert_warned = False
            cfg = self._vllm_config.kv_transfer_config
            self.is_producer = cfg.is_kv_producer
            if self._v2_enabled and self.connector_worker is not None:
                self._install_hooks()
                logger.info("MooncakeLayerwiseConnectorFFv2: v2 dedup enabled (role=%s, "
                            "sig_layers=%s).", role, SIG_LAYERS)

        # -- wiring ----------------------------------------------------------------------
        def _install_hooks(self) -> None:
            worker = self.connector_worker
            if self.is_producer:
                self._client = _SigRequestClient()
                self._patch_sender(worker)
            else:
                self._engine = DedupEngine()
                self._applier = AliasApplier(
                    self._engine, self._write_block_table, self._note_failed_blocks)
                host = worker.side_channel_host
                port = worker.side_channel_port + FF_V2_PORT_OFFSET + worker.tp_rank
                self._server = _SigReplyServer(host, port, self._engine)
                self._server.start()

        def _patch_sender(self, worker) -> None:
            """Wrap ``_transfer_kv_cache`` on the sending thread.

            That method is the only place that sees a whole send task *and* its layer group *and*
            runs off the forward path, which is exactly the three things the exchange needs. It is
            patched lazily because the thread is constructed in ``register_kv_caches``, after
            ``__init__``."""
            self._sender_patched = False

        def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
            super().register_kv_caches(kv_caches)
            # Keep the tensors ourselves. The vendored worker declares `self.kv_caches` and never
            # assigns it — it walks these tensors for their base addresses and drops them — so
            # reading `worker.kv_caches` yields {} forever, which is exactly what made the first
            # Ascend v2 run inert for a whole benchmark with an all-zero stats file and no warning.
            self._kv_caches = dict(kv_caches)
            if not self._v2_enabled or not self.is_producer:
                return
            worker = self.connector_worker
            thread = getattr(worker, "kv_send_layer_thread", None)
            if thread is None or getattr(self, "_sender_patched", False):
                return
            orig_transfer = thread._transfer_kv_cache
            orig_meta = thread.get_transfer_meta

            def _wrapped_transfer(send_task):
                try:
                    self._exchange_for(worker, thread, send_task)
                except Exception as e:  # pragma: no cover - never break the transfer
                    logger.warning("BFF v2 signature exchange failed: %s", e)
                return orig_transfer(send_task)

            def _wrapped_meta(send_task, req_id, req_meta, layer_group_idx):
                """Swap in the surviving block ids for the duration of the original call.

                Filtering here rather than mutating ``req_meta`` keeps the base's own bookkeeping
                (done-signal accounting, ``trans_count``, chunk handling) reading the untouched
                lists."""
                key = (get_external_request_id(req_id), int(layer_group_idx))
                kept = self._skip.get(key)
                if kept is None:
                    return orig_meta(send_task, req_id, req_meta, layer_group_idx)
                gi = int(layer_group_idx)
                keep_remote, keep_local = kept
                save_r = req_meta.remote_block_ids[gi]
                save_l = req_meta.local_block_ids[gi]
                req_meta.remote_block_ids[gi] = keep_remote
                req_meta.local_block_ids[gi] = keep_local
                try:
                    return orig_meta(send_task, req_id, req_meta, layer_group_idx)
                finally:
                    req_meta.remote_block_ids[gi] = save_r
                    req_meta.local_block_ids[gi] = save_l

            thread._transfer_kv_cache = _wrapped_transfer
            thread.get_transfer_meta = _wrapped_meta
            self._sender_patched = True
            logger.info("BFF v2: sender hooks installed on KVCacheSendingLayerThread.")

        # -- producer: ask before writing --------------------------------------------------
        def _exchange_for(self, worker, thread, send_task) -> None:
            """One exchange per (send task, fusion group), covering every request in the task.

            Every path that declines to ask increments a named counter. An exchange that never
            happens is otherwise indistinguishable in the stats from one that happened and found
            nothing, and that ambiguity cost a whole benchmark run."""
            self._send_tasks += 1
            stats = self._stats
            layer_name = send_task.layer_name
            meta = worker.layer_metadata.get(layer_name)
            if meta is None:
                stats.note_skip("empty_group")
                return self._warn_if_inert()
            gi = int(meta.tensor_group_idx[0])
            if gi <= 0:
                return          # group 0 is the warmup group — never fused, and not a problem
            key = (id(send_task.send_request), gi)
            if key in self._asked:
                return          # later layers of the same group reuse the first answer
            self._asked.add(key)
            if len(self._asked) > 4096:
                self._asked = {key}

            if not self._kv_caches:
                stats.note_skip("no_kv_tensors")
                return self._warn_if_inert()

            layer_names = self._signature_layers(worker, gi, layer_name)
            is_mla = bool(getattr(worker, "use_mla", False))
            num_blocks = getattr(getattr(worker, "kv_cache_config", None), "num_blocks", None)
            by_host: dict[tuple, dict] = {}
            for req_id, req_meta in send_task.send_request.items():
                if gi >= len(req_meta.local_block_ids) or gi >= len(req_meta.remote_block_ids):
                    stats.note_skip("empty_group")
                    continue
                local_ids = list(req_meta.local_block_ids[gi])
                remote_ids = list(req_meta.remote_block_ids[gi])
                if not local_ids or len(remote_ids) != len(local_ids):
                    # Mismatched lengths mean the base is doing something (chunking, TP resharding)
                    # this decision cannot reason about. Send it whole.
                    stats.note_skip("length_mismatch")
                    continue
                try:
                    payload = signatures_for_group(self._kv_caches, layer_names, local_ids,
                                                   is_mla, self._jl, num_blocks=num_blocks)
                except KVLayoutError as e:
                    # Refusing is the point: a mis-indexed signature does not fail, it merges
                    # blocks that are not alike. Once per run is enough noise.
                    stats.note_skip("kv_layout_refused")
                    if not self._inert_warned:
                        self._inert_warned = True
                        logger.error("BFF v2: cannot index the KV cache by block id (%s) — dedup "
                                     "is DISABLED for this run.", e)
                    return
                if payload is None:
                    stats.note_skip("no_signature")
                    continue
                ext = get_external_request_id(req_id)
                slot = by_host.setdefault((req_meta.remote_host, req_meta.remote_port),
                                          {"blocks": {}, "sigs": {}, "meta": {}})
                slot["blocks"][ext] = remote_ids
                slot["sigs"][ext] = payload
                slot["meta"][ext] = (req_id, local_ids, remote_ids)

            if not by_host:
                return self._warn_if_inert()

            for (host, port), slot in by_host.items():
                if port is None:
                    stats.note_skip("no_peer")
                    continue
                stats.exchanges += 1
                answer = self._client.ask(host, int(port) + FF_V2_PORT_OFFSET + worker.tp_rank,
                                          gi, slot["blocks"], slot["sigs"])
                if not answer:
                    stats.sig_phase_failed += 1
                for ext, sentinels in (answer or {}).items():
                    entry = slot["meta"].get(ext)
                    if entry is None or len(sentinels) != len(entry[2]):
                        stats.note_skip("length_mismatch")
                        continue    # D answered about something else; send that request whole
                    _rid, local_ids, _remote = entry
                    keep_remote, keep_local = filter_sentinels(list(sentinels), local_ids)
                    if len(keep_remote) != len(entry[2]):
                        self._skip[(ext, gi)] = (keep_remote, keep_local)
                        stats.blocks_withheld += len(entry[2]) - len(keep_remote)
                    # Deliberately does NOT count planned/dropped blocks: the decode already counts
                    # those authoritatively (it is the side that decides), and both processes dump
                    # into the same stats directory, so counting here would double every figure the
                    # collector sums. The producer's job in the stats is only to say whether it
                    # asked — and blocks_withheld, which is its own independent check that the two
                    # sides agree about what was skipped.

        def _warn_if_inert(self) -> None:
            """Say it out loud, once, when v2 is installed but has never asked anything.

            The first Ascend run produced a perfectly healthy-looking benchmark and an all-zero
            stats file; nothing in either said "this feature did not run"."""
            if self._inert_warned or self._send_tasks < 64 or self._stats.exchanges:
                return
            self._inert_warned = True
            reasons = " ".join(f"{k}={v}" for k, v in self._stats.skip_reasons.items() if v)
            logger.warning(
                "BFF v2 is INERT: %d send tasks and not one signature exchange (%s). No block will "
                "ever be deduplicated in this run.", self._send_tasks, reasons or "no reason "
                "recorded — this is a bug in the accounting, not a quiet success")

        def dump_producer_stats(self) -> None:
            self._stats.dump()

        def _signature_layers(self, worker, gi: int, layer_name: str) -> list[str]:
            if SIG_LAYERS != "group":
                return [layer_name]
            return [ln for ln, m in worker.layer_metadata.items()
                    if int(m.tensor_group_idx[0]) == gi and ln in self._kv_caches]

        # -- consumer: apply the aliases ---------------------------------------------------
        @staticmethod
        def _write_block_table(runner, rid, gi, new_blocks) -> bool:
            from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
                _ff_write_runner_block_table,
            )
            return _ff_write_runner_block_table(runner, rid, gi, new_blocks)

        def _note_failed_blocks(self, block_ids) -> None:
            """Blocks D declined that were then never written and could not be aliased.

            Routed into vLLM's KV-load-failure path so the owning request recomputes locally —
            slower, never wrong."""
            worker = self.connector_worker
            store = getattr(worker, "_ff_failed_blocks", None)
            if store is None:
                store = worker._ff_failed_blocks = set()
            store |= {int(b) for b in block_ids}

        def get_block_ids_with_load_errors(self) -> set[int]:
            worker = self.connector_worker
            store = getattr(worker, "_ff_failed_blocks", None)
            if not store:
                return set()
            worker._ff_failed_blocks = set()
            return store

        def start_load_kv(self, forward_context, **kwargs) -> None:
            super().start_load_kv(forward_context, **kwargs)
            self._ff_step += 1
            if self._applier is None:
                # Producer: no aliases to apply, but its skip counters are the only record of
                # whether the exchange ever ran, so they still have to reach the stats file.
                if self._v2_enabled and self._stats.should_dump(self._ff_step):
                    self._stats.dump()
                return
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                self._applier.apply(getattr(_bp, "_ACTIVE_RUNNER", None))
                stats = self._engine.stats
                if stats.should_dump(self._ff_step):
                    stats.dump()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF v2 consumer apply failed: %s", e)

        def get_finished(self, finished_req_ids: set[str]):
            """The transport's "KV has landed" signal.

            Releasing a request's aliases any earlier is the bug that made the first GPU v2 run
            apply 22 of 26,531 aliases: the apply path expires a map whose owner has not been
            batched within ~1.2 s, and an owner cannot be batched until its KV arrives."""
            done_sending, done_recving = super().get_finished(finished_req_ids)
            if self._engine is not None:
                for rid in (done_recving or ()):
                    self._engine.release(get_external_request_id(rid))
            return done_sending, done_recving

    def register_mooncake_layerwise_ff_v2() -> None:
        """Register ``MooncakeLayerwiseConnectorFFv2`` beside v1 (idempotent)."""
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        if "MooncakeLayerwiseConnectorFFv2" in KVConnectorFactory._registry:
            return
        KVConnectorFactory.register_connector(
            "MooncakeLayerwiseConnectorFFv2",
            "kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff_v2",
            "MooncakeLayerwiseConnectorFFv2",
        )
        logger.info("Fast fusion Ascend patch: registered MooncakeLayerwiseConnectorFFv2.")
