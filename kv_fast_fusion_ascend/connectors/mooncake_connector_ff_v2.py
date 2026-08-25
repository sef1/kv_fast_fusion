"""BFF v2 for the Ascend NON-layerwise (pull) connector — the decode decides what not to READ.

The decision logic is the shared, transport-free core in :mod:`kv_fast_fusion.pd_dedup_v2`, the same
one the GPU Mooncake connector and the Ascend layerwise connector use. This file owns only what the
NPU pull transport does differently — and of the three transports now running v2, this one is the
simplest, for a reason worth stating up front.

**Nothing about the decision crosses the wire.**

======================  ===========================  ==========  ==================================
transport               who moves the bytes          who decides how "don't send" travels
======================  ===========================  ==========  ==================================
GPU Mooncake            D pulls, **P writes**        D          sentinels ride ``req_blocks`` to P
Ascend layerwise        **P pushes**                 D          P asks D; sentinels ride the reply
**Ascend pull (here)**  **D reads** (RDMA read)      D          **nowhere — D omits the read**
======================  ===========================  ==========  ==================================

Because D both decides and executes the transfer, a declined block is simply a read D never issues.
The positional-pairing hazard that :func:`filter_sentinels` exists to guard becomes purely local:
D filters its own ``(remote_ids, local_ids)`` pairs. It is still the one thing that must not be got
wrong — dropping from one list only, or filtering AFTER ``group_concurrent_contiguous``, pairs every
survivor against the wrong source block and writes the wrong KV with no error anywhere — but the
failure can no longer be caused by a peer.

**What P has to do.** Only answer questions. D asks for signatures of the blocks it is about to read;
P computes them on demand from its registered KV cache. There is no producer forward-path hook at
all: no ``save_kv_layer`` work, no chunked-prefill accumulation, none of the ~14% of prefill wall
time v1 spent clustering. v1's redirect channel, its ``FFRowStash``, and the whole
resolve/hold/expire path on the consumer are simply unused here — v2 never emits a redirect that
might not resolve.

**Its own channel, always.** REQ/REP on our own port with our own tag, with the roles inverted from
the layerwise version: here **D is the client and P the server**, which is the direction this
transport already runs in (D knows P's address from ``kv_transfer_params``; P never learns D's,
which is why v1's redirects had to ride the params dict instead). Never a rider on the vendored
``KVCacheSendingThread`` ROUTER — v1's module docstring records that coupling to the base's control
plane broke the decode node outright when a vendored thread body grew a parameter.

**Best-effort by construction.** No reply, a timeout, a dead peer ⇒ no signatures ⇒ ``plan()``
returns its input unchanged ⇒ the request is read in full, exactly as v1 would. A slow producer costs
compression, never a stall.

Everything above the ``_ASCEND_AVAILABLE`` gate is pure and imports on any box, so the filtering and
codec logic stay unit-testable without an NPU. v1 is untouched; this registers as
``MooncakeConnectorFFv2`` beside it.
"""

import os
import threading
import time
from typing import Any, ClassVar

from vllm.logger import init_logger

logger = init_logger("vllm.mooncake_connector_ff_v2_ascend")

CONNECTOR_NAME = "MooncakeConnectorFFv2"

# Port for this transport's signature exchange. Offset from the base connector's side channel, and
# distinct from layerwise v2's 21000 so both connectors can be loaded in one process without
# colliding.
FF_PULL_V2_PORT_OFFSET = int(os.environ.get("BFF_MOONCAKE_FF_PULL_V2_PORT_OFFSET", "22000"))
# Seconds D waits for P's signatures before reading the request whole.
#
# Its own knob, and a much larger default than the layerwise transport's 2s, because the two are not
# the same amount of work. There, D answers from a dict it already holds. Here, P has to gather the
# blocks on the NPU and sync them back to the host (signatures_for_group ends in a `.cpu()`) on a
# side thread, while that same device is saturated with prefill. The exchange runs on the recv
# thread, never the forward path, so a generous timeout costs one request's KV latency — whereas a
# tight one costs the whole optimisation, silently, by falling back to a full read every time.
# Deliberately NOT falling back to BFF_V2_SIG_TIMEOUT: the harness exports that unconditionally with
# the layerwise default of 2, so "use it if set" would mean "always 2" and this default would be
# dead code.
SIG_EXCHANGE_TIMEOUT = float(os.environ.get("BFF_PULL_V2_SIG_TIMEOUT", "10"))
# How long register_kv_caches waits for the producer's REP socket to bind before saying so.
SIG_SERVER_BIND_TIMEOUT = float(os.environ.get("BFF_PULL_V2_SIG_BIND_TIMEOUT", "5"))
# Full ACL graph is refused by default even though v2 does no forward-path work — see
# MooncakeConnectorFFv2.requires_piecewise_for_cudagraph for why the GPU v2 reasoning does not carry
# over, and what setting this to 1 actually tests.
ALLOW_FULL_GRAPH = os.environ.get("BFF_V2_ALLOW_FULL_GRAPH", "0") == "1"

# Message tags for our own channel.
MSG_SIG_REQUEST = b"bff_pull_v2_sig_req"
MSG_SIG_REPLY = b"bff_pull_v2_sig_rep"


# =================================================================================================
# pure helpers (no NPU, no vllm_ascend)
# =================================================================================================
def filter_sentinels(planned, *paired):
    """Drop the positions D declined, from EVERY list in the pairing.

    ``planned`` is the block list the engine planned over, with :data:`SENTINEL` where the decode
    declined; each list in ``paired`` is positionally paired with it by the transfer that follows.
    A declined block has to be removed from all of them at the same index. Dropping it from one list
    only — or filtering after ``group_concurrent_contiguous`` has coalesced runs — pairs every
    subsequent survivor with the wrong source block and reads the wrong KV into it, with no error
    anywhere. That is the whole reason the sentinel is a placeholder rather than a deletion.

    Two lists reach this function on the pull transport and they are in DIFFERENT block-id spaces:
    ``planned`` holds D's ids (that is what the engine records, so that is what it must be given)
    and its companion holds P's. They line up only positionally, which is exactly why the filter is
    written in terms of positions and never re-derives one list from the other.

    A companion may be SHORTER than ``planned``: a prefix-cache hit on D shortens its side from the
    front and ``align_per_group`` has already tail-aligned the pair, so indices correspond but the
    lengths need not. Kept index-safe rather than assuming equal length.

    Returns ``(planned_kept, *paired_kept)``, everything unchanged when nothing was declined — so a
    request the producer never answered for costs one ``any()`` scan."""
    if not planned or not any(b < 0 for b in planned):
        return (planned, *paired)
    kept = [i for i, b in enumerate(planned) if b >= 0]
    return ([planned[i] for i in kept],
            *([lst[i] for i in kept if i < len(lst)] for lst in paired))


def signature_request_and_plan_groups(aligned, groups=None):
    """The two block-id spaces this transport has to keep apart, from one aligned pair per group.

    Returns ``(ask, plan)``, both ``{group: [block ids]}``:

    * ``ask`` is keyed by the **producer's** ids, because P answers by indexing its own KV cache;
    * ``plan`` is keyed by the **decode's**, because everything ``DedupEngine`` records from the
      answer addresses D — the alias victim, the representative, the residency index, and the ids
      handed to ``get_block_ids_with_load_errors``, which vLLM reads as D's own blocks.

    Getting this backwards does not raise. It cost a whole run: the engine planned over P's ids, so
    ``AliasApplier`` could not find a single victim in D's block table (709 of 726 aliases went to
    recompute), the seventeen that matched by coincidence wrote a foreign block id into D's table,
    and eighteen declined blocks were reported as failures against unrelated D blocks — which turned
    into 31- and 35-request reschedule cascades.

    Safe only because ``align_per_group`` has already made the two lists equal-length and
    positionally paired, so P's signature rows line up with D's ids slot for slot. Groups with
    nothing to pull are dropped from both: asking about them wastes a payload and invites an
    empty-signature reply that looks like a failure.

    **Group 0 is never eligible.** It is the warmup group — ``layers[0:2] + layers[-2:]``, so on
    Qwen2.5-7B the first two and the last two layers — and every other BFF path refuses it by name:
    the GPU v2 sender, the Ascend layerwise v2 sender, and v1's own producer fusion. This connector
    was the only one without the guard, because it subclasses v1's *transport* while the group
    policy lived in v1's *fusion* path, which v2 deletes. The cost of the omission: 99.16% of group
    0 aliased in one run, 11,582 blocks collapsed onto 23 representatives, so nearly every request
    shared one of 23 physical blocks for its first and last two layers. The output stayed fluent —
    each substitution was individually tiny, mostly cosine >0.98 — but the model stopped answering
    the prompt and stopped emitting EOS, and 27.6% of requests ran to the token cap.

    ``groups`` restricts further, and is the ``BFF_FF_GROUPS`` selection (``None`` = every eligible
    group). Taken as a parameter rather than read from v1 so this stays pure and importable off the
    Ascend stack."""
    ask, plan = {}, {}
    for gi, (remote_ids, local_ids) in enumerate(aligned):
        if gi <= 0 or not remote_ids or not local_ids:
            continue
        if groups is not None and gi not in groups:
            continue
        ask[int(gi)] = [int(b) for b in remote_ids]
        plan[int(gi)] = [int(b) for b in local_ids]
    return ask, plan


def sig_request_msg(groups_to_ids: dict) -> tuple:
    """D → P: ``(tag, {group: [P block ids]})``."""
    return (MSG_SIG_REQUEST, {int(gi): [int(b) for b in ids]
                              for gi, ids in groups_to_ids.items() if ids})


def sig_reply_msg(payloads: dict) -> tuple:
    """P → D: ``(tag, {group: signature payload})``. An empty dict is a valid answer meaning
    "nothing to describe" and must lead to a full read, never to an error."""
    return (MSG_SIG_REPLY, {int(gi): p for gi, p in (payloads or {}).items() if p is not None})


def parse_sig_reply(msg) -> dict:
    """P's reply → ``{group: payload}``, or ``{}`` for anything unrecognisable.

    Deliberately total: every malformed-answer path has to degrade to a full read, because the
    alternative is refusing to serve a request over a compression optimisation."""
    if not msg or len(msg) < 2 or msg[0] != MSG_SIG_REPLY or not isinstance(msg[1], dict):
        return {}
    out = {}
    for gi, payload in msg[1].items():
        try:
            out[int(gi)] = payload
        except (TypeError, ValueError):
            continue
    return out


# =================================================================================================
# Ascend/NPU-only section
# =================================================================================================
try:
    import msgspec
    import zmq
    from vllm.config import VllmConfig  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: F401
    from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
    from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: F401
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
        group_concurrent_contiguous,
    )

    from kv_fast_fusion import pd_dedup_v2
    from kv_fast_fusion.pd_dedup_v2 import AliasApplier, DedupEngine, KVLayoutError
    from kv_fast_fusion_ascend.connectors import mooncake_connector_ff as v1
    from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff_v2 import (
        signatures_for_group,
    )

    _ASCEND_AVAILABLE = True
except Exception as _e:  # pragma: no cover - optional dependency
    logger.info("MooncakeConnectorFFv2: Ascend stack unavailable (%s); only the pure glue is "
                "importable.", _e)
    _ASCEND_AVAILABLE = False


if _ASCEND_AVAILABLE:

    class _SigServer(threading.Thread):
        """Producer side: answer "give me signatures for these blocks".

        A REP socket of our own — own port, own tag. Every failure path answers with an empty
        payload, which the decode reads as "pull it all", so a producer that cannot describe its
        blocks degrades to stock behaviour rather than failing a transfer."""

        def __init__(self, host: str, port: int, worker):
            super().__init__(daemon=True, name="BFF-pullv2-SigServer")
            self._host, self._port = host, port
            self._worker = worker
            self._dec = msgspec.msgpack.Decoder()
            self._enc = msgspec.msgpack.Encoder()
            # Set once the REP socket is actually bound. Without it a bind failure kills this thread
            # with nothing but an unraisable traceback, and the only visible symptom is a timeout on
            # the OTHER node thousands of requests later — which is exactly how the first on-box run
            # left "did P ever listen?" undecidable.
            self.ready = threading.Event()
            self.bind_error: Exception | None = None
            # The producer's own tally. Deliberately NOT a DedupStats dump: the decode is the only
            # side that decides here, and a second ledger in the same stats directory would
            # double-count every figure the collector sums. P's unique information is just whether
            # it could answer.
            self.served = 0
            self.failed = 0
            self.fail_reasons: dict[str, int] = {}
            self._next_report = 1

        def run(self):
            path = make_zmq_path("tcp", self._host, self._port)
            ctx = zmq.Context()
            try:
                sock = make_zmq_socket(ctx=ctx, path=path, socket_type=zmq.REP, bind=True)
            except Exception as e:
                # Named loudly rather than left to an unraisable traceback. The likeliest cause is a
                # port already taken — the offset puts us in Linux's ephemeral range (32768-60999),
                # so a collision with a transient socket is possible; BFF_MOONCAKE_FF_PULL_V2_PORT_
                # OFFSET moves us out of the way. The decode degrades to full reads either way.
                self.bind_error = e
                logger.error("BFF pull-v2: signature server could NOT bind %s (%s). This producer "
                             "cannot answer signature requests, so every decode that asks it will "
                             "time out and read in full. Move the port with "
                             "BFF_MOONCAKE_FF_PULL_V2_PORT_OFFSET.", path, e)
                ctx.destroy(linger=0)
                return
            logger.info("BFF pull-v2 signature server (REP) bound on %s", path)
            self.ready.set()
            try:
                while True:
                    try:
                        msg = self._dec.decode(sock.recv())
                        reply = self._handle(msg)
                    except Exception as e:  # pragma: no cover - never kill the listener
                        logger.warning("BFF pull-v2 signature server error: %s", e)
                        reply = sig_reply_msg({})
                    try:
                        sock.send(self._enc.encode(reply))
                    except Exception as e:  # pragma: no cover
                        logger.warning("BFF pull-v2 signature reply failed: %s", e)
            finally:
                ctx.destroy(linger=0)

        def _note_failure(self, reason: str) -> None:
            self.failed += 1
            self.fail_reasons[reason] = self.fail_reasons.get(reason, 0) + 1
            self._worker.note_sig_failure(reason)

        def _report(self) -> None:
            """Log the tally on a widening cadence: the first request (the one that says D reached
            us at all), then 10, 100, 1000..."""
            if self.served + self.failed < self._next_report:
                return
            self._next_report *= 10
            logger.info("BFF pull-v2 signature server: served %d request(s), %d failed%s.",
                        self.served, self.failed,
                        (" " + str(self.fail_reasons)) if self.fail_reasons else "")

        def _handle(self, msg):
            if not msg or msg[0] != MSG_SIG_REQUEST:
                return sig_reply_msg({})
            groups_to_ids = msg[1] or {}
            out = {}
            ok = False
            for gi, ids in groups_to_ids.items():
                try:
                    out[int(gi)] = self._worker.signatures_for_group(int(gi), ids)
                    ok = True
                except KVLayoutError as e:
                    # Counted, not swallowed: it means the cache cannot be indexed by connector
                    # block ids, which would make every signature in the run meaningless.
                    self._note_failure("kv_layout")
                    logger.warning("BFF pull-v2: cannot index KV for group %s (%s).", gi, e)
                except Exception as e:  # pragma: no cover - defensive
                    self._note_failure("sig_error")
                    logger.warning("BFF pull-v2: signature build failed for group %s: %s", gi, e)
            if ok:
                self.served += 1
            self._report()
            return sig_reply_msg(out)

    class _SigClient:
        """Decode side: one REQ socket per producer peer, used from the recv thread.

        Synchronous by nature — D must know the answer before it reads — but it runs on the transfer
        thread, so the round trip delays one request's KV rather than the model."""

        def __init__(self):
            self._lock = threading.Lock()
            self._ctx = None
            self._socks: dict[tuple, Any] = {}
            self._enc = msgspec.msgpack.Encoder()
            self._dec = msgspec.msgpack.Decoder()
            self._announced: set = set()

        def ask(self, host, port, groups_to_ids: dict) -> dict:
            """Return ``{group: signature payload}``, or ``{}`` on any failure."""
            if host is None or port is None or not groups_to_ids:
                return {}
            with self._lock:
                t0 = time.perf_counter()
                try:
                    sock = self._sock_for(host, int(port))
                    sock.send(self._enc.encode(sig_request_msg(groups_to_ids)))
                    return parse_sig_reply(self._dec.decode(sock.recv()))
                except Exception as e:
                    # A REQ socket that timed out is stuck in the wrong state; drop it so the next
                    # exchange starts clean, and read this request whole.
                    #
                    # The elapsed time is the diagnosis: at (or just over) SIG_EXCHANGE_TIMEOUT the
                    # producer never answered — either it is not listening or it is too slow, which
                    # its own log distinguishes. Well under it means the socket was refused outright,
                    # a different problem entirely.
                    logger.warning("BFF pull-v2: signature exchange with %s:%s failed after "
                                   "%.0f ms of a %.0f ms budget (%s) — reading the request in "
                                   "full.", host, port, (time.perf_counter() - t0) * 1e3,
                                   SIG_EXCHANGE_TIMEOUT * 1e3, e)
                    self._drop(host, port)
                    return {}

        def _sock_for(self, host, port):
            key = (host, port)
            s = self._socks.get(key)
            if s is None:
                if self._ctx is None:
                    self._ctx = zmq.Context()
                path = make_zmq_path("tcp", host, port)
                if key not in self._announced:
                    # Says out loud which address D derived, so a port-arithmetic drift between the
                    # two sides is a one-line comparison against the producer's bind log rather than
                    # an inference from a timeout.
                    self._announced.add(key)
                    logger.info("BFF pull-v2: signature peer for this producer is %s", path)
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

    class KVCacheRecvingThreadFFv2(v1.KVCacheRecvingThreadFF):
        """v1's group-aware pull, with the blocks D can satisfy locally never read at all.

        Injected after construction by the worker's ``register_kv_caches``, same as
        ``base_addr_groups``: neither is knowable until the caches are registered."""

        # NOT `engine`. The vendored KVCacheRecvingThread keeps the Mooncake TransferEngine under
        # that exact name (mooncake_connector.py:326) and v1's _transfer_kv_cache calls
        # `self.engine.batch_transfer_sync_read` — injecting the dedup engine as `engine` replaced
        # the transport with an object that cannot transfer, and every request on the node died with
        # "'DedupEngine' object has no attribute 'batch_transfer_sync_read'". Any attribute added
        # here has to be checked against that __init__; the test suite does it mechanically.
        sig_client: "_SigClient | None" = None
        dedup_engine: "DedupEngine | None" = None
        sig_port_offset: int = FF_PULL_V2_PORT_OFFSET
        _logged_first_decline = False
        # Groups already reported by _warn_on_runaway_groups. Deliberately class-level and shared:
        # the warning is once per group for the process, not once per request or per thread.
        _warned_runaway: ClassVar[set] = set()

        def _plan_aligned(self, req_meta, aligned):
            """Ask P for signatures of the blocks we are about to read, and decide.

            Planned on the ALIGNED lists, not the raw ones: ``align_per_group`` has already trimmed
            P's list to the tail that corresponds to what D actually allocated (a prefix-cache hit
            shortens D's side from the front), so planning here means the plan's slots and the
            transfer's slots are the same slots by construction — no re-keying, and no chance of an
            off-by-one between the two.

            Two block-id spaces meet here and are kept apart by
            :func:`signature_request_and_plan_groups`: P is asked about ITS blocks, the engine is
            given D's. See that function for what conflating them cost.

            Returns ``{group: planned_ids}`` — D's ids with SENTINEL in the declined positions — or
            ``{}`` on every failure path, which leaves the read whole."""
            if (self.dedup_engine is None or self.sig_client is None
                    or not pd_dedup_v2.V2_ENABLED):
                return {}
            ask, plan_groups = signature_request_and_plan_groups(aligned, v1._FF_GROUPS)
            if not ask:
                return {}
            # EXTERNAL id, not `remote_request_id` itself: that is P's local request id, while the
            # applier walks D's runner, whose requests carry D's local ids. vLLM appends a
            # per-EngineCore suffix, so the two are different strings for one request and only the
            # stripped form is common to both. Keying the engine on P's id would mean no alias ever
            # resolved, with no error anywhere.
            ext_id = v1._ext_of(req_meta["remote_request_id"])
            # Counted BEFORE the ask, so it counts attempts. DedupStats.is_inert() reads
            # `exchanges == 0` as "v2 is installed but never ran", which is the difference between
            # "nothing was worth merging" and "the mechanism never engaged" — the two readings of an
            # all-zero saving that a benchmark cannot otherwise tell apart.
            self.dedup_engine.stats.exchanges += 1
            sigs = self.sig_client.ask(
                req_meta["remote_host"],
                int(req_meta["remote_handshake_port"]) + self.sig_port_offset,
                ask)                                  # P's ids — P indexes its own KV cache
            if not sigs:
                self.dedup_engine.stats.sig_phase_failed += 1
                return {}
            # D's ids — row i of P's signature payload describes plan slot i, because
            # align_per_group made the two lists equal-length and positionally paired.
            n_groups = max(plan_groups) + 1
            wrapped = [plan_groups.get(gi, []) for gi in range(n_groups)]
            try:
                planned = self.dedup_engine.plan({ext_id: wrapped}, {ext_id: sigs})
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF pull-v2: planning failed (%s) — reading in full.", e)
                self.dedup_engine.forget([ext_id])
                return {}
            out = planned.get(ext_id)
            if not out:
                return {}
            result = {gi: out[gi] for gi in plan_groups if gi < len(out)}
            self._warn_on_runaway_groups(result)
            return result

        def _warn_on_runaway_groups(self, planned) -> None:
            """Say so when a group is being deduped almost entirely.

            A group whose blocks are nearly all interchangeable is not a compression win — it is a
            group that should not be deduped at all, because whatever distinguishes its blocks is
            below the signature's resolution. The warmup group said exactly this at 99.16% for a
            whole run and the number sat unread in the stats file while the benchmark reported a
            flattering 19.7% overall saving. Cheap enough to leave on: one ratio per group."""
            for gi, plan_g in planned.items():
                if gi in KVCacheRecvingThreadFFv2._warned_runaway or not plan_g:
                    continue
                declined = sum(1 for b in plan_g if b < 0)
                if declined < 0.9 * len(plan_g):
                    continue
                KVCacheRecvingThreadFFv2._warned_runaway.add(gi)
                logger.warning(
                    "BFF pull-v2: group %d declined %d of %d blocks (%.0f%%) in one request. A "
                    "group this uniform is not compressible, it is under-resolved by the signature "
                    "— check that it is a fusion group and consider excluding it via BFF_FF_GROUPS.",
                    gi, declined, len(plan_g), 100.0 * declined / len(plan_g))

        def _align_and_group(self, req_meta, local_groups, remote_groups, tp_num_need_pulls):
            """v1's tail-align + coalesce, with the declined positions removed in between.

            The order is the whole correctness argument: filter AFTER ``align_per_group`` (so the
            pair is tail-aligned and indices correspond) and BEFORE ``group_concurrent_contiguous``
            (so a declined block breaks a contiguous run rather than being absorbed into one and
            dragged across the wire anyway)."""
            aligned = v1.align_per_group(local_groups, remote_groups)
            planned = self._plan_aligned(req_meta, aligned)
            grouped = []
            n_declined = 0
            for gi, (remote_ids, local_ids) in enumerate(aligned):
                plan_g = planned.get(gi)
                if plan_g is not None and len(plan_g) == len(local_ids):
                    before = len(local_ids)
                    # The planned list is D's, so it filters the LOCAL side directly and the remote
                    # side by position. Reading the remote ids back out of `plan_g` instead would
                    # feed D's block ids to the transfer as source addresses.
                    local_ids, remote_ids = filter_sentinels(list(plan_g), remote_ids)
                    n_declined += before - len(local_ids)
                if not local_ids:
                    grouped.append(([], []))
                elif tp_num_need_pulls == 1:
                    grouped.append(group_concurrent_contiguous(remote_ids, local_ids))
                else:
                    grouped.append(([[b] for b in remote_ids], [[b] for b in local_ids]))
            if n_declined:
                if not KVCacheRecvingThreadFFv2._logged_first_decline:
                    KVCacheRecvingThreadFFv2._logged_first_decline = True
                    logger.info("BFF pull-v2: first declined read — %d block(s) satisfied locally "
                                "and never fetched.", n_declined)
                if self.dedup_engine is not None:
                    self.dedup_engine.stats.note_skip("declined", n_declined)
            return grouped

        def _after_transfer(self, req_meta) -> None:
            """The "KV has landed" signal.

            Only here do two things become true: this request's aliases may be applied, and the
            blocks it did read may serve as representatives for later requests. Releasing any
            earlier is the bug that made the first GPU v2 run apply 22 of 26,531 aliases — the apply
            path expires a map whose owner has not been batched, and an owner cannot be batched
            until its KV has actually arrived."""
            if self.dedup_engine is not None:
                self.dedup_engine.release(v1._ext_of(req_meta["remote_request_id"]))

    class MooncakeConnectorWorkerFFv2(v1.MooncakeConnectorWorkerFF):
        """v1's worker plus the signature server (on P) and the dedup engine (on D)."""

        _RECV_THREAD_CLS = KVCacheRecvingThreadFFv2

        def __init__(self, vllm_config, engine_id, kv_cache_config=None):
            # `_dedup_engine`, never `_engine`: the vendored worker's `self.engine` is the Mooncake
            # TransferEngine, and keeping the two one underscore apart is how the recv thread's
            # collision got written in the first place.
            self._dedup_engine = None
            self._sig_client = None
            self._sig_server = None
            self._jl = [None]        # JL projection cache, must outlive the calls
            self._proj = [None]      # SimHash projection cache — same reason
            self._ff_failed_blocks: set = set()
            self._group_layers: dict[int, set] = {}
            super().__init__(vllm_config, engine_id, kv_cache_config)

        def register_kv_caches(self, kv_caches):
            super().register_kv_caches(kv_caches)
            # Inverse of v1's layer->group map; AliasApplier and the signature builder both want
            # group->layers. Filled IN PLACE, never reassigned: AliasApplier holds this dict by
            # reference and is built lazily, so swapping in a new object here would leave the
            # applier looking at an empty one and silently disable every scale it can place.
            self._group_layers.clear()
            for ln, gi in self._layer_group.items():
                self._group_layers.setdefault(int(gi), set()).add(ln)

            host = self.side_channel_host
            port = self.side_channel_port + FF_PULL_V2_PORT_OFFSET + self.tp_rank
            if self.kv_role == "kv_producer":
                self._sig_server = _SigServer(host, port, self)
                self._sig_server.start()
                self._await_sig_server()
                self._warm_signatures()
            else:
                self._dedup_engine = DedupEngine()
                self._sig_client = _SigClient()
                if self.kv_recv_thread is not None:
                    self.kv_recv_thread.dedup_engine = self._dedup_engine
                    self.kv_recv_thread.sig_client = self._sig_client
                logger.info("BFF pull-v2: decode dedup engine armed (V2_DEDUP=%s, sig timeout "
                            "%.1fs).", pd_dedup_v2.V2_ENABLED, SIG_EXCHANGE_TIMEOUT)

        def _await_sig_server(self) -> None:
            """Fail loudly at startup instead of silently at request time.

            The server binds on its own thread, so without this a bind failure surfaces only as a
            decode-side timeout on some later node — which is undecidable from that node's log."""
            if self._sig_server.ready.wait(SIG_SERVER_BIND_TIMEOUT):
                return
            logger.error("BFF pull-v2: signature server did not come up within %.0fs (%s). This "
                         "producer will not answer signature requests; decodes reading from it fall "
                         "back to full reads and v2 buys nothing here.",
                         SIG_SERVER_BIND_TIMEOUT, self._sig_server.bind_error or "no bind error "
                         "reported — still starting?")

        def _warm_signatures(self) -> None:
            """Build the projections once, now, rather than inside the decode's first timeout.

            signatures_for_group ends in an NPU->CPU sync and, on its first call, also builds the
            fixed-seed JL and SimHash projections. Paying that on the first real request means paying
            it on a side thread of a node that is by then busy with prefill, inside a bounded window
            — a slow first answer that costs compression for no reason. The result is discarded; only
            the cached projections in _jl/_proj are wanted."""
            gi = next((g for g in sorted(self._group_layers) if self._group_layers[g]), None)
            if gi is None or not getattr(self, "num_blocks", 0):
                return
            try:
                _t0 = time.perf_counter()
                self.signatures_for_group(gi, [0])
                logger.info("BFF pull-v2: signature path warmed on group %d in %.0f ms.",
                            gi, (time.perf_counter() - _t0) * 1e3)
            except Exception as e:
                # Never fatal: a producer that cannot warm up can still try per request, and if it
                # cannot do that either the decode degrades to full reads.
                logger.warning("BFF pull-v2: signature warm-up failed (%s); the first real exchange "
                               "will pay for it instead.", e)

        # -- producer side ----------------------------------------------------------------
        def signatures_for_group(self, gi: int, block_ids):
            """Signature payload for one group's blocks, computed on demand from the registered KV.

            No forward-path hook is involved: this reads whatever is in the cache when D asks, which
            is why v2 needs none of v1's save_kv_layer accumulation."""
            layer_names = sorted(self._group_layers.get(int(gi), ()))
            if not layer_names or not block_ids:
                return None
            is_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
            return signatures_for_group(
                self.kv_caches, layer_names, [int(b) for b in block_ids], is_mla,
                self._jl, num_blocks=self.num_blocks, proj_holder=self._proj)

        def note_sig_failure(self, reason: str) -> None:
            # On the producer `_dedup_engine` is always None (only the decode decides), so this is a
            # no-op there by design — _SigServer keeps and logs its own tally instead. Kept for the
            # symmetric case and for tests that drive the server against a decode-side worker.
            eng = self._dedup_engine
            if eng is not None:
                eng.stats.note_failure(reason)

        # -- consumer side ----------------------------------------------------------------
        def note_failed_blocks(self, block_ids) -> None:
            """Blocks D declined that were then never read and could not be aliased.

            Routed into vLLM's KV-load-failure path so the owning request recomputes locally —
            slower, never wrong."""
            self._ff_failed_blocks |= {int(b) for b in block_ids}

        def take_failed_blocks(self) -> set:
            out, self._ff_failed_blocks = self._ff_failed_blocks, set()
            return out

    class MooncakeConnectorFFv2(v1.MooncakeConnectorFF):
        """The Ascend pull connector where the DECODE decides which blocks are worth reading.

        v1's producer fusion engine, its redirect wire format and the whole resolve/hold/expire path
        on the consumer are simply not used: v2 never emits a redirect that might not resolve, so
        there is nothing to resolve, hold, or expire."""

        _WORKER_CLS = MooncakeConnectorWorkerFFv2

        def __init__(self, vllm_config, role, kv_cache_config=None):
            super().__init__(vllm_config, role, kv_cache_config)
            # v2 does no producer forward-path work at all. Dropping v1's engine here is what
            # removes the ~14% of prefill wall time it spent clustering, and it must be dropped
            # rather than left idle: a live producer engine would keep filling the row stash that
            # nothing in v2 ever drains.
            self._ff_producer = None
            self._ff_applier = None
            self._ff_step = 0

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
            """No-op: signatures are computed on demand from the registered KV cache."""

        @classmethod
        def requires_piecewise_for_cudagraph(cls, extra_config: dict) -> bool:
            """Still True by default, for a DIFFERENT reason than v1's.

            v1 demanded PIECEWISE because it ran real Python per layer inside ``save_kv_layer``; the
            GPU v2 therefore returns False, since v2 does no forward-path work. That reasoning does
            not carry to Ascend: the full-graph corruption measured here (garbage from the first
            decoded token, clean at one KV-cache group, broken at seven) was tied to the seven block
            tables in ``AscendAttentionBackendImpl.update_graph_params``, which re-reads only
            ``seq_lens`` per replay and takes ``block_table`` from the tuple frozen at capture. That
            has nothing to do with save_kv_layer and v2 does not fix it.

            Set ``BFF_V2_ALLOW_FULL_GRAPH=1`` to test exactly that: v2 removes one of the two
            candidate explanations, so a clean full-graph run under v2 would prove the corruption was
            save_kv_layer, and a corrupt one would prove it is the block tables. Off by default
            because the failure mode is silent."""
            return not ALLOW_FULL_GRAPH

        def _applier(self) -> "AliasApplier":
            a = getattr(self, "_ff_applier", None)
            if a is None:
                worker = self.connector_worker
                a = self._ff_applier = AliasApplier(
                    worker._dedup_engine, _write_block_table, worker.note_failed_blocks,
                    normalize_req_id=v1._ext_of,
                    group_layers=worker._group_layers)
            return a

        def start_load_kv(self, forward_context, **kwargs) -> None:
            # Connector-level signature (forward_context), NOT the worker's start_load_kv(metadata).
            super().start_load_kv(forward_context, **kwargs)
            self._ff_step += 1
            if self.connector_worker is None or self.connector_worker._dedup_engine is None:
                return
            self._v2_apply()
            # Dumped from here, on the engine's own cadence. Without it this arm produces no
            # bff_stats_*.json at all and the whole verification ladder — wire saving, applied vs
            # recomputed, inert-or-not — is unreadable, which is exactly how an earlier run came back
            # as "bff stats: none found (fusion may not have engaged)".
            stats = self.connector_worker._dedup_engine.stats
            if stats.should_dump(self._ff_step):
                stats.dump()

        def _v2_apply(self) -> None:
            """Apply landed aliases; see :class:`~kv_fast_fusion.pd_dedup_v2.AliasApplier`."""
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                applier = self._applier()
                before = applier._engine.stats.applied
                applier.apply(getattr(_bp, "_ACTIVE_RUNNER", None))
                n = applier._engine.stats.applied - before
                if n:
                    logger.info("BFF pull-v2 apply | aliases_applied=%d | recompute(cum)=%d",
                                n, applier._engine.stats.recomputed)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF pull-v2 consumer apply failed: %s", e)

        def get_block_ids_with_load_errors(self) -> set:
            worker = self.connector_worker
            if worker is None:
                return set()
            return worker.take_failed_blocks()

    def _write_block_table(runner, rid, gi, new_blocks) -> bool:
        from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
            _ff_write_runner_block_table,
        )
        return _ff_write_runner_block_table(runner, rid, gi, new_blocks)

    def register_mooncake_connector_ff_v2() -> None:
        """Register ``MooncakeConnectorFFv2`` (idempotent)."""
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        if CONNECTOR_NAME in KVConnectorFactory._registry:
            return
        KVConnectorFactory.register_connector(
            CONNECTOR_NAME,
            "kv_fast_fusion_ascend.connectors.mooncake_connector_ff_v2",
            "MooncakeConnectorFFv2",
        )
        logger.info("BFF Ascend: registered %s (pull transport, decode-side dedup).",
                    CONNECTOR_NAME)

else:  # pragma: no cover - exercised only off the Ascend stack

    def register_mooncake_connector_ff_v2() -> None:
        logger.warning("MooncakeConnectorFFv2 not registered: the Ascend stack is unavailable.")
