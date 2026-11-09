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
import queue
import threading
import time
from typing import Any, ClassVar

from vllm.logger import init_logger

from kv_fast_fusion_ascend import landed_promotion
from kv_fast_fusion_ascend.build import BFF_BUILD
from kv_fast_fusion_ascend.diagnostics import (  # noqa: F401 - names re-exported for the tests
    AUDIT_HOT_BLOCKS,
    TRACE_SLOTS,
    VERIFY_DEGRADED_COS,
    VERIFY_MAX_NORM_ERR,
    VERIFY_MIN_COS,
    VERIFY_SPACING_S,
    VERIFY_TRANSFER,
    XFER_AUDIT,
    TraceDiagnostics,
    TransferDiagnostics,
    VerifySampler,
    _sync_device,
    best_matching_row,
    classify_mismatch,
    compare_signature_rows,
    cross_match,
    hot_block_collisions,
    slot_trace_fault,
    write_slot,
)

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
# Full ACL graph is refused, and this escape hatch is now known to produce GARBAGE — see
# MooncakeConnectorFFv2.requires_piecewise_for_cudagraph. Kept only so the experiment stays
# reproducible; it is not a tuning knob.
ALLOW_FULL_GRAPH = os.environ.get("BFF_V2_ALLOW_FULL_GRAPH", "0") == "1"
# Producer-side signature precompute (Update 9). When set, the producer computes each request's block
# signatures at SAVE time (off the decode's critical path — P has slack, D is the bottleneck) and the
# _SigServer answers exchanges from that cache, so the decode's exchange stops paying for the
# producer's on-demand NPU drains (~79 s / 16x the actual transfer at con512). A cache MISS falls back
# to the existing on-demand compute, so the result is never wrong, only slower. Default off.
PRODUCER_SIGCACHE = os.environ.get("BFF_V2_PRODUCER_SIGCACHE", "0") == "1"

# Bridge for PRODUCER_SIGCACHE: the scheduler-side `request_finished_all_groups` (which has the
# per-group block ids) enqueues a precompute job here, and the producer worker's drain thread (which
# has `kv_caches`) consumes it. At TP=1 scheduler and worker share the process, so this in-process
# queue joins the two halves without any wire/protocol change; at TP>1 the worker is a separate
# process and the queue simply stays empty there, so the decode falls back to on-demand (no win, no
# harm). Bounded so a stall can never grow it without limit — a dropped job is just an on-demand miss.
_PRODUCER_SIG_JOBS: "queue.Queue" = queue.Queue(maxsize=4096)
# How many finished requests to sign per forward step in wait_for_save. Bounds the compute one step
# adds so it stays inside the transport's window. For the SIGCACHE queue (drain_precompute) a backlog
# drains over later steps; for the PIGGYBACK (precompute_and_offer) there is no later step — the
# staple happens in this one — so the cap is a hard coverage limit. Raised 8 -> 32 in Update 27: with
# max_num_batched_tokens=65536 the producer completes 24-27 prefills in a single step, so 8 dropped
# 37 % of them (piggyback_misses 190/512), and each drop cost the decode a ~374 ms on-demand round trip
# ahead of its transfer — far more than the signing it avoided, since that signing then happens anyway.
SIG_PRECOMPUTE_PER_STEP = int(os.environ.get("BFF_V2_SIG_PRECOMPUTE_PER_STEP", "32"))
# Upper bound on the D-side piggyback stash (remote_request_id -> {gi: payload}). Entries are popped
# when the recv thread plans the request; the cap only backstops the rare un-consumed ones (a request
# whose batch was gate-closed before the pop). Generous vs the ~512 concurrency. See Update 13.
_PIGGYBACK_STASH_CAP = int(os.environ.get("BFF_V2_PIGGYBACK_STASH_CAP", "8192"))
# Usage-gated dedup (Update 11): run the signature exchange + aliasing ONLY when the decode's KV
# cache usage is at least this fraction — i.e. only when dedup's memory relief is worth its cost. The
# pull-heavy ramp (low usage) then runs dedup-free at baseline speed, and dedup self-throttles (its
# frees drop usage back below the gate). 0.0 disables the gate (dedup always on, the prior behaviour);
# 0.97 is a sensible "only under real pressure". Usage unknown (None) is treated as below the gate.
DEDUP_KV_THRESHOLD = float(os.environ.get("BFF_V2_DEDUP_KV_THRESHOLD", "0.0"))
# Ceiling on how much of ONE request may be satisfied from other requests' blocks. Above it the
# request is read whole. See cap_request_decline for why a per-block cosine bar cannot express this.
MAX_REQ_DECLINE = float(os.environ.get("BFF_V2_MAX_REQ_DECLINE", "0.5"))
# Clip instead of drop. A request over MAX_REQ_DECLINE is read in full today, and on GLM Magicoder those
# 10-12 near-duplicate requests hold ~70% of all declinable blocks. With BFF_V2_CLIP_FRAC > 0 such a
# request keeps its LOWEST-error aliases, up to this fraction of its blocks, and fetches the rest —
# the blocks that differ most are the ones that make it its own question. 0 (default) = off: the
# request is read in full exactly as before. See clip_request_decline.
CLIP_FRAC = float(os.environ.get("BFF_V2_CLIP_FRAC", "0"))
# Per-block bar for the aliases a clip keeps, on top of BFF_MAX_REL_ERR. 1.0 = no extra bar.
CLIP_MAX_REL_ERR = float(os.environ.get("BFF_V2_CLIP_MAX_REL_ERR", "1.0"))
# A clip never aliases a request's last N blocks: the question usually sits at the end.
CLIP_TAIL = int(os.environ.get("BFF_V2_CLIP_TAIL", "2"))
# BFF_STEP_PROFILE=1 also times the clip and the landing copy (stats: clip_ms_total,
# landing_copy_ms_total). Off, nothing is timed and the stats report None.
_PROFILE_OVERHEAD = os.environ.get("BFF_STEP_PROFILE", "0") == "1"
# Under BFF_V2_MATERIALIZE, copy each alias at LANDING on the recv thread, and re-read a refused one
# from the producer, instead of applying on the forward path and recomputing a refusal. A refusal on
# the forward path re-prefills the whole prompt on D (KV-load failure) — on the GLM con48 runs each
# one stalled the batch for 25-50 s and four of them cost the run ~100 s. 0 restores the old path.
MATERIALIZE_ON_LANDING = os.environ.get("BFF_V2_MATERIALIZE_ON_LANDING", "1") == "1"

# Message tags for our own channel.
MSG_SIG_REQUEST = b"bff_pull_v2_sig_req"
MSG_SIG_REPLY = b"bff_pull_v2_sig_rep"
# Batched form: one message carries several requests, mirroring the GPU connector's signature phase
# (which answers for every pending send in one reply).
#
# DISTINCT TAGS, not a reshaped payload, because the two shapes are indistinguishable once decoded:
# single is ``{group: payload}`` and batched is ``{slot: {group: payload}}``, both int-keyed dicts.
# A producer one version behind would answer a batched question in the single shape and the decode
# would read group indices as batch slots — silently handing request 0's signatures to whichever
# request sat in slot 0. With its own tag an older producer simply does not recognise it, replies
# empty, and every request is read in full.
MSG_SIG_REQUEST_BATCH = b"bff_pull_v2_sig_req_batch"
MSG_SIG_REPLY_BATCH = b"bff_pull_v2_sig_rep_batch"
# Most requests an exchange may carry. The batch is not made of concurrent callers — the vendored
# recv thread is strictly serial (see KVCacheRecvingThreadFFv2.run) — it is whatever has piled up in
# `request_queue`.
#
# 8, NOT the 32 this shipped with, because the exchange head-of-line blocks. The recv thread starts
# no transfer at all while the producer works, which the one-at-a-time path never did: a 32-request
# batch measured up to ~5 s of that. 8 keeps most of the saving at a quarter of the stall. The cap
# also bounds one exchange's device work on the producer and keeps its batch-scaled timeout
# meaningful.
# Run the signature exchange on its own thread instead of in front of the transfer.
#
# Measured cost of NOT doing this, per request on the serial recv thread: 132 ms of ZMQ round trip
# plus 63 ms of planning, which caps the transport at ~7.6 req/s against the ~100/s the transfers
# alone sustain — and produces a ~40 s ramp at `Running=1, Waiting=199`. dedup=1 is the correct
# configuration (F1 0.4707/0.4798 against the baseline's 0.4803) and this is what it costs.
ASYNC_SIG = os.environ.get("BFF_PULL_V2_ASYNC_SIG", "0") == "1"
# How long a request may wait for its signatures before it is read in full. This, not the threading,
# is what removes the ceiling: under a burst dedup degrades to a plain read instead of queueing.
SIG_DEADLINE_MS = float(os.environ.get("BFF_PULL_V2_SIG_DEADLINE_MS", "200"))
# Off the critical path a bigger batch has no latency cost and a real throughput one: 199 queued
# requests at 8 per round trip is 25 trips (~9.5 s), at 64 it is 4 (~1.5 s). It also amortises the
# producer's single NPU drain per exchange, which is what its 380 ms actually is.
MAX_SIG_BATCH = int(os.environ.get("BFF_PULL_V2_SIG_BATCH", "64" if ASYNC_SIG else "8"))
# Where an async exchange leaves this request's own answer. On the request, not in a cache shared by
# everything in flight: with the exchange a batch ahead of the transfer, a shared cache is how one
# request is handed another's signatures — the same mis-attribution the producer/decode block-id
# spaces already cost this connector once.
SIG_SLOT = "_bff_pull_v2_sig"
# Move `plan` onto the exchange thread too, not just the exchange. Requires ASYNC_SIG. Even with the
# exchange off Thread B, plan (~63 ms/request, and it HOLDS the GIL) is left serial on the transfer
# thread — a ~14 req/s ceiling. On its own sub-flag because it is the change that makes the otherwise
# lock-free dedup engine concurrent: it can be measured, and rolled back, independently of the
# exchange move.
ASYNC_PLAN = ASYNC_SIG and os.environ.get("BFF_PULL_V2_ASYNC_PLAN", "0") == "1"
# Where the exchange thread leaves this request's plan, same reasoning as SIG_SLOT: on the request,
# so plan slot i can never be paired against another request's blocks.
PLAN_SLOT = "_bff_pull_v2_plan"
# Milliseconds to wait for a SECOND request when the queue drains empty. Zero, and measurement says
# keep it there: batching is throughput-neutral (see _SigClient), so paying latency to manufacture a
# batch buys nothing, and the request kept waiting is holding allocated KV blocks — the scarce
# resource here. A knob so the question stays answerable without a rebuild, not a recommendation.
SIG_LINGER_MS = float(os.environ.get("BFF_PULL_V2_SIG_LINGER_MS", "0"))


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


def declined_pairs(planned, local_ids, remote_ids) -> dict:
    """``{local_id: remote_id}`` for every position D declined — the blocks that are NOT read.

    Same positional pairing as :func:`filter_sentinels`, which drops these; kept here so a refused
    alias can still be re-read from the producer. ``planned`` carries a sentinel where ``local_ids``
    carries the id, so the id is read from ``local_ids`` and never from ``planned``."""
    return {int(local_ids[i]): int(remote_ids[i]) for i, b in enumerate(planned)
            if b < 0 and i < len(local_ids) and i < len(remote_ids)}


def repull_segments(descriptors, victims):
    """``(srcs, dsts, lens, missing)`` re-reading ``victims`` (``[(gi, local_id)]``).

    ``descriptors`` is the emission loop's ``{(gi, local_id): [(src, dst, len), ...]}`` for the
    declined blocks — every address of the block, so a re-read restores all of its layers. A victim
    with no entry is ``missing`` and cannot be re-read."""
    descriptors = descriptors or {}
    srcs, dsts, lens, missing = [], [], [], []
    for key in victims:
        segs = descriptors.get((int(key[0]), int(key[1])))
        if not segs:
            missing.append(key)
            continue
        for s, d, n in segs:
            srcs.append(s)
            dsts.append(d)
            lens.append(n)
    return srcs, dsts, lens, missing


def decline_fraction(planned):
    """``(declined, total)`` over one request's whole plan — every group, not one at a time.

    The harm is per REQUEST: a prompt whose blocks are individually replaceable is still a prompt
    the model never sees if enough of them are replaced at once."""
    declined = total = 0
    for plan_g in planned.values():
        total += len(plan_g)
        declined += sum(1 for b in plan_g if b < 0)
    return declined, total


def cap_request_decline(planned, max_frac=None):
    """Refuse a plan that replaces most of a request, keeping the rest untouched.

    Returns ``(planned, capped)``. Over the ceiling the plan is dropped whole and the request is
    read normally; the caller must also ``forget()`` its staged aliases so none are applied later.

    A per-block cosine bar cannot express this, because the bar is per block while the harm is per
    request. At ``BFF_MAX_REL_ERR=0.3`` the floor is cosine 0.954, which is a reasonable bar for one
    block and a catastrophic one for nineteen of twenty: the model then attends to a coherent prompt
    that is not the one it was asked. A run measured exactly that — 19 of 20 blocks declined in all
    six fusion groups of one request — with the local prefix cache at 0.5%, so the blocks were
    similar, not identical. Fetching one request in full is cheap; answering the wrong prompt is
    not."""
    frac = MAX_REQ_DECLINE if max_frac is None else max_frac
    if not planned or frac >= 1.0:
        return planned, False
    declined, total = decline_fraction(planned)
    if total <= 0 or declined <= frac * total:
        return planned, False
    return {}, True


def clip_request_decline(planned, plan_ids, errs, frac, max_err=1.0, tail=0):
    """Keep a capped request's best aliases instead of none. Returns ``(planned, keep)``.

    ``planned`` is ``{group: ids}`` with a negative sentinel at declined positions, ``plan_ids`` the
    same lists before planning (the decode's own block ids), ``errs`` ``{group: {victim: rel_err}}``
    as :meth:`DedupEngine.alias_errors` reports it. Up to ``int(frac * total blocks)`` declined
    positions survive, lowest ``rel_err`` first; a position with unknown error, an error over
    ``max_err``, or within the group's last ``tail`` blocks is always fetched. Every other declined
    position gets its own id back, so it is read like any undeclined block.

    ``keep`` is ``{group: {victim}}`` — what the engine must retain (:meth:`DedupEngine.retain`).
    Empty ``keep`` means nothing qualified: the caller reads the request in full, as the cap would.
    Pure, so the selection is testable off-NPU."""
    total = sum(len(ids) for ids in planned.values())
    budget = int(frac * total)
    cands = []
    if budget > 0:
        for gi, ids in planned.items():
            own = plan_ids.get(gi) or []
            for slot, b in enumerate(ids):
                if b >= 0 or slot >= len(own) or slot >= len(ids) - tail:
                    continue
                victim = int(own[slot])
                e = (errs.get(gi) or {}).get(victim)
                if e is None or e > max_err:
                    continue
                cands.append((e, gi, slot, victim))
    cands.sort()
    keep: dict = {}
    for _e, gi, _slot, victim in cands[:budget]:
        keep.setdefault(gi, set()).add(victim)
    out = {}
    for gi, ids in planned.items():
        own = plan_ids.get(gi) or []
        mine = keep.get(gi, ())
        out[gi] = [b if b >= 0 or slot >= len(own) or int(own[slot]) in mine
                   else int(own[slot]) for slot, b in enumerate(ids)]
    return out, keep


def select_rank_sigs(staged: dict, tp_rank: int) -> dict:
    """``{req: sigs}`` as this decode rank should see it.

    At TP>1 a request's piggybacked signatures arrive as ``{"ranks": {rank: {gi: payload}}}`` (each
    producer rank signs its own head shard); keep this rank's set, and drop the request when its rank
    is missing so that rank falls back to the on-demand exchange. TP=1 entries pass through."""
    out = {}
    for rid, sigs in (staged or {}).items():
        if isinstance(sigs, dict) and "ranks" in sigs:
            mine = (sigs.get("ranks") or {}).get(tp_rank)
            if mine:
                out[rid] = mine
        elif sigs:
            out[rid] = sigs
    return out


def piggyback_sign_here(sig_piggyback: bool, is_producer: bool, dedup_enabled: bool) -> bool:
    """Sign this step's finished prefills only where the result is used: on the producer, and only
    when the decode will dedup with it.

    ``wait_for_save`` runs on every worker, and on the DECODE every newly admitted request looks like
    a prefill that finished this step. Ungated, the decode signed each arriving request — a runner
    cProfile measured 61 calls per 2,000 steps at ~35 ms each (a host-to-device copy queued behind the
    in-flight forward), 1.1 ms/step, for payloads nothing on the decode ever reads."""
    return bool(sig_piggyback and is_producer and dedup_enabled)


def claim_piggyback_sign(holder, meta) -> bool:
    """True the first time ``holder`` asks for this step's connector ``meta``, False after.

    Signing is reached from ``wait_for_save`` and, at TP>1, from ``get_kv_connector_stats`` — in
    either order, because with a speculative config vllm-ascend defers finalize and the stats are
    taken first. Holds the object, not ``id(meta)``: a freed metadata's id can be reused by the next
    step's, which would silently skip that step."""
    if meta is None or getattr(holder, "_ff_signed_meta", None) is meta:
        return False
    holder._ff_signed_meta = meta
    return True


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


def drain_queue(get_nowait, first, max_items=None, linger_ms=None):
    """``first`` plus everything already queued behind it, up to ``max_items`` in total.

    This is what actually creates a batch on this transport. The vendored recv thread handles one
    request at a time, so no two signature exchanges are ever in flight together and a coalescer
    built for concurrent callers batched exactly one request, 512 times. Taking the queue in drained
    runs instead makes the batch size equal to the backlog.

    What that is worth is settled and modest: see :class:`_SigClient` — it cuts the producer's
    per-request cost ~30% and leaves end-to-end throughput unchanged. Draining costs nothing, so it
    stays; it is not a lever to pull on when a run is slow.

    ``get_nowait`` is injected (rather than the queue itself) so the drain is testable with a list,
    and ``queue.Empty`` — not ``Exception`` — ends the drain: anything else coming out of the queue
    is a real fault and must not be silently read as "nothing left".

    ``linger_ms`` waits that long for a second item when the queue is immediately empty. Off by
    default; see :data:`SIG_LINGER_MS`."""
    limit = max(1, int(MAX_SIG_BATCH if max_items is None else max_items))
    wait = (SIG_LINGER_MS if linger_ms is None else linger_ms) / 1000.0
    batch = [first]
    deadline = None
    while len(batch) < limit:
        try:
            batch.append(get_nowait())
            continue
        except queue.Empty:
            pass
        # Nothing queued. Lingering can only ever help the very first item: once anything has been
        # drained we are working a backlog rather than waiting for one, so the wait must not restart
        # per item — that would make a 32-deep drain pay the linger 32 times.
        if wait <= 0 or len(batch) > 1:
            break
        now = time.monotonic()
        if deadline is None:
            deadline = now + wait
        elif now >= deadline:
            break
        time.sleep(0.001)
    return batch


def group_by_peer(keys):
    """``[key, ...]`` → ``{key: [positions]}``, in first-seen order, skipping ``None``.

    One exchange per producer, with each request's answer found again by its position. Positions
    rather than request ids for the same reason the wire carries slots and not ids: a position is
    minted here and cannot be confused with an identity."""
    out: dict = {}
    for i, k in enumerate(keys):
        if k is None:
            continue
        out.setdefault(k, []).append(i)
    return out


def ask_shape(ask):
    """``{group: ids}`` → ``{group: len(ids)}``, the cheap fingerprint of a signature request.

    The prefetch builds a request's ask on the recv loop and the plan rebuilds it moments later
    inside ``_plan_aligned``. Both derive it from the same ``req_meta`` through the same pure
    functions, so they agree by construction — but "agree by construction" is exactly what was
    believed about the producer and decode block-id spaces, and that cost a run: signatures were
    applied against the wrong table with no error anywhere. Comparing shapes at the point of use
    turns any future divergence into a full read instead of a silent mis-pairing."""
    return {int(gi): len(ids) for gi, ids in (ask or {}).items()}


def claim_prefetched(cache, rid, ask):
    """Take ``rid``'s prefetched signatures out of ``cache``. Returns ``(sigs, mismatch)``.

    POP, never peek. A cached answer that outlived its request would be handed to a later one —
    the same failure mode as confusing the producer's block ids with the decode's, which resolved no
    aliases, applied 17 wrong ones, and reported nothing anywhere.

    The shape check is the other half of that guard. The prefetch builds a request's ask from
    ``req_meta`` and the plan rebuilds it moments later from the aligned lists; they agree by
    construction today, but "agrees by construction" is precisely what was believed about the two
    block-id spaces. If they ever diverge, row *i* of the payload describes a different block than
    plan slot *i*, so ``mismatch`` is returned and the caller reads the request in full."""
    entry = (cache or {}).pop(rid, None)
    if entry is None:
        return {}, None
    shape, sigs = entry
    want = ask_shape(ask)
    if shape != want:
        return {}, (shape, want)
    return sigs, None


def partition_by_age(items, deadline_s, now, max_batch):
    """Split ``[(stamp, item)]`` into ``(fresh, stale, defer)`` for one round trip.

    * **stale** — waited longer than ``deadline_s``. These go straight through WITHOUT an exchange,
      and that is the whole point of the design rather than a degradation of it: under a burst the
      alternative is to queue, and queueing behind a 380 ms round trip is what produced 40 s of the
      decode running ONE request while two hundred waited.
    * **fresh** — up to ``max_batch``, exchanged together.
    * **defer** — the rest, carried to the next round trip *with their original stamps*, so they age
      toward the deadline instead of waiting forever behind a queue that keeps refilling.

    Arrival order is preserved, so the oldest are served first and nothing starves. Stamps are
    caller-supplied so this stays pure and testable with no clock."""
    fresh, stale, defer = [], [], []
    for stamp, item in items:
        if now - stamp >= deadline_s:
            stale.append((stamp, item))
        elif len(fresh) < max(1, int(max_batch)):
            fresh.append((stamp, item))
        else:
            defer.append((stamp, item))
    return fresh, stale, defer


class SignaturePipeline(threading.Thread):
    """Runs the signature exchange on its OWN thread, so a transfer never waits for the producer.

    The problem it solves. The recv thread did exchange, then plan, then transfer, strictly in
    order: ~132 ms of ZMQ round trip plus ~63 ms of planning per request, which caps the whole
    transport at about **7.6 requests a second** against the ~100/s the transfers alone sustain. It
    is also the ramp — ``Running=1`` with ``Waiting`` climbing to 199 for four straight samples while
    the stock connector held ``Waiting=0`` through its entire KV fill.

    Splitting the thread alone would only move that ceiling. **The deadline is what removes it**: a
    request that has waited too long is forwarded unplanned and read in full, so a burst costs
    compression instead of liveness. See :func:`partition_by_age`.

    Transport-free by construction — ``exchange`` is injected and the queues hold whatever the caller
    puts in them — so the part that can stall a recv worker is testable with threads and no NPU.

    ``exchange`` is allowed to raise: a failure is contained here and the batch is forwarded
    unplanned, because the alternative is a recv thread that stops feeding the decode."""

    def __init__(self, inbox, exchange, deadline_s, max_batch, clock=None, on_error=None):
        super().__init__(daemon=True, name="BFFSignaturePipeline")
        self.inbox = inbox
        self.ready: queue.Queue = queue.Queue()
        self._exchange = exchange
        self._deadline_s = float(deadline_s)
        self._max_batch = max(1, int(max_batch))
        self._clock = clock or time.monotonic
        self._on_error = on_error
        self.exchanged = 0
        self.passed_through = 0
        self.deferred_carried = 0
        self.batches = 0
        # Busy time on THIS thread — the whole point of the accounting. It is charged only to the
        # round trip, because a run that shows this exchange living here rather than on the transfer
        # thread is the proof that "off the critical path" is real and not inferred from an absence.
        self.busy_s = 0.0
        self._start_time = None
        self._next_report = 1

    def summary(self, now) -> str:
        wall = (now - self._start_time) if self._start_time is not None else 0.0
        served = self.exchanged + self.passed_through
        duty = (self.busy_s / wall * 100.0) if wall > 0 else 0.0
        per = (self.busy_s / self.batches * 1e3) if self.batches else 0.0
        return (f"{served} request(s) in {self.batches} round trip(s), exchanged {self.exchanged}, "
                f"deadline-skipped {self.passed_through} | busy {self.busy_s:.1f}s of {wall:.1f}s "
                f"= {duty:.1f}% of ITS OWN thread ({per:.0f} ms/round trip)")

    def _report(self, now) -> None:
        served = self.exchanged + self.passed_through
        if served < self._next_report:
            return
        self._next_report = self._next_report * 10 if self._next_report < 100 else \
            self._next_report + 100
        logger.info("BFF pull-v2 signature pipeline: %s", self.summary(now))

    def _next_raw(self, blocking: bool):
        """Whatever is on the inbox now; block for the first item only when there is no backlog."""
        if blocking:
            first = self.inbox.get()
            return drain_queue(self.inbox.get_nowait, first)
        out = []
        try:
            while True:
                out.append(self.inbox.get_nowait())
        except queue.Empty:
            pass
        return out

    def run(self):
        self._start_time = self._clock()
        deferred: list = []
        while True:
            try:
                # Blocking only when nothing is carried over. With a backlog this spins through
                # round trips as fast as the producer answers, which is the whole throughput story.
                raw = self._next_raw(blocking=not deferred)
                now = self._clock()
                batch = deferred + [(now, item) for item in raw if item is not None]
                for item in raw:
                    if item is None:
                        # The vendored loop logs and acknowledges these; keep that here so the
                        # transfer thread never sees one.
                        logger.warning("Received a None request!")
                        self.inbox.task_done()
                fresh, stale, deferred = partition_by_age(
                    batch, self._deadline_s, now, self._max_batch)
                self.deferred_carried = len(deferred)
                # Stale FIRST, and before the round trip: they are late already, and making them
                # wait for an exchange they are not part of is the exact stall being removed.
                for _stamp, item in stale:
                    self.passed_through += 1
                    self.ready.put(item)
                if not fresh:
                    if stale:
                        self._report(now)
                    continue
                self.batches += 1
                t0 = self._clock()
                try:
                    self._exchange([item for _stamp, item in fresh])
                    self.exchanged += len(fresh)
                except Exception as e:  # noqa: BLE001 - see the class docstring
                    if self._on_error is not None:
                        self._on_error(e)
                finally:
                    self.busy_s += self._clock() - t0
                    # ALWAYS forwarded, answered or not. A request left here is a request whose
                    # blocks are never read and whose producer is never signalled.
                    for _stamp, item in fresh:
                        self.ready.put(item)
                    self._report(self._clock())
            # Deliberately blind: this thread dying strands every queued request — their KV is
            # never read and their producer is never signalled, so the decode waits on them
            # forever. No exception is narrow enough to be worth that.
            except Exception as e:  # noqa: BLE001
                if self._on_error is not None:
                    self._on_error(e)


class PendingAsk:
    """One caller's slot in a batched exchange.

    ``done`` is set exactly once, by whichever thread served the batch, and ALWAYS — a caller left
    unset would block its recv worker for the rest of the run."""

    __slots__ = ("done", "payload", "result")

    def __init__(self, payload):
        self.payload = payload
        self.result: Any = None
        self.done = threading.Event()


class BatchCoalescer:
    """Serve concurrent callers to the same peer with one round trip.

    Leader/follower, no background thread and no timer: whoever wins a peer's lock drains everything
    queued for it and serves the lot. Because the leader holds that lock across the round trip, the
    next batch grows to exactly what arrived while it was in flight — the batch size self-tunes to
    the load with no tuning knob.

    ``send(key, items)`` is injected: it sets ``item.result`` on each item it can answer. It is
    allowed to raise, and a raise is CONTAINED here rather than propagated — otherwise the outcome
    would depend on whether a caller happened to be the leader (exception) or a follower (no
    answer), and a future sender that forgot its own try/except could take down a recv worker. Every
    caller gets the same thing on failure: no answer, which reads as "ask for it all". Kept
    transport-free so this logic — the part that can hang a recv worker — is testable with threads
    and no NPU.

    The wait is a plain lock acquisition rather than a condition variable because there is no lost
    wakeup to guard against: a caller either finds its item already done, or acquires the lock and
    becomes the leader for whatever remains."""

    def __init__(self, send, max_batch: int = 32):
        self._send = send
        self._max_batch = max(1, int(max_batch))
        self._lock = threading.Lock()
        self._peer_locks: dict[Any, Any] = {}
        self._queue: dict[Any, list] = {}
        self.batches = 0
        self.batched_items = 0
        self.failures = 0

    def ask(self, key, payload):
        """Queue ``payload`` for ``key`` and return this caller's own result."""
        item = PendingAsk(payload)
        with self._lock:
            self._queue.setdefault(key, []).append(item)

        while not item.done.is_set():
            with self._peer_lock(key):
                # Re-checked inside the lock: while we waited for it, the previous leader may have
                # picked our item up and answered it.
                if item.done.is_set():
                    break
                with self._lock:
                    queued = self._queue.get(key) or []
                    batch, rest = queued[:self._max_batch], queued[self._max_batch:]
                    if rest:
                        self._queue[key] = rest
                    else:
                        self._queue.pop(key, None)
                if not batch:
                    continue     # taken by a leader still in flight; wait for the lock again
                try:
                    self._send(key, batch)
                    self.batches += 1
                    self.batched_items += len(batch)
                except Exception as e:      # noqa: BLE001 - see the class docstring
                    self.failures += 1
                    logger.warning("BFF: batched exchange with %s failed for %d caller(s) (%s) — "
                                   "they proceed without an answer.", key, len(batch), e)
                finally:
                    # Unconditional: a raising sender costs these callers their compression, never
                    # their liveness.
                    for it in batch:
                        it.done.set()
        return item.result

    def _peer_lock(self, key):
        with self._lock:
            lk = self._peer_locks.get(key)
            if lk is None:
                lk = self._peer_locks[key] = threading.Lock()
            return lk


def signature_batch_plan(per_slot: dict) -> dict:
    """``{slot: {group: ids}}`` → ``{group: (slots, flat_ids, lengths)}``, one entry per group.

    The producer computes each GROUP once for the whole batch, so every slot's ids for that group
    are concatenated in a fixed order and the resulting rows are sliced back by ``lengths``. This is
    the arithmetic that keeps slots apart, and getting it wrong does not raise — it hands one
    request's signatures to another, which is the same class of silent mis-attribution that the
    producer/decode block-id spaces already cost this connector once.

    ``slots`` and ``lengths`` are positionally paired with each other and with the slices of
    ``flat_ids``: slot ``slots[i]`` owns ``lengths[i]`` rows, starting after all earlier lengths."""
    out: dict[int, tuple] = {}
    for slot, per_group in (per_slot or {}).items():
        for gi, ids in (per_group or {}).items():
            if not ids:
                continue
            slots, flat, lengths = out.setdefault(int(gi), ([], [], []))
            slots.append(int(slot))
            flat.extend(int(b) for b in ids)
            lengths.append(len(ids))
    return out


def sig_request_batch_msg(per_slot: dict) -> tuple:
    """D → P: ``(tag, {slot: {group: [P block ids]}})`` for several requests at once.

    ``slot`` is the caller's position in this exchange, nothing more — it is minted per message and
    means nothing outside it. Request ids are deliberately NOT sent: the producer needs only "which
    blocks", and keeping ids off the wire means a slot can never be confused with an identity."""
    out = {}
    for slot, groups in (per_slot or {}).items():
        one = {int(gi): [int(b) for b in ids] for gi, ids in (groups or {}).items() if ids}
        if one:
            out[int(slot)] = one
    return (MSG_SIG_REQUEST_BATCH, out)


def sig_reply_batch_msg(per_slot: dict) -> tuple:
    """P → D: ``(tag, {slot: {group: payload}})``, answering in the slots it was asked in.

    A slot the producer could not describe is omitted rather than nulled, and the decode reads its
    absence as "no signatures" — that request is then pulled in full, which is always safe."""
    out = {}
    for slot, payloads in (per_slot or {}).items():
        one = {int(gi): p for gi, p in (payloads or {}).items() if p is not None}
        if one:
            out[int(slot)] = one
    return (MSG_SIG_REPLY_BATCH, out)


def parse_sig_reply_batch(msg) -> dict:
    """P's batched reply → ``{slot: {group: payload}}``, or ``{}`` for anything unrecognisable.

    Total for the same reason as :func:`parse_sig_reply`, and it is the path an older producer lands
    on: it does not know :data:`MSG_SIG_REQUEST_BATCH`, so it answers with the single-request tag,
    which fails the tag check here and every request in the batch is read in full."""
    if (not msg or len(msg) < 2 or msg[0] != MSG_SIG_REPLY_BATCH
            or not isinstance(msg[1], dict)):
        return {}
    out: dict[int, dict] = {}
    for slot, groups in msg[1].items():
        if not isinstance(groups, dict):
            continue
        try:
            key = int(slot)
        except (TypeError, ValueError):
            continue
        one = {}
        for gi, payload in groups.items():
            try:
                one[int(gi)] = payload
            except (TypeError, ValueError):
                continue
        if one:
            out[key] = one
    return out


def dedup_gate_open(usage, threshold) -> bool:
    """Should dedup (the exchange + aliasing) run at this KV-cache usage? (Update 11.)

    Open (True) when ``threshold <= 0`` — the gate is disabled, dedup always on, the prior behaviour.
    Otherwise open only when ``usage`` is known AND at/above ``threshold``. Unknown usage (None) keeps
    it CLOSED so an unmeasurable state behaves like baseline rather than silently deduping. Pure, so
    the policy is unit-testable without a block pool."""
    if threshold <= 0.0:
        return True
    return usage is not None and usage >= threshold


def assemble_sig_payload(rows: list) -> dict | None:
    """Merge per-block signature payloads (each a 1-row ``SignatureCodec.encode`` dict) into one
    n-block slot payload, byte-identical to ``SignatureCodec.encode`` over the stacked rows.

    This is what lets the producer cache signatures per ``(group, block_id)`` yet still answer a
    decode exchange in the exact per-slot format the decode already parses — so the wire protocol and
    the decode are untouched (Update 9). ``rows`` are in slot order, one per block. Returns None if
    any row is missing, so the caller reads that slot in full rather than pairing short.

    Pure and vllm-free (bytes + lists only) so it is unit-testable off-NPU."""
    if not rows or any(r is None for r in rows):
        return None
    out = {
        "sig": b"".join(r["sig"] for r in rows),
        "dim": rows[0]["dim"],
        "norms": [r["norms"][0] for r in rows],
        "hashes": [r["hashes"][0] for r in rows],
    }
    if rows[0].get("kvn") is not None:
        out["kvn"] = b"".join(r["kvn"] for r in rows)
        out["kvn_v"] = b"".join(r["kvn_v"] for r in rows)
        out["kvn_layers"] = rows[0]["kvn_layers"]
    return out


def sig_payload_shape(sigs) -> dict:
    """``{group: rows}`` read off the PAYLOADS themselves, not off the ask that requested them.

    ``ask_shape`` fingerprints the question; this fingerprints the answer. Storing the answer's own
    shape is what lets ``claim_prefetched`` catch a producer that sent a different number of rows
    than D asked for — the piggyback path used to store the ask's shape beside an untrimmed payload,
    so the guard passed and the disagreement only surfaced deep inside ``DedupEngine._plan`` as
    ``signature/block mismatch``, discarding the whole request."""
    return {int(gi): len((p or {}).get("norms") or ()) for gi, p in (sigs or {}).items()}


def trim_sig_payload_to(payload, n: int):
    """Keep this payload's **last** ``n`` block rows, or ``None`` if it cannot be made to fit.

    The tail, never the head, because that is the transport's alignment convention throughout:
    ``align_per_group`` keeps P's *last* ``len(local)`` ids, since a decode-side prefix-cache hit
    covers the *front* of the sequence. The producer's piggybacked payload is signed over P's WHOLE
    block list, so it is longer by exactly that cached head; pairing row 0 with D's block 0 would
    shift every alias by the hit length and substitute KV from the wrong positions — fluently, with
    no error anywhere, which is the failure mode that cost the group-0 run.

    ``None`` when the payload is SHORTER than the ask or internally inconsistent: short is not a
    prefix hit, it is a producer/decode disagreement, and the honest answer to that is a full read.
    Pure bytes/list arithmetic — no torch, no device — so it is unit-testable off-NPU."""
    n = int(n)
    if not payload or n <= 0:
        return None
    try:
        dim = int(payload["dim"])
        norms = list(payload["norms"])
        hashes = list(payload["hashes"])
        sig = payload["sig"]
        rows = len(norms)
        # Every parallel array must agree about the row count before any of them is sliced.
        if rows < n or len(hashes) != rows or dim <= 0 or len(sig) != rows * dim * 2:
            return None
        if rows == n:
            return payload
        drop = rows - n
        out = dict(payload)
        out["sig"] = sig[drop * dim * 2:]
        out["norms"] = norms[-n:]
        out["hashes"] = hashes[-n:]
        if payload.get("kvn") is not None:
            stride = int(payload["kvn_layers"]) * 4     # fp32 per layer, k and v alike
            kvn, kvn_v = payload["kvn"], payload["kvn_v"]
            if stride <= 0 or len(kvn) != rows * stride or len(kvn_v) != rows * stride:
                return None
            out["kvn"] = kvn[drop * stride:]
            out["kvn_v"] = kvn_v[drop * stride:]
        return out
    except Exception:  # noqa: BLE001 - a malformed payload costs compression, never liveness
        return None


def fit_piggyback_to_ask(sigs, ask):
    """Tail-trim a whole piggybacked ``{group: payload}`` onto ``ask``, or ``None`` to go ask P.

    All-or-nothing per request: a group the producer did not sign, or one that cannot be trimmed to
    the asked length, means this request's answer is incomplete, and a partial dict would fail
    ``claim_prefetched``'s shape check anyway. Falling back to the on-demand exchange costs a round
    trip; guessing costs correctness."""
    if not sigs or not ask:
        return None
    out = {}
    for gi, ids in ask.items():
        fitted = trim_sig_payload_to(sigs.get(int(gi)), len(ids))
        if fitted is None:
            return None
        out[int(gi)] = fitted
    return out


def split_cached_blocks(block_ids: list, cache: dict, gi: int) -> tuple:
    """Partition ``block_ids`` into (cached rows in slot order with None for misses, missing ids).

    ``cache`` is keyed ``(gi, block_id) -> per-block payload``. Returned ``rows`` is index-aligned
    with ``block_ids`` (None where absent); ``missing`` is the de-duplicated list of ids to compute.
    Pure, so the cache-hit/miss bookkeeping is testable without a device."""
    rows = [cache.get((int(gi), int(b))) for b in block_ids]
    missing = [int(b) for b, r in zip(block_ids, rows) if r is None]
    # de-dup preserving order — a slot can ask the same id twice (a prefix shared within itself)
    seen: set = set()
    missing = [b for b in missing if not (b in seen or seen.add(b))]
    return rows, missing


def warm_group(group_layers: dict, skip=lambda gi: False):
    """The group to build the signature projections on at startup, or None.

    The first group that is really signed: group 0 is never signed (see
    signature_request_and_plan_groups), and the JL matrix is sized by the group's layer count, so a
    matrix warmed on it is discarded by the first real signing. On GLM-5.1 (40/39 layers) that
    rebuild is ~3.8 s on the prefill node's forward thread, on top of the ~3.8 s the warm-up spent."""
    return next((g for g in sorted(group_layers)
                 if g > 0 and group_layers[g] and not skip(g)), None)


def warm_groups(group_layers: dict, skip=lambda gi: False) -> list:
    """One signed group per distinct layer count: the groups whose JL matrices the warm-up builds.

    The JL matrix is cached per width (pd_dedup_v2.jl_projection), so the split layout (signed
    groups of 20, 20 and 19 layers) needs two matrices; warming only :func:`warm_group` left the
    19-layer build (~1.9 s) to the first real request."""
    seen, out = set(), []
    for g in sorted(group_layers):
        n = len(group_layers[g] or ())
        if g > 0 and n and not skip(g) and n not in seen:
            seen.add(n)
            out.append(g)
    return out


def cache_list_device(kv_caches):
    """Device of a list of per-layer KV caches, tolerating both layouts BFF sees.

    A layer's cache is a ``(key, value)`` tuple here (``signature_matrix`` treats ``first[0]`` as the
    key), not a stacked ``[2, num_blocks, ...]`` tensor — so reading ``kv_caches[0].device`` blindly
    dereferences ``.device`` on a tuple, which is the exact crash the first materialize run hit
    ('tuple' object has no attribute 'device'). Pure and vllm_ascend-free so it is unit-testable
    off-NPU; the copy callback in the NPU section calls it."""
    if not kv_caches:
        return None
    probe = kv_caches[0]
    probe = probe[0] if isinstance(probe, (list, tuple)) else probe
    return probe.device


def cache_parts(kv_cache) -> list:
    """Every per-block tensor of one layer's cache, each indexed by block id on dim 0.

    A tuple is taken whole — under DSA it is ``(latent KV, rope key, indexer key[, indexer key
    scale])``, and a copy that stops at ``[0]``/``[1]`` leaves the lightning indexer selecting from
    the destination block's stale keys. A stacked ``[2, num_blocks, ...]`` tensor yields its K and V
    halves. ``None`` entries (an optional tensor a layer does not have) are skipped."""
    if isinstance(kv_cache, (list, tuple)):
        return [t for t in kv_cache if t is not None]
    return [kv_cache[0], kv_cache[1]]


# =================================================================================================
# Ascend/NPU-only section
# =================================================================================================
try:
    import msgspec
    import torch
    import zmq
    from vllm.config import VllmConfig  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: F401
    from vllm.utils.network_utils import make_zmq_path, make_zmq_socket
    from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: F401
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
        group_concurrent_contiguous,
    )

    from kv_fast_fusion import pd_dedup_v2, pd_lsh
    from kv_fast_fusion.pd_dedup_v2 import (
        AliasApplier,
        DedupEngine,
        KVLayoutError,
        SignatureCodec,
        project_signatures,
        signature_matrix,
        signature_matrix_by_layer,
        signatures_for_group,
    )
    from kv_fast_fusion_ascend.connectors import mooncake_connector_ff as v1
    from kv_fast_fusion_ascend.connectors.merge_stats import (
        BFFMergeStats,
        tp_group_if_sharded,
    )
    from kv_fast_fusion_ascend.hybrid_layout import build_hybrid_layout, model_declares_mamba

    _ASCEND_AVAILABLE = True
except Exception as _e:  # pragma: no cover - optional dependency
    logger.info("MooncakeConnectorFFv2: Ascend stack unavailable (%s); only the pure glue is "
                "importable.", _e)
    _ASCEND_AVAILABLE = False


if _ASCEND_AVAILABLE:

    _SYNC_WARNED = False

    def _alloc_stats() -> str:
        """The NPU caching allocator's retry count and reserved memory, for the signing profile line.

        A retry is the allocator failing to find a cached block, releasing its cache and asking the
        device again — the suspected cause of a 40 ms projection whose warm repeat takes 3.5 ms."""
        try:
            st = torch.npu.memory_stats()
            return (f"allocator: {int(st.get('num_alloc_retries', 0))} retries, "
                    f"{torch.npu.memory_reserved() / 2**30:.2f} GiB reserved")
        except Exception:  # noqa: BLE001 - telemetry must never fail a signing
            return "allocator: n/a"

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
            self.batches = 0
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
            cache = ""
            if PRODUCER_SIGCACHE:
                h = getattr(self._worker, "_sig_cache_hits", 0)
                m = getattr(self._worker, "_sig_cache_misses", 0)
                tot = h + m
                # The number that says the precompute is doing its job: a high hit rate means the
                # exchange was served from cache and paid no on-demand NPU drain.
                rate = 100.0 * h / tot if tot else 0.0
                cache = f" | precompute cache {h}/{tot} blocks ({rate:.1f}% hit)"
            # Every served request is a round trip the piggyback failed to cover, so the overflow the
            # per-step cap dropped belongs on the same line as the served count — it is usually why.
            over = getattr(self._worker, "_sig_precompute_overflow", 0)
            if over:
                cache += f" | piggyback overflow {over} (cap {SIG_PRECOMPUTE_PER_STEP}/step)"
            logger.info("BFF pull-v2 signature server: served %d request(s) in %d exchange(s) "
                        "(%.1f per exchange), %d failed%s%s.", self.served, self.batches,
                        self.served / self.batches if self.batches else 0.0, self.failed,
                        (" " + str(self.fail_reasons)) if self.fail_reasons else "", cache)

        def _handle(self, msg):
            """Answer either shape. Both tags are served so a decode one version behind still
            works; the decode only ever sends the batched one."""
            if not msg:
                return sig_reply_msg({})
            if msg[0] == MSG_SIG_REQUEST_BATCH:
                return self._handle_batch(msg[1] or {})
            if msg[0] != MSG_SIG_REQUEST:
                return sig_reply_msg({})
            out = self._signatures_for({0: msg[1] or {}}).get(0, {})
            return sig_reply_msg(out)

        def _handle_batch(self, per_slot: dict):
            return sig_reply_batch_msg(self._signatures_for(per_slot))

        def _signatures_for(self, per_slot: dict) -> dict:
            """``{slot: {group: ids}}`` → ``{slot: {group: payload}}``, computing each GROUP once
            for the whole batch.

            Batching across SLOTS was the first hoist and it was worth ~30% (222 ms → ~156 ms per
            request). Batching across GROUPS is the second and it is worth far more: each group's
            signatures end in three device-to-host transfers, and each transfer is an NPU queue
            drain on a node busy prefilling, so seven groups cost 21 drains and an exchange measured
            **5.7 s**. One pass over every group in the batch costs 3, regardless of group count.

            Slots are kept apart by construction: the ids are concatenated in slot order and the
            resulting rows are sliced back by the same offsets, so row *i* of a slot's payload is
            still that slot's block *i*. A group one slot did not ask about simply contributes no
            rows and is absent from its answer."""
            # PRODUCER_SIGCACHE: answer from the save-time precompute cache (miss → on-demand
            # fallback inside serve_signatures_cached). This is what moves the NPU drains off the
            # decode's blocking exchange onto prefill — see Update 9.
            if PRODUCER_SIGCACHE:
                out = self._worker.serve_signatures_cached(per_slot)
                served = sum(1 for v in out.values() if v)
                if served:
                    self.served += served
                self.batches += 1
                self._report()
                return out
            out: dict[int, dict] = {slot: {} for slot in per_slot}
            plan = signature_batch_plan(per_slot)
            try:
                payloads_by_group, failures = self._worker.signatures_for_groups_split(
                    {gi: (flat, lengths) for gi, (_slots, flat, lengths) in plan.items()})
            except Exception as e:  # pragma: no cover - defensive
                # Only reachable if the pass itself fails rather than one group in it; per-group
                # faults come back in `failures` so the other groups still answer.
                self._note_failure("sig_error")
                logger.warning("BFF pull-v2: signature build failed for the whole batch: %s", e)
                payloads_by_group, failures = {}, []
            for gi, e in failures:
                if isinstance(e, KVLayoutError):
                    # Counted, not swallowed: it means the cache cannot be indexed by connector
                    # block ids, which would make every signature in the run meaningless. Logged
                    # ONCE per group — the producer marks such a group unsignable and stops
                    # retrying it, so a repeat here would mean the sticky set is not working.
                    self._note_failure("kv_layout")
                    logger.warning_once(
                        "BFF pull-v2: cannot index KV for group %s (%s) — this group is now "
                        "excluded from dedup for the rest of the run." % (gi, e))
                else:
                    self._note_failure("sig_error")
                    logger.warning("BFF pull-v2: signature build failed for group %s: %s", gi, e)
            for gi, (slots, _flat, _lengths) in plan.items():
                for slot, payload in zip(slots, payloads_by_group.get(gi, ())):
                    if payload is not None:
                        out[slot][gi] = payload

            served = sum(1 for v in out.values() if v)
            if served:
                self.served += served
            self.batches += 1
            self._report()
            return out

    class _SigClient:
        """Decode side: one REQ socket per producer peer, used from the recv thread.

        **Several requests are served by one exchange**, which cuts the producer's per-request cost
        from ~222 ms to ~156 ms at con512. Some of that is a device sync — ``signatures_for_group``
        ends in a ``.cpu()``, once per group per request — but only about a third of it; the rest is
        per-request device work (gather, LSH, encode) that batching cannot remove.

        **Where the batch comes from.** Not from concurrent callers: the vendored recv thread is a
        single thread handling one request at a time (its ``ThreadPoolExecutor(max_workers=32)`` is
        constructed and never referenced), so :meth:`ask` and its coalescer can only ever see one
        caller and measured exactly that — 512 requests in 512 exchanges, "1.0 per exchange" in the
        producer's own log. The batch comes from :meth:`ask_many`, which the recv loop calls with a
        drained run of the request queue. :class:`BatchCoalescer` is kept because it is correct for
        any future caller that does run threads, and because :meth:`ask_many` reuses its item
        protocol.

        **MEASURED: batching this is throughput-neutral. Do not expect a run to get faster.**
        With it live the producer reported 21.6 requests per exchange during ramp-up and 3.26 over
        the run, and con512 came back at 83.6 ms per decode step against 81.3 ms with the identical
        code batching nothing — a difference fully explained by that run's 4.2% longer sequences.

        It also does NOT fix the decode's ramp. The theory it was built on was that the ~200
        requests holding allocated KV at ``running=1`` were queued behind this exchange; batching it
        21x left the ramp unchanged (``running`` still 1 for the first 40 s), so they were waiting on
        something else. Do not re-derive that theory. At con512 the exchange is at most a fifth of
        the gap; per-step cost is the other four fifths (83.6 vs the baseline's 61.6 ms, at identical
        concurrency). Kept because a 30% cut in producer load is real and costs nothing to hold."""

        def __init__(self):
            self._lock = threading.Lock()
            self._ctx = None
            self._socks: dict[tuple, Any] = {}
            self._enc = msgspec.msgpack.Encoder()
            self._dec = msgspec.msgpack.Decoder()
            self._announced: set = set()
            self._batcher = BatchCoalescer(self._exchange, MAX_SIG_BATCH)
            self._direct_batches = 0
            self._direct_items = 0
            # Wall time inside the round trip. Mostly waiting on the producer, so it should NOT
            # dominate dedup's per-step cost — if it does, the recv thread is the bottleneck and
            # not the CPU work around it.
            self.exchange_ms = 0.0

        @property
        def batches(self) -> int:
            return self._batcher.batches + self._direct_batches

        @property
        def batched_requests(self) -> int:
            return self._batcher.batched_items + self._direct_items

        def ask_many(self, host, port, payloads: list) -> list:
            """One round trip for several requests at once; answers come back positionally.

            The batched path, and the only one this transport actually uses. A payload that is
            empty takes no slot on the wire but still gets its ``{}`` back in place, so the caller's
            indexing into ``payloads`` never has to account for who was skipped.

            Failure is total and safe: :meth:`_exchange` contains its own errors, so a dead or slow
            producer leaves every result ``{}`` and every request in the batch is read in full."""
            out: list = [{} for _ in payloads]
            if host is None or port is None:
                return out
            live = [(i, p) for i, p in enumerate(payloads) if p]
            if not live:
                return out
            key = (host, int(port))
            items = [PendingAsk(p) for _, p in live]
            self._exchange(key, items)
            self._direct_batches += 1
            self._direct_items += len(items)
            for (i, _), item in zip(live, items):
                out[i] = item.result or {}
            return out

        def ask(self, host, port, groups_to_ids: dict) -> dict:
            """Return ``{group: signature payload}`` for THIS caller, or ``{}`` on any failure.

            Batching is invisible from here: the caller asks about one request and gets one
            request's answer."""
            if host is None or port is None or not groups_to_ids:
                return {}
            return self._batcher.ask((host, int(port)), groups_to_ids) or {}

        def _exchange(self, key, batch) -> None:
            """One round trip for ``batch``, filling in each item's result.

            Raising here is safe and deliberate on failure: :class:`BatchCoalescer` marks every item
            done regardless, so a dead producer costs these requests their compression rather than
            their liveness — they are simply read in full."""
            host, port = key
            t0 = time.perf_counter()
            try:
                sock = self._sock_for(host, port)
                # The producer's work scales with the batch, so the budget does too. Set per call
                # rather than at socket creation, since the batch size is only known here.
                sock.setsockopt(zmq.RCVTIMEO, int(SIG_EXCHANGE_TIMEOUT * 1000 * len(batch)))
                sock.send(self._enc.encode(
                    sig_request_batch_msg({i: it.payload for i, it in enumerate(batch)})))
                answer = parse_sig_reply_batch(self._dec.decode(sock.recv()))
                for i, it in enumerate(batch):
                    it.result = answer.get(i, {})
            except Exception as e:
                # A REQ socket that timed out is stuck in the wrong state; drop it so the next
                # exchange starts clean, and read these requests whole.
                #
                # The elapsed time is the diagnosis: at (or just over) the budget the producer never
                # answered — either it is not listening or it is too slow, which its own log
                # distinguishes. Well under it means the socket was refused outright, a different
                # problem entirely.
                logger.warning("BFF pull-v2: signature exchange with %s:%s for %d request(s) "
                               "failed after %.0f ms of a %.0f ms budget (%s) — reading them in "
                               "full.", host, port, len(batch), (time.perf_counter() - t0) * 1e3,
                               SIG_EXCHANGE_TIMEOUT * 1e3 * len(batch), e)
                self._drop(host, port)
            finally:
                self.exchange_ms += (time.perf_counter() - t0) * 1e3

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

    class KVCacheRecvingThreadFFv2(TransferDiagnostics, v1.KVCacheRecvingThreadFF):
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
        # Landing-time materialize: the worker's copy callback (own stream, completes before it
        # returns) and its KV-load-failure sink, for a refusal whose re-read cannot run.
        ff_copy_alias_blocks = None
        ff_note_failed_blocks = None
        # BFF_V2_FREE_AT_PROMOTION: the worker's outbox for what landing actually copied, which the
        # scheduler turns into real frees at promotion (landed_promotion).
        ff_landed_sink = None
        # The worker's signatures_for_group_split, for BFF_V2_VERIFY_TRANSFER only. The decode uses
        # the producer's own method against its OWN cache, which is what makes the two sides
        # comparable: same code, same fixed-seed projection, same group layout.
        local_sigs = None
        # The cross-group form of `local_sigs`, for the one caller that recomputes every group of a
        # request at once. Bound beside it; `local_sigs` stays for the single-group diagnostics
        # (_stability, _replay), which run only after a mismatch and so are not worth batching.
        local_sigs_multi = None
        sig_port_offset: int = FF_PULL_V2_PORT_OFFSET
        _logged_first_decline = False
        _logged_first_cap = False
        # Groups already reported by _warn_on_runaway_groups. Deliberately class-level and shared:
        # the warning is once per group for the process, not once per request or per thread.
        _warned_runaway: ClassVar[set] = set()

        @property
        def _sig_cache(self) -> dict:
            """This batch's prefetched signatures, ``{remote_request_id: (shape, sigs)}``.

            Lazy rather than set in ``__init__`` because we deliberately do not override the
            vendored constructor — see the class docstring on what injecting into it cost."""
            cache = self.__dict__.get("_sig_cache_d")
            if cache is None:
                cache = self.__dict__["_sig_cache_d"] = {}
            return cache

        @property
        def piggyback_sigs(self) -> dict:
            """Signatures stapled onto kv_transfer_params (Update 13), ``{remote_request_id: {gi:
            payload}}``, staged by the worker's ``start_load_kv``. Consumed (popped) in
            ``_prefetch_signatures`` before any ask. Lazy for the same reason as ``_sig_cache``."""
            d = self.__dict__.get("_piggyback_d")
            if d is None:
                d = self.__dict__["_piggyback_d"] = {}
            return d

        # The block ids of the request being verified right now, so the stability re-read touches
        # exactly the block that failed. Set immediately before _localise, read only by it.
        _verify_ids: ClassVar[dict] = {}
        # Ask v1's emission loop to record per-block descriptors. Only worth its cost when a
        # mismatch might actually be replayed.
        # Either diagnostic needs the per-block descriptor list: verification to replay ONE failing
        # block, the audit to re-issue the whole request.
        capture_descriptors: bool = VERIFY_TRANSFER > 0 or XFER_AUDIT > 0

        @property
        def _verify_stash(self) -> dict:
            """``{remote_request_id: (plan_groups, sigs)}`` awaiting post-transfer verification."""
            st = self.__dict__.get("_verify_stash_d")
            if st is None:
                st = self.__dict__["_verify_stash_d"] = {}
            return st

        def _verify_ready(self) -> bool:
            """Gates the FETCH, which runs a batch ahead of the claim in _plan_aligned."""
            return VERIFY_TRANSFER > 0 and KVCacheRecvingThreadFFv2._verify_sampler.ready()

        @property
        def _sig_pipeline(self):
            """Thread A, started on first use. Per instance — it owns this thread's request queue."""
            p = self.__dict__.get("_sig_pipeline_obj")
            if p is None:
                p = self.__dict__["_sig_pipeline_obj"] = SignaturePipeline(
                    self.request_queue, self._exchange_for,
                    SIG_DEADLINE_MS / 1000.0, MAX_SIG_BATCH,
                    on_error=lambda e: logger.warning(
                        "BFF pull-v2: signature pipeline error (%s) — those requests are read in "
                        "full.", e))
                p.start()
                logger.info(
                    "BFF pull-v2: ASYNC signature exchange on (deadline %.0f ms, batch cap %d). "
                    "A request that waits longer than the deadline is read in full rather than "
                    "queueing behind a round trip.", SIG_DEADLINE_MS, MAX_SIG_BATCH)
            return p

        @property
        def _engine_lock(self):
            """Serialises every dedup-engine MUTATION — plan, forget, release.

            The engine is otherwise lock-free because it was only ever touched from the one recv
            thread. ``BFF_PULL_V2_ASYNC_PLAN`` breaks that: plan runs on the exchange thread while
            release runs on the transfer thread, so their read-modify-writes on the resident set and
            the alias tables can now interleave. This lock is the entire safety argument — held
            across each whole plan-decide and each release, it keeps the maps consistent while still
            letting the two threads run. With async-plan off it is uncontended (one thread) and
            costs a few nanoseconds, so there is a single code path rather than a branch."""
            lk = self.__dict__.get("_engine_lock_obj")
            if lk is None:
                lk = self.__dict__["_engine_lock_obj"] = threading.Lock()
            return lk

        def _exchange_for(self, batch) -> None:
            """Thread A: ask each producer once for the batch and leave each answer ON its request.

            With ``ASYNC_PLAN`` off this touches NO dedup-engine state, not even its counters — the
            read-only side of the split. With it on, planning moves here too: the same
            :meth:`_plan_aligned` runs under :attr:`_engine_lock` and its result rides on the request
            in ``PLAN_SLOT``, so the transfer thread does no engine work at all. Either way the
            exchange itself is producer I/O and is what this thread exists to overlap."""
            if self.sig_client is None or not (pd_dedup_v2.V2_ENABLED or self._verify_ready()):
                return
            # Usage gate (Update 11) — same policy as the sync path: below threshold, skip the
            # exchange so this batch is read in full. Verify-only runs are never gated.
            if pd_dedup_v2.V2_ENABLED and self._dedup_gate_closed():
                return
            asks = [self._ask_for(req_meta) for req_meta in batch]
            keys = [self._peer_of(req_meta) if ask else None
                    for req_meta, ask in zip(batch, asks)]
            for (host, port), positions in group_by_peer(keys).items():
                answers = self.sig_client.ask_many(host, port, [asks[i] for i in positions])
                for i, sigs in zip(positions, answers):
                    batch[i][SIG_SLOT] = (ask_shape(asks[i]), sigs)
            if ASYNC_PLAN:
                for req_meta in batch:
                    req_meta[PLAN_SLOT] = self._plan_for(req_meta)

        def _plan_for(self, req_meta) -> dict:
            """Thread A: this request's plan, computed the moment its signatures land.

            The transfer thread will re-derive the same ``aligned`` in ``_align_and_group`` — it is
            pure list maths over ``req_meta`` and cheap — so only the plan travels, in ``PLAN_SLOT``.
            Returns ``{}`` on any failure, which Thread B reads as 'nothing declined, transfer in
            full', exactly the synchronous fallback."""
            local = req_meta.get("local_block_ids")
            remote = req_meta.get("remote_block_ids")
            if not local or not v1.flatten_group_lists(local):
                return {}
            try:
                aligned = v1.align_per_group(local, remote)
                return self._plan_aligned(req_meta, aligned)
            except Exception as e:  # noqa: BLE001 - a failed plan costs compression, never liveness
                logger.warning("BFF pull-v2: async plan failed (%s) — this request is read in "
                               "full.", e)
                return {}

        def _run_async(self):
            """Thread B: transfer whatever the pipeline has made ready, and never wait on a peer.

            The vendored body is preserved where it matters — one ``try`` per request, so a single
            failure costs one request — but the queue it reads is the pipeline's, not the raw
            request queue. ``task_done`` still pairs one-for-one with the ``get`` the pipeline made,
            because ``_handle_request`` calls it in its own ``finally``."""
            pipeline = self._sig_pipeline
            ready = pipeline.ready
            while True:
                try:
                    req_data = ready.get()
                except Exception as e:  # noqa: BLE001 - the transfer thread must outlive any fault
                    logger.error(f"Error in KVCacheTransferThread: {e}")
                    continue
                try:
                    self._handle_request(req_data)
                except Exception as e:  # noqa: BLE001 - vendored shape: one failure, one request
                    logger.error(f"Error in KVCacheTransferThread: {e}")
                # Mirror the pipeline's own counters into the stats dump. Read-only of another
                # thread's plain-int/float attributes: torn reads at worst report a slightly stale
                # number in a diagnostic, never a wrong transfer. Cheap enough to do per request so
                # the final dump reflects the whole run.
                if self.dedup_engine is not None:
                    st = self.dedup_engine.stats
                    st.pipeline_busy_ms = pipeline.busy_s * 1e3
                    st.pipeline_batches = pipeline.batches
                    st.pipeline_exchanged = pipeline.exchanged
                    # The pipeline is the single source of truth for deadline skips — a skipped
                    # request is forwarded straight to transfer and may never enter _take_prefetched
                    # (it never does under ASYNC_PLAN), so counting it there would miss it.
                    st.sig_deadline_skipped = pipeline.passed_through

        def run(self):
            """The vendored loop, taking the queue in DRAINED RUNS so the exchange can batch.

            Vendored ``KVCacheRecvingThread.run`` pops one request and handles it synchronously.
            That is what made the batched signature protocol inert: with a single thread there is
            never a second caller to batch with, and the producer logged "1.0 per exchange" for all
            512 requests. Draining the queue first turns the backlog into the batch.

            The backlog is real — at con512 the decode reached 99.8% KV usage at 5 running requests
            while the baseline's waiting count stayed 0 for its entire KV fill — but draining it is
            NOT what fixes that. Batching the exchange 21x left the ramp unchanged; what removed it
            was cutting the KV-cache group count. Keep the drain (it is free and it does cut producer
            load ~30%), and do not reach for it again when a run is slow.

            Everything else is kept faithful to the vendored body on purpose, because this is
            control-plane code we do not own: ``ready_event`` is still set first, ``task_done`` is
            still called for a ``None``, and ``_handle_request`` is still wrapped PER REQUEST so one
            failure costs one request — batching must not turn a single fault into a lost batch.
            A test parses the vendored source and fails if that shape changes underneath us."""
            # Before ready_event, so the duty cycle's clock starts at the thread rather than at the
            # first thing that charges it. This loop does NOT call super().run(), so v1's priming
            # never reaches here — and the first thing to touch the timer would be the `finally`
            # that charges the first signature exchange, which is how it once read 858.6%.
            self._recv_timer                            # noqa: B018 - primes `started`
            # Unconditional, so a run's mode is line 1 of the log rather than inferred from a MISSING
            # line — the exact trap a stale build fell into, running synchronously while everyone
            # read it as async.
            logger.info(
                "BFF pull-v2 recv thread: async exchange %s, async plan %s (deadline %.0f ms, "
                "batch cap %d).", "ON" if ASYNC_SIG else "OFF", "ON" if ASYNC_PLAN else "OFF",
                SIG_DEADLINE_MS, MAX_SIG_BATCH)
            self.ready_event.set()
            if ASYNC_SIG:
                # The exchange moves to its own thread and this one only transfers. Kept as a
                # separate body rather than branching inside the loop: this is control-plane code we
                # do not own, and the synchronous path has to stay recognisably the vendored shape.
                self._run_async()
                return
            while True:
                try:
                    first = self.request_queue.get()
                except Exception as e:  # pragma: no cover - defensive
                    logger.error(f"Error in KVCacheTransferThread: {e}")
                    continue
                try:
                    batch = drain_queue(self.request_queue.get_nowait, first)
                except Exception as e:  # pragma: no cover - defensive
                    # Separate from the get above on purpose: the head is already off the queue, so
                    # a failed drain must not drop it — that would leak its blocks and never signal
                    # the producer. Handle it alone and let the backlog wait.
                    logger.error(f"Error in KVCacheTransferThread: {e}")
                    batch = [first]
                live = []
                for req_data in batch:
                    if req_data is None:
                        logger.warning("Received a None request!")
                        self.request_queue.task_done()
                    else:
                        live.append(req_data)
                if not live:
                    continue
                _t = time.perf_counter()
                try:
                    self._prefetch_signatures(live)
                except Exception as e:  # pragma: no cover - defensive
                    # Never fatal: with no cached signatures every request is read in full.
                    logger.warning("BFF pull-v2: signature prefetch failed (%s) — this batch is "
                                   "read in full.", e)
                finally:
                    # Per BATCH, and it blocks the thread exactly as a transfer does — so it belongs
                    # in the duty cycle. It is also the only phase here that waits on ANOTHER node,
                    # which is why it is named separately from the transfer it delays.
                    self._recv_timer.note_phase(
                        "exchange", (time.perf_counter() - _t) * 1e3)
                for req_data in live:
                    try:
                        self._handle_request(req_data)
                    except Exception as e:
                        logger.error(f"Error in KVCacheTransferThread: {e}")

        def _peer_of(self, req_meta):
            """The producer's signature endpoint for this request, or ``None`` if unaddressable."""
            host = req_meta.get("remote_host")
            port = req_meta.get("remote_handshake_port")
            if host is None or port is None:
                return None
            return (host, int(port) + self.sig_port_offset)

        def _ask_for(self, req_meta) -> dict:
            """This request's signature question, ``{group: P block ids}``, or ``{}``.

            Pure list math on ``req_meta`` — no device, and none of the remote metadata that
            ``_transfer_kv_cache`` fetches — which is why it can run a whole batch ahead of the
            transfer."""
            local_groups = req_meta.get("local_block_ids")
            remote_groups = req_meta.get("remote_block_ids")
            if not local_groups or not v1.flatten_group_lists(local_groups):
                return {}       # full prefix-cache hit: nothing to pull, so nothing to ask about
            try:
                aligned = v1.align_per_group(local_groups, remote_groups)
                ask, _plan = signature_request_and_plan_groups(aligned, v1._FF_GROUPS)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF pull-v2: could not build a signature request (%s) — this "
                               "request is read in full.", e)
                return {}
            return ask

        def _dedup_gate_closed(self) -> bool:
            """Usage gate (Update 11): True ⇒ SKIP dedup this batch because KV usage is below
            DEDUP_KV_THRESHOLD (dedup's memory relief is not yet worth its exchange cost). Counts
            closed batches for the log. Never gates the verify-only path — that must observe an
            unmodified transfer regardless.

            The usage is whatever the SCHEDULER last shipped on the connector metadata, not a local
            read. The block pool lives only in the scheduler process, so at TP>1 a local read is
            always None and the gate — which is deliberately closed-when-unknown — was permanently
            shut for the whole run, silently, whenever the threshold was above zero. The local read
            remains as the fallback for the TP=1 case, where it is the same object anyway."""
            if DEDUP_KV_THRESHOLD <= 0.0 or self.dedup_engine is None:
                return False
            usage = self._kv_usage
            if usage is None:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                usage = _bp.kv_cache_usage()
            if dedup_gate_open(usage, DEDUP_KV_THRESHOLD):
                return False
            self.dedup_engine.stats.dedup_gated_batches += 1
            return True

        def _prefetch_signatures(self, batch) -> None:
            """Ask each producer once for the whole drained run, before any of it is transferred.

            One exchange per peer, answers matched back by position. The cache is cleared at entry,
            not at exit: anything left from a previous batch is by definition an answer nobody
            claimed, and letting it survive would offer one request another request's signatures."""
            cache = self._sig_cache
            cache.clear()
            if (self.dedup_engine is None or self.sig_client is None
                    or not (pd_dedup_v2.V2_ENABLED or self._verify_ready())):
                return
            # Usage gate: below threshold, skip the exchange entirely — the batch is read in full,
            # like baseline. Not applied when only verification is active (V2 disabled), so a verify
            # run still exchanges. See _dedup_gate_closed.
            if pd_dedup_v2.V2_ENABLED and self._dedup_gate_closed():
                return
            asks = [self._ask_for(req_meta) for req_meta in batch]
            stats = self.dedup_engine.stats
            # Piggyback (Update 13): a request whose signatures rode in on kv_transfer_params is
            # answered from the local stash with NO round trip; only the misses reach the producer.
            # This is the whole point — the ask, not the compute, was the ~172 ms cost.
            piggy = self.piggyback_sigs if v1.SIG_PIGGYBACK else None
            keys = []
            for req_meta, ask in zip(batch, asks):
                if not ask:
                    keys.append(None)
                    continue
                rid = req_meta.get("remote_request_id")
                if piggy is not None:
                    psig = piggy.pop(rid, None)
                    # TAIL-trim to the ask before caching (Update 27). P signs its whole block list
                    # while this ask is tail-aligned, so a decode-side prefix-cache hit leaves the
                    # payload longer by the cached head. Untrimmed, the row/id disagreement used to
                    # reach DedupEngine._plan as `signature/block mismatch` and the request was read
                    # in full — the freeing was thrown away for an alignment the transport already
                    # knows how to do. The shape stored is the PAYLOAD's, so claim_prefetched still
                    # guards the pairing rather than trusting the question.
                    fitted = fit_piggyback_to_ask(psig, ask) if psig else None
                    if fitted is not None:
                        if sig_payload_shape(fitted) != sig_payload_shape(psig):
                            stats.piggyback_trimmed += 1
                        cache[rid] = (sig_payload_shape(fitted), fitted)
                        stats.piggyback_hits += 1
                        keys.append(None)          # answered locally — no exchange
                        continue
                    stats.piggyback_misses += 1
                keys.append(self._peer_of(req_meta))
            for (host, port), positions in group_by_peer(keys).items():
                answers = self.sig_client.ask_many(host, port, [asks[i] for i in positions])
                for i, sigs in zip(positions, answers):
                    # Counted here, and counted as ATTEMPTS: DedupStats.is_inert() reads
                    # `exchanges == 0` as "v2 is installed but never ran", which is the difference
                    # between "nothing was worth merging" and "the mechanism never engaged" — the
                    # two readings of an all-zero saving a benchmark cannot otherwise tell apart.
                    stats.exchanges += 1
                    if sigs:
                        cache[batch[i]["remote_request_id"]] = (ask_shape(asks[i]), sigs)
                    else:
                        stats.sig_phase_failed += 1

        def _take_prefetched(self, req_meta, ask) -> dict:
            """Claim this request's prefetched signatures, or ``{}`` to read it in full."""
            rid = req_meta.get("remote_request_id")
            if ASYNC_SIG:
                # Async: the answer rode in on the request, so the "cache" can only ever hold this
                # one. Counted HERE rather than on the exchange thread — see _exchange_for for why
                # that side stays clear of the engine, stats included.
                slot = req_meta.pop(SIG_SLOT, None)
                stats = self.dedup_engine.stats
                if slot is None:
                    # No exchange happened: the request passed the deadline and is read in full. NOT
                    # counted here — the pipeline's `passed_through` is the single source of truth
                    # for deadline skips (mirrored into sig_deadline_skipped in _run_async), because
                    # under ASYNC_PLAN a skipped request never reaches this method at all.
                    return {}
                stats.exchanges += 1
                sigs, mismatch = claim_prefetched({rid: slot}, rid, ask)
                if not sigs and mismatch is None:
                    stats.sig_phase_failed += 1
                if mismatch is not None:
                    stats.sig_phase_failed += 1
                    logger.warning(
                        "BFF pull-v2: prefetched signatures do not match the planned request (%s "
                        "vs %s) — reading it in full rather than pairing rows against the wrong "
                        "blocks.", *mismatch)
                return sigs
            sigs, mismatch = claim_prefetched(self._sig_cache, rid, ask)
            if mismatch is not None:
                self.dedup_engine.stats.sig_phase_failed += 1
                logger.warning(
                    "BFF pull-v2: prefetched signatures do not match the planned request (%s vs "
                    "%s) — reading it in full rather than pairing rows against the wrong blocks.",
                    *mismatch)
            return sigs

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
                    or not (pd_dedup_v2.V2_ENABLED or self._verify_ready())):
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
            # Already fetched, by `run` for this whole drained batch — P was asked about ITS ids.
            # `exchanges` and `sig_phase_failed` are counted there, at the point the question is
            # actually put to the producer.
            sigs = self._take_prefetched(req_meta, ask)
            if not sigs:
                return {}
            # One lock, held across the whole plan-decide and (in _after_transfer) each release.
            # Under ASYNC_PLAN this runs on the exchange thread while releases run on the transfer
            # thread; the lock is what keeps the engine's resident set and alias tables consistent
            # between them. Uncontended and near-free when async-plan is off. See _engine_lock.
            with self._engine_lock:
                if KVCacheRecvingThreadFFv2._verify_sampler.take():
                    # Claimed HERE, not at verification time: the sampler decides which requests get
                    # checked, and that decision has to be made while the signatures are still in
                    # hand. Held for _after_transfer, because the comparison is only meaningful once
                    # the KV has landed. plan_groups is D's ids, positionally paired with P's rows;
                    # `ask` is P's ids for the same rows, kept so a failing group can be re-requested.
                    self._verify_stash[req_meta.get("remote_request_id")] = (ask, plan_groups, sigs)
                if not pd_dedup_v2.V2_ENABLED:
                    # Verification only. It must never plan and never alias — the whole point is to
                    # observe an unmodified transfer.
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
                stats = self.dedup_engine.stats
                declined, total = decline_fraction(result)
                stats.note_request_decline(declined, total)
                pre_cap = result
                result, capped = cap_request_decline(pre_cap)
                kept = None
                if capped and CLIP_FRAC > 0:
                    _clip_t0 = time.perf_counter()
                    clipped, keep = clip_request_decline(
                        pre_cap, plan_groups, self.dedup_engine.alias_errors(ext_id), CLIP_FRAC,
                        CLIP_MAX_REL_ERR, CLIP_TAIL)
                    kept = sum(len(v) for v in keep.values()) or None
                    if kept:
                        stats.clip_aliases_dropped += self.dedup_engine.retain(ext_id, keep)
                        stats.clip_aliases_kept += kept
                        stats.requests_clipped += 1
                        result, capped = clipped, False
                    if _PROFILE_OVERHEAD:
                        stats.clip_ms = (stats.clip_ms or 0.0) + (
                            time.perf_counter() - _clip_t0) * 1e3
                        stats.clip_calls += 1
                stats.note_request(ext_id, declined, total, capped or kept is not None,
                                   clipped=kept)
                if capped:
                    # The aliases are already staged in the engine, so dropping the plan is not
                    # enough — forget() must retract them or they would be applied to blocks this
                    # request is now going to fetch normally. uncount: they are fetched, so they
                    # are not a wire saving either.
                    self.dedup_engine.forget([ext_id], uncount=True)
                    stats.requests_capped += 1
                    if not KVCacheRecvingThreadFFv2._logged_first_cap:
                        KVCacheRecvingThreadFFv2._logged_first_cap = True
                        logger.warning(
                            "BFF pull-v2: a request would have had %d of %d blocks (%.0f%%) "
                            "replaced by other requests' KV — over the %.0f%% ceiling, so it is "
                            "being read in full. That much substitution answers a neighbouring "
                            "prompt, however similar each block is on its own. Tune with "
                            "BFF_V2_MAX_REQ_DECLINE.",
                            declined, total, 100.0 * declined / total, 100.0 * MAX_REQ_DECLINE)
                    return {}
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
            # Under ASYNC_PLAN the exchange thread already planned this request and left the result
            # in PLAN_SLOT; popping it here is all the transfer thread does with the engine. Off, it
            # plans inline exactly as the vendored-derived path always has. One decision, one slot.
            if ASYNC_PLAN:
                planned = req_meta.pop(PLAN_SLOT, None) or {}
            else:
                planned = self._plan_aligned(req_meta, aligned)
            grouped = []
            n_declined = 0
            declined: dict[int, dict] = {}
            req_meta[v1.FF_DECLINED_SLOT] = declined
            for gi, (remote_ids, local_ids) in enumerate(aligned):
                plan_g = planned.get(gi)
                if plan_g is not None and len(plan_g) == len(local_ids):
                    before = len(local_ids)
                    pairs = declined_pairs(plan_g, local_ids, remote_ids)
                    if pairs:
                        declined[gi] = pairs
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
            self._verify_transfer(req_meta)
            self._audit_self_consistency(req_meta)
            if self.dedup_engine is None:
                return
            ext_id = v1._ext_of(req_meta["remote_request_id"])
            if not self._materialize_on_landing(req_meta):
                # Same lock as plan/forget: under ASYNC_PLAN this release (transfer thread) races the
                # exchange thread's plan for other requests, and the lock is what serialises their
                # touches on the engine's shared maps.
                with self._engine_lock:
                    self.dedup_engine.release(ext_id)
                return
            if self.last_segment_count == 0:
                # Fully deduped: no pull ran, so nothing has yet landed a previous tenant's in-flight
                # write to these freshly allocated blocks — the copy below must not be clobbered by
                # one. _issue_transfer does this for every pull. Outside both locks: it is a device
                # drain, and the forward thread must not wait on it.
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                _bp.flush_before_load()
            landed = {} if self.ff_landed_sink is not None else None
            # Every alias is copied, the request's LAST block included: it has not decoded yet, so
            # its own block holds nothing to protect. Only SHARING a partly filled last block is
            # unsafe (its decode would write into the rep), and only the scheduler knows the token
            # count — landed_promotion refuses that swap as `victim_hot`. Refusing it here cost
            # ~35% of all landed candidates in a re-read each (GLM err 0.5: 56 of ~160).
            with self._engine_lock:
                refused = self.dedup_engine.release_materialized(
                    ext_id, self.ff_copy_alias_blocks, None, landed=landed)
            if landed:
                # Before the vendored `finally` reports this request done: vLLM drains get_finished
                # before get_kv_connector_stats each step, so the map can never trail the "done".
                self.ff_landed_sink(ext_id, landed)
            if refused:
                self._repull_refused(req_meta, ext_id, refused)

        def _materialize_on_landing(self, req_meta) -> bool:
            """Realise aliases here, at landing, rather than on the forward path?

            Only for the copy variant (a redirect must write the runner's block table), only without
            ratio scales (they live on the runner, which this thread cannot reach), and only for a
            single-pull transfer: with several pulls per request the declined descriptors cover the
            last pull only, so a refusal could not be re-read in full."""
            return (MATERIALIZE_ON_LANDING and pd_dedup_v2.MATERIALIZE_ALIASES
                    and pd_dedup_v2.SCALE_MODE != "ratio"
                    and self.ff_copy_alias_blocks is not None
                    and int(req_meta.get("tp_num_need_pulls", 1)) == 1)

        def _repull_refused(self, req_meta, ext_id, refused) -> None:
            """Read the refused victims from the producer, which still holds them.

            Runs before the vendored ``finally`` tells P this request is done, which is what makes
            the re-read possible at all; by the forward path P has freed the blocks and a refusal
            costs a full local re-prefill instead. Only a re-read that cannot run goes there."""
            stats = self.dedup_engine.stats
            victims = [(gi, v) for gi, vs in refused.items() for v in vs]
            srcs, dsts, lens, missing = repull_segments(self.declined_descriptors, victims)
            failed = list(missing)
            ok = []
            if srcs and self.declined_session_id is not None:
                fetch = [x for x in victims if x not in missing]
                try:
                    self._issue_transfer(req_meta, self.declined_session_id, srcs, dsts, lens)
                    mini = [([], []) for _ in range(max(gi for gi, _v in fetch) + 1)]
                    for gi, v in fetch:
                        mini[gi][1].append([v])
                    self._reformat_after_pull(mini, req_meta["offset"],
                                              req_meta["tp_num_need_pulls"])
                    ok = fetch
                except Exception as e:  # noqa: BLE001 - a failed re-read degrades to recompute
                    logger.warning("BFF pull-v2: re-read of %d refused alias(es) failed (%s) — "
                                   "recomputing them.", len(fetch), e)
                    failed.extend(fetch)
            else:
                failed = victims
            stats.repulled += len(ok)
            if failed:
                stats.repull_failed += len(failed)
                stats.note_recompute(len(failed), rid=ext_id)
                if self.ff_note_failed_blocks is not None:
                    self.ff_note_failed_blocks([v for _gi, v in failed])

    class MooncakeConnectorWorkerFFv2(v1.MooncakeConnectorWorkerFF):
        """v1's worker plus the signature server (on P) and the dedup engine (on D)."""

        _RECV_THREAD_CLS = KVCacheRecvingThreadFFv2
        _logged_freed = False

        def __init__(self, vllm_config, engine_id, kv_cache_config=None):
            # `_dedup_engine`, never `_engine`: the vendored worker's `self.engine` is the Mooncake
            # TransferEngine, and keeping the two one underscore apart is how the recv thread's
            # collision got written in the first place.
            self._dedup_engine = None
            self._sig_client = None
            self._sig_server = None
            # Pool occupancy as last shipped by the scheduler; see start_load_kv and
            # _dedup_gate_closed. None until the first step, which reads as "closed".
            self._kv_usage: float | None = None
            # Hybrid geometry, derived in register_kv_caches once the tensors exist.
            self._layout = None
            # Groups whose KV cannot be indexed by block id. Sticky: the layout is a property of
            # the allocation, so a group that refuses once refuses forever, and retrying it is a
            # gather plus a log line per exchange for the rest of the run.
            self._unsignable: set[int] = set()
            self._jl = [None]        # JL projection cache, must outlive the calls
            self._proj = [None]      # SimHash projection cache — same reason
            self._ff_failed_blocks: set = set()
            # The recv thread notes failures too (a landing re-read that could not run), while the
            # forward thread takes them; an unlocked swap could hand a block id over twice.
            self._ff_failed_lock = threading.Lock()
            # Landed aliases awaiting the scheduler (BFF_V2_FREE_AT_PROMOTION): filled by the recv
            # thread, drained by get_kv_connector_stats on the forward thread.
            self._ff_landed_outbox: dict = {}
            self._ff_landed_lock = threading.Lock()
            self._group_layers: dict[int, set] = {}
            # Producer-side signature precompute cache (PRODUCER_SIGCACHE): (gi, block_id) -> per-block
            # payload, filled at save time and read by the _SigServer thread, so guarded by a lock.
            # Naturally bounded by the block pool (each id is overwritten by the next request that
            # finishes on it, before any decode asks), so no eviction hook is needed. See Update 9.
            self._sig_row_cache: dict = {}
            self._sig_cache_lock = threading.Lock()
            self._sig_cache_hits = 0
            self._sig_cache_misses = 0
            # Requests that completed prefill in a step but lost the piggyback to the per-step cap —
            # each one is an on-demand round trip the decode pays in front of its transfer.
            self._sig_precompute_overflow = 0
            # BFF_STEP_PROFILE: piggyback signing time on the PRODUCER's forward thread, logged every 50
            # signed requests. CUMULATIVE: read the slope between two lines, not one line's average.
            # Until 2026-11-04a the first signing rebuilt the JL matrix (see warm_group), so one line
            # read as "44 ms/request, 2% of wall time" when the truth was ~3.8 s once plus ~4.5
            # ms/request (~0.2%).
            self._sign_ms = 0.0
            self._sign_reqs = 0
            self._sign_t0: float | None = None
            # The first .cpu() of a signing drains the NPU queue, i.e. waits for the prefill forward
            # still in flight. Draining explicitly first separates that wait from the signing's own
            # work; the total is unchanged.
            self._sign_drain_ms = 0.0
            # Phase split of signature_rows_for (BFF_STEP_PROFILE), and the per-layer projection timed
            # against the concat for the first _SIG_ALT_CHECKS calls (time and max |1 - cos| between
            # the two signatures). project_launch = until the projection call returns (host dispatch,
            # including a JL build); project_device = the wait after it (device execution).
            self._sig_phase_ms = {"project_launch": 0.0, "project_device": 0.0, "hash": 0.0,
                                  "d2h": 0.0, "encode": 0.0}
            # Cold-first test (calls _SIG_ALT_CHECKS.._SIG_ALT_CHECKS*2): the per-layer projection runs
            # FIRST on the just-written KV, the real signing right after. Measured 3.9 vs 3.0 ms on
            # GLM-5.1: a first read after the forward is not slow.
            self._sig_calls = 0
            self._sig_skip_phase = False
            self._sig_cold = {"calls": 0, "cold_ms": 0.0, "warm_after_ms": 0.0}
            self._sig_alt = {"calls": 0, "concat_ms": 0.0, "by_layer_ms": 0.0, "max_dev": 0.0,
                             "spent_ms": 0.0}
            # TP>1 piggyback: signatures signed here cannot reach the scheduler's _FF_SIGS (another
            # process), so they wait here for get_kv_connector_stats to carry them across.
            self._sig_outbox: dict = {}
            # NOTE (Update 34): this init used to be bracketed by mem probes, hunting the flat
            # 0.88 GiB by which BFF's `Available KV cache memory` trails the baseline's. It cannot be
            # here. `ensure_kv_transfer_initialized` — which constructs this object — runs in
            # NPUWorker.initialize_from_config (vllm_ascend/worker/worker.py:516), *after*
            # determine_available_memory (:327) has already sized the pool. Whatever this allocates is
            # charged to the run, never to the KV budget. The probes now bracket the profiling itself
            # (step_profile.install_mem_probe); do not re-add them here.
            super().__init__(vllm_config, engine_id, kv_cache_config)

        def start_load_kv(self, metadata):
            """Worker-side per-step entry. Stage this step's piggybacked signatures (Update 13) onto
            the recv thread BEFORE ``super()`` enqueues the requests, so a request's sigs are present
            in ``piggyback_sigs`` by the time the (serial) recv thread dequeues it. ``metadata.ff_sigs``
            crossed from the scheduler with the rest of the metadata; empty at TP>1 fallback or when
            piggyback is off, in which case this is a no-op and D asks as before."""
            # The scheduler's pool occupancy, for the dedup gate. Plain attribute rather than a
            # queue: the gate wants the latest reading, and a step that ships nothing else still
            # ships this. See _dedup_gate_closed for why a local read is not an option at TP>1.
            self._kv_usage = getattr(metadata, "kv_usage", None)
            # The scheduler's frees since the last step: at TP>1 the only way the dedup index hears
            # of them (FreedBlockLog). Bound here, before this step's forward can write a reissued
            # block, and under the engine's own lock, which a landing copy holds until its device
            # work is done — so a representative is either copied intact or seen as gone.
            freed = getattr(metadata, "freed_blocks", None)
            if freed and self._dedup_engine is not None:
                self._dedup_engine.on_blocks_freed(freed)
                self._dedup_engine.stats.freed_shipped += len(freed)
                if not MooncakeConnectorWorkerFFv2._logged_freed:
                    MooncakeConnectorWorkerFFv2._logged_freed = True
                    logger.info("BFF pull-v2: scheduler frees reach the dedup index (%d id(s) in "
                                "the first batch).", len(freed))
            if v1.SIG_PIGGYBACK and self.kv_recv_thread is not None:
                sigs = getattr(metadata, "ff_sigs", None)
                if sigs:
                    sigs = select_rank_sigs(sigs, int(getattr(self, "tp_rank", 0)))
                if sigs:
                    staged = self.kv_recv_thread.piggyback_sigs
                    staged.update(sigs)
                    # Bound the stash: entries are normally popped when the recv thread plans the
                    # request, but a gate-closed batch returns before the pop, so drop the oldest
                    # rather than let un-consumed sigs from gated batches accumulate over a long run.
                    while len(staged) > _PIGGYBACK_STASH_CAP:
                        staged.pop(next(iter(staged)))
            return super().start_load_kv(metadata)

        def register_kv_caches(self, kv_caches):
            super().register_kv_caches(kv_caches)
            # Inverse of v1's layer->group map; AliasApplier and the signature builder both want
            # group->layers. Filled IN PLACE, never reassigned: AliasApplier holds this dict by
            # reference and is built lazily, so swapping in a new object here would leave the
            # applier looking at an empty one and silently disable every scale it can place.
            self._group_layers.clear()
            for ln, gi in self._layer_group.items():
                self._group_layers.setdefault(int(gi), set()).add(ln)

            # The hybrid geometry, derived once from the tensors that were just handed over. Fatal
            # if it cannot be derived: a wrong scale signs the wrong rows and nothing notices.
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            self._layout = build_hybrid_layout(
                self._kv_cache_config,
                runner=getattr(_bp, "_ACTIVE_RUNNER", None),
                kv_caches=kv_caches,
                configured_block_size=self.vllm_config.cache_config.block_size,
                expect_mamba=model_declares_mamba(self.vllm_config))
            self._layout.log_engagement("pull-v2")

            host = self.side_channel_host
            port = self.side_channel_port + FF_PULL_V2_PORT_OFFSET + self.tp_rank
            if self.kv_role == "kv_producer":
                self._sig_server = _SigServer(host, port, self)
                self._sig_server.start()
                self._await_sig_server()
                self._warm_signatures()
                logger.info("BFF pull-v2 producer: signature precompute %s.",
                            "ON (computed in wait_for_save on the forward thread, served from cache)"
                            if PRODUCER_SIGCACHE else "OFF (computed on demand per exchange)")
            else:
                self._dedup_engine = DedupEngine()
                self._sig_client = _SigClient()
                if self.kv_recv_thread is not None:
                    self.kv_recv_thread.dedup_engine = self._dedup_engine
                    self.kv_recv_thread.sig_client = self._sig_client
                    self.kv_recv_thread.ff_copy_alias_blocks = _make_landing_copy(self)
                    self.kv_recv_thread.ff_note_failed_blocks = self.note_failed_blocks
                    if (landed_promotion.FREE_AT_PROMOTION and pd_dedup_v2.MATERIALIZE_ALIASES
                            and MATERIALIZE_ON_LANDING):
                        self.kv_recv_thread.ff_landed_sink = self.note_landed
                    # The producer's own method, pointed at THIS node's cache — that sameness is
                    # what makes the two sides' signatures comparable at all.
                    self.kv_recv_thread.local_sigs = self.signatures_for_group_split
                    self.kv_recv_thread.local_sigs_multi = self.signatures_for_groups_split
                logger.info("BFF pull-v2: decode dedup engine armed (V2_DEDUP=%s, sig timeout "
                            "%.1fs, KV-usage gate %s, sig compute dtype %s, piggyback %s).",
                            pd_dedup_v2.V2_ENABLED, SIG_EXCHANGE_TIMEOUT,
                            f">= {DEDUP_KV_THRESHOLD:.2f}" if DEDUP_KV_THRESHOLD > 0
                            else "off (dedup always on)",
                            str(pd_dedup_v2.SIG_COMPUTE_DTYPE).replace("torch.", ""),
                            "ON" if v1.SIG_PIGGYBACK else "off")
                # Unconditional so line 1 of the log states the apply mode — Variant A materialize
                # (copy into own blocks, no table write, full graph legal) vs the default redirect
                # (pointer share + free, PIECEWISE-only). A stale build or an unset flag is then
                # visible without inference, the trap the async work kept falling into.
                logger.info("BFF pull-v2 [build %s]: alias apply mode = %s%s%s.", BFF_BUILD,
                            "MATERIALIZE (copy into own blocks; no post-capture table write; "
                            "retains own blocks, no orphan free — memory NOT shared, concurrency at "
                            "baseline, by design)"
                            if pd_dedup_v2.MATERIALIZE_ALIASES else "redirect (pointer share + free)",
                            ", FULL_DECODE_ONLY permitted"
                            if (ALLOW_FULL_GRAPH and pd_dedup_v2.MATERIALIZE_ALIASES) else "",
                            ("; copied at LANDING on the recv thread, refusals re-read from P"
                             "; last block: copied at landing, hotness decided at promotion"
                             + ("; free-at-promotion ON (scheduler frees rank-agreed copies)"
                                if landed_promotion.FREE_AT_PROMOTION else
                                "; free-at-promotion off (BFF_V2_FREE_AT_PROMOTION=1 frees them)")
                             if MATERIALIZE_ON_LANDING and pd_dedup_v2.SCALE_MODE != "ratio"
                             else "; copied on the forward path, refusals recomputed")
                            if pd_dedup_v2.MATERIALIZE_ALIASES else "")
                pc = getattr(self.vllm_config, "parallel_config", None)
                if (pd_dedup_v2.V2_ENABLED and not pd_dedup_v2.MATERIALIZE_ALIASES
                        and int(getattr(pc, "tensor_parallel_size", 1) or 1) > 1):
                    logger.error(
                        "BFF pull-v2: redirect mode at TP>1 frees by ONE rank's merge map, but each "
                        "rank plans from its own signatures and nothing reconciles them — other ranks "
                        "can be left reading freed blocks. Use BFF_V2_MATERIALIZE=1 with "
                        "BFF_V2_FREE_AT_PROMOTION=1 instead.")
                if VERIFY_TRANSFER > 0:
                    logger.warning(
                        "BFF pull-v2: TRANSFER VERIFICATION on for the first %d request(s). Each "
                        "one recomputes signatures on this device after its KV lands, so this is a "
                        "diagnostic setting, not a benchmark one.", VERIFY_TRANSFER)

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
            gis = warm_groups(self._group_layers, self._skip_group)
            if not gis or not getattr(self, "num_blocks", 0):
                return
            try:
                for gi in gis:
                    _t0 = time.perf_counter()
                    self.signatures_for_group(gi, [0])
                    logger.info("BFF pull-v2: signature path warmed on group %d (%d layers) in "
                                "%.0f ms.", gi, len(self._group_layers[gi]),
                                (time.perf_counter() - _t0) * 1e3)
            except Exception as e:
                # Never fatal: a producer that cannot warm up can still try per request, and if it
                # cannot do that either the decode degrades to full reads.
                logger.warning("BFF pull-v2: signature warm-up failed (%s); the first real exchange "
                               "will pay for it instead.", e)

        # -- hybrid geometry ----------------------------------------------------------------
        # Three one-liners rather than reaching into `self._layout` at each site, because each of
        # them has a safe answer when the layout could not be built (non-hybrid fallbacks) and
        # spelling that out four times is how two of them end up different.
        def _group_scale(self, gi: int) -> int:
            return self._layout.scale(gi) if self._layout is not None else 1

        def _group_rows(self, gi: int) -> int:
            """The group's TENSOR row count — the pool's only when scale is 1.

            `self.num_blocks` (the vendored base's, taken from the FIRST layer's tensor) is an
            attention-derived figure applied to every group. On a hybrid model that is 4824 for a
            402-block pool, so every mamba group refuses and every attention group is judged
            against the wrong number depending on layer order."""
            if self._layout is not None:
                return self._layout.tensor_rows(gi)
            return self.num_blocks

        def _skip_group(self, gi: int) -> bool:
            """Groups that must never be signed: mamba state, and anything already found
            unindexable. A mamba block holds one request's recurrent state, not a token window —
            there is nothing to match, and aliasing one hands another request's state over."""
            if gi in self._unsignable:
                return True
            return self._layout is not None and self._layout.is_mamba(gi)

        # -- producer side ----------------------------------------------------------------
        def signatures_for_group(self, gi: int, block_ids):
            """Signature payload for one group's blocks, computed on demand from the registered KV.

            No forward-path hook is involved: this reads whatever is in the cache when D asks, which
            is why v2 needs none of v1's save_kv_layer accumulation."""
            gi = int(gi)
            layer_names = sorted(self._group_layers.get(gi, ()))
            if not layer_names or not block_ids or self._skip_group(gi):
                return None
            is_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
            return signatures_for_group(
                self.kv_caches, layer_names, [int(b) for b in block_ids], is_mla,
                self._jl, num_blocks=self._group_rows(gi), proj_holder=self._proj,
                scale=self._group_scale(gi))

        def signatures_for_groups_split(self, plan: dict):
            """``{group: (flat ids, lengths)}`` → ``({group: [payload per length]}, failures)``.

            **The device-to-host transfers are hoisted across GROUPS, not just slots.** That is the
            whole point of this method, and the measurement behind it: the producer logged
            ``signature path warmed on group 0 in 400 ms`` — one group — because each group's
            signatures end in three transfers, and each transfer is a full NPU queue drain on a node
            busy prefilling. Seven groups x three drains put an exchange at **5.7 s**, during which
            the decode's serial recv thread does nothing and every queued KV pull waits behind it.
            That is the ramp: BFF sat at ``Running=1, Waiting=191`` with KV at 89% while the stock
            connector held ``Waiting=0`` through its entire fill.

            Computing every group before syncing anything makes it three drains for the whole
            exchange regardless of group count. The three could be packed into one buffer for a
            further 3x, but that needs dtype punning across int64/fp16/fp32 and is not worth it.

            Slots stay apart exactly as before — the concatenation is (group-major, then slot order)
            and every row is sliced back by the same offsets it was written at.

            ``failures`` is ``[(group, exception)]`` rather than a raise, because one group whose KV
            cannot be indexed must not cost the other six their signatures; the caller counts and
            logs them the way it always has."""
            groups, sigs, norms_l = [], [], []
            failures: list[tuple[int, Exception]] = []
            is_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
            out: dict[int, list] = {}
            for gi, (block_ids, lengths) in sorted((plan or {}).items()):
                gi = int(gi)
                out[gi] = [None] * len(lengths)
                layer_names = sorted(self._group_layers.get(gi, ()))
                layers = [self.kv_caches[ln] for ln in layer_names if ln in self.kv_caches]
                if not layers or not block_ids or self._skip_group(gi):
                    continue
                try:
                    sig, norms = project_signatures(layers, [int(b) for b in block_ids], is_mla,
                                                  self._jl, num_blocks=self._group_rows(gi),
                                                  scale=self._group_scale(gi))
                except KVLayoutError as e:
                    # Sticky: the layout is a property of the allocation, so a group that cannot be
                    # indexed now will never be indexable. Retrying it costs the gather again and
                    # logs again, once per group per exchange, for the whole run.
                    self._unsignable.add(gi)
                    failures.append((gi, e))
                    continue
                except Exception as e:  # noqa: BLE001 - reported per group, never fatal to the batch
                    failures.append((gi, e))
                    continue
                if sig is None:
                    continue
                groups.append((gi, sig.shape[0], list(lengths)))
                sigs.append(sig)
                norms_l.append(norms)
            if not sigs:
                return out, failures
            # Nothing above this line touched the host. Everything below it drains the queue.
            sig_all = sigs[0] if len(sigs) == 1 else torch.cat(sigs, dim=0)
            norms_all = norms_l[0] if len(norms_l) == 1 else torch.cat(norms_l, dim=0)
            proj = pd_lsh.get_proj(self._proj, sig_all.shape[1], sig_all.device)
            # The three drains, together, once — for every group in the exchange.
            hashes = pd_lsh.sub_hashes_device(sig_all, proj).cpu().tolist()
            sig_host = sig_all.to(torch.float16).cpu()
            norms_host = norms_all.detach().float().cpu()
            # Sliced on the HOST from here on, so SignatureCodec.encode's own `.cpu()` calls are
            # no-ops rather than two more drains per slot.
            base = 0
            for gi, rows, lengths in groups:
                per_slot, off = [], base
                for n in lengths:
                    if n <= 0:
                        per_slot.append(None)
                        continue
                    per_slot.append(SignatureCodec.encode(
                        sig_host[off:off + n], norms_host[off:off + n], hashes[off:off + n]))
                    off += n
                out[gi] = per_slot
                base += rows
            return out, failures

        def signatures_for_group_split(self, gi: int, block_ids, lengths):
            """One group's signatures for SEVERAL requests at once, split back by ``lengths``.

            Same result as calling :meth:`signatures_for_group` once per request, at a fraction of
            the cost: every device-to-host transfer is hoisted out of the per-request loop.

            ``lengths`` are consumed in order and must sum to ``len(block_ids)``; the caller built
            both from the same list, so slot *i*'s rows are ``block_ids[offset:offset+lengths[i]]``.
            Returns one payload (or None) per length.

            Delegates to :meth:`signatures_for_groups_split` so there is exactly ONE encoding path:
            the two agreeing byte for byte is then true by construction rather than by test."""
            out, failures = self.signatures_for_groups_split({int(gi): (block_ids, lengths)})
            for bad_gi, exc in failures:
                if bad_gi == int(gi):
                    raise exc
            return out.get(int(gi), [None] * len(lengths))

        # -- producer signature precompute (PRODUCER_SIGCACHE, Update 9) -------------------
        def signature_rows_for(self, gi: int, block_ids):
            """Per-block signature payloads for ONE group — a 1-row ``SignatureCodec.encode`` dict per
            block, in ``block_ids`` order (None for a block that cannot be signed).

            This is the unit the producer caches by ``(gi, block_id)``; slot payloads are reassembled
            from these with :func:`assemble_sig_payload`, so a cached answer is byte-identical to the
            on-demand ``signatures_for_groups_split`` path (same ``signature_matrix`` + projections +
            3-arg encode). One device pass for the whole ``block_ids`` list, like the on-demand path.
            Never raises the per-group compute — a failure returns Nones, which the caller treats as a
            miss and reads in full."""
            ids = [int(b) for b in block_ids]
            if not ids:
                return []
            gi = int(gi)
            layer_names = sorted(self._group_layers.get(gi, ()))
            layers = [self.kv_caches[ln] for ln in layer_names if ln in self.kv_caches]
            if not layers or self._skip_group(gi):
                return [None] * len(ids)
            is_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
            prof = _PROFILE_OVERHEAD
            cold_test = False
            if prof:
                self._sig_calls += 1
                n_alt = self._SIG_ALT_CHECKS
                cold_test = n_alt < self._sig_calls <= 2 * n_alt
                self._sig_skip_phase = cold_test
                if cold_test:
                    self._sig_cold_first(layers, ids, is_mla, gi)
            t = time.perf_counter() if prof else 0.0
            try:
                sig, norms = project_signatures(layers, ids, is_mla, self._jl,
                                              num_blocks=self._group_rows(gi),
                                              scale=self._group_scale(gi))
            except KVLayoutError:
                # The docstring's promise: this path never raises, it misses. It did raise —
                # `signature_matrix` is called unguarded here — which turned a precompute for an
                # unindexable group into a failed step rather than a full read.
                self._unsignable.add(gi)
                return [None] * len(ids)
            if sig is None:
                return [None] * len(ids)
            if prof:
                t_start = t
                t = self._sig_phase("project_launch", t, sync=False)
                t = self._sig_phase("project_device", t)
                if cold_test:
                    self._sig_cold_done((t - t_start) * 1e3)
                else:
                    self._sig_alt_check(layers, ids, is_mla, gi, sig)
                t = time.perf_counter()
            proj = pd_lsh.get_proj(self._proj, sig.shape[1], sig.device)
            hashes_dev = pd_lsh.sub_hashes_device(sig, proj)
            if prof:
                t = self._sig_phase("hash", t)
            hashes = hashes_dev.cpu().tolist()
            sig_host = sig.to(torch.float16).cpu()
            norms_host = norms.detach().float().cpu()
            if prof:
                t = self._sig_phase("d2h", t)
            out = [SignatureCodec.encode(sig_host[j:j + 1], norms_host[j:j + 1], hashes[j:j + 1])
                   for j in range(len(ids))]
            if prof:
                self._sig_phase("encode", t, sync=False)
            return out

        _SIG_ALT_CHECKS = 20

        def _sig_phase(self, name: str, t0: float, sync: bool = True) -> float:
            """Charge ``now - t0`` to phase ``name`` (after draining the device, so the phase's own
            kernels land in it) and return now. BFF_STEP_PROFILE only."""
            if sync:
                _sync_device()
            now = time.perf_counter()
            if not self._sig_skip_phase:
                self._sig_phase_ms[name] += (now - t0) * 1e3
            return now

        def _sig_cold_first(self, layers, ids, is_mla, gi) -> None:
            """Run the per-layer projection BEFORE the real signing, on the KV the forward just wrote,
            and time it. BFF_STEP_PROFILE only; its result is discarded."""
            try:
                _sync_device()
                t0 = time.perf_counter()
                signature_matrix_by_layer(layers, ids, is_mla, self._jl,
                                          scale=self._group_scale(gi))
                _sync_device()
                self._sig_cold["cold_ms"] += (time.perf_counter() - t0) * 1e3
                self._sig_alt["spent_ms"] += (time.perf_counter() - t0) * 1e3
            except Exception as e:  # noqa: BLE001 - a diagnostic must never fail a signing
                logger.warning("BFF pull-v2 producer: cold-first projection test failed (%s).", e)

        def _sig_cold_done(self, warm_ms: float) -> None:
            c = self._sig_cold
            c["calls"] += 1
            c["warm_after_ms"] += warm_ms
            if c["calls"] == self._SIG_ALT_CHECKS:
                logger.info("BFF pull-v2 producer: cold-first test over %d calls — a projection run "
                            "FIRST on the just-written KV %.2f ms/call, the real one right after "
                            "%.2f ms/call (BFF_STEP_PROFILE).", c["calls"],
                            c["cold_ms"] / c["calls"], c["warm_after_ms"] / c["calls"])

        def _sig_alt_check(self, layers, ids, is_mla, gi, sig) -> None:
            """Time the per-layer projection against the concat on the same blocks, and how far apart
            the two signatures are. First _SIG_ALT_CHECKS calls only; never used for the plan."""
            alt = self._sig_alt
            if alt["calls"] >= self._SIG_ALT_CHECKS:
                return
            try:
                _sync_device()
                t0 = time.perf_counter()
                ref, _n = signature_matrix(layers, ids, is_mla, self._jl,
                                           num_blocks=self._group_rows(gi),
                                           scale=self._group_scale(gi))
                _sync_device()
                t1 = time.perf_counter()
                by_layer, _n2 = signature_matrix_by_layer(layers, ids, is_mla, self._jl,
                                                          scale=self._group_scale(gi))
                _sync_device()
                t2 = time.perf_counter()
                dev = float((1.0 - (ref * by_layer).sum(dim=1)).abs().max().item())
            except Exception as e:  # noqa: BLE001 - a diagnostic must never fail a signing
                logger.warning("BFF pull-v2 producer: per-layer projection check failed (%s).", e)
                alt["calls"] = self._SIG_ALT_CHECKS
                return
            alt["calls"] += 1
            alt["spent_ms"] += (t2 - t0) * 1e3 + 0.0
            alt["concat_ms"] += (t1 - t0) * 1e3
            alt["by_layer_ms"] += (t2 - t1) * 1e3
            alt["max_dev"] = max(alt["max_dev"], dev)
            if alt["calls"] == self._SIG_ALT_CHECKS:
                logger.info("BFF pull-v2 producer: projection over %d calls, repeated warm on the "
                            "same blocks — concat %.2f ms/call, per-layer %.2f ms/call, max |1-cos| "
                            "between them %.2e; the real path signs with %s (BFF_STEP_PROFILE).",
                            alt["calls"], alt["concat_ms"] / alt["calls"],
                            alt["by_layer_ms"] / alt["calls"], alt["max_dev"],
                            pd_dedup_v2.SIG_PROJECTION)

        def precompute_signatures(self, blocks_by_group) -> int:
            """Compute + cache per-block signatures for ``{gi: [block_ids]}`` at SAVE time.

            Overwrites existing ``(gi, block_id)`` entries — that is what keeps a recycled block id
            fresh, so the cache never serves stale KV and needs no eviction hook (see Update 9).
            Returns the count cached. Never fatal: a failed group is skipped and falls back to the
            decode's on-demand exchange."""
            if not PRODUCER_SIGCACHE or not blocks_by_group:
                return 0
            n = 0
            for gi, ids in blocks_by_group.items():
                ids = [int(b) for b in ids]
                if not ids:
                    continue
                try:
                    rows = self.signature_rows_for(int(gi), ids)
                except Exception as e:  # noqa: BLE001 - never fatal; on-demand covers the miss
                    logger.warning("BFF pull-v2 producer: precompute failed for group %s (%s).",
                                   gi, e)
                    continue
                with self._sig_cache_lock:
                    for b, row in zip(ids, rows):
                        if row is not None:
                            self._sig_row_cache[(int(gi), int(b))] = row
                            n += 1
            return n

        def drain_precompute(self, max_jobs: int = SIG_PRECOMPUTE_PER_STEP) -> int:
            """Compute + cache signatures for up to ``max_jobs`` finished requests, ON THE CALLER'S
            THREAD (the producer forward thread, from ``wait_for_save``).

            This is the whole reason it is NOT a background thread: an UNPACED background drain floods
            the producer NPU and stalls the adxl transport serving the pulls (measured: 485 transfer
            timeouts). Run from ``wait_for_save`` it is sequenced with the forward the transport
            already coexists with, and bounded per step so no single step's compute exceeds the
            transport's window. Returns the number of jobs drained."""
            n = 0
            for _ in range(max(0, int(max_jobs))):
                try:
                    job = _PRODUCER_SIG_JOBS.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.precompute_signatures(job)
                except Exception as e:  # noqa: BLE001 - a bad job is skipped, never fatal
                    logger.warning("BFF pull-v2 producer: precompute job failed (%s).", e)
                n += 1
            return n

        def precompute_and_offer(self, fuse_reqs, max_reqs: int = SIG_PRECOMPUTE_PER_STEP) -> int:
            """Piggyback (Update 13): sign the requests that finished prefill THIS step and hand the
            assembled per-request payloads to the scheduler via ``_FF_SIGS`` for stapling onto
            ``kv_transfer_params``. Called from ``wait_for_save``, AFTER ``super().wait_for_save()``,
            so the KV these ids point at has settled (the save barrier, when enabled, is what makes the
            same-step read safe — reading un-settled KV is the F1-0.198 corruption from Update 10).

            ``fuse_reqs`` is ``[(rid, ext_id, per_group_block_ids)]`` from the connector metadata — the
            requests completing this step, exactly what ``request_finished`` will ship moments later.
            Bounded per step (``max_reqs``) so the compute never exceeds the transport's window, the
            same discipline as :meth:`drain_precompute`. Group 0 (warmup) is never signed. Best-effort:
            any failure just leaves that request to D's on-demand exchange.

            **The overflow cannot be carried to the next step** (Update 27): ``request_finished``
            staples this same engine step, so a request not signed HERE ships without ``ff_sigs`` and
            is lost to the piggyback for good. That is not hypothetical — with
            ``max_num_batched_tokens=65536`` the producer completes 24-27 prefills in a step, a cap of
            8 dropped 37 % of them (``piggyback_misses=190`` of 512), and every one cost the decode a
            ~374 ms on-demand round trip IN FRONT OF ITS TRANSFER. Hence the raised default and the
            overflow counter: the bound stays (it is what keeps producer compute inside the transport's
            window), but a bound that silently drops a third of the coverage is worse than the compute
            it saves, because that compute is merely deferred to a worse moment."""
            if not v1.SIG_PIGGYBACK or not fuse_reqs:
                return 0
            if not _PROFILE_OVERHEAD:
                return self._precompute_and_offer(fuse_reqs, max_reqs)
            t0 = time.perf_counter()
            if self._sign_t0 is None:
                self._sign_t0 = t0
            _sync_device()
            t1 = time.perf_counter()
            alt_before = self._sig_alt["spent_ms"]
            try:
                return self._precompute_and_offer(fuse_reqs, max_reqs)
            finally:
                before = self._sign_reqs // 50
                self._sign_drain_ms += (t1 - t0) * 1e3
                # The projection check is a diagnostic run on top of the real signing; not its cost.
                self._sign_ms += ((time.perf_counter() - t1) * 1e3
                                  - (self._sig_alt["spent_ms"] - alt_before))
                self._sign_reqs += min(len(fuse_reqs), max(0, int(max_reqs)))
                if self._sign_reqs // 50 > before:
                    wall = (time.perf_counter() - self._sign_t0) * 1e3
                    total = self._sign_ms + self._sign_drain_ms
                    logger.info("BFF pull-v2 producer: piggyback signing %.1f ms over %d request(s) "
                                "= %.2f ms/request (%.2f waiting for the in-flight forward + %.2f "
                                "signing), %.2f%% of wall time since the first signing "
                                "(BFF_STEP_PROFILE). Signing phases per request: %s.", total,
                                self._sign_reqs, total / max(1, self._sign_reqs),
                                self._sign_drain_ms / max(1, self._sign_reqs),
                                self._sign_ms / max(1, self._sign_reqs),
                                100.0 * total / wall if wall > 0 else 0.0,
                                " ".join(f"{k} {v / max(1, self._sign_reqs):.2f} ms"
                                         for k, v in self._sig_phase_ms.items())
                                + f" (projection={pd_dedup_v2.SIG_PROJECTION}; {_alloc_stats()})")

        def _precompute_and_offer(self, fuse_reqs, max_reqs) -> int:
            """:meth:`precompute_and_offer` without the timing wrapper."""
            n = 0
            cap = max(0, int(max_reqs))
            overflow = len(fuse_reqs) - cap
            if overflow > 0:
                self._sig_precompute_overflow += overflow
                logger.warning_once(
                    "BFF pull-v2 producer: %d request(s) completed prefill this step but the "
                    "piggyback cap is %d — the rest ship without signatures and cost the decode an "
                    "on-demand exchange each. Raise BFF_V2_SIG_PRECOMPUTE_PER_STEP.",
                    len(fuse_reqs), cap)
            for (_rid, ext_id, groups) in list(fuse_reqs)[:cap]:
                payload_by_group: dict = {}
                for gi, ids in enumerate(groups):
                    if gi <= 0 or not ids:      # skip the warmup group and empty groups
                        continue
                    try:
                        rows = self.signature_rows_for(int(gi), [int(b) for b in ids])
                    except Exception as e:  # noqa: BLE001 - never fatal; D falls back to the exchange
                        logger.warning("BFF pull-v2 producer: piggyback sign failed for group %s "
                                       "(%s).", gi, e)
                        continue
                    payload = assemble_sig_payload(rows)
                    if payload is not None:
                        payload_by_group[int(gi)] = payload
                if payload_by_group:
                    if tp_group_if_sharded() is not None:
                        self._sig_outbox[ext_id] = payload_by_group
                    else:
                        v1._FF_SIGS.offer(ext_id, payload_by_group)
                    n += 1
            return n

        def serve_signatures_cached(self, per_slot) -> dict:
            """Cache-first core for the _SigServer: ``{slot: {gi: [ids]}}`` -> ``{slot: {gi: payload}}``.

            Assembles each slot payload from the ``(gi, block_id)`` cache, computing only the MISSING
            ids on demand (one pass per group, preserving the drain-batching) and back-filling. A miss
            is slower, never wrong. Hit/miss are counted BEFORE back-fill so the telemetry reflects
            what the precompute actually covered."""
            out: dict = {slot: {} for slot in per_slot}
            missing_by_group: dict = {}
            with self._sig_cache_lock:
                for groups in per_slot.values():
                    for gi, ids in (groups or {}).items():
                        rows, miss = split_cached_blocks(list(ids), self._sig_row_cache, int(gi))
                        self._sig_cache_hits += sum(1 for r in rows if r is not None)
                        self._sig_cache_misses += len(miss)
                        if miss:
                            missing_by_group.setdefault(int(gi), set()).update(miss)
            for gi, miss in missing_by_group.items():
                ids = sorted(miss)
                try:
                    rows = self.signature_rows_for(int(gi), ids)
                except Exception as e:  # noqa: BLE001 - fallback failure -> read in full
                    logger.warning("BFF pull-v2: on-demand signature fallback failed for group %s "
                                   "(%s).", gi, e)
                    rows = [None] * len(ids)
                with self._sig_cache_lock:
                    for b, row in zip(ids, rows):
                        if row is not None:
                            self._sig_row_cache[(int(gi), int(b))] = row
            with self._sig_cache_lock:
                for slot, groups in per_slot.items():
                    for gi, ids in (groups or {}).items():
                        payload = assemble_sig_payload(
                            [self._sig_row_cache.get((int(gi), int(b))) for b in ids])
                        if payload is not None:
                            out[slot][int(gi)] = payload
            return out

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
            slower, never wrong.

            Accepts a flat iterable of ids or the per-group list-of-lists that BFF stores block ids
            as, because both shapes exist in this file and ``int()`` on a list raises `TypeError`
            on the FIRST failed pull — i.e. exactly when the recompute path is what is keeping the
            run correct. Flattening costs nothing; the failure it avoids ends the run."""
            flat: set[int] = set()
            for b in block_ids:
                if isinstance(b, (list, tuple, set)):
                    flat.update(int(x) for x in b)
                else:
                    flat.add(int(b))
            with self._ff_failed_lock:
                self._ff_failed_blocks |= flat

        def note_landed(self, ext_id, landed) -> None:
            with self._ff_landed_lock:
                self._ff_landed_outbox[ext_id] = landed

        def take_landed(self) -> dict:
            with self._ff_landed_lock:
                out, self._ff_landed_outbox = self._ff_landed_outbox, {}
            return out

        def take_failed_blocks(self) -> set:
            with self._ff_failed_lock:
                out, self._ff_failed_blocks = self._ff_failed_blocks, set()
            return out

    class MooncakeConnectorFFv2(TraceDiagnostics, v1.MooncakeConnectorFF):
        """The Ascend pull connector where the DECODE decides which blocks are worth reading.

        v1's producer fusion engine, its redirect wire format and the whole resolve/hold/expire path
        on the consumer are simply not used: v2 never emits a redirect that might not resolve, so
        there is nothing to resolve, hold, or expire."""

        _WORKER_CLS = MooncakeConnectorWorkerFFv2
        _warned_piggyback_barrier = False
        _said_piggyback_skipped = False
        _said_piggyback_stats = False

        def __init__(self, vllm_config, role, kv_cache_config=None):
            super().__init__(vllm_config, role, kv_cache_config)
            # v2 does no producer forward-path work at all. Dropping v1's engine here is what
            # removes the ~14% of prefill wall time it spent clustering, and it must be dropped
            # rather than left idle: a live producer engine would keep filling the row stash that
            # nothing in v2 ever drains.
            self._ff_producer = None
            self._ff_applier = None
            self._ff_step = 0
            self._hot_collisions = 0
            self._traced: dict = {}          # BFF_V2_TRACE_SLOTS: rid -> (position, slot)
            # TP>1 only: this step's block-merge map, staged by _v2_apply for the stats carrier.
            self._ff_pending_merges = None

        def get_kv_connector_stats(self):
            """Under TP>1 the scheduler is a separate process, so this step's D-side block-merge map
            rides the connector-stats carrier; at TP=1 the in-process ``_ACTIVE_RUNNER`` channel is
            used instead and this falls through to the base.

            Without this the aliases apply, the block tables are rewritten, ``aliases_applied``
            counts up — and the scheduler frees nothing, because the map never leaves the worker
            process. That is the whole "no block free under TP=2" symptom."""
            merges = self._ff_pending_merges
            self._ff_pending_merges = None
            # The producer's piggybacked signatures, at TP>1 (see precompute_and_offer): this rank's
            # share, keyed by rank so the scheduler can staple every rank's set and each decode rank
            # can pick its own.
            sigs: dict = {}
            worker = self.connector_worker
            if worker is not None and tp_group_if_sharded() is not None:
                deferred = getattr(self, "_ff_signed_meta", None) is not getattr(
                    self, "_connector_metadata", None)
                self._ff_piggyback_sign()
                n = len(getattr(worker, "_sig_outbox", None) or {})
                if n and not MooncakeConnectorFFv2._said_piggyback_stats:
                    MooncakeConnectorFFv2._said_piggyback_stats = True
                    logger.info("BFF pull-v2 producer: piggyback: %d request(s) signed into "
                                "connector stats (deferred finalize=%s).", n, deferred)
            outbox = getattr(worker, "_sig_outbox", None) if worker is not None else None
            if outbox:
                worker._sig_outbox = {}
                rank = int(getattr(worker, "tp_rank", 0))
                sigs = {ext_id: {rank: payload} for ext_id, payload in outbox.items()}
            # Landed aliases ride this carrier at EVERY TP size: the scheduler needs all ranks' maps
            # to agree before it frees anything, and at TP=1 the one rank is simply all of them.
            landed = worker.take_landed() if worker is not None else {}
            if landed:
                rank = int(getattr(worker, "tp_rank", 0))
                landed = {ext_id: {rank: m} for ext_id, m in landed.items()}
            sharded = tp_group_if_sharded() is not None
            if ((merges or sigs) and sharded) or landed:
                data = {"bff_merges": (merges or {}) if sharded else {},
                        "bff_sigs": sigs if sharded else {}}
                if landed:
                    data["bff_landed"] = landed
                return BFFMergeStats(data=data)
            return super().get_kv_connector_stats()

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
            """No-op: signatures are computed on demand from the registered KV cache."""

        def wait_for_save(self):
            """As v1 (the optional producer write barrier), plus: under PRODUCER_SIGCACHE, drain a
            bounded batch of finished requests into the signature cache HERE — on the forward thread.

            vLLM calls this once per step in the worker's ``_get_kv_connector_output`` finally, on the
            same thread that just ran the forward. Computing the signatures here is what makes them
            coordinated with the forward the adxl transport already coexists with, instead of an
            uncoordinated background thread that floods the NPU and stalls the pulls (measured: 485
            transfer timeouts). Bounded per step so no single step overruns the transport's window."""
            super().wait_for_save()
            if PRODUCER_SIGCACHE and self.connector_worker is not None:
                try:
                    self.connector_worker.drain_precompute()
                except Exception as e:  # noqa: BLE001 - precompute must never fail the step
                    logger.warning("BFF pull-v2 producer: wait_for_save precompute drain failed "
                                   "(%s).", e)
            # Piggyback (Update 13): sign this step's just-finished requests and stage them for the
            # scheduler to staple onto kv_transfer_params. AFTER super().wait_for_save() so the save
            # barrier (when on) has settled the KV — see precompute_and_offer.
            self._ff_piggyback_sign()

        def _ff_piggyback_sign(self) -> None:
            """Sign this step's finished prefills once per connector metadata.

            Called from ``wait_for_save`` and, at TP>1, from ``get_kv_connector_stats``. With a
            speculative config vllm-ascend defers finalize (``defer_finalize=True``), so vLLM takes
            the connector stats BEFORE ``wait_for_save`` runs — signing only there filled the outbox
            one step after ``request_finished`` had already stapled an empty set (GLM-5.1 + MTP:
            ``piggyback_hits 0`` on every rank). Whichever call comes first signs; the other is a
            no-op. The forward has completed by the stats call, and group 0 (where the MTP layer's
            not-yet-written KV lives) is never signed."""
            if not claim_piggyback_sign(self, getattr(self, "_connector_metadata", None)):
                return
            sign_here = piggyback_sign_here(v1.SIG_PIGGYBACK, getattr(self, "_is_producer", False),
                                            pd_dedup_v2.V2_ENABLED)
            if (v1.SIG_PIGGYBACK and not sign_here
                    and not MooncakeConnectorFFv2._said_piggyback_skipped):
                MooncakeConnectorFFv2._said_piggyback_skipped = True
                logger.info("BFF pull-v2: piggyback signing is producer-only — skipped on this node "
                            "(producer=%s, V2_DEDUP=%s).",
                            getattr(self, "_is_producer", False), pd_dedup_v2.V2_ENABLED)
            if sign_here and self.connector_worker is not None:
                if not v1._SAVE_BARRIER and not MooncakeConnectorFFv2._warned_piggyback_barrier:
                    MooncakeConnectorFFv2._warned_piggyback_barrier = True
                    logger.warning(
                        "BFF pull-v2: BFF_V2_SIG_PIGGYBACK=1 without BFF_PD_SAVE_BARRIER=1 — the "
                        "same-step signature read may race the async KV write. Enable the save "
                        "barrier if F1 drops.")
                try:
                    fuse_reqs = getattr(self._connector_metadata, "fuse_reqs", None)
                    self.connector_worker.precompute_and_offer(fuse_reqs or [])
                except Exception as e:  # noqa: BLE001 - piggyback must never fail the step
                    logger.warning("BFF pull-v2 producer: wait_for_save piggyback sign failed "
                                   "(%s).", e)

        def request_finished_all_groups(self, request, block_ids):
            """As v1 (hand over the KV, delay the free), plus: under PRODUCER_SIGCACHE, enqueue this
            finished request's per-group block ids for the worker's precompute thread.

            ``block_ids`` is already per group (list of lists) — the same ids that go into
            ``kv_transfer_params`` — so the worker signs exactly what D will pull. Enqueue is
            best-effort: a full queue or any error just leaves those blocks to the on-demand exchange.
            This is scheduler-side; the worker drains the module queue in the same process (TP=1)."""
            res = super().request_finished_all_groups(request, block_ids)
            if PRODUCER_SIGCACHE and block_ids:
                try:
                    groups = {int(gi): [int(b) for b in g]
                              for gi, g in enumerate(block_ids) if g}
                    if groups:
                        _PRODUCER_SIG_JOBS.put_nowait(groups)
                except queue.Full:
                    pass
                except Exception as e:  # noqa: BLE001 - enqueue must never fail the request
                    logger.warning("BFF pull-v2 producer: could not enqueue precompute (%s).", e)
            return res

        @classmethod
        def requires_piecewise_for_cudagraph(cls, extra_config: dict) -> bool:
            """Always True in practice, for a DIFFERENT reason than v1's — and now a MEASURED one.

            v1 demanded PIECEWISE because it ran real Python per layer inside ``save_kv_layer``; the
            GPU v2 therefore returns False, since v2 does no forward-path work. That reasoning does
            not carry to Ascend, and ``BFF_V2_ALLOW_FULL_GRAPH=1`` was added to find out which of the
            two candidate explanations was right. **The experiment has been run and it settled the
            question against full graph.** v2 has no ``save_kv_layer`` at all, and full graph still
            produced token-level garbage from the first decoded token — F1 0.2704 against 0.4947 for
            the identical configuration under PIECEWISE, AST validity 8.61% against 15.43%.

            **Mechanism, corrected by Update 36.** It was attributed to ``update_graph_params``
            "taking ``block_table`` from the tuple frozen at capture". That is the paged branch, which
            only runs for shapes in ``pa_shape_list`` (empty by default). The branch that runs is FIA,
            and it re-reads ``attn_metadata[key].block_tables`` LIVE — but pairs ``key`` with the
            captured task by position, zipping dict order against capture order. Captures append in
            forward order (layer 0..27); ``attn_metadata`` is filled group by group, and the legacy
            warmup group ``{0,1,26,27}`` makes that ``0,1,26,27,2,3,...``. So about half the layers
            replayed against ANOTHER group's block table. That is also why every check here was clean:
            the slot trace saw 1,008,893 clean observations and the device-vs-host check no divergence,
            because the live tables were right — the wrong layer was reading them.
            ``fast_fusion_ascend_patch._patch_attn_metadata_forward_order`` now puts the metadata in
            forward order under any full-graph, multi-group run. Until an F1 run confirms it, full
            graph stays opt-in below.

            The knob is kept so the result stays reproducible, not as something to tune. Turning it
            on costs half the F1 and the failure is silent unless you read the text.

            **The downgrade has a memory price, measured (Update 35).** vLLM sizes the KV pool from a
            profile forward at ``max_num_batched_tokens``, and the attention-split graph peaks higher
            than a fused one: torch peak 8.27 GiB here against 7.39 GiB for the FULL_DECODE_ONLY
            baseline decode (the baseline's own PIECEWISE prefill also reads 8.27), weights and
            non-torch memory identical. That 0.88 GiB is the whole of BFF's smaller
            ``Available KV cache memory`` — 2.5 % of KV at util 1.0, 9.8 % at 0.5. So the answer is
            gated on ``BFF_PD_FUSE``, like v1's: with no group split there are no multiple block
            tables for a replay to mis-read, and forcing PIECEWISE would charge a stock-layout run
            that price for nothing — which is also what made a ``BFF_PD_FUSE=0`` probe look like it
            exonerated the connector when the connector was the cause."""
            if os.environ.get("BFF_PD_FUSE", "0") != "1":
                return False
            # Variant A materialization removes the post-capture block-table write entirely (aliases
            # are realised by KV copy into the request's OWN blocks, so the table never diverges from
            # what the graph captured). That is the exact mechanism the 0.2704 result was attributed
            # to, so materialize is the one mode in which permitting FULL_DECODE_ONLY is a real test
            # rather than a known-broken reproduction. It still requires the operator to opt in via
            # BFF_V2_ALLOW_FULL_GRAPH, so nothing changes silently.
            if ALLOW_FULL_GRAPH and pd_dedup_v2.MATERIALIZE_ALIASES:
                logger.warning(
                    "BFF pull-v2: FULL_DECODE_ONLY permitted under BFF_V2_MATERIALIZE — aliases are "
                    "materialised by KV copy, so no block-table write lands after graph capture. "
                    "This is the decisive full-graph test; F1 must stay ~0.47, not fall to 0.27.")
                return False
            if ALLOW_FULL_GRAPH:
                logger.warning(
                    "BFF pull-v2: BFF_V2_ALLOW_FULL_GRAPH=1 permits FULL_DECODE_ONLY. A 2026-08-26 "
                    "run gave F1 0.2704 vs 0.4947 under PIECEWISE (garbage from the first token); "
                    "Update 36 traced that to full-graph replay pairing layers with another KV-cache "
                    "group's block table, which the metadata reorder now fixes. UNVERIFIED until an "
                    "F1 run holds ~0.49 — confirm the 'BFF full-graph metadata order' line, and do not "
                    "trust a throughput-only run to show corruption.")
            return not ALLOW_FULL_GRAPH

        def _applier(self) -> "AliasApplier":
            a = getattr(self, "_ff_applier", None)
            if a is None:
                worker = self.connector_worker
                a = self._ff_applier = AliasApplier(
                    worker._dedup_engine, _write_block_table, worker.note_failed_blocks,
                    normalize_req_id=v1._ext_of,
                    group_layers=worker._group_layers,
                    # Lets the applier locate each request's write frontier and refuse to alias a
                    # block the decode has not finished writing. See pd_dedup_v2.PROTECT_HOT_BLOCKS.
                    block_size=self._block_size(),
                    # Variant A: injected so an alias can be realised by copying the representative's
                    # KV into the request's own block instead of redirecting the table. Inert unless
                    # BFF_V2_MATERIALIZE is set (pd_dedup_v2.MATERIALIZE_ALIASES).
                    copy_blocks=_make_copy_alias_blocks(worker))
            return a

        def _block_size(self) -> int:
            cache = getattr(self.connector_worker.vllm_config, "cache_config", None)
            return int(getattr(cache, "block_size", 0) or 0)

        def start_load_kv(self, forward_context, **kwargs) -> None:
            # Connector-level signature (forward_context), NOT the worker's start_load_kv(metadata).
            super().start_load_kv(forward_context, **kwargs)
            self._ff_step += 1
            if self.connector_worker is None or self.connector_worker._dedup_engine is None:
                return
            # Bounds the FORWARD-PATH share of dedup's cost directly. Everything else the mechanism
            # does runs on the recv thread, so `hook_ms_total` staying small is what says the cost
            # is over there competing for the GIL rather than in here.
            _hook_t0 = time.perf_counter()
            self._v2_apply()
            self.connector_worker._dedup_engine.stats.hook_ms += (
                time.perf_counter() - _hook_t0) * 1e3
            # Dumped from here, on the engine's own cadence. Without it this arm produces no
            # bff_stats_*.json at all and the whole verification ladder — wire saving, applied vs
            # recomputed, inert-or-not — is unreadable, which is exactly how an earlier run came back
            # as "bff stats: none found (fusion may not have engaged)".
            stats = self.connector_worker._dedup_engine.stats
            if stats.should_dump(self._ff_step):
                # Carried over here rather than counted in the engine: the client is the only thing
                # that knows how many requests rode each round trip, and the producer's own log
                # stops reporting at 100 served — which is inside the ramp, not the whole run.
                client = self.connector_worker._sig_client
                if client is not None:
                    stats.sig_batches = client.batches
                    stats.sig_batched_requests = client.batched_requests
                    stats.exchange_ms = client.exchange_ms
                # Profiler (Update 14): the decode KV occupancy per running request, once per dump, so
                # the 120-vs-198 concurrency gap is a greppable time series in the decode log next to
                # loggers.py's Running/KV samples. Comparable across the baseline / dedup=0 / dedup=1
                # arms; a rising blocks_per_running with a stalled running count localises the gap.
                try:
                    from kv_fast_fusion import fast_fusion_block_pool as _bp
                    runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                    idx = getattr(getattr(runner, "input_batch", None), "req_id_to_index", None)
                    snap = _bp.block_occupancy(len(idx) if idx is not None else 0)
                    if snap is not None:
                        logger.info("BFF pull-v2 occupancy | blocks total=%d free=%d used=%d | "
                                    "running=%d | blocks_per_running=%.1f", *snap)
                        # Update 21: split `used` into LIVE (blocks owned by RUNNING requests) vs HELD
                        # (pinned by the dedup machinery). used >> live localises the concurrency cap to
                        # the hold; the per-map counts then name which structure to release.
                        try:
                            reqs = getattr(runner, "requests", None) or {}
                            live = 0
                            per_req: dict = {}
                            for rid in (idx or {}):
                                bids = getattr(reqs.get(rid), "block_ids", None)
                                if bids:  # BFF: list of per-group id lists; stay layout-agnostic
                                    for g in bids:
                                        live += len(g) if isinstance(g, (list, tuple)) else 1
                                    per_req[rid] = bids
                            eng = getattr(self.connector_worker, "_dedup_engine", None)
                            held = eng.held_block_counts() if eng is not None else {}
                            used = snap[2]
                            # `shared` and `held` are opposite signs of the same subtraction and must
                            # be reported separately: live > used means aliasing RELEASED victims and
                            # several requests now reference one block — the mechanism working. Printed
                            # as a single signed "held" it read as `held=-2692`, which is what made a
                            # working freeing path look like a leak. Never render it signed again.
                            total = snap[0] or 1
                            shared = max(0, live - used)
                            logger.info(
                                "BFF pull-v2 occupancy split | used=%d live=%d | shared=%d (%.1f%% of "
                                "pool) held=%d | resident=%d pending_alias=%d alias_ready=%d "
                                "pending_resident=%d",
                                used, live, shared, 100.0 * shared / total, max(0, used - live),
                                held.get("resident", 0), held.get("pending_alias", 0),
                                held.get("alias_ready", 0), held.get("pending_resident", 0))
                            # Update 32: is the sharing dedup creates BATCHABLE? `redundancy` bounds
                            # the KV bandwidth a shared/unique attention split could recover;
                            # `batchable` is the part whose sharers sit at the same slot, which is
                            # the only part one shared GEMM could serve. Same walk, no extra pass.
                            sh = pd_dedup_v2.sharing_stats(per_req)
                            # Stash the PEAK-refs sample, not the most recent one. These dumps run to
                            # the end of the drain, where the handful of surviving requests give
                            # refs=300 / redundancy=0 % / max_fanout=1 — and read alone out of
                            # bff_stats that says "this run had no sharing at all" about a run whose
                            # steady state was 10-12 %. The question is what the sharing looks like at
                            # concurrency, so keep the busiest sample.
                            prev_sh = stats.sharing
                            if prev_sh is None or sh["refs"] >= prev_sh.get("refs", 0):
                                stats.sharing = sh
                            logger.info(
                                "BFF pull-v2 sharing | refs=%d distinct=%d redundancy=%.1f%% "
                                "batchable=%.1f%% | fanout %s | mean_shared=%.2f max=%d "
                                "aligned=%.1f%% | ALIGNED fanout %s max_aligned=%d | per-group %s",
                                sh["refs"], sh["distinct"], 100.0 * sh["redundancy"],
                                100.0 * sh["batchable"],
                                " ".join(f"{k}:{v}" for k, v in sh["fanout"].items()),
                                sh["mean_fanout_shared"], sh["max_fanout"],
                                100.0 * sh["aligned_frac"],
                                " ".join(f"{k}:{v}" for k, v in sh["aligned_fanout"].items()),
                                sh["max_aligned_fanout"],
                                " ".join(
                                    "g%s(batch=%.1f%%,maxal=%d)"
                                    % (g, 100.0 * v["batchable"], v["max_aligned_fanout"])
                                    for g, v in sh["per_group"].items()))
                        except Exception as e:  # noqa: BLE001 - profiling must never break the dump
                            logger.warning("BFF pull-v2: occupancy split failed (%s).", e)
                except Exception as e:  # noqa: BLE001 - profiling must never break the dump
                    logger.warning("BFF pull-v2: occupancy snapshot failed (%s).", e)
                stats.dump()

        def _v2_apply(self) -> None:
            """Apply landed aliases; see :class:`~kv_fast_fusion.pd_dedup_v2.AliasApplier`."""
            self._ff_pending_merges = None
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                applier = self._applier()
                runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                before = applier._engine.stats.applied
                applier.apply(runner)
                # apply() publishes in-process by assigning runner._updated_block_tables, which the
                # scheduler cannot see at TP>1. `pending_merges` is the same map; stage it for
                # get_kv_connector_stats to carry across the process boundary.
                self._ff_pending_merges = applier.pending_merges or None
                n = applier._engine.stats.applied - before
                if n:
                    hot = sum(applier._engine.stats.hot_block_aliases.values())
                    logger.info("BFF pull-v2 apply | aliases_applied=%d | via=%s | "
                                "materialized(cum)=%d | recompute(cum)=%d | hot-block refused(cum)=%d",
                                n, "copy" if pd_dedup_v2.MATERIALIZE_ALIASES else "redirect",
                                applier._engine.stats.materialized,
                                applier._engine.stats.recomputed, hot)
                _audit_t0 = time.perf_counter()
                self._audit_hot_blocks(runner)
                applier._engine.stats.audit_ms += (time.perf_counter() - _audit_t0) * 1e3
                self._trace_slots(runner)
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

    def _make_copy_alias_blocks(worker):
        """Build the Variant A copy callback bound to this worker's registered KV caches.

        ``copy(runner, gi, pairs) -> bool`` copies, for group ``gi``, each ``(src_rep, dst_victim)``
        block pair inside every layer of the group, so the request's own physical block ends up
        holding the representative's KV. Returns True iff the device copy was issued. Unlike the
        redirect path this writes NOTHING to the block table — the table keeps pointing at the
        request's own (now-filled) blocks — which is what lets FULL_DECODE_ONLY stay correct: there
        is no post-capture table mutation for the replayed graph to miss.

        Copies EVERY tensor of the layer's cache (:func:`cache_parts`), not just ``[0]``/``[1]``.
        It used to call ``AscendAttentionBackend.copy_blocks``, which copies exactly two: under DSA
        (GLM-5.1, ``use_sparse``) a layer's cache is ``(latent KV, rope key, indexer key[, indexer
        key scale])``, so the victim — declined, never read — kept its previous tenant's indexer
        keys, and the lightning indexer chose its top-k tokens from them while sparse attention read
        the copied KV. Defensive by construction — any layout surprise returns False so the alias
        degrades to a recompute (``note_failed_blocks``) rather than crashing the forward."""
        def _copy(runner, gi: int, pairs) -> bool:
            if not pairs:
                return False
            try:
                import torch
                layer_names = sorted(worker._group_layers.get(int(gi), ()))
                kv_caches = [worker.kv_caches[ln] for ln in layer_names if ln in worker.kv_caches]
                if not kv_caches:
                    return False
                dev = cache_list_device(kv_caches)
                # [representative, own victim] per pair.
                src = torch.tensor([int(s) for s, _d in pairs], dtype=torch.int64, device=dev)
                dst = torch.tensor([int(d) for _s, d in pairs], dtype=torch.int64, device=dev)
                for kv_cache in kv_caches:
                    for part in cache_parts(kv_cache):
                        part[dst] = part[src]
                return True
            except Exception as e:  # noqa: BLE001 - a copy that cannot run must recompute, not crash
                logger.warning("BFF pull-v2 materialize copy failed (group %s, %d pairs): %s",
                               gi, len(pairs), e)
                return False
        return _copy

    def _make_landing_copy(worker):
        """The materialize copy for the RECV thread: ``copy(gi, pairs) -> bool``, done on return.

        On its own stream, synchronised before returning, because the caller holds the dedup
        engine's lock across it (DedupEngine.release_materialized) and the forward thread takes that
        lock to invalidate freed blocks: done-on-return is what makes "copied intact or seen as gone"
        true. A dedicated stream waits only for this copy — syncing the default stream would also
        wait out the in-flight forward while the forward thread is blocked on the lock. The sources
        are prompt KV that has landed (by a host-synchronous read) and the destinations belong to a
        request that has not run, so no stream ordering is needed — with one caveat shared with the
        forward-path copy: an owner's LAST prompt block keeps receiving its decode's writes in the
        free slots, and a copy of it carries whatever those slots hold."""
        copy = _make_copy_alias_blocks(worker)

        def _landing(gi: int, pairs) -> bool:
            try:
                import torch
                stream = worker.__dict__.get("_ff_landing_stream")
                if stream is None:
                    dev = cache_list_device(list(worker.kv_caches.values()))
                    stream = worker._ff_landing_stream = torch.npu.Stream(device=dev)
                _t0 = time.perf_counter() if _PROFILE_OVERHEAD else None
                with torch.npu.stream(stream):
                    ok = copy(None, gi, pairs)
                stream.synchronize()
                if _t0 is not None and worker._dedup_engine is not None:
                    st = worker._dedup_engine.stats
                    st.landing_copy_ms = (st.landing_copy_ms or 0.0) + (
                        time.perf_counter() - _t0) * 1e3
                    st.landing_copy_calls += 1
                    st.landing_copy_pairs += len(pairs)
                return ok
            except Exception as e:  # noqa: BLE001 - a copy that cannot run is re-read instead
                logger.warning("BFF pull-v2 landing copy failed (group %s, %d pairs): %s",
                               gi, len(pairs), e)
                return False
        return _landing

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
