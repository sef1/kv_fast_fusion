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
# Check, after each apply, that no physical block sits in two live requests' write frontiers.
#
# OFF by default now: its question is closed. It found 0 collisions across 1,008,893 slot
# observations and every run since, and it is O(batch x groups) on the FORWARD PATH every step
# — ~700 inner iterations per step at con512, paid whether or not a single alias was applied. Kept
# as a knob so the check stays reproducible, not as something to leave on.
AUDIT_HOT_BLOCKS = os.environ.get("BFF_V2_AUDIT_HOT_BLOCKS", "0") == "1"
# Verify, for the first N requests, that the KV which landed on D is the KV P had.
#
# Every other check in the transfer path is structural — lengths, coverage, group reachability —
# and none of them is about CONTENT. This one recomputes D's own signatures for the blocks it just
# received and compares them, row for row, against the producer's. The JL projection is fixed-seed
# (signature_matrix), so both sides derive the same one and a correct transfer agrees to fp16
# round-trip noise; crossed layers or groups do not come close. Diagnostic only: it never plans, it
# never aliases, and a mismatch is reported rather than raised.
VERIFY_TRANSFER = int(os.environ.get("BFF_V2_VERIFY_TRANSFER", "0"))
# Seconds between checks, so the budget is spent ACROSS the run. See VerifySampler for why this
# exists: the first version spent all 32 checks on the first 32 requests, which all ran before the
# KV cache had ever filled, and reported a clean result about the wrong window.
VERIFY_SPACING_S = float(os.environ.get("BFF_V2_VERIFY_SPACING_S", "15"))
# Cosine below this (or a norm off by more than 2%) is called a mismatch. Loose enough to absorb the
# fp16 wire format and the NPU's own reduction order, tight enough that a wrong block cannot pass:
# unrelated KV blocks in these runs sit far below it, which is why dedup needs a 0.75+ bar to alias.
VERIFY_MIN_COS = float(os.environ.get("BFF_V2_VERIFY_MIN_COS", "0.99"))
# Norm-error bar. 10%, NOT the 2% this shipped with, and the difference was measured: correct blocks
# drift to ~2% norm error and ~0.991 cosine once the decode is saturated, so a 2% bar sat exactly on
# the noise floor and flagged a block at `cos 0.9956, norm err 2.0%` as corrupt while the check
# twenty seconds later passed with a LOWER cosine (0.99141). The cosine bar is what actually
# separates the populations — correct blocks 0.991+, wrong ones 0.74 — so the norm check is here to
# catch a block whose content is right but whose magnitude is not, and it should fire on 20%, not on
# drift.
VERIFY_MAX_NORM_ERR = float(os.environ.get("BFF_V2_VERIFY_MAX_NORM_ERR", "0.10"))
# How close a block must still be to ITS OWN row to be called merely degraded rather than foreign.
# Below the mismatch bar but far above the ~0.75 that unrelated KV scores here.
VERIFY_DEGRADED_COS = float(os.environ.get("BFF_V2_VERIFY_DEGRADED_COS", "0.95"))
# Trace the physical write slot of the first N requests to decode, step by step. Off by default
# because it logs once per request per step; on, it is the direct test of "new K/V is written to an
# invalid address, or to the same slot over and over".
TRACE_SLOTS = int(os.environ.get("BFF_V2_TRACE_SLOTS", "0"))
# Ceiling on how much of ONE request may be satisfied from other requests' blocks. Above it the
# request is read whole. See cap_request_decline for why a per-block cosine bar cannot express this.
MAX_REQ_DECLINE = float(os.environ.get("BFF_V2_MAX_REQ_DECLINE", "0.5"))

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
MAX_SIG_BATCH = int(os.environ.get("BFF_PULL_V2_SIG_BATCH", "8"))
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


def hot_block_collisions(per_request, block_size):
    """Physical blocks two live requests are both still writing into.

    ``per_request`` is ``{req_id: (num_computed_tokens, [per-group block id lists])}``. For each
    request the blocks from ``num_computed_tokens // block_size`` onward are its **hot** region: the
    decode writes newly generated K/V there, at ``position % block_size``. Two requests can never
    legitimately share such a block — each owns its own tail — so any overlap means both are writing
    to the same physical slots, last write wins, and attention stops seeing distinct K/V for their
    new tokens. That is the reported symptom stated as an assertion, and it is the check that can
    also *refute* the theory: a clean run says addressing is fine and the damage is substitution
    error instead.

    Returns ``{(group, block_id): [req_id, ...]}`` for the colliding blocks only, so a healthy step
    allocates nothing. Pure, so the whole audit is testable with a dict."""
    seen: dict[tuple, list] = {}
    out: dict[tuple, list] = {}
    if not block_size:
        return out
    for rid, (n_computed, groups) in per_request.items():
        hot_from = int(n_computed) // int(block_size)
        for gi, ids in enumerate(groups or ()):
            for b in list(ids)[hot_from:]:
                b = int(b)
                if b < 0:
                    continue          # the null block is shared by construction
                key = (gi, b)
                holders = seen.setdefault(key, [])
                # ONE entry per request. A single request holding the same block twice is a merge
                # inside its own table — the normal result of aliasing — not two writers racing.
                # Counting it would make the audit fire on every successful merge and bury the
                # signal it exists to carry.
                if rid in holders:
                    continue
                holders.append(rid)
                if len(holders) > 1:
                    out[key] = holders
    return out


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


def write_slot(n_computed, group_blocks, block_size):
    """The physical KV slot the request's NEXT token will be written to, or None.

    ``num_computed_tokens`` is the position of that token, so it lands at
    ``block_table[pos // block_size] * block_size + pos % block_size`` — the same arithmetic the
    attention backend's slot mapping does. Exposed as a pure function so the trace can be checked
    without an NPU, and so the check is the arithmetic itself rather than a paraphrase of it."""
    if not block_size or not group_blocks:
        return None
    idx, off = divmod(int(n_computed), int(block_size))
    if idx >= len(group_blocks):
        return None               # the block for this position has not been allocated yet
    return int(group_blocks[idx]) * int(block_size) + off


def slot_trace_fault(prev, cur, block_size):
    """Name what is wrong between two consecutive ``(n_computed, slot)`` observations, or None.

    The conjecture in its literal form. If the decode's new K/V goes to the same address over and
    over, or fails to move to the next physical block at a boundary, attention sees static K/V for
    new tokens and the model locks into a repetition loop. Three things can go wrong, and they are
    distinguishable:

    * ``slot_repeated`` — the position advanced but the slot did not: every new token overwrites the
      previous one.
    * ``slot_not_advanced_in_block`` — one token forward WITHIN a block must be exactly one slot
      forward.
    * ``block_not_advanced`` — the position crossed a block boundary but the slot stayed inside the
      same physical block, which is precisely "fails to index the next physical block".

    Only consecutive single-token steps are judged; anything else returns None rather than guessing,
    because a resumed or chunked request legitimately jumps."""
    if prev is None or cur is None or not block_size:
        return None
    (p_pos, p_slot), (c_pos, c_slot) = prev, cur
    if p_slot is None or c_slot is None or c_pos != p_pos + 1:
        return None
    if c_slot == p_slot:
        return "slot_repeated"
    if p_pos // block_size == c_pos // block_size:
        return None if c_slot == p_slot + 1 else "slot_not_advanced_in_block"
    # A boundary crossing: the new slot must be the base of a DIFFERENT physical block.
    if c_slot // block_size == p_slot // block_size:
        return "block_not_advanced"
    return None if c_slot % block_size == 0 else "slot_not_advanced_in_block"


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


def classify_mismatch(producer_moved: bool, decode_matches_new: bool) -> str:
    """Name the cause of a surviving mismatch from two follow-up comparisons.

    A mismatch on its own says only "the decode's KV does not match the signature the producer gave
    for it". That has three very different causes and they need different fixes, so the diagnostic
    re-asks the producer for the same blocks and compares three ways:

    * ``producer_moved`` — P's blocks changed between the signature and the read, and the decode
      holds P's *current* content. The transfer is faithful; the signature was stale. This is not
      only a reporting nicety: dedup decides what to skip from those same signatures, so stale ones
      mean it is aliasing against KV that has since moved.
    * ``transfer_wrong`` — P is stable and the decode still differs. The bytes that arrived are not
      the bytes P had, and the block ids in the report localise it.
    * ``producer_unstable`` — P moved AND the decode matches neither. Nothing can be concluded about
      the transfer until P holds still.
    * ``transient`` — everything agrees on the re-check, so the first read saw the block before it
      had settled. Points at the reader, not the transfer.
    """
    if producer_moved:
        return "producer_moved" if decode_matches_new else "producer_unstable"
    return "transient" if decode_matches_new else "transfer_wrong"


class VerifySampler:
    """A fixed verification budget, spent ACROSS the run rather than on the front of it.

    The first version simply counted down from N, so all N checks landed on the first N requests —
    which on a con512 run all complete during ramp-up, before the KV cache has ever reached 100%. It
    returned "4,368 blocks, 0 mismatched" and that was true, but it was evidence about the
    unsaturated phase only. A transfer fault that needs blocks to be recycled under pressure could
    not have shown up in that sample, and the phase it missed is exactly the one where the quality
    regression lives.

    Spaced in TIME, not by request index, for two reasons: saturation is a time phenomenon (KV fills
    60-80 s in, whatever the request rate), and spacing by index would need to know in advance how
    many requests the run will have.

    ``clock`` is injected so the tests are deterministic instead of sleeping."""

    def __init__(self, budget, spacing_s, clock=None):
        self.left = max(0, int(budget))
        self.spacing_s = max(0.0, float(spacing_s))
        self._clock = clock or time.monotonic
        # None rather than a timestamp: the first check is due immediately, so a short run still
        # verifies something instead of waiting out one spacing interval and checking nothing.
        self._next = None

    def ready(self) -> bool:
        """Would :meth:`take` succeed right now? Gates the *fetch*, which happens a batch early."""
        return self.left > 0 and (self._next is None or self._clock() >= self._next)

    def take(self) -> bool:
        """Claim one check, re-arming the spacing. False when spent or too soon."""
        if not self.ready():
            return False
        self.left -= 1
        self._next = self._clock() + self.spacing_s
        return True


def best_matching_row(vec, rows, skip=None):
    """``(index, cos)`` of the row in ``rows`` closest to ``vec``, or ``(-1, -1.0)`` for none.

    BEST, not first-above-a-bar: the question is *which* block this is, so a search that stopped at
    the first plausible candidate could name a neighbour that merely passed the threshold while the
    real one sat higher up. ``skip`` excludes the row we already know does not match, so a
    near-duplicate of the expected block cannot be reported as the answer to "where did it come
    from".

    Rows are the unit vectors ``signature_matrix`` produces, so the cosine is a plain dot product."""
    best_i, best_cos = -1, -1.0
    for i, row in enumerate(rows or ()):
        if skip is not None and i == skip:
            continue
        cos = sum(float(x) * float(y) for x, y in zip(vec, row))
        if cos > best_cos:
            best_i, best_cos = i, cos
    return best_i, best_cos


def cross_match(row, rows_p, rows_d, min_cos=None):
    """Name what a wrong block actually holds. Returns ``(kind, index, cos)``.

    A confirmed ``transfer_wrong`` says the decode's block *i* is not what the producer had there.
    Three things it could be instead, and they need different fixes:

    * ``source_permuted`` — the decode's block holds one of the producer's OTHER blocks, so the
      transfer read the wrong source address. ``index`` is the producer row it actually got.
    * ``destination_permuted`` — the producer's block *i* turned up in a different one of the
      decode's blocks, so the data landed at the wrong destination and that block was clobbered too.
      ``index`` is where it went.
    * ``degraded`` — it still matches its OWN row closely; only the magnitude moved. Checked first,
      because the two searches below skip this row and would otherwise answer "no other row matches"
      about a block that was essentially correct all along.
    * ``foreign`` — nothing matches. The content belongs to no row of this request: another
      request's KV, or a block reallocated mid-transfer. That points at ownership, not arithmetic.

    Source is checked first because it is the narrower claim: if the decode's block is exactly one of
    the producer's other blocks, that is a read-side address fault regardless of where anything else
    ended up."""
    bar = VERIFY_MIN_COS if min_cos is None else min_cos
    if row < 0 or row >= len(rows_d or ()) or row >= len(rows_p or ()):
        return "foreign", -1, -1.0
    # Self FIRST. The searches below deliberately skip this row, so without this a block holding its
    # own correct content — failed only on a norm error at the noise floor — finds no other row and
    # is reported as belonging to another request. That happened: `cos 0.9956, norm err 2.0%` came
    # back labelled `foreign`, which is a diagnostic being confidently wrong.
    self_cos = sum(float(x) * float(y) for x, y in zip(rows_d[row], rows_p[row]))
    if self_cos >= VERIFY_DEGRADED_COS:
        return "degraded", row, self_cos
    src_i, src_cos = best_matching_row(rows_d[row], rows_p, skip=row)
    if src_cos >= bar:
        return "source_permuted", src_i, src_cos
    dst_i, dst_cos = best_matching_row(rows_p[row], rows_d, skip=row)
    if dst_cos >= bar:
        return "destination_permuted", dst_i, dst_cos
    return "foreign", -1, max(src_cos, dst_cos)


def compare_signature_rows(sig_p, norms_p, sig_d, norms_d, min_cos=None, max_norm_err=None):
    """Row-for-row agreement between the producer's signatures and the decode's own.

    Returns ``(mismatches, worst_cos, worst_norm_err)``, where each mismatch is
    ``(row, cos, norm_err)``. Row *i* on both sides is the same block — ``align_per_group`` made the
    two lists equal-length and positionally paired — so a correct transfer agrees to fp16 noise.

    BOTH halves are compared on purpose. The vectors are unit-normalised by ``signature_matrix``, so
    cosine alone is blind to scale: a block whose KV was transferred at the wrong magnitude, or a
    partially-written block, can still point the same way. The norms carry that, and comparing them
    costs one subtraction per row.

    A length disagreement is itself the fault, not a reason to compare the overlap — it means the
    two sides do not agree about how many blocks this request has, so it is reported as row ``-1``
    rather than silently truncating to the shorter list.

    Pure Python over sequences of floats, with no torch and no NPU, so the arithmetic that decides
    "the transfer is correct" is testable on CPU like ``align_per_group`` and ``filter_sentinels``."""
    bar = VERIFY_MIN_COS if min_cos is None else min_cos
    nbar = VERIFY_MAX_NORM_ERR if max_norm_err is None else max_norm_err
    if len(sig_p) != len(sig_d):
        return [(-1, 0.0, 1.0)], 0.0, 1.0
    mismatches, worst_cos, worst_err = [], 1.0, 0.0
    for i, (a, b) in enumerate(zip(sig_p, sig_d)):
        cos = sum(float(x) * float(y) for x, y in zip(a, b))
        np_i, nd_i = float(norms_p[i]), float(norms_d[i])
        err = abs(np_i - nd_i) / max(abs(np_i), abs(nd_i), 1e-6)
        worst_cos = min(worst_cos, cos)
        worst_err = max(worst_err, err)
        if cos < bar or err > nbar:
            mismatches.append((i, cos, err))
    return mismatches, worst_cos, worst_err


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
        signature_matrix,
    )
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

    _SYNC_WARNED = False

    def _sync_device() -> None:
        """Wait for the device before reading KV that arrived by RDMA.

        VERIFICATION ONLY — never on the transfer path. The one ``torch.npu.synchronize()`` in the
        vendored receive path sits inside ``reformat_kv_cache``
        (vllm_ascend/.../kv_p2p/mooncake_connector.py) under a FIXME saying that skipping it crashes
        GQA scenarios and that upstream has not found the root cause. At tp=1 with NZ off,
        ``_reformat_after_pull`` returns before reaching it, so nothing orders the RDMA write against
        the next NPU read.

        That makes an unsynchronised read after the transfer a candidate for seeing memory that is
        not yet visible — which is exactly what this diagnostic does, so it has to rule itself out
        before its mismatches mean anything about the transfer.

        Defensive about the API: a build without ``torch.npu`` must degrade to "no barrier", not
        take the diagnostic down."""
        global _SYNC_WARNED
        try:
            npu = getattr(torch, "npu", None)
            if npu is not None:
                npu.synchronize()
        except Exception as e:  # noqa: BLE001 - a barrier that cannot run must not fail the check
            # Said once, and worth saying: without the barrier a mismatch cannot be told apart from
            # an unsynchronised read, which is the entire question this diagnostic exists to settle.
            if not _SYNC_WARNED:
                _SYNC_WARNED = True
                logger.warning("BFF pull-v2: device barrier unavailable (%s) — verification "
                               "mismatches cannot be distinguished from unsynchronised reads.", e)

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
            logger.info("BFF pull-v2 signature server: served %d request(s) in %d exchange(s) "
                        "(%.1f per exchange), %d failed%s.", self.served, self.batches,
                        self.served / self.batches if self.batches else 0.0, self.failed,
                        (" " + str(self.fail_reasons)) if self.fail_reasons else "")

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

            This is where batching pays, and the size of the prize is known: ~30% of the producer's
            per-request cost, no more. ``signatures_for_group`` ends in a device-to-host sync, so
            answering per request drained the NPU queue once per group PER REQUEST, and gathering
            every slot's blocks for a group into one call makes that once per batch. Measured
            222 ms → ~156 ms per request; the remainder is per-request gather/LSH/encode that no
            amount of batching removes. End to end it is throughput-neutral — see :class:`_SigClient`
            before spending anything more here.

            Slots are kept apart by construction: the ids are concatenated in slot order and the
            resulting rows are sliced back by the same offsets, so row *i* of a slot's payload is
            still that slot's block *i*. A group one slot did not ask about simply contributes no
            rows and is absent from its answer."""
            out: dict[int, dict] = {slot: {} for slot in per_slot}
            for gi, (slots, flat, lengths) in signature_batch_plan(per_slot).items():
                try:
                    payloads = self._worker.signatures_for_group_split(gi, flat, lengths)
                except KVLayoutError as e:
                    # Counted, not swallowed: it means the cache cannot be indexed by connector
                    # block ids, which would make every signature in the run meaningless.
                    self._note_failure("kv_layout")
                    logger.warning("BFF pull-v2: cannot index KV for group %s (%s).", gi, e)
                    continue
                except Exception as e:  # pragma: no cover - defensive
                    self._note_failure("sig_error")
                    logger.warning("BFF pull-v2: signature build failed for group %s: %s", gi, e)
                    continue
                for slot, payload in zip(slots, payloads):
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
        # The worker's signatures_for_group_split, for BFF_V2_VERIFY_TRANSFER only. The decode uses
        # the producer's own method against its OWN cache, which is what makes the two sides
        # comparable: same code, same fixed-seed projection, same group layout.
        local_sigs = None
        sig_port_offset: int = FF_PULL_V2_PORT_OFFSET
        _logged_first_decline = False
        _logged_first_cap = False
        _logged_verify_ok = False
        # Verification budget for the PROCESS, spaced across the run. Class-level because it is an
        # allowance for the run, not per thread. See VerifySampler for why it is spaced at all.
        _verify_sampler: ClassVar = VerifySampler(VERIFY_TRANSFER, VERIFY_SPACING_S)
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

        # The block ids of the request being verified right now, so the stability re-read touches
        # exactly the block that failed. Set immediately before _localise, read only by it.
        _verify_ids: ClassVar[dict] = {}

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
            self.ready_event.set()
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
                try:
                    self._prefetch_signatures(live)
                except Exception as e:  # pragma: no cover - defensive
                    # Never fatal: with no cached signatures every request is read in full.
                    logger.warning("BFF pull-v2: signature prefetch failed (%s) — this batch is "
                                   "read in full.", e)
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
            asks = [self._ask_for(req_meta) for req_meta in batch]
            keys = [self._peer_of(req_meta) if ask else None
                    for req_meta, ask in zip(batch, asks)]
            stats = self.dedup_engine.stats
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
            sigs, mismatch = claim_prefetched(
                self._sig_cache, req_meta.get("remote_request_id"), ask)
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
            if KVCacheRecvingThreadFFv2._verify_sampler.take():
                # Claimed HERE, not at verification time: the sampler decides which requests get
                # checked, and that decision has to be made while the signatures are still in hand.
                # Held for _after_transfer, because the comparison is only meaningful once the KV has
                # actually landed. plan_groups is D's ids, positionally paired with P's rows; `ask`
                # is P's ids for the same rows, kept so a failing group can be re-requested.
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
            result, capped = cap_request_decline(result)
            if capped:
                # The aliases are already staged in the engine, so dropping the plan is not enough —
                # forget() must retract them or they would be applied to blocks this request is now
                # going to fetch normally.
                self.dedup_engine.forget([ext_id])
                stats.requests_capped += 1
                if not KVCacheRecvingThreadFFv2._logged_first_cap:
                    KVCacheRecvingThreadFFv2._logged_first_cap = True
                    logger.warning(
                        "BFF pull-v2: a request would have had %d of %d blocks (%.0f%%) replaced by "
                        "other requests' KV — over the %.0f%% ceiling, so it is being read in full. "
                        "That much substitution answers a neighbouring prompt, however similar each "
                        "block is on its own. Tune with BFF_V2_MAX_REQ_DECLINE.",
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
            self._verify_transfer(req_meta)
            if self.dedup_engine is not None:
                self.dedup_engine.release(v1._ext_of(req_meta["remote_request_id"]))

        def _diagnose_mismatch(self, req_meta, gi, ask, rows_p, norms_p, rows_d, norms_d) -> str:
            """Re-ask the producer for the failing group and name the cause. See classify_mismatch.

            Only ever reached from the mismatch path, so the common case pays nothing. A second
            round trip is cheap next to being unable to tell "the transfer is broken" from "the
            signature was stale"."""
            ids = (ask or {}).get(int(gi)) or []
            peer = self._peer_of(req_meta)
            if not ids or peer is None or self.sig_client is None:
                return "unknown"
            try:
                again = (self.sig_client.ask_many(peer[0], peer[1], [{int(gi): ids}]) or [{}])[0]
                payload = (again or {}).get(int(gi))
                if payload is None:
                    return "unknown"
                sig_p2, norms_p2, _h = SignatureCodec.decode(payload)
                rows_p2 = sig_p2.tolist()
                moved, _c, _e = compare_signature_rows(rows_p2, norms_p2, rows_p, norms_p)
                vs_decode, _c2, _e2 = compare_signature_rows(rows_p2, norms_p2, rows_d, norms_d)
                return classify_mismatch(bool(moved), not vs_decode)
            except Exception as e:  # noqa: BLE001 - diagnosis must never fail a request
                logger.warning("BFF pull-v2: could not re-check group %s with the producer (%s).",
                               gi, e)
                return "unknown"

        def _localise(self, gi, row, p_rows, rows_d, stats) -> str:
            """Say what the wrong block actually holds, as a phrase to append to the report.

            Only reached on a confirmed ``transfer_wrong``, so the O(rows^2) search runs a handful of
            times in a run at most. Reuses :func:`cross_match` within the failing group first — a
            permuted source or destination there is the narrowest and most actionable answer — and
            widens to the OTHER groups only when the content is foreign to this one, which is where a
            group-index fault would show up. All groups share one JL projection (same layer count,
            same input width), so their signatures live in the same space and are comparable."""
            rows_p, _norms_p = p_rows.get(int(gi), ([], []))
            kind, idx, cos = cross_match(row, rows_p, rows_d)
            stats.verify_localised[kind] = stats.verify_localised.get(kind, 0) + 1
            if kind == "degraded":
                return (f" It still matches its OWN row at cos {cos:.4f} — the content is right and "
                        f"only the magnitude moved. This is drift, not a wrong block.")
            if kind != "foreign":
                return f" It holds this request's {kind.split('_')[0]} row {idx} (cos {cos:.4f})."
            best = (None, -1, cos)
            for other, (rows_o, _n) in p_rows.items():
                if int(other) == int(gi):
                    continue
                i, c = best_matching_row(rows_d[row], rows_o)
                if c > best[2]:
                    best = (int(other), i, c)
            if best[0] is not None and best[2] >= VERIFY_MIN_COS:
                # Reclassify: cross_match called it foreign for its own group, but it belongs to
                # another one. Move the single count rather than clearing the tally.
                stats.verify_localised["foreign"] = max(
                    0, stats.verify_localised.get("foreign", 0) - 1)
                stats.verify_localised["group_permuted"] = (
                    stats.verify_localised.get("group_permuted", 0) + 1)
                return (f" It holds GROUP {best[0]} row {best[1]} (cos {best[2]:.4f}) — the block id "
                        f"was read against the wrong group.")
            return (f" It matches no row of this request in any group (best cos {best[2]:.4f})."
                    f"{self._stability(int(gi), row, rows_d, stats)}")

        def _stability(self, gi, row, rows_d, stats) -> str:
            """Is the foreign content sitting still, or is something writing it?

            Two stories survive a foreign verdict and they need different fixes: the block was never
            written and still holds a recycled tenant's KV, or it was written and then clobbered by a
            concurrent owner. Reading it a second time separates them — a second owner keeps
            generating tokens into it, a stale tenant does not.

            The ids came from the same stash the first read used, so this re-reads exactly the block
            that failed."""
            ids = self._verify_ids.get(int(gi)) or []
            if not ids or row >= len(ids) or self.local_sigs is None:
                return " Content belongs to another request or to a block reallocated mid-transfer."
            try:
                _sync_device()
                again = self.local_sigs(int(gi), ids, [len(ids)])[0]
                if again is None:
                    return " Content belongs to another request or a reallocated block."
                sig2, _n2, _h = SignatureCodec.decode(again)
                cos = sum(float(x) * float(y) for x, y in zip(rows_d[row], sig2.tolist()[row]))
            except Exception as e:  # noqa: BLE001 - a diagnostic must never fail a request
                logger.warning("BFF pull-v2: stability re-read failed for group %s (%s).", gi, e)
                return ""
            if cos >= VERIFY_MIN_COS:
                stats.verify_localised["foreign_static"] = (
                    stats.verify_localised.get("foreign_static", 0) + 1)
                return (f" It is STATIC across a re-read (cos {cos:.4f}): never written by this "
                        f"transfer, still holding a recycled tenant's KV.")
            stats.verify_localised["foreign_changing"] = (
                stats.verify_localised.get("foreign_changing", 0) + 1)
            return (f" It CHANGED across a re-read (cos {cos:.4f}): something else is writing this "
                    f"block, so two requests own it.")

        def _verify_transfer(self, req_meta) -> None:
            """Is the KV that just landed the KV the producer had? (BFF_V2_VERIFY_TRANSFER)

            Recomputes this decode's own signatures for the blocks it received and compares them,
            row for row, against the producer's. Both sides run the same ``signature_matrix`` over
            the same group's layers with the same fixed-seed projection, so a correct transfer
            agrees to fp16 noise — and layers or groups crossed anywhere between P's cache and D's
            do not come close.

            Called after the transfer and BEFORE ``release``, so the blocks are still exactly what
            arrived: once released they may be aliased onto by later requests, and a mismatch then
            would be ambiguous between a bad transfer and a bad alias.

            **What it has measured so far.** Sampling only the first 32 requests reported 0 of 4,368
            mismatched — but those all completed during ramp-up, before the KV cache ever filled, so
            it was evidence about the wrong window (see :class:`VerifySampler`). Spread across the
            run at 98-100% KV it reported **3 of 1,518, worst cos 0.90538**, with per-check worst
            cosines drifting from 0.9998 early to 0.992-0.994 under load. That gradient is the
            interesting part: a transfer reading the WRONG block would sit near the unrelated-block
            range (mostly <0.75 in this workload, which is why dedup at a 0.75 bar finds only ~4%
            aliasable) and would not degrade smoothly with load. Partial visibility does look like
            this, which is why the barrier above goes in ahead of any conclusion, and why a
            surviving mismatch is classified rather than asserted.

            Never raises. A diagnostic that can fail a request is worse than no diagnostic."""
            rid = req_meta.get("remote_request_id")
            entry = self._verify_stash.pop(rid, None)
            if entry is None or self.local_sigs is None or self.dedup_engine is None:
                return
            ask, plan_groups, sigs = entry
            stats = self.dedup_engine.stats
            checked = mismatched = 0
            worst = 1.0
            try:
                # Decoded up front, all groups, because a cross-group source search needs the
                # producer's rows for groups other than the failing one. Cheap: unpacking a payload
                # is host work over a few hundred fp16 rows.
                p_rows = {}
                for gi, payload in sigs.items():
                    if payload is not None:
                        sp, np_, _hh = SignatureCodec.decode(payload)
                        p_rows[int(gi)] = (sp.tolist(), np_)
                for gi, payload in sorted(sigs.items()):
                    local_ids = plan_groups.get(int(gi)) or []
                    if not local_ids or payload is None:
                        continue
                    _sync_device()
                    mine = self.local_sigs(int(gi), local_ids, [len(local_ids)])[0]
                    if mine is None:
                        continue
                    sig_d, norms_d, _h2 = SignatureCodec.decode(mine)
                    rows_p, norms_p = p_rows[int(gi)]
                    rows_d = sig_d.tolist()
                    bad, worst_cos, worst_err = compare_signature_rows(
                        rows_p, norms_p, rows_d, norms_d)
                    stats.verify_checked += len(local_ids)
                    stats.verify_mismatched += len(bad)
                    stats.verify_worst_cos = min(stats.verify_worst_cos, worst_cos)
                    checked += len(local_ids)
                    mismatched += len(bad)
                    worst = min(worst, worst_cos)
                    if bad:
                        row, cos, err = bad[0]
                        verdict = self._diagnose_mismatch(
                            req_meta, int(gi), ask, rows_p, norms_p, rows_d, norms_d)
                        stats.verify_verdicts[verdict] = (
                            stats.verify_verdicts.get(verdict, 0) + 1)
                        where = ""
                        if verdict == "transfer_wrong":
                            self._verify_ids = plan_groups
                            where = self._localise(int(gi), row, p_rows, rows_d, stats)
                        logger.error(
                            "BFF pull-v2 VERIFY [%s]: group %d, %d of %d block(s) do not match the "
                            "signature the producer gave for them — first at row %d (cos %.4f, "
                            "norm err %.1f%%), decode block %s.%s",
                            verdict, gi, len(bad), len(local_ids), row, cos, 100.0 * err,
                            local_ids[row] if 0 <= row < len(local_ids) else "n/a", where)
                    elif not KVCacheRecvingThreadFFv2._logged_verify_ok:
                        # Only the first clean check is logged in full; the rest would be noise.
                        # The per-request line below still fires for every check, so the decode log
                        # shows WHEN the checks landed — which is the whole point of spacing them,
                        # and the only way to tell a run that reached saturation from one that did
                        # not.
                        KVCacheRecvingThreadFFv2._logged_verify_ok = True
                        logger.info(
                            "BFF pull-v2 VERIFY: group %d matched the producer on all %d block(s) "
                            "(worst cos %.5f, worst norm err %.2f%%).",
                            gi, len(local_ids), worst_cos, 100.0 * worst_err)
                if checked:
                    # One line per CHECK, always. Its timestamp is what says which phase of the run
                    # was sampled: correlate it against the scheduler's "GPU KV cache usage" lines
                    # to tell a check taken under saturation from one taken during ramp-up. A clean
                    # result from the ramp alone is evidence about the wrong window.
                    logger.info("BFF pull-v2 VERIFY: %d block(s) checked, %d mismatched (worst cos "
                                "%.5f) | %d check(s) left in the budget.",
                                checked, mismatched, worst,
                                KVCacheRecvingThreadFFv2._verify_sampler.left)
            except Exception as e:  # noqa: BLE001 - a diagnostic must never fail a request
                logger.warning("BFF pull-v2: transfer verification failed for %s (%s); the "
                               "request itself is unaffected.", rid, e)

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
                    # The producer's own method, pointed at THIS node's cache — that sameness is
                    # what makes the two sides' signatures comparable at all.
                    self.kv_recv_thread.local_sigs = self.signatures_for_group_split
                logger.info("BFF pull-v2: decode dedup engine armed (V2_DEDUP=%s, sig timeout "
                            "%.1fs).", pd_dedup_v2.V2_ENABLED, SIG_EXCHANGE_TIMEOUT)
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

        def signatures_for_group_split(self, gi: int, block_ids, lengths):
            """One group's signatures for SEVERAL requests at once, split back by ``lengths``.

            Same result as calling :meth:`signatures_for_group` once per request, at a fraction of
            the cost: every device-to-host transfer is hoisted out of the per-request loop.

            There are three of them per call — the SimHash bucket ids, the fp16 vectors, and the
            norms — and each is a full NPU queue drain on a producer that is busy prefilling. Paid
            per request per group they came to ~3,600 drains in a 512-request run, which is what
            made an exchange cost 211 ms and put ~108 s of the run in front of the decode's KV
            reads. Paid once per group per batch they cost 3.

            ``lengths`` are consumed in order and must sum to ``len(block_ids)``; the caller built
            both from the same list, so slot *i*'s rows are ``block_ids[offset:offset+lengths[i]]``.
            Returns one payload (or None) per length."""
            n_out = len(lengths)
            layer_names = sorted(self._group_layers.get(int(gi), ()))
            layers = [self.kv_caches[ln] for ln in layer_names if ln in self.kv_caches]
            if not layers or not block_ids:
                return [None] * n_out
            is_mla = bool(getattr(self.vllm_config.model_config, "use_mla", False))
            sig, norms = signature_matrix(layers, [int(b) for b in block_ids], is_mla,
                                          self._jl, num_blocks=self.num_blocks)
            if sig is None:
                return [None] * n_out
            proj = pd_lsh.get_proj(self._proj, sig.shape[1], sig.device)
            # The three syncs, together, once.
            hashes = pd_lsh.sub_hashes_device(sig, proj).cpu().tolist()
            sig_host = sig.to(torch.float16).cpu()
            norms_host = norms.detach().float().cpu()
            # Sliced on the HOST from here on, so SignatureCodec.encode's own `.cpu()` calls are
            # no-ops rather than a fourth and fifth drain per slot.
            out, off = [], 0
            for n in lengths:
                if n <= 0:
                    out.append(None)
                    continue
                out.append(SignatureCodec.encode(
                    sig_host[off:off + n], norms_host[off:off + n], hashes[off:off + n]))
                off += n
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
        # One ERROR for the run, not one per step: a collision repeats every step until the
        # requests finish, and the first occurrence is the one worth reading.
        _logged_hot_collision = False
        _logged_mirror_drift = False

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

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
            """No-op: signatures are computed on demand from the registered KV cache."""

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

            So the cause is the seven block tables in
            ``AscendAttentionBackendImpl.update_graph_params``, which re-reads only ``seq_lens`` per
            replay and takes ``block_table`` from the tuple frozen at capture. Note what that
            implies, and why it is invisible to every check in this file: the connector writes
            correct block ids into the LIVE table — the slot trace saw 1,008,893 clean observations
            and the device-vs-host check saw no divergence — while the replayed graph reads the
            table captured earlier. We validate a table the graph does not use.

            The knob is kept so the result stays reproducible, not as something to tune. Turning it
            on costs half the F1 and the failure is silent unless you read the text."""
            if ALLOW_FULL_GRAPH:
                logger.warning(
                    "BFF pull-v2: BFF_V2_ALLOW_FULL_GRAPH=1 permits FULL_DECODE_ONLY, which is "
                    "KNOWN BROKEN with BFF's multi-group block tables — a measured run gave F1 "
                    "0.2704 against 0.4947 under PIECEWISE, with token-level garbage from the "
                    "first decoded token. Unset it unless you are deliberately reproducing that.")
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
                    block_size=self._block_size())
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
                stats.dump()

        def _v2_apply(self) -> None:
            """Apply landed aliases; see :class:`~kv_fast_fusion.pd_dedup_v2.AliasApplier`."""
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                applier = self._applier()
                runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                before = applier._engine.stats.applied
                applier.apply(runner)
                n = applier._engine.stats.applied - before
                if n:
                    hot = sum(applier._engine.stats.hot_block_aliases.values())
                    logger.info("BFF pull-v2 apply | aliases_applied=%d | recompute(cum)=%d | "
                                "hot-block refused(cum)=%d", n,
                                applier._engine.stats.recomputed, hot)
                _audit_t0 = time.perf_counter()
                self._audit_hot_blocks(runner)
                applier._engine.stats.audit_ms += (time.perf_counter() - _audit_t0) * 1e3
                self._trace_slots(runner)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF pull-v2 consumer apply failed: %s", e)

        def _trace_slots(self, runner) -> None:
            """Follow a few requests' physical write slot step by step (BFF_V2_TRACE_SLOTS=N).

            Off by default — it logs per request per step. On, it is the check that catches an
            addressing bug which has NOTHING to do with aliasing, so a run cannot come back
            inconclusive: if the hot-block audit is clean and this is clean, new tokens are getting
            distinct K/V and the damage is substitution error instead."""
            if TRACE_SLOTS <= 0 or runner is None:
                return
            batched = getattr(getattr(runner, "input_batch", None), "req_id_to_index", None)
            if not batched:
                return
            bs = self._block_size()
            for rid in batched:
                if rid not in self._traced and len(self._traced) >= TRACE_SLOTS:
                    continue
                st = getattr(runner, "requests", {}).get(rid)
                n = getattr(st, "num_computed_tokens", None) if st is not None else None
                bids = getattr(st, "block_ids", None) if st is not None else None
                if not isinstance(n, int) or not bids:
                    continue
                # Group 1: the first fusion group, i.e. one that aliasing can actually touch.
                gi = 1 if len(bids) > 1 else 0
                self._check_mirror_matches_device(runner, rid, gi, bids[gi])
                cur = (n, write_slot(n, bids[gi], bs))
                fault = slot_trace_fault(self._traced.get(rid), cur, bs)
                self._traced[rid] = cur
                logger.info("BFF pull-v2 slot | %s g%d pos=%d slot=%s%s",
                            rid, gi, n, cur[1], f" FAULT={fault}" if fault else "")
                if fault:
                    logger.error("BFF pull-v2: %s wrote token %d to slot %s — %s. New K/V is not "
                                 "reaching a fresh address, which is what a repetition loop looks "
                                 "like from the attention's side.", rid, n, cur[1], fault)

        def _check_mirror_matches_device(self, runner, rid, gi, mirror_row) -> None:
            """The trace reads ``runner.requests[rid].block_ids[gi]``; attention reads the DEVICE
            table. ``_ff_write_runner_block_table`` writes both from the same row, so they must
            agree — but if they ever did not, every slot this trace reports would be an address the
            hardware never used, and the whole trace would be reassuring about the wrong table.

            Three views have to agree, and only the third is the one attention reads:
              * ``st.block_ids[gi]`` — the runner's Python list, what this trace's arithmetic uses;
              * ``block_table.np`` — the pinned host row, what the runner's next commit publishes;
              * ``block_table.gpu`` — the DEVICE tensor the forward indexes.

            The device read is a host-device sync, so it only runs under BFF_V2_TRACE_SLOTS and only
            until the first mismatch. Comparing the first two alone cannot see a failed publish:
            both are host-side, and a redirect that never reached the device would look perfect.

            Compared over the row's real length only: the device table is rectangular and padded to
            the widest request in the batch."""
            if MooncakeConnectorFFv2._logged_mirror_drift:
                return
            try:
                ridx = runner.input_batch.req_id_to_index.get(rid)
                bt = runner.input_batch.block_table[gi]
                n = min(len(mirror_row), int(bt.num_blocks_per_row[ridx]))
                host_row = [int(x) for x in bt.block_table.np[ridx, :n]]
                device_row = [int(x) for x in bt.block_table.gpu[ridx, :n].cpu()]
            except Exception:       # noqa: BLE001 - a diagnostic must never break the step
                return
            mirror = [int(x) for x in mirror_row[:n]]
            if mirror == host_row == device_row:
                return
            MooncakeConnectorFFv2._logged_mirror_drift = True
            which = "host table" if mirror != host_row else "DEVICE table"
            other = host_row if mirror != host_row else device_row
            bad = next((i for i, (a, b) in enumerate(zip(mirror, other)) if a != b), None)
            logger.error(
                "BFF pull-v2: %s group %d — the runner's block list and the %s disagree at position "
                "%s (%s vs %s). Attention reads the device table, so the addresses this trace "
                "reports are not the ones being written. runner=%s host=%s device=%s",
                rid, gi, which, bad,
                mirror[bad] if bad is not None else "?", other[bad] if bad is not None else "?",
                mirror[:8], host_row[:8], device_row[:8])

        def _audit_hot_blocks(self, runner) -> None:
            """After applying: is any physical block in two live requests' write frontiers?

            The direct test of "attention is no longer receiving distinct K/V for newly generated
            tokens". Two requests writing their new tokens into the same physical block overwrite
            each other slot for slot, so both read whichever wrote last, and the model locks into a
            repetition loop. Distinct requests never legitimately share a hot block, so this has no
            false-positive mode — and a clean run refutes the theory rather than merely failing to
            confirm it.

            Runs AFTER apply, deliberately: the question is whether the block tables we just wrote
            put two requests on the same slots. Costs one pass over each request's last block or
            two."""
            if not AUDIT_HOT_BLOCKS or runner is None:
                return
            batched = getattr(getattr(runner, "input_batch", None), "req_id_to_index", None)
            if not batched:
                return
            per_request = {}
            for rid in batched:
                st = getattr(runner, "requests", {}).get(rid)
                bids = getattr(st, "block_ids", None) if st is not None else None
                n = getattr(st, "num_computed_tokens", None) if st is not None else None
                if bids is not None and isinstance(n, int):
                    per_request[rid] = (n, bids)
            hits = hot_block_collisions(per_request, self._block_size())
            if not hits:
                return
            self._hot_collisions += len(hits)
            if not MooncakeConnectorFFv2._logged_hot_collision:
                MooncakeConnectorFFv2._logged_hot_collision = True
                (gi, blk), rids = next(iter(hits.items()))
                logger.error(
                    "BFF pull-v2: %d physical block(s) are in the write frontier of MORE THAN ONE "
                    "live request — e.g. group %d block %d shared by %s. Both will write their "
                    "newly generated K/V to the same slots, so neither sees its own. This is KV "
                    "corruption, not a compression trade-off.", len(hits), gi, blk, sorted(rids))

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
