"""Diagnostics for the Ascend pull connector. All off by default, none on the normal path.

Moved out of ``connectors/mooncake_connector_ff_v2.py`` unchanged, with the same flags and defaults,
so that file reads as the transfer path. They stay because they found real bugs. The pure functions
are CPU-testable. The two mixins are inherited by the connector classes and read the state those
classes own (``local_sigs``, ``last_descriptors``, ``sig_client``, ``_traced`` and so on).

    BFF_V2_VERIFY_TRANSFER   compare landed KV against the producer's signatures
    BFF_XFER_AUDIT           re-issue a transfer and look for blocks that changed
    BFF_V2_TRACE_SLOTS       follow requests' physical write slots step by step
    BFF_V2_AUDIT_HOT_BLOCKS  look for a block in two live requests' write frontiers
"""

import os
import time
from typing import ClassVar

from vllm.logger import init_logger

try:
    import torch

    from kv_fast_fusion.pd_dedup_v2 import SignatureCodec
except ImportError:  # pragma: no cover - the pure helpers stay importable without torch
    torch = None
    SignatureCodec = None

logger = init_logger("vllm.mooncake_connector_ff_v2_ascend")

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
# SELF-consistency audit: 0 off, N audits every Nth request. Transfer, snapshot this decode's own
# signatures, re-issue the same descriptors, snapshot again — a block that CHANGED was written wrong
# the first time.
#
# The point is that it never contacts the producer. VERIFY_TRANSFER's blocking exchange costs 2-5 s
# per request on the serial recv thread, so it checked ~11 requests a run and found 2-6 events: it
# proved the bug is real and can say nothing about its shape. This covers every block of an audited
# request (~161 of them at 7 groups) for a re-transfer and four device drains, which is the volume a
# rate and a distribution actually need.
XFER_AUDIT = int(os.environ.get("BFF_XFER_AUDIT", "0"))
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


def _sync_device() -> None:
    """Wait for the device before reading KV that arrived by RDMA.

    VERIFICATION ONLY — never on the DECODE's transfer path. There is now a second barrier in
    this connector, ``v1._sync_producer_writes``, and it is neither a diagnostic nor on this
    side: it drains the PRODUCER's queue in ``wait_for_save`` so the decode cannot read KV the
    producer has not written yet. Different node, different race, different switch
    (``BFF_PD_SAVE_BARRIER``). Do not merge them.

    The one ``torch.npu.synchronize()`` in the
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


class TransferDiagnostics:
    """Did the KV that landed match what the producer had? Mixed into the recv thread."""

    _logged_verify_ok = False
    # Verification budget for the PROCESS, spaced across the run. Class-level because it is an
    # allowance for the run, not per thread. See VerifySampler for why it is spaced at all.
    _verify_sampler: ClassVar = VerifySampler(VERIFY_TRANSFER, VERIFY_SPACING_S)
    # Requests seen by the audit, for its every-Nth stride. On the class, not the instance: the
    # stride should thin the audit across the node, not once per recv thread.
    _audit_seen = 0

    def _audit_self_consistency(self, req_meta) -> None:
        """Did this request's KV change when the identical descriptors were issued again?

        A block that changed was written wrong the first time. That is the whole check, and it
        needs **no producer round trip** — which is the only reason it can run at volume where
        :meth:`_verify_transfer` cannot: that one blocks the serial recv thread for 2-5 s asking
        P for signatures, so it audited ~11 requests a run and found 2-6 events. Enough to prove
        the loss is real; nothing at all about its shape.

        It cannot say WHICH of the two reads was wrong, only that they disagree, so it measures
        ``P(disagree) ~ 2p`` — a known factor of two. ``_replay`` already established the
        mechanism (five re-issues across two runs, every one delivering the producer's exact
        KV), so the direction is not in question and only the rate and the distribution are.

        Runs before ``release``, like the verification, so the blocks are still exactly what
        arrived rather than something a later request has aliased onto.

        Never raises. Note it DOES re-issue into live KV: that is safe precisely because a
        re-issue delivers the correct bytes, and it leaves the block at least as correct as it
        found it. Never gated on ``VERIFY_TRANSFER`` — the two are independent."""
        if XFER_AUDIT <= 0 or self.dedup_engine is None:
            return
        TransferDiagnostics._audit_seen += 1
        if TransferDiagnostics._audit_seen % XFER_AUDIT:
            return
        descs = self.last_descriptors or {}
        if not descs or self.last_session_id is None:
            return
        stats = self.dedup_engine.stats
        t0 = time.perf_counter()
        try:
            by_group: dict[int, list[int]] = {}
            for gi, lb in descs:
                by_group.setdefault(int(gi), []).append(int(lb))
            # Sorted so row i of a group's signature is block by_group[gi][i] on BOTH snapshots.
            for ids in by_group.values():
                ids.sort()
            plan = {gi: (ids, [len(ids)]) for gi, ids in by_group.items() if ids}
            if not plan:
                return
            _sync_device()
            first = self._recompute_local(plan)
            srcs, dsts, lens = [], [], []
            for segs in descs.values():
                for d in segs:
                    srcs.append(d[0])
                    dsts.append(d[1])
                    lens.append(d[2])
            if self.engine.batch_transfer_sync_read(
                    self.last_session_id, srcs, dsts, lens) < 0:
                return
            _sync_device()
            second = self._recompute_local(plan)
        except Exception as e:  # noqa: BLE001 - an audit must never fail a request
            logger.warning("BFF pull-v2: self-consistency audit failed (%s).", e)
            return
        finally:
            stats.audit_ms_self += (time.perf_counter() - t0) * 1e3
        self._score_audit(by_group, first, second, stats)

    def _score_audit(self, by_group, first, second, stats) -> None:
        """Compare the two snapshots block by block and bucket EVERY block, bad or not."""
        buckets = self._audit_buckets
        stats.audit_requests += 1
        bad_here = 0
        for gi, ids in sorted(by_group.items()):
            a, b = first.get(gi), second.get(gi)
            if a is None or b is None:
                continue
            rows_a = SignatureCodec.decode(a)[0].tolist()
            rows_b = SignatureCodec.decode(b)[0].tolist()
            for row, block_id in enumerate(ids):
                if row >= len(rows_a) or row >= len(rows_b):
                    break
                cos = sum(float(x) * float(y) for x, y in zip(rows_a[row], rows_b[row]))
                changed = cos < VERIFY_MIN_COS
                stats.audit_blocks += 1
                stats.audit_worst_cos = min(stats.audit_worst_cos, cos)
                if changed:
                    stats.audit_blocks_bad += 1
                    bad_here += 1
                buckets.add(self._bucket_for(gi, block_id), changed)
        if bad_here:
            stats.audit_requests_bad += 1
        stats.audit_buckets = buckets.as_dict()

    @property
    def _audit_buckets(self):
        """Per-instance, accumulated across the run and mirrored into stats after each request.

        Kept here rather than rebuilt from ``stats.audit_buckets`` every time: the stats dict is
        a report, and reading a report back in as state is how a denominator gets lost."""
        b = self.__dict__.get("_audit_buckets_obj")
        if b is None:
            from kv_fast_fusion_ascend.connectors import mooncake_connector_ff as v1

            b = self.__dict__["_audit_buckets_obj"] = v1.AuditBuckets()
        return b

    def _bucket_for(self, gi, block_id) -> str:
        """Where this block sits inside its registered region, or the group if that is unknown.

        BFF registers 8 regions of 4.22 GiB against the stock connector's 56 of 618 MiB, so only
        BFF's addressing crosses 2^32 — the most obvious structural difference between the two.
        The naive version is already disconfirmed (failing ids span both sides of the boundary),
        which is why this feeds a rate-per-bucket rather than a list."""
        rm = self.region_map
        descs = (self.last_descriptors or {}).get((int(gi), int(block_id)))
        if rm is None or not descs:
            return f"group{int(gi)}"
        return rm.bucket(descs[0][0])

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
                f"{self._stability(int(gi), row, rows_p, rows_d, stats)}")

    def _replay(self, gi, block_id, rows_p, rows_d, row, stats) -> str:
        """Re-issue the failing block's OWN descriptor, then look again.

        The last question the passive checks cannot answer. Two causes survive a static-foreign
        verdict and they have different owners:

        * ``retry_matches_producer`` — the same descriptor delivers exactly the producer's KV on
          a second attempt, so the first one was lost inside the engine. Upstream, and retry is a
          plausible mitigation. Judged against the PRODUCER, not merely against the old bad
          content: "it changed" is not "it is now right", and reporting the weaker comparison as
          the stronger one would overstate the finding.
        * ``retry_changed_still_wrong`` — it moved but landed on neither. Nothing in this model
          explains that, so it is counted separately rather than folded into a success.
        * ``retry_same`` — the identical foreign content arrives again, deterministically. Then
          the remote address does not point at the block we believe it does, and the arithmetic
          is OURS. The cross-match cannot see this case: it compares only against the rows the
          producer told us about, so KV fetched from another request's block on P is "foreign"
          by construction.

        The descriptor is logged either way. If the arithmetic is wrong, that line shows it
        without needing the round trip at all."""
        descs = (self.last_descriptors or {}).get((int(gi), int(block_id)))
        if not descs or self.last_session_id is None:
            return " Content is another request's KV or a reallocated block."
        shown = descs[0]
        note = (f" Descriptor: local=0x{shown[0]:x} remote=0x{shown[1]:x} len={shown[2]} "
                f"({len(descs)} address(es) for this block).")
        try:
            srcs = [d[0] for d in descs]
            dsts = [d[1] for d in descs]
            lens = [d[2] for d in descs]
            ret = self.engine.batch_transfer_sync_read(
                self.last_session_id, srcs, dsts, lens)
            if ret < 0:
                return note + f" Replay itself failed (ret={ret})."
            _sync_device()
            again = self.local_sigs(int(gi), self._verify_ids.get(int(gi)) or [],
                                    [len(self._verify_ids.get(int(gi)) or [])])[0]
            if again is None:
                return note + " Replay produced no signature to compare."
            sig2, _n, _h = SignatureCodec.decode(again)
            fresh = sig2.tolist()[row]
            # Two comparisons, not one. Against the OLD bad content says whether anything moved;
            # against the PRODUCER says whether it moved to the right place. Only the second
            # licenses "the write was lost" — the first was being reported as if it did.
            cos = sum(float(x) * float(y) for x, y in zip(rows_d[row], fresh))
            cos_p = sum(float(x) * float(y) for x, y in zip(rows_p[row], fresh))
        except Exception as e:  # noqa: BLE001 - a diagnostic must never fail a request
            logger.warning("BFF pull-v2: descriptor replay failed for group %s (%s).", gi, e)
            return note
        if cos >= VERIFY_MIN_COS:
            stats.verify_localised["retry_same"] = (
                stats.verify_localised.get("retry_same", 0) + 1)
            return note + (f" Replaying it returned the SAME content (cos {cos:.4f}) — the "
                           f"remote address does not point at the intended block. Ours.")
        if cos_p >= VERIFY_MIN_COS:
            stats.verify_localised["retry_matches_producer"] = (
                stats.verify_localised.get("retry_matches_producer", 0) + 1)
            return note + (f" Replaying it delivered exactly the producer's KV (cos "
                           f"{cos_p:.4f}) — the first write was LOST, not misaddressed. "
                           f"Upstream.")
        stats.verify_localised["retry_changed_still_wrong"] = (
            stats.verify_localised.get("retry_changed_still_wrong", 0) + 1)
        return note + (f" Replaying it changed the block (cos {cos:.4f} against the old "
                       f"content) but it still does not match the producer (cos {cos_p:.4f}). "
                       f"Neither a lost write nor a wrong address explains that.")

    def _stability(self, gi, row, rows_p, rows_d, stats) -> str:
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
            # STATIC does NOT mean "never written". A block written correctly and then
            # overwritten ONCE by another owner reads exactly the same. Replaying its own
            # descriptor is what separates a lost write from a wrong address.
            return (f" It is STATIC across a re-read (cos {cos:.4f})."
                    f"{self._replay(gi, ids[row], rows_p, rows_d, row, stats)}")
        stats.verify_localised["foreign_changing"] = (
            stats.verify_localised.get("foreign_changing", 0) + 1)
        return (f" It CHANGED across a re-read (cos {cos:.4f}): something else is writing this "
                f"block, so two requests own it.")

    def _recompute_local(self, plan: dict) -> dict:
        """``{group: (ids, [len(ids)])}`` → ``{group: payload}``, in one pass over the device.

        Prefers the cross-group entry point. The per-group fallback exists so a worker that
        binds only ``local_sigs`` still verifies — it is correct, just back to one drain per
        group, which is what this whole change is removing."""
        if not plan:
            return {}
        multi = self.local_sigs_multi
        if multi is not None:
            out, _failures = multi(plan)
            return {gi: (rows[0] if rows else None) for gi, rows in out.items()}
        return {gi: self.local_sigs(gi, ids, lengths)[0]
                for gi, (ids, lengths) in plan.items()}

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
            # ONE barrier and ONE recompute for the whole request, not one of each per group.
            # Both are NPU queue drains on this decode's serial recv thread while the device is
            # generating, so per group they cost 7 barriers + 21 transfers — around 2.6 s per
            # verified request, which is a diagnostic that changes the run it is measuring.
            # Hoisted, they cost 1 + 3 regardless of group count.
            _sync_device()
            mine_by_group = self._recompute_local(
                {int(gi): (plan_groups.get(int(gi)) or [],
                           [len(plan_groups.get(int(gi)) or [])])
                 for gi, payload in sigs.items()
                 if payload is not None and plan_groups.get(int(gi))})
            for gi, payload in sorted(sigs.items()):
                local_ids = plan_groups.get(int(gi)) or []
                if not local_ids or payload is None:
                    continue
                mine = mine_by_group.get(int(gi))
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
                elif not TransferDiagnostics._logged_verify_ok:
                    # Only the first clean check is logged in full; the rest would be noise.
                    # The per-request line below still fires for every check, so the decode log
                    # shows WHEN the checks landed — which is the whole point of spacing them,
                    # and the only way to tell a run that reached saturation from one that did
                    # not.
                    TransferDiagnostics._logged_verify_ok = True
                    logger.info(
                        "BFF pull-v2 VERIFY: group %d matched the producer on all %d block(s) "
                        "(worst cos %.5f, worst norm err %.2f%%).",
                        gi, len(local_ids), worst_cos, 100.0 * worst_err)
            if checked:
                # Once per REQUEST, whatever the block count. A request with three bad blocks is
                # one damaged answer, not three.
                stats.verify_requests += 1
                if mismatched:
                    stats.verify_requests_bad += 1
            if checked and self.last_segment_count:
                # Logged per VERIFIED request, not once per process. The first request of a run
                # has the freshest, least fragmented block pool — it measured 56 segments over
                # 119 blocks — while the failures land minutes later under churn, where the run
                # count and so the segment count are higher. Any batch-size argument needs the
                # number from where it breaks, not from where it is smallest.
                logger.info("BFF pull-v2 VERIFY: this request used %d transfer segment(s).",
                            self.last_segment_count)
            if checked:
                # One line per CHECK, always. Its timestamp is what says which phase of the run
                # was sampled: correlate it against the scheduler's "GPU KV cache usage" lines
                # to tell a check taken under saturation from one taken during ramp-up. A clean
                # result from the ramp alone is evidence about the wrong window.
                logger.info("BFF pull-v2 VERIFY: %d block(s) checked, %d mismatched (worst cos "
                            "%.5f) | %d check(s) left in the budget.",
                            checked, mismatched, worst,
                            TransferDiagnostics._verify_sampler.left)
        except Exception as e:  # noqa: BLE001 - a diagnostic must never fail a request
            logger.warning("BFF pull-v2: transfer verification failed for %s (%s); the "
                           "request itself is unaffected.", rid, e)


class TraceDiagnostics:
    """Are new tokens reaching distinct physical slots? Mixed into the connector."""

    # One ERROR for the run, not one per step: a collision repeats every step until the
    # requests finish, and the first occurrence is the one worth reading.
    _logged_hot_collision = False
    _logged_mirror_drift = False

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
        if TraceDiagnostics._logged_mirror_drift:
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
        TraceDiagnostics._logged_mirror_drift = True
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
        if not TraceDiagnostics._logged_hot_collision:
            TraceDiagnostics._logged_hot_collision = True
            (gi, blk), rids = next(iter(hits.items()))
            logger.error(
                "BFF pull-v2: %d physical block(s) are in the write frontier of MORE THAN ONE "
                "live request — e.g. group %d block %d shared by %s. Both will write their "
                "newly generated K/V to the same slots, so neither sees its own. This is KV "
                "corruption, not a compression trade-off.", len(hits), gi, blk, sorted(rids))
