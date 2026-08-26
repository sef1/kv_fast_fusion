"""Unit tests for the Mooncake BFF v2 connector (decode decides what to pull).

The point of v2 is that a deduplicated block is **never requested**, so the saving is on the wire
and on the producer's KV residency — the resource that was actually scarce. Two things have to hold
for that to be safe rather than merely smaller, and both are pinned here:

* the pull request the producer receives must still pair every surviving block with the SAME source
  block it would have got with no dedup (`_build_transfer_params` aligns positionally, so this is a
  silent-corruption hazard, not a performance detail);
* a block that is aliased is never written, so an alias that cannot be applied has to become a
  recompute — the opposite polarity to v1, where a failed apply was harmless.

CPU only; no Transfer Engine, no GPU, no P/D topology.
"""

import types

import pytest
import torch

from kv_fast_fusion import pd_dedup_v2, pd_lsh
from kv_fast_fusion.connectors import mooncake_connector_ff_v2 as v2

requires_mooncake = pytest.mark.skipif(
    not hasattr(v2, "MooncakeConnectorWorkerFFv2"), reason="mooncake stack unavailable")


def _worker():
    w = v2.MooncakeConnectorWorkerFFv2.__new__(v2.MooncakeConnectorWorkerFFv2)
    import threading

    from kv_fast_fusion.pd_dedup_v2 import DedupEngine
    w._jl = [None]
    # Both projection caches must be per-worker and must OUTLIVE the call. Passing a throwaway
    # `[None]` to pd_lsh.get_proj rebuilds a fixed-seed matrix on every group of every signature
    # request, on the producer's critical path — see test_the_projection_is_cached_across_calls.
    w._proj = [None]
    w._ff_lock = threading.Lock()
    # The decision state moved into the shared, transport-free engine (pd_dedup_v2), which the
    # Ascend connectors use too; the worker is now transport only.
    w._engine = DedupEngine(lock=w._ff_lock)
    w._ff_failed_blocks = set()
    w._group_layers = {1: {"l0"}}
    return w


def _kv(vectors):
    """FlashAttention-shaped KV [2, nblocks, block, heads, dim]; block 0 stays the null block."""
    n = len(vectors)
    k = torch.zeros(n + 1, 1, 1, len(vectors[0]))
    for i, v in enumerate(vectors):
        k[i + 1, 0, 0] = torch.tensor(v, dtype=torch.float32)
    return torch.stack([k, k.clone()])


def _payload(rows):
    m = torch.tensor(rows, dtype=torch.float32)
    norms = m.norm(dim=1).clamp(min=1e-6)
    sig = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    return v2.SignatureCodec.encode(sig, norms, pd_lsh.sub_hashes(sig, proj))


# =====================================================================================
# signatures
# =====================================================================================
def test_signature_roundtrip_preserves_the_cosine_decision():
    """fp16 on the wire has to stay faithful enough to verify a merge, since the cosine IS the
    merge decision — a signature that drifts changes what gets merged."""
    torch.manual_seed(3)
    raw = torch.randn(6, 64)
    sig = raw / raw.norm(dim=1, keepdim=True)
    norms = raw.norm(dim=1)
    proj = pd_lsh.get_proj([None], 64, sig.device)
    hashes = pd_lsh.sub_hashes(sig, proj)

    got_sig, got_norms, got_hashes = v2.SignatureCodec.decode(
        v2.SignatureCodec.encode(sig, norms, hashes))

    assert got_hashes == hashes
    assert got_norms == pytest.approx(norms.tolist(), rel=1e-6)
    before = sig @ sig.T
    after = got_sig @ got_sig.T
    assert torch.allclose(before, after, atol=2e-3), "fp16 must not move any pairwise cosine"


@requires_mooncake
def test_producer_computes_signatures_without_a_forward_hook():
    """v2's producer reads the registered KV cache on demand, which is why it needs no
    save_kv_layer work and no PIECEWISE cudagraph constraint."""
    w = _worker()
    w.device_kv_caches = {"l0": _kv([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])}

    out = w.signatures_for({1: [1, 2]})

    assert set(out) == {1}
    sig, norms, hashes = v2.SignatureCodec.decode(out[1])
    assert sig.shape[0] == 2 and len(norms) == 2 and len(hashes) == 2
    assert torch.allclose(sig.norm(dim=1), torch.ones(2), atol=1e-3), "signatures arrive normalised"


@requires_mooncake
def test_signatures_skip_groups_with_no_blocks_or_no_layers():
    w = _worker()
    w.device_kv_caches = {"l0": _kv([[1.0, 0.0]])}
    assert w.signatures_for({1: []}) == {}
    assert w.signatures_for({7: [1]}) == {}, "group with no registered layers"


# =====================================================================================
# planning the pull
# =====================================================================================
@requires_mooncake
def test_a_duplicate_is_dropped_from_the_pull_request():
    """The whole point: the second copy is never asked for, so its bytes never cross the wire."""
    w = _worker()
    v = [1.0, 0.0, 0.5, 0.25]
    req_blocks = {"rA": [[], [41]], "rB": [[], [42]]}
    sigs = {"rA": {1: _payload([v])}, "rB": {1: _payload([v])}}

    planned = w.plan_pull(req_blocks, sigs, threshold=0.75)

    assert planned["rA"][1] == [41], "the first copy is still pulled"
    assert planned["rB"][1] == [v2._SENTINEL], "the second is not requested"
    assert w.take_pending_alias("rB") == {1: {42: (41, "rA", None)}}
    assert w._engine.stats.dropped_batch == {1: 1} and w._engine.stats.planned == {1: 2}


@requires_mooncake
def test_a_dropped_block_keeps_its_position_in_the_request():
    """The corruption guard. `_build_transfer_params` pairs P's blocks with D's POSITIONALLY, so a
    dropped block must leave a hole, not shorten the list — otherwise every block after it would be
    filled from the wrong source."""
    w = _worker()
    dup = [1.0, 0.0, 0.0, 0.0]
    rows = [[0.0, 1.0, 0.0, 0.0], dup, [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    req_blocks = {"rA": [[], [40]], "rB": [[], [50, 51, 52, 53]]}
    sigs = {"rA": {1: _payload([dup])}, "rB": {1: _payload(rows)}}

    planned = w.plan_pull(req_blocks, sigs, threshold=0.75)

    assert planned["rB"][1] == [50, v2._SENTINEL, 52, 53], "same length, hole in the middle"
    assert w.take_pending_alias("rB") == {1: {51: (40, "rA", None)}}


@requires_mooncake
def test_the_sentinel_leaves_every_surviving_pair_exactly_as_it_was():
    """End to end through the real pairing code: the blocks that ARE pulled must be paired with the
    same producer blocks they would have been paired with had nothing been deduplicated."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        group_concurrent_contiguous,
    )
    p_blocks = [900, 901, 902, 903]
    d_full = [50, 51, 52, 53]
    d_sent = [50, v2._SENTINEL, 52, 53]

    kept = [i for i, b in enumerate(d_sent) if b >= 0]
    src, dst = group_concurrent_contiguous([p_blocks[i] for i in kept],
                                           [d_sent[i] for i in kept])
    got = dict(zip([b for run in src for b in run], [b for run in dst for b in run]))

    full_src, full_dst = group_concurrent_contiguous(p_blocks, d_full)
    want = dict(zip([b for r in full_src for b in r], [b for r in full_dst for b in r]))

    assert got == {900: 50, 902: 52, 903: 53}
    assert all(want[p] == d for p, d in got.items()), "a survivor changed source block"


@requires_mooncake
def test_dissimilar_blocks_leave_the_pull_request_untouched():
    """No match must mean a byte-identical request to what v1/vanilla would have sent."""
    w = _worker()
    req_blocks = {"rA": [[], [41]], "rB": [[], [42]]}
    sigs = {"rA": {1: _payload([[1.0, 0.0, 0.0, 0.0]])},
            "rB": {1: _payload([[0.0, 1.0, 0.0, 0.0]])}}

    planned = w.plan_pull(req_blocks, sigs, threshold=0.75)

    assert planned == {"rA": [[], [41]], "rB": [[], [42]]}
    assert w.take_pending_alias("rB") is None


@requires_mooncake
def test_groups_without_signatures_are_passed_through_verbatim():
    """Only fused groups get signatures; group 0 (the warmup/sliding-window group) must be pulled
    exactly as the scheduler allocated it."""
    w = _worker()
    req_blocks = {"rA": [[7, 8], [41]]}
    planned = w.plan_pull(req_blocks, {"rA": {1: _payload([[1.0, 0.0]])}}, threshold=0.75)
    assert planned["rA"][0] == [7, 8], "unplanned group untouched"


@requires_mooncake
def test_a_block_count_mismatch_pulls_the_request_in_full():
    """P and D disagreeing is a bug, not a merge opportunity: never guess at the alignment."""
    w = _worker()
    req_blocks = {"rA": [[], [41, 42]]}
    planned = w.plan_pull(req_blocks, {"rA": {1: _payload([[1.0, 0.0]])}}, threshold=0.75)
    assert planned["rA"][1] == [41, 42]
    assert w.take_pending_alias("rA") is None


@requires_mooncake
def test_dedup_can_be_switched_off(monkeypatch):
    """The control arm: v2 plumbing active, no blocks withheld."""
    from kv_fast_fusion import pd_dedup_v2
    monkeypatch.setattr(pd_dedup_v2, "V2_ENABLED", False)
    w = _worker()
    v = [1.0, 0.0, 0.5, 0.25]
    req_blocks = {"rA": [[], [41]], "rB": [[], [42]]}
    sigs = {"rA": {1: _payload([v])}, "rB": {1: _payload([v])}}
    assert w.plan_pull(req_blocks, sigs, 0.75) == req_blocks


@requires_mooncake
def test_a_resident_block_serves_a_later_pull():
    """Across pulls, not just within one — the capability v1 tried to get from the producer's
    index and mostly failed to, because it was guessing at decode residency."""
    w = _worker()
    v = [1.0, 0.0, 0.5, 0.25]
    m = torch.tensor([v], dtype=torch.float32)
    norms = m.norm(dim=1)
    sig = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    w.note_resident(1, sig, pd_lsh.sub_hashes(sig, proj), norms.tolist(), [900], owner="old")

    planned = w.plan_pull({"rA": [[], [41]]}, {"rA": {1: _payload([v])}}, threshold=0.75)

    assert planned["rA"][1] == [v2._SENTINEL], "served entirely from a block already on D"
    assert w.take_pending_alias("rA") == {1: {41: (900, "old", None)}}
    assert w._engine.stats.dropped_resident == {1: 1}


@requires_mooncake
def test_freed_blocks_stop_serving():
    """The invalidation that makes resident aliasing safe under preemption."""
    w = _worker()
    v = [1.0, 0.0, 0.5, 0.25]
    m = torch.tensor([v], dtype=torch.float32)
    norms = m.norm(dim=1)
    sig = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    w.note_resident(1, sig, pd_lsh.sub_hashes(sig, proj), norms.tolist(), [900], owner="old")

    w._on_blocks_freed([900])           # what the block pool calls on every release

    assert not w.is_resident(1, 900)
    planned = w.plan_pull({"rA": [[], [41]]}, {"rA": {1: _payload([v])}}, threshold=0.75)
    assert planned["rA"][1] == [41], "a freed block can never serve a later request"


@requires_mooncake
def test_a_completed_pull_makes_its_blocks_alias_targets():
    """Residency is earned by a transfer that finished, never by one that was merely planned —
    aliasing to a block still being written would read whatever was there before."""
    w = _worker()
    w.finished_recving_reqs = set()
    v = [1.0, 0.0, 0.5, 0.25]
    w.plan_pull({"rA": [[], [41]]}, {"rA": {1: _payload([v])}}, threshold=0.75)
    assert not w.is_resident(1, 41), "not until the KV has actually landed"

    pull_metas = {"rA": types.SimpleNamespace(
        pull_tasks_count=1, d_req_id="rA", local_block_ids=[[], [41]])}
    w.process_pulling_result(
        types.SimpleNamespace(ok_reqs=["rA"], err_reqs=None, ff_redirects=None), pull_metas)

    assert w.is_resident(1, 41)
    assert "rA" in w.finished_recving_reqs


@requires_mooncake
def test_an_alias_is_not_appliable_until_its_transfer_completes():
    """Regression, and the most expensive bug in v2 so far.

    The apply path expires a map whose owner has not been batched within _FF_APPLY_MAX_AGE forward
    steps (~1.2 s), and treats the expiry as "this block was never written" → recompute. An owner
    cannot be batched until its KV lands, which is a whole remote round trip. Staging the aliases at
    DECISION time therefore started that clock a full transfer too early and expired essentially all
    of them: the first v2 run applied 22 of 26,531 and re-prefilled 493 of 500 requests on the
    decode. Aliases must not reach the apply path until the pull is acknowledged."""
    w = _worker()
    w.finished_recving_reqs = set()
    v = [1.0, 0.0, 0.5, 0.25]
    w.plan_pull({"rA": [[], [41]], "rB": [[], [42]]},
                {"rA": {1: _payload([v])}, "rB": {1: _payload([v])}}, threshold=0.75)

    assert w._engine._pending_alias.get("rB"), "the decision was taken"
    assert w.drain_pending_alias() == {}, "but nothing is appliable while the KV is in flight"

    pull_metas = {"rB": types.SimpleNamespace(
        pull_tasks_count=1, d_req_id="rB", local_block_ids=[[], [42]])}
    w.process_pulling_result(
        types.SimpleNamespace(ok_reqs=["rB"], err_reqs=None, ff_redirects=None), pull_metas)

    assert w.drain_pending_alias() == {"rB": {1: {42: (41, "rA", None)}}}, "appliable once it lands"


@requires_mooncake
def test_a_failed_pull_discards_its_aliases():
    """Nothing was written, so an alias from that pull must never be applied."""
    w = _worker()
    v = [1.0, 0.0, 0.5, 0.25]
    w.plan_pull({"rA": [[], [41]], "rB": [[], [42]]},
                {"rA": {1: _payload([v])}, "rB": {1: _payload([v])}}, threshold=0.75)
    assert w._engine._pending_alias.get("rB")

    w._forget_pending(["rA", "rB"])

    assert w._engine._pending_alias == {} and w._engine._pending_resident == {}


# =====================================================================================
# the wire
# =====================================================================================
@requires_mooncake
def test_the_structs_carry_the_signature_phase():
    """Both fields ride the EXISTING structs (omit_defaults), so a v1 peer never sees them and one
    decoder serves both versions."""
    import msgspec

    from kv_fast_fusion.connectors import mooncake_connector_ff as ff
    payload = _payload([[1.0, 0.0, 0.5], [0.0, 1.0, 0.25]])

    meta = ff.MooncakeXferMetadataFF(
        remote_hostname="h", remote_port=1, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"rA": ("t1", [[], [41, v2._SENTINEL]])},
        kv_caches_base_addr=[7], want_signatures=True)
    got = msgspec.msgpack.Decoder(ff.MooncakeXferMetadataFF).decode(
        msgspec.msgpack.Encoder().encode(meta))
    assert got.want_signatures is True
    assert got.req_blocks["rA"][1] == [[], [41, v2._SENTINEL]], "the sentinel survives the wire"

    resp = ff.MooncakeXferResponseFF(
        status=v2.MooncakeXferResponseStatus.FINISH, signatures={"rA": {1: payload}})
    got_r = msgspec.msgpack.Decoder(ff.MooncakeXferResponseFF).decode(
        msgspec.msgpack.Encoder().encode(resp))
    sig, norms, hashes = v2.SignatureCodec.decode(got_r.signatures["rA"][1])
    assert sig.shape == (2, 3) and len(norms) == 2 and len(hashes) == 2


@requires_mooncake
def test_a_v1_pull_request_still_decodes(monkeypatch):
    """The new fields must be invisible when unset, or a v1 producer would reject a v1 decode."""
    import msgspec

    from kv_fast_fusion.connectors import mooncake_connector_ff as ff
    meta = ff.MooncakeXferMetadataFF(
        remote_hostname="h", remote_port=1, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"rA": ("t1", [[41]])}, kv_caches_base_addr=[7])
    raw = msgspec.msgpack.Encoder().encode(meta)
    assert b"want_signatures" not in raw, "omit_defaults keeps it off the wire"
    assert msgspec.msgpack.Decoder(ff.MooncakeXferMetadataFF).decode(raw).want_signatures is False


class _FakeSock:
    def __init__(self):
        self.sent = []

    async def send_multipart(self, parts):
        self.sent.append(parts)


def _ready_send_meta(block_ids, ready=True):
    import asyncio as _a
    ev = _a.Event()
    if ready:
        ev.set()
    return types.SimpleNamespace(local_block_ids=block_ids, ready=ev, sending=0,
                                 transfer_id="t1", p_req_id="p1")


@requires_mooncake
def test_phase_one_answers_with_signatures_and_transfers_nothing():
    """Phase 1 must not move KV — that is what makes the extra round trip affordable."""
    import asyncio as _a

    import msgspec

    from kv_fast_fusion.connectors import mooncake_connector_ff as ff
    w = _worker()
    w.device_kv_caches = {"l0": _kv([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])}
    w._encoder = msgspec.msgpack.Encoder()
    sm = _ready_send_meta([[], [1, 2]])
    w.reqs_need_send = {"t1": sm}
    sock = _FakeSock()
    meta = types.SimpleNamespace(req_blocks={"rA": ("t1", [[], [41, 42]])})

    _a.run(w._send_signatures(b"id", sock, meta))

    assert len(sock.sent) == 1
    resp = msgspec.msgpack.Decoder(ff.MooncakeXferResponseFF).decode(sock.sent[0][1])
    sig, _n, _h = v2.SignatureCodec.decode(resp.signatures["rA"][1])
    assert sig.shape[0] == 2, "one signature per block the transfer would write"
    assert sm.sending == 0, "the abort-timeout guard is released again"


@requires_mooncake
def test_phase_one_skips_a_prefill_that_is_not_ready(monkeypatch):
    """Best-effort: a producer that has not finished prefilling costs compression, never a stall."""
    import asyncio as _a

    import msgspec
    monkeypatch.setattr(v2, "_SIG_READY_TIMEOUT", 0.01)
    w = _worker()
    w.device_kv_caches = {"l0": _kv([[1.0, 0.0]])}
    w._encoder = msgspec.msgpack.Encoder()
    sm = _ready_send_meta([[], [1]], ready=False)
    w.reqs_need_send = {"t1": sm}
    sock = _FakeSock()

    _a.run(w._send_signatures(b"id", sock, types.SimpleNamespace(
        req_blocks={"rA": ("t1", [[], [41]])})))

    from kv_fast_fusion.connectors import mooncake_connector_ff as ff
    resp = msgspec.msgpack.Decoder(ff.MooncakeXferResponseFF).decode(sock.sent[0][1])
    assert resp.signatures is None
    assert sm.sending == 0


@requires_mooncake
def test_phase_one_ignores_the_warmup_group():
    """Group 0 is never fused, so it must never be offered for dedup either."""
    import asyncio as _a

    import msgspec

    from kv_fast_fusion.connectors import mooncake_connector_ff as ff
    w = _worker()
    w._group_layers = {0: {"l0"}, 1: {"l0"}}
    w.device_kv_caches = {"l0": _kv([[1.0, 0.0], [0.0, 1.0]])}
    w._encoder = msgspec.msgpack.Encoder()
    w.reqs_need_send = {"t1": _ready_send_meta([[1], [2]])}
    sock = _FakeSock()

    _a.run(w._send_signatures(b"id", sock, types.SimpleNamespace(
        req_blocks={"rA": ("t1", [[7], [41]])})))

    resp = msgspec.msgpack.Decoder(ff.MooncakeXferResponseFF).decode(sock.sent[0][1])
    assert set(resp.signatures["rA"]) == {1}


# =====================================================================================
# applying the aliases — where v2's polarity differs from v1's
# =====================================================================================
class _FakeBlockTable:
    """Mirrors the runner's rectangular table: rows are padded to a common width and the real
    length lives in num_blocks_per_row, which is what write_runner_block_table honours."""

    def __init__(self, rows):
        self.num_blocks_per_row = [len(r) for r in rows]
        width = max(len(r) for r in rows)
        padded = [r + [0] * (width - len(r)) for r in rows]
        import numpy
        self.block_table = types.SimpleNamespace(
            np=numpy.array(padded), gpu=torch.tensor(padded))


def _fake_runner(blocks_by_rid, batched=None, computed=None):
    """blocks_by_rid: {req_id: [per-group block id lists]}; group 1 only, one row per request.

    `computed` sets each request's num_computed_tokens, which is what locates its write frontier.
    Left unset it is 0, and since the frontier check is inert unless the applier was given a block
    size, that keeps every pre-existing test byte-identical."""
    order = list(blocks_by_rid)
    rows = [blocks_by_rid[r][1] for r in order]
    ib = types.SimpleNamespace(
        req_id_to_index={r: i for i, r in enumerate(order)
                         if batched is None or r in batched},
        block_table=types.SimpleNamespace(block_tables={1: _FakeBlockTable(rows)}))
    reqs = {r: types.SimpleNamespace(block_ids=[list(g) for g in blocks_by_rid[r]],
                                     num_computed_tokens=(computed or {}).get(r, 0))
            for r in order}
    return types.SimpleNamespace(input_batch=ib, requests=reqs, _updated_block_tables=None)


def _connector(worker):
    conn = v2.MooncakeConnectorFFv2.__new__(v2.MooncakeConnectorFFv2)
    conn.connector_worker = worker
    conn._ff_pending = {}
    conn._ff_pending_merges = None
    conn._ff_applied = 0
    conn._ff_step = 100
    return conn


@requires_mooncake
def test_an_alias_is_applied_and_its_orphan_staged_for_free(monkeypatch):
    from kv_fast_fusion import fast_fusion_block_pool as bp
    w = _worker()
    w._engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}
    w._engine._planner._resident.setdefault(1, set()).add(41)      # rep landed and is still held
    w._engine._resident_owner[(1, 41)] = "rA"                      # ...and still owned by rA
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]})
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)

    _connector(w)._ff_consumer_apply()

    assert runner._updated_block_tables == {"rB": {1: [50, 41]}}
    assert runner.requests["rB"].block_ids[1] == [50, 41], "the runner table really was rewritten"
    assert w._ff_failed_blocks == set(), "nothing to recompute"
    assert w._engine.stats.applied == 1


@requires_mooncake
def test_an_alias_to_a_freed_representative_forces_a_recompute(monkeypatch):
    """v2's one real hazard. The victim block was never written, so refusing the substitution
    cannot mean 'keep your own copy' the way it did in v1 — it has to mean recompute.

    The refusal is deferred to the deadline rather than taken on sight, because a representative
    that is not resident YET is indistinguishable from one that is gone (see the retry test below).
    What must not change is that a rep which never arrives still ends in recompute."""
    from kv_fast_fusion import fast_fusion_block_pool as bp
    w = _worker()
    w._engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}       # 41 deliberately NOT resident
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]})
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)

    conn = _connector(w)
    conn._ff_consumer_apply()
    assert w._ff_failed_blocks == set(), "still within the deadline — the rep may yet register"

    for _ in range(pd_dedup_v2.APPLY_MAX_AGE + 2):
        conn._ff_consumer_apply()

    assert runner._updated_block_tables is None, "no table rewrite"
    assert w._ff_failed_blocks == {51}, "the never-written block goes to the load-failure path"
    assert w._engine.stats.recomputed == 1


@requires_mooncake
def test_a_representative_registered_late_is_applied_not_recomputed(monkeypatch):
    """The race this retry exists for. A representative from the SAME transfer becomes resident
    only when ITS request's release() runs, and nothing orders that before the victim's — ok_reqs
    can list them either way, in either response message. Failing on the first look turned that
    ordering into a full re-prefill: 601-765 requests per run, every one rep_not_resident."""
    from kv_fast_fusion import fast_fusion_block_pool as bp
    w = _worker()
    w._engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]})
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)

    conn = _connector(w)
    conn._ff_consumer_apply()                       # rep has not registered yet
    assert w._ff_failed_blocks == set() and w._engine.stats.applied == 0

    w._engine._planner._resident.setdefault(1, set()).add(41)     # rA's transfer lands
    w._engine._resident_owner[(1, 41)] = "rA"
    conn._ff_consumer_apply()

    assert w._engine.stats.applied == 1, "the alias is rescued, not recomputed"
    assert w._ff_failed_blocks == set()
    assert runner._updated_block_tables == {"rB": {1: [50, 41]}}


@requires_mooncake
def test_a_recycled_representative_is_refused_even_though_it_is_resident(monkeypatch):
    """The hazard the retry would otherwise open. Waiting for a rep means the block id can be freed
    and REISSUED to another request inside the deadline; residency would then be true again with
    entirely different KV behind it. Applying that is the one failure here no counter would catch,
    so identity — not presence — is what the apply checks."""
    from kv_fast_fusion import fast_fusion_block_pool as bp
    w = _worker()
    w._engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]})
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)

    # Block 41 is resident again — but it now belongs to rC, not the rA we chose.
    w._engine._planner._resident.setdefault(1, set()).add(41)
    w._engine._resident_owner[(1, 41)] = "rC"

    conn = _connector(w)
    for _ in range(pd_dedup_v2.APPLY_MAX_AGE + 2):
        conn._ff_consumer_apply()

    assert w._engine.stats.applied == 0, "never point a victim at another request's KV"
    assert w._engine.stats.fail_reasons["rep_recycled"] == 1
    assert w._ff_failed_blocks == {51}, "recompute is the safe outcome"


@requires_mooncake
def test_an_owner_that_never_gets_batched_forces_a_recompute(monkeypatch):
    """The map is held for a while — a request can complete its recv a step before it is scheduled —
    but it must not be held forever, and dropping it silently would strand unwritten blocks."""
    from kv_fast_fusion import fast_fusion_block_pool as bp
    from kv_fast_fusion import pd_dedup_v2
    w = _worker()
    w._engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}
    w._engine._planner._resident.setdefault(1, set()).add(41)
    w._engine._resident_owner[(1, 41)] = "rA"
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]}, batched=["rA"])
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)
    conn = _connector(w)

    conn._ff_consumer_apply()
    assert w._ff_failed_blocks == set(), "still waiting for it to be scheduled"
    assert "rB" in conn._applier().pending

    for _ in range(pd_dedup_v2.APPLY_MAX_AGE + 1):
        conn._ff_consumer_apply()      # steps pass; the owner never appears in input_batch

    assert w._ff_failed_blocks == {51}
    assert "rB" not in conn._applier().pending
    assert w._engine.stats.fail_reasons["owner_never_batched"] == 1


@requires_mooncake
@pytest.mark.parametrize("scenario,reason", [("rep_freed", "rep_not_resident"),
                                             ("victim_gone", "victim_not_in_table")])
def test_each_failure_names_its_own_cause(monkeypatch, scenario, reason):
    """One pooled counter for four distinct failures is what forced the last root cause to be
    inferred from step-rate arithmetic. Each cause implies a different fix — staging, pinning
    representatives, or a stale map — so they must be told apart in the stats."""
    from kv_fast_fusion import fast_fusion_block_pool as bp
    w = _worker()
    victim = 51 if scenario == "rep_freed" else 99      # 99 is not in rB's table
    w._engine._alias_ready = {"rB": {1: {victim: (41, "rA", None)}}}
    if scenario == "victim_gone":
        w._engine._planner._resident.setdefault(1, set()).add(41)
        w._engine._resident_owner[(1, 41)] = "rA"
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]})
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)

    conn = _connector(w)
    # rep_not_resident is retried to the deadline (the rep may simply not have been registered
    # yet), victim_not_in_table is terminal on sight — so drive past the deadline to see both.
    for _ in range(pd_dedup_v2.APPLY_MAX_AGE + 2):
        conn._ff_consumer_apply()

    assert w._engine.stats.fail_reasons[reason] == 1
    assert sum(w._engine.stats.fail_reasons.values()) == 1, "exactly one cause is charged"
    assert w._ff_failed_blocks == {victim}


# =====================================================================================
# reporting
# =====================================================================================
@requires_mooncake
def test_stats_report_wire_saving_not_a_producer_claim():
    """v1's headline counted merges the producer merely SENT, 73% of which never landed. Here the
    only number reported is blocks that were never requested, which cannot be inflated."""
    w = _worker()
    v = [1.0, 0.0, 0.5, 0.25]
    w.plan_pull({"rA": [[], [41]], "rB": [[], [42]]},
                {"rA": {1: _payload([v])}, "rB": {1: _payload([v])}}, 0.75)

    s = w._engine.stats.stats_dict()
    assert s["bff_version"] == 2
    assert s["blocks_planned"] == 2 and s["blocks_not_requested"] == 1
    assert s["blocks_not_requested_same_pull"] == 1
    assert s["wire_saving_pct"] == 50.0
    assert s["wire_saving_per_group"]["1"] == {"planned": 2, "not_requested": 1, "pct": 50.0}
    assert sum(s["lsh_accept_cos"]["1"].values()) == 1


@requires_mooncake
def test_the_reported_saving_equals_the_holes_in_the_pull_request():
    """The number reported must BE the number of blocks not asked for — counting what the planner
    proposed instead of what was withheld is exactly how v1's headline drifted from reality."""
    torch.manual_seed(5)
    w = _worker()
    rows = torch.randn(10, 16)
    rows[4] = rows[1]
    rows[8] = rows[1]
    req_blocks, sigs = {}, {}
    for i in range(10):
        req_blocks[f"r{i}"] = [[], [100 + i]]
        sigs[f"r{i}"] = {1: _payload([rows[i].tolist()])}

    planned = w.plan_pull(req_blocks, sigs, threshold=0.75)

    holes = sum(1 for g in planned.values() for b in g[1] if b == v2._SENTINEL)
    s = w._engine.stats.stats_dict()
    assert holes == 2, "the two duplicates, and nothing else"
    assert s["blocks_not_requested"] == holes
    assert s["blocks_planned"] == 10


# =====================================================================================
# what v2 no longer does
# =====================================================================================
@requires_mooncake
def test_v2_needs_no_forward_path_hook():
    """save_kv_layer was where v1 spent 10.8 ms/group and why it demanded PIECEWISE cudagraphs."""
    conn = v2.MooncakeConnectorFFv2.__new__(v2.MooncakeConnectorFFv2)
    assert conn.save_kv_layer("l0", torch.zeros(2, 2, 1, 1, 1), None) is None
    assert v2.MooncakeConnectorFFv2.requires_piecewise_for_cudagraph({}) is False


@requires_mooncake
def test_v2_builds_its_own_worker():
    assert v2.MooncakeConnectorFFv2._WORKER_CLS is v2.MooncakeConnectorWorkerFFv2


def test_registration_is_idempotent(monkeypatch):
    calls = []
    registry = {}
    fake = types.SimpleNamespace(
        _registry=registry,
        register_connector=lambda name, path, cls: (calls.append(name),
                                                    registry.setdefault(name, cls)))
    import vllm.distributed.kv_transfer.kv_connector.factory as f
    monkeypatch.setattr(f, "KVConnectorFactory", fake)
    v2.register_mooncake_connector_ff_v2()
    v2.register_mooncake_connector_ff_v2()
    assert calls == ["MooncakeConnectorFFv2"]


# =====================================================================================
# the GPU path must not move when the shared core is changed for Ascend
# =====================================================================================
# `signature_matrix` grew a layout dispatch (`key_blocks`) and a `num_blocks` guard so the Ascend
# (K, V) tuple layout could use the same core. Both are additions for a shape the GPU never sends,
# but "should be inert" is not a property you get to assert without a test — a silent change here
# moves every merge decision, and the symptom would be a throughput number, not an error.
def test_the_gpu_indexing_form_is_bitwise_unchanged():
    """`kv[0, idx]` was the GPU formulation before the Ascend port; `kv[0].index_select(0, idx)` is
    what key_blocks does now. Pinned as bitwise equality, including a repeated index."""
    torch.manual_seed(0)
    kv = torch.randn(2, 64, 8, 4, 16)                 # [2, num_blocks, block, heads, dim]
    idx = torch.tensor([3, 17, 0, 63, 17], dtype=torch.long)

    assert torch.equal(kv[0, idx], pd_dedup_v2.key_blocks(kv, idx, is_mla=False))


def test_the_num_blocks_guard_is_inert_when_the_caller_omits_it():
    """The GPU connector calls signature_matrix WITHOUT num_blocks, so the Ascend block_size_scale
    refusal must not be reachable from it — with or without a shape that would trip the check."""
    torch.manual_seed(0)
    layers = [torch.randn(2, 8, 4, 2, 8)]
    sig_a, norms_a = pd_dedup_v2.signature_matrix(layers, [1, 2, 3], False, [None])
    sig_b, norms_b = pd_dedup_v2.signature_matrix(layers, [1, 2, 3], False, [None], num_blocks=None)

    assert torch.equal(sig_a, sig_b) and torch.equal(norms_a, norms_b)


# =====================================================================================
# aliasing a block the decode is still writing into
#
# A request's last prompt block is almost always partially filled, and the decode keeps writing its
# newly generated K/V into the free slots. Substituting THAT block points those writes at the
# representative's physical slots — shared with its owner and with every other request aliasing it —
# so they overwrite each other and all read whichever wrote last. Attention stops seeing distinct
# K/V for new tokens and the model locks into a verbatim repetition loop, after a window as long as
# the block had slots left. Block size 128 throughout, matching BFF's requirement.
# =====================================================================================
def _hot_applier(engine, block_size=128):
    written, failed = [], set()
    applier = pd_dedup_v2.AliasApplier(
        engine,
        lambda r, rid, gi, blocks: (written.append((rid, gi, list(blocks))) or True),
        failed.update, block_size=block_size)
    return applier, written, failed


def _staged(victim, rep, owner="rA"):
    engine = pd_dedup_v2.DedupEngine(resident=False)
    engine._alias_ready = {"rB": {1: {victim: (rep, owner, None)}}}
    engine._planner._resident.setdefault(1, set()).add(rep)
    engine._resident_owner[(1, rep)] = owner
    return engine


def test_a_finished_block_is_still_aliased():
    """The safe, common case — and where all of v2's compression comes from. rB holds four blocks
    (512 tokens) and has computed 512, so nothing is left to write: every position is cold."""
    engine = _staged(victim=51, rep=41)
    applier, written, failed = _hot_applier(engine)

    applier.apply(_fake_runner({"rA": [[], [41]], "rB": [[], [50, 51, 52, 53]]},
                               computed={"rA": 128, "rB": 512}))

    assert written == [("rB", 1, [50, 41, 52, 53])], "a cold block must still alias"
    assert not failed
    assert engine.stats.hot_block_aliases == {}


def test_the_block_the_request_is_still_writing_is_refused():
    """rB has computed 300 of its 512 slots, so the next token goes to block index 2 — position 2
    and beyond are hot. Aliasing position 2 would send rB's new K/V into rA's block 41."""
    engine = _staged(victim=52, rep=41)
    applier, written, failed = _hot_applier(engine)

    applier.apply(_fake_runner({"rA": [[], [41]], "rB": [[], [50, 51, 52, 53]]},
                               computed={"rA": 128, "rB": 300}))

    assert written == [], "the block table must not be rewritten"
    assert failed == {52}, "the victim holds nothing, so it must go to recompute"
    assert engine.stats.fail_reasons["victim_still_writing"] == 1
    assert engine.stats.hot_block_aliases == {1: 1}
    assert engine.stats.hot_slots == 128 - 300 % 128, "the window before output degenerates"


def test_one_hot_victim_does_not_cost_the_whole_group_its_aliases():
    """`_substitute` reports a single reason for an entire mapping, so refusing inside it would fail
    every alias in the group. With ~8 aliases per request-group that turns one unsafe block into
    eight recomputes — the guard has to be finer than the reason it reports."""
    engine = _staged(victim=52, rep=41)
    engine._alias_ready["rB"][1][50] = (42, "rA", None)      # a second, COLD victim
    engine._planner._resident[1].add(42)
    engine._resident_owner[(1, 42)] = "rA"
    applier, written, failed = _hot_applier(engine)

    applier.apply(_fake_runner({"rA": [[], [41, 42]], "rB": [[], [50, 51, 52, 53]]},
                               computed={"rA": 256, "rB": 300}))

    assert written == [("rB", 1, [42, 51, 52, 53])], "the cold alias survives, the hot one does not"
    assert failed == {52}
    assert engine.stats.applied == 1


def test_the_guard_can_be_turned_off_and_still_counts(monkeypatch):
    """The reproduce configuration. BFF_V2_PROTECT_HOT_BLOCKS=0 restores the old behaviour so the
    damage can be observed deliberately — but the counter still fires, so a run measures the
    exposure either way."""
    monkeypatch.setattr(pd_dedup_v2, "PROTECT_HOT_BLOCKS", False)
    engine = _staged(victim=52, rep=41)
    applier, written, failed = _hot_applier(engine)

    applier.apply(_fake_runner({"rA": [[], [41]], "rB": [[], [50, 51, 52, 53]]},
                               computed={"rA": 128, "rB": 300}))

    assert written == [("rB", 1, [50, 51, 41, 53])], "unguarded, the hot block IS substituted"
    assert not failed
    assert engine.stats.hot_block_aliases == {1: 1}, "and it is still counted"


def test_without_a_block_size_the_check_is_inert():
    """Every caller that does not pass one — the GPU and layerwise connectors today — must behave
    exactly as before. A frontier guessed from a wrong block size would refuse safe aliases and hide
    the real ones, so not knowing it disables the check rather than approximating it."""
    engine = _staged(victim=52, rep=41)
    applier, written, _failed = _hot_applier(engine, block_size=None)

    applier.apply(_fake_runner({"rA": [[], [41]], "rB": [[], [50, 51, 52, 53]]},
                               computed={"rA": 128, "rB": 300}))

    assert written == [("rB", 1, [50, 51, 41, 53])]
    assert engine.stats.hot_block_aliases == {}


def test_a_partially_filled_last_block_is_hot_but_a_full_one_is_not():
    """The boundary that decides how often this fires at all. 512 computed tokens over four
    128-blocks leaves nothing partial, so block 3 is finished; one token more and block 4 opens."""
    engine = pd_dedup_v2.DedupEngine(resident=False)
    applier = pd_dedup_v2.AliasApplier(engine, lambda *a: True, set().update, block_size=128)

    assert applier._hot_from(types.SimpleNamespace(num_computed_tokens=512)) == (4, 0)
    assert applier._hot_from(types.SimpleNamespace(num_computed_tokens=513)) == (4, 127)
    assert applier._hot_from(types.SimpleNamespace(num_computed_tokens=500)) == (3, 12)


def test_a_refused_hot_victim_is_counted_once_even_when_its_group_retries():
    """A group whose representative is not resident YET is retried for APPLY_MAX_AGE steps. If the
    refused victim stayed in the retry state it would be re-counted on every one of those steps, and
    the run would report an exposure several times larger than it was — the number this whole
    exercise exists to read."""
    engine = _staged(victim=52, rep=41)
    engine._alias_ready["rB"][1][50] = (77, "rA", None)     # cold victim, rep NOT resident -> retry
    applier, written, failed = _hot_applier(engine)
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51, 52, 53]]},
                          computed={"rA": 128, "rB": 300})

    for _ in range(3):
        applier.apply(runner)

    assert written == [], "the cold alias keeps retrying, the hot one is gone"
    assert engine.stats.hot_block_aliases == {1: 1}, "counted once, not once per retry"
    assert engine.stats.fail_reasons["victim_still_writing"] == 1
    assert failed == {52}


def test_the_stats_report_the_exposure_and_whether_it_was_blocked():
    """The two numbers the run has to come back with: non-zero says the mechanism was live, and the
    guard flag says whether this run was measuring it or suffering it."""
    engine = _staged(victim=52, rep=41)
    applier, _w, _f = _hot_applier(engine)
    applier.apply(_fake_runner({"rA": [[], [41]], "rB": [[], [50, 51, 52, 53]]},
                               computed={"rA": 128, "rB": 300}))

    s = engine.stats.stats_dict()

    assert s["hot_block_aliases"] == 1
    assert s["hot_block_aliases_per_group"] == {1: 1}
    assert s["hot_block_guarded"] is True
    assert s["alias_failure_reasons"]["victim_still_writing"] == 1


# =====================================================================================
# the per-request decline histogram
#
# The run-level wire saving cannot tell "every request gives up a little" from "a few requests give
# up nearly everything", and only the second destroys those requests. This is the number that can.
# =====================================================================================
def test_each_request_lands_in_the_bucket_for_its_own_fraction():
    s = pd_dedup_v2.DedupStats()

    s.note_request_decline(1, 20)        # 5%
    s.note_request_decline(4, 20)        # 20%
    s.note_request_decline(19, 20)       # 95%

    assert s.request_decline_frac["0-10%"] == 1
    assert s.request_decline_frac["10-25%"] == 1
    assert s.request_decline_frac["90-100%"] == 1


def test_the_same_average_reads_differently_when_it_is_concentrated():
    """Ten requests at 10% and ten at 5%+95% both average ~10% overall. The histogram is what makes
    those two runs distinguishable — the second has ten ruined requests."""
    spread, concentrated = pd_dedup_v2.DedupStats(), pd_dedup_v2.DedupStats()
    for _ in range(20):
        spread.note_request_decline(2, 20)
    for _ in range(10):
        concentrated.note_request_decline(0, 20)
        concentrated.note_request_decline(19, 20)

    assert spread.request_decline_frac["90-100%"] == 0
    assert concentrated.request_decline_frac["90-100%"] == 10


def test_bucket_edges_fall_on_the_lower_bound():
    s = pd_dedup_v2.DedupStats()

    for declined in (0, 10, 25, 50, 75, 90, 100):
        s.note_request_decline(declined, 100)

    assert s.request_decline_frac == {"0-10%": 1, "10-25%": 1, "25-50%": 1,
                                      "50-75%": 1, "75-90%": 1, "90-100%": 2}


def test_a_request_with_nothing_planned_is_not_recorded():
    """Otherwise every request the connector declined to plan for would pile into the 0-10% bucket
    and dilute the distribution the run is being read for."""
    s = pd_dedup_v2.DedupStats()

    s.note_request_decline(0, 0)

    assert sum(s.request_decline_frac.values()) == 0


def test_the_histogram_and_the_cap_count_reach_the_stats_file():
    s = pd_dedup_v2.DedupStats()
    s.note_request_decline(19, 20)
    s.requests_capped = 3

    d = s.stats_dict()

    assert d["request_decline_frac"]["90-100%"] == 1
    assert d["requests_capped"] == 3


def test_the_default_applier_never_reaches_the_ambiguity_branch():
    """`normalize_req_id` defaults to identity and the runner's batch is a dict, so two entries can
    never collide. The GPU stats reporting owner_id_ambiguous=0 is not luck — it is unreachable."""
    engine = pd_dedup_v2.DedupEngine(resident=False)
    engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    engine._resident_owner[(1, 41)] = "rA"
    runner = _fake_runner({"rA": [[], [41]], "rB": [[], [50, 51]]})
    written = []
    applier = pd_dedup_v2.AliasApplier(
        engine,
        lambda r, rid, gi, blocks: (written.append((rid, gi, list(blocks))) or True),
        set().update)

    applier.apply(runner)

    assert written == [("rB", 1, [50, 41])]
    assert engine.stats.fail_reasons["owner_id_ambiguous"] == 0


# =====================================================================================
# batching visibility in the stats file
# =====================================================================================
def test_a_transport_that_does_not_batch_reports_none_not_zero():
    """0 round trips and 1.0 requests-per-round-trip are very different findings, and the GPU
    connector produces neither — it has no batched signature phase at all. Reporting 0 would read as
    "batching never engaged", which is the exact failure the Ascend pull transport had."""
    st = pd_dedup_v2.DedupStats()

    got = st.stats_dict()

    assert got["sig_batches"] is None
    assert got["sig_requests_per_exchange"] is None


def test_the_requests_per_round_trip_ratio_is_reported():
    st = pd_dedup_v2.DedupStats()
    st.sig_batches = 20
    st.sig_batched_requests = 512

    got = st.stats_dict()

    assert got["sig_batches"] == 20
    assert got["sig_requests_per_exchange"] == 25.6


def test_one_request_per_round_trip_is_visible_as_exactly_that():
    """The inert case, which cost a whole run before anyone noticed: it must read 1.0, not blank."""
    st = pd_dedup_v2.DedupStats()
    st.sig_batches = 512
    st.sig_batched_requests = 512

    assert st.stats_dict()["sig_requests_per_exchange"] == 1.0
