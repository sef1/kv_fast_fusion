"""Unit tests for BFF v2 on the Ascend layerwise connector.

The decision logic is shared with GPU and tested in ``kv_fast_fusion/tests``; what is specific here
is the transport, and it has one hazard worth more than all the others combined:

**the producer pairs its block ids against the decode's positionally.** A block the decode declines
must therefore leave a *hole* that is removed from both sides at the same index — never a shortened
list on one side. Get that wrong and every surviving block after the hole is filled from the wrong
source, silently, with no error anywhere. :func:`filter_sentinels` is that function and most of this
file exists to pin it.

CPU only: no NPU, no Transfer Engine, no vllm_ascend import needed.
"""

import torch

from kv_fast_fusion import pd_dedup_v2, pd_lsh
from kv_fast_fusion.pd_dedup_v2 import SENTINEL, DedupEngine, SignatureCodec
from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff_v2 as a2


# =====================================================================================
# the positional-pairing guard
# =====================================================================================
def test_a_declined_block_is_removed_from_both_sides_at_the_same_index():
    """The corruption guard. P zips its own ids against D's, so dropping position 1 from D's list
    without dropping it from P's would shift every later pair by one."""
    remote = [50, SENTINEL, 52, 53]
    local = [900, 901, 902, 903]

    keep_remote, keep_local = a2.filter_sentinels(remote, local)

    assert keep_remote == [50, 52, 53]
    assert keep_local == [900, 902, 903], "P's block 901 must go with D's declined 51"
    assert dict(zip(keep_local, keep_remote)) == {900: 50, 902: 52, 903: 53}


def test_every_surviving_pair_is_the_pair_it_would_have_been():
    """Stated as the invariant rather than an example: dedup must not move any survivor."""
    remote = [50, 51, 52, 53, 54]
    local = [900, 901, 902, 903, 904]
    want = dict(zip(local, remote))

    for declined in ([1], [0], [4], [1, 3], [0, 4], [1, 2, 3]):
        r = [SENTINEL if i in declined else b for i, b in enumerate(remote)]
        keep_remote, keep_local = a2.filter_sentinels(r, local)
        got = dict(zip(keep_local, keep_remote))
        assert len(got) == len(remote) - len(declined)
        assert all(want[p] == d for p, d in got.items()), f"survivor moved, declined={declined}"


def test_nothing_declined_is_returned_untouched():
    """A non-v2 request must produce a byte-identical transfer to stock."""
    remote, local = [50, 51], [900, 901]
    keep_remote, keep_local = a2.filter_sentinels(remote, local)
    assert keep_remote is remote and keep_local is local


def test_an_all_declined_group_yields_an_empty_transfer():
    keep_remote, keep_local = a2.filter_sentinels([SENTINEL, SENTINEL], [900, 901])
    assert keep_remote == [] and keep_local == []


def test_a_short_local_list_cannot_index_out_of_range():
    """Defensive: the base can hand back mismatched lengths under chunking/TP resharding, and a
    crash on the sender thread would take the transfer down."""
    keep_remote, keep_local = a2.filter_sentinels([50, SENTINEL, 52], [900])
    assert keep_remote == [50, 52] and keep_local == [900]


# =====================================================================================
# producer-side signatures
# =====================================================================================
def _kv(vectors):
    """Layerwise-shaped KV [2, nblocks, ...]; block 0 stays the null block."""
    n = len(vectors)
    k = torch.zeros(n + 1, 1, 1, len(vectors[0]))
    for i, v in enumerate(vectors):
        k[i + 1, 0, 0] = torch.tensor(v, dtype=torch.float32)
    return torch.stack([k, k.clone()])


def test_signatures_come_straight_from_the_kv_cache():
    """No forward-path hook: the producer reads the registered cache on demand, which is what keeps
    the exchange off the model's critical path."""
    caches = {"l0": _kv([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])}

    payload = a2.signatures_for_group(caches, ["l0"], [1, 2], False, [None])

    sig, norms, hashes = SignatureCodec.decode(payload)
    assert sig.shape[0] == 2 and len(norms) == 2 and len(hashes) == 2
    assert torch.allclose(sig.norm(dim=1), torch.ones(2), atol=1e-3)


def test_signatures_are_none_when_the_group_has_nothing_to_describe():
    caches = {"l0": _kv([[1.0, 0.0]])}
    assert a2.signatures_for_group(caches, ["l0"], [], False, [None]) is None
    assert a2.signatures_for_group(caches, ["missing"], [1], False, [None]) is None


def test_more_layers_give_a_different_signature_than_one():
    """BFF_SIG_LAYERS=group vs first is a real choice, not a formality: the concat over a group's
    layers is a different (and strictly more informative) fingerprint."""
    caches = {"l0": _kv([[1.0, 0.0], [0.0, 1.0]]),
              "l1": _kv([[1.0, 0.0], [1.0, 0.0]])}
    one = SignatureCodec.decode(a2.signatures_for_group(caches, ["l0"], [1, 2], False, [None]))[0]
    both = SignatureCodec.decode(
        a2.signatures_for_group(caches, ["l0", "l1"], [1, 2], False, [None]))[0]

    # The JL projection preserves cosines only approximately, so the "orthogonal" case is a small
    # number rather than zero — which is itself worth knowing: the signature is a estimate, and the
    # threshold has to sit well clear of that noise.
    assert abs(float(one[0] @ one[1])) < 0.1, "near-orthogonal on layer 0 alone"
    assert float(both[0] @ both[1]) > 0.4, "the shared layer-1 content makes them look alike"


# =====================================================================================
# the decode's answer
# =====================================================================================
def _payload(rows):
    m = torch.tensor(rows, dtype=torch.float32)
    norms = m.norm(dim=1).clamp(min=1e-6)
    sig = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    return SignatureCodec.encode(sig, norms, pd_lsh.sub_hashes(sig, proj))


def _answer(engine, gi, req_blocks, sigs):
    """What _SigReplyServer._handle does, without needing zmq or vllm_ascend."""
    wrapped = {rid: [[] for _ in range(gi + 1)] for rid in req_blocks}
    for rid, ids in req_blocks.items():
        wrapped[rid][gi] = [int(b) for b in ids]
    planned = engine.plan(wrapped, {rid: {gi: p} for rid, p in sigs.items()})
    return {rid: planned[rid][gi] for rid in req_blocks}


def test_the_decode_declines_a_duplicate_and_keeps_the_list_length():
    """The reply is a sentinel list, never a short list — see the pairing guard above."""
    engine = DedupEngine(resident=False)
    v = [1.0, 0.0, 0.5, 0.25]

    reply = _answer(engine, 1, {"rA": [41], "rB": [42]},
                    {"rA": _payload([v]), "rB": _payload([v])})

    assert reply["rA"] == [41], "the first copy is still sent"
    assert reply["rB"] == [SENTINEL], "the second is declined, in place"
    assert len(reply["rB"]) == 1


def test_dissimilar_blocks_are_all_requested():
    engine = DedupEngine(resident=False)
    reply = _answer(engine, 1, {"rA": [41], "rB": [42]},
                    {"rA": _payload([[1.0, 0.0, 0.0, 0.0]]),
                     "rB": _payload([[0.0, 1.0, 0.0, 0.0]])})
    assert reply == {"rA": [41], "rB": [42]}


def test_the_error_budget_reaches_this_transport_too(monkeypatch):
    """The rel_err gate lives in the shared core, so it must apply here without extra wiring: two
    perfectly aligned blocks of different magnitudes are not interchangeable."""
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.20)
    engine = DedupEngine(resident=False)
    v = [1.0, 0.0, 0.5, 0.25]
    m = torch.tensor([v], dtype=torch.float32)
    sig = m / m.norm(dim=1, keepdim=True)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    small = SignatureCodec.encode(sig, torch.tensor([10.0]), pd_lsh.sub_hashes(sig, proj))
    big = SignatureCodec.encode(sig, torch.tensor([16.0]), pd_lsh.sub_hashes(sig, proj))

    reply = _answer(engine, 1, {"rA": [41], "rB": [42]}, {"rA": small, "rB": big})

    assert reply["rB"] == [42], "a 1.6x magnitude gap is not a free substitution"


def test_a_declined_block_becomes_a_pending_alias_not_an_applied_one():
    """The reply commits D to an alias, but only once the KV lands — releasing earlier is the bug
    that made the first GPU v2 run apply 22 of 26,531 aliases."""
    engine = DedupEngine(resident=False)
    v = [1.0, 0.0, 0.5, 0.25]

    _answer(engine, 1, {"rA": [41], "rB": [42]}, {"rA": _payload([v]), "rB": _payload([v])})

    assert engine.pending_alias("rB") == {1: {42: (41, "rA")}}
    assert engine.drain_ready() == {}, "not appliable while the KV is in flight"
    engine.release("rB")
    assert engine.drain_ready() == {"rB": {1: {42: (41, "rA")}}}


def test_a_request_that_never_lands_never_becomes_appliable():
    engine = DedupEngine(resident=False)
    v = [1.0, 0.0, 0.5, 0.25]
    _answer(engine, 1, {"rA": [41], "rB": [42]}, {"rA": _payload([v]), "rB": _payload([v])})

    engine.forget(["rB"])
    engine.release("rB")

    assert engine.drain_ready() == {}


def test_dedup_off_answers_with_the_request_unchanged(monkeypatch):
    """The control arm must produce a transfer byte-identical to stock."""
    monkeypatch.setattr(pd_dedup_v2, "V2_ENABLED", False)
    engine = DedupEngine(resident=False)
    v = [1.0, 0.0, 0.5, 0.25]
    reply = _answer(engine, 1, {"rA": [41], "rB": [42]},
                    {"rA": _payload([v]), "rB": _payload([v])})
    assert reply == {"rA": [41], "rB": [42]}


# =====================================================================================
# end to end: what the producer actually transfers
# =====================================================================================
def test_the_decode_reply_survives_the_round_trip_into_a_transfer_plan():
    """Reply → filter_sentinels → the (local, remote) pairs the RDMA write would use. This is the
    join the two halves have to agree on, and the only place a v2 bug becomes silent corruption."""
    engine = DedupEngine(resident=False)
    dup = [1.0, 0.0, 0.0, 0.0]
    rows = [[0.0, 1.0, 0.0, 0.0], dup, [0.0, 0.0, 1.0, 0.0]]
    local = {"rA": [900], "rB": [910, 911, 912]}
    remote = {"rA": [40], "rB": [50, 51, 52]}

    reply = _answer(engine, 1, remote,
                    {"rA": _payload([dup]), "rB": _payload(rows)})

    pairs = {}
    for rid in remote:
        keep_remote, keep_local = a2.filter_sentinels(list(reply[rid]), local[rid])
        pairs.update(dict(zip(keep_local, keep_remote)))

    assert reply["rB"] == [50, SENTINEL, 52], "rB's middle block matched rA's"
    assert pairs == {900: 40, 910: 50, 912: 52}, "911 is not transferred; the rest are unmoved"
