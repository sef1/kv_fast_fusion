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

import pytest
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


# =====================================================================================
# the KV layout — the bug that made the first NPU run inert
# =====================================================================================
# The Ascend worker hands each layer as a (K, V) pair of tensors with the block dim at 0. The GPU
# connector hands one stacked [2, num_blocks, ...] tensor. Indexing one as if it were the other does
# not raise — it silently reads inside block 0 — so this is stated as an equivalence between the two
# layouts rather than a smoke test.
def _gpu_stacked(vectors):
    """GPU layout: one tensor [2, nblocks, block, heads, dim]; dim 0 selects K vs V."""
    n = len(vectors)
    k = torch.zeros(n + 1, 1, 1, len(vectors[0]))
    for i, v in enumerate(vectors):
        k[i + 1, 0, 0] = torch.tensor(v, dtype=torch.float32)
    v_cache = torch.full_like(k, -7.0)      # deliberately different, so picking V would show
    return torch.stack([k, v_cache])


def _ascend_pair(vectors):
    """Ascend layout: [K, V] as separate tensors, each [nblocks, block, heads, dim]."""
    stacked = _gpu_stacked(vectors)
    return [stacked[0].clone(), stacked[1].clone()]


def test_the_two_kv_layouts_give_the_same_signature():
    """The equivalence the layout fix exists to guarantee."""
    rows = [[1.0, 0.0, 0.5, 0.25], [0.0, 1.0, 0.25, 0.5]]
    gpu = a2.signatures_for_group({"l0": _gpu_stacked(rows)}, ["l0"], [1, 2], False, [None])
    npu = a2.signatures_for_group({"l0": _ascend_pair(rows)}, ["l0"], [1, 2], False, [None])

    g_sig, g_norms, g_hashes = SignatureCodec.decode(gpu)
    n_sig, n_norms, n_hashes = SignatureCodec.decode(npu)
    assert torch.allclose(g_sig, n_sig, atol=1e-6)
    assert g_norms == n_norms and g_hashes == n_hashes


def test_the_ascend_layout_reads_k_not_v():
    """A (K, V) pair indexed as if it were stacked would read V, or block 0's interior. Both are
    wrong and neither raises, so pin the content."""
    rows = [[1.0, 0.0], [0.0, 1.0]]
    pair = _ascend_pair(rows)
    idx = torch.tensor([1, 2])

    got = pd_dedup_v2.key_blocks(pair, idx, is_mla=False)

    assert torch.equal(got, pair[0].index_select(0, idx))
    assert not torch.equal(got, pair[1].index_select(0, idx)), "read V instead of K"


def test_a_block_id_that_cannot_index_the_cache_is_refused():
    """block_size_scale != 1 means the tensor holds a multiple of the connector's blocks, so a
    connector block id addresses the wrong memory. Refusing beats merging blocks that are not
    alike, which is what a plausible-looking wrong signature would do."""
    caches = {"l0": _ascend_pair([[1.0, 0.0]] * 4)}       # 5 rows incl. the null block
    with pytest.raises(pd_dedup_v2.KVLayoutError):
        a2.signatures_for_group(caches, ["l0"], [1], False, [None], num_blocks=2)

    # ...and the matching count is accepted.
    assert a2.signatures_for_group(caches, ["l0"], [1], False, [None], num_blocks=5) is not None


def test_a_non_tensor_cache_entry_is_refused_not_guessed_at():
    with pytest.raises(pd_dedup_v2.KVLayoutError):
        pd_dedup_v2.key_blocks(["not a tensor"], torch.tensor([0]), is_mla=False)
    with pytest.raises(pd_dedup_v2.KVLayoutError):
        pd_dedup_v2.key_blocks([], torch.tensor([0]), is_mla=False)


# =====================================================================================
# inertness has to be loud
# =====================================================================================
def test_an_inert_run_is_reported_as_inert_not_as_zero_saving():
    """The whole point of the skip counters. A connector that never asked produces the same
    '0.0% saving' as one that asked and found nothing; only these tell them apart."""
    from kv_fast_fusion.pd_dedup_v2 import DedupStats
    inert = DedupStats()
    inert.note_skip("no_kv_tensors", 12)

    asked_and_found_nothing = DedupStats()
    asked_and_found_nothing.exchanges = 12
    asked_and_found_nothing.planned[1] = 900

    assert inert.is_inert() and inert.stats_dict()["inert"] is True
    assert not asked_and_found_nothing.is_inert()
    assert inert.stats_dict()["wire_saving_pct"] == 0.0
    assert asked_and_found_nothing.stats_dict()["wire_saving_pct"] == 0.0, (
        "identical headline — the reason is the only thing that distinguishes them")
    assert inert.stats_dict()["exchange_skip_reasons"]["no_kv_tensors"] == 12


def test_an_empty_kv_cache_is_the_reason_recorded():
    """The exact first-run bug: worker.kv_caches was permanently {}."""
    assert a2.signatures_for_group({}, ["l0"], [1, 2], False, [None]) is None


def test_ascend_shaped_kv_all_the_way_to_a_transfer_plan():
    """The end-to-end path, from tensors in the shape the NPU worker actually hands over to the
    (local, remote) pairs the RDMA write would use.

    Every earlier test exercises one hop with a synthetic payload; this one starts from real
    Ascend-layout KV, which is the hop the first NPU run got wrong — `worker.kv_caches` was empty,
    so nothing downstream ever ran and every stat was a truthful-looking zero."""
    dup = [1.0, 0.0, 0.0, 0.0]
    caches = {"l0": _ascend_pair([[0.0, 1.0, 0.0, 0.0], dup, [0.0, 0.0, 1.0, 0.0]])}
    engine = DedupEngine(resident=False)

    # P: signatures for its own blocks 1..3, plus a separate request holding a copy of `dup`.
    sig_b = a2.signatures_for_group(caches, ["l0"], [1, 2, 3], False, [None])
    sig_a = a2.signatures_for_group({"l0": _ascend_pair([dup])}, ["l0"], [1], False, [None])

    # D: decides, in the arrival order P sent them.
    reply = _answer(engine, 1, {"rA": [40], "rB": [50, 51, 52]}, {"rA": sig_a, "rB": sig_b})

    # P: turns the answer into what it will actually write.
    keep_remote, keep_local = a2.filter_sentinels(list(reply["rB"]), [910, 911, 912])

    assert reply["rB"] == [50, SENTINEL, 52], "rB's middle block is rA's block again"
    assert dict(zip(keep_local, keep_remote)) == {910: 50, 912: 52}
    assert 911 not in keep_local, "the duplicate is never transferred"


# =====================================================================================
# the two request-id spaces
# =====================================================================================
# vLLM PR #27987 appends a 9-character per-EngineCore suffix to the request id, so P's local id, D's
# local id and the id they agree on over the wire are three different strings. The engine is keyed
# by the wire (external) id; the runner and scheduler are keyed by this node's local id. The first
# NPU run that got as far as planning matched one against the other and applied 0 of 5216 aliases,
# every one of them ageing out as `owner_never_batched` — a counter that reads as "the scheduler was
# slow", which is why it is worth a test that names the real cause.
SUFFIX = "-8d919ad3"        # len 9, the shape get_external_request_id strips


def _applier_runner(req_blocks, batched=None):
    """A runner keyed the way the NPU keys it: LOCAL ids, suffix and all."""
    import types
    reqs = {rid: types.SimpleNamespace(block_ids=[list(g) for g in groups])
            for rid, groups in req_blocks.items()}
    names = list(req_blocks) if batched is None else batched
    return types.SimpleNamespace(
        requests=reqs,
        input_batch=types.SimpleNamespace(
            req_id_to_index={rid: i for i, rid in enumerate(names)}))


def _applier(engine, normalize=None):
    """AliasApplier with a recording writer that mimics _ff_write_runner_block_table: it refuses
    (returns False) for a rid that is not in this step's input_batch, which is the coupling the
    whole apply path depends on."""
    written, failed = [], set()

    def write(runner, rid, gi, blocks):
        if rid not in runner.input_batch.req_id_to_index:
            return False
        runner.requests[rid].block_ids[gi] = list(blocks)
        written.append((rid, gi, list(blocks)))
        return True

    a = pd_dedup_v2.AliasApplier(engine, write, failed.update, normalize_req_id=normalize)
    return a, written, failed


def test_the_applier_finds_the_owner_through_the_local_id_suffix():
    """The fix. Engine keyed by external id, runner keyed by local id, and the write must still go
    out under the LOCAL id — the block table and the merge channel are both addressed that way."""
    ext_a, ext_b = "chatcmpl-aaa", "chatcmpl-bbb"
    engine = DedupEngine(resident=False)
    engine._alias_ready = {ext_b: {1: {51: (41, ext_a)}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    runner = _applier_runner({ext_a + SUFFIX: [[], [41]], ext_b + SUFFIX: [[], [50, 51]]})
    applier, written, failed = _applier(engine, normalize=a2.to_external_request_id)

    applier.apply(runner)

    assert written == [(ext_b + SUFFIX, 1, [50, 41])], "written under the LOCAL id"
    assert applier.pending_merges == {ext_b + SUFFIX: {1: [50, 41]}}, \
        "the merge channel is keyed by the id the scheduler knows"
    assert failed == set(), "nothing had to be recomputed"
    assert engine.stats.applied == 1


def test_an_id_space_mismatch_ages_every_alias_out_as_owner_never_batched():
    """The bug, pinned as the symptom it actually produced, so this exact reading of the stats can
    never again be mistaken for a scheduling problem. Identity normalisation (the default) against
    suffixed runner ids finds nothing at all."""
    ext_a, ext_b = "chatcmpl-aaa", "chatcmpl-bbb"
    engine = DedupEngine(resident=False)
    engine._alias_ready = {ext_b: {1: {51: (41, ext_a)}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    runner = _applier_runner({ext_a + SUFFIX: [[], [41]], ext_b + SUFFIX: [[], [50, 51]]})
    applier, written, failed = _applier(engine)          # no normalizer

    for _ in range(pd_dedup_v2.APPLY_MAX_AGE + 2):
        applier.apply(runner)

    assert written == []
    assert engine.stats.applied == 0
    assert failed == {51}, "the never-written block still has to reach the recompute path"
    assert engine.stats.fail_reasons["owner_never_batched"] == 1


def test_the_default_normalizer_is_identity():
    """CUDA runs with matching ids (the stable-id patch) and must be untouched by this change."""
    rid = "chatcmpl-aaa"
    engine = DedupEngine(resident=False)
    engine._alias_ready = {rid: {1: {51: (41, "owner")}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    runner = _applier_runner({"owner": [[], [41]], rid: [[], [50, 51]]})
    applier, written, failed = _applier(engine)

    applier.apply(runner)

    assert written == [(rid, 1, [50, 41])]
    assert failed == set()


def test_two_batched_requests_sharing_an_external_id_are_refused_not_guessed_at():
    """Normalisation only has to be injective across the batch, and the vendored transport already
    assumes that (its request_map is keyed by external id). But if it ever is not, an alias map
    cannot be attributed to a request, and picking one would rewrite the WRONG request's block
    table — the only silent, unrecoverable failure in this path. It must refuse and recompute."""
    ext = "chatcmpl-aaa"
    locals_ = [ext + "-11111111", ext + "-22222222"]
    assert len({a2.to_external_request_id(r) for r in locals_}) == 1, "the collision under test"

    engine = DedupEngine(resident=False)
    engine._alias_ready = {ext: {1: {51: (41, "owner")}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    runner = _applier_runner({"owner" + SUFFIX: [[], [41]],
                              locals_[0]: [[], [50, 51]], locals_[1]: [[], [60, 51]]})
    before = [list(g) for g in runner.requests[locals_[1]].block_ids]
    applier, written, failed = _applier(engine, normalize=a2.to_external_request_id)

    applier.apply(runner)

    assert written == [], "no table was rewritten on a guess"
    assert runner.requests[locals_[1]].block_ids == before, "the other request is untouched"
    assert failed == {51}, "the unwritten block goes to the recompute path instead"
    assert engine.stats.fail_reasons["owner_id_ambiguous"] == 1
    assert applier.pending == {}, "and it is not left to age out under a misleading reason"
