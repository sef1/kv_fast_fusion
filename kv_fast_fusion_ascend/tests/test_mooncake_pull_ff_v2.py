"""Unit tests for BFF v2 on the Ascend NON-layerwise (pull) connector.

v2 lets the DECODE decide which of the producer's blocks are worth reading: where a block is close
enough to one D already holds, D aliases its own and never issues the read. Unlike v1 that saves
wire bandwidth, not just KV capacity.

On this transport D both decides and executes the transfer, so nothing about the decision crosses
the wire — which makes the positional-pairing hazard purely local, and therefore fully testable here
on CPU with no NPU, no Mooncake and no vllm_ascend.

The file is organised around the three things that can silently corrupt KV:
  * dropping a declined position from one side of the pairing but not the other,
  * filtering after the runs have been coalesced instead of before, and
  * confusing the two block-id spaces — P's, which the signature request must speak, and D's, which
    everything the engine records must speak.
"""

import ast
import os
import re
import time

import pytest

from kv_fast_fusion_ascend.connectors import mooncake_connector_ff_v2 as v2

SENT = -1


# =====================================================================================
# filter_sentinels — the one function that must never be got wrong
# =====================================================================================
def test_a_declined_position_vanishes_from_both_lists():
    """The transfer pairs planned[i] (D's block) with remote[i] (P's). Dropping from one side only
    shifts every survivor onto the wrong source block and reads the wrong KV, with no error."""
    planned = [70, SENT, 72, 73]          # D's blocks, one declined
    remote = [10, 11, 12, 13]             # P's blocks, same slots

    local, r = v2.filter_sentinels(planned, remote)

    assert local == [70, 72, 73]
    assert r == [10, 12, 13], "the remote slot for the declined block must go too"
    assert list(zip(local, r)) == [(70, 10), (72, 12), (73, 13)], "surviving pairs are unchanged"


def test_a_list_with_nothing_declined_is_returned_unchanged():
    """A request the producer never answered for must cost one scan and nothing else."""
    planned, remote = [7, 8, 9], [1, 2, 3]

    p, r = v2.filter_sentinels(planned, remote)

    assert p is planned and r is remote, "no copying on the common path"


def test_every_position_declined_yields_an_empty_read():
    """The fully-deduped request. It must produce an empty transfer, not a partial one."""
    p, r = v2.filter_sentinels([SENT, SENT], [10, 11])

    assert p == [] and r == []


def test_a_prefix_cache_hit_still_pairs_the_right_tail():
    """The real prefix-hit case, which is why the filter runs AFTER align_per_group.

    D allocated only the uncomputed tail, so alignment has already trimmed P's list to that tail and
    the two arrive here the SAME length. Declining inside the tail must keep the pairing intact."""
    planned, remote = [72, SENT, 74], [12, 13, 14]    # P sent 10..14; D kept the last three

    p, r = v2.filter_sentinels(planned, remote)

    assert list(zip(p, r)) == [(72, 12), (74, 14)]


def test_an_unequal_length_pair_truncates_rather_than_mis_indexing():
    """Documents the defensive branch, not a reachable state.

    `align_per_group` raises when local is longer and tail-trims remote when it is shorter, so by
    the time a pair reaches this function the lengths are equal — `_align_and_group` re-checks that
    before filtering at all. If a future caller ever breaks that invariant, dropping the positions
    it cannot index is the safe failure: a block that is never read is a recompute, whereas indexing
    past the end would pair a survivor with someone else's KV."""
    p, r = v2.filter_sentinels([70, SENT, 72, 73], [10, 11])

    assert p == [70, 72, 73]
    assert r == [10], "only positions the companion can actually index survive"
    assert all(x in (10, 11) for x in r), "no remote id is invented"


def test_empty_input_is_handled():
    assert v2.filter_sentinels([], []) == ([], [])


def test_more_than_one_companion_is_filtered_at_the_same_indices():
    """The arity the connector actually uses is two lists in two different id spaces. A filter that
    handled only one would leave the other holding the declined slot."""
    planned = [70, SENT, 72]
    remote = [10, 11, 12]
    tags = ["a", "b", "c"]

    p, r, t = v2.filter_sentinels(planned, remote, tags)

    assert (p, r, t) == ([70, 72], [10, 12], ["a", "c"])


def test_a_companion_is_never_read_back_out_of_the_planned_list():
    """The two lists are disjoint id spaces, so deriving one from the other is not merely untidy —
    it substitutes D's block ids for P's as transfer SOURCE addresses. Asserted by giving them
    ranges that cannot be confused."""
    p, r = v2.filter_sentinels([700, SENT, 702], [10, 11, 12])

    assert all(x < 100 for x in r), "the remote side must stay in the producer's range"
    assert all(x >= 700 for x in p), "the planned side must stay in the decode's range"


# =====================================================================================
# the ordering invariant: filter BEFORE coalescing
# =====================================================================================
def _coalesce(remote, local):
    """The contiguous-run pairing the real transfer uses, reimplemented minimally.

    Mirrors vllm_ascend's `group_concurrent_contiguous`, which is not importable here: runs of
    consecutive ids on BOTH sides are merged into one segment."""
    if not remote:
        return [], []
    gr, gl = [[remote[0]]], [[local[0]]]
    for i in range(1, len(remote)):
        if remote[i] == remote[i - 1] + 1 and local[i] == local[i - 1] + 1:
            gr[-1].append(remote[i])
            gl[-1].append(local[i])
        else:
            gr.append([remote[i]])
            gl.append([local[i]])
    return gr, gl


def test_declining_a_block_breaks_the_contiguous_run():
    """The reason the filter must run BEFORE coalescing.

    Blocks 10,11,12 are contiguous and would coalesce into one segment. Declining the middle slot
    must split them into two — if the filter ran after, 11 would already be inside a merged run and
    would be dragged across the wire despite being declined, or worse, the run's length would no
    longer match its ids."""
    planned, remote = [70, SENT, 72], [10, 11, 12]

    local, r = v2.filter_sentinels(planned, remote)
    gr, gl = _coalesce(r, local)

    assert gr == [[10], [12]], "the declined block must split the run"
    assert gl == [[70], [72]]
    assert sum(len(seg) for seg in gr) == 2, "only two blocks are actually read"


def test_a_run_with_nothing_declined_still_coalesces():
    """The filter must not disturb the common path — coalescing is what keeps the transfer cheap."""
    local, r = v2.filter_sentinels([70, 71, 72], [10, 11, 12])
    gr, gl = _coalesce(r, local)

    assert gr == [[10, 11, 12]] and gl == [[70, 71, 72]], "one segment, as before v2"


def test_segment_lengths_still_match_their_ids_after_filtering():
    """The transfer computes each segment's byte length from len(local_block_id). A filter that
    left a length out of step with its ids would read the wrong number of bytes."""
    local, r = v2.filter_sentinels([70, 71, SENT, 73, 74], [10, 11, 12, 13, 14])

    gr, gl = _coalesce(r, local)

    assert [len(s) for s in gr] == [len(s) for s in gl]
    assert sum(len(s) for s in gl) == 4


# =====================================================================================
# the coalescer — concurrency, where a lost wakeup wedges a recv worker for the rest of the run
# =====================================================================================
def test_concurrent_callers_to_one_peer_share_round_trips():
    """The whole point. 32 recv workers asking at once must not each pay the producer's latency;
    whoever leads drains the queue and the rest ride along."""
    import threading

    started = threading.Event()
    release = threading.Event()

    def send(_key, batch):
        started.set()
        release.wait(5)                      # hold the leader so the others pile up behind it
        for it in batch:
            it.result = f"ans-{it.payload}"

    c = v2.BatchCoalescer(send, max_batch=32)
    got, threads = {}, []
    for i in range(8):
        t = threading.Thread(target=lambda i=i: got.__setitem__(i, c.ask("p", i)))
        t.start()
        threads.append(t)
        if i == 0:
            started.wait(5)                  # make thread 0 the leader deterministically
    release.set()
    for t in threads:
        t.join(5)

    assert got == {i: f"ans-{i}" for i in range(8)}, "every caller gets its OWN answer"
    assert c.batches < 8, f"callers must share round trips, got {c.batches} for 8 asks"
    assert c.batched_items == 8


def test_a_raising_sender_never_leaves_a_caller_waiting():
    """A dead producer must cost compression, not liveness. If `done` were set only on the success
    path, the recv worker that was waiting would block for the rest of the run."""
    def send(_key, _batch):
        raise RuntimeError("producer is gone")

    c = v2.BatchCoalescer(send, max_batch=8)

    assert c.ask("p", 1) is None, "no answer — and no exception, and no hang"
    assert c.failures == 1


def test_every_caller_of_a_failed_batch_returns_the_same_way():
    """Containment has to be symmetric. If the exception propagated, the outcome would depend on
    whether a caller happened to lead its batch or follow in one — and a future sender that forgot
    its own try/except could wedge a recv worker. Leader and followers alike get no answer."""
    import threading

    started, release = threading.Event(), threading.Event()
    seen = []

    def send(_key, batch):
        seen.append(len(batch))
        if not started.is_set():
            started.set()
            release.wait(5)                  # hold leader #1 so the rest queue behind it
        raise RuntimeError("boom")

    c = v2.BatchCoalescer(send, max_batch=8)
    got = {}
    threads = [threading.Thread(target=lambda i=i: got.__setitem__(i, c.ask("p", i)))
               for i in range(5)]
    threads[0].start()
    started.wait(5)
    for t in threads[1:]:
        t.start()
        # Give each a moment to queue while leader #1 is still held, so they land in ONE batch.
        time.sleep(0.02)
    release.set()
    for t in threads:
        t.join(5)

    assert not any(t.is_alive() for t in threads), "nobody hangs"
    assert got == dict.fromkeys(range(5)), "every caller returns None"
    assert max(seen) > 1, f"the followers really were batched together, saw {seen}"
    assert c.failures == len(seen)


def test_peers_are_batched_separately():
    """Two producers are two sockets and two answers. Mixing their queues would send one peer's
    block ids to the other, which owns entirely different blocks."""
    seen = []

    def send(key, batch):
        seen.append((key, [it.payload for it in batch]))
        for it in batch:
            it.result = key

    c = v2.BatchCoalescer(send, max_batch=8)

    assert c.ask("pA", 1) == "pA"
    assert c.ask("pB", 2) == "pB"
    assert seen == [("pA", [1]), ("pB", [2])]


def test_the_batch_size_is_capped():
    """One exchange's device work has to stay bounded, or its timeout stops meaning anything."""
    import threading

    started, release = threading.Event(), threading.Event()
    sizes = []

    def send(_key, batch):
        sizes.append(len(batch))
        if not started.is_set():
            started.set()
            release.wait(5)
        for it in batch:
            it.result = it.payload

    c = v2.BatchCoalescer(send, max_batch=2)
    threads = [threading.Thread(target=lambda i=i: c.ask("p", i)) for i in range(6)]
    threads[0].start()
    started.wait(5)
    for t in threads[1:]:
        t.start()
    release.set()
    for t in threads:
        t.join(5)

    assert max(sizes) <= 2, f"batches must respect the cap, saw {sizes}"
    assert sum(sizes) == 6, "and nothing is dropped"


def test_a_single_uncontended_caller_costs_one_exchange():
    """The common path when the decode is not busy — batching must not add a round trip."""
    calls = []

    def send(_key, batch):
        calls.append(len(batch))
        for it in batch:
            it.result = "ok"

    c = v2.BatchCoalescer(send, max_batch=32)

    assert c.ask("p", 1) == "ok"
    assert calls == [1]


# =====================================================================================
# splitting one group's rows back to the slots that asked
# =====================================================================================
def test_each_slot_gets_back_exactly_the_rows_it_asked_about():
    """The producer computes a group ONCE for the batch, so the rows must be sliced back by the
    same offsets they were concatenated at. A mis-slice does not raise — it hands one request's
    signatures to another, the same silent mis-attribution the block-id spaces already cost us."""
    plan = v2.signature_batch_plan({0: {1: [10, 11]}, 2: {1: [20, 21, 22]}})

    slots, flat, lengths = plan[1]

    assert flat == [10, 11, 20, 21, 22], "concatenated in slot order"
    assert lengths == [2, 3]
    # Replay the split the server does and check every slot recovers its own ids.
    off, recovered = 0, {}
    for slot, n in zip(slots, lengths):
        recovered[slot] = flat[off:off + n]
        off += n
    assert recovered == {0: [10, 11], 2: [20, 21, 22]}


def test_a_group_only_one_slot_asked_about_is_planned_alone():
    plan = v2.signature_batch_plan({0: {1: [10], 2: [30]}, 1: {1: [20]}})

    assert plan[1] == ([0, 1], [10, 20], [1, 1])
    assert plan[2] == ([0], [30], [1])


def test_empty_and_absent_groups_never_enter_the_plan():
    """A group with no blocks contributes no rows; leaving it in would make `lengths` disagree with
    the slots and shift every later slice."""
    plan = v2.signature_batch_plan({0: {1: [], 2: [30]}, 1: {}, 2: None})

    assert set(plan) == {2}
    assert plan[2] == ([0], [30], [1])


def test_the_lengths_always_sum_to_the_flat_ids():
    """The invariant the server's zip depends on."""
    plan = v2.signature_batch_plan({s: {1: list(range(s + 1))} for s in range(4)})

    for _slots, flat, lengths in plan.values():
        assert sum(lengths) == len(flat)


def test_an_empty_batch_plans_nothing():
    assert v2.signature_batch_plan({}) == {}


# =====================================================================================
# the BATCHED signature exchange
#
# The producer's cost is device-sync latency, not signature compute: signatures_for_group ends in a
# `.cpu()`, so answering per request drained the NPU queue once per group PER REQUEST. Batching
# collapses that, mirroring the GPU connector, which answers for every pending send in one reply.
# =====================================================================================
def test_the_batched_request_keeps_each_slot_s_groups_apart():
    msg = v2.sig_request_batch_msg({0: {1: [10, 11]}, 1: {1: [20], 2: [30]}})

    assert msg[0] == v2.MSG_SIG_REQUEST_BATCH
    assert msg[1] == {0: {1: [10, 11]}, 1: {1: [20], 2: [30]}}


def test_the_batched_tags_are_distinct_from_the_single_request_ones():
    """Both payloads decode to an int-keyed dict of dicts, so the SHAPES cannot tell them apart:
    single is {group: payload}, batched is {slot: {group: payload}}. A producer one version behind
    would answer a batched question in the single shape and the decode would read group indices as
    slot numbers — handing request 0's signatures to whichever request sat in slot 0, silently.

    Separate tags are what make that impossible: the old producer does not recognise the batched
    tag, replies with the single one, and the parser below rejects it."""
    assert v2.MSG_SIG_REQUEST_BATCH != v2.MSG_SIG_REQUEST
    assert v2.MSG_SIG_REPLY_BATCH != v2.MSG_SIG_REPLY
    assert v2.parse_sig_reply_batch(v2.sig_reply_msg({1: {"s": 1}})) == {}, \
        "a single-shaped reply must never be read as a batch"
    assert v2.parse_sig_reply(v2.sig_reply_batch_msg({0: {1: {"s": 1}}})) == {}, \
        "and the converse"


def test_a_batched_reply_round_trips_per_slot():
    reply = v2.sig_reply_batch_msg({0: {1: "pa"}, 2: {1: "pb", 3: "pc"}})

    assert v2.parse_sig_reply_batch(reply) == {0: {1: "pa"}, 2: {1: "pb", 3: "pc"}}


def test_a_slot_the_producer_could_not_answer_is_absent_not_empty():
    """Absence is how the decode learns to pull that request in full. An empty dict left in place
    would say the same thing, but only by accident of truthiness."""
    assert v2.sig_reply_batch_msg({0: {1: "p"}, 1: {}, 2: None})[1] == {0: {1: "p"}}


def test_slots_are_dropped_from_the_request_when_they_have_no_blocks():
    assert v2.sig_request_batch_msg({0: {1: [10]}, 1: {1: []}, 2: {}})[1] == {0: {1: [10]}}


@pytest.mark.parametrize("bad", [
    None, (), (b"wrong", {}), (v2.MSG_SIG_REPLY_BATCH,), (v2.MSG_SIG_REPLY_BATCH, None),
    (v2.MSG_SIG_REPLY_BATCH, "nope"), (v2.MSG_SIG_REPLY_BATCH, {0: "not-a-dict"}),
])
def test_any_malformed_batched_reply_reads_as_no_signatures(bad):
    """Every unrecognisable answer degrades the WHOLE batch to full reads. Refusing to serve
    requests over a compression optimisation is never acceptable, and a batch multiplies that."""
    assert v2.parse_sig_reply_batch(bad) == {}


def test_json_style_string_keys_are_coerced_at_both_levels():
    """msgpack preserves int keys, but nothing guarantees a future transport will. String slots or
    groups would silently find no signatures and quietly disable v2 for the batch."""
    got = v2.parse_sig_reply_batch((v2.MSG_SIG_REPLY_BATCH, {"1": {"3": {"s": 1}}}))

    assert got == {1: {3: {"s": 1}}}


# =====================================================================================
# the signature exchange codec
# =====================================================================================
def test_the_request_carries_the_groups_and_their_block_ids():
    msg = v2.sig_request_msg({1: [10, 11], 2: [20]})

    assert msg[0] == v2.MSG_SIG_REQUEST
    assert msg[1] == {1: [10, 11], 2: [20]}
    assert all(type(k) is int for k in msg[1])


def test_empty_groups_are_dropped_from_the_request():
    """Asking about a group with no blocks wastes a payload and invites an empty-signature reply
    that looks like a failure."""
    assert v2.sig_request_msg({1: [10], 2: []})[1] == {1: [10]}


def test_a_reply_round_trips():
    reply = v2.sig_reply_msg({1: {"sig": "payload"}})

    assert v2.parse_sig_reply(reply) == {1: {"sig": "payload"}}


def test_json_style_string_group_keys_are_coerced():
    """msgpack preserves int keys, but nothing guarantees a future transport will. Indexing by int
    against string keys would silently find no signatures and quietly disable v2."""
    assert v2.parse_sig_reply((v2.MSG_SIG_REPLY, {"3": {"s": 1}})) == {3: {"s": 1}}


def test_a_group_with_no_signature_is_omitted_not_nulled():
    """`signatures_for_group` returns None when there is nothing to describe. A None in the payload
    would reach DedupEngine.plan as a decodable object."""
    assert v2.sig_reply_msg({1: None, 2: {"s": 1}})[1] == {2: {"s": 1}}


@pytest.mark.parametrize("bad", [
    None, (), (b"wrong-tag", {}), (v2.MSG_SIG_REPLY,), (v2.MSG_SIG_REPLY, None),
    (v2.MSG_SIG_REPLY, "not-a-dict"),
])
def test_any_malformed_reply_reads_as_no_signatures(bad):
    """Every unrecognisable answer must degrade to a full read. Refusing to serve a request over a
    compression optimisation is the one outcome that is never acceptable."""
    assert v2.parse_sig_reply(bad) == {}


# =====================================================================================
# the two block-id spaces
# =====================================================================================
def test_the_producer_is_asked_about_its_blocks_and_the_engine_planned_over_the_decode_s():
    """The bug that cost a whole run.

    P must be asked about P's blocks — it answers by indexing its own KV cache. But everything the
    engine records from that answer addresses D: the alias victim, the representative, the residency
    index, and the ids handed to get_block_ids_with_load_errors, which vLLM reads as D's own blocks.
    Planning over P's ids left AliasApplier unable to find a single victim in D's block table."""
    aligned = [([], []), ([10, 11], [70, 71]), ([20], [80])]

    ask, plan = v2.signature_request_and_plan_groups(aligned)

    assert ask == {1: [10, 11], 2: [20]}, "the request carries the PRODUCER's block ids"
    assert plan == {1: [70, 71], 2: [80]}, "the engine is planned over the DECODE's"


def test_groups_are_keyed_by_index_and_empties_dropped():
    """DedupEngine speaks {group: ids}; the transport holds a positional per-group list. An
    off-by-one here would attribute every block in the request to the wrong group."""
    aligned = [([], []), ([10, 11], [70, 71]), ([], []), ([30], [90])]

    ask, plan = v2.signature_request_and_plan_groups(aligned)

    assert ask == {1: [10, 11], 3: [30]}
    assert plan == {1: [70, 71], 3: [90]}


def test_an_all_empty_request_yields_nothing_to_ask_about():
    """Asking about a group with no blocks wastes a payload and invites an empty-signature reply
    that looks like a failure."""
    assert v2.signature_request_and_plan_groups([([], []), ([], [])]) == ({}, {})


def test_a_group_the_decode_did_not_allocate_is_asked_about_by_neither_side():
    """A full prefix-cache hit on one group leaves P's list non-empty and D's empty. There is
    nothing to read and nothing to alias, so it must not reach the engine at all — an entry with a
    populated `ask` and an empty `plan` would trip plan()'s row-count guard for the whole group."""
    ask, plan = v2.signature_request_and_plan_groups([([], []), ([10, 11], [])])

    assert ask == {} and plan == {}


def test_the_two_spaces_stay_the_same_length_per_group():
    """plan() refuses a group whose signature row count does not match its block count, and the row
    count comes from `ask`. align_per_group is what makes them equal; this pins that the helper does
    not disturb it."""
    aligned = [([], []), ([10, 11, 12], [70, 71, 72])]

    ask, plan = v2.signature_request_and_plan_groups(aligned)

    assert [len(v) for v in ask.values()] == [len(v) for v in plan.values()]


# =====================================================================================
# the warmup group is never eligible
# =====================================================================================
def test_the_warmup_group_is_never_asked_about_or_planned():
    """Group 0 holds layers[0:2] + layers[-2:] — the first two and last two layers. Every other BFF
    path refuses it by name; this connector was the only one without the guard because it subclasses
    v1's transport while the policy lived in v1's fusion path, which v2 deletes.

    Unguarded, one run aliased 99.16% of it: 11,582 blocks onto 23 representatives, so nearly every
    request shared one of 23 physical blocks for its first and last two layers. The output stayed
    fluent — each substitution was cosine >0.98 — but the model stopped answering the prompt and
    stopped emitting EOS, and 27.6% of requests ran to the token cap."""
    aligned = [([10, 11], [70, 71]), ([20, 21], [80, 81])]

    ask, plan = v2.signature_request_and_plan_groups(aligned)

    assert 0 not in ask, "the producer must not even be asked for the warmup group"
    assert 0 not in plan, "and it must never reach the engine"
    assert ask == {1: [20, 21]} and plan == {1: [80, 81]}


def test_an_explicit_selection_restricts_further_but_cannot_re_admit_the_warmup_group():
    """BFF_FF_GROUPS is an A/B knob over the ELIGIBLE groups, which never include 0 — same
    semantics as `_parse_ff_groups`, so one knob means the same thing on both transports."""
    aligned = [([10], [70]), ([20], [80]), ([30], [90]), ([40], [100])]

    ask, plan = v2.signature_request_and_plan_groups(aligned, groups={0, 1, 3})

    assert set(ask) == {1, 3}, "group 0 stays out even when named explicitly"
    assert set(plan) == {1, 3}


def test_no_selection_means_every_group_except_the_warmup_one():
    """`None` is 'every eligible group', not 'every group' — the distinction the omission turned on:
    unset must not mean the warmup group is included."""
    aligned = [([10], [70]), ([20], [80]), ([30], [90])]

    ask, _plan = v2.signature_request_and_plan_groups(aligned, groups=None)

    assert set(ask) == {1, 2}


def test_an_empty_selection_disables_the_exchange_entirely():
    """Distinct from None. A caller that computed 'no groups' must ask about nothing, not
    everything."""
    assert v2.signature_request_and_plan_groups([([10], [70]), ([20], [80])], groups=set()) == ({},
                                                                                               {})


# =====================================================================================
# end to end: plan -> release -> apply, across two disjoint block-id spaces
#
# pd_dedup_v2 is pure torch, so the whole decision-to-block-table path runs here with no NPU. This
# is where the id-space bug actually showed up on hardware, so it is where it is pinned.
# =====================================================================================
def _payload(rows):
    """One group's signature payload for `rows`, exactly as the producer builds it."""
    import torch

    from kv_fast_fusion import pd_lsh
    from kv_fast_fusion.pd_dedup_v2 import SignatureCodec

    m = torch.tensor(rows, dtype=torch.float32)
    norms = m.norm(dim=1).clamp(min=1e-6)
    sig = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    return SignatureCodec.encode(sig, norms, pd_lsh.sub_hashes(sig, proj))


def _runner(blocks_by_rid):
    """Minimal stand-in for NPUModelRunner: {req_id: [per-group block id lists]}."""
    import types

    return types.SimpleNamespace(
        input_batch=types.SimpleNamespace(
            req_id_to_index={r: i for i, r in enumerate(blocks_by_rid)}),
        requests={r: types.SimpleNamespace(block_ids=[list(g) for g in gs])
                  for r, gs in blocks_by_rid.items()},
        _updated_block_tables=None)


def _plan_release_apply(plan_ids, sig_rows):
    """Run the real engine over `plan_ids` (per request, group 1), then apply.

    Returns ``(writes, failed, stats)``. `plan_ids` is what the connector feeds `plan()` — the
    question this test exists to answer is which id space that has to be."""
    from kv_fast_fusion.pd_dedup_v2 import AliasApplier, DedupEngine

    engine = DedupEngine(resident=True)
    engine.plan({rid: [[], list(ids)] for rid, ids in plan_ids.items()},
                {rid: {1: _payload(sig_rows[rid])} for rid in plan_ids},
                threshold=0.8)
    for rid in plan_ids:
        engine.release(rid)

    writes, failed = [], set()
    applier = AliasApplier(
        engine,
        lambda _r, rid, gi, blocks: (writes.append((rid, gi, list(blocks))) or True),
        failed.update)
    # The runner only ever holds the DECODE's block ids.
    applier.apply(_runner({"rA": [[], [40]], "rB": [[], [41]]}))
    return writes, failed, engine.stats


# Two requests in one transfer. rB's block is near-identical to rA's, so rB aliases rA and never
# reads. P calls those blocks 4000/4001; D calls them 40/41 — deliberately disjoint, because in the
# failing run the two spaces overlapped often enough for 17 aliases to "apply" by coincidence.
_SIGS = {"rA": [[1.0, 0.0, 0.0, 0.0]], "rB": [[1.0, 0.02, 0.0, 0.0]]}
_DECODE_IDS = {"rA": [40], "rB": [41]}
_PRODUCER_IDS = {"rA": [4000], "rB": [4001]}


def test_an_alias_planned_over_the_decode_s_ids_reaches_the_block_table():
    """The fix, end to end: victim 41 is replaced by representative 40, both D's own ids."""
    writes, failed, stats = _plan_release_apply(_DECODE_IDS, _SIGS)

    assert writes == [("rB", 1, [40])], "rB's slot must now point at rA's block"
    assert not failed, "nothing goes to recompute when the alias resolves"
    assert stats.applied == 1


def test_planning_over_the_producer_s_ids_cannot_apply_and_sends_the_block_to_recompute():
    """The bug, reproduced. Identical inputs but P's ids: the engine records victim 4001 against
    representative 4000, `_substitute` cannot find 4001 in D's block table, and the block is
    reported as a KV load failure instead — which is what turned 18 declined blocks into 31- and
    35-request reschedule cascades on the box."""
    writes, failed, stats = _plan_release_apply(_PRODUCER_IDS, _SIGS)

    assert writes == [], "no alias can be placed"
    assert stats.applied == 0
    assert failed == {4001}, "and the failure is reported against a block id D does not own"
    assert stats.fail_reasons.get("victim_not_in_table") == 1


def test_the_helper_feeds_the_engine_the_space_that_actually_applies():
    """Ties the two tests above to the connector: whichever list
    `signature_request_and_plan_groups` puts in `plan` is what reaches `DedupEngine.plan`."""
    aligned = [([], []), (_PRODUCER_IDS["rA"] + _PRODUCER_IDS["rB"],
                          _DECODE_IDS["rA"] + _DECODE_IDS["rB"])]

    _ask, plan = v2.signature_request_and_plan_groups(aligned)

    assert plan[1] == [40, 41], "the appliable space, per the tests above"


# =====================================================================================
# hot-block collisions: are two live requests writing new tokens to the same slots?
#
# The direct test of "attention is no longer receiving distinct K/V for newly generated tokens".
# Each request writes its new K/V at `position % 128` of block `position // 128`, so two requests
# sharing a block in their write frontier overwrite each other slot for slot and both read whichever
# wrote last. Block size 128 throughout, matching BFF's requirement.
# =====================================================================================
def test_two_requests_writing_into_the_same_block_are_reported():
    """rA has computed 300 tokens and rB 300, so both are writing into their block index 2 — and
    aliasing has pointed both at physical block 99."""
    hits = v2.hot_block_collisions({"rA": (300, [[], [10, 11, 99]]),
                                    "rB": (300, [[], [20, 21, 99]])}, 128)

    assert list(hits) == [(1, 99)]
    assert sorted(hits[(1, 99)]) == ["rA", "rB"]


def test_sharing_a_finished_block_is_not_a_collision():
    """The intended, correct state after aliasing: both requests read block 99 for a region neither
    will write again. Reporting this would make the audit fire on every successful merge and drown
    the signal it exists to carry."""
    hits = v2.hot_block_collisions({"rA": (300, [[], [99, 11, 12]]),
                                    "rB": (300, [[], [99, 21, 22]])}, 128)

    assert hits == {}


def test_the_frontier_moves_with_the_request_s_progress():
    """Same block list, different `num_computed_tokens`, opposite verdicts — and the boundary is
    exact. The shared block sits at index 1. At 255 computed tokens the frontier is index 1, so the
    requests are still writing there and it collides; one token later the frontier is index 2, the
    shared block is finished, and sharing it is the intended result of a merge.

    Note the frontier covers every block from there ON, not just the partial one: a block that has
    not been written yet is as unsafe to alias as one being written."""
    blocks = {"rA": [[], [10, 99, 12]], "rB": [[], [20, 99, 22]]}

    assert v2.hot_block_collisions({r: (255, b) for r, b in blocks.items()}, 128) != {}
    assert v2.hot_block_collisions({r: (256, b) for r, b in blocks.items()}, 128) == {}


def test_a_request_never_collides_with_itself():
    """One request legitimately holds the same block twice after aliasing — that is a merge within
    its own table, not two writers. Only DISTINCT requests can corrupt each other."""
    hits = v2.hot_block_collisions({"rA": (0, [[], [99, 99, 99]])}, 128)

    assert hits == {}, "the same rid appearing twice is not two writers"


def test_groups_are_kept_apart():
    """Each KV-cache group has its own block table and its own block-id space, so block 99 of group
    1 and block 99 of group 2 are different memory. Pooling them would invent collisions."""
    hits = v2.hot_block_collisions({"rA": (0, [[], [99], []]),
                                    "rB": (0, [[], [], [99]])}, 128)

    assert hits == {}


def test_the_null_block_is_not_a_collision():
    """Padding positions carry -1 (or the null block) and are shared by construction."""
    hits = v2.hot_block_collisions({"rA": (0, [[], [-1, 5]]), "rB": (0, [[], [-1, 6]])}, 128)

    assert hits == {}


def test_a_healthy_step_allocates_no_result():
    """This runs after every apply on the decode's critical path, so the common case has to be a
    scan and nothing more."""
    assert v2.hot_block_collisions({"rA": (0, [[], [1, 2]]), "rB": (0, [[], [3, 4]])}, 128) == {}


def test_an_unknown_block_size_disables_the_audit():
    """Without it there is no frontier, and guessing one would report collisions on cold blocks —
    i.e. on every successful merge."""
    assert v2.hot_block_collisions({"rA": (300, [[], [99]]), "rB": (300, [[], [99]])}, 0) == {}


# =====================================================================================
# the per-request ceiling: compression vs. replacing the request
#
# The cosine bar is per BLOCK; the harm is per REQUEST. At BFF_MAX_REL_ERR=0.3 the floor is cosine
# 0.954 — a reasonable bar for one block and a catastrophic one for nineteen of twenty, because the
# model then attends to a coherent prompt that is not the one it was asked.
# =====================================================================================
def test_the_declined_fraction_is_counted_across_every_group():
    """A request half-replaced in each of six groups is a half-replaced request. Counting per group
    would report six modest numbers and hide the one that matters."""
    planned = {1: [70, SENT, 72, SENT], 2: [80, SENT, 82, 83]}

    assert v2.decline_fraction(planned) == (3, 8)


def test_an_untouched_request_reports_nothing_declined():
    assert v2.decline_fraction({1: [70, 71]}) == (0, 2)
    assert v2.decline_fraction({}) == (0, 0)


def test_a_request_mostly_replaced_is_read_in_full():
    """The measured case: 19 of 20 blocks declined in every group. Dropping the plan means the
    request transfers exactly as stock would — one full read, against answering someone else's
    prompt."""
    planned = {1: [SENT] * 19 + [99], 2: [SENT] * 19 + [98]}

    out, capped = v2.cap_request_decline(planned, max_frac=0.5)

    assert capped is True
    assert out == {}, "nothing is declined, so every block is fetched"


def test_ordinary_compression_passes_through_untouched():
    """Where all of v2's saving comes from — this must not be disturbed."""
    planned = {1: [70, SENT, 72, 73], 2: [80, 81, 82, 83]}

    out, capped = v2.cap_request_decline(planned, max_frac=0.5)

    assert capped is False
    assert out is planned, "no copying on the common path"


def test_the_ceiling_is_exact_and_inclusive():
    """Exactly at the ceiling is allowed; one block past it is not. An off-by-one here silently
    moves the policy by a whole block on short requests."""
    at = {1: [70, 71, SENT, SENT]}                       # 2 of 4 = 0.50
    over = {1: [70, SENT, SENT, SENT]}                   # 3 of 4 = 0.75

    assert v2.cap_request_decline(at, max_frac=0.5)[1] is False
    assert v2.cap_request_decline(over, max_frac=0.5)[1] is True


def test_a_ceiling_of_one_disables_the_cap():
    """1.0 must be inert, so the cap can be taken out of the picture for an A/B without also
    changing the code path."""
    planned = {1: [SENT] * 20}

    assert v2.cap_request_decline(planned, max_frac=1.0) == (planned, False)


def test_an_empty_plan_is_not_capped():
    """A request nobody planned for must not be counted as a capped one — that would report the
    guard firing on requests it never saw."""
    assert v2.cap_request_decline({}, max_frac=0.5) == ({}, False)
    assert v2.cap_request_decline({1: []}, max_frac=0.5)[1] is False


# =====================================================================================
# the slot trace: is each new token reaching a fresh address?
#
# The conjecture in its literal form, and the check that keeps a run from coming back inconclusive:
# it catches an addressing fault that has nothing to do with aliasing.
# =====================================================================================
def test_the_write_slot_is_the_block_table_arithmetic():
    """position 300 with block size 128 -> block index 2, offset 44. If the table's third entry is
    block 7, the token lands at 7*128 + 44."""
    assert v2.write_slot(300, [10, 11, 7, 12], 128) == 7 * 128 + 44


def test_a_position_with_no_block_yet_has_no_slot():
    """The block for the next token is allocated by the scheduler, not by us. Reporting a slot here
    would invent an address."""
    assert v2.write_slot(300, [10, 11], 128) is None
    assert v2.write_slot(0, [], 128) is None


def test_advancing_one_token_inside_a_block_advances_one_slot():
    assert v2.slot_trace_fault((300, 940), (301, 941), 128) is None


def test_a_slot_that_does_not_move_is_the_reported_symptom():
    """Every new token overwriting the previous one — attention then sees static K/V for all of
    them, which is what a hard verbatim loop looks like from the inside."""
    assert v2.slot_trace_fault((300, 940), (301, 940), 128) == "slot_repeated"


def test_a_slot_that_moves_by_the_wrong_amount_inside_a_block_is_caught():
    assert v2.slot_trace_fault((300, 940), (301, 999), 128) == "slot_not_advanced_in_block"


def test_crossing_a_block_boundary_must_reach_a_different_physical_block():
    """position 255 -> 256 crosses from block index 1 to 2. Landing at the base of a different
    physical block is correct; staying inside the same one is exactly 'fails to index the next
    physical block'."""
    assert v2.slot_trace_fault((255, 7 * 128 + 127), (256, 9 * 128), 128) is None
    assert v2.slot_trace_fault((255, 7 * 128 + 127), (256, 7 * 128), 128) == "block_not_advanced"


def test_a_non_consecutive_step_is_not_judged():
    """A resumed or chunked request legitimately jumps. Guessing there would report faults on
    healthy requests and discredit the ones that matter."""
    assert v2.slot_trace_fault((300, 940), (400, 1040), 128) is None
    assert v2.slot_trace_fault(None, (301, 941), 128) is None
    assert v2.slot_trace_fault((300, 940), (301, 941), 0) is None


def test_a_missing_slot_is_not_a_fault():
    """`write_slot` returns None when the block is not allocated yet; that is absence of evidence,
    not a fault."""
    assert v2.slot_trace_fault((300, None), (301, 941), 128) is None
    assert v2.slot_trace_fault((300, 940), (301, None), 128) is None


# =====================================================================================
# attribute collision with the vendored thread we subclass
# =====================================================================================
def _vendored_recv_thread_source():
    """The vendored mooncake_connector.py, or None if this box has no vllm_ascend checkout."""
    import importlib.util

    candidates = []
    try:
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is not None and spec.submodule_search_locations:
            candidates.append(next(iter(spec.submodule_search_locations)))
    except Exception:  # noqa: BLE001 - find_spec raises for a half-installed package
        pass
    root = os.environ.get("VLLM_ASCEND_ROOT")
    if root:
        candidates.append(os.path.join(root, "vllm_ascend"))
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    candidates.append(os.path.join(os.path.dirname(repo), "vllm-ascend", "vllm_ascend"))
    for base in candidates:
        fp = os.path.join(base, "distributed", "kv_transfer", "kv_p2p", "mooncake_connector.py")
        if os.path.isfile(fp):
            return open(fp, errors="replace").read()
    return None


def _self_assigned_in_init(source: str, class_name: str) -> set:
    """Every ``self.X = ...`` target in ``class_name.__init__``."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for fn in node.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                continue
            names = set()
            for stmt in ast.walk(fn):
                targets = (stmt.targets if isinstance(stmt, ast.Assign)
                           else [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
                for t in targets:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == "self"):
                        names.add(t.attr)
            return names
    return set()


def _v2_injected_attributes() -> set:
    """Names v2 puts onto the recv thread: its own class-level attributes, plus everything the
    worker assigns through ``self.kv_recv_thread.<name> = ...``."""
    import inspect

    src = inspect.getsource(v2)
    names = set(re.findall(r"self\.kv_recv_thread\.(\w+)\s*=", src))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheRecvingThreadFFv2":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    names |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
    return names


def test_v2_injects_no_attribute_the_vendored_thread_already_owns():
    """The bug that killed the first on-box run, generalised.

    v2 declared ``engine`` on its recv-thread subclass and the worker assigned the DedupEngine into
    it — but the vendored ``KVCacheRecvingThread.__init__`` keeps the Mooncake TransferEngine under
    that exact name, and v1's transfer calls ``self.engine.batch_transfer_sync_read``. Every request
    on the node died with ``'DedupEngine' object has no attribute 'batch_transfer_sync_read'``.

    Asserted against the vendored SOURCE rather than a name list, because the hazard is that the
    vendored file changes under us: a future upstream ``self.sig_client = ...`` would collide just as
    silently. Skipped where there is no vllm_ascend checkout to read."""
    source = _vendored_recv_thread_source()
    if source is None:
        pytest.skip("no vllm_ascend checkout on this box")

    vendored = _self_assigned_in_init(source, "KVCacheRecvingThread")
    assert "engine" in vendored, "sanity: the vendored thread should still own `engine`"

    injected = _v2_injected_attributes()
    assert injected, "sanity: v2 should inject something onto the recv thread"

    collisions = injected & vendored
    assert not collisions, (
        f"v2 injects {sorted(collisions)} onto KVCacheRecvingThread, which the vendored "
        f"__init__ already assigns. Overwriting one of those replaces a piece of the transport; "
        f"`engine` is the TransferEngine itself.")


def test_the_dedup_engine_is_reachable_under_its_new_name():
    """Renaming is only half the fix — the worker has to inject the new name, and the thread has to
    read the same one. A rename that missed either side would leave dedup permanently off with no
    error at all."""
    import inspect

    src = inspect.getsource(v2)
    assert "self.kv_recv_thread.dedup_engine = self._dedup_engine" in src
    assert "self.dedup_engine.plan(" in src, "the planner must read the injected attribute"
    assert "self.kv_recv_thread.engine = " not in src, "the fatal assignment must not come back"


# =====================================================================================
# class substitution — silent if it breaks
# =====================================================================================
def test_the_connector_builds_its_halves_from_overridable_attributes():
    """v1 originally constructed `MooncakeConnectorWorkerFF` by name, which made v2's override a
    no-op: v2 would have run with v1's worker — no dedup engine, no signature server — and the only
    symptom would have been a benchmark showing no improvement.

    Asserted on the source because the classes need the NPU stack to instantiate, while the failure
    they guard is a silent one that a CPU test can still catch."""
    import inspect

    from kv_fast_fusion_ascend.connectors import mooncake_connector_ff as v1_mod

    src = inspect.getsource(v1_mod)
    ctor = src[src.index("class MooncakeConnectorFF("):]
    ctor = ctor[:ctor.index("def request_finished_all_groups")]

    assert "self._WORKER_CLS(" in ctor, "the worker must come from the overridable attribute"
    assert "self._SCHEDULER_CLS(" in ctor, "so must the scheduler"
    assert "= MooncakeConnectorWorkerFF(" not in ctor, "no hard-coded construction by name"
