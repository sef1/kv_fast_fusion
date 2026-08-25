"""Unit tests for BFF v2 on the Ascend NON-layerwise (pull) connector.

v2 lets the DECODE decide which of the producer's blocks are worth reading: where a block is close
enough to one D already holds, D aliases its own and never issues the read. Unlike v1 that saves
wire bandwidth, not just KV capacity.

On this transport D both decides and executes the transfer, so nothing about the decision crosses
the wire — which makes the positional-pairing hazard purely local, and therefore fully testable here
on CPU with no NPU, no Mooncake and no vllm_ascend.

The file is organised around the two things that can silently corrupt KV:
  * dropping a declined position from one side of the pairing but not the other, and
  * filtering after the runs have been coalesced instead of before.
"""

import pytest

from kv_fast_fusion_ascend.connectors import mooncake_connector_ff_v2 as v2

SENT = -1


# =====================================================================================
# filter_sentinels — the one function that must never be got wrong
# =====================================================================================
def test_a_declined_position_vanishes_from_both_lists():
    """The transfer pairs remote[i] with local[i]. Dropping from one side only shifts every
    survivor onto the wrong source block and reads the wrong KV, with no error anywhere."""
    remote = [10, SENT, 12, 13]
    local = [70, 71, 72, 73]

    r, l = v2.filter_sentinels(remote, local)

    assert r == [10, 12, 13]
    assert l == [70, 72, 73], "the local slot for the declined block must go too"
    assert list(zip(r, l)) == [(10, 70), (12, 72), (13, 73)], "surviving pairs are unchanged"


def test_a_list_with_nothing_declined_is_returned_unchanged():
    """A request the producer never answered for must cost one scan and nothing else."""
    remote, local = [1, 2, 3], [7, 8, 9]

    r, l = v2.filter_sentinels(remote, local)

    assert r is remote and l is local, "no copying on the common path"


def test_every_position_declined_yields_an_empty_read():
    """The fully-deduped request. It must produce an empty transfer, not a partial one."""
    r, l = v2.filter_sentinels([SENT, SENT], [70, 71])

    assert r == [] and l == []


def test_a_prefix_cache_hit_still_pairs_the_right_tail():
    """The real prefix-hit case, which is why the filter runs AFTER align_per_group.

    D allocated only the uncomputed tail, so alignment has already trimmed P's list to that tail and
    the two arrive here the SAME length. Declining inside the tail must keep the pairing intact."""
    remote, local = [12, SENT, 14], [72, 73, 74]      # P sent 10..14; D kept the last three

    r, l = v2.filter_sentinels(remote, local)

    assert list(zip(r, l)) == [(12, 72), (14, 74)]


def test_an_unequal_length_pair_truncates_rather_than_mis_indexing():
    """Documents the defensive branch, not a reachable state.

    `align_per_group` raises when local is longer and tail-trims remote when it is shorter, so by
    the time a pair reaches this function the lengths are equal — `_align_and_group` re-checks that
    before filtering at all. If a future caller ever breaks that invariant, dropping the positions
    it cannot index is the safe failure: a block that is never read is a recompute, whereas indexing
    past the end would pair a survivor with someone else's KV."""
    r, l = v2.filter_sentinels([10, SENT, 12, 13], [72, 73])

    assert r == [10, 12, 13]
    assert l == [72], "only positions local can actually index survive"
    assert all(x in (72, 73) for x in l), "no local id is invented"


def test_empty_input_is_handled():
    assert v2.filter_sentinels([], []) == ([], [])


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

    Blocks 10,11,12 are contiguous and would coalesce into one segment. Declining 11 must split
    them into two segments — if the filter ran after, 11 would already be inside a merged run and
    would be dragged across the wire despite being declined, or worse, the run's length would no
    longer match its ids."""
    remote, local = [10, SENT, 12], [70, 71, 72]

    r, l = v2.filter_sentinels(remote, local)
    gr, gl = _coalesce(r, l)

    assert gr == [[10], [12]], "the declined block must split the run"
    assert gl == [[70], [72]]
    assert sum(len(seg) for seg in gr) == 2, "only two blocks are actually read"


def test_a_run_with_nothing_declined_still_coalesces():
    """The filter must not disturb the common path — coalescing is what keeps the transfer cheap."""
    r, l = v2.filter_sentinels([10, 11, 12], [70, 71, 72])
    gr, gl = _coalesce(r, l)

    assert gr == [[10, 11, 12]] and gl == [[70, 71, 72]], "one segment, as before v2"


def test_segment_lengths_still_match_their_ids_after_filtering():
    """The transfer computes each segment's byte length from len(local_block_id). A filter that
    left a length out of step with its ids would read the wrong number of bytes."""
    remote = [10, 11, SENT, 13, 14]
    local = [70, 71, 72, 73, 74]

    gr, gl = _coalesce(*v2.filter_sentinels(remote, local))

    assert [len(s) for s in gr] == [len(s) for s in gl]
    assert sum(len(s) for s in gl) == 4


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
# group wrapping for the engine
# =====================================================================================
def test_groups_are_keyed_by_index_and_empties_dropped():
    """DedupEngine speaks {group: ids}; the transport holds a positional per-group list. An
    off-by-one here would attribute every block in the request to the wrong group."""
    assert v2.wrap_groups_for_engine([[], [10, 11], [], [30]]) == {1: [10, 11], 3: [30]}


def test_wrapping_an_all_empty_request_yields_nothing_to_ask_about():
    assert v2.wrap_groups_for_engine([[], []]) == {}


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
