"""Free-at-promotion: turning landing-time copies into real frees, without leaving full graph.

The copy at landing keeps every read correct but frees nothing; capacity needs the victim block back.
The scheduler can swap victim→representative before the request's first schedule (the worker then
receives the rep's id as part of the request's own table, so nothing is written after graph
capture). It may only do so when it can PROVE the swap is right, because a wrong swap is silent
KV substitution:

* every TP rank landed the same alias (ranks plan from their own signature shards);
* the owner is alive and loaded, still holds the rep, and has finished writing it.

Anything unproven is refused, and the request keeps its already-valid copy.
"""

import types

from kv_fast_fusion import fast_fusion_scheduler as fs
from kv_fast_fusion import pd_dedup_v2
from kv_fast_fusion_ascend import fast_fusion_ascend_patch as patch
from kv_fast_fusion_ascend import landed_promotion as lp
from kv_fast_fusion_ascend.connectors.merge_stats import BFFMergeStats

BS = 128


# =====================================================================================
# consensus across TP ranks
# =====================================================================================
def test_an_alias_every_rank_landed_survives():
    per_rank = {0: {1: {51: (41, "rA")}}, 1: {1: {51: [41, "rA"]}}}   # list form after transport

    consensus, disagreed = lp.consensus_landed(per_rank, 2)

    assert consensus == {1: {51: (41, "rA")}}
    assert disagreed == 0


def test_an_alias_only_some_ranks_chose_is_dropped():
    per_rank = {0: {1: {51: (41, "rA"), 52: (42, "rA")}},
                1: {1: {51: (41, "rA"), 52: (49, "rC")}}}       # rank 1 chose another rep for 52

    consensus, disagreed = lp.consensus_landed(per_rank, 2)

    assert consensus == {1: {51: (41, "rA")}}
    assert disagreed == 2, "both ranks' versions of victim 52 are counted as dropped"


def test_nothing_is_decided_until_every_rank_reported():
    assert lp.consensus_landed({0: {1: {51: (41, "rA")}}}, 2) == (None, 0)


# =====================================================================================
# proving each swap from the scheduler's own state
# =====================================================================================
def _owner(blocks, computed):
    return {"rA": ([[], blocks], computed)}.get


def test_a_provable_swap_points_the_victim_at_the_rep():
    merged, refused = lp.promotion_swaps([[], [50, 51, 52]], {1: {51: (41, "rA")}},
                                         _owner([40, 41, 42], 3 * BS), BS)

    assert merged == {1: [50, 41, 52]}
    assert sum(refused.values()) == 0


def test_a_finished_owner_refuses_the_swap():
    merged, refused = lp.promotion_swaps([[], [50, 51]], {1: {51: (41, "rA")}},
                                         {}.get, BS)

    assert merged == {} and refused["owner_gone"] == 1


def test_an_owner_still_loading_refuses_the_swap():
    merged, refused = lp.promotion_swaps([[], [50, 51]], {1: {51: (41, "rA")}},
                                         {"rA": "loading"}.get, BS)

    assert merged == {} and refused["owner_loading"] == 1


def test_a_rep_no_longer_in_its_owner_s_table_refuses_the_swap():
    """Preempted or reissued since the worker copied it: the id is not that KV any more."""
    merged, refused = lp.promotion_swaps([[], [50, 51]], {1: {51: (41, "rA")}},
                                         _owner([40, 43], 2 * BS), BS)

    assert merged == {} and refused["rep_moved"] == 1


def test_a_rep_its_owner_is_still_writing_refuses_the_swap():
    """The owner's frontier block keeps receiving its decode's tokens; sharing it would hand those
    to the victim. Position 1 of the owner's table is unfinished at 200 computed tokens."""
    merged, refused = lp.promotion_swaps([[], [50, 51]], {1: {51: (41, "rA")}},
                                         _owner([40, 41], 200), BS)

    assert merged == {} and refused["rep_hot"] == 1


def test_a_victim_not_in_the_request_s_table_refuses_the_swap():
    merged, refused = lp.promotion_swaps([[], [50, 52]], {1: {51: (41, "rA")}},
                                         _owner([40, 41], 2 * BS), BS)

    assert merged == {} and refused["victim_moved"] == 1


def test_the_block_size_is_read_per_group():
    merged, _ = lp.promotion_swaps([[], [50, 51]], {1: {51: (41, "rA")}},
                                   _owner([41], 256), lambda gi: 256 if gi == 1 else 0)

    assert merged == {1: [50, 41]}


# =====================================================================================
# the worker's half: what landing copied, and the carrier across ranks
# =====================================================================================
def test_landing_reports_exactly_what_it_copied():
    import torch

    from kv_fast_fusion import pd_lsh
    engine = pd_dedup_v2.DedupEngine(resident=True)
    m = torch.randn(2, 4)
    n = m.norm(dim=1)
    sig = m / n.unsqueeze(1)
    engine.note_resident(1, sig, pd_lsh.sub_hashes(sig, pd_lsh.get_proj([None], 4, sig.device)),
                         n.tolist(), [41, 42], owner="rA")
    engine._pending_alias = {"rB": {1: {51: (41, "rA", None), 52: (42, "rA", None)}}}
    landed = {}

    engine.release_materialized("rB", lambda gi, pairs: True, hot_victims={1: 52}, landed=landed)

    assert landed == {1: {51: (41, "rA")}}, "the refused hot victim is not reported as copied"


def test_the_carrier_unions_landed_maps_rank_by_rank():
    acc = BFFMergeStats(data={"bff_landed": {"rB": {0: {1: {51: (41, "rA")}}}}})
    acc.aggregate(BFFMergeStats(data={"bff_landed": {"rB": {1: {1: {51: (41, "rA")}}}}}))

    assert set(acc.data["bff_landed"]["rB"]) == {0, 1}
    assert not acc.is_empty()


# =====================================================================================
# the scheduler glue, end to end against the real merge handler
# =====================================================================================
class _Block:
    def __init__(self, bid):
        self.block_id, self.ref_cnt, self.is_null, self.block_hash = bid, 1, False, None


class _Pool:
    enable_caching = False

    def __init__(self, n):
        self.blocks = [_Block(i) for i in range(n)]
        self.free = 0

    def get_num_free_blocks(self):
        return self.free

    def touch(self, blocks):
        for b in blocks:
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free += 1


def _scheduler(owner_computed=3 * BS):
    from vllm.v1.request import RequestStatus
    pool = _Pool(64)
    g0 = types.SimpleNamespace(req_to_blocks={}, num_cached_block={}, block_size=BS)
    g1 = types.SimpleNamespace(req_to_blocks={}, num_cached_block={}, block_size=BS)
    g1.req_to_blocks["rA-1"] = [pool.blocks[i] for i in (40, 41, 42)]
    g1.req_to_blocks["rB-1"] = [pool.blocks[i] for i in (50, 51, 52)]
    g1.num_cached_block["rB-1"] = 3
    reqs = {
        "rA-1": types.SimpleNamespace(status=RequestStatus.RUNNING,
                                      num_computed_tokens=owner_computed),
        "rB-1": types.SimpleNamespace(status=RequestStatus.WAITING,
                                      num_computed_tokens=3 * BS, request_id="rB-1"),
    }
    sched = types.SimpleNamespace(
        requests=reqs,
        vllm_config=types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(tensor_parallel_size=2)),
        kv_cache_manager=types.SimpleNamespace(coordinator=types.SimpleNamespace(
            single_type_managers=[g0, g1], block_pool=pool)))
    sched._handle_block_merging_with_counts = types.MethodType(
        fs._handle_block_merging_with_counts, sched)
    sched._bff_landed = {"rB": {0: {1: {51: (41, "rA")}}, 1: {1: {51: (41, "rA")}}}}
    return sched, pool, g1


def test_promotion_frees_the_victim_and_shares_the_rep(monkeypatch):
    monkeypatch.setattr(patch, "_ext_of", lambda rid: rid.rsplit("-", 1)[0])
    sched, pool, g1 = _scheduler()

    patch._bff_landed_promotion_apply(sched, sched.requests["rB-1"])

    assert [b.block_id for b in g1.req_to_blocks["rB-1"]] == [50, 41, 52]
    assert pool.blocks[51].ref_cnt == 0 and pool.free == 1, "the victim is back in the pool"
    assert pool.blocks[41].ref_cnt == 2, "the rep is held by its owner and the victim's request"
    assert sched._bff_landed == {}, "consumed once"


def test_promotion_keeps_the_copy_when_the_rep_is_unproven(monkeypatch):
    monkeypatch.setattr(patch, "_ext_of", lambda rid: rid.rsplit("-", 1)[0])
    sched, pool, g1 = _scheduler(owner_computed=BS + 10)     # owner still writing block 41

    patch._bff_landed_promotion_apply(sched, sched.requests["rB-1"])

    assert [b.block_id for b in g1.req_to_blocks["rB-1"]] == [50, 51, 52]
    assert pool.free == 0


def test_the_ingest_stashes_maps_until_promotion():
    sched = types.SimpleNamespace(requests={})
    out = types.SimpleNamespace(kv_connector_output=types.SimpleNamespace(
        kv_connector_stats=BFFMergeStats(data={"bff_landed": {"rB": {0: {1: {51: (41, "rA")}}}}})))

    assert patch._bff_ingest_landed(sched, out) == 1
    assert sched._bff_landed == {"rB": {0: {1: {51: (41, "rA")}}}}
