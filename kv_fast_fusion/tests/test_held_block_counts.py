"""held_block_counts (Update 21): the live-vs-held occupancy split for dedup=1.

dedup=1 saturates the KV pool at far fewer running requests than dedup=0 on the identical pool, so a
large share is pinned OUTSIDE the running set. This helper sizes the maps that could hold it, so the
decode-log occupancy split can say `used >> live` and name which structure (resident registry / pending
aliases) to release. Pure dict arithmetic — no device, no transport.
"""
from kv_fast_fusion.pd_dedup_v2 import DedupEngine


def _engine():
    return DedupEngine(resident=True)


def test_counts_are_zero_on_a_fresh_engine():
    assert _engine().held_block_counts() == {
        "resident": 0, "pending_alias": 0, "alias_ready": 0, "pending_resident": 0,
    }


def test_resident_registry_is_counted_per_group_block():
    e = _engine()
    # (group, block_id) -> owner req_id. Each entry is one pinned physical block.
    e._resident_owner = {(1, 10): "a", (1, 11): "a", (2, 10): "b"}
    assert e.held_block_counts()["resident"] == 3


def test_pending_alias_and_alias_ready_count_victim_blocks():
    e = _engine()
    # {req_id: {group: {victim_block: (rep, owner, scale)}}}
    e._pending_alias = {"r1": {1: {100: (5, "o", None), 101: (6, "o", None)}, 2: {200: (7, "o", None)}}}
    e._alias_ready = {"r2": {1: {300: (8, "o", None)}}}
    counts = e.held_block_counts()
    assert counts["pending_alias"] == 3   # 100,101 (g1) + 200 (g2)
    assert counts["alias_ready"] == 1


def test_pending_resident_counts_the_staged_block_ids():
    e = _engine()
    # {req_id: {group: (sig, hsh, nrm, ids, kvn)}} — ids (index 3) are the staged blocks.
    e._pending_resident = {"r1": {1: (None, None, None, [10, 11, 12], [None, None, None]),
                                  2: (None, None, None, [20], [None])}}
    assert e.held_block_counts()["pending_resident"] == 4  # 3 + 1


def test_all_buckets_sum_independently():
    e = _engine()
    e._resident_owner = {(1, 1): "a"}
    e._pending_alias = {"r": {1: {2: (0, "", None)}}}
    e._alias_ready = {"r": {1: {3: (0, "", None), 4: (0, "", None)}}}
    e._pending_resident = {"r": {1: (None, None, None, [5, 6], [None, None])}}
    assert e.held_block_counts() == {
        "resident": 1, "pending_alias": 1, "alias_ready": 2, "pending_resident": 2,
    }
