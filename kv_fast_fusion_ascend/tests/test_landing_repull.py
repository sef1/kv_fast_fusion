"""The glue around landing-time materialize on the Ascend pull connector.

Two channels the fix depends on, both testable without an NPU:

* the declined pairs and their per-block descriptors, which let a refused alias be RE-READ from the
  producer instead of recomputed on the decode (a recompute re-prefills the whole prompt through the
  decode's token budget and stalls every running request);
* ``FreedBlockLog``, the scheduler's frees shipped on the step metadata. The dedup index lives in
  the worker and the pool frees only in the scheduler, so at TP>1 nothing else tells the index that
  a representative was reissued.
"""

import types

import pytest

import kv_fast_fusion.fast_fusion_block_pool as bp
from kv_fast_fusion_ascend.connectors import mooncake_connector_ff as v1
from kv_fast_fusion_ascend.connectors import mooncake_connector_ff_v2 as v2

SENT = -1


# =====================================================================================
# declined_pairs — the blocks NOT read, in both id spaces
# =====================================================================================
def test_each_declined_position_pairs_d_s_id_with_p_s():
    planned = [10, SENT, 12, SENT]
    local = [10, 11, 12, 13]
    remote = [70, 71, 72, 73]

    assert v2.declined_pairs(planned, local, remote) == {11: 71, 13: 73}


def test_the_local_id_is_read_from_the_local_list_never_the_planned_one():
    """``planned`` holds the sentinel at a declined position; reading the id from it would key the
    re-read on -1."""
    assert v2.declined_pairs([SENT], [11], [71]) == {11: 71}


def test_nothing_declined_is_an_empty_map():
    assert v2.declined_pairs([10, 11], [10, 11], [70, 71]) == {}


def test_a_short_companion_drops_the_position_rather_than_mis_indexing():
    assert v2.declined_pairs([SENT, SENT], [11, 12], [71]) == {11: 71}


def test_declined_pairs_agree_with_filter_sentinels():
    """Every position is either transferred (filter_sentinels keeps it) or declined (here) —
    never both, never neither."""
    planned = [10, SENT, 12, SENT, 14]
    local = [10, 11, 12, 13, 14]
    remote = [70, 71, 72, 73, 74]

    kept_local, kept_remote = v2.filter_sentinels(planned, remote)
    declined = v2.declined_pairs(planned, local, remote)

    assert sorted(list(kept_local) + list(declined)) == local
    assert sorted(list(kept_remote) + list(declined.values())) == remote


# =====================================================================================
# repull_segments — a re-read restores every address of the block
# =====================================================================================
def test_a_refused_victim_is_re_read_at_every_address_it_spans():
    descs = {(1, 51): [(0x100, 0x900, 64), (0x200, 0xA00, 64)],
             (1, 52): [(0x300, 0xB00, 64)]}

    srcs, dsts, lens, missing = v2.repull_segments(descs, [(1, 51)])

    assert (srcs, dsts, lens) == ([0x100, 0x200], [0x900, 0xA00], [64, 64])
    assert missing == []


def test_a_victim_without_descriptors_is_reported_missing():
    srcs, _dsts, _lens, missing = v2.repull_segments({(1, 51): [(1, 2, 3)]}, [(1, 51), (2, 60)])

    assert missing == [(2, 60)]
    assert srcs == [1]


def test_no_descriptors_at_all_misses_everything():
    assert v2.repull_segments(None, [(1, 51)])[3] == [(1, 51)]


# =====================================================================================
# FreedBlockLog — scheduler frees, carried to the worker on the step metadata
# =====================================================================================
def test_the_log_drains_what_it_noted_and_starts_empty_again():
    log = v1.FreedBlockLog()
    log.note([5, 6])
    log.note((7,))

    assert log.drain() == [5, 6, 7]
    assert log.drain() == []


def test_the_log_sees_every_block_the_patched_pool_frees(monkeypatch):
    """End to end through the real patched_free_blocks: only blocks that reach ref_cnt 0 are frees,
    and those are what the worker's index must forget."""
    monkeypatch.setattr(bp, "_ON_BLOCKS_FREED", [], raising=False)
    monkeypatch.setattr(bp, "_FREE_FLUSH_MODE", "0", raising=False)
    monkeypatch.setattr(bp, "_KEEP_FUSED_HASH", True, raising=False)
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", None, raising=False)
    log = v1.FreedBlockLog()
    assert log.install()

    pool = types.SimpleNamespace(free_block_queue=types.SimpleNamespace(append=lambda b: None))
    blocks = [types.SimpleNamespace(block_id=5, ref_cnt=1, is_null=False),
              types.SimpleNamespace(block_id=6, ref_cnt=2, is_null=False)]   # still referenced
    bp.patched_free_blocks(pool, blocks)

    assert log.drain() == [5]


@pytest.mark.skipif(not hasattr(v1, "MooncakeConnectorMetadataFF"),
                    reason="the metadata class needs vllm_ascend")
def test_the_metadata_carries_freed_blocks_by_default():
    assert v1.MooncakeConnectorMetadataFF().freed_blocks == []
