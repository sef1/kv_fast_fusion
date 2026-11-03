"""The shared-block write check: which blocks a step writes, and when that write is a bug. CPU only."""

import logging
import types

from kv_fast_fusion_ascend import shared_write_check as swc

BS = 128


class _Blk:
    def __init__(self, block_id, ref_cnt=1, is_null=False):
        self.block_id, self.ref_cnt, self.is_null = block_id, ref_cnt, is_null


def test_a_decode_step_writes_only_the_block_holding_its_next_token():
    assert list(swc.write_span(300, 1, BS)) == [2]


def test_a_chunk_spanning_a_boundary_writes_both_blocks():
    assert list(swc.write_span(250, 10, BS)) == [1, 2]


def test_nothing_scheduled_writes_nothing():
    assert list(swc.write_span(300, 0, BS)) == []


def test_a_write_into_an_unshared_block_is_fine():
    row = [_Blk(10), _Blk(11), _Blk(12)]
    assert swc.shared_writes(row, 300, 1, BS) == []


def test_a_write_into_a_shared_block_is_reported():
    shared = _Blk(11, ref_cnt=3)
    row = [_Blk(10, ref_cnt=2), shared, _Blk(12)]
    # A recompute of the prompt from token 0: the shared prefix block 0 AND the representative.
    got = swc.shared_writes(row, 0, 2 * BS, BS)
    assert [(p, b.block_id) for p, b in got] == [(0, 10), (1, 11)]


def test_a_read_of_a_shared_block_is_not_a_write():
    """Decoding past a shared prompt block reads it; only the frontier block is written."""
    row = [_Blk(10, ref_cnt=4), _Blk(11, ref_cnt=4), _Blk(12)]
    assert swc.shared_writes(row, 2 * BS + 5, 1, BS) == []


def test_null_and_unallocated_blocks_are_skipped():
    row = [_Blk(0, ref_cnt=9, is_null=True)]
    assert swc.shared_writes(row, 0, 3 * BS, BS) == []


def _scheduler(rows, computed):
    managers = [types.SimpleNamespace(block_size=BS, req_to_blocks=rows)]
    return types.SimpleNamespace(
        kv_cache_manager=types.SimpleNamespace(
            coordinator=types.SimpleNamespace(single_type_managers=managers)),
        requests={rid: types.SimpleNamespace(num_computed_tokens=n) for rid, n in computed.items()})


def test_check_step_names_the_writer_and_the_other_holders(caplog):
    rep = _Blk(41, ref_cnt=2)
    rows = {"writer-aaaaaaaa": [rep, _Blk(50)], "owner-bbbbbbbb": [_Blk(40), rep]}
    sched = _scheduler(rows, {"writer-aaaaaaaa": 0, "owner-bbbbbbbb": 300})
    sched._bff_swap_roles = {}
    out = types.SimpleNamespace(num_scheduled_tokens={"writer-aaaaaaaa": 200,
                                                      "owner-bbbbbbbb": 1})
    caplog.set_level(logging.INFO, logger="vllm.bff_shared_write_check")

    assert swc.check_step(sched, out) == 1
    line = next(r.getMessage() for r in caplog.records if "shared-block write" in r.getMessage())
    assert "req=writer-aaaaaaaa g=0 pos=0 block=41 ref_cnt=2" in line
    assert "other holders: owner-bbbbbbbb@1" in line


def test_check_step_reports_each_write_once(caplog):
    rep = _Blk(41, ref_cnt=2)
    sched = _scheduler({"w-aaaaaaaa": [rep], "o-bbbbbbbb": [rep]}, {"w-aaaaaaaa": 0})
    sched._bff_swap_roles = {}
    out = types.SimpleNamespace(num_scheduled_tokens={"w-aaaaaaaa": 10})
    caplog.set_level(logging.INFO, logger="vllm.bff_shared_write_check")

    swc.check_step(sched, out)
    swc.check_step(sched, out)

    assert sum("shared-block write" in r.getMessage() for r in caplog.records) == 1


def test_a_landing_copy_into_a_block_only_this_request_holds_is_fine():
    blocks = {51: _Blk(51, ref_cnt=1)}
    assert swc.landed_anomalies([[], [50, 51]], {1: {51: (41, "O")}}, blocks) == []


def test_a_landing_copy_into_a_shared_block_is_reported():
    blocks = {51: _Blk(51, ref_cnt=2)}
    assert swc.landed_anomalies([[], [50, 51]], {1: {51: (41, "O")}}, blocks) == [(1, 51, 2, True)]


def test_a_landing_copy_into_a_block_the_request_no_longer_holds_is_reported():
    blocks = {51: _Blk(51, ref_cnt=1)}
    assert swc.landed_anomalies([[], [50, 52]], {1: {51: (41, "O")}}, blocks) == [(1, 51, 1, False)]


def test_a_pull_never_targets_a_prefix_hit_block():
    hit = _Blk(41, ref_cnt=3)
    hit.block_hash = "hO"
    fresh = _Blk(60)
    fresh.block_hash = None
    assert swc.check_pull_targets(types.SimpleNamespace(request_id="r"),
                                  types.SimpleNamespace(blocks=([hit, fresh],))) == 0


def test_a_pull_into_a_shared_unhashed_block_is_reported():
    shared = _Blk(60, ref_cnt=2)
    shared.block_hash = None
    assert swc.check_pull_targets(types.SimpleNamespace(request_id="r"),
                                  types.SimpleNamespace(blocks=([shared],))) == 1


def test_install_runs_the_check_before_the_original_advances_tokens():
    seen = []

    class _Sched:
        def __init__(self):
            self.kv_cache_manager = types.SimpleNamespace(
                coordinator=types.SimpleNamespace(single_type_managers=[]))
            self.requests = {}

        def _update_after_schedule(self, out):
            seen.append("orig")

        def _preempt_request(self, request, timestamp):
            seen.append("preempt")

    orig_check = swc.check_step
    swc.check_step = lambda s, o: seen.append("check")
    try:
        assert swc.install(_Sched)
        assert not swc.install(_Sched)          # idempotent
        s = _Sched()
        s._update_after_schedule(types.SimpleNamespace(num_scheduled_tokens={}))
        s._preempt_request(types.SimpleNamespace(request_id="r", num_computed_tokens=5), 0.0)
    finally:
        swc.check_step = orig_check
    assert seen == ["check", "orig", "preempt"]
