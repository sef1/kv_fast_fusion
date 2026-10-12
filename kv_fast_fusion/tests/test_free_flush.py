"""FREE_FLUSH (Update 16): the decode-side KV write-race fix.

A block is freed on the host the instant its request finishes, but its last attention KV write can
still be in flight on the NPU graph stream. If the block is reallocated and the next request's KV is
RDMA'd in before that stale write lands, the stale write clobbers the fresh KV -> wrong output.
`patched_free_blocks` must, when BFF_FREE_FLUSH=1, drain the NPU ONCE before any ref-0 block re-enters
the free queue (becomes reallocatable) -- and only when something is actually being freed.

Runs off-NPU: `_flush_npu` is monkeypatched to a recorder, so no torch/NPU is needed.
"""
import types

import kv_fast_fusion.fast_fusion_block_pool as m


class _Queue:
    def __init__(self):
        self.appended = []

    def append(self, block):
        self.appended.append(block.block_id)


def _pool():
    return types.SimpleNamespace(free_block_queue=_Queue())


def _blocks(specs):
    # specs: list of (block_id, ref_cnt_after_decrement_target). patched_free_blocks decrements once,
    # so start ref_cnt one above the target we want it to reach.
    return [types.SimpleNamespace(block_id=bid, ref_cnt=target + 1, is_null=False)
            for bid, target in specs]


def _run(monkeypatch, free_flush, specs):
    calls = {"n": 0}
    monkeypatch.setattr(m, "_flush_npu", lambda: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(m, "_FREE_FLUSH", free_flush, raising=False)
    monkeypatch.setattr(m, "_FREE_FLUSH_MODE", "1" if free_flush else "0")
    monkeypatch.setattr(m, "_KEEP_FUSED_HASH", True, raising=False)  # skip _maybe_evict path
    monkeypatch.setattr(m, "_ACTIVE_RUNNER", None, raising=False)
    monkeypatch.setattr(m, "_ON_BLOCKS_FREED", [], raising=False)
    pool = _pool()
    m.patched_free_blocks(pool, _blocks(specs))
    return calls["n"], pool.free_block_queue.appended


def test_flush_fires_once_before_freeing_when_enabled(monkeypatch):
    """One ref-0 block freed -> exactly one drain, and the block still reaches the free queue."""
    n, freed = _run(monkeypatch, True, [(5, 0), (6, 0)])   # both hit ref_cnt 0
    assert n == 1, "one flush per free EVENT, not per block"
    assert freed == [5, 6]


def test_no_flush_when_disabled(monkeypatch):
    """Flag off -> never drains (the baseline path is byte-for-byte unchanged)."""
    n, freed = _run(monkeypatch, False, [(5, 0)])
    assert n == 0 and freed == [5]


def test_no_flush_when_nothing_is_actually_freed(monkeypatch):
    """A release that only drops a ref (block still held, ref_cnt > 0) must not pay a device sync."""
    n, freed = _run(monkeypatch, True, [(5, 1)])   # decrements to 1, still referenced
    assert n == 0 and freed == []


def test_a_barrier_is_on_by_default_and_it_is_the_recv_side_one():
    """The write-race fix ships ON without any flag, or the F1-0.44 corruption is latent. The default
    is the load-side barrier; BFF_FREE_FLUSH=0 is the opt-out (unset here in CI)."""
    assert m._FREE_FLUSH_MODE == "recv"


def _recv_state(monkeypatch, mode):
    calls = {"n": 0}
    monkeypatch.setattr(m, "_flush_npu", lambda: calls.__setitem__("n", calls["n"] + 1))
    monkeypatch.setattr(m, "_FREE_FLUSH_MODE", mode)
    monkeypatch.setattr(m, "_FREE_FLUSH", mode == "1")
    monkeypatch.setattr(m, "_RECV_FLUSH",
                        {"calls": 0, "ms": 0.0, "next_report": 1, "announced": False,
                         "armed": False})
    monkeypatch.setattr(m, "_KEEP_FUSED_HASH", True, raising=False)
    monkeypatch.setattr(m, "_ACTIVE_RUNNER", None, raising=False)
    monkeypatch.setattr(m, "_ON_BLOCKS_FREED", [], raising=False)
    return calls


def test_recv_mode_moves_the_barrier_from_the_free_to_the_load(monkeypatch):
    """The scheduler thread must not drain the device on a free; the recv thread drains once per
    load instead. This is the 5 ms/step the cProfile window found."""
    calls = _recv_state(monkeypatch, "recv")
    m.flush_before_load()
    m.flush_before_load()
    assert calls["n"] == 2 and m._RECV_FLUSH["calls"] == 2
    pool = _pool()
    m.patched_free_blocks(pool, _blocks([(5, 0)]))
    assert calls["n"] == 2, "once a load-side barrier exists, a free does not drain"
    assert pool.free_block_queue.appended == [5]


def test_recv_mode_still_drains_on_free_until_a_load_side_barrier_exists(monkeypatch):
    """A transport that never calls flush_before_load (the layerwise connectors) must not be left
    with no barrier at all just because recv is the default."""
    calls = _recv_state(monkeypatch, "recv")
    m.patched_free_blocks(_pool(), _blocks([(5, 0)]))
    assert calls["n"] == 1


def test_default_mode_keeps_the_barrier_on_the_free_only(monkeypatch):
    calls = _recv_state(monkeypatch, "1")
    m.flush_before_load()
    assert calls["n"] == 0, "the load-side barrier is recv-mode only"
    m.patched_free_blocks(_pool(), _blocks([(5, 0)]))
    assert calls["n"] == 1


def test_off_mode_has_no_barrier_anywhere(monkeypatch):
    calls = _recv_state(monkeypatch, "0")
    m.patched_free_blocks(_pool(), _blocks([(5, 0)]))
    m.flush_before_load()
    assert calls["n"] == 0


def test_the_live_mode_is_announced_once(monkeypatch):
    _recv_state(monkeypatch, "recv")
    said = []
    monkeypatch.setattr(m, "_flush_log", lambda: types.SimpleNamespace(
        info=lambda fmt, *a: said.append(fmt % a)))
    m.flush_before_load()
    m.flush_before_load()
    assert sum("free-flush mode: recv" in x for x in said) == 1
    assert any("free-flush(recv) | barriers=1" in x for x in said)
