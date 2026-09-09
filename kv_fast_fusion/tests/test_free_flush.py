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


def test_free_flush_is_on_by_default():
    """The write-race fix ships ON: a stock BFF run must drain before reallocation without any flag,
    or the F1-0.44 corruption is latent. BFF_FREE_FLUSH=0 is the opt-out (unset here in CI)."""
    assert m._FREE_FLUSH is True
