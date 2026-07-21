"""cache_full_blocks must tolerate blocks that already carry a hash.

BFF's `add_block_alias` (BFF_ALIAS_FUSED, lever 3) registers a representative block under ANOTHER
request's prefix-cache key, so a later request can prefix-hit a block that already carries its
original owner's hash. Stock `BlockPool.cache_full_blocks` asserts every block in its range is
unhashed, so that hit kills EngineCore — observed on NPU via
`_update_waiting_for_remote_kv -> cache_blocks -> assert blk.block_hash is None`.

`patched_cache_full_blocks` skips already-hashed blocks by handing each maximal run of UNHASHED
blocks to the stock implementation (never a copy of it, so it survives vLLM version drift). These
tests drive the segmentation directly with a recording stub in place of stock: they assert the
absolute bounds are right and that stock never sees a hashed block.

Runs off-cluster/off-NPU. pytest is not installed in the venv, so:
    PYTHONPATH=<repo root> python kv_fast_fusion/tests/test_cache_full_blocks_alias.py
"""
import types

import kv_fast_fusion.fast_fusion_block_pool as m


class _RecordingStock:
    """Stands in for the captured stock BlockPool.cache_full_blocks."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = []          # (num_cached_blocks, num_full_blocks) per invocation

    def __call__(self, pool, request, blocks, num_cached_blocks, num_full_blocks,
                 block_size, kv_cache_group_id):
        # The whole point of the wrapper: stock must never receive an already-hashed block.
        for i in range(num_cached_blocks, num_full_blocks):
            assert blocks[i].block_hash is None, (
                f"stock received an already-hashed block at absolute index {i}")
        self.calls.append((num_cached_blocks, num_full_blocks))


def _blocks(n, hashed_idx=()):
    return [types.SimpleNamespace(block_id=i, block_hash=("h" if i in hashed_idx else None))
            for i in range(n)]


def _run(n, hashed_idx, num_cached=0, num_full=None):
    """Invoke the wrapper over blocks[num_cached:num_full]; return (calls, skip delta)."""
    blocks = _blocks(n, hashed_idx)
    stock = _RecordingStock(blocks)
    orig, before = m._ORIG_CACHE_FULL, m._alias_hash_skips
    m._ORIG_CACHE_FULL = stock
    try:
        m.patched_cache_full_blocks(
            object(), object(), blocks, num_cached,
            n if num_full is None else num_full, 16, 0)
    finally:
        m._ORIG_CACHE_FULL = orig
    return stock.calls, m._alias_hash_skips - before


def test_no_hashed_blocks_is_a_single_passthrough():
    # The common case must be exactly one stock call over the original range — no behavior change.
    calls, skips = _run(5, hashed_idx=())
    assert calls == [(0, 5)], calls
    assert skips == 0


def test_hashed_prefix_skips_to_the_remainder():
    # The alias-hit shape: get_computed_blocks returns a contiguous already-cached prefix.
    calls, skips = _run(5, hashed_idx={0, 1})
    assert calls == [(2, 5)], calls
    assert skips == 2


def test_hashed_block_in_the_middle_splits_the_range():
    calls, skips = _run(6, hashed_idx={2})
    assert calls == [(0, 2), (3, 6)], calls
    assert skips == 1


def test_all_hashed_makes_no_stock_call():
    calls, skips = _run(3, hashed_idx={0, 1, 2})
    assert calls == [], calls
    assert skips == 3


def test_hashed_suffix_keeps_the_leading_run():
    calls, skips = _run(4, hashed_idx={3})
    assert calls == [(0, 3)], calls
    assert skips == 1


def test_respects_a_nonzero_num_cached_blocks():
    # Blocks below num_cached_blocks are already cached and must not be inspected or re-sent,
    # even though block 0 is hashed.
    calls, skips = _run(6, hashed_idx={0, 4}, num_cached=2)
    assert calls == [(2, 4), (5, 6)], calls
    assert skips == 1          # only block 4 is in range; block 0 is below num_cached


def test_empty_range_returns_without_calling_stock():
    calls, skips = _run(4, hashed_idx=(), num_cached=3, num_full=3)
    assert calls == []
    assert skips == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} passed")
