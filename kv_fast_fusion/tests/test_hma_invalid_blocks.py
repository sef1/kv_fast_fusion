"""Unit tests for the HMA-aware `_update_requests_with_invalid_blocks`
(`kv_fast_fusion.fast_fusion_scheduler`).

vLLM's own version opens with a single-group assumption it documents as a TODO:

    # TODO (davidb): add support for hybrid memory allocator
    (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)

Under BFF a request has one block list per KV-cache group (7 in the default split), so that unpack
raises `ValueError: too many values to unpack (expected 1)` and kills EngineCore — which is what
happened the first time a Mooncake KV pull failed (2026-08-13). These tests pin both that the
multi-group case works and that the single-group case still behaves exactly like stock.

CPU only; no GPU, no engine, no transport.
"""

import types

import pytest

from kv_fast_fusion.fast_fusion_scheduler import _update_requests_with_invalid_blocks
from vllm.v1.request import RequestStatus

BLOCK = 16


def _sched(block_ids_per_req):
    """A stand-in scheduler: `block_ids_per_req[req_id]` is the per-group block-id tuple."""
    return types.SimpleNamespace(
        block_size=BLOCK,
        kv_cache_manager=types.SimpleNamespace(
            get_block_ids=lambda req_id: block_ids_per_req[req_id]),
    )


def _req(req_id, computed_tokens, status=RequestStatus.WAITING_FOR_REMOTE_KVS):
    return types.SimpleNamespace(
        request_id=req_id, status=status,
        num_computed_tokens=computed_tokens, num_cached_tokens=0,
        num_external_computed_tokens=computed_tokens)


def _run(sched, requests, invalid, evict_blocks=True):
    return _update_requests_with_invalid_blocks.__get__(sched)(
        requests, invalid, evict_blocks)


def test_multi_group_does_not_raise_and_truncates_at_the_bad_block():
    """The crash case: 7 groups. Block index 2 of group 5 failed to load, so the request's computed
    tokens must be truncated to 2 * block_size."""
    groups = tuple([100 * g + i for i in range(4)] for g in range(7))
    sched = _sched({"r0": groups})
    req = _req("r0", computed_tokens=4 * BLOCK)

    affected, tokens, evicted = _run(sched, [req], {502})

    assert affected == {"r0"}
    assert req.num_computed_tokens == 2 * BLOCK, "truncated at the failed position"
    assert tokens == 2 * BLOCK
    assert 502 in evicted and 503 in evicted, "the bad block and its tail"
    assert 2 in evicted and 3 in evicted, "the SAME tail in every other group too"


def test_single_group_matches_stock_behaviour():
    sched = _sched({"r0": ([10, 11, 12, 13],)})
    req = _req("r0", computed_tokens=4 * BLOCK)

    affected, tokens, evicted = _run(sched, [req], {12})

    assert affected == {"r0"}
    assert req.num_computed_tokens == 2 * BLOCK
    assert tokens == 2 * BLOCK
    assert evicted == {12, 13}


def test_earliest_affected_position_across_groups_wins():
    """Group 3 fails at index 1, group 0 at index 3 — the truncation must use the earlier one."""
    groups = tuple([100 * g + i for i in range(4)] for g in range(4))
    sched = _sched({"r0": groups})
    req = _req("r0", computed_tokens=4 * BLOCK)

    _run(sched, [req], {301, 3})

    assert req.num_computed_tokens == 1 * BLOCK


def test_unaffected_request_is_untouched():
    sched = _sched({"r0": ([10, 11], [20, 21])})
    req = _req("r0", computed_tokens=2 * BLOCK)

    affected, tokens, evicted = _run(sched, [req], {999})

    assert affected == set() and tokens == 0 and evicted == set()
    assert req.num_computed_tokens == 2 * BLOCK


def test_blocks_beyond_the_computed_range_are_ignored():
    """Only blocks that may hold externally computed tokens count."""
    sched = _sched({"r0": ([10, 11, 12, 13],)})
    req = _req("r0", computed_tokens=2 * BLOCK)     # only indices 0..1 are computed

    affected, _tokens, _evicted = _run(sched, [req], {13})

    assert affected == set()


def test_evict_blocks_false_collects_nothing():
    """Async loads are not cached yet, so nothing is evicted."""
    sched = _sched({"r0": ([10, 11], [20, 21])})
    req = _req("r0", computed_tokens=2 * BLOCK)

    _affected, _tokens, evicted = _run(sched, [req], {20}, evict_blocks=False)

    assert evicted == set()


def test_a_block_shared_with_an_earlier_request_is_only_recomputed_once():
    """Stock semantics: the second request keeps the block as computed, since the first will
    recompute it — but it is still reported as affected."""
    sched = _sched({"r0": ([10, 11],), "r1": ([10, 11],)})
    r0, r1 = _req("r0", 2 * BLOCK), _req("r1", 2 * BLOCK)

    affected, _tokens, _evicted = _run(sched, [r0, r1], {11})

    assert affected == {"r0", "r1"}
    assert r0.num_computed_tokens == 1 * BLOCK, "first request truncates"
    assert r1.num_computed_tokens == r1.num_cached_tokens, "second falls back to cached only"


def test_groups_of_unequal_length_are_tolerated():
    """BFF's group 0 is a sliding-window group, so its block list can be shorter."""
    sched = _sched({"r0": ([10, 11], [20, 21, 22, 23])})
    req = _req("r0", computed_tokens=4 * BLOCK)

    affected, _tokens, _evicted = _run(sched, [req], {23})

    assert affected == {"r0"}
    assert req.num_computed_tokens == 3 * BLOCK


@pytest.mark.parametrize("n_groups", [1, 2, 7])
def test_never_raises_for_any_group_count(n_groups):
    """The regression itself: stock raised ValueError for n_groups != 1."""
    groups = tuple([100 * g + i for i in range(3)] for g in range(n_groups))
    sched = _sched({"r0": groups})
    _run(sched, [_req("r0", 3 * BLOCK)], {101 if n_groups > 1 else 1})
