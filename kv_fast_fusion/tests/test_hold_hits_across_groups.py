"""No group may take another group's unclaimed prefix hit as a new block (vLLM issue #33775). CPU only.

The coordinator below is vLLM 0.18.0's loop: per group, touch its hits, then take its external blocks.
"""

from collections import deque

import pytest

from kv_fast_fusion.fast_fusion_block_pool import install_hold_hits_across_groups


class Block:
    def __init__(self, i, is_null=False):
        self.block_id, self.ref_cnt, self.is_null = i, 0, is_null


class Pool:
    def __init__(self, n):
        self.blocks = [Block(i) for i in range(n)]
        self.free = deque(self.blocks)

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free.remove(b)
            b.ref_cnt += 1

    def get_new_blocks(self, n):
        assert n <= len(self.free), "out of blocks"
        out = [self.free.popleft() for _ in range(n)]
        for b in out:
            b.ref_cnt += 1
        return out

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0 and not b.is_null:
                self.free.append(b)


def coordinator(n_groups, pool, fail_at=None):
    class Coordinator:                       # a fresh class each time: the install wraps the class
        def __init__(self):
            self.single_type_managers, self.block_pool = list(range(n_groups)), pool
            self.tables = [[] for _ in range(n_groups)]

        def allocate_new_computed_blocks(self, request_id, new_computed_blocks, n_local, n_external):
            for g in self.single_type_managers:
                if g == fail_at:
                    raise RuntimeError("boom")
                self.block_pool.touch(new_computed_blocks[g])
                self.tables[g] = (list(new_computed_blocks[g])
                                  + self.block_pool.get_new_blocks(n_external))
    return Coordinator


def scenario():
    """The box's shape: group 3's hits are at the head of the free queue, in reverse order."""
    pool = Pool(40)
    hits = [[pool.blocks[10 + 3 * g + i] for i in range(3)] for g in range(4)]
    head = list(reversed(hits[3]))
    pool.free = deque(head + [b for b in pool.blocks if b not in head])
    return pool, hits


def ids(c):
    return [b.block_id for t in c.tables for b in t]


def test_without_the_hold_one_block_lands_in_two_groups():
    pool, hits = scenario()
    c = coordinator(4, pool)()
    c.allocate_new_computed_blocks("r", hits, 3, 1)
    assert len(set(ids(c))) < len(ids(c)), "the bug this guards against"
    assert any(b.ref_cnt == 2 for b in pool.blocks)


def test_with_the_hold_every_block_is_used_once_and_counted_once():
    pool, hits = scenario()
    cls = coordinator(4, pool)
    assert install_hold_hits_across_groups(cls) is True
    c = cls()
    c.allocate_new_computed_blocks("r", hits, 3, 1)
    assert len(set(ids(c))) == len(ids(c)) == 16
    assert [b.ref_cnt for t in c.tables for b in t] == [1] * 16
    assert len(pool.free) == 24 and all(b.ref_cnt == 0 for b in pool.free)


def test_a_hit_shared_with_a_running_request_keeps_that_reference():
    pool, hits = scenario()
    pool.touch([hits[3][0]])
    cls = coordinator(4, pool)
    install_hold_hits_across_groups(cls)
    cls().allocate_new_computed_blocks("r", hits, 3, 1)
    assert hits[3][0].ref_cnt == 2


def test_exactly_enough_blocks_is_still_enough():
    pool, hits = scenario()
    need = {b.block_id for row in hits for b in row}
    spare = [b for b in pool.free if b.block_id not in need][:4]
    pool.free = deque([b for b in pool.free if b.block_id in need or b in spare])
    cls = coordinator(4, pool)
    install_hold_hits_across_groups(cls)
    c = cls()
    c.allocate_new_computed_blocks("r", hits, 3, 1)
    assert len(set(ids(c))) == 16 and not pool.free


def test_a_failure_leaves_no_reference_behind():
    pool, hits = scenario()
    cls = coordinator(4, pool, fail_at=0)
    install_hold_hits_across_groups(cls)
    with pytest.raises(RuntimeError):
        cls().allocate_new_computed_blocks("r", hits, 3, 1)
    assert all(b.ref_cnt == 0 for b in pool.blocks) and len(pool.free) == 40


def test_one_group_and_no_hits_go_straight_through_and_null_blocks_are_left_alone():
    pool = Pool(8)
    cls = coordinator(1, pool)
    install_hold_hits_across_groups(cls)
    calls = []
    pool.touch = lambda b, t=pool.touch: (calls.append(len(b)), t(b))
    cls().allocate_new_computed_blocks("r", [[pool.blocks[3]]], 1, 1)
    assert calls == [1], "one group: only the original touch"
    pool2 = Pool(8)
    cls2 = coordinator(2, pool2)
    install_hold_hits_across_groups(cls2)
    null = Block(-1, is_null=True)
    cls2().allocate_new_computed_blocks("r", [[null], []], 1, 1)
    assert null.ref_cnt == 1, "only the original touch"


def test_installing_twice_and_on_a_two_phase_vllm_does_nothing():
    cls = coordinator(2, Pool(4))
    assert install_hold_hits_across_groups(cls) is True
    assert install_hold_hits_across_groups(cls) is False

    class TwoPhase:
        def allocate_new_computed_blocks(self, request_id, blocks, n_local, n_external):
            for i, m in enumerate(self.single_type_managers):
                m.add_local_computed_blocks(request_id, blocks[i], n_local, n_external)
    assert install_hold_hits_across_groups(TwoPhase) is False
