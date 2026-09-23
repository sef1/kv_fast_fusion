"""sharing_stats (Update 32): is the sharing dedup creates BATCHABLE?

The user's Hydragen question (arXiv 2402.05099): aliasing makes requests share physical blocks, so why
not split attention into a shared part (one matrix-matrix GEMM over the shared keys) and a unique
part? Today sharing saves memory but no bandwidth — paged attention re-reads a shared block once per
sharer — which is why balanced-4's +7.2 % concurrency came back as −6.8 % per-request efficiency.

Two numbers decide whether a split could ever pay, and they are NOT the same number:
  * `redundancy`  — every re-read, the ceiling on recoverable bandwidth;
  * `batchable`   — only re-reads whose sharers sit at the SAME slot. Keys are stored post-RoPE and
                    the mask is positional, so sharers at different positions cannot share a GEMM.
A high `redundancy` with a low `batchable` means the sharing is real and the idea is still dead —
the distinction `wire_saving_pct` cannot make, and the trap this file exists to keep us out of.

Pure dict arithmetic over block ids: no device, no torch, no engine.
"""
from kv_fast_fusion.pd_dedup_v2 import sharing_stats


def test_no_sharing_is_zero_redundancy_and_no_divide_by_zero():
    s = sharing_stats({"a": [[1, 2, 3]], "b": [[4, 5, 6]]})
    assert s["refs"] == 6 and s["distinct"] == 6
    assert s["redundancy"] == 0.0 and s["batchable"] == 0.0
    assert s["mean_fanout_shared"] == 0.0 and s["aligned_frac"] == 0.0
    assert s["max_fanout"] == 1
    # And the empty case must not raise on the first dump of a run.
    empty = sharing_stats({})
    assert empty["refs"] == 0 and empty["redundancy"] == 0.0 and empty["max_fanout"] == 0


def test_redundancy_counts_every_re_read_of_a_shared_block():
    """Four requests holding block 9 at slot 0: 4 references, 1 distinct → 3 of 4 reads are re-reads."""
    s = sharing_stats({r: [[9]] for r in "abcd"})
    assert s["refs"] == 4 and s["distinct"] == 1
    assert s["redundancy"] == 0.75
    assert s["max_fanout"] == 4 and s["mean_fanout_shared"] == 4.0
    # All four hold it at slot 0, so all of it is batchable: one read serves four.
    assert s["aligned_frac"] == 1.0 and s["batchable"] == 0.75


def test_sharing_at_DIFFERENT_slots_is_redundant_but_NOT_batchable():
    """The whole point of the second metric. Same physical block, different positions in each
    sequence — post-RoPE keys and a positional mask mean one shared GEMM cannot serve them."""
    s = sharing_stats({"a": [[9, 1]], "b": [[2, 9]], "c": [[3, 9]]})
    assert s["refs"] == 6 and s["distinct"] == 4      # 9 at slot 0, 9 at slot 1, plus 1,2,3
    assert s["redundancy"] > 0.0, "the re-reads are real"
    # Block 9: slots [0, 1, 1] → the largest same-slot cohort is 2, so only 1 read is saveable.
    assert s["aligned_frac"] < 1.0
    assert s["batchable"] < s["redundancy"], "misaligned sharing must not be counted as batchable"
    assert s["batchable"] == round(1 / 6, 4)


def test_a_block_a_single_request_holds_twice_counts_as_the_re_read_it_is():
    """Fan-out is references, not distinct requests: attention reads the block once per slot either
    way, so a self-repeat is bandwidth a split could recover just like a cross-request share."""
    s = sharing_stats({"a": [[7, 7, 7]]})
    assert s["refs"] == 3 and s["distinct"] == 1 and s["max_fanout"] == 3
    assert s["redundancy"] == round(2 / 3, 4)


def test_the_same_block_id_in_two_GROUPS_is_two_different_blocks():
    """Groups are separate physical regions — block 5 of group 0 and block 5 of group 1 are unrelated
    memory. Pooling them would invent sharing that does not exist."""
    s = sharing_stats({"a": [[5], [5]], "b": [[5], [5]]})
    assert s["distinct"] == 2, "(group, block) is the key, not block"
    assert s["refs"] == 4 and s["redundancy"] == 0.5


def test_fanout_histogram_buckets_and_mean_over_shared_blocks_only():
    per_req = {
        "a": [[1, 2, 3, 10]],
        "b": [[2, 3, 10]],
        "c": [[3, 10]],
        "d": [[10]],
        "e": [[10]],
    }
    s = sharing_stats(per_req)
    # block 1 x1, block 2 x2, block 3 x3, block 10 x5
    assert s["fanout"]["1"] == 1 and s["fanout"]["2"] == 1
    assert s["fanout"]["3-4"] == 1 and s["fanout"]["5-8"] == 1
    assert s["fanout"]["9-16"] == 0 and s["fanout"]["17+"] == 0
    assert s["max_fanout"] == 5
    # mean over SHARED blocks (2,3,10) = (2+3+5)/3, not diluted by the unshared one.
    assert s["mean_fanout_shared"] == round(10 / 3, 2)


def test_ragged_and_malformed_group_lists_are_tolerated():
    """The dump reads live runner state; a request mid-allocation can have empty or non-list groups,
    and a profiling helper must never be the thing that breaks the stats dump."""
    s = sharing_stats({"a": [[1], [], None], "b": [[1]], "c": []})
    assert s["refs"] == 2 and s["distinct"] == 1 and s["redundancy"] == 0.5


# =====================================================================================
# max_aligned_fanout and per_group (Update 39). `batchable` is an aggregate; what a shared/unique
# attention split is actually worth is the largest cohort holding ONE block at ONE slot, because
# serving a cohort of m costs one read plus a log-sum-exp merge. At m=2 the split is strictly worse
# than re-reading, whatever `redundancy` says — our balanced-4 runs show one hub block at fan-out
# 1,046-2,506 whose sharers sit at different slots, which is why redundancy ~12 % is batchable ~2 %.
# =====================================================================================
def test_max_aligned_fanout_is_the_cohort_at_ONE_slot_not_the_fanout():
    """Five requests hold block 7, but only three of them at the same slot — a shared GEMM can serve
    those three, and re-reading is all that is left for the other two."""
    per_req = {                    # distinct filler ids, so block 7 is the only shared one
        "a": [[7, 101, 102]],      # block 7 at slot 0
        "b": [[7, 103, 104]],      # slot 0
        "c": [[7, 105, 106]],      # slot 0
        "d": [[107, 7, 108]],      # slot 1
        "e": [[109, 110, 7]],      # slot 2
    }
    s = sharing_stats(per_req)
    assert s["max_fanout"] == 5, "five references to block 7"
    assert s["max_aligned_fanout"] == 3, "but the largest single-slot cohort is three"
    assert s["aligned_fanout"]["3-4"] == 1


def test_a_hub_block_shared_at_every_different_slot_is_worth_nothing_to_a_split():
    """The measured shape on balanced-4: huge fan-out, no two sharers aligned. redundancy is high and
    the split is still dead — the distinction wire_saving_pct cannot make."""
    # block 99 sits at a different slot in each request; every other id is unique to its request
    per_req = {f"r{i}": [[1000 * i + j for j in range(i)] + [99]] for i in range(1, 9)}
    s = sharing_stats(per_req)
    assert s["max_fanout"] == 8, "eight requests read block 99"
    assert s["max_aligned_fanout"] == 1, "no two of them at the same slot — no cohort at all"
    assert s["redundancy"] > 0.1 and s["batchable"] == 0.0


def test_per_group_splits_the_headline_and_sums_back_to_the_pool():
    """Only some groups are dedup-eligible and the layouts differ 4x on batchable, so which group
    carries the aligned sharing is the whole question."""
    per_req = {
        "a": [[1, 2], [5, 6]],
        "b": [[1, 2], [7, 8]],     # group 0 fully aligned with a; group 1 shares nothing
        "c": [[1, 2], [9, 10]],
    }
    s = sharing_stats(per_req)
    assert set(s["per_group"]) == {"0", "1"}
    assert sum(g["refs"] for g in s["per_group"].values()) == s["refs"]
    assert s["per_group"]["0"]["max_aligned_fanout"] == 3, "three requests hold 1 and 2 at one slot"
    assert s["per_group"]["1"]["max_aligned_fanout"] == 0, "no shared block at all in group 1"
    assert s["per_group"]["1"]["batchable"] == 0.0
    assert s["per_group"]["0"]["batchable"] > 0.5


def test_the_new_fields_survive_an_empty_and_a_single_request_batch():
    for per_req in ({}, {"solo": [[1, 2, 3]]}):
        s = sharing_stats(per_req)
        assert s["max_aligned_fanout"] == 0
        assert all(v == 0 for v in s["aligned_fanout"].values())
        assert isinstance(s["per_group"], dict)
