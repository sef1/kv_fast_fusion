"""The scan bounds what BFF could free, so its three distinctions must be exact: a near-duplicate
counts only while both prompts are live; a matching prefix chain is prefix caching's, not BFF's;
and distinct blocks count as nothing. CPU only, on raw token ids."""

from f1_benchmark import scan_block_redundancy as scan

BS = 8


def _blk(seed):
    return [seed * 100 + i for i in range(BS)]


def test_a_near_duplicate_in_a_live_prompt_counts():
    a = _blk(1) + _blk(2)
    b = _blk(9) + _blk(2)                       # different first block, so not a prefix hit

    c = scan.scan([a, b], block_size=BS, window=2, ngram=2, jaccard=0.8)

    assert c["live_near_dup"] == 1 and c["live_exact_dup"] == 1
    assert c.get("prefix_cached", 0) == 0


def test_a_duplicate_that_has_left_the_window_does_not_count():
    a = _blk(1) + _blk(2)
    filler = _blk(5) + _blk(6)
    b = _blk(9) + _blk(2)

    c = scan.scan([a, filler, b], block_size=BS, window=2, ngram=2, jaccard=0.8)

    assert c.get("live_near_dup", 0) == 0


def test_a_shared_prefix_is_prefix_caching_not_bff():
    a = _blk(1) + _blk(2)
    b = _blk(1) + _blk(2)

    c = scan.scan([a, b], block_size=BS, window=2, ngram=2, jaccard=0.8)

    assert c["prefix_cached"] == 2
    assert c.get("live_near_dup", 0) == 0


def test_distinct_blocks_count_nothing():
    c = scan.scan([_blk(1) + _blk(2), _blk(3) + _blk(4)], block_size=BS, window=2, ngram=2,
                  jaccard=0.8)

    assert c["blocks"] == 4 and c.get("live_near_dup", 0) == 0


def test_a_near_but_not_exact_block_counts_as_near_only():
    base = _blk(2)
    near = base[:-1] + [77777]                  # one token changed: most 2-grams survive

    c = scan.scan([_blk(1) + base, _blk(9) + near], block_size=BS, window=2, ngram=2,
                  jaccard=0.7)

    assert c["live_near_dup"] == 1 and c.get("live_exact_dup", 0) == 0
