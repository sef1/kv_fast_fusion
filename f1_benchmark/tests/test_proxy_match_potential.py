"""proxy_match_potential's replay: the real planner, fed synthetic block signatures."""
import torch

from f1_benchmark import proxy_match_potential as pm


def _norm(x):
    n = x.norm(dim=1)
    return x / n.unsqueeze(1), n


def _prompt(blocks, first_token):
    """Token ids: block k holds token value blocks[k]; the first token is set apart."""
    ids = [v for b in blocks for v in [b] * pm.BLOCK]
    ids[0] = first_token
    return ids


def test_middle_layers_scale_the_box_split():
    assert pm.middle_layers(79) == list(range(20, 59))
    assert pm.middle_layers(27) == list(range(7, 20))


def test_prefix_blocks_counts_identical_leading_blocks():
    a = _prompt([1, 2, 3], 0)
    b = _prompt([1, 2, 9], 0)
    c = _prompt([1, 2, 3], 5)
    assert pm.prefix_blocks(b, [a]) == 2
    assert pm.prefix_blocks(c, [a]) == 0
    assert pm.prefix_blocks(a, []) == 0


def test_repeated_blocks_after_a_different_start_are_dropped_and_others_kept():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(4, 64, generator=g)
    other = torch.randn(4, 64, generator=g)
    s1, n1 = _norm(base)
    # Request 2: different first block, then the same three blocks plus small noise.
    b2 = base.clone()
    b2[0] = torch.randn(64, generator=g)
    b2[1:] += 0.05 * torch.randn(3, 64, generator=g)
    s2, n2 = _norm(b2)
    s3, n3 = _norm(other)
    data = {"ids": [_prompt([1, 2, 3, 4], 0), _prompt([7, 2, 3, 4], 0), _prompt([8, 9, 10, 11], 0)],
            "sigs": [s1, s2, s3], "norms": [n1, n2, n3]}
    t = pm.simulate_one(data, live=8, cache=64)
    assert t["blocks"] == 12 and t["prefix"] == 0 and t["planned"] == 12
    assert t["dropped"] == 3
    assert pm.within(t["cells"], 0.75, 0.5) == 3


def test_partner_gone_from_the_live_window_matches_nothing():
    g = torch.Generator().manual_seed(1)
    base = torch.randn(3, 64, generator=g)
    s, n = _norm(base)
    filler = [_norm(torch.randn(3, 64, generator=g)) for _ in range(2)]
    data = {"ids": [_prompt([1, 2, 3], i) for i in range(4)],
            "sigs": [s, filler[0][0], filler[1][0], s.clone()],
            "norms": [n, filler[0][1], filler[1][1], n.clone()]}
    assert pm.simulate_one(data, live=8, cache=0)["dropped"] == 3
    assert pm.simulate_one(data, live=2, cache=0)["dropped"] == 0


def _tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, generator=g)
    return x, (x * x).sum(1)


def test_window_matches_same_position():
    x, sq = _tokens(3 * pm.BLOCK, 3)
    s, a, o = pm.window_matches(x, sq, x.clone(), sq.clone(), 3, self_partner=False)
    assert s.all() and not a.any() and o.all()


def test_window_matches_shift_by_whole_blocks_is_aligned():
    x, sq = _tokens(4 * pm.BLOCK, 4)
    b, sqb = x[pm.BLOCK:], sq[pm.BLOCK:]          # B's block j is A's block j+1
    s, a, o = pm.window_matches(b, sqb, x, sq, 3, self_partner=False)
    assert not s.any() and a.all() and o.all()


def test_window_matches_shift_by_a_few_tokens_is_any_offset_only():
    x, sq = _tokens(4 * pm.BLOCK, 5)
    b, sqb = x[5:], sq[5:]                         # five tokens fewer in front
    s, a, o = pm.window_matches(b, sqb, x, sq, 3, self_partner=False)
    assert not s.any() and not a.any() and o.all()


def test_window_matches_unrelated_tokens_do_not_match():
    xa, sqa = _tokens(3 * pm.BLOCK, 6)
    xb, sqb = _tokens(3 * pm.BLOCK, 7)
    s, a, o = pm.window_matches(xb, sqb, xa, sqa, 3, self_partner=False)
    assert not o.any()


def test_window_matches_self_only_counts_earlier_windows():
    x, sq = _tokens(pm.BLOCK, 8)
    rep = torch.cat([x, x, x])                     # the same block three times
    r_sq = torch.cat([sq, sq, sq])
    s, a, o = pm.window_matches(rep, r_sq, rep, r_sq, 3, self_partner=True)
    assert o.tolist() == [False, True, True] and not s.any()


def test_prefix_cached_blocks_are_not_planned():
    g = torch.Generator().manual_seed(2)
    s, n = _norm(torch.randn(3, 64, generator=g))
    data = {"ids": [_prompt([1, 2, 3], 0), _prompt([1, 2, 3], 0)], "sigs": [s, s.clone()],
            "norms": [n, n.clone()]}
    t = pm.simulate_one(data, live=8, cache=64)
    assert t["prefix"] == 3 and t["planned"] == 3 and t["dropped"] == 0
