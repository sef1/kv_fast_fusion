"""Unit tests for the SimHash cross-request index (`kv_fast_fusion.pd_lsh`) and its use by the GPU
Mooncake connector's fusion engine.

Two properties matter most and are pinned hardest:

* a bucket collision is a CANDIDATE, never a match — a redirect requires the exact cosine to clear
  the threshold (the Ascend port records that merging on raw bucket hits corrupted decode output at
  high concurrency);
* a block registered in one step can be matched from a LATER step with the matrix window disabled —
  that is the whole point of the index.

CPU only. Mirrors the coverage in kv_fast_fusion_ascend/tests/test_mooncake_layerwise_ff.py.
"""

import pytest
import torch

from kv_fast_fusion import pd_lsh
from kv_fast_fusion.connectors import mooncake_connector_ff as mc

THR = 0.75


def _norm(rows):
    m = torch.tensor(rows, dtype=torch.float32)
    return m / m.norm(dim=1, keepdim=True).clamp(min=1e-6)


def _index_with(vecs, owners, slots=None):
    """Build an index holding `vecs` (list of row vectors), one per owner."""
    cur = _norm(vecs)
    proj = pd_lsh.get_proj([None], cur.shape[1], cur.device)
    hashes = pd_lsh.sub_hashes(cur, proj)
    idx = pd_lsh.LshIndex()
    slots = slots if slots is not None else list(range(len(vecs)))
    idx.register(cur, hashes, [(i, hash(owners[i]), slots[i], owners[i])
                               for i in range(len(vecs))])
    return idx, proj


def _probe(idx, proj, vecs, owners, threshold=THR):
    cur = _norm(vecs)
    return idx.probe(cur, pd_lsh.sub_hashes(cur, proj), owners, threshold)


# =====================================================================================
# probe semantics
# =====================================================================================
def test_identical_block_from_another_request_matches():
    v = [1.0, 0.0, 0.5, 0.25]
    idx, proj = _index_with([v], ["reqA"], slots=[3])
    matched, hits = _probe(idx, proj, [v], ["reqB"])
    assert matched == [True]
    assert hits == [(0, hash("reqA"), 3)], "carries the rep's key and slot back"


def test_dissimilar_block_is_not_redirected():
    """The verify decision: even if the bucket were shared, cosine below threshold means no merge."""
    idx, proj = _index_with([[1.0, 0.0, 0.0, 0.0]], ["reqA"])
    matched, hits = _probe(idx, proj, [[0.0, 1.0, 0.0, 0.0]], ["reqB"])
    assert matched == [False] and hits == []


def test_never_redirects_to_a_rep_from_the_same_request():
    """Self-merge would point a request at its own block — no compression, and it breaks the
    'rep is resident elsewhere' assumption the consumer relies on."""
    v = [1.0, 0.0, 0.5, 0.25]
    idx, proj = _index_with([v], ["reqA"])
    matched, _hits = _probe(idx, proj, [v], ["reqA"])
    assert matched == [False]


def test_probe_agrees_with_a_bruteforce_cosine_oracle():
    """Any hit the index reports must be a genuine above-threshold neighbour (no false accepts)."""
    torch.manual_seed(7)
    reps = torch.randn(40, 32).tolist()
    idx, proj = _index_with(reps, [f"r{i}" for i in range(40)])
    queries = torch.randn(20, 32).tolist()

    _matched, hits = _probe(idx, proj, queries, ["q"] * 20)

    rn, qn = _norm(reps), _norm(queries)
    for (i, _rep_key, rep_slot) in hits:
        assert float(qn[i] @ rn[rep_slot]) > THR, (
            f"query {i} accepted a rep at cosine {float(qn[i] @ rn[rep_slot]):.3f}")


def test_probe_on_an_empty_index_matches_nothing():
    idx = pd_lsh.LshIndex()
    proj = pd_lsh.get_proj([None], 4, torch.device("cpu"))
    cur = _norm([[1.0, 0.0, 0.0, 0.0]])
    assert idx.probe(cur, pd_lsh.sub_hashes(cur, proj), ["q"], THR) == ([False], [])


def test_accepted_cosines_are_binned():
    v = [1.0, 0.0, 0.5, 0.25]
    idx, proj = _index_with([v], ["reqA"])
    _probe(idx, proj, [v], ["reqB"])
    assert sum(idx.accept_cos) == 1
    assert idx.accept_cos[-1] == 1, "an exact duplicate lands in the top bin"


# =====================================================================================
# capacity management
# =====================================================================================
def test_register_grows_and_reports_size():
    torch.manual_seed(3)
    reps = torch.randn(300, 16).tolist()
    idx, _proj = _index_with(reps, [f"r{i}" for i in range(300)])
    assert idx.size() == 300
    assert idx.mat.shape[0] >= 300, "capacity grew by doubling"


def test_evict_and_compact_keeps_the_index_probeable(monkeypatch):
    """After a batch evict the surviving rows are renumbered; buckets must point at the new ids."""
    torch.manual_seed(11)
    reps = torch.randn(20, 16)
    reps[19] = reps[0]                        # last is a duplicate of the first
    idx, proj = _index_with(reps.tolist(), [f"r{i}" for i in range(20)])

    for row in range(10):                     # evict the oldest half, then compact
        idx.evict(row)
    idx.compact()

    assert idx.size() == 10
    assert set(idx.meta) == set(range(10)), "rows renumbered densely"
    for rows in (b for t in idx.tables for b in t.values()):
        assert all(r < idx.size() for r in rows), "no bucket points past the compacted matrix"
    matched, _hits = _probe(idx, proj, [reps[19].tolist()], ["q"])
    assert matched == [True], "a surviving duplicate is still findable"


def test_register_evicts_when_over_capacity(monkeypatch):
    monkeypatch.setattr(pd_lsh, "LSH_MAX_ENTRIES", 8)
    torch.manual_seed(5)
    reps = torch.randn(8, 16).tolist()
    idx, _proj = _index_with(reps, [f"r{i}" for i in range(8)])
    assert idx.size() == 8

    cur = _norm(torch.randn(2, 16).tolist())
    proj = pd_lsh.get_proj([None], 16, cur.device)
    idx.register(cur, pd_lsh.sub_hashes(cur, proj),
                 [(i, hash(f"n{i}"), i, f"n{i}") for i in range(2)])

    assert idx.size() <= 8 - 8 // 2 + 2, "oldest half dropped before inserting"
    assert set(idx.meta) == set(range(idx.size()))


def test_compact_on_a_fully_evicted_index_resets_it():
    idx, _proj = _index_with([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    idx.evict(0)
    idx.evict(1)
    idx.compact()
    assert idx.size() == 0 and idx.mat is None
    assert all(not t for t in idx.tables)


# =====================================================================================
# integration with the fusion engine
# =====================================================================================
def _kv(vecs):
    n = len(vecs)
    k = torch.zeros(n + 1, 1, 1, len(vecs[0]))     # +1: block id 0 is the null block
    for i, v in enumerate(vecs):
        k[i + 1, 0, 0] = torch.tensor(v, dtype=torch.float32)
    return torch.stack([k, k.clone()])


def _lsh_fusion(monkeypatch):
    monkeypatch.setattr(mc, "_PD_CROSS_INDEX", "lsh")
    monkeypatch.setattr(mc, "_PD_ENCODED_BATCH", 0)     # matrix window off: LSH must stand alone
    return mc.FFProducerFusion({1: {"l0"}})


def test_lsh_matches_across_steps_without_the_matrix_window(monkeypatch):
    """The capability the matrix backend cannot provide: a rep registered in step 1 is matched in
    step 2 with BFF_PD_ENCODED_BATCH_SIZE=0."""
    f = _lsh_fusion(monkeypatch)
    v = [1.0, 0.0, 0.0, 0.5]

    assert f.on_layer(1, "l0", _kv([v]), [("t-A", [[], [1]])], 1, False) == {}
    rows = f.on_layer(1, "l0", _kv([v]), [("t-B", [[], [1]])], 2, False)

    assert rows == {"t-B": [[0, mc._tid_hash("t-A"), 0]]}
    assert f.cross_redir_total == 1 and f.within_redir_total == 0
    assert f._lsh[1].size() == 1


def test_lsh_leaves_a_dissimilar_later_block_alone(monkeypatch):
    f = _lsh_fusion(monkeypatch)
    f.on_layer(1, "l0", _kv([[1.0, 0.0, 0.0, 0.0]]), [("t-A", [[], [1]])], 1, False)
    rows = f.on_layer(1, "l0", _kv([[0.0, 1.0, 0.0, 0.0]]), [("t-B", [[], [1]])], 2, False)
    assert rows == {} and f.cross_redir_total == 0


def test_lsh_stats_are_reported(monkeypatch):
    f = _lsh_fusion(monkeypatch)
    v = [1.0, 0.0, 0.0, 0.5]
    f.on_layer(1, "l0", _kv([v]), [("t-A", [[], [1]])], 1, False)
    f.on_layer(1, "l0", _kv([v]), [("t-B", [[], [1]])], 2, False)

    s = f.stats_dict()
    assert s["cross_index"] == "lsh"
    assert s["lsh_entries"] == {"1": 1}
    assert sum(s["lsh_accept_cos"]["1"].values()) == 1


def test_matrix_backend_is_unaffected(monkeypatch):
    """Default path must behave exactly as before: within-batch merge, no LSH index built."""
    monkeypatch.setattr(mc, "_PD_CROSS_INDEX", "matrix")
    monkeypatch.setattr(mc, "_PD_ENCODED_BATCH", 0)
    f = mc.FFProducerFusion({1: {"l0"}})
    dup = [1.0, 0.0]
    rows = f.on_layer(1, "l0", _kv([dup, dup]),
                      [("t-A", [[], [1]]), ("t-B", [[], [2]])], 1, False)
    assert rows == {"t-B": [[0, mc._tid_hash("t-A"), 0]]}
    assert f.within_redir_total == 1 and f.cross_redir_total == 0
    assert f._lsh == {}


def test_lsh_falls_back_to_matrix_under_tp(monkeypatch):
    """The index is a single host-side structure with no cross-rank coherence, so TP>1 must not
    use it (ranks would otherwise reach different merge decisions and desync the block table)."""
    monkeypatch.setattr(mc, "_PD_CROSS_INDEX", "lsh")
    monkeypatch.setattr(mc, "_PD_ENCODED_BATCH", 0)
    f = mc.FFProducerFusion({1: {"l0"}})
    f.tp_group = object()                       # pretend TP>1
    v = [1.0, 0.0, 0.0, 0.5]
    f.on_layer(1, "l0", _kv([v]), [("t-A", [[], [1]])], 1, False)
    assert f._lsh == {}, "no LSH index built under TP>1"


# =====================================================================================
# owner-scoped eviction
#
# A representative is only usable while its KV is still resident on the decode instance. The index
# has no way to know that on its own, so the connector tells it — and these pin the contract that
# makes that possible: eviction at REQUEST granularity, because a request's blocks die together.
# =====================================================================================
def test_evict_owner_removes_all_of_one_requests_reps():
    v1, v2, v3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    idx, proj = _index_with([v1, v2, v3], ["reqA", "reqA", "reqB"], slots=[0, 1, 0])
    assert idx.size() == 3 and idx.n_owners() == 2

    assert idx.evict_owner("reqA") == 2, "both of reqA's reps go"

    assert idx.size() == 1 and idx.n_owners() == 1
    assert _probe(idx, proj, [v1], ["q"]) == ([False], []), "a dead rep is never returned"
    matched, hits = _probe(idx, proj, [v3], ["q"])
    assert matched == [True] and hits == [(0, hash("reqB"), 0)], "the survivor still matches"


def test_evict_owner_leaves_buckets_pointing_only_at_live_rows():
    torch.manual_seed(19)
    reps = torch.randn(12, 16).tolist()
    idx, _proj = _index_with(reps, [f"r{i // 3}" for i in range(12)])   # 4 owners x 3 reps
    idx.evict_owners(["r0", "r2"])
    assert idx.size() == 6 and set(idx.meta) == set(range(6))
    for rows in (b for t in idx.tables for b in t.values()):
        assert all(r < idx.size() for r in rows)
    assert set(idx.by_owner) == {"r1", "r3"}


def test_evicting_an_unknown_owner_is_a_noop():
    """Feedback is broadcast to every producer, so most ids a producer sees are not its own."""
    idx, _proj = _index_with([[1.0, 0.0]], ["reqA"])
    assert idx.evict_owner("never-heard-of-it") == 0
    assert idx.size() == 1


def test_owner_bound_evicts_the_oldest_whole_request(monkeypatch):
    """The backstop for lost feedback. Evicting whole requests (not rows) keeps the index a set of
    plausibly-live requests rather than a mix of half-dead ones."""
    monkeypatch.setattr(pd_lsh, "LSH_MAX_OWNERS", 2)
    torch.manual_seed(23)
    reps = torch.randn(6, 16).tolist()
    idx, _proj = _index_with(reps, [f"r{i // 2}" for i in range(6)])    # r0, r1, r2 x 2 reps

    assert idx.n_owners() == 2, "oldest request dropped once the bound was exceeded"
    assert set(idx.by_owner) == {"r1", "r2"}
    assert idx.size() == 4
    assert all(m[2] != "r0" for m in idx.meta.values())


def test_direct_row_eviction_keeps_the_owner_view_consistent(monkeypatch):
    """The LSH_MAX_ENTRIES path evicts rows, not owners; by_owner must not keep dangling rows or a
    later evict_owners would resurrect them."""
    monkeypatch.setattr(pd_lsh, "LSH_MAX_ENTRIES", 4)
    torch.manual_seed(29)
    reps = torch.randn(4, 16).tolist()
    idx, _proj = _index_with(reps, ["a", "a", "b", "b"])

    cur = _norm(torch.randn(1, 16).tolist())
    proj = pd_lsh.get_proj([None], 16, cur.device)
    idx.register(cur, pd_lsh.sub_hashes(cur, proj), [(0, 99, 0, "c")])

    live = {o for m in idx.meta.values() for o in [m[2]]}
    assert set(idx.by_owner) == live, "owner view matches what actually survived"
    assert all(r < idx.size() for rows in idx.by_owner.values() for r in rows)


# =====================================================================================
# substitution error
#
# A redirect does not deduplicate — it replaces the owner's KV with the rep's. Cosine is scale-free
# and cannot see a magnitude mismatch, so it is not on its own a measure of what that costs. These
# pin the quantity that is: rel_err = ||k_owner - k_rep|| / ||k_owner||.
# =====================================================================================
def test_rel_err_is_zero_only_for_an_exact_match():
    assert pd_lsh.rel_err(1.0, 1.0) == 0.0
    assert pd_lsh.rel_err(1.0, 1.5) == pytest.approx(0.5), "same direction, 1.5x the magnitude"
    assert pd_lsh.rel_err(0.9, 1.0) == pytest.approx(0.4472, abs=1e-4)


def test_equal_cosine_different_norms_bin_differently():
    """The whole reason norms are tracked: at cosine 0.93 a matched-magnitude substitution is a
    ~0.37 relative error while a 1.3x one is ~0.45 — and cosine reports them as identical."""
    cur = _norm([[1.0, 0.0, 0.5, 0.25]])
    proj = pd_lsh.get_proj([None], 4, cur.device)
    h = pd_lsh.sub_hashes(cur, proj)
    same, bigger = pd_lsh.LshIndex(), pd_lsh.LshIndex()
    same.register(cur, h, [(0, 1, 0, "reqA")], norms=[10.0])
    bigger.register(cur, h, [(0, 1, 0, "reqA")], norms=[13.0])

    for idx in (same, bigger):
        idx.probe(cur, h, ["reqB"], THR, norms=[10.0])

    assert same.accept_rel_err[0] == 1, "exact substitution lands in the smallest-error bin"
    assert bigger.accept_rel_err[0] == 0
    assert sum(bigger.accept_rel_err) == 1 and bigger.accept_rel_err.index(1) > 0


def test_rel_err_is_not_recorded_without_norms():
    """Norms are optional; callers that do not pass them still get cosines, just no error histogram
    (rather than a silently wrong one computed against an assumed ratio of 1)."""
    v = [1.0, 0.0, 0.5, 0.25]
    idx, proj = _index_with([v], ["reqA"])
    _probe(idx, proj, [v], ["reqB"])
    assert sum(idx.accept_cos) == 1 and sum(idx.accept_rel_err) == 0


def test_norms_survive_eviction_and_compaction():
    torch.manual_seed(31)
    reps = torch.randn(6, 8).tolist()
    cur = _norm(reps)
    proj = pd_lsh.get_proj([None], 8, cur.device)
    h = pd_lsh.sub_hashes(cur, proj)
    idx = pd_lsh.LshIndex()
    idx.register(cur, h, [(i, i, i, f"r{i // 2}") for i in range(6)],
                 norms=[float(i + 1) for i in range(6)])

    idx.evict_owners(["r0"])                        # drops rows 0,1 -> norms 1.0, 2.0

    assert idx.size() == 4
    assert sorted(idx.row_norm.values()) == [3.0, 4.0, 5.0, 6.0]
    assert set(idx.row_norm) == set(range(4)), "renumbered with the rows"


# =====================================================================================
# substitution-error budget (BFF_MAX_REL_ERR)
# =====================================================================================
# Cosine is scale-free: two blocks pointing the same way but sized differently score 1.0, while
# swapping one for the other injects error proportional to the size gap. These pin the gate that
# closes that hole, and the arithmetic bound on how much it can ever buy.
def _index_with_norms(vecs, owners, norms):
    cur = _norm(vecs)
    proj = pd_lsh.get_proj([None], cur.shape[1], cur.device)
    hashes = pd_lsh.sub_hashes(cur, proj)
    idx = pd_lsh.LshIndex()
    idx.register(cur, hashes, [(i, hash(owners[i]), i, owners[i]) for i in range(len(vecs))],
                 norms)
    return idx, proj


def _probe_norms(idx, proj, vecs, owners, norms, threshold=THR):
    cur = _norm(vecs)
    return idx.probe(cur, pd_lsh.sub_hashes(cur, proj), owners, threshold, norms)


def test_min_rel_err_is_the_floor_no_norm_can_beat():
    """rel_err = sqrt(1 + r^2 - 2*r*cos) is minimised at r = cos, giving sqrt(1-cos^2). This is why
    a substitution-error budget implies a cosine bar: 0.20 forces cos >= 0.98 whatever the norms."""
    for cos in (0.5, 0.75, 0.9, 0.95, 0.98):
        floor = pd_lsh.min_rel_err(cos)
        best = min(pd_lsh.rel_err(cos, r / 1000.0) for r in range(1, 3000))
        assert floor == pytest.approx(best, abs=1e-3)
        assert floor == pytest.approx((1 - cos * cos) ** 0.5)


def test_a_perfectly_aligned_block_of_the_wrong_size_is_rejected(monkeypatch):
    """The case a cosine bar cannot see: cosine 1.0, but the rep is 1.6x the owner's magnitude, so
    substituting it injects 60% relative error."""
    v = [1.0, 0.0, 0.5, 0.25]
    idx, proj = _index_with_norms([v], ["rep"], [16.0])

    matched, _ = _probe_norms(idx, proj, [v], ["owner"], [10.0])
    assert matched == [True], "cosine alone accepts it"

    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.20)
    idx2, proj2 = _index_with_norms([v], ["rep"], [16.0])
    matched2, hits2 = _probe_norms(idx2, proj2, [v], ["owner"], [10.0])
    assert matched2 == [False] and hits2 == []
    assert idx2.rejected_by_rel_err == 1, "charged to the error budget, not the cosine bar"


def test_the_budget_picks_the_lower_error_candidate_not_the_closer_one(monkeypatch):
    """Ranking by cosine takes a near-parallel block of the wrong size over a slightly less aligned
    block of the right size — the worse substitution of the two."""
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.30)
    aligned = [1.0, 0.0, 0.0, 0.0]                       # cosine 1.00, norm 1.5x  -> rel_err 0.50
    slightly_off = [0.99, (1 - 0.99 ** 2) ** 0.5, 0.0, 0.0]   # cosine 0.99, norm 1x -> rel_err 0.14
    idx, proj = _index_with_norms([aligned, slightly_off], ["a", "b"], [15.0, 10.0])

    matched, hits = _probe_norms(idx, proj, [aligned], ["owner"], [10.0])

    assert matched == [True]
    assert hits[0][1] == hash("b"), "took the equal-magnitude block, not the better-aligned one"


@pytest.mark.parametrize("budget,expect", [(1.0, True), (0.30, False)])
def test_a_cos_090_pair_cannot_pass_a_030_budget(monkeypatch, budget, expect):
    """The floor in practice: at cosine 0.90 the best achievable error is 0.436, so no norm ratio
    rescues it. A 0.30 budget must reject it however the magnitudes are chosen."""
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", budget)
    owner = [1.0, 0.0, 0.0, 0.0]
    rep = [0.90, (1 - 0.90 ** 2) ** 0.5, 0.0, 0.0]
    # r = cos is the error-minimising ratio, i.e. the best case for the rep.
    idx, proj = _index_with_norms([rep], ["rep"], [9.0])
    matched, _ = _probe_norms(idx, proj, [owner], ["owner"], [10.0])
    assert matched == [expect]


def test_the_budget_is_inert_by_default():
    """Every earlier run used a pure cosine bar; the default must reproduce it exactly."""
    assert pd_lsh.MAX_REL_ERR == 1.0
    v = [1.0, 0.0, 0.5, 0.25]
    idx, proj = _index_with_norms([v], ["rep"], [16.0])
    matched, _ = _probe_norms(idx, proj, [v], ["owner"], [10.0])
    assert matched == [True] and idx.rejected_by_rel_err == 0
