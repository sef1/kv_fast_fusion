"""Unit tests for the decode-side dedup planner (`kv_fast_fusion.pd_dedup_plan`).

The claim this module exists to make good on is narrow and worth stating: **the decode only ever
aliases to a block it holds**. In v1 the producer guessed at decode residency and was wrong 71% of
the time (7,714 redirects applied against 18,726 unresolved), and every block was transferred
anyway. Here an unresolvable alias is not a rare outcome to be counted — it is unrepresentable, and
a block that is aliased is never requested, so its bytes never cross the wire.

CPU only.
"""

import pytest
import torch

from kv_fast_fusion import pd_lsh
from kv_fast_fusion.pd_dedup_plan import DedupPlanner, IncomingBlock

THR = 0.75


def _sig(rows):
    """Normalised signature matrix + bucket ids + norms, as the producer would ship them."""
    m = torch.tensor(rows, dtype=torch.float32)
    norms = m.norm(dim=1).clamp(min=1e-6)
    cur = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], cur.shape[1], cur.device)
    return cur, pd_lsh.sub_hashes(cur, proj), norms.tolist()


def _incoming(specs):
    """specs: [(req_id, slot, block_id)] in the order the decode would request them."""
    return [IncomingBlock(req_id=r, group=1, slot=s, block_id=b, row=i)
            for i, (r, s, b) in enumerate(specs)]


# =====================================================================================
# satisfied by a block already resident
# =====================================================================================
def test_a_block_matching_a_resident_block_is_not_requested():
    v = [1.0, 0.0, 0.5, 0.25]
    p = DedupPlanner()
    sigs, hashes, norms = _sig([v])
    p.register(1, sigs, hashes, norms, block_ids=[900])       # already on D

    plan = p.plan(1, _incoming([("rA", 0, 41)]), sigs, hashes, norms, THR)

    assert plan.keep[("rA", 1)] == [], "nothing left to pull"
    assert plan.alias[("rA", 1)] == {0: 900}, "served by the resident block"
    assert plan.n_resident == 1 and plan.n_batch == 0


def test_a_dissimilar_block_is_still_requested():
    p = DedupPlanner()
    res_sig, res_h, res_n = _sig([[1.0, 0.0, 0.0, 0.0]])
    p.register(1, res_sig, res_h, res_n, block_ids=[900])

    sigs, hashes, norms = _sig([[0.0, 1.0, 0.0, 0.0]])
    plan = p.plan(1, _incoming([("rA", 0, 41)]), sigs, hashes, norms, THR)

    assert plan.keep[("rA", 1)] == [0] and plan.alias == {}
    assert plan.n_dropped() == 0


@pytest.mark.parametrize("threshold,expect_pulled", [(0.75, False), (0.99, True)])
def test_the_threshold_decides_a_near_duplicate(threshold, expect_pulled):
    """The bar is an error budget, not a preference: cos 0.95 is a 0.32 relative substitution
    error. The SAME pair must be dropped under a loose bar and pulled under a strict one — which is
    the knob that governs quality, unchanged from the producer-side design."""
    p = DedupPlanner()
    res_sig, res_h, res_n = _sig([[1.0, 0.0, 0.0, 0.0]])
    p.register(1, res_sig, res_h, res_n, block_ids=[900])

    near = [0.95, (1 - 0.95 ** 2) ** 0.5, 0.0, 0.0]        # cosine 0.95 with the resident block
    sigs, hashes, norms = _sig([near])
    plan = p.plan(1, _incoming([("rA", 0, 41)]), sigs, hashes, norms, threshold)

    assert (plan.keep[("rA", 1)] == [0]) is expect_pulled


# =====================================================================================
# satisfied by another block in the same pull
# =====================================================================================
def test_a_duplicate_within_the_same_pull_is_requested_once():
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, norms = _sig([v, v])
    p = DedupPlanner()

    plan = p.plan(1, _incoming([("rA", 0, 41), ("rB", 0, 42)]), sigs, hashes, norms, THR)

    assert plan.keep[("rA", 1)] == [0], "the first copy is pulled"
    assert plan.keep[("rB", 1)] == [], "the second is not"
    assert plan.alias[("rB", 1)] == {0: 41}, "and points at the block that WILL be written"
    assert plan.n_batch == 1 and plan.n_resident == 0


def test_a_requests_own_duplicate_blocks_are_shared():
    """Unlike the producer index, same-request sharing is a real saving here and must not be
    excluded — the decode is deduplicating its own memory, not merging across the wire."""
    v = [0.0, 1.0, 0.0, 0.5]
    sigs, hashes, norms = _sig([v, v])
    plan = DedupPlanner().plan(1, _incoming([("rA", 0, 41), ("rA", 1, 42)]),
                               sigs, hashes, norms, THR)
    assert plan.keep[("rA", 1)] == [0]
    assert plan.alias[("rA", 1)] == {1: 41}


def test_an_aliased_block_never_becomes_a_representative():
    """No chains: a rep must be a block that is actually written, or the third copy would point at
    memory nobody ever fills."""
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, norms = _sig([v, v, v])
    plan = DedupPlanner().plan(
        1, _incoming([("rA", 0, 41), ("rB", 0, 42), ("rC", 0, 43)]), sigs, hashes, norms, THR)

    assert plan.keep[("rA", 1)] == [0]
    reps = {d[0] for d in (plan.alias[("rB", 1)], plan.alias[("rC", 1)])}
    assert reps == {41}, "both later copies point at the one block being written"


def test_every_alias_target_is_a_block_that_gets_written_or_is_resident():
    """The invariant the whole design rests on, checked over a random mix."""
    torch.manual_seed(11)
    resident_ids = [900, 901, 902]
    res = torch.randn(3, 24)
    p = DedupPlanner()
    rs, rh, rn = _sig(res.tolist())
    p.register(1, rs, rh, rn, block_ids=resident_ids)

    rows = torch.randn(12, 24)
    rows[3] = res[1]                      # a resident duplicate
    rows[7] = rows[2]                     # a within-pull duplicate
    sigs, hashes, norms = _sig(rows.tolist())
    incoming = _incoming([(f"r{i}", 0, 100 + i) for i in range(12)])

    plan = p.plan(1, incoming, sigs, hashes, norms, THR)

    written = {b.block_id for b in incoming if b.slot in plan.keep[(b.req_id, 1)]}
    for slots in plan.alias.values():
        for rep in slots.values():
            assert rep in written or rep in resident_ids, (
                f"alias target {rep} is neither written in this pull nor resident")


# =====================================================================================
# residency bookkeeping
# =====================================================================================
def test_forgetting_a_freed_block_stops_it_being_offered():
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, norms = _sig([v])
    p = DedupPlanner()
    p.register(1, sigs, hashes, norms, block_ids=[900])
    assert p.size(1) == 1

    assert p.forget(1, [900]) == 1
    assert p.size(1) == 0

    plan = p.plan(1, _incoming([("rA", 0, 41)]), sigs, hashes, norms, THR)
    assert plan.keep[("rA", 1)] == [0], "a freed block can never serve a later request"


def test_forgetting_an_unknown_block_is_harmless():
    p = DedupPlanner()
    assert p.forget(1, [12345]) == 0
    assert p.size(1) == 0


# =====================================================================================
# reporting
# =====================================================================================
def test_quality_of_each_merge_is_recorded():
    """Same instrumentation as the producer path: the accepted cosine and the substitution error
    it implies, so a v2 run is directly comparable with a v1 one."""
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, norms = _sig([v, v])
    plan = DedupPlanner().plan(1, _incoming([("rA", 0, 41), ("rB", 0, 42)]),
                               sigs, hashes, norms, THR)
    assert sum(plan.accept_cos) == 1
    assert plan.accept_cos[-1] == 1, "an exact duplicate lands in the top cosine bin"
    assert plan.accept_rel_err[0] == 1, "and in the smallest substitution-error bin"


def test_an_empty_pull_plans_nothing():
    sigs, hashes, norms = _sig([[1.0, 0.0]])
    plan = DedupPlanner().plan(1, [], sigs, hashes, norms, THR)
    assert plan.keep == {} and plan.alias == {} and plan.n_dropped() == 0


def test_scale_mismatch_is_charged_as_substitution_error():
    """Cosine cannot see a magnitude mismatch; rel_err must."""
    p = DedupPlanner()
    rs, rh, _rn = _sig([[1.0, 0.0, 0.5, 0.25]])
    p.register(1, rs, rh, [13.0], block_ids=[900])

    sigs, hashes, _n = _sig([[1.0, 0.0, 0.5, 0.25]])
    plan = p.plan(1, _incoming([("rA", 0, 41)]), sigs, hashes, [10.0], THR)

    assert plan.n_resident == 1
    assert plan.accept_rel_err[0] == 0, "a 1.3x magnitude gap is not a free substitution"
    assert sum(plan.accept_rel_err) == 1


def test_planner_reports_index_size_per_group():
    p = DedupPlanner()
    sigs, hashes, norms = _sig([[1.0, 0.0], [0.0, 1.0]])
    p.register(2, sigs, hashes, norms, block_ids=[900, 901])
    assert p.size(2) == 2 and p.size(1) == 0
    assert p.plan(1, _incoming([("rA", 0, 41)]), sigs, hashes, norms, THR).keep[("rA", 1)] == [0], (
        "an empty group index requests everything"
    )


def test_registering_nothing_is_a_noop():
    p = DedupPlanner()
    p.register(1, None, None, None, block_ids=[])
    assert p.size(1) == 0


@pytest.mark.parametrize("threshold", [0.5, 0.9, 0.99])
def test_threshold_monotonically_reduces_what_is_dropped(threshold):
    torch.manual_seed(3)
    rows = torch.randn(20, 16)
    sigs, hashes, norms = _sig(rows.tolist())
    incoming = _incoming([(f"r{i}", 0, 100 + i) for i in range(20)])
    plan = DedupPlanner().plan(1, incoming, sigs, hashes, norms, threshold)
    kept = sum(len(v) for v in plan.keep.values())
    assert kept + plan.n_dropped() == 20, "every block is either pulled or aliased, never both"


# =====================================================================================
# substitution-error budget on the same-pull path
# =====================================================================================
# The two halves of one decision must obey the same rule. Resident candidates are gated inside
# pd_lsh.probe; same-pull candidates are compared here, and were previously ranked on raw cosine.
def test_same_pull_merges_obey_the_error_budget_too(monkeypatch):
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.20)
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, _n = _sig([v, v])

    plan = DedupPlanner().plan(1, _incoming([("rA", 0, 41), ("rB", 0, 42)]),
                               sigs, hashes, [10.0, 16.0], THR)

    assert plan.keep[("rB", 1)] == [0], "a 1.6x magnitude gap is not a free substitution"
    assert plan.n_dropped() == 0
    assert plan.rejected_by_rel_err == 1


def test_same_pull_still_merges_a_true_duplicate_under_the_budget(monkeypatch):
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.20)
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, norms = _sig([v, v])

    plan = DedupPlanner().plan(1, _incoming([("rA", 0, 41), ("rB", 0, 42)]),
                               sigs, hashes, norms, THR)

    assert plan.alias[("rB", 1)] == {0: 41}
    assert plan.rejected_by_rel_err == 0


def test_rejections_are_binned_by_cosine_so_the_reachable_ones_can_be_counted(monkeypatch):
    """Counting rejections says how many the norm stopped; only their COSINES say whether any could
    ever be recovered. ``min_rel_err(cos) = sqrt(1-cos^2)`` is a floor no norm ratio beats, so a
    rejection below ``min_cos_for_budget`` is unreachable however well the rep is rescaled — the
    question 'would stretching the norm back help?' is exactly this histogram."""
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.20)
    v = [1.0, 0.0, 0.5, 0.25]
    sigs, hashes, _n = _sig([v, v])          # identical directions: cosine 1.0

    plan = DedupPlanner().plan(1, _incoming([("rA", 0, 41), ("rB", 0, 42)]),
                               sigs, hashes, [10.0, 16.0], THR)

    assert plan.rejected_by_rel_err == 1
    assert sum(plan.reject_cos) == plan.rejected_by_rel_err, "every rejection lands in a bin"
    # cos 1.0 is above the 0.980 floor for a 0.20 budget: lost to the norm ratio ALONE, so a
    # better-matched representative could have taken it. That is the recoverable case.
    top = pd_lsh.ACCEPT_COS_LABELS[plan.reject_cos.index(1)]
    assert float(top.split("-")[1]) > pd_lsh.min_cos_for_budget(0.20)


def test_a_rejection_below_the_budget_floor_is_reported_as_unreachable(monkeypatch):
    """The other half: a pair whose cosine alone already violates the budget. No norm ratio can
    rescue it, so it must not be counted as something better matching would recover."""
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.20)      # floor: cos >= 0.980
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.90, 0.4359, 0.0, 0.0]                          # cos ~= 0.90, below the floor
    sigs, hashes, norms = _sig([a, b])

    plan = DedupPlanner().plan(1, _incoming([("rA", 0, 41), ("rB", 0, 42)]),
                               sigs, hashes, norms, THR)

    assert plan.n_dropped() == 0
    assert plan.rejected_by_rel_err == sum(plan.reject_cos)
    for i, n in enumerate(plan.reject_cos):
        if n:
            assert float(pd_lsh.ACCEPT_COS_LABELS[i].split("-")[1]) <= pd_lsh.min_cos_for_budget(0.20)
