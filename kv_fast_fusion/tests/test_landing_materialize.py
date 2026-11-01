"""Landing-time materialize (DedupEngine.release_materialized) and the frees that keep it exact.

The GLM con48 runs lost ~10% to a handful of stalls: an alias whose representative was freed
between decision and the forward-path apply was refused, the victim had never been fetched, and the
only recovery was a full local re-prefill on the decode (25-50 s of batch-wide stall each). Copying
at landing — while the producer still holds every declined block — turns each refusal into a
re-read instead. What has to hold:

* a valid alias is copied at landing and nothing is left for the forward path to retry;
* a representative that is gone or reissued is REFUSED (never copied) and returned for re-read,
  counted under its cause but not as a recompute;
* the copy runs under the engine lock, which the forward thread takes to invalidate a free;
* scheduler frees invalidate the resident index without scanning it.

CPU only.
"""

import torch

from kv_fast_fusion import pd_dedup_v2, pd_lsh


def _rows(n, dim=4):
    torch.manual_seed(0)
    m = torch.randn(n, dim)
    norms = m.norm(dim=1)
    sig = m / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], dim, sig.device)
    return sig, pd_lsh.sub_hashes(sig, proj), norms.tolist()


def _engine(reps=(41, 42), owner="rA"):
    """rB landed with aliases 51->41 and 52->42, both reps resident and owned by rA."""
    engine = pd_dedup_v2.DedupEngine(resident=True)
    sig, hashes, norms = _rows(len(reps))
    engine.note_resident(1, sig, hashes, norms, list(reps), owner=owner)
    engine._pending_alias = {"rB": {1: {50 + i + 1: (rep, owner, None)
                                        for i, rep in enumerate(reps)}}}
    return engine


class _Copy:
    def __init__(self, ok=True, engine=None):
        self.ok, self.calls, self.engine, self.locked = ok, [], engine, []

    def __call__(self, gi, pairs):
        self.calls.append((gi, sorted(pairs)))
        if self.engine is not None:
            self.locked.append(self.engine.lock.locked())
        return self.ok


def test_every_valid_alias_is_copied_at_landing_and_none_is_left_for_the_forward_path():
    engine = _engine()
    copy = _Copy()

    refused = engine.release_materialized("rB", copy)

    assert refused == {}
    assert copy.calls == [(1, [(41, 51), (42, 52)])]
    assert engine.drain_ready() == {}, "the forward-path applier must find nothing to retry"
    assert (engine.stats.applied, engine.stats.materialized, engine.stats.recomputed) == (2, 2, 0)


def test_a_representative_freed_before_landing_is_refused_not_copied():
    """The stall's cause: the rep's owner finished. The victim must be re-read, not filled from a
    block the pool may already have handed to someone else."""
    engine = _engine()
    engine.on_blocks_freed([41])
    copy = _Copy()

    refused = engine.release_materialized("rB", copy)

    assert refused == {1: {51: "rep_not_resident"}}
    assert copy.calls == [(1, [(42, 52)])], "the surviving alias is still copied"
    assert engine.stats.fail_reasons["rep_not_resident"] == 1
    assert engine.stats.recomputed == 0, "a refusal at landing is a re-read, not a recompute"
    assert engine.stats.resident_invalidated == 1


def test_a_representative_reissued_to_another_request_is_refused():
    """Same id, different KV: residency alone would let it through."""
    engine = _engine()
    sig, hashes, norms = _rows(1)
    engine.note_resident(1, sig, hashes, norms, [41], owner="rC")
    copy = _Copy()

    refused = engine.release_materialized("rB", copy)

    assert refused == {1: {51: "rep_recycled"}}
    assert copy.calls == [(1, [(42, 52)])]


def test_the_victim_the_decode_writes_first_is_refused():
    engine = _engine()
    copy = _Copy()

    refused = engine.release_materialized("rB", copy, hot_victims={1: 52})

    assert refused == {1: {52: "victim_still_writing"}}
    assert copy.calls == [(1, [(41, 51)])]


def test_a_copy_that_cannot_run_refuses_its_whole_batch():
    engine = _engine()

    refused = engine.release_materialized("rB", _Copy(ok=False))

    assert refused == {1: {51: "block_table_write_refused", 52: "block_table_write_refused"}}
    assert engine.stats.materialized == 0
    assert engine.stats.recomputed == 0


def test_the_copy_runs_under_the_engine_lock():
    """The safety argument: the forward thread invalidates frees under this lock, so a copy holding
    it either completes before the invalidation or sees it."""
    engine = _engine()
    copy = _Copy(engine=engine)

    engine.release_materialized("rB", copy)

    assert copy.locked == [True]


def test_a_rep_from_the_same_pull_becomes_resident_before_it_is_validated():
    """A same-pull representative is registered by this very release; validating before
    registering would refuse every one of them."""
    engine = pd_dedup_v2.DedupEngine(resident=True)
    sig, hashes, norms = _rows(1)
    engine._pending_resident = {"rB": {1: (sig, hashes, norms, [41], [None])}}
    engine._pending_alias = {"rB": {1: {51: (41, "rB", None)}}}
    copy = _Copy()

    refused = engine.release_materialized("rB", copy)

    assert refused == {}
    assert copy.calls == [(1, [(41, 51)])]
    assert engine.resident_owner(1, 41) == "rB"


def test_a_request_with_no_aliases_copies_nothing():
    engine = pd_dedup_v2.DedupEngine(resident=True)
    copy = _Copy()

    assert engine.release_materialized("rB", copy) == {}
    assert copy.calls == []


def test_frees_invalidate_only_the_blocks_named():
    engine = _engine(reps=(41, 42, 43))

    engine.on_blocks_freed([42, 999])

    assert engine.resident_owner(1, 41) == "rA"
    assert engine.resident_owner(1, 42) is None
    assert engine.resident_owner(1, 43) == "rA"
    assert engine.stats.resident_invalidated == 1, "an id that was never resident is not counted"


def test_a_forward_path_failure_still_counts_a_recompute():
    """note_failure is now note_refused + note_recompute; the forward path's accounting must not
    move."""
    stats = pd_dedup_v2.DedupStats()

    stats.note_failure("rep_recycled", 2)
    stats.note_refused("rep_not_resident")

    assert stats.recomputed == 2
    assert stats.fail_reasons["rep_recycled"] == 2
    assert stats.fail_reasons["rep_not_resident"] == 1


def test_the_new_counters_reach_the_stats_file():
    stats = pd_dedup_v2.DedupStats()
    stats.repulled, stats.repull_failed, stats.freed_shipped, stats.resident_invalidated = 3, 1, 9, 2

    d = stats.stats_dict()

    assert (d["aliases_repulled"], d["repull_failed"], d["freed_blocks_shipped"],
            d["resident_invalidated"]) == (3, 1, 9, 2)
