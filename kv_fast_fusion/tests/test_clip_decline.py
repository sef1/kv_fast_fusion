"""BFF_V2_CLIP_FRAC, engine side: per-alias errors, retain, and the capped-request uncount. CPU only.

Drives the real DedupEngine.plan over resident representatives, so the errors come from the planner
exactly as they would on the box.
"""

import pytest
import torch

from kv_fast_fusion import pd_dedup_v2, pd_lsh


def _sigs(raw):
    """Raw block vectors -> (unit signatures, bucket hashes, norms), as the producer encodes them."""
    norms = raw.norm(dim=1)
    sig = raw / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], raw.shape[1], sig.device)
    return sig, pd_lsh.sub_hashes(sig, proj), norms


@pytest.fixture
def planned(monkeypatch):
    """rB's 4 group-1 blocks (60..63): 60-62 copy residents 40-42 of rA with growing noise, 63 is
    unrelated. Returns (engine, planned ids)."""
    monkeypatch.setattr(pd_dedup_v2, "V2_ENABLED", True)
    engine = pd_dedup_v2.DedupEngine(resident=True)
    torch.manual_seed(0)
    raw = torch.randn(3, 8)
    sig, hashes, norms = _sigs(raw)
    engine.note_resident(1, sig, hashes, norms.tolist(), [40, 41, 42], owner="rA")
    torch.manual_seed(1)
    noise = torch.stack([0.0 * torch.randn(8), 0.15 * torch.randn(8), 0.3 * torch.randn(8)])
    torch.manual_seed(7)
    incoming = torch.cat([raw + noise * raw.norm(dim=1, keepdim=True), torch.randn(1, 8)])
    i_sig, i_hash, i_norms = _sigs(incoming)
    payload = pd_dedup_v2.SignatureCodec.encode(i_sig, i_norms, i_hash)
    out = engine.plan({"rB": [[], [60, 61, 62, 63]]}, {"rB": {1: payload}}, threshold=0.5)
    return engine, out["rB"]


def test_every_staged_alias_carries_its_error_lowest_for_the_closest_copy(planned):
    engine, ids = planned
    errs = engine.alias_errors("rB")[1]
    assert set(errs) <= {60, 61, 62} and 60 in errs
    assert errs[60] == pytest.approx(0.0, abs=1e-3)
    assert all(errs[60] <= e for e in errs.values())


def test_retain_keeps_only_the_named_aliases_and_uncounts_the_rest(planned):
    engine, _ids = planned
    staged = set(engine._pending_alias["rB"][1])
    before = engine.stats.dropped_resident.get(1, 0)

    dropped = engine.retain("rB", {1: {60}})

    assert dropped == len(staged) - 1
    assert set(engine._pending_alias["rB"][1]) == {60}
    assert engine.stats.dropped_resident.get(1, 0) == before - dropped


def test_forgetting_a_capped_request_takes_its_declines_back(planned):
    engine, _ids = planned
    assert engine.stats.dropped_resident.get(1, 0) > 0

    engine.forget(["rB"], uncount=True)

    assert engine.stats.dropped_resident.get(1, 0) == 0
    assert "rB" not in engine._pending_alias and "rB" not in engine._pending_meta


def test_a_plain_forget_leaves_the_counters_as_they_were(planned):
    engine, _ids = planned
    before = dict(engine.stats.dropped_resident)

    engine.forget(["rB"])

    assert engine.stats.dropped_resident == before


def test_landing_drops_the_meta_with_the_alias(planned):
    engine, _ids = planned
    engine.release("rB")
    assert "rB" not in engine._pending_meta


def test_a_clip_shows_in_the_ledger_and_an_unclipped_request_has_no_clip_key():
    st = pd_dedup_v2.DedupStats()
    st.note_request("plain", 2, 40, False)
    st.note_request("clipped", 30, 40, True, clipped=10)
    assert "clipped" not in st.request_ledger["plain"]
    assert st.request_ledger["clipped"]["clipped"] == 10


# ---- match potential: every planned block's best candidate, accepted or not --------------------
def test_the_grid_cell_bins_cosine_by_lower_edge_and_error_by_upper_edge():
    assert pd_lsh.potential_cell(0.99, 0.05) == "0.95|0.1"
    assert pd_lsh.potential_cell(0.72, 0.55) == "0.7|0.6"
    assert pd_lsh.potential_cell(0.2, 1.5) == "0|inf"
    assert pd_lsh.potential_cell(0.5, None) == "0.5|?"


def test_probe_reports_one_best_per_block_including_rejected_ones(monkeypatch):
    monkeypatch.setattr(pd_lsh, "LSH_EXACT", True)       # every live row is a candidate
    torch.manual_seed(0)
    sig, hashes, norms = _sigs(torch.randn(4, 8))
    idx = pd_lsh.LshIndex()
    idx.register(sig, hashes, [(i, 40 + i, i, "rA") for i in range(4)], norms.tolist())
    torch.manual_seed(3)
    q_sig, q_hash, q_norms = _sigs(torch.randn(2, 8))
    plain = idx.probe(q_sig, q_hash, ["rB", "rB"], 0.99, q_norms.tolist())
    best: list = []
    with_best = idx.probe(q_sig, q_hash, ["rB", "rB"], 0.99, q_norms.tolist(), best_out=best)

    assert with_best == plain, "recording the best candidate must not change what matches"
    assert plain[0] == [False, False], "random blocks clear no 0.99 bar"
    assert len(best) == 2 and all(b is not None and b[0] < 0.99 for b in best), \
        "the rejected best candidates are still recorded"


def test_a_plan_records_every_block_in_the_potential_grid(planned, monkeypatch):
    # stats_dict reads the per-group threshold through the connector package, which needs a vLLM
    # with the p2p connector; this test is only about the grid.
    monkeypatch.setattr(pd_dedup_v2, "threshold_for", lambda gi: 0.75)
    engine, _ids = planned
    pot = engine.stats.match_potential[1]
    assert sum(pot.values()) + engine.stats.match_potential_none.get(1, 0) == 4
    assert engine.stats.stats_dict()["match_potential"]["1"]["cells"] == dict(sorted(pot.items()))


def test_the_per_layer_projection_is_the_concat_projection():
    """concat(K_l) @ JL == sum_l K_l @ JL[l-th slice]: same signature, no [N, L*w] intermediate."""
    torch.manual_seed(5)
    layers = [(torch.randn(10, 4, 1, 6), torch.randn(10, 4, 1, 6)) for _ in range(3)]
    ids = [1, 4, 7]
    a, na = pd_dedup_v2.signature_matrix(layers, ids, False, [None], compute_dtype=torch.float32)
    b, nb = pd_dedup_v2.signature_matrix_by_layer(layers, ids, False, [None],
                                                  compute_dtype=torch.float32)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(na, nb, rtol=1e-5)


def test_the_per_layer_projection_refuses_a_layout_it_cannot_index():
    """The shape check the concat path has must hold for the default path too."""
    layers = [(torch.randn(10, 4, 1, 6), torch.randn(10, 4, 1, 6))]
    with pytest.raises(pd_dedup_v2.KVLayoutError):
        pd_dedup_v2.signature_matrix_by_layer(layers, [1], False, [None], num_blocks=12)


def test_the_producer_signs_per_layer_by_default():
    assert pd_dedup_v2.SIG_PROJECTION == "by_layer"


def test_alternating_group_widths_build_each_projection_once(monkeypatch):
    """Split layout (signed groups of 20, 20, 19 layers): the single-slot cache rebuilt the JL matrix
    twice per request (~1.9 s host randn each), dropping the prefill from ~2,080 to ~740 tok/s."""
    torch.manual_seed(6)
    wide = [(torch.randn(10, 4, 1, 6), torch.randn(10, 4, 1, 6)) for _ in range(3)]
    narrow = wide[:2]
    holder = [None]
    fresh_wide, _ = pd_dedup_v2.signature_matrix_by_layer(wide, [1, 2], False, [None],
                                                          compute_dtype=torch.float32)
    fresh_narrow, _ = pd_dedup_v2.signature_matrix_by_layer(narrow, [1, 2], False, [None],
                                                            compute_dtype=torch.float32)
    builds = []
    real = torch.randn
    monkeypatch.setattr(pd_dedup_v2.torch, "randn",
                        lambda *a, **k: builds.append(a) or real(*a, **k))
    for _ in range(3):   # three "requests": g1, g2 (same width), g3 (narrower)
        for layers, fresh in ((wide, fresh_wide), (wide, fresh_wide), (narrow, fresh_narrow)):
            s, _ = pd_dedup_v2.signature_matrix_by_layer(layers, [1, 2], False, holder,
                                                         compute_dtype=torch.float32)
            assert torch.equal(s, fresh), "a cached projection must sign exactly as a fresh one"
    assert len(builds) == 2, "one build per width, none per request"
    # The concat path shares the cache: same width, no new build.
    pd_dedup_v2.signature_matrix(wide, [1, 2], False, holder, compute_dtype=torch.float32)
    assert len(builds) == 2


def test_the_projection_cache_is_bounded():
    holder = [None]
    for w in range(1, pd_dedup_v2.JL_CACHE_WIDTHS + 3):
        pd_dedup_v2.jl_projection(holder, w, "cpu", torch.float32)
    assert len(holder[0]) == pd_dedup_v2.JL_CACHE_WIDTHS
