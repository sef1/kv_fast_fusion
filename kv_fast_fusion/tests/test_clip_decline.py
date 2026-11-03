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
