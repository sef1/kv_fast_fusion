"""A request's last block is always read, never planned (pd_dedup_v2.SHARE_LAST_BLOCK off). CPU only.

Box, 2026-11-15: short prompts (one partly filled block) whose block was declined had another prompt's KV
copied in at landing and answered it from the first token, 8 of 8. The producer signs all 128 slots and
the unused ones hold the block's previous tenant, so a partial block can match a block that is not this
prompt at all.
"""

import pytest
import torch

from kv_fast_fusion import pd_dedup_v2, pd_lsh


def _sigs(raw):
    norms = raw.norm(dim=1)
    sig = raw / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], raw.shape[1], sig.device)
    return sig, pd_lsh.sub_hashes(sig, proj), norms


@pytest.fixture
def engine(monkeypatch):
    """Resident blocks 40..42 of rA in group 1."""
    monkeypatch.setattr(pd_dedup_v2, "V2_ENABLED", True)
    eng = pd_dedup_v2.DedupEngine(resident=True)
    torch.manual_seed(0)
    raw = torch.randn(3, 8)
    sig, hashes, norms = _sigs(raw)
    eng.note_resident(1, sig, hashes, norms.tolist(), [40, 41, 42], owner="rA")
    eng._raw = raw
    return eng


def _payload(raw):
    sig, hashes, norms = _sigs(raw)
    return pd_dedup_v2.SignatureCodec.encode(sig, norms, hashes)


def test_a_one_block_request_identical_to_a_resident_block_is_still_read(engine):
    out = engine.plan({"rS": [[], [70]]}, {"rS": {1: _payload(engine._raw[:1])}}, threshold=0.5)
    assert out["rS"][1] == [70], "the only block of a short prompt is never declined"
    assert "rS" not in engine._pending_alias
    assert "rS" not in engine._pending_resident, "nor offered to later requests as a representative"
    assert engine.stats.last_block_kept == 1


def test_a_longer_request_shares_its_full_blocks_and_reads_its_last(engine):
    out = engine.plan({"rB": [[], [60, 61, 62]]}, {"rB": {1: _payload(engine._raw)}}, threshold=0.5)
    ids = out["rB"][1]
    assert ids[0] == pd_dedup_v2.SENTINEL and ids[1] == pd_dedup_v2.SENTINEL
    assert ids[2] == 62, "identical to resident 42, but it is the last block"
    assert set(engine._pending_alias["rB"][1]) == {60, 61}


def test_rows_stay_paired_with_slots_across_requests_in_one_batch(engine):
    """rC's skipped last row must not shift rD's rows: rD's first block equals resident 41."""
    torch.manual_seed(5)
    other = torch.randn(2, 8)
    req = {"rC": [[], [80, 81]], "rD": [[], [90, 91]]}
    sigs = {"rC": {1: _payload(other)}, "rD": {1: _payload(torch.stack([engine._raw[1], other[0]]))}}
    out = engine.plan(req, sigs, threshold=0.5)
    assert out["rD"][1] == [pd_dedup_v2.SENTINEL, 91]
    assert engine._pending_alias["rD"][1][90][0] == 41
    assert out["rC"][1] == [80, 81]


def test_the_old_behaviour_is_one_switch_away(engine, monkeypatch):
    monkeypatch.setattr(pd_dedup_v2, "SHARE_LAST_BLOCK", True)
    out = engine.plan({"rS": [[], [70]]}, {"rS": {1: _payload(engine._raw[:1])}}, threshold=0.5)
    assert out["rS"][1] == [pd_dedup_v2.SENTINEL]
    assert engine.stats.last_block_kept == 0
