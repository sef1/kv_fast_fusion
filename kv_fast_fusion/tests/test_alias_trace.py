"""BFF_V2_ALIAS_TRACE: alias insert / hit / drop, freed-victim hits and re-arrivals. CPU only.

Drives the real add_block_alias, patched_get_computed_blocks and patched_maybe_evict_cached_block
against a stand-in for vLLM's BlockHashToBlockMap with the same insert / get_one_block / pop
semantics (a key holds one block, or a {block_id: block} dict once a second block shares it).
"""

import logging
import types

import pytest

from kv_fast_fusion import fast_fusion_block_pool as bp


class _Map:
    def __init__(self):
        self._c = {}

    def get_one_block(self, key):
        v = self._c.get(key)
        if isinstance(v, dict):
            return next(iter(v.values()))
        return v

    def insert(self, key, block):
        v = self._c.get(key)
        if v is None:
            self._c[key] = block
        elif isinstance(v, dict):
            v[block.block_id] = block
        else:
            self._c[key] = {v.block_id: v, block.block_id: block}

    def pop(self, key, block_id):
        v = self._c.pop(key, None)
        if v is None:
            return None
        if isinstance(v, dict):
            b = v.pop(block_id, None)
            if v:
                self._c[key] = v
            return b
        if v.block_id == block_id:
            return v
        self._c[key] = v
        return None


class _Blk:
    def __init__(self, block_id, ref_cnt=1, block_hash=None):
        self.block_id, self.ref_cnt, self.block_hash, self.is_null = (
            block_id, ref_cnt, block_hash, False)


@pytest.fixture
def trace(monkeypatch, caplog):
    """Clean module state, the key builder the Ascend patch installs, and a captured log."""
    monkeypatch.setattr(bp, "_BLOCK_ALIASES", {})
    monkeypatch.setattr(bp, "_ALIAS_ORIGIN", {})
    monkeypatch.setattr(bp, "_FREED_AT_PROMOTION", {})
    monkeypatch.setattr(bp, "_RECOMPUTED_EXT", set())
    monkeypatch.setattr(bp, "_ALIAS_TRACE", True)
    monkeypatch.setattr(bp, "_ALIAS_ENABLED", True)
    monkeypatch.setattr(bp, "_KEY_OF", lambda h, g: (h, g))
    monkeypatch.setattr(bp, "_EXT_OF", lambda rid: rid.rsplit("-", 1)[0])
    caplog.set_level(logging.INFO, logger="vllm.bff_alias_trace")
    return caplog


def _lines(caplog, tag):
    return [r.getMessage() for r in caplog.records if tag in r.getMessage()]


def test_an_insert_says_what_the_key_held_before(trace):
    pool = types.SimpleNamespace(cached_block_hash_to_block=_Map())
    victim, rep = _Blk(51, ref_cnt=0, block_hash=("hS", 1)), _Blk(41, ref_cnt=2, block_hash="hO")
    pool.cached_block_hash_to_block.insert(("hS", 1), victim)

    bp.add_block_alias(pool, ("hS", 1), rep, req="S-aaaa", pos=0)

    (line,) = _lines(trace, "BFF alias insert")
    assert "req=S-aaaa pos=0 rep=41 rep_ref=2 rep_hashed=True key_had=51" in line
    # One key now answers with the victim first, the representative second.
    assert pool.cached_block_hash_to_block.get_one_block(("hS", 1)) is victim


def _lookup(monkeypatch, request, blocks_by_group, n):
    cb = types.SimpleNamespace(blocks=blocks_by_group)
    monkeypatch.setattr(bp, "_ORIG_GET_COMPUTED", lambda self, req: (cb, n))
    return bp.patched_get_computed_blocks(None, request)


def test_a_hit_through_the_alias_names_whom_it_was_made_for(trace, monkeypatch):
    pool = types.SimpleNamespace(cached_block_hash_to_block=_Map())
    rep = _Blk(41, ref_cnt=3, block_hash="hO")
    bp.add_block_alias(pool, ("hS", 1), rep, req="S-aaaa", pos=0)
    returning = types.SimpleNamespace(request_id="S-bbbb", block_hashes=["hS"])

    _lookup(monkeypatch, returning, ([_Blk(9)], [rep]), 128)

    (line,) = _lines(trace, "BFF alias hit")
    assert "req=S-bbbb g=1 pos=0 block=41 via=alias of=S-aaaa@0 ref_cnt=3 hit_tokens=128" in line


def test_the_owners_own_hit_on_its_block_is_not_an_alias_hit(trace, monkeypatch):
    pool = types.SimpleNamespace(cached_block_hash_to_block=_Map())
    rep = _Blk(41, block_hash="hO")
    bp.add_block_alias(pool, ("hS", 1), rep, req="S-aaaa", pos=0)
    owner = types.SimpleNamespace(request_id="O-cccc", block_hashes=["hO"])

    _lookup(monkeypatch, owner, ([_Blk(9)], [rep]), 128)

    assert _lines(trace, "BFF alias hit") == []


def test_a_hit_on_a_victim_freed_at_promotion_is_reported(trace, monkeypatch):
    bp._FREED_AT_PROMOTION[51] = "S"
    req = types.SimpleNamespace(request_id="R-dddd", block_hashes=["hS"])

    _lookup(monkeypatch, req, ([_Blk(9)], [_Blk(51, ref_cnt=0)]), 128)

    (line,) = _lines(trace, "BFF alias hit")
    assert "block=51 via=freed_victim of=S ref_cnt=0" in line


def test_a_recomputed_request_s_next_lookup_is_a_rearrival(trace, monkeypatch):
    bp._RECOMPUTED_EXT.add("S")
    _lookup(monkeypatch, types.SimpleNamespace(request_id="S-eeee", block_hashes=[]), ([], []), 0)

    (line,) = _lines(trace, "BFF rearrival")
    assert "req=S-eeee local_hit_tokens=0" in line


def test_evicting_the_rep_drops_its_aliases_and_says_so(trace, monkeypatch):
    pool = types.SimpleNamespace(cached_block_hash_to_block=_Map())
    rep = _Blk(41, ref_cnt=0, block_hash="hO")
    bp.add_block_alias(pool, ("hS", 1), rep, req="S-aaaa", pos=0)
    bp._FREED_AT_PROMOTION[41] = "X"
    monkeypatch.setattr(bp, "_ORIG_MAYBE_EVICT", lambda self, block: True)

    bp.patched_maybe_evict_cached_block(pool, rep)

    (line,) = _lines(trace, "BFF alias drop")
    assert "rep=41 keys=1 ref_cnt=0 for=S-aaaa@0" in line
    assert pool.cached_block_hash_to_block.get_one_block(("hS", 1)) is None
    assert 41 not in bp._FREED_AT_PROMOTION and 41 not in bp._ALIAS_ORIGIN
