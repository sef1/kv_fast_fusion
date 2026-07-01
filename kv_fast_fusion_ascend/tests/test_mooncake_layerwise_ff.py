"""Unit tests for the transport-agnostic fusion glue in
``kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff`` (BFF Mooncake M1, raw mode).

These exercise the pure producer clustering + consumer resolve WITHOUT the Ascend/NPU stack or ZMQ
(the connector subclass is guarded behind an import of vllm_ascend). Run with:
    python -m pytest kv_fast_fusion_ascend/tests/test_mooncake_layerwise_ff.py -q
or standalone:
    python kv_fast_fusion_ascend/tests/test_mooncake_layerwise_ff.py
"""

import torch

from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
    MooncakeFFProducer,
    _ext_hash,
    resolve_redirect_rows,
)


def _make_k(num_blocks, block_size=4, heads=2, head_dim=8, seed=0):
    """A paged K cache indexed by physical block id: shape [num_blocks, block_size, heads, dim]."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_blocks, block_size, heads, head_dim, generator=g)


def _cache_with_blocks(block_vecs, num_blocks=32, block_size=4, heads=2, head_dim=8, seed=0):
    """Build a paged cache of ``num_blocks`` rows and place each (block_id -> vector) at its row."""
    g = torch.Generator().manual_seed(seed)
    cache = torch.randn(num_blocks, block_size, heads, head_dim, generator=g)
    for bid, vec in block_vecs.items():
        cache[bid] = vec
    return cache


def test_producer_detects_duplicate_across_requests():
    """Two requests whose group-K blocks are identical must produce a redirect from the longer
    (owner) request's block to the shorter (representative) request's block."""
    # Group of 2 layers, 1 block per request, 2 requests with IDENTICAL K → should merge.
    prod = MooncakeFFProducer()
    # physical block ids: req A -> [10], req B -> [20]
    requests = [("reqA", [10]), ("reqB", [20])]
    group_layers = {"model.layers.2.attn", "model.layers.3.attn"}

    # identical K at physical blocks 10 (A) and 20 (B), across both layers
    shared = _make_k(1, seed=1)[0]
    base = _cache_with_blocks({10: shared, 20: shared})
    step_id = 12345
    prod.reset_step(step_id)
    out = prod.on_layer(gi=1, layer_name="model.layers.2.attn", k_cache=base,
                        group_layer_names=group_layers, requests=requests)
    assert out is None, "group not complete after 1 of 2 layers"
    out = prod.on_layer(gi=1, layer_name="model.layers.3.attn", k_cache=base,
                        group_layer_names=group_layers, requests=requests)
    assert out is not None, "group complete after 2 layers"

    # exactly one redirect, from the owner to the rep. Owner is the request whose block got merged
    # away (labels[i] != i); with equal-length reqs the nr_tree picks the left as rep.
    total_rows = sum(len(v) for v in out.values())
    assert total_rows == 1, f"expected 1 redirect, got {out}"
    owner_ext = next(iter(out))
    (owner_slot, rep_hash, rep_slot) = out[owner_ext][0]
    assert owner_slot == 0 and rep_slot == 0
    rep_ext = "reqA" if owner_ext == "reqB" else "reqB"
    assert rep_hash == _ext_hash(rep_ext)


def test_producer_no_merge_when_dissimilar():
    prod = MooncakeFFProducer()
    requests = [("reqA", [10]), ("reqB", [20])]
    group_layers = {"L0", "L1"}
    k = _cache_with_blocks({}, seed=7)  # blocks 10 and 20 are distinct random rows
    prod.reset_step(1)
    assert prod.on_layer(1, "L0", k, group_layers, requests) is None
    out = prod.on_layer(1, "L1", k, group_layers, requests)
    assert out == {}, f"dissimilar blocks must not merge, got {out}"


def test_consumer_resolve_repoints_and_reports():
    """Feed the producer's rows into the consumer resolver with a fake D block-id table and assert
    the owner slot is repointed at the representative's physical block."""
    # D-side per-group block ids: group 0 unused, group 1 is the fusion group.
    ext2blocks = {
        "reqA": [[0], [100]],   # rep: group1 physical block 100
        "reqB": [[0], [200]],   # owner: group1 physical block 200 (to be repointed to 100)
    }
    hash2ext = {_ext_hash("reqA"): "reqA", _ext_hash("reqB"): "reqB"}
    rows = [(0, _ext_hash("reqA"), 0)]  # owner slot 0 -> reqA slot 0
    new_blocks, n_applied, n_unresolved = resolve_redirect_rows(
        ext2blocks, hash2ext, "reqB", gi=1, rows=rows)
    assert new_blocks == [100], f"owner block table should point at rep block 100, got {new_blocks}"
    assert n_applied == 1 and n_unresolved == 0


def test_consumer_resolve_unresolved_when_rep_absent():
    ext2blocks = {"reqB": [[0], [200]]}  # rep reqA not resident
    hash2ext = {_ext_hash("reqB"): "reqB"}
    rows = [(0, _ext_hash("reqA"), 0)]
    new_blocks, n_applied, n_unresolved = resolve_redirect_rows(
        ext2blocks, hash2ext, "reqB", gi=1, rows=rows)
    assert new_blocks is None and n_applied == 0 and n_unresolved == 1


def test_end_to_end_producer_to_consumer():
    """Full round-trip: cluster on P, resolve on D, and confirm the merged-away request ends up
    sharing the representative's physical block."""
    prod = MooncakeFFProducer()
    requests = [("reqA", [10]), ("reqB", [20])]
    group_layers = {"L0", "L1"}
    shared = _make_k(1, seed=3)[0]
    base = _cache_with_blocks({10: shared, 20: shared})
    prod.reset_step(99)
    prod.on_layer(1, "L0", base, group_layers, requests)
    out = prod.on_layer(1, "L1", base, group_layers, requests)
    assert sum(len(v) for v in out.values()) == 1

    owner_ext = next(iter(out))
    rep_ext = "reqA" if owner_ext == "reqB" else "reqB"
    # D physical blocks (arbitrary but distinct)
    dblocks = {"reqA": [[0], [100]], "reqB": [[0], [200]]}
    hash2ext = {_ext_hash("reqA"): "reqA", _ext_hash("reqB"): "reqB"}
    new_blocks, na, nu = resolve_redirect_rows(dblocks, hash2ext, owner_ext, 1, out[owner_ext])
    assert na == 1 and nu == 0
    assert new_blocks == dblocks[rep_ext][1], "owner must share the representative's D block"


if __name__ == "__main__":
    test_producer_detects_duplicate_across_requests()
    test_producer_no_merge_when_dissimilar()
    test_consumer_resolve_repoints_and_reports()
    test_consumer_resolve_unresolved_when_rep_absent()
    test_end_to_end_producer_to_consumer()
    print("OK: all mooncake layerwise FF glue tests passed")
