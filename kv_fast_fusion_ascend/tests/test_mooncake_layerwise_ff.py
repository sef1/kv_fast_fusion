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
    out = prod.on_layer(gi=1, layer_name="model.layers.2.attn", caches=base,
                        group_layer_names=group_layers, requests=requests)
    assert out is None, "group not complete after 1 of 2 layers"
    out = prod.on_layer(gi=1, layer_name="model.layers.3.attn", caches=base,
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
    new_blocks, n_applied, n_unresolved, n_owner_missing = resolve_redirect_rows(
        ext2blocks, hash2ext, "reqB", gi=1, rows=rows)
    assert new_blocks == [100], f"owner block table should point at rep block 100, got {new_blocks}"
    assert n_applied == 1 and n_unresolved == 0 and n_owner_missing == 0


def test_consumer_resolve_unresolved_when_rep_absent():
    ext2blocks = {"reqB": [[0], [200]]}  # rep reqA not resident
    hash2ext = {_ext_hash("reqB"): "reqB"}
    rows = [(0, _ext_hash("reqA"), 0)]
    new_blocks, n_applied, n_unresolved, n_owner_missing = resolve_redirect_rows(
        ext2blocks, hash2ext, "reqB", gi=1, rows=rows)
    assert new_blocks is None and n_applied == 0 and n_unresolved == 1 and n_owner_missing == 0


def test_rep_safe_skips_stale_overlay_rep():
    # BFF_FF_REP_SAFE: the owner set (overlay, res_*) still carries a STALE rep reqA, but the live rep set
    # does NOT → the rep must count as unresolved (owner keeps its own block), never a wrong repoint.
    owner_set = {"reqB": [[0], [200, 201]], "reqA": [[0], [999]]}   # overlay still has stale reqA
    hash2ext = {_ext_hash("reqA"): "reqA", _ext_hash("reqB"): "reqB"}
    rows = [(0, _ext_hash("reqA"), 0)]                              # reqB slot0 -> reqA slot0
    # rep resolved from LIVE-only set that lacks reqA → unresolved, owner unchanged
    nb, na, nu, nom = resolve_redirect_rows(
        owner_set, hash2ext, "reqB", gi=1, rows=rows,
        rep_ext2blocks={"reqB": [[0], [200, 201]]}, rep_hash2ext={})
    assert nb is None and na == 0 and nu == 1 and nom == 0
    # when the rep IS live, the same row applies (repoint slot0 -> 999)
    nb2, na2, nu2, nom2 = resolve_redirect_rows(
        owner_set, hash2ext, "reqB", gi=1, rows=rows,
        rep_ext2blocks={"reqA": [[0], [999]]}, rep_hash2ext={_ext_hash("reqA"): "reqA"})
    assert nb2 == [999, 201] and na2 == 1 and nu2 == 0


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
    new_blocks, na, nu, nom = resolve_redirect_rows(dblocks, hash2ext, owner_ext, 1, out[owner_ext])
    assert na == 1 and nu == 0 and nom == 0
    assert new_blocks == dblocks[rep_ext][1], "owner must share the representative's D block"


def test_lsh_cross_request_across_steps_without_encoded_batch():
    """Gate decoupling: with the default lsh backend and BFF_PD_ENCODED_BATCH_SIZE=0, a block seen in
    a LATER step must still cross-match the earlier step's registered rep. Before the decoupling this
    path only ran when the matrix FIFO window (_PD_ENCODED_BATCH) was > 0, so it produced 0 redirects
    here. The connector is imported with the env default (0), so this exercises the =0 case directly."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    if m._PD_CROSS_INDEX != "lsh":
        print("  (skipped: BFF_PD_CROSS_INDEX != lsh)")
        return
    assert m._PD_ENCODED_BATCH == 0, "test asserts the =0 (matrix-window-off) case"

    prod = MooncakeFFProducer()
    group_layers = {"L0", "L1"}
    shared = _make_k(1, seed=5)[0]

    # Step 1: reqA registers its rep into the LSH index (has_remote=True so it is registered).
    base1 = _cache_with_blocks({10: shared})
    prod.reset_step(1)
    prod.on_layer(1, "L0", base1, group_layers, [("reqA", [10], True)])
    out1 = prod.on_layer(1, "L1", base1, group_layers, [("reqA", [10], True)])
    assert sum(len(v) for v in out1.values()) == 0, f"single new block, nothing to merge: {out1}"

    # Step 2: reqB has an identical block in a different step -> LSH cross-match to reqA's rep.
    base2 = _cache_with_blocks({20: shared}, seed=9)
    prod.reset_step(2)
    prod.on_layer(1, "L0", base2, group_layers, [("reqB", [20], True)])
    out2 = prod.on_layer(1, "L1", base2, group_layers, [("reqB", [20], True)])
    assert out2 == {"reqB": [(0, _ext_hash("reqA"), 0)]}, \
        f"reqB must cross-redirect to reqA's rep across steps at ENCODED_BATCH=0, got {out2}"


def test_intra_req_ff_disables_within_batch_cc():
    """BFF_PD_INTRA_REQ_FF=0 must skip the within-batch cc merge (two same-step requests with identical
    blocks no longer fuse), while cross-request LSH still works: every unmatched block is registered, so
    an identical block in a LATER step cross-matches. Toggle the module constant directly (it's read
    live inside _build_send_rows) and restore it afterwards."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    if m._PD_CROSS_INDEX != "lsh":
        print("  (skipped: BFF_PD_CROSS_INDEX != lsh)")
        return
    saved = m._PD_INTRA_REQ_FF
    m._PD_INTRA_REQ_FF = False
    try:
        prod = MooncakeFFProducer()
        group_layers = {"L0", "L1"}
        shared = _make_k(1, seed=11)[0]

        # Step 1: two requests, identical blocks, same step. With within-batch cc OFF they must NOT
        # merge (index is empty so cross finds nothing) — but both reps get registered.
        base1 = _cache_with_blocks({10: shared, 20: shared})
        prod.reset_step(1)
        prod.on_layer(1, "L0", base1, group_layers, [("reqA", [10], True), ("reqB", [20], True)])
        out1 = prod.on_layer(1, "L1", base1, group_layers,
                             [("reqA", [10], True), ("reqB", [20], True)])
        assert out1 == {}, f"within-batch cc disabled must not merge same-step dupes, got {out1}"

        # Step 2: a later request with the same block cross-matches a registered rep from step 1.
        base2 = _cache_with_blocks({30: shared}, seed=13)
        prod.reset_step(2)
        prod.on_layer(1, "L0", base2, group_layers, [("reqC", [30], True)])
        out2 = prod.on_layer(1, "L1", base2, group_layers, [("reqC", [30], True)])
        assert list(out2.keys()) == ["reqC"], f"reqC must cross-match a step-1 rep, got {out2}"
        (owner_slot, rep_hash, rep_slot) = out2["reqC"][0]
        assert rep_hash in (_ext_hash("reqA"), _ext_hash("reqB")), \
            f"rep must be one of step-1's registered reqs, got hash {rep_hash}"
    finally:
        m._PD_INTRA_REQ_FF = saved


def _lsh_fill(prod, gi, n_entries, d, seed=0):
    """Register n_entries synthetic reps into group gi's LSH index; returns the normalized vectors."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n_entries, d, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    proj = m._lsh_get_proj(prod._lsh_proj, d, torch.device("cpu"))
    sh = m._lsh_sub_hashes(v, proj)
    ext_ids = [f"reg{j}" for j in range(n_entries)]
    prod._lsh_register(gi, v, sh, ext_ids, list(range(n_entries)), list(range(n_entries)),
                       list(range(n_entries)), [True] * n_entries)
    return v


def test_lsh_probe_matches_bruteforce_oracle():
    """The matrix-backed probe must pick exactly the rep a brute-force cosine scan would, for every
    block that has a bucket candidate above THRESHOLD. Guards the index_select verify rewrite."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    d, E, N = 64, 2000, 32
    prod = m.MooncakeFFProducer()
    reg = _lsh_fill(prod, 1, E, d, seed=3)

    # Probe blocks: half are near-duplicates of registered reps (guaranteed hits), half random.
    g = torch.Generator().manual_seed(99)
    q = torch.randn(N, d, generator=g)
    q[: N // 2] = reg[: N // 2] + 0.01 * torch.randn(N // 2, d, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    proj = m._lsh_get_proj(prod._lsh_proj, d, torch.device("cpu"))
    sh = m._lsh_sub_hashes(q, proj)
    ext_ids = [f"probe{i}" for i in range(N)]
    matched, hits = prod._lsh_probe(1, q, sh, ext_ids, list(range(N)))

    idx = prod._lsh[1]
    hit_by_block = {i: (rh, rs) for (i, rh, rs) in hits}
    for i in range(N):
        # Oracle: brute-force the exact candidate set the probe was allowed to consider.
        cand = set()
        for t, h in enumerate(sh[i]):
            cand.update(idx["tables"][t].get(h, ()))
        cand = sorted(cand)
        if not cand:
            assert i not in hit_by_block, f"block {i} hit with no candidates"
            continue
        rows = torch.tensor(cand, dtype=torch.long)
        sims = idx["mat"].index_select(0, rows) @ q[i]
        bv, bj = sims.max(dim=0)
        if bv.item() > m.THRESHOLD:
            assert matched[i], f"block {i}: oracle found {bv.item():.4f} > thr but probe missed"
            exp_hash, exp_slot, _ = idx["meta"][cand[int(bj.item())]]
            assert hit_by_block[i] == (exp_hash, exp_slot), f"block {i} picked the wrong rep"
        else:
            assert not matched[i], f"block {i}: probe hit but oracle best is {bv.item():.4f}"
    assert sum(matched) >= N // 4, f"expected the planted near-duplicates to hit, got {sum(matched)}"


def test_lsh_evict_compacts_and_still_probes():
    """Over-cap registration LRU-drops the oldest half and compacts `mat`; survivors must still be
    found, and _lsh_size must track the compacted row count."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    d = 32
    saved = m._LSH_MAX_ENTRIES
    m._LSH_MAX_ENTRIES = 64
    try:
        prod = m.MooncakeFFProducer()
        _lsh_fill(prod, 1, 64, d, seed=5)               # fill exactly to the cap
        assert prod._lsh_size(1) == 64
        late = _lsh_fill(prod, 1, 8, d, seed=6)         # triggers evict-half + compact
        idx = prod._lsh[1]
        assert prod._lsh_size(1) == len(idx["meta"]) == len(idx["owner"]) == len(idx["lru"])
        assert idx["mat"].shape[0] >= prod._lsh_size(1), "mat capacity must cover n_rows"
        assert prod._lsh_size(1) == 40, f"64 - 32 evicted + 8 new = 40, got {prod._lsh_size(1)}"
        # Compaction must renumber rows to a dense 0..n-1 and leave every bucket pointing at a live row.
        assert set(idx["meta"]) == set(range(40)), "rows must be dense after compaction"
        for t in idx["tables"]:
            for bucket in t.values():
                for r in bucket:
                    assert r in idx["meta"], "a bucket references an evicted/stale row"
        # A late-registered rep is still findable by an identical probe.
        proj = m._lsh_get_proj(prod._lsh_proj, d, torch.device("cpu"))
        qs = m._lsh_sub_hashes(late[:1], proj)
        matched, hits = prod._lsh_probe(1, late[:1], qs, ["other"], [0])
        assert matched[0] and hits, "a surviving rep must still be probe-able after compaction"
    finally:
        m._LSH_MAX_ENTRIES = saved


def test_redirect_channel_push_pull_contract():
    """Fire-and-forget wire contract for the FF redirect channel: a PUSH of
    ``(_FF_REDIRECT_MSG, ext_id, gi, rows)`` (msgpack) must decode on the PULL side into the same
    ``{ext_id: {gi: rows}}`` structure the recv thread builds — with NO ACK. The real send/recv thread
    classes are NPU-guarded (need vllm_ascend), so this replicates just their message handling."""
    try:
        import msgspec
        import zmq
    except Exception:
        print("  (skipped push/pull contract test: pyzmq/msgspec unavailable)")
        return

    _FF_REDIRECT_MSG = b"bff_redirect_msg"   # must match the connector's tag
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(f"tcp://127.0.0.1:{port}")
    enc = msgspec.msgpack.Encoder()
    dec = msgspec.msgpack.Decoder(type=tuple)

    # producer submits two groups for one request
    push.send(enc.encode((_FF_REDIRECT_MSG, "reqB", 1, [[0, 12345, 0]])))
    push.send(enc.encode((_FF_REDIRECT_MSG, "reqB", 2, [[1, 67890, 3]])))

    pending: dict = {}
    for _ in range(2):
        assert pull.poll(3000), "message not received on PULL socket"
        msg = dec.decode(pull.recv())
        assert msg[0] == _FF_REDIRECT_MSG
        _tag, ext_id, gi, rows = msg
        pending.setdefault(ext_id, {})[int(gi)] = rows
    ctx.destroy(linger=0)

    assert pending == {"reqB": {1: [[0, 12345, 0]], 2: [[1, 67890, 3]]}}, pending


if __name__ == "__main__":
    test_producer_detects_duplicate_across_requests()
    test_producer_no_merge_when_dissimilar()
    test_consumer_resolve_repoints_and_reports()
    test_consumer_resolve_unresolved_when_rep_absent()
    test_end_to_end_producer_to_consumer()
    test_lsh_cross_request_across_steps_without_encoded_batch()
    test_intra_req_ff_disables_within_batch_cc()
    test_lsh_probe_matches_bruteforce_oracle()
    test_lsh_evict_compacts_and_still_probes()
    test_redirect_channel_push_pull_contract()
    print("OK: all mooncake layerwise FF glue tests passed")
