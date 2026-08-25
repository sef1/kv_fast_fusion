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
    FusionTask,
    MooncakeFFProducer,
    _classify_owner_miss,
    _ext_hash,
    _ff_write_runner_block_table,
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


def test_lsh_probe_bins_accepted_cosines():
    """Every accepted merge must land in the accept-cosine histogram bin matching its verify cosine.
    Bucket collision at cos~0.78 is unreliable (P~12% at 16 tables x 20 bits), so the probe REUSES
    the reps' own sub-hashes — candidates guaranteed, and the verify cosine is exactly the crafted
    one."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    d = 64
    prod = m.MooncakeFFProducer()
    g = torch.Generator().manual_seed(5)
    u = torch.randn(2, d, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    proj = m._lsh_get_proj(prod._lsh_proj, d, torch.device("cpu"))
    sh_u = m._lsh_sub_hashes(u, proj)
    prod._lsh_register(1, u, sh_u, ["repA", "repB"], [0, 1], [0, 1], [0, 1], [True, True])

    def _at_cos(base, c):
        # Unit vector at exact cosine c to unit `base`: c*base + sqrt(1-c^2)*w with w ⊥ base.
        w = torch.randn(d, generator=g)
        w = w - (w @ base) * base
        w = w / w.norm()
        return c * base + (1.0 - c * c) ** 0.5 * w

    q = torch.stack([_at_cos(u[0], 0.99), _at_cos(u[1], 0.78)])
    matched, hits = prod._lsh_probe(1, q, sh_u, ["own0", "own1"], [0, 1])
    assert matched == [True, True] and len(hits) == 2, (matched, hits)
    hist = dict(zip(m._ACCEPT_COS_LABELS, prod._accept_cos[1]))
    assert hist["0.98-1.00"] == 1, hist
    assert hist["0.75-0.80"] == 1, hist
    assert sum(prod._accept_cos[1]) == len(hits)


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


def test_ff_groups_parsing():
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    # Unset / empty / degenerate all mean "all eligible groups", never "fuse nothing" — a typo
    # must not silently disable fusion.
    for raw in (None, "", "   ", ",,", " , "):
        assert m._parse_groups(raw) is None, raw
    assert m._parse_groups("1,2,3") == {1, 2, 3}
    assert m._parse_groups("1, 2,") == {1, 2}      # tolerate spaces + trailing comma
    assert m._parse_groups("3") == {3}


def test_ff_groups_filters_fusion_set():
    """The knob intersects with the eligible set; it can never ADD an ineligible group."""
    eligible = {1, 2, 3, 4, 5, 6}

    def apply(selected):
        return eligible if selected is None else eligible & selected

    assert apply(None) == eligible                      # default = unchanged behavior
    assert apply({1, 2, 3}) == {1, 2, 3}                 # the measured-productive groups
    assert apply({1, 99}) == {1}                         # 99 is not eligible -> ignored
    assert apply({0}) == set()                           # group 0 is warmup, never eligible


# --------------------------------------------------------------------------------------------
# Write-success coupling: the free MUST be withheld unless the device block table was rewritten.
# This is the exact invariant the con512 corruption violated — a just-recv'd owner not yet in the
# step's input_batch had its blocks freed while its table was never repointed, so it decoded
# against freed-then-reallocated KV. _ff_write_runner_block_table returns whether it wrote; the
# apply site frees only on True.
# --------------------------------------------------------------------------------------------

class _FakeBlockTable:
    """Mirrors vllm_ascend's BlockTable closely enough for the write path.

    `.np` is a numpy VIEW of the pinned `.cpu` tensor, exactly as CpuGpuBuffer builds it — the
    connector publishes to the device by copying that row, so a fixture with independent arrays
    would pass a test the real thing would fail."""

    def __init__(self, rows, cols, use_hybrid_blocks=False):
        self.num_blocks_per_row = [cols] * rows
        self.use_hybrid_blocks = use_hybrid_blocks
        self.physical_block_size = 128
        self.block_size = 64 if use_hybrid_blocks else 128
        self.blocks_per_phys_block = 2 if use_hybrid_blocks else 1
        self.block_table = type("BT", (), {})()
        self.block_table.cpu = torch.zeros(rows, cols, dtype=torch.long)
        self.block_table.np = self.block_table.cpu.numpy()
        self.block_table.gpu = torch.zeros(rows, cols, dtype=torch.long)


class _FakeReqState:
    def __init__(self, ngroups, cols):
        self.block_ids = [[0] * cols for _ in range(ngroups)]


class _FakeRunner:
    """Minimal stand-in for the NPU model runner: only what _ff_write_runner_block_table touches."""
    def __init__(self, resident_rids, ngroups=2, cols=4, use_hybrid_blocks=False):
        self.input_batch = type("IB", (), {})()
        self.input_batch.req_id_to_index = {rid: i for i, rid in enumerate(resident_rids)}
        self.input_batch.block_table = [
            _FakeBlockTable(len(resident_rids), cols, use_hybrid_blocks)
            for _ in range(ngroups)]
        self.requests = {rid: _FakeReqState(ngroups, cols) for rid in resident_rids}


def test_save_kv_positions_resolve_by_name_once():
    """The producer hook binds save_kv_layer's arguments by NAME but must resolve them exactly once,
    at install time — a Signature.bind() per call would land on the prefill forward thread for every
    attention layer of every step, which is the latency this project exists to measure.

    So the resolver returns positional indices. They must track the name through an appended
    parameter and through a reordering, and must return None (fusion goes inert, transfer untouched)
    when the base no longer exposes the names at all."""
    from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import (
        resolve_save_kv_positions,
    )

    # The signature the connector ships today (bound method → no self).
    def current(layer_name, kv_layer, attn_metadata, connector_metadata, **kwargs): pass
    assert resolve_save_kv_positions(current) == (0, 1, 3)

    # A drifted base that APPENDS a parameter — exactly the class of change that broke the vendored
    # listener (update_task gained side_channel_path). Indices must be unaffected.
    def appended(layer_name, kv_layer, attn_metadata, connector_metadata, side_channel_path=None): pass
    assert resolve_save_kv_positions(appended) == (0, 1, 3)

    # A drifted base that REORDERS: positional binding would silently hand fusion the wrong tensor.
    def reordered(kv_layer, layer_name, connector_metadata, attn_metadata): pass
    assert resolve_save_kv_positions(reordered) == (1, 0, 2)

    # Names gone entirely → None, so the caller disables fusion loudly instead of guessing.
    def unrecognized(a, b, c): pass
    assert resolve_save_kv_positions(unrecognized) is None

    # And the indices must agree with what the values actually are for a real call.
    i_ln, i_kv, i_cm = resolve_save_kv_positions(reordered)
    args = ("KV", "L0", "CM", "AM")          # reordered(kv_layer, layer_name, connector_metadata, ...)
    assert (args[i_ln], args[i_kv], args[i_cm]) == ("L0", "KV", "CM")


def test_write_returns_false_for_absent_rid():
    """An owner not in this step's input_batch → no write, so the caller must withhold the free."""
    runner = _FakeRunner(["reqA"])
    assert _ff_write_runner_block_table(runner, "reqGHOST", 1, [100, 101]) is False


def test_write_returns_true_and_repoints_resident_rid():
    runner = _FakeRunner(["reqA"], ngroups=2, cols=4)
    assert _ff_write_runner_block_table(runner, "reqA", 1, [100, 101]) is True
    # device mirror (np + gpu) and the request's own block_ids all repointed
    assert list(runner.input_batch.block_table[1].block_table.np[0, :2]) == [100, 101]
    assert runner.input_batch.block_table[1].block_table.gpu[0, :2].tolist() == [100, 101]
    assert runner.requests["reqA"].block_ids[1][:2] == [100, 101]


def test_the_device_row_agrees_with_the_host_row_after_every_write():
    """The device tensor is what attention reads; `.np` is what the runner's own commit will publish
    next step. They must agree after each write, including a second write over the first.

    This pins the INVARIANT, not the mechanism: copying from the pinned `.cpu` row and building a
    tensor from the Python list both satisfy it. The reason to prefer the former is that it is a
    pinned, non-blocking copy rather than a pageable one on the forward thread — a cost, not a
    correctness property, so it is documented at the call site rather than asserted here."""
    runner = _FakeRunner(["reqA"], ngroups=2, cols=4)

    _ff_write_runner_block_table(runner, "reqA", 1, [100, 101])
    bt = runner.input_batch.block_table[1].block_table
    assert bt.gpu[0, :2].tolist() == bt.np[0, :2].tolist() == [100, 101]

    _ff_write_runner_block_table(runner, "reqA", 1, [7, 8, 9])
    assert bt.gpu[0, :3].tolist() == bt.np[0, :3].tolist() == [7, 8, 9]


def test_a_buffer_without_a_pinned_mirror_still_publishes():
    """The fallback path. A block table whose buffer has no `.cpu` — a fake, or a future buffer
    type — must still reach the device rather than silently leaving it stale."""
    runner = _FakeRunner(["reqA"], ngroups=2, cols=4)
    del runner.input_batch.block_table[1].block_table.cpu
    runner.input_batch.block_table[1].block_table.np = \
        runner.input_batch.block_table[1].block_table.np.copy()

    assert _ff_write_runner_block_table(runner, "reqA", 1, [100, 101]) is True
    assert runner.input_batch.block_table[1].block_table.gpu[0, :2].tolist() == [100, 101]


def test_a_hybrid_block_table_is_refused_rather_than_corrupted():
    """Hybrid mode stores LOGICAL block ids, expanded per physical block, and counts logical blocks
    in num_blocks_per_row. Writing physical ids at a physical stride would be wrong in value, wrong
    in stride, and bounded by a count in the other unit — corrupting the head of the sequence, which
    is the prompt, in silence.

    It cannot arise at block size 128 (the backend's only supported kernel size), so refusing costs
    nothing today and cannot corrupt tomorrow. False is the existing 'I did not write' signal, so
    the caller withholds the block free and redirects degrade to off."""
    import kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff as mod
    mod._HYBRID_WARNED = False
    runner = _FakeRunner(["reqA"], ngroups=2, cols=4, use_hybrid_blocks=True)

    assert _ff_write_runner_block_table(runner, "reqA", 1, [100, 101]) is False

    bt = runner.input_batch.block_table[1].block_table
    assert bt.np[0].tolist() == [0, 0, 0, 0], "the table must be left exactly as it was"
    assert bt.gpu[0].tolist() == [0, 0, 0, 0]
    assert runner.requests["reqA"].block_ids[1] == [0, 0, 0, 0]


def test_a_refused_hybrid_write_withholds_the_free():
    """The coupling that matters: a refusal must reach the apply site as 'do not free', or the
    request would keep a table nobody repointed while its blocks were handed to someone else."""
    import kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff as mod
    mod._HYBRID_WARNED = False
    runner = _FakeRunner(["live"], use_hybrid_blocks=True)

    updated = {}
    if _ff_write_runner_block_table(runner, "live", 1, [100, 101]):
        updated["live"] = [100, 101]

    assert updated == {}


def test_free_is_coupled_to_write_success():
    """Replays the apply-site decision: `updated` (which drives the scheduler free) gets an entry
    IFF the write returned True. A resident owner is freed; an absent one is not."""
    runner = _FakeRunner(["live"])
    updated = {}
    owner_not_written = 0
    for rid, new_blocks in (("live", [100, 101]), ("gone", [200, 201])):
        if rid is not None and _ff_write_runner_block_table(runner, rid, 1, new_blocks):
            updated.setdefault(rid, {})[1] = new_blocks
        else:
            owner_not_written += 1
    assert updated == {"live": {1: [100, 101]}}, updated
    assert owner_not_written == 1


# --------------------------------------------------------------------------------------------
# owner-miss classifier (BFF_FF_AUDIT diagnostic): split owner_unresident into the two causes
# so the pending run says which few-line fix to make. never_snap = the overlay never held it;
# pruned = the overlay held it once and lost it before the recv landed.
# --------------------------------------------------------------------------------------------

def test_classify_owner_miss_never_snapshotted():
    ever = set()                       # ext was never put into the overlay
    assert _classify_owner_miss("reqZ", ever) == "never_snap"


def test_classify_owner_miss_pruned():
    ever = {"reqZ"}                     # ext DID enter the overlay, then was removed before landing
    assert _classify_owner_miss("reqZ", ever) == "pruned"


# --------------------------------------------------------------------------------------------
# Promotion-time apply (_bff_promotion_apply): the owner's req_to_blocks must be rewritten at the
# instant it leaves WAITING_FOR_REMOTE_KVS — before its first schedule — via the existing
# _handle_block_merging_with_counts machinery. The con512 audit proved the old recv-step window is
# structurally dead (owner joins input_batch only at the NEXT schedule → device write fails →
# nothing is ever freed).
# --------------------------------------------------------------------------------------------
import threading

from vllm.v1.request import RequestStatus

import kv_fast_fusion.fast_fusion_block_pool as _bp_mod
from kv_fast_fusion_ascend.fast_fusion_ascend_patch import _bff_promotion_apply


class _FakeSource:
    """Stands in for the published _FFRedirectRecvThread: lock + pending + promo_stats."""
    def __init__(self, pending):
        self.lock = threading.Lock()
        self.pending = pending
        self.promo_stats = {"promo_applied": 0, "promo_unresolved": 0, "promo_no_rows": 0,
                            "promo_merge_calls": 0, "promo_pending_dropped": 0,
                            "promo_unres_rep_loading": 0, "promo_unres_rep_gone": 0,
                            "repgone_revive_live": 0, "repgone_revive_cached": 0,
                            "repgone_truly_gone": 0, "repgone_no_history": 0}


class _FakeKVBlock:
    def __init__(self, ref_cnt=0, block_hash=None):
        self.ref_cnt = ref_cnt
        self.block_hash = block_hash


class _FakePool:
    """Minimal BlockPool stand-in: .blocks indexed by block_id (list, like the real pool)."""
    def __init__(self, blocks_by_id):
        n = max(blocks_by_id) + 1
        self.blocks = [_FakeKVBlock() for _ in range(n)]
        for bid, blk in blocks_by_id.items():
            self.blocks[bid] = blk


class _FakeBlock:
    def __init__(self, bid):
        self.block_id = bid


class _FakeManager:
    def __init__(self, req_to_blocks):
        self.req_to_blocks = {rid: [_FakeBlock(b) for b in bids]
                              for rid, bids in req_to_blocks.items()}


class _FakeSchedReq:
    def __init__(self, status):
        self.status = status


class _FakeScheduler:
    """Only what _bff_promotion_apply touches; _handle_block_merging_with_counts is recorded."""
    def __init__(self, requests, managers, block_pool=None):
        self.requests = requests
        self.kv_cache_manager = type("KM", (), {})()
        self.kv_cache_manager.coordinator = type("CO", (), {})()
        self.kv_cache_manager.coordinator.single_type_managers = managers
        self.kv_cache_manager.coordinator.block_pool = block_pool
        self.merge_calls = []

    def _handle_block_merging_with_counts(self, request_blocks):
        self.merge_calls.append(request_blocks)


def test_promotion_apply_rewrites_and_feeds_merge():
    # Owner "own-abcd1234" (ext "own") promoted with a pending redirect: group1 slot0 -> rep slot0.
    # Rep "rep-abcd1234" (ext "rep") is load-complete (RUNNING). Expect the merge machinery to be
    # fed {owner_rid: {1: [rep_block, own_block1]}} and the pending entry consumed.
    owner_rid, rep_rid = "own-abcd1234", "rep-abcd1234"
    pending = {"own": {1: [[0, _ext_hash("rep"), 0]]}}
    managers = [_FakeManager({}),                                            # group 0 (warmup)
                _FakeManager({owner_rid: [200, 201], rep_rid: [100, 101]})]  # group 1
    requests = {rep_rid: _FakeSchedReq(RequestStatus.RUNNING)}
    src = _FakeSource(pending)
    sched = _FakeScheduler(requests, managers)
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        owner = _FakeSchedReq(RequestStatus.WAITING)
        owner.request_id = owner_rid
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == [{owner_rid: {1: [100, 201]}}], sched.merge_calls
    assert src.pending == {}                                   # consumed exactly once
    assert src.promo_stats["promo_applied"] == 1
    assert src.promo_stats["promo_merge_calls"] == 1
    assert src.promo_stats["promo_unresolved"] == 0


def test_promotion_apply_skips_still_loading_rep():
    # The rep is still WAITING_FOR_REMOTE_KVS: its blocks exist but its KV is half-arrived.
    # Repointing there would be silent corruption → the row must count unresolved, no merge fed.
    owner_rid, rep_rid = "own-abcd1234", "rep-abcd1234"
    pending = {"own": {1: [[0, _ext_hash("rep"), 0]]}}
    managers = [_FakeManager({}),
                _FakeManager({owner_rid: [200, 201], rep_rid: [100, 101]})]
    requests = {rep_rid: _FakeSchedReq(RequestStatus.WAITING_FOR_REMOTE_KVS)}
    src = _FakeSource(pending)
    sched = _FakeScheduler(requests, managers)
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        owner = _FakeSchedReq(RequestStatus.WAITING)
        owner.request_id = owner_rid
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == []
    assert src.promo_stats["promo_unresolved"] == 1
    assert src.promo_stats["promo_applied"] == 0
    # Cause split: the rep IS present, just still loading — must land in the loading bucket only.
    assert src.promo_stats["promo_unres_rep_loading"] == 1
    assert src.promo_stats["promo_unres_rep_gone"] == 0


def test_promotion_apply_classifies_rep_gone():
    # The rep's request has already finished — no request in scheduler.requests carries its hash.
    # The row must count unresolved AND land in the rep-gone bucket (rep-lifetime problem).
    owner_rid = "own-abcd1234"
    pending = {"own": {1: [[0, _ext_hash("rep"), 0]]}}
    managers = [_FakeManager({}),
                _FakeManager({owner_rid: [200, 201]})]
    src = _FakeSource(pending)
    sched = _FakeScheduler({}, managers)                       # rep nowhere to be found
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        owner = _FakeSchedReq(RequestStatus.WAITING)
        owner.request_id = owner_rid
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == []
    assert src.promo_stats["promo_unresolved"] == 1
    assert src.promo_stats["promo_unres_rep_gone"] == 1
    assert src.promo_stats["promo_unres_rep_loading"] == 0


def test_promotion_apply_mixed_split_sums_to_unresolved():
    # Three rows in one promotion: one resolves (RUNNING rep), one hits a still-loading rep, one a
    # finished rep. The split buckets must sum exactly to promo_unresolved, and the resolvable row
    # must still be applied (partial merges are fed, unresolved slots keep the owner's own blocks).
    owner_rid, rep_a, rep_b = "own-abcd1234", "repa-abcd1234", "repb-abcd1234"
    pending = {"own": {1: [[0, _ext_hash("repa"), 0],
                           [1, _ext_hash("repb"), 0],
                           [2, _ext_hash("repc"), 0]]}}
    managers = [_FakeManager({}),
                _FakeManager({owner_rid: [200, 201, 202],
                              rep_a: [100, 101], rep_b: [110, 111]})]
    requests = {rep_a: _FakeSchedReq(RequestStatus.RUNNING),
                rep_b: _FakeSchedReq(RequestStatus.WAITING_FOR_REMOTE_KVS)}
    src = _FakeSource(pending)
    sched = _FakeScheduler(requests, managers)
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        owner = _FakeSchedReq(RequestStatus.WAITING)
        owner.request_id = owner_rid
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == [{owner_rid: {1: [100, 201, 202]}}], sched.merge_calls
    assert src.promo_stats["promo_applied"] == 1
    assert src.promo_stats["promo_unresolved"] == 2
    assert src.promo_stats["promo_unres_rep_loading"] == 1
    assert src.promo_stats["promo_unres_rep_gone"] == 1


def test_promotion_apply_no_rows_is_noop():
    owner = _FakeSchedReq(RequestStatus.WAITING)
    owner.request_id = "own-abcd1234"
    src = _FakeSource({})
    sched = _FakeScheduler({}, [_FakeManager({})])
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == []
    assert src.promo_stats["promo_no_rows"] == 1


def test_promotion_apply_skips_preempted_resume():
    # A preempted-resume re-enters with partial state; its redirect targeted a dead lifetime.
    # The pending entry must NOT be consumed (the janitor drops it on finish).
    owner = _FakeSchedReq(RequestStatus.PREEMPTED)
    owner.request_id = "own-abcd1234"
    pending = {"own": {1: [[0, _ext_hash("rep"), 0]]}}
    src = _FakeSource(pending)
    sched = _FakeScheduler({}, [_FakeManager({})])
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == []
    assert src.pending == pending          # untouched


def test_promotion_apply_unpublished_source_is_noop():
    owner = _FakeSchedReq(RequestStatus.WAITING)
    owner.request_id = "own-abcd1234"
    sched = _FakeScheduler({}, [_FakeManager({})])
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = None
    try:
        _bff_promotion_apply(sched, owner)     # must not raise
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert sched.merge_calls == []


def test_promotion_repgone_revivability_buckets():
    # Stage 1a: a rep-gone row (rep finished → absent from scheduler.requests) is classified by
    # whether its OLD physical block still holds KV, via _REP_HISTORY + the pool. Owner block at
    # group1 slot0 → rep "goner" slot0, whose old block id is 100. Vary the pool state of block 100.
    import kv_fast_fusion_ascend.fast_fusion_ascend_patch as _fp
    owner_rid = "own-abcd1234"
    rep_ext, rep_bid = "goner", 100

    def _run(block100):
        _fp._REP_HISTORY.clear()
        _fp._REP_HISTORY[_ext_hash(rep_ext)] = [[], [rep_bid]]     # rep's OLD blocks: gi1 slot0 = 100
        pending = {"own": {1: [[0, _ext_hash(rep_ext), 0]]}}
        managers = [_FakeManager({}), _FakeManager({owner_rid: [200, 201]})]  # rep NOT in managers
        pool = _FakePool({rep_bid: block100})
        src = _FakeSource(pending)
        sched = _FakeScheduler({}, managers, block_pool=pool)       # rep absent from live requests
        prev = _bp_mod._FF_PENDING_SOURCE
        _bp_mod._FF_PENDING_SOURCE = src
        try:
            owner = _FakeSchedReq(RequestStatus.WAITING)
            owner.request_id = owner_rid
            _bff_promotion_apply(sched, owner)
        finally:
            _bp_mod._FF_PENDING_SOURCE = prev
        return src.promo_stats

    s = _run(_FakeKVBlock(ref_cnt=2, block_hash=None))
    assert s["promo_unres_rep_gone"] == 1 and s["repgone_revive_live"] == 1, s

    s = _run(_FakeKVBlock(ref_cnt=0, block_hash=("h",)))
    assert s["repgone_revive_cached"] == 1 and s["repgone_revive_live"] == 0, s

    s = _run(_FakeKVBlock(ref_cnt=0, block_hash=None))
    assert s["repgone_truly_gone"] == 1, s

    # No history for the rep → no_history bucket (window too small in the real run).
    _fp._REP_HISTORY.clear()
    pending = {"own": {1: [[0, _ext_hash("unknown"), 0]]}}
    managers = [_FakeManager({}), _FakeManager({owner_rid: [200, 201]})]
    src = _FakeSource(pending)
    sched = _FakeScheduler({}, managers, block_pool=_FakePool({0: _FakeKVBlock()}))
    prev = _bp_mod._FF_PENDING_SOURCE
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        owner = _FakeSchedReq(RequestStatus.WAITING)
        owner.request_id = owner_rid
        _bff_promotion_apply(sched, owner)
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
    assert src.promo_stats["repgone_no_history"] == 1, src.promo_stats
    # ...and the split attributes it to the specific cause (rep never recorded here).
    assert src.promo_stats["repgone_nohist_missing"] == 1, src.promo_stats
    # The owner recorded ITSELF into the ring (available as a rep for future owners).
    assert _ext_hash("own") in _fp._REP_HISTORY


def test_repgone_nohist_split_distinguishes_causes():
    """The three nohist_* causes must be told apart: a rep that was never recorded is a rep-LIFETIME
    miss (content-hash naming fixes it), whereas a rep recorded with a shorter block list for this
    group is a P/D block-table SHAPE mismatch (it does not). They were one label until the con128
    run made the combined bucket the largest loss category at 768 of 1574 rows."""
    from kv_fast_fusion_ascend import fast_fusion_ascend_patch as p
    pool = _FakePool({0: _FakeKVBlock(ref_cnt=1)})
    h = _ext_hash("rep")
    p._REP_HISTORY.clear()
    assert p._rep_gone_bucket(pool, h, 1, 0) == "nohist_missing"
    # Recorded, but only group 0 exists → the row references a group the rep never had.
    p._REP_HISTORY[h] = [[0]]
    assert p._rep_gone_bucket(pool, h, 1, 0) == "nohist_gi_oob"
    # Group exists but is SHORTER than rep_slot → shape mismatch, not a lifetime problem.
    p._REP_HISTORY[h] = [[0], [0]]
    assert p._rep_gone_bucket(pool, h, 1, 5) == "nohist_slot_oob"
    # In range and live → revivable, not a nohist case at all.
    assert p._rep_gone_bucket(pool, h, 1, 0) == "revive_live"
    p._REP_HISTORY.clear()


def test_skip_transfer_bandwidth_accounting():
    # Stage 0: the GROSS skip-transfer ceiling in block-layer units. Group 0 (warmup, non-fusion)
    # contributes to the denominator only; group 1 (3 layers) redirects 3 blocks → each skip omits
    # all 3 of its group's layers. fraction = Σ redir×layers / total = (3×3)/40.
    import tempfile, os, json
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    prod = m.MooncakeFFProducer()
    for _ in range(2):
        prod.note_transferred(8)             # group 0: 2 layers × 8 blocks (denominator only)
    for _ in range(3):
        prod.note_transferred(8)             # group 1: 3 layers × 8 blocks
    prod.redir_total[1] = 3
    prod.blk_total[1] = 8
    prod.layers_per_group[1] = 3
    d = tempfile.mkdtemp()
    prod.dump_stats(d)
    s = json.load(open(os.path.join(d, f"bff_stats_{os.getpid()}.json")))
    assert s["total_block_layers"] == 40, s["total_block_layers"]
    assert s["skip_block_layers"] == 9, s["skip_block_layers"]
    assert abs(s["skip_bandwidth_fraction"] - 9 / 40) < 1e-9, s["skip_bandwidth_fraction"]


# --------------------------------------------------------------------------------------------
# Step 1: the fusion check moved off the prefill forward thread. The whole safety argument is that
# deferring changes WHERE the work runs and nothing else, so these pin exactly that.
# --------------------------------------------------------------------------------------------
class _async_config:
    """Force the production con512 fusion config (lsh cross-index, within-batch cc off) for the
    duration of a test, since the MODULE defaults leave intra_req_ff on and would otherwise close
    the async gate and silently skip these."""

    def __enter__(self):
        from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
        self._m = m
        self._saved = (m._PD_CROSS_INDEX, m._PD_INTRA_REQ_FF)
        m._PD_CROSS_INDEX, m._PD_INTRA_REQ_FF = "lsh", False
        return m

    def __exit__(self, *exc):
        self._m._PD_CROSS_INDEX, self._m._PD_INTRA_REQ_FF = self._saved
        return False


def _run_fusion_steps(prod, use_task, steps, group_layers=("L0", "L1"), gi=1):
    """Drive ``steps`` (each a list of (ext_id, cache, block_ids)) through a producer, either
    inline or via the prepare/run task split, and collect the per-step redirect dicts."""
    group_layers = set(group_layers)
    out = []
    for si, reqs in enumerate(steps):
        prod.reset_step(si + 1)
        req_tuples = [(ext, list(bids), True) for (ext, _c, bids) in reqs]
        cache = reqs[0][1]
        res = None
        for ln in sorted(group_layers):
            res = prod.on_layer(gi, ln, cache, group_layers, req_tuples, want_task=use_task)
        if use_task and isinstance(res, FusionTask):
            res = prod.run_fusion_task(res)
        out.append(res)
    return out


def test_async_task_matches_inline_send_rows():
    """GOLDEN EQUIVALENCE: prepare-task + run_fusion_task must produce byte-identical redirect rows
    to the inline path, over a multi-step sequence that exercises cross-step LSH matching (so the
    index state, not just one step's output, is compared). This is the regression guard for the
    whole async move: if it ever diverges, the worker is not doing the same computation."""
    shared = _make_k(1, seed=5)[0]
    other = _make_k(1, seed=77)[0]
    # reqA registers a rep; reqB/reqC match it in later steps; reqD is dissimilar (no redirect).
    steps = [
        [("reqA", _cache_with_blocks({10: shared}), [10])],
        [("reqB", _cache_with_blocks({20: shared}, seed=9), [20])],
        [("reqC", _cache_with_blocks({28: shared, 29: other}, seed=3), [28, 29])],
        [("reqD", _cache_with_blocks({30: other}, seed=11), [30])],
    ]
    with _async_config():
        inline = _run_fusion_steps(MooncakeFFProducer(), False, steps)
        viatask = _run_fusion_steps(MooncakeFFProducer(), True, steps)
    assert inline == viatask, f"async diverged from inline:\n  inline={inline}\n  task  ={viatask}"
    # Guard against the comparison passing vacuously (both all-empty).
    assert sum(len(v) for d in inline for v in d.values()) >= 2, f"expected redirects, got {inline}"


def test_async_task_is_host_resident_after_prepare():
    """The task handed to the worker must not reference the paged KV cache — those blocks are
    recycled once the step ends, so a retained view would read another request's KV later."""
    prod = MooncakeFFProducer()
    cache = _cache_with_blocks({10: _make_k(1, seed=5)[0]})
    with _async_config() as m:
        prod.reset_step(1)
        prod.on_layer(1, "L0", cache, {"L0", "L1"}, [("reqA", [10], True)], want_task=True)
        task = prod.on_layer(1, "L1", cache, {"L0", "L1"}, [("reqA", [10], True)], want_task=True)
    assert isinstance(task, m.FusionTask), f"expected a FusionTask, got {type(task)}"
    assert task.reps_cpu.device.type == "cpu", task.reps_cpu.device
    assert task.sub_hashes_dev.device.type == "cpu", task.sub_hashes_dev.device
    # Mutating the cache after prepare must not change what the worker sees.
    snapshot = task.reps_cpu.clone()
    cache.zero_()
    assert torch.equal(task.reps_cpu, snapshot), "task tensors alias the paged cache"


def test_can_defer_refuses_tp_group_and_intra_req():
    """The async gate must refuse every path that is not pure host work. tp_group is the critical
    one: the matrix backend all_reduces, and a collective on a side thread deadlocks against the
    model's own collectives."""
    with _async_config() as m:
        assert MooncakeFFProducer._can_defer(None), "production config must be deferrable"
        assert not MooncakeFFProducer._can_defer(object()), "must refuse to defer under TP>1"
        m._PD_INTRA_REQ_FF = True
        assert not MooncakeFFProducer._can_defer(None), "must refuse when within-batch cc is on"
        m._PD_INTRA_REQ_FF = False
        m._PD_CROSS_INDEX = "matrix"
        assert not MooncakeFFProducer._can_defer(None), "must refuse the matrix backend"


def test_on_layer_falls_back_to_inline_when_gate_closed():
    """want_task=True with a closed gate must still return the redirect dict, so the caller's type
    branch degrades to the synchronous path rather than silently dropping fusion. Closed here via
    intra_req_ff (the other gate, TP>1, would need a real process group — the inline path really
    does all_reduce, which is exactly why it can never be deferred)."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    prod = MooncakeFFProducer()
    shared = _make_k(1, seed=5)[0]
    cache = _cache_with_blocks({10: shared, 11: shared}, seed=2)
    prod.reset_step(1)
    reqs = [("reqA", [10], True), ("reqB", [11], True)]
    orig = m._PD_INTRA_REQ_FF
    try:
        m._PD_INTRA_REQ_FF = True
        prod.on_layer(1, "L0", cache, {"L0", "L1"}, reqs, want_task=True)
        out = prod.on_layer(1, "L1", cache, {"L0", "L1"}, reqs, want_task=True)
    finally:
        m._PD_INTRA_REQ_FF = orig
    assert isinstance(out, dict), f"closed gate must return the inline dict, got {type(out)}"
    assert sum(len(v) for v in out.values()) == 1, f"within-batch merge expected, got {out}"


def test_worker_queue_drops_instead_of_blocking():
    """A full queue must shed the task and count it — fusion may lose compression under load, but it
    must never block the prefill forward thread."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    prod = MooncakeFFProducer()
    worker = m._FFFusionWorkerThread(prod, lambda *a: None)   # not started: queue cannot drain
    task = FusionTask(gi=1, reps_cpu=torch.zeros(1, 4), sub_hashes_dev=torch.zeros(1, 2),
                      flat_req_local=[0], flat_slot=[0], ext_ids=["r"], ext_has_remote=[True],
                      n_blocks=1)
    accepted = sum(worker.submit(task) for _ in range(m._FF_WORKER_QUEUE + 5))
    assert accepted == m._FF_WORKER_QUEUE, f"accepted {accepted}, cap {m._FF_WORKER_QUEUE}"
    assert prod.worker_dropped == 5, prod.worker_dropped


def test_forward_work_moves_off_the_forward_thread():
    """The Step 1 deliverable, asserted directly: on the async path the clustering cost must land on
    the WORKER side of the split, not the forward side. Measured structurally rather than by wall
    clock (which is far too noisy at these sizes): after _prepare_fusion_task returns, no probe or
    register work has happened yet; it only happens once run_fusion_task is called."""
    prod = MooncakeFFProducer()
    shared = _make_k(1, seed=5)[0]
    cache = _cache_with_blocks({10: shared, 11: shared}, seed=2)
    reqs = [("reqA", [10], True), ("reqB", [11], True)]
    with _async_config():
        prod.reset_step(1)
        for ln in ("L0", "L1"):
            task = prod.on_layer(1, ln, cache, {"L0", "L1"}, reqs, want_task=True)
        assert isinstance(task, FusionTask)
        # Forward thread is done: it did the device prep and the copies, and nothing else.
        assert prod.probe_ms == 0.0 and prod.register_ms == 0.0, \
            f"probe/register leaked onto the forward thread: {prod.probe_ms}/{prod.register_ms}"
        assert prod.dedup_ms == 0.0, f"clustering ran on the forward thread: {prod.dedup_ms}"
        assert prod.group_completions == 0, "group counted before the worker ran it"
        prod.run_fusion_task(task)
    assert prod.register_ms > 0.0, "register never ran on the worker"
    assert prod.dedup_ms > 0.0, "worker did not account its clustering time"
    assert prod.group_completions == 1


def test_forward_ms_accumulates_per_hook():
    prod = MooncakeFFProducer()
    prod.note_forward(1.5)
    prod.note_forward(2.5)
    assert prod.forward_calls == 2 and abs(prod.forward_ms - 4.0) < 1e-9


def test_stats_dump_fires_on_wall_clock_and_at_exit():
    """A run that ends before its next step-cadence multiple must still emit real cumulative numbers.
    The first async-worker run did not, and every derived metric in it was measured against a step-1
    snapshot with a 17-entry LSH index."""
    import tempfile, os, json, time as _time
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    d = tempfile.mkdtemp()
    path = os.path.join(d, f"bff_stats_{os.getpid()}.json")
    prod = MooncakeFFProducer()
    prod.steps = 7                      # not 1, not a multiple of the step cadence
    prod.blk_total[1] = 123
    prod._last_dump_t = _time.monotonic()
    prod.maybe_dump_stats(d)
    assert not os.path.exists(path), "dumped despite neither cadence being due"
    prod._last_dump_t = _time.monotonic() - (m._PD_STATS_MAX_AGE_S + 1)
    prod.maybe_dump_stats(d)
    assert json.load(open(path))["total_blocks"] == 123, "wall-clock trigger did not dump"
    # And the exit hook writes the final cumulative state regardless of cadence.
    os.remove(path)
    prod.blk_total[1] = 456
    prod._dump_at_exit(d)
    assert json.load(open(path))["total_blocks"] == 456, "atexit dump did not write"


def test_ff_groups_none_selects_no_fusion_groups():
    """BFF_FF_GROUPS=none must yield an EMPTY selection, distinct from unset (=all). That is the
    control arm which keeps the KV-cache group split and every patch active while doing no fusion,
    so a throughput comparison against stock can separate the split's cost from fusion's."""
    from kv_fast_fusion_ascend.connectors.mooncake_layerwise_connector_ff import _parse_groups
    assert _parse_groups(None) is None, "unset means all eligible groups"
    assert _parse_groups("  ") is None
    assert _parse_groups(",,") is None, "a value parsing to nothing must not disable fusion"
    assert _parse_groups("1,2") == {1, 2}
    for raw in ("none", "NONE", " off "):
        got = _parse_groups(raw)
        assert got is not None and len(got) == 0, f"{raw!r} must select no groups, got {got!r}"


def test_stats_dump_is_not_gated_behind_the_fusion_filter():
    """The wall-clock backstop must be reachable on EVERY layer hook, not only when a fusion group
    completes. It was originally called from the tail of _ff_producer_accumulate, behind four early
    returns — under BFF_FF_GROUPS=1 that tail runs on ~1 layer in 28, so a prefill node's ledger
    froze at its step-1 snapshot for an entire run and under-reported its shipped redirects."""
    # Read the source text rather than the class: the connector is Ascend-gated and does not import
    # off-NPU, but this invariant is exactly the kind that regresses unnoticed on a dev box.
    import ast
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    tree = ast.parse(open(m.__file__.replace(".pyc", ".py")).read())
    bodies = {n.name: ast.unparse(n) for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name in ("_ff_producer_accumulate", "_ff_install_worker_hooks")}
    assert set(bodies) == {"_ff_producer_accumulate", "_ff_install_worker_hooks"}, sorted(bodies)
    assert "maybe_dump_stats" not in bodies["_ff_producer_accumulate"], \
        "dump must NOT be called from _ff_producer_accumulate — its early returns (non-fusion group, "
    assert "maybe_dump_stats" in bodies["_ff_install_worker_hooks"], \
        "dump must be driven from the save_kv_layer hook so it runs on every layer"


def test_decode_free_ledger_survives_a_short_run():
    """The decode ledger is the GROUND TRUTH for compression, and its event cadence silently
    truncated it: a con32 run had 38 merge events, so only event 1 was ever written and the file
    reported blocks_freed_total=16 against a real 785 — turning ~1.7x compression into a reported
    1.007x. A run must leave the true cumulative totals on disk however it ends."""
    import tempfile, os, json, time as _time
    import kv_fast_fusion.fast_fusion_scheduler as fs
    d = tempfile.mkdtemp()
    path = os.path.join(d, f"bff_decode_stats_{os.getpid()}.json")
    saved = (fs._PD_STATS_DIR, dict(fs._bff_decode_stats), dict(fs._bff_decode_dump_state))
    try:
        fs._PD_STATS_DIR = d
        fs._bff_decode_stats.update(blocks_freed_total=0, merge_events=0)
        fs._bff_decode_dump_state.update(last_t=_time.monotonic(), atexit=False)
        for _ in range(38):                      # fewer events than the dump cadence (50)
            fs._bff_record_decode_free(20)
        got = json.load(open(path))
        assert got["merge_events"] == 1, "event-1 snapshot expected before any backstop fires"
        # Wall-clock backstop: the next event must publish the true cumulative totals.
        fs._bff_decode_dump_state["last_t"] = _time.monotonic() - (fs._PD_STATS_MAX_AGE_S + 1)
        fs._bff_record_decode_free(20)
        got = json.load(open(path))
        assert got["merge_events"] == 39 and got["blocks_freed_total"] == 780, got
        # ...and the exit hook publishes whatever accumulated after the last dump.
        fs._bff_record_decode_free(5)
        fs._bff_dump_decode_stats()
        got = json.load(open(path))
        assert got["merge_events"] == 40 and got["blocks_freed_total"] == 785, got
    finally:
        fs._PD_STATS_DIR = saved[0]
        fs._bff_decode_stats.clear(); fs._bff_decode_stats.update(saved[1])
        fs._bff_decode_dump_state.clear(); fs._bff_decode_dump_state.update(saved[2])


def test_late_map_after_promotion_is_counted_and_dropped():
    """A redirect map arriving AFTER its owner's promotion must be counted as late and discarded.
    It cannot be applied: once a request has been scheduled the scheduler ships only newly allocated
    blocks, so a later req_to_blocks rewrite never reaches the worker's device table — freeing the
    owner's original block then would be a use-after-free."""
    from kv_fast_fusion_ascend import fast_fusion_ascend_patch as p
    owner_rid = "own-abcd1234"
    src = _FakeSource({})
    sched = _FakeScheduler({}, [_FakeManager({}), _FakeManager({owner_rid: [200]})])
    prev = _bp_mod._FF_PENDING_SOURCE
    p._PROMOTED_SEEN.clear()
    _bp_mod._FF_PENDING_SOURCE = src
    try:
        # Promote with nothing pending → recorded as seen, counted as no_rows.
        owner = _FakeSchedReq(RequestStatus.WAITING)
        owner.request_id = owner_rid
        _bff_promotion_apply(sched, owner)
        assert src.promo_stats["promo_no_rows"] == 1, src.promo_stats
        # The map shows up one step too late.
        src.pending["own"] = {1: [[0, _ext_hash("rep"), 0], [1, _ext_hash("rep"), 1]]}
        p._bff_sweep_late_maps(sched)
        assert src.promo_stats["promo_rows_late"] == 2, src.promo_stats
        assert src.promo_stats["promo_maps_late"] == 1, src.promo_stats
        assert "own" not in src.pending, "late map must be dropped, not left to accumulate"
        # A map for a request that has NOT been promoted must be left alone for its promotion.
        src.pending["other"] = {1: [[0, _ext_hash("rep"), 0]]}
        p._bff_sweep_late_maps(sched)
        assert "other" in src.pending, "un-promoted owner's map must survive the sweep"
        assert src.promo_stats["promo_rows_late"] == 2, src.promo_stats
    finally:
        _bp_mod._FF_PENDING_SOURCE = prev
        p._PROMOTED_SEEN.clear()


def test_worker_thread_ships_redirects_end_to_end():
    """The running worker must finish a real task and ship each owner's rows to that owner's target,
    with the gi it was computed for."""
    from kv_fast_fusion_ascend.connectors import mooncake_layerwise_connector_ff as m
    shipped = []
    prod = MooncakeFFProducer()
    worker = m._FFFusionWorkerThread(prod, lambda *a: shipped.append(a))
    worker.start()
    assert worker.ready.wait(timeout=5), "worker never signalled ready"
    shared = _make_k(1, seed=5)[0]
    with _async_config():
        # Step 1 registers reqA's rep; step 2's reqB matches it and must ship one redirect.
        for si, (ext, bid, seed) in enumerate([("reqA", 10, 0), ("reqB", 20, 9)]):
            cache = _cache_with_blocks({bid: shared}, seed=seed)
            prod.reset_step(si + 1)
            reqs = [(ext, [bid], True)]
            for ln in ("L0", "L1"):
                task = prod.on_layer(1, ln, cache, {"L0", "L1"}, reqs, want_task=True)
            assert isinstance(task, FusionTask)
            task.targets = {ext: ("10.0.0.1", 5000)}
            assert worker.submit(task)
            worker.join_queue()      # deterministic: this task is fully processed before the next
    assert len(shipped) == 1, f"expected exactly one shipped map, got {shipped}"
    host, port, ext_id, gi, rows = shipped[0]
    assert (host, port, ext_id, gi) == ("10.0.0.1", 5000, "reqB", 1), shipped[0]
    assert rows == [(0, _ext_hash("reqA"), 0)], rows


if __name__ == "__main__":
    test_producer_detects_duplicate_across_requests()
    test_producer_no_merge_when_dissimilar()
    test_consumer_resolve_repoints_and_reports()
    test_consumer_resolve_unresolved_when_rep_absent()
    test_end_to_end_producer_to_consumer()
    test_lsh_cross_request_across_steps_without_encoded_batch()
    test_intra_req_ff_disables_within_batch_cc()
    test_lsh_probe_matches_bruteforce_oracle()
    test_lsh_probe_bins_accepted_cosines()
    test_lsh_evict_compacts_and_still_probes()
    test_redirect_channel_push_pull_contract()
    test_ff_groups_parsing()
    test_ff_groups_filters_fusion_set()
    test_write_returns_false_for_absent_rid()
    test_write_returns_true_and_repoints_resident_rid()
    test_free_is_coupled_to_write_success()
    test_classify_owner_miss_never_snapshotted()
    test_classify_owner_miss_pruned()
    test_promotion_apply_rewrites_and_feeds_merge()
    test_promotion_apply_skips_still_loading_rep()
    test_promotion_apply_classifies_rep_gone()
    test_promotion_apply_mixed_split_sums_to_unresolved()
    test_skip_transfer_bandwidth_accounting()
    test_promotion_repgone_revivability_buckets()
    test_promotion_apply_no_rows_is_noop()
    test_promotion_apply_skips_preempted_resume()
    test_promotion_apply_unpublished_source_is_noop()
    test_async_task_matches_inline_send_rows()
    test_async_task_is_host_resident_after_prepare()
    test_can_defer_refuses_tp_group_and_intra_req()
    test_on_layer_falls_back_to_inline_when_gate_closed()
    test_worker_queue_drops_instead_of_blocking()
    test_worker_thread_ships_redirects_end_to_end()
    test_late_map_after_promotion_is_counted_and_dropped()
    test_forward_work_moves_off_the_forward_thread()
    test_forward_ms_accumulates_per_hook()
    test_stats_dump_fires_on_wall_clock_and_at_exit()
    test_repgone_nohist_split_distinguishes_causes()
    test_decode_free_ledger_survives_a_short_run()
    test_ff_groups_none_selects_no_fusion_groups()
    test_stats_dump_is_not_gated_behind_the_fusion_filter()
    test_save_kv_positions_resolve_by_name_once()
    print("OK: all mooncake layerwise FF glue tests passed")


# --------------------------------------------------------------------------------------------
# BFF_FF_GROUPS: restrict fusion to the groups that actually pay.
# Measured at con512: groups 1-3 = 90.9% of redirects but 23% of the LSH index; groups 4-6 =
# 9.1% of redirects but 77% of the index. Excluding a group must remove it from the producer's
# fusion set entirely (no clustering/hash/probe/register) while leaving the rest untouched.
# --------------------------------------------------------------------------------------------

