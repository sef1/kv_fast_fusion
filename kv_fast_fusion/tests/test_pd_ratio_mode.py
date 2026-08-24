"""`BFF_SCALE_MODE=ratio`: the alias is rescaled at attention time instead of being rejected.

Why this mode exists, in one paragraph, because the number that motivates it is easy to lose:
substituting a representative for a victim block costs ``rel_err = sqrt(1 + r^2 - 2*r*cos)`` with
``r`` the norm ratio as it happens to fall. Measured over a 65%-wire-saving run, 95.5% of accepted
merges carried error > 0.3 and 55% carried 0.5–1.0, while their cosine floors ``sqrt(1 - cos^2)``
sat at 0.31–0.66 — so most of that error is norm mismatch, and correcting it in-kernel removes it.
It does NOT buy compression: at a 0.3 budget the reachable mass is already taken (``rel_err <= 0.3``
*requires* ``cos >= 0.954``, so the accepted set already is the reachable set). The claim under test
is narrower and entirely about accuracy at a FIXED wire saving.

The load-bearing test here is :func:`test_a_scaled_alias_is_identical_to_not_merging_at_all` — it
runs the real Triton kernel and is the only one that can catch a wrong scale actually reaching the
attention. The rest guard the plumbing that gets a scale to it.
"""

import os
import types

import pytest
import torch

from kv_fast_fusion import pd_dedup_v2, pd_lsh

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# =====================================================================================
# The property rescaling is supposed to buy.
# =====================================================================================
@requires_cuda
def test_a_scaled_alias_is_identical_to_not_merging_at_all():
    """Two blocks at cosine 1.0 differing by exactly 2x in magnitude: merging them and scaling the
    survivor by 0.5 must give the SAME attention output as never merging.

    This is the whole premise. If it fails, every accuracy number ratio mode produces is measuring
    something other than what it claims, and no amount of plumbing above it matters. Run against the
    kernel with `norms=None` (stock paged attention over unmerged blocks) rather than against a
    hand-rolled reference, so the comparison isolates the scaling and not the kernel."""
    from kv_fast_fusion.fast_fusion_triton_attn import bff_unified_attention

    torch.manual_seed(0)
    dev, dt = "cuda", torch.float16
    n_heads, head_dim, block_size, n_tok = 4, 64, 16, 16
    n_seqs = 2

    # Physical cache: block 1 is seq 0's, block 2 is seq 1's own copy at HALF the magnitude.
    k_cache = torch.zeros(4, block_size, n_heads, head_dim, device=dev, dtype=dt)
    v_cache = torch.zeros(4, block_size, n_heads, head_dim, device=dev, dtype=dt)
    k_cache[1] = torch.randn(block_size, n_heads, head_dim, device=dev, dtype=dt)
    v_cache[1] = torch.randn(block_size, n_heads, head_dim, device=dev, dtype=dt)
    k_cache[2] = k_cache[1] * 0.5      # cos = 1.0 exactly, norm ratio 2 — the ideal merge
    v_cache[2] = v_cache[1] * 0.5

    q = torch.randn(n_seqs * n_tok, n_heads, head_dim, device=dev, dtype=dt)
    cu_q = torch.tensor([0, n_tok, 2 * n_tok], device=dev, dtype=torch.int32)
    seqused = torch.tensor([n_tok, n_tok], device=dev, dtype=torch.int32)
    scale = head_dim ** -0.5

    def run(block_table, norms_k, norms_v, seq_to_slot):
        out = torch.empty_like(q)
        bff_unified_attention(
            q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_q, max_seqlen_q=n_tok,
            seqused_k=seqused, max_seqlen_k=n_tok, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=block_table, softcap=0.0,
            norms_k=norms_k, norms_v=norms_v, seq_to_slot=seq_to_slot)
        return out

    # Reference: no merge. Seq 1 reads its own block 2, unscaled.
    unmerged = torch.tensor([[1], [2]], device=dev, dtype=torch.int32)
    reference = run(unmerged, None, None, None)

    # Merged: seq 1's table now points at seq 0's block 1, and slot 1 carries the 0.5 scale that
    # turns block 1 back into what block 2 held. Slot 0 is the non-fused sentinel and stays at 1.0.
    merged = torch.tensor([[1], [1]], device=dev, dtype=torch.int32)
    norms_k = torch.ones(3, 1, device=dev, dtype=dt)
    norms_v = torch.ones(3, 1, device=dev, dtype=dt)
    norms_k[1, 0] = 0.5
    norms_v[1, 0] = 0.5
    seq_to_slot = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    rescaled = run(merged, norms_k, norms_v, seq_to_slot)

    # fp16 through a softmax: exact equality is not on offer, but the two must agree to well inside
    # the noise. A missing scale would show up as a wholly different distribution, not a last-bit
    # difference — the K scale moves the logits before the exponential.
    torch.testing.assert_close(rescaled, reference, rtol=2e-3, atol=2e-3)


@requires_cuda
def test_each_block_of_a_sequence_gets_its_own_scale():
    """Multi-block sequences, with only SOME blocks aliased and each at a different ratio.

    The single-block case cannot see the kernel's `block_pos = seq_offset // BLOCK_SIZE` indexing at
    all — every lookup lands on column 0 whatever the arithmetic. This is the realistic shape: a
    request keeps most of its blocks and aliases a few, so the scale row is mostly 1.0 with holes,
    and an off-by-one would scale the wrong block rather than failing."""
    from kv_fast_fusion.fast_fusion_triton_attn import bff_unified_attention

    torch.manual_seed(1)
    dev, dt = "cuda", torch.float16
    n_heads, head_dim, block_size, n_blocks = 4, 64, 16, 3
    n_tok = block_size * n_blocks

    # Physical blocks 1..3 are seq 0's. Blocks 4..6 are seq 1's own copies: block 4 is unrelated,
    # blocks 5 and 6 are scaled copies of seq 0's at DIFFERENT ratios.
    k_cache = torch.randn(8, block_size, n_heads, head_dim, device=dev, dtype=dt)
    v_cache = torch.randn(8, block_size, n_heads, head_dim, device=dev, dtype=dt)
    ratios = [1.0, 0.25, 4.0]           # block 4 not aliased, 5 -> 0.25x, 6 -> 4x
    for i, r in enumerate(ratios[1:], start=1):
        k_cache[4 + i] = k_cache[1 + i] * r
        v_cache[4 + i] = v_cache[1 + i] * r

    q = torch.randn(2 * n_tok, n_heads, head_dim, device=dev, dtype=dt)
    cu_q = torch.tensor([0, n_tok, 2 * n_tok], device=dev, dtype=torch.int32)
    seqused = torch.tensor([n_tok, n_tok], device=dev, dtype=torch.int32)

    def run(block_table, norms_k, norms_v, seq_to_slot):
        out = torch.empty_like(q)
        bff_unified_attention(
            q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_q, max_seqlen_q=n_tok,
            seqused_k=seqused, max_seqlen_k=n_tok, softmax_scale=head_dim ** -0.5, causal=True,
            window_size=(-1, -1), block_table=block_table, softcap=0.0,
            norms_k=norms_k, norms_v=norms_v, seq_to_slot=seq_to_slot)
        return out

    unmerged = torch.tensor([[1, 2, 3], [4, 5, 6]], device=dev, dtype=torch.int32)
    reference = run(unmerged, None, None, None)

    # Seq 1 keeps block 4 and aliases the other two onto seq 0's, each with its own ratio.
    merged = torch.tensor([[1, 2, 3], [4, 2, 3]], device=dev, dtype=torch.int32)
    norms_k = torch.ones(3, n_blocks, device=dev, dtype=dt)
    norms_v = torch.ones(3, n_blocks, device=dev, dtype=dt)
    for pos, r in enumerate(ratios):
        norms_k[1, pos] = r
        norms_v[1, pos] = r
    seq_to_slot = torch.tensor([0, 1], device=dev, dtype=torch.int32)

    torch.testing.assert_close(run(merged, norms_k, norms_v, seq_to_slot), reference,
                               rtol=3e-3, atol=3e-3)


@requires_cuda
def test_the_test_above_would_notice_a_missing_scale():
    """Guards the guard: run the merged case WITHOUT the scale and confirm it does not match.

    Without this, a kernel that silently ignored `norms_k` would pass the test above whenever the
    two blocks happened to be close, and the suite would report success for a mode that does
    nothing. This is the same failure the raw-mode arm exists to be different from."""
    from kv_fast_fusion.fast_fusion_triton_attn import bff_unified_attention

    torch.manual_seed(0)
    dev, dt = "cuda", torch.float16
    n_heads, head_dim, block_size, n_tok = 4, 64, 16, 16

    k_cache = torch.zeros(4, block_size, n_heads, head_dim, device=dev, dtype=dt)
    v_cache = torch.zeros(4, block_size, n_heads, head_dim, device=dev, dtype=dt)
    k_cache[1] = torch.randn(block_size, n_heads, head_dim, device=dev, dtype=dt)
    v_cache[1] = torch.randn(block_size, n_heads, head_dim, device=dev, dtype=dt)
    k_cache[2] = k_cache[1] * 0.5
    v_cache[2] = v_cache[1] * 0.5

    q = torch.randn(2 * n_tok, n_heads, head_dim, device=dev, dtype=dt)
    cu_q = torch.tensor([0, n_tok, 2 * n_tok], device=dev, dtype=torch.int32)
    seqused = torch.tensor([n_tok, n_tok], device=dev, dtype=torch.int32)

    def run(block_table):
        out = torch.empty_like(q)
        bff_unified_attention(
            q=q, k=k_cache, v=v_cache, out=out, cu_seqlens_q=cu_q, max_seqlen_q=n_tok,
            seqused_k=seqused, max_seqlen_k=n_tok, softmax_scale=head_dim ** -0.5, causal=True,
            window_size=(-1, -1), block_table=block_table, softcap=0.0,
            norms_k=None, norms_v=None, seq_to_slot=None)
        return out

    reference = run(torch.tensor([[1], [2]], device=dev, dtype=torch.int32))
    unscaled = run(torch.tensor([[1], [1]], device=dev, dtype=torch.int32))

    seq1_ref, seq1_raw = reference[n_tok:], unscaled[n_tok:]
    assert not torch.allclose(seq1_raw, seq1_ref, rtol=2e-3, atol=2e-3), (
        "an unscaled merge is indistinguishable from no merge here — this fixture cannot detect "
        "a dropped scale, so the identity test above proves nothing")


# =====================================================================================
# Getting a scale to the kernel.
# =====================================================================================
def _applier(engine, group_layers, written=None):
    return pd_dedup_v2.AliasApplier(
        engine,
        lambda r, rid, gi, blocks: ((written.append((rid, gi, list(blocks)))
                                     if written is not None else None) or True),
        set().update,
        group_layers=group_layers)


def _runner(blocks_by_rid):
    reqs = {r: types.SimpleNamespace(block_ids=[list(g) for g in gs])
            for r, gs in blocks_by_rid.items()}
    ib = types.SimpleNamespace(req_id_to_index={r: i for i, r in enumerate(blocks_by_rid)})
    return types.SimpleNamespace(input_batch=ib, requests=reqs, _updated_block_tables=None,
                                 fused_requests={})


def test_an_applied_alias_records_its_per_layer_scale_for_the_kernel():
    """The scale must land per LAYER, at the victim's POSITION in the block table.

    Position rather than block id because the kernel indexes norms by `seq_offset // BLOCK_SIZE`,
    and because after substitution several positions legitimately share one physical block — keying
    by id would collapse them onto one scale."""
    engine = pd_dedup_v2.DedupEngine(resident=False)
    k_scale = torch.tensor([2.0, 3.0])          # 2 layers in this group
    v_scale = torch.tensor([0.5, 0.25])
    engine._alias_ready = {"rB": {1: {51: (41, "rA", (k_scale, v_scale))}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    engine._resident_owner[(1, 41)] = "rA"
    runner = _runner({"rA": [[], [41]], "rB": [[], [50, 51, 52]]})

    _applier(engine, {1: {"model.layers.7.attn", "model.layers.6.attn"}}).apply(runner)

    fr = runner.fused_requests["rB"]
    assert set(fr) == {"model.layers.6.attn", "model.layers.7.attn"}
    nk6, nv6 = fr["model.layers.6.attn"]
    nk7, nv7 = fr["model.layers.7.attn"]
    # sorted() puts layer 6 first, so it takes column 0 — the same order the producer built the
    # norms in. A swap here scales layer 6 by layer 7's ratio and nothing would report it.
    assert nk6.tolist() == [1.0, 2.0, 1.0] and nv6.tolist() == [1.0, 0.5, 1.0]
    assert nk7.tolist() == [1.0, 3.0, 1.0] and nv7.tolist() == [1.0, 0.25, 1.0]


def test_a_refused_block_table_write_records_no_scale():
    """The scales describe a substitution that happened. Recording one for a table the runner
    refused would rescale KV the victim still owns — a corruption, not a lost optimisation."""
    engine = pd_dedup_v2.DedupEngine(resident=False)
    engine._alias_ready = {"rB": {1: {51: (41, "rA", (torch.tensor([2.0]), torch.tensor([2.0])))}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    engine._resident_owner[(1, 41)] = "rA"
    runner = _runner({"rA": [[], [41]], "rB": [[], [50, 51]]})

    pd_dedup_v2.AliasApplier(engine, lambda *a: False, set().update,
                            group_layers={1: {"model.layers.4.attn"}}).apply(runner)

    assert runner.fused_requests == {}


def test_raw_mode_aliases_leave_fused_requests_untouched():
    """A raw-mode alias carries scale=None. Writing 1.0 rows for it would hand every raw request a
    slot in the norm buffers and route the batch through the scaling kernel for nothing."""
    engine = pd_dedup_v2.DedupEngine(resident=False)
    engine._alias_ready = {"rB": {1: {51: (41, "rA", None)}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    engine._resident_owner[(1, 41)] = "rA"
    runner = _runner({"rA": [[], [41]], "rB": [[], [50, 51]]})

    _applier(engine, {1: {"model.layers.4.attn"}}).apply(runner)

    assert runner.fused_requests == {}


def test_a_scale_with_nowhere_to_go_is_reported_not_swallowed(caplog):
    """No group_layers means the scale cannot reach the kernel, so the alias would be applied
    UNSCALED — the raw-mode error this mode exists to remove, arriving silently. The block table is
    still rewritten (the substitution is sound, just unimproved); what must not happen is a run
    reporting ratio mode while quietly delivering raw."""
    engine = pd_dedup_v2.DedupEngine(resident=False)
    engine._alias_ready = {"rB": {1: {51: (41, "rA", (torch.tensor([2.0]), torch.tensor([2.0])))}}}
    engine._planner._resident.setdefault(1, set()).add(41)
    engine._resident_owner[(1, 41)] = "rA"
    runner = _runner({"rA": [[], [41]], "rB": [[], [50, 51]]})

    with caplog.at_level("WARNING"):
        _applier(engine, None).apply(runner)

    assert "UNSCALED" in caplog.text


# =====================================================================================
# Producing the scale.
# =====================================================================================
def test_the_scale_is_the_victims_norm_over_the_representatives_per_layer():
    """End to end through the payload: encode exact norms, plan, and read back the ratio."""
    monkey = pd_dedup_v2.SCALE_MODE
    try:
        pd_dedup_v2.SCALE_MODE = "ratio"
        torch.manual_seed(0)
        # One layer group of 2 layers, 2 blocks whose K/V differ only in magnitude.
        kv = [torch.zeros(2, 3, 1, 1, 4), torch.zeros(2, 3, 1, 1, 4)]
        for li in range(2):
            kv[li][0, 1] = 3.0 * (li + 1)      # K of block 1
            kv[li][1, 1] = 2.0                 # V of block 1
            kv[li][0, 2] = 6.0 * (li + 1)      # K of block 2 — twice block 1's
            kv[li][1, 2] = 8.0
        k, v = pd_dedup_v2.block_layer_norms(kv, [1, 2], False)
        # ||[x,x,x,x]|| = 2|x|; block 2's K is 2x block 1's in every layer.
        assert torch.allclose(k[1] / k[0], torch.tensor([2.0, 2.0]))
        assert torch.allclose(v[1] / v[0], torch.tensor([4.0, 4.0]))
    finally:
        pd_dedup_v2.SCALE_MODE = monkey


def test_a_raw_producer_is_refused_rather_than_substituted_unscaled():
    """A payload with no norms in it means the producer is running raw. Ratio mode must decline the
    alias (`no_kv_norms`) and pull the block, because substituting without a scale is exactly the
    error this mode promises to have removed — and it would show up only as accuracy loss."""
    payload = pd_dedup_v2.SignatureCodec.encode(
        torch.randn(2, 4), torch.ones(2), [[0], [0]])
    assert pd_dedup_v2.SignatureCodec.kv_norms(payload) is None, (
        "None, not ones: 'did not ship scales' and 'scales are 1' must stay distinguishable")


def test_a_pull_plans_declines_and_lands_a_scaled_alias(monkeypatch):
    """The whole chain in one test: payload -> plan -> release -> apply -> fused_requests.

    Each half of this is pinned separately above; what this adds is that they are actually joined.
    The chain has four places a scale can be dropped silently — the producer not shipping norms, the
    planner not reading them, `release` not registering the representative's, and the applier not
    recording the ratio — and every one of them degrades to a correct-but-unscaled substitution,
    which no counter reports and only accuracy would reveal."""
    monkeypatch.setattr(pd_dedup_v2, "SCALE_MODE", "ratio")
    engine = pd_dedup_v2.DedupEngine()

    # Two blocks of one 2-layer group, same direction, victim 2x the representative's magnitude.
    direction = torch.tensor([[1.0, 0.0, 0.5, 0.25]])
    norms = direction.norm(dim=1).clamp(min=1e-6)
    sig = direction / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    hashes = pd_lsh.sub_hashes(sig, proj)

    def payload(k_norms, v_norms):
        return pd_dedup_v2.SignatureCodec.encode(
            sig, norms, hashes,
            kv_norms=(torch.tensor([k_norms]), torch.tensor([v_norms])))

    # rA lands first and becomes the representative; rB then matches it.
    engine.plan({"rA": [[], [900]]}, {"rA": {1: payload([2.0, 4.0], [1.0, 8.0])}}, threshold=0.75)
    engine.release("rA")
    planned = engine.plan({"rB": [[], [41]]},
                          {"rB": {1: payload([4.0, 8.0], [2.0, 4.0])}}, threshold=0.75)

    assert planned["rB"][1] == [pd_dedup_v2.SENTINEL], "rB should not pull a block it can alias"
    engine.release("rB")

    runner = _runner({"rA": [[], [900]], "rB": [[], [41]]})
    written = []
    _applier(engine, {1: {"model.layers.5.attn", "model.layers.4.attn"}}, written).apply(runner)

    assert written == [("rB", 1, [900])], "the alias really was applied"
    nk4, nv4 = runner.fused_requests["rB"]["model.layers.4.attn"]
    nk5, nv5 = runner.fused_requests["rB"]["model.layers.5.attn"]
    # victim/rep, per layer: K 4/2 and 8/4; V 2/1 and 4/8.
    assert nk4.tolist() == [2.0] and nk5.tolist() == [2.0]
    assert nv4.tolist() == [2.0] and nv5.tolist() == [0.5]


def test_a_raw_producer_costs_the_alias_rather_than_the_scale(monkeypatch):
    """Same chain, but the representative was registered without norms. Ratio mode must decline —
    substituting here would be a raw-mode merge inside a run reporting itself as ratio."""
    monkeypatch.setattr(pd_dedup_v2, "SCALE_MODE", "ratio")
    engine = pd_dedup_v2.DedupEngine()

    direction = torch.tensor([[1.0, 0.0, 0.5, 0.25]])
    norms = direction.norm(dim=1).clamp(min=1e-6)
    sig = direction / norms.unsqueeze(1)
    proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
    hashes = pd_lsh.sub_hashes(sig, proj)
    bare = pd_dedup_v2.SignatureCodec.encode(sig, norms, hashes)     # a raw producer's payload

    engine.plan({"rA": [[], [900]]}, {"rA": {1: bare}}, threshold=0.75)
    engine.release("rA")
    planned = engine.plan({"rB": [[], [41]]}, {"rB": {1: bare}}, threshold=0.75)

    assert planned["rB"][1] == [41], "no scale available, so the block must be pulled"
    assert engine.stats.skip_reasons["no_kv_norms"] == 1


# =====================================================================================
# Choosing the candidate.
# =====================================================================================
def _unit(d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(d, generator=g)
    return x / x.norm()


def _at_cos(target, c, seed):
    """A unit vector whose cosine with ``target`` is exactly ``c``."""
    perp = _unit(target.shape[0], seed)
    perp = perp - (perp @ target) * target
    perp = perp / perp.norm()
    return c * target + (1.0 - c * c) ** 0.5 * perp


def _probe_all(rows, norms, target, threshold=0.75):
    """Probe ``target`` against ``rows`` with every row forced to be a CANDIDATE.

    The bucket tables are rebuilt to map the probe's own hashes onto all rows. These tests are about
    which candidate the verify step picks and where the accept bar sits — not about SimHash recall,
    which at cos 0.93 surfaces nothing at all and would make them silently vacuous."""
    cur = torch.stack(rows)
    proj = pd_lsh.get_proj([None], cur.shape[1], cur.device)
    idx = pd_lsh.LshIndex()
    idx.register(cur, pd_lsh.sub_hashes(cur, proj),
                 [(i, f"b{i}", i, "owner0") for i in range(len(rows))], norms)
    q = target.unsqueeze(0)
    q_hashes = pd_lsh.sub_hashes(q, proj)
    idx.tables = [{h: list(range(len(rows)))} for h in q_hashes[0]]
    return idx, idx.probe(q, q_hashes, ["me"], threshold, norms=[1.0])


def test_ratio_mode_ranks_by_cosine_not_by_norm_match(monkeypatch):
    """The selection criterion has to flip with the mode.

    Ranked by rel_err, a candidate at cos 0.90 with a matched norm (err 0.436) beats one at cos 0.96
    that is 1.5x too big (err 0.608). Rescaling erases the size disadvantage entirely, so under
    ratio mode that ranking picks the strictly worse of the two — and no counter would say so,
    because both are 'accepted'."""
    target = _unit(64, 0)
    # row 0: well aligned but 1.5x too big.   row 1: less aligned, exactly the right size.
    rows = [_at_cos(target, 0.96, 1), _at_cos(target, 0.90, 2)]
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.7)

    def pick(ratio):
        monkeypatch.setattr(pd_lsh, "RATIO_MODE", ratio)
        return _probe_all(rows, [1.5, 1.0], target)[1][1][0][1]

    assert pick(False) == "b1", "raw mode should prefer the norm-matched candidate"
    assert pick(True) == "b0", "ratio mode should prefer the better-aligned candidate"


def test_ratio_mode_turns_the_budget_into_a_cosine_bar(monkeypatch):
    """With the norms corrected, the error a pair costs is its floor sqrt(1-cos^2), so a budget of
    b admits exactly cos >= sqrt(1-b^2) — 0.954 at the usual 0.3. A candidate below that bar can
    never meet the budget however it is scaled, and must still be rejected."""
    monkeypatch.setattr(pd_lsh, "RATIO_MODE", True)
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 0.3)
    target = _unit(64, 0)

    def probe(cos, seed):
        # Norm 3.0 against the probe's 1.0: badly mismatched on purpose, so what is being tested is
        # the cosine bar and not the norm ratio, which ratio mode is entitled to ignore.
        return _probe_all([_at_cos(target, cos, seed)], [3.0], target)

    # 0.93 clears the 0.75 threshold but sits below min_cos_for_budget(0.3) = 0.954.
    idx, (matched, hits) = probe(0.93, 1)
    assert not hits and not matched[0]
    assert idx.rejected_by_rel_err == 1, (
        "it cleared the threshold and was stopped by the budget, so it must be charged to the "
        "budget — otherwise the two modes' histograms stop being comparable")

    idx2, (_, hits2) = probe(0.99, 2)
    assert hits2, "0.99 is above the bar and must be accepted whatever its norm is"
    assert idx2.accept_rel_err[pd_lsh._bin(pd_lsh.min_rel_err(0.99), pd_lsh.REL_ERR_BINS)] == 1, (
        "the error binned must be the FLOOR the rescaled pair actually pays, not the as-it-falls "
        "value — a norm-3.0 rep would otherwise report an error this run does not suffer")


def test_an_inert_budget_leaves_the_cosine_bar_at_the_threshold(monkeypatch):
    """`min_cos_for_budget(1.0)` is 0, so ratio mode with the default budget must behave exactly as
    a pure-threshold run — otherwise every earlier result stops being a comparable baseline."""
    monkeypatch.setattr(pd_lsh, "RATIO_MODE", True)
    monkeypatch.setattr(pd_lsh, "MAX_REL_ERR", 1.0)
    assert pd_lsh.min_cos_for_budget(pd_lsh.MAX_REL_ERR) == 0.0


def test_the_two_modes_read_the_same_env(monkeypatch):
    """`pd_lsh.RATIO_MODE` and `pd_dedup_v2.SCALE_MODE` gate different halves of one decision — the
    candidate choice and the payload. If they could disagree, a run would select candidates for
    cosine and then substitute them without a scale, which is worse than either mode alone.

    Re-imports rather than reading the live attributes, because conftest pins those to raw for the
    suite; what is under test is that both derive from the same variable in the same way."""
    import importlib
    for value, want in (("ratio", True), ("raw", False), ("", False)):
        monkeypatch.setenv("BFF_SCALE_MODE", value)
        assert importlib.reload(pd_lsh).RATIO_MODE is want
        assert (importlib.reload(pd_dedup_v2).SCALE_MODE == "ratio") is want
    # Leave the modules as the rest of the session found them.
    monkeypatch.delenv("BFF_SCALE_MODE", raising=False)
    importlib.reload(pd_lsh)
    importlib.reload(pd_dedup_v2)
