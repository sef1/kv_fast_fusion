"""Unit tests for the BFF group layout on the Ascend NON-layerwise (pull) connector.

The vendored connector is single-group to the bone: one flat block list is applied to every
registered base address. BFF gives each of its seven KV-cache groups its own block table, so that
flat list transfers the wrong physical blocks for six of them. It does not raise — it serves
plausible garbage, and every counter still reads as healthy.

That is the whole reason this file exists. The mapping and alignment logic is kept as module-level
pure functions precisely so it can be pinned here, on CPU, with no NPU, no Transfer Engine and no
vllm_ascend import.

CPU only.
"""

import pytest

from kv_fast_fusion_ascend.connectors import mooncake_connector_ff as mc


class _Group:
    def __init__(self, layer_names):
        self.layer_names = list(layer_names)


def _groups(n_groups, layers_per_group, prefix="layer"):
    """n_groups x layers_per_group named layers, in the BFF split's shape."""
    out, k = [], 0
    for _ in range(n_groups):
        names = []
        for _ in range(layers_per_group):
            names.append(f"{prefix}.{k}")
            k += 1
        out.append(_Group(names))
    return out


# =====================================================================================
# layer -> group
# =====================================================================================
def test_every_layer_of_every_group_is_mapped():
    groups = _groups(7, 4)
    m = mc.build_layer_group_map(groups)

    assert len(m) == 28
    assert m["layer.0"] == 0 and m["layer.3"] == 0
    assert m["layer.4"] == 1
    assert m["layer.27"] == 6


def test_the_attn_suffix_mismatch_is_tolerated_in_both_directions():
    """The runner's KV-cache config and the connector's layer names disagree about a trailing
    '.attn' depending on the model. Neither spelling may be treated as an unknown layer."""
    m = {"model.layers.0.attn": 3}
    assert mc.group_of(m, "model.layers.0.attn") == 3
    assert mc.group_of(m, "model.layers.0") == 3

    m2 = {"model.layers.0": 5}
    assert mc.group_of(m2, "model.layers.0.attn") == 5


def test_an_unknown_layer_is_none_not_zero():
    """Defaulting to group 0 is the failure this whole file guards: it transfers the warmup group's
    blocks in place of a fusion group's, silently."""
    assert mc.group_of({"a": 1}, "b") is None


# =====================================================================================
# base address -> groups
# =====================================================================================
def test_one_allocation_per_layer_gives_one_group_per_address():
    """The Ascend layout: register_kv_caches appends K and V per layer, in kv_caches order."""
    groups = _groups(7, 4)
    layer_names = [ln for g in groups for ln in g.layer_names]
    base_addrs = [1000 + 16 * j for j in range(len(layer_names) * 2)]   # 56 distinct addresses

    out = mc.build_base_addr_groups(base_addrs, layer_names, mc.build_layer_group_map(groups), 2, 7)

    assert len(out) == 56
    assert all(len(gs) == 1 for gs in out), "distinct allocations cannot share groups"
    assert out[0] == [0] and out[1] == [0], "K and V of layer 0 are both group 0"
    assert out[8] == [1], "layer 4 (address 8 = 4*2) is group 1"
    assert {g for gs in out for g in gs} == set(range(7))


def test_an_allocation_shared_by_several_groups_carries_all_of_them():
    """The GPU layout, which the first GPU connector got wrong: vLLM's uniform-page-size packing
    puts one layer from EVERY group into a single tensor. Tagging one group per address there
    corrupted six of seven groups on the decode. Keying on the address value is what makes both
    layouts come out right."""
    groups = _groups(3, 1)                      # 3 groups, 1 layer each
    layer_names = [ln for g in groups for ln in g.layer_names]
    shared_k, shared_v = 4096, 8192
    base_addrs = [shared_k, shared_v] * 3       # every layer reports the SAME K and V allocation

    out = mc.build_base_addr_groups(base_addrs, layer_names, mc.build_layer_group_map(groups), 2, 3)

    assert out == [[0, 1, 2]] * 6, "each shared allocation must receive all three groups' blocks"


def test_a_group_no_address_can_reach_is_refused():
    """The failure that must never be a warning: a group nothing carries is a group whose KV is
    never transferred, and no counter anywhere would show it."""
    groups = _groups(3, 1)
    layer_names = [ln for g in groups for ln in g.layer_names]
    layer_group = mc.build_layer_group_map(groups)
    layer_group["layer.2"] = 1                  # group 2 now owns no layer

    with pytest.raises(mc.KVGroupLayoutError, match="not reachable"):
        mc.build_base_addr_groups([1, 2, 3, 4, 5, 6], layer_names, layer_group, 2, 3)


def test_a_layer_in_no_group_is_refused_rather_than_guessed():
    groups = _groups(2, 1)
    layer_group = mc.build_layer_group_map(groups)

    with pytest.raises(mc.KVGroupLayoutError, match="belongs to no KV-cache group"):
        mc.build_base_addr_groups([1, 2, 3, 4], ["layer.0", "stranger"], layer_group, 2, 2)


def test_an_unexpected_address_count_is_refused():
    """If the base class ever changes its registration order, every index below it is wrong. Fail
    at startup rather than transferring against a shifted map."""
    groups = _groups(2, 1)
    with pytest.raises(mc.KVGroupLayoutError, match="base addresses"):
        mc.build_base_addr_groups([1, 2, 3], ["layer.0", "layer.1"],
                                  mc.build_layer_group_map(groups), 2, 2)


def test_the_single_group_case_reduces_to_stock():
    """With one group this must be indistinguishable from the flat behaviour it replaces."""
    groups = _groups(1, 4)
    layer_names = groups[0].layer_names
    out = mc.build_base_addr_groups(list(range(8)), layer_names,
                                    mc.build_layer_group_map(groups), 2, 1)
    assert out == [[0]] * 8


# =====================================================================================
# per-group tail alignment
# =====================================================================================
def test_each_group_is_tail_aligned_independently():
    """A prefix-cache hit shortens D's list from the FRONT, so P's last len(local) ids are the ones
    that correspond. Groups need not shorten by the same amount, which is why stock's single
    alignment cannot be reused."""
    local = [[70, 71], [80, 81, 82]]
    remote = [[10, 11, 12, 13], [20, 21, 22]]

    out = mc.align_per_group(local, remote)

    assert out[0] == ([12, 13], [70, 71]), "group 0 keeps P's LAST two"
    assert out[1] == ([20, 21, 22], [80, 81, 82]), "group 1 was already aligned"


def test_alignment_never_moves_a_surviving_pair():
    """Stated as the invariant rather than an example: whatever the trim, position i on D must
    still be fed by the block P intended for it."""
    remote = [100, 101, 102, 103, 104]
    for keep in range(1, 6):
        local = list(range(900, 900 + keep))
        (r, ll), = mc.align_per_group([local], [remote])
        assert len(r) == len(ll) == keep
        # The kept remote ids are the tail, in order, paired positionally.
        assert r == remote[-keep:]
        assert list(zip(ll, r)) == list(zip(local, remote[-keep:]))


def test_a_group_with_nothing_to_pull_keeps_its_slot():
    """Dropping it would renumber every later group, and the transfer indexes by group."""
    out = mc.align_per_group([[], [80]], [[10, 11], [20]])
    assert out == [([], []), ([20], [80])]


def test_a_decode_asking_for_more_than_the_prefill_sent_is_refused():
    """Silently truncating here would pair D's blocks with the wrong source."""
    with pytest.raises(mc.KVGroupLayoutError, match="only offered"):
        mc.align_per_group([[1, 2, 3]], [[10, 11]])


def test_a_group_the_prefill_never_sent_is_refused_not_skipped():
    """Asymmetric on purpose, and the asymmetry is the safety property.

    P offering MORE groups than D wants is fine — D had a prefix hit and allocated nothing there.
    D wanting a group P never sent is not: those layers would decode against whatever was already
    in the block, which is the silent-corruption class this connector exists to avoid. Skipping the
    group (the tempting 'be liberal' reading) would produce exactly that."""
    with pytest.raises(mc.KVGroupLayoutError, match="only offered"):
        mc.align_per_group([[70], [80]], [[10]])


def test_a_prefill_group_the_decode_does_not_want_is_simply_unused():
    """The safe direction: D allocated nothing for group 1, so there is nothing to pull into."""
    out = mc.align_per_group([[70], []], [[10], [20, 21]])
    assert out == [([10], [70]), ([], [])]


def test_flatten_counts_every_group():
    assert mc.flatten_group_lists([[1, 2], [], [3]]) == [1, 2, 3]
    assert mc.flatten_group_lists([[], []]) == []


# =====================================================================================
# transfer-engine registration dedup
# =====================================================================================
def test_the_shared_tensor_layout_registers_each_region_once():
    """The startup crash this guards: `Transfer Engine does not support overlapped memory region`.

    BFF's split makes vLLM size the pool by max_layers_per_group (4), so a 28-layer / 7-group model
    gets 4 allocations, each shared_by one layer per group. The NPU allocator honours shared_by, so
    those 7 layers all report the SAME data_ptr — and the vendored PULL connector appends one entry
    per (layer, K/V) unconditionally. 56 pointers, 8 distinct, and Mooncake refuses the second
    registration of a region it already holds. The LAYERWISE connector guards the same append with
    `if data_ptr() not in ptrs`, which is exactly why it survives the split and the pull one does
    not."""
    # 4 shared tensors, split into K and V => 8 distinct regions, each reported by 7 layers.
    distinct = [(0x1000 + 0x100 * i, 0x2000 + 0x100 * i) for i in range(4)]
    ptrs, sizes = [], []
    for _group in range(7):
        for k_addr, v_addr in distinct:
            ptrs += [k_addr, v_addr]
            sizes += [4096, 4096]
    assert len(ptrs) == 56 and len(set(ptrs)) == 8

    kept_ptrs, kept_sizes = mc.dedup_registration_regions(ptrs, sizes)

    assert len(kept_ptrs) == 8
    assert set(kept_ptrs) == set(ptrs), "dedup must drop repeats, never whole regions"
    assert len(kept_sizes) == len(kept_ptrs), "sizes stay index-aligned with ptrs"
    assert kept_ptrs == sorted(set(ptrs), key=ptrs.index), "first occurrence wins, order preserved"


def test_the_ascend_layout_is_left_completely_alone():
    """One allocation per layer — the layout the connector was written against. Dedup must be a
    no-op there, or it would silently unregister real regions."""
    ptrs = [1000 + 16 * j for j in range(56)]
    sizes = [4096] * 56

    assert mc.dedup_registration_regions(ptrs, sizes) == (ptrs, sizes)


def test_dedup_keeps_the_size_that_belongs_to_the_kept_pointer():
    """Sizes are consumed positionally by register_memory(ptr, size). Pairing a kept pointer with a
    dropped duplicate's size would register the wrong extent."""
    kept_ptrs, kept_sizes = mc.dedup_registration_regions(
        [0xA, 0xB, 0xA, 0xC], [10, 20, 999, 30])

    assert kept_ptrs == [0xA, 0xB, 0xC]
    assert kept_sizes == [10, 20, 30]


# =====================================================================================
# transfer amplification
# =====================================================================================
def _shared_layout(n_groups=7, slots=4):
    """Both engines under BFF: `slots` allocations, each K/V, each reported by every group."""
    addrs = []
    for _group in range(n_groups):
        for slot in range(slots):
            addrs += [0x1000 + 0x100 * slot, 0x2000 + 0x100 * slot]
    return addrs


def test_the_shared_layout_collapses_to_the_distinct_regions():
    """The 7x amplification: 56 address pairs, 8 distinct. Left alone, _transfer_kv_cache emits
    every segment 7 (duplicate addresses) x 7 (the union of groups each address maps to) = 49 times
    instead of 7."""
    local = _shared_layout()
    remote = _shared_layout()

    keep = mc.transfer_indices(local, remote)

    assert len(local) == 56
    assert len(keep) == 8
    assert keep == sorted(keep), "indices stay ascending so k still selects the right block_len"
    assert len({(local[k], remote[k]) for k in keep}) == 8
    # Every distinct pair in the input survives somewhere in the kept set.
    assert {(local[k], remote[k]) for k in keep} == set(zip(local, remote))


def test_the_per_layer_layout_is_untouched():
    """The Ascend norm: 28 layers x K/V, all distinct. Nothing may be dropped."""
    local = [1000 + 16 * j for j in range(56)]
    remote = [9000 + 16 * j for j in range(56)]

    assert mc.transfer_indices(local, remote) == list(range(56))


def test_a_shared_decode_against_a_per_layer_prefill_drops_nothing():
    """Why the key is the PAIR and not the local address.

    If P ever runs a per-layer layout while D runs the shared one, D's region is paired with seven
    DIFFERENT remote regions. Keying on the local address alone would keep one and discard six —
    six sevenths of the model's KV, silently. Pair-keying degrades to a no-op instead."""
    local = _shared_layout()                       # 8 distinct, each repeated 7x
    remote = [9000 + 16 * j for j in range(56)]    # 56 distinct

    keep = mc.transfer_indices(local, remote)

    assert keep == list(range(56)), "every distinct remote region must still be pulled"
    assert len({local[k] for k in keep}) == 8, "the local side really is the degenerate one"


def test_indices_are_preserved_not_renumbered():
    """block_len is chosen by `k % len(self.block_len)` — the K/V alternation for MLA. Renumbering
    the survivors 0..n would pick the wrong cache's block length for every odd entry."""
    local = [0xA, 0xA, 0xB, 0xB]
    remote = [0x1, 0x2, 0x1, 0x2]

    # All four pairs are distinct despite only two distinct local addresses.
    assert mc.transfer_indices(local, remote) == [0, 1, 2, 3]

    # And when pairs do repeat, the ORIGINAL index of the first occurrence is what survives.
    assert mc.transfer_indices([0xA, 0xB, 0xA, 0xB], [0x1, 0x2, 0x1, 0x2]) == [0, 1]


# =====================================================================================
# phase B: producer row stash (worker -> scheduler handoff)
# =====================================================================================
def test_rows_are_collected_per_group_and_consumed_once():
    """The stash exists because rows are PRODUCED in save_kv_layer (worker) and must LEAVE in
    request_finished (scheduler). Consume-once matters: a retried request_finished must not ship
    the same redirect map twice."""
    stash = mc.FFRowStash()
    stash.add("ext-a", 1, [(0, 111, 3)])
    stash.add("ext-a", 2, [(1, 222, 4), (2, 333, 5)])
    stash.add("ext-b", 1, [(0, 444, 6)])

    got = stash.take("ext-a")
    assert got == {1: [[0, 111, 3]], 2: [[1, 222, 4], [2, 333, 5]]}
    assert stash.take("ext-a") is None, "consume-once"
    assert stash.take("ext-b") == {1: [[0, 444, 6]]}


def test_empty_rows_never_create_an_entry():
    """A group that completed with nothing worth redirecting must not produce an empty map — that
    would ship `ff_redirects={gi: []}` and make the consumer count a fusion that never happened."""
    stash = mc.FFRowStash()
    stash.add("ext-a", 1, [])
    assert stash.take("ext-a") is None


def test_the_stash_is_bounded_and_evicts_oldest():
    """Requests aborted before request_finished never collect their rows. Unbounded, that leaks for
    the life of the process."""
    stash = mc.FFRowStash(cap=3)
    for i in range(5):
        stash.add(f"ext-{i}", 1, [(0, i, 0)])

    assert stash.take("ext-0") is None and stash.take("ext-1") is None
    assert stash.take("ext-4") == {1: [[0, 4, 0]]}
    assert stash.dropped == 2


def test_rows_are_normalized_to_plain_ints():
    """They are about to be JSON-encoded into kv_transfer_params; numpy ints would not survive."""
    stash = mc.FFRowStash()
    stash.add("e", 1, [(True, 2, 3)])
    (row,) = stash.take("e")[1]
    assert row == [1, 2, 3]
    assert all(type(v) is int for v in row)


# =====================================================================================
# phase B: the redirect field's JSON round trip
# =====================================================================================
def test_group_indices_survive_being_stringified_by_json():
    """kv_transfer_params crosses the proxy as JSON, which turns dict keys into strings. Indexing
    the result by int would silently find nothing and fusion would appear to do nothing at all."""
    out = mc.normalize_ff_redirects({"3": [[0, 99, 1]], "5": [[2, 88, 3]]})
    assert out == {3: [[0, 99, 1]], 5: [[2, 88, 3]]}
    assert all(type(k) is int for k in out)


def test_a_missing_or_empty_redirect_field_is_simply_no_fusion():
    """Must stay non-fatal, unlike the transfer path: a dropped redirect costs compression, never
    correctness."""
    for raw in (None, {}, [], "nonsense", {"1": []}):
        assert mc.normalize_ff_redirects(raw) is None


def test_malformed_rows_are_dropped_not_guessed():
    out = mc.normalize_ff_redirects({"1": [[0, 1, 2], [9, 9], [3, 4, 5]], "x": [[0, 1, 2]]})
    assert out == {1: [[0, 1, 2], [3, 4, 5]]}, "short row and non-int group both dropped"


# =====================================================================================
# phase B: consumer sink contract with the promotion hook
# =====================================================================================
def test_the_pending_source_matches_what_the_promotion_hook_expects():
    """_bff_promotion_apply consumes `.lock` + `.pending` as {ext_id: {gi: rows}} and pops per
    request. Pinning the shape here because the hook lives in another file and would fail silently
    (promo_no_rows) rather than raise if this drifted."""
    src = mc.FFPendingSource()
    src.offer("ext-a", {"1": [[0, 5, 1]]})

    assert hasattr(src, "lock") and hasattr(src, "pending")
    assert src.pending == {"ext-a": {1: [[0, 5, 1]]}}, "keys coerced to int"

    with src.lock:
        popped = src.pending.pop("ext-a", None)
    assert popped == {1: [[0, 5, 1]]}


def test_offering_nothing_leaves_the_hook_with_nothing_to_do():
    src = mc.FFPendingSource()
    src.offer("ext-a", {})
    assert src.pending == {}


def test_a_second_group_merges_instead_of_replacing_the_first():
    """The producer emits one map per fusion GROUP, and the promotion hook pops the whole
    {gi: rows} dict at once. An offer that overwrote would silently drop every group but the last —
    losing most of the compression with no error anywhere."""
    src = mc.FFPendingSource()
    src.offer("ext-a", {1: [[0, 5, 1]]})
    src.offer("ext-a", {2: [[1, 6, 2]]})

    assert src.pending["ext-a"] == {1: [[0, 5, 1]], 2: [[1, 6, 2]]}


def test_the_redirect_field_is_consumed_from_the_params_dict():
    """update_state_after_alloc runs on EVERY allocation for a request, not just the first. A
    non-destructive read re-offers rows the promotion hook already applied; the sweep then drops
    them as 'arrived after promotion'. That is what made one run discard 853 rows while applying
    371 blocks — one late map per apply, exactly 1:1.

    This pins the `pop` semantics at the level the connector relies on: a second read of the same
    params dict must find nothing."""
    params = {"ff_redirects": {"1": [[0, 5, 1]]}, "do_remote_prefill": True}

    first = mc.normalize_ff_redirects(params.pop("ff_redirects", None))
    second = mc.normalize_ff_redirects(params.pop("ff_redirects", None))

    assert first == {1: [[0, 5, 1]]}
    assert second is None, "a repeated allocation must not resurrect consumed rows"
    assert "do_remote_prefill" in params, "only the fusion field is consumed"


def test_the_pending_sink_is_bounded():
    """A request aborted between allocation and promotion is removed by neither the promotion hook
    nor the late sweep, so its rows would otherwise be held for the life of the process."""
    src = mc.FFPendingSource(cap=3)
    for i in range(5):
        src.offer(f"ext-{i}", {1: [[0, i, 0]]})

    assert len(src.pending) == 3
    assert "ext-0" not in src.pending and "ext-4" in src.pending
    assert src.dropped == 2


def test_promo_stats_keys_exist_for_the_hook_to_increment():
    """The hook does `stats["promo_no_rows"] += 1` without a guard, so a missing key is a KeyError
    inside scheduling."""
    src = mc.FFPendingSource()
    for key in ("promo_applied", "promo_unresolved", "promo_no_rows", "promo_merge_calls"):
        assert key in src.promo_stats


# =====================================================================================
# phase B: group selection
# =====================================================================================
def test_ff_groups_parsing_matches_the_layerwise_spelling():
    """One A/B knob must mean the same thing on both transports."""
    assert mc._parse_ff_groups("1,2,3") == {1, 2, 3}
    assert mc._parse_ff_groups(" 2 , 4 ") == {2, 4}
    assert mc._parse_ff_groups(None) is None
    assert mc._parse_ff_groups("") is None
    assert mc._parse_ff_groups("   ") is None


# =====================================================================================
# where the group layout is read from
# =====================================================================================
class _Cfg:
    def __init__(self, groups):
        self.kv_cache_groups = groups


class _Runner:
    def __init__(self, groups):
        self.kv_cache_config = _Cfg(groups)


def test_the_layout_is_readable_without_any_active_runner():
    """The `no active BFF runner` failure, in its real shape.

    In a process where the connector module is what first imports kv_fast_fusion — the factory
    loads it lazily during ensure_kv_transfer_initialized — NPUModelRunner was already constructed,
    so no __init__ patch can publish _ACTIVE_RUNNER. The kv_cache_config the factory passes is
    available at exactly that moment, and is the same post-split layout."""
    groups = _groups(7, 4)

    assert mc.resolve_kv_cache_groups(_Cfg(groups), None) is groups


def test_the_runner_still_serves_as_the_fallback():
    """Kept so a connector constructed on the old (2-arg) path is not regressed."""
    groups = _groups(7, 4)

    assert mc.resolve_kv_cache_groups(None, _Runner(groups)) is groups


def test_the_config_wins_over_the_runner():
    """Both are the same object in practice; if they ever diverge, the one the factory handed this
    connector for THIS role is the authoritative one."""
    from_cfg, from_runner = _groups(7, 4), _groups(3, 2)

    assert mc.resolve_kv_cache_groups(_Cfg(from_cfg), _Runner(from_runner)) is from_cfg


def test_an_empty_config_falls_through_rather_than_being_believed():
    """A KVCacheConfig with no groups is not a layout, it is a missing layout. Believing it would
    map every address to nothing and refuse later with a confusing message."""
    groups = _groups(7, 4)

    assert mc.resolve_kv_cache_groups(_Cfg([]), _Runner(groups)) is groups


def test_no_layout_anywhere_is_still_fatal():
    """The point of the whole file: never guess. A default group list here transfers real KV against
    the wrong block table, and nothing downstream can detect it."""
    with pytest.raises(mc.KVGroupLayoutError, match="unreadable"):
        mc.resolve_kv_cache_groups(None, None)

    with pytest.raises(mc.KVGroupLayoutError, match="unreadable"):
        mc.resolve_kv_cache_groups(_Cfg([]), _Runner([]))


# =====================================================================================
# descriptor_coverage — every planned block must actually get written
# =====================================================================================
# v2's transfer verification found blocks on the decode holding content that matches no row of their
# request in any group — ~0.19% of transferred blocks, one per affected request, clustered under
# allocation churn. A block that receives no descriptor is never written and still holds its previous
# tenant's KV, which looks exactly like that. The existing per-request check only asserts each GROUP
# emitted something, so one missing block inside a group that emitted others passes it today.
def _grouped(*per_group):
    """`grouped` as _transfer_kv_cache sees it: per group, (remote runs, local runs)."""
    return [([list(r) for r in remote], [list(l) for l in local]) for remote, local in per_group]


def test_a_normal_emission_covers_every_planned_block():
    grouped = _grouped(([[10, 11], [20]], [[70, 71], [80]]),
                       ([[30]], [[90]]))
    keep, addr_groups = [0, 1], {0: [0, 1], 1: [0, 1]}

    covered, n = mc.descriptor_coverage(grouped, keep, addr_groups)

    assert covered == mc.planned_blocks(grouped)
    assert n == 2 * 3, "two addresses x three runs"


def test_a_group_beyond_the_aligned_list_is_reported_as_a_gap():
    """`if gi >= len(grouped): continue` skips a whole group silently. It cannot happen today, which
    is exactly why nothing would notice if it started to."""
    grouped = _grouped(([[10]], [[70]]))
    covered, _n = mc.descriptor_coverage(grouped, [0], {0: [0, 5]})

    assert covered == {(0, 70)}, "group 5 does not exist and must not invent coverage"


def test_uneven_run_lists_lose_blocks_and_the_audit_sees_it():
    """`zip(grouped_remote, grouped_local)` truncates to the shorter side. group_concurrent_contiguous
    returns equal lengths, so this is a guard on a future change, not a live bug."""
    grouped = [([[10]], [[70], [71]])]          # one remote run, two local runs

    covered, _n = mc.descriptor_coverage(grouped, [0], {0: [0]})
    missing = mc.planned_blocks(grouped) - covered

    assert missing == {(0, 71)}, "the truncated run is never written"


def test_a_block_missing_from_the_emission_is_named():
    grouped = _grouped(([[10, 11]], [[70, 71]]))
    covered, _n = mc.descriptor_coverage(grouped, [0], {0: []})   # no group emitted

    assert mc.planned_blocks(grouped) - covered == {(0, 70), (0, 71)}


def test_duplicate_addresses_do_not_inflate_coverage_or_hide_gaps():
    """`keep` already drops repeated (local, remote) pairs; coverage is a SET, so re-emitting the
    same block through several addresses must not read as covering a different one."""
    grouped = _grouped(([[10], [11]], [[70], [71]]))

    one, n1 = mc.descriptor_coverage(grouped, [0], {0: [0]})
    both, n2 = mc.descriptor_coverage(grouped, [0, 1], {0: [0], 1: [0]})

    assert one == both == {(0, 70), (0, 71)}
    assert n2 == 2 * n1, "the segment COUNT still reflects the duplication"


def test_coverage_is_empty_for_a_request_with_nothing_to_pull():
    assert mc.descriptor_coverage([], [], {}) == (set(), 0)
    assert mc.planned_blocks([]) == set()


# =====================================================================================
# chunk_segments — bounding the batch must not itself lose a write
# =====================================================================================
# Verification found ~0.19% of transferred blocks never written while this connector's descriptor
# list audited COMPLETE, so the write was lost inside batch_transfer_sync_read. Splitting into
# bounded calls tests whether segment count is the cause. An off-by-one here would silently drop the
# tail — indistinguishable from the bug being chased, which is why this is pinned.
def _flat(chunks):
    src = [x for c in chunks for x in c[0]]
    dst = [x for c in chunks for x in c[1]]
    ln = [x for c in chunks for x in c[2]]
    return src, dst, ln


def test_chunking_is_a_strict_partition():
    src, dst, ln = list(range(10)), list(range(100, 110)), list(range(200, 210))

    chunks = mc.chunk_segments(src, dst, ln, 3)

    assert [len(c[0]) for c in chunks] == [3, 3, 3, 1], "no segment lost, none duplicated"
    assert _flat(chunks) == (src, dst, ln), "order preserved and every segment present exactly once"


def test_the_three_lists_are_always_cut_at_the_same_boundaries():
    """src[i], dst[i] and lengths[i] describe ONE segment. Cutting them differently would pair a
    source address with another segment's destination — writing the right bytes to the wrong block."""
    src, dst, ln = list(range(7)), list(range(100, 107)), list(range(200, 207))

    for c_src, c_dst, c_ln in mc.chunk_segments(src, dst, ln, 2):
        assert len(c_src) == len(c_dst) == len(c_ln)
        for a, b, c in zip(c_src, c_dst, c_ln):
            assert b == a + 100 and c == a + 200, "a segment's three fields stayed together"


def test_an_exact_multiple_produces_no_empty_trailing_chunk():
    """An empty final batch would be handed to the engine as a zero-segment transfer."""
    chunks = mc.chunk_segments(list(range(6)), list(range(6)), list(range(6)), 3)

    assert len(chunks) == 2
    assert all(c[0] for c in chunks)


def test_unlimited_is_the_default_and_means_one_call():
    """0 must reproduce today's behaviour exactly, or turning the knob off changes the transport."""
    src, dst, ln = list(range(5)), list(range(5)), list(range(5))

    for cap in (0, -1, None):
        assert mc.chunk_segments(src, dst, ln, cap) == [(src, dst, ln)]


def test_a_list_shorter_than_the_cap_is_one_chunk():
    src = [1, 2]
    assert mc.chunk_segments(src, src, src, 100) == [(src, src, src)]


def test_an_empty_transfer_issues_no_call_at_all():
    """Zero segments must not become one empty batch_transfer_sync_read."""
    assert mc.chunk_segments([], [], [], 0) == []
    assert mc.chunk_segments([], [], [], 8) == []


def test_a_cap_of_one_still_covers_everything():
    src = [1, 2, 3]
    chunks = mc.chunk_segments(src, src, src, 1)

    assert len(chunks) == 3
    assert _flat(chunks)[0] == src


# =====================================================================================
# block_segment — replaying one block must target what the transfer targeted
# =====================================================================================
# _transfer_kv_cache coalesces consecutive blocks into ONE segment addressed by the run's first id.
# Replaying a single block re-derives its address; if that arithmetic disagrees with the run's, the
# replay answers a question nobody asked — and the replay is the test that decides whether a
# ~0.2% never-written block is a lost write (upstream) or a wrong remote address (ours).
def test_the_first_block_of_a_run_reproduces_the_runs_own_addresses():
    """The run's segment uses local_id[0] and remote_id[0]. j=0 must land on exactly those."""
    base_local, base_remote, block_len = 0x1000, 0x9000, 128
    local_ids, remote_ids = [70, 71, 72], [10, 11, 12]

    run_src = base_local + local_ids[0] * block_len + 0 * block_len
    run_dst = base_remote + remote_ids[0] * block_len

    src, dst, ln = mc.block_segment(base_local, base_remote, local_ids[0], remote_ids[0],
                                    block_len, block_len)

    assert (src, dst) == (run_src, run_dst)
    assert ln == block_len, "one block's worth, not the whole run's"


def test_later_blocks_stride_by_exactly_one_block():
    base_local, base_remote, block_len = 0x1000, 0x9000, 128
    first = mc.block_segment(base_local, base_remote, 70, 10, block_len, block_len)
    third = mc.block_segment(base_local, base_remote, 72, 12, block_len, block_len)

    assert third[0] - first[0] == 2 * block_len
    assert third[1] - first[1] == 2 * block_len


def test_the_run_length_equals_the_sum_of_its_blocks():
    """The segment carries `inner_block_len * len(run)`; the per-block replays must tile it exactly,
    or a replay would write a different number of bytes than the transfer did."""
    block_len, ids = 128, [70, 71, 72, 73]
    per_block = [mc.block_segment(0x1000, 0x9000, lb, rb, block_len, block_len)
                 for lb, rb in zip(ids, [10, 11, 12, 13])]

    assert sum(d[2] for d in per_block) == block_len * len(ids)
    assert [d[0] for d in per_block] == [0x1000 + i * block_len for i in ids]


def test_local_and_remote_use_their_own_id_spaces():
    """src is the LOCAL destination, dst the REMOTE source. Deriving one from the other would send
    the decode's block ids to the producer as addresses."""
    src, dst, _ln = mc.block_segment(0x1000, 0x9000, 700, 10, 128, 128)

    assert src == 0x1000 + 700 * 128
    assert dst == 0x9000 + 10 * 128


def test_the_inner_offset_applies_only_to_the_local_side():
    """Mirrors the emission: src adds inner_offset * inner_block_len, dst does not. At tp=1 the
    offset is 0 and the two coincide, which is why a mistake here would stay hidden."""
    src0, dst0, _ = mc.block_segment(0x1000, 0x9000, 70, 10, 256, 128, inner_offset=0)
    src1, dst1, _ = mc.block_segment(0x1000, 0x9000, 70, 10, 256, 128, inner_offset=1)

    assert src1 - src0 == 128
    assert dst1 == dst0


def test_the_async_transfer_path_is_off_by_default():
    """It changes how every KV byte is moved. A knob that alters the transport must not be on until
    someone asks — the sync path is what nine runs of evidence were collected against."""
    import inspect

    src = inspect.getsource(mc)

    assert 'os.environ.get("BFF_XFER_ASYNC", "0") == "1"' in src
    fn = src[src.index("        def _one_batch(self, session_id, src, dst, lengths):"):]
    fn = fn[:fn.index("        def _issue_transfer(")]
    assert fn.index("if not self._XFER_ASYNC:") < fn.index("batch_transfer_async_read"), \
        "the sync call stays the default branch"
    assert "batch_transfer_sync_read" in fn, "and is still reachable unchanged"


def test_the_async_poll_is_bounded():
    """It runs on the recv thread, which every queued request is waiting behind. An unbounded poll
    would turn a lost completion into a hung decode."""
    import inspect

    src = inspect.getsource(mc)
    fn = src[src.index("        def _one_batch(self, session_id, src, dst, lengths):"):]
    fn = fn[:fn.index("        def _issue_transfer(")]

    assert "deadline = time.perf_counter() + self._XFER_ASYNC_TIMEOUT" in fn
    assert "if time.perf_counter() >= deadline:" in fn, \
        "the deadline must actually be compared, not merely computed"
    assert "while True:" in fn and "return 0" in fn, "it exits, it does not spin forever"


def test_a_negative_handle_is_returned_rather_than_polled():
    """Polling a handle the engine refused to create would block for the whole timeout on every
    failed submit."""
    import inspect

    src = inspect.getsource(mc)
    fn = src[src.index("        def _one_batch(self, session_id, src, dst, lengths):"):]
    fn = fn[:fn.index("        def _issue_transfer(")]

    assert fn.index("if handle < 0:") < fn.index("deadline =")
