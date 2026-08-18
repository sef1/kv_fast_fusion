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
