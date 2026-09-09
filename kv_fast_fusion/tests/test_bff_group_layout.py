"""bff_group_layout (Update 17): the pure BFF storage-group partition.

Concurrency ∝ 1/(max_layers_per_group × num_groups); the pool is sized by the fattest group. A
BALANCED partition (BFF_BALANCED_GROUPS>=2) holds that product at its floor (== total layers ==
baseline concurrency) while minimizing group count, so the per-step per-group block-table tax falls
without a concurrency cliff. N=0 must reproduce the legacy warmup(4)+uniform-chunk layout exactly.

Pure list arithmetic — no vllm, no NPU.
"""
from kv_fast_fusion.fast_fusion_core import bff_group_layout


def _layers(n):
    return [f"l{i}" for i in range(n)]


def _flat(layout):
    out = []
    for names, _ in layout:
        out.extend(names)
    return out


def test_balanced_two_groups_are_even_distinct_and_cover_all_layers():
    layout = bff_group_layout(_layers(28), balanced_groups=2, group_size=4)
    sizes = [len(names) for names, _ in layout]
    spec_ids = [sid for _, sid in layout]
    assert sizes == [14, 14], "N=2 on 28 layers must be perfectly balanced"
    assert spec_ids == [0, 1], "adjacent groups must carry distinct spec ids"
    # Partition: every layer exactly once, order preserved.
    assert _flat(layout) == _layers(28)


def test_balanced_sizes_differ_by_at_most_one_when_uneven():
    layout = bff_group_layout(_layers(29), balanced_groups=2, group_size=4)
    sizes = sorted(len(names) for names, _ in layout)
    assert sizes == [14, 15], "uneven split must differ by <= 1 (minimizes max_layers)"
    assert _flat(layout) == _layers(29)


def test_balanced_holds_the_concurrency_product_at_its_floor():
    # max_layers_per_group * num_groups == total layers for a balanced divisible split — the floor
    # that equals baseline (single-group) concurrency, unlike an unbalanced coarse split.
    layout = bff_group_layout(_layers(28), balanced_groups=2, group_size=4)
    max_layers = max(len(names) for names, _ in layout)
    assert max_layers * len(layout) == 28


def test_balanced_n3_keeps_three_groups_with_at_least_two_distinct_specs():
    layout = bff_group_layout(_layers(28), balanced_groups=3, group_size=4)
    assert len(layout) == 3, "same-spec non-adjacent groups are NOT merged"
    assert len({sid for _, sid in layout}) >= 2, "verify_and_split needs >1 distinct spec value"
    # adjacent groups differ
    ids = [sid for _, sid in layout]
    assert all(ids[i] != ids[i + 1] for i in range(len(ids) - 1))
    assert _flat(layout) == _layers(28)


def test_default_layout_reproduces_warmup_plus_uniform_chunks():
    layout = bff_group_layout(_layers(28), balanced_groups=0, group_size=4)
    names = [n for n, _ in layout]
    spec_ids = [sid for _, sid in layout]
    # warmup group first: first 2 + last 2 layers, on the alt spec (id 1).
    assert names[0] == ["l0", "l1", "l26", "l27"]
    assert spec_ids[0] == 1
    # then the middle 24 layers chunked by 4 → 6 fusion groups on per_layer_spec (id 0).
    assert len(layout) == 7
    assert spec_ids[1:] == [0, 0, 0, 0, 0, 0]
    assert all(len(n) == 4 for n in names[1:])
    # union of warmup + fusion == all layers (warmup is non-contiguous, so not _flat order).
    assert sorted(_flat(layout)) == sorted(_layers(28))


def test_default_layout_respects_group_size():
    layout = bff_group_layout(_layers(28), balanced_groups=0, group_size=24)
    # 24 middle layers, chunk 24 → 1 fusion group; warmup + 1 fusion = 2 groups.
    assert len(layout) == 2
    assert [sid for _, sid in layout] == [1, 0]
