"""The hybrid geometry, on a box with no NPU.

The failure this module exists to prevent is silent: a wrong ``scale`` reads the wrong tensor rows,
transfers the wrong bytes, and leaves every counter looking healthy. So the tests here are mostly
about *refusing* — that two derivations disagree, that a mamba group was handed a scale, that an
allocation is unaligned — rather than about the happy path, which is one line of arithmetic.

The numbers are Qwen3-Next's: 4 groups (1 attention + 3 mamba) of 12 layers over 402 pool blocks,
with the attention tensor at 4824 rows (scale 12).
"""

import pytest

from kv_fast_fusion_ascend.hybrid_layout import (
    GroupGeometry,
    HybridLayout,
    HybridLayoutError,
    RegroupedHybridError,
    block_len_table,
    build_hybrid_layout,
    hybrid_registration_regions,
    mamba_state_index,
    model_declares_mamba,
    narrow_mamba_groups,
    scales_from_tensors,
)

POOL_BLOCKS = 402
SCALE = 12


# -- doubles ---------------------------------------------------------------------------------
class _Tensor:
    def __init__(self, rows):
        self.shape = (rows,)


class _AttnSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class _MambaSpec:
    """Duck-typed the way the real one is detected: it carries per-state shapes and dtypes."""

    def __init__(self, block_size):
        self.block_size = block_size
        self.shapes = ((4, 128), (8, 128))
        self.dtypes = ("bf16", "bf16")


class _Group:
    def __init__(self, spec, layer_names):
        self.kv_cache_spec = spec
        self.layer_names = layer_names


class _Config:
    def __init__(self, groups, num_blocks=POOL_BLOCKS, tensors=()):
        self.kv_cache_groups = groups
        self.num_blocks = num_blocks
        self.kv_cache_tensors = list(tensors)


class _BlockTable:
    def __init__(self, scale):
        self.blocks_per_phys_block = scale


class _Runner:
    def __init__(self, scales):
        self.input_batch = type("_IB", (), {"block_table": [_BlockTable(s) for s in scales]})()


def _qwen3_next(attn_rows=POOL_BLOCKS * SCALE, mamba_rows=POOL_BLOCKS):
    """1 attention group + 3 mamba groups, 12 layers each, as vLLM splits Qwen3-Next."""
    groups = [_Group(_AttnSpec(SCALE * 128), [f"attn.{i}" for i in range(12)])]
    for g in range(3):
        groups.append(_Group(_MambaSpec(SCALE * 128), [f"mamba.{g}.{i}" for i in range(12)]))
    kv_caches = {}
    for ln in groups[0].layer_names:
        kv_caches[ln] = (_Tensor(attn_rows), _Tensor(attn_rows))  # (k_cache, v_cache)
    for group in groups[1:]:
        for ln in group.layer_names:
            kv_caches[ln] = [_Tensor(mamba_rows), _Tensor(mamba_rows)]  # [conv, ssm]
    return _Config(groups), kv_caches


# -- derivation ------------------------------------------------------------------------------
def test_qwen3_next_geometry_from_tensors():
    cfg, kv_caches = _qwen3_next()
    layout = build_hybrid_layout(cfg, kv_caches=kv_caches)

    assert layout.is_hybrid and layout.is_split
    assert [g.scale for g in layout.groups] == [SCALE, 1, 1, 1]
    assert [g.is_mamba for g in layout.groups] == [False, True, True, True]
    assert layout.tensor_rows(0) == POOL_BLOCKS * SCALE
    assert layout.tensor_rows(1) == POOL_BLOCKS


def test_block_table_alone_is_enough_when_tensors_are_not_allocated_yet():
    """``register_kv_caches`` runs after the tables exist but the connector may be built before
    the caches are handed over."""
    cfg, _ = _qwen3_next()
    layout = build_hybrid_layout(cfg, runner=_Runner([SCALE, 1, 1, 1]))
    assert [g.scale for g in layout.groups] == [SCALE, 1, 1, 1]


def test_tensors_and_block_table_disagreeing_is_fatal():
    cfg, kv_caches = _qwen3_next()
    with pytest.raises(HybridLayoutError, match="indexed differently"):
        build_hybrid_layout(cfg, runner=_Runner([SCALE // 2, 1, 1, 1]), kv_caches=kv_caches)


def test_a_scaled_mamba_group_is_fatal():
    """Subdividing a recurrent state block aliases one request's state onto another's."""
    cfg, kv_caches = _qwen3_next(mamba_rows=POOL_BLOCKS * 2)
    with pytest.raises(HybridLayoutError, match="never be subdivided"):
        build_hybrid_layout(cfg, kv_caches=kv_caches)


def test_rows_not_a_multiple_of_the_pool_is_fatal():
    cfg, kv_caches = _qwen3_next(attn_rows=POOL_BLOCKS * SCALE + 1)
    with pytest.raises(HybridLayoutError, match="not a whole multiple"):
        build_hybrid_layout(cfg, kv_caches=kv_caches)


def test_non_hybrid_model_is_scale_one_everywhere():
    """The Qwen2.5 shape every earlier BFF run used — this module must be a no-op there."""
    groups = [_Group(_AttnSpec(128), [f"attn.{i}" for i in range(28)])]
    kv_caches = {ln: (_Tensor(POOL_BLOCKS), _Tensor(POOL_BLOCKS)) for ln in groups[0].layer_names}
    layout = build_hybrid_layout(_Config(groups), kv_caches=kv_caches)
    assert not layout.is_hybrid and not layout.is_split
    assert layout.scale(0) == 1


def test_a_hybrid_model_regrouped_as_attention_is_fatal_not_inert():
    """The failure that costs a run: BFF's rebuild leaves attention-only groups, `is_hybrid` is
    False, and every hybrid path in this package quietly does not run. Both witnesses must trip."""
    groups = [_Group(_AttnSpec(SCALE * 128), [f"attn.{i}" for i in range(12)])]
    kv = {ln: (_Tensor(POOL_BLOCKS * SCALE), _Tensor(POOL_BLOCKS * SCALE))
          for ln in groups[0].layer_names}
    cfg = _Config(groups)

    # witness 1: the model config says it is hybrid
    with pytest.raises(RegroupedHybridError, match="none of the 1 KV-cache groups"):
        build_hybrid_layout(cfg, kv_caches=kv, expect_mamba=True)

    # witness 2: the runner's own naming of recurrent layers, with no config to ask
    kv["model.layers.3.linear_attn"] = [_Tensor(POOL_BLOCKS), _Tensor(POOL_BLOCKS)]
    with pytest.raises(RegroupedHybridError):
        build_hybrid_layout(cfg, kv_caches=kv)


def test_a_non_hybrid_model_is_not_flagged():
    groups = [_Group(_AttnSpec(128), [f"attn.{i}" for i in range(4)])]
    kv = {ln: (_Tensor(POOL_BLOCKS), _Tensor(POOL_BLOCKS)) for ln in groups[0].layer_names}
    assert not build_hybrid_layout(_Config(groups), kv_caches=kv).is_hybrid


def test_regrouped_is_a_subclass_so_a_tolerant_caller_must_opt_out_explicitly():
    """The pull connector tolerates 'groups not readable yet' by catching HybridLayoutError. This
    must NOT be absorbed by that: the groups are readable and they are wrong."""
    assert issubclass(RegroupedHybridError, HybridLayoutError)


def test_model_declares_mamba_reads_the_config_not_the_groups():
    class _Cfg:
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]

    class _Mc:
        hf_text_config = _Cfg()

    class _Vc:
        model_config = _Mc()

    assert model_declares_mamba(_Vc())
    _Cfg.layer_types = ["full_attention"] * 4
    assert not model_declares_mamba(_Vc())


def test_missing_groups_is_fatal_not_defaulted():
    with pytest.raises(HybridLayoutError, match="no kv_cache_groups"):
        build_hybrid_layout(_Config([]))


def test_scales_from_tensors_reads_the_first_cache_not_the_block_axis():
    """An attention layer's KV is a 2-tuple; iterating a bare tensor would walk its blocks."""
    cfg, kv_caches = _qwen3_next()
    assert scales_from_tensors(cfg.kv_cache_groups, kv_caches, POOL_BLOCKS)[0] == SCALE


# -- id-space conversion ---------------------------------------------------------------------
def test_to_kernel_expands_in_order():
    layout = HybridLayout([GroupGeometry(0, False, 4, POOL_BLOCKS, 512)])
    assert layout.to_kernel(0, [0, 2]) == [0, 1, 2, 3, 8, 9, 10, 11]


def test_round_trip_is_identity_on_pool_ids():
    layout = HybridLayout([GroupGeometry(0, False, SCALE, POOL_BLOCKS, SCALE * 128)])
    ids = [0, 7, 401]
    assert layout.to_pool(0, layout.to_kernel(0, ids)) == ids


def test_mamba_groups_pass_through_both_conversions():
    layout = HybridLayout([GroupGeometry(0, True, 1, POOL_BLOCKS, 128)])
    assert layout.to_kernel(0, [5, 9]) == [5, 9]
    assert layout.to_pool(0, [5, 9]) == [5, 9]


def test_to_pool_dedups_and_keeps_order():
    layout = HybridLayout([GroupGeometry(0, False, 4, POOL_BLOCKS, 512)])
    assert layout.to_pool(0, [8, 9, 0, 1, 10]) == [2, 0]


# -- transfer geometry -----------------------------------------------------------------------
def _qwen3_next_layout():
    cfg, kv_caches = _qwen3_next()
    return build_hybrid_layout(cfg, kv_caches=kv_caches), kv_caches


def test_a_mamba_group_transfers_only_the_block_holding_the_state():
    """Pulling all of a mamba group's blocks wastes the wire; pulling the wrong one hands the
    decode a mid-prompt recurrent state — fluent, wrong text and no error anywhere."""
    layout, _ = _qwen3_next_layout()
    local = [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]]
    remote = [[60, 61, 62], [70, 71, 72], [80, 81, 82], [90, 91, 92]]

    got_l, got_r = narrow_mamba_groups(layout, local, remote)

    assert got_l[0] == [10, 11, 12], "the attention group keeps every block"
    assert got_l[1:] == [[22], [32], [42]], "each mamba group keeps only its last block"
    assert got_r[1:] == [[72], [82], [92]]


def test_speculative_lookahead_moves_the_state_block_back():
    layout, _ = _qwen3_next_layout()
    local = [[10], [20, 21, 22, 23]]
    got_l, _ = narrow_mamba_groups(layout, local, local, num_speculative_tokens=2)
    assert got_l[1] == [21]


def test_outside_align_mode_the_group_holds_one_block_at_index_zero():
    assert mamba_state_index(1, 0, align=False) == 0
    assert mamba_state_index(4, 0, align=False) == 0
    assert mamba_state_index(4, 0, align=True) == 3


def test_narrowing_is_a_no_op_on_a_non_hybrid_model():
    """Every pre-Qwen3-Next run goes through this call; it must not touch the lists."""
    local, remote = [[1, 2, 3]], [[4, 5, 6]]
    assert narrow_mamba_groups(None, local, remote) == (local, remote)


def test_block_len_is_per_address_and_scaled_for_attention():
    """The vendored base takes ONE length from the first layer and applies it everywhere. Under
    hybrid that is wrong for whichever cache is not first, in both region size and stride."""
    layout, _ = _qwen3_next_layout()
    kv_caches = {
        "attn.0": (_Sized(16, 4), _Sized(16, 4)),        # k, v: 64 bytes per ROW
        "mamba.0.0": [_Sized(8, 2), _Sized(32, 2)],      # conv, ssm: different from each other
    }
    group_of = {"attn.0": 0, "mamba.0.0": 1}

    table = block_len_table(layout, kv_caches, group_of.get)

    # Attention rows are folded `scale` to a pool block; mamba is 1:1 with the pool.
    assert table == [64 * SCALE, 64 * SCALE, 16, 64]


def test_a_layer_with_no_group_is_refused_not_defaulted():
    layout, _ = _qwen3_next_layout()
    with pytest.raises(HybridLayoutError, match="no KV-cache group"):
        block_len_table(layout, {"orphan": (_Sized(4, 2),)}, lambda ln: None)


def test_block_len_table_is_none_on_a_non_hybrid_model():
    """None is the signal to keep the base's arithmetic untouched, so no measured run moves."""
    assert block_len_table(None, {}, lambda ln: 0) is None


class _Sized:
    """A tensor stand-in with just the two attributes the byte math reads."""

    def __init__(self, per_row_elems, elem_size):
        self.shape = (POOL_BLOCKS, per_row_elems)
        self._elem = elem_size

    def element_size(self):
        return self._elem


# -- registration ----------------------------------------------------------------------------
class _KVCacheTensor:
    def __init__(self, size, shared_by):
        self.size = size
        self.shared_by = shared_by


MIB2 = 2 * 1024 * 1024


def test_one_region_per_allocation_at_the_aligned_base():
    """Views sit at arbitrary offsets inside the allocation; only the allocation base is aligned."""
    tensors = [_KVCacheTensor(3 * MIB2, ["attn.0", "mamba.0.0"])]
    layer_base_addrs = {
        "attn.0": [MIB2 * 4 + 1234, MIB2 * 5 + 99],   # k, v — unaligned views
        "mamba.0.0": [MIB2 * 4, MIB2 * 4 + 777],      # conv, ssm — conv is at the base
    }
    ptrs, sizes = hybrid_registration_regions(tensors, layer_base_addrs)
    assert ptrs == [MIB2 * 4]
    assert sizes == [3 * MIB2]


def test_unaligned_allocation_base_is_refused():
    tensors = [_KVCacheTensor(MIB2, ["attn.0"])]
    with pytest.raises(HybridLayoutError, match="2 MiB aligned"):
        hybrid_registration_regions(tensors, {"attn.0": [MIB2 + 4096]})


def test_overlapping_allocations_are_refused():
    tensors = [_KVCacheTensor(2 * MIB2, ["a"]), _KVCacheTensor(2 * MIB2, ["b"])]
    addrs = {"a": [MIB2 * 4], "b": [MIB2 * 5]}
    with pytest.raises(HybridLayoutError, match="overlaps"):
        hybrid_registration_regions(tensors, addrs)


def test_an_allocation_with_no_registered_view_is_refused():
    """Silently skipping it leaves those layers' transfers reading unmapped memory."""
    tensors = [_KVCacheTensor(MIB2, ["attn.0"])]
    with pytest.raises(HybridLayoutError, match="no registered view"):
        hybrid_registration_regions(tensors, {})
