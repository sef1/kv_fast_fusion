"""BFF regroups attention layers and must carry Mamba groups through untouched.

The failure these pin is silent: rebuilding every group from ``groups[0]`` keeps the attention
layers, relabels or drops the recurrent-state ones, and keeps the attention spec's
``page_size_padded`` for a Mamba partner that is gone. Nothing raises at the rebuild; the model
just decodes garbage, with ``BFF_V2_DEDUP=0`` and every dedup counter at zero.

Shapes are Qwen3-Next's: 12 attention layers and 36 Mamba layers, which vLLM splits into 1
attention group + 3 Mamba groups of 12.
"""

import pytest

from kv_fast_fusion.fast_fusion_core import bff_group_layout
from kv_fast_fusion.hybrid_groups import is_mamba_spec, split_hybrid_groups


class _Attn:
    def __init__(self, page=4096, window=None):
        self.page_size_bytes = page
        self.sliding_window = window


class _Mamba:
    def __init__(self, page=4096):
        self.page_size_bytes = page
        self.shapes = ((4, 8), (8, 8))
        self.dtypes = ("bf16", "bf16")


class _Uniform:
    """vLLM's UniformTypeKVCacheSpecs: one group wrapping a per-layer spec dict."""

    def __init__(self, per_layer):
        self.kv_cache_specs = per_layer


class _Group:
    def __init__(self, layer_names, spec):
        self.layer_names = layer_names
        self.kv_cache_spec = spec


def _qwen3_next_groups():
    attn = _Attn()
    mamba = _Mamba()
    return [
        _Group([f"attn.{i}" for i in range(12)], attn),
        _Group([f"lin.a.{i}" for i in range(12)], mamba),
        _Group([f"lin.b.{i}" for i in range(12)], mamba),
        _Group([f"lin.c.{i}" for i in range(12)], mamba),
    ]


def test_mamba_groups_are_split_off_and_returned_unmodified():
    groups = _qwen3_next_groups()
    attn_layers, attn_spec, mamba_groups = split_hybrid_groups(groups)

    assert attn_layers == [f"attn.{i}" for i in range(12)]
    assert isinstance(attn_spec, _Attn)
    assert mamba_groups == groups[1:], "same objects, same order — nothing rebuilt"
    assert all(a is b for a, b in zip(mamba_groups, groups[1:]))


def test_a_non_hybrid_model_is_untouched():
    """The legacy rebuild path must run byte-for-byte for every pre-Qwen3-Next model."""
    assert split_hybrid_groups([_Group(["a", "b"], _Attn())]) == (None, None, [])


def test_the_attention_spec_keeps_its_padding_only_because_mamba_stays():
    """The whole point: `page_size_padded` pads the attention page to match a Mamba page. The
    rebuild is only safe to keep that spec if the Mamba groups it pairs with come through too."""
    groups = _qwen3_next_groups()
    attn_layers, attn_spec, mamba_groups = split_hybrid_groups(groups)
    assert attn_spec.page_size_bytes == mamba_groups[0].kv_cache_spec.page_size_bytes


def test_a_uniform_type_wrapper_is_unwrapped_to_a_concrete_spec():
    inner = _Attn()
    groups = [_Group(["a0", "a1"], _Uniform({"a0": inner, "a1": inner})),
              _Group(["m0", "m1"], _Mamba())]
    _, attn_spec, _ = split_hybrid_groups(groups)
    assert attn_spec is inner


def test_striped_attention_groups_are_restored_to_model_order():
    """vLLM deals same-type layers round-robin across groups (`layers[i::n]`). BFF's layouts are
    contiguous ranges of the MODEL, so they must be applied to the original order."""
    spec = _Attn()
    original = [f"attn.{i}" for i in range(6)]
    groups = [_Group(original[0::2], spec), _Group(original[1::2], spec),
              _Group(["m0"], _Mamba())]
    attn_layers, _, _ = split_hybrid_groups(groups)
    assert attn_layers == original


def test_mamba_without_any_attention_is_refused_not_rebuilt():
    with pytest.raises(ValueError, match="no attention group"):
        split_hybrid_groups([_Group(["m0"], _Mamba())])


def test_mamba_detection_by_class_name_and_by_shape():
    class MambaSpec:        # the real class, by name, however it is imported
        pass

    assert is_mamba_spec(MambaSpec())
    assert is_mamba_spec(_Mamba())
    assert not is_mamba_spec(_Attn())
    assert not is_mamba_spec(None)


def test_bff_layout_over_only_the_attention_layers_still_has_a_warmup_group():
    """Group 0 is the never-fused warmup group every consumer assumes. Regrouping the 12 attention
    layers must still put one there, with the two distinct spec ids the verify step needs."""
    attn_layers, _, _ = split_hybrid_groups(_qwen3_next_groups())
    layout = bff_group_layout(attn_layers, balanced_groups=0, group_size=4)

    assert layout[0][0] == attn_layers[0:2] + attn_layers[-2:]
    assert layout[0][1] == 1 and {sid for _, sid in layout[1:]} == {0}
    assert sorted(ln for names, _ in layout for ln in names) == sorted(attn_layers)
