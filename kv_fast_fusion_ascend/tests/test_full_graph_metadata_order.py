"""Full-graph replay must pair each captured attention task with ITS OWN layer's metadata (Update 36).

vllm-ascend's FIA ``update_graph_params`` zips ``list(attn_metadata.keys())`` against
``graph_params.attn_params[num_tokens]``. The params are appended in forward order; the dict is filled
KV-cache group by group. Under BFF's legacy layout the warmup group is ``{0, 1, 26, 27}``, so the two
orders disagree and layers replay against another group's block table — the F1 0.2704 full-graph run.
These tests pin (1) that the mechanism is real on the actual layouts BFF builds, (2) the pure reorder,
and (3) the runner wrapper's gating. CPU only: the runner is faked.
"""

import sys
import types

import kv_fast_fusion_ascend.fast_fusion_ascend_patch as patch
from kv_fast_fusion.fast_fusion_core import bff_group_layout

N_LAYERS = 28


def _names(n=N_LAYERS):
    return [f"model.layers.{i}.self_attn.attn" for i in range(n)]


def _group_ordered_meta(layout):
    """attn_metadata exactly as _build_attention_metadata fills it: group by group."""
    meta = {}
    for gi, (names, _spec) in enumerate(layout):
        for name in names:
            meta[name] = f"meta-of-group-{gi}"
    return meta


def _cross_group_pairings(layout):
    """How many captured layers the positional zip hands ANOTHER group's metadata."""
    group_of = {name: gi for gi, (names, _s) in enumerate(layout) for name in names}
    meta_keys = list(_group_ordered_meta(layout))
    return sum(1 for captured, key in zip(_names(), meta_keys) if group_of[captured] != group_of[key])


# --- (1) the mechanism, on the layouts BFF really builds -------------------------------------------
def test_the_legacy_layout_pairs_many_layers_with_another_groups_block_table():
    """This is the corruption: a layer attending over block ids that belong to a different group."""
    legacy = bff_group_layout(_names(), balanced_groups=0, group_size=4)
    assert [int(n.split(".")[2]) for n in legacy[0][0]] == [0, 1, 26, 27], "the warmup quarantine"
    assert _cross_group_pairings(legacy) >= 10, "roughly half the layers read the wrong group"


def test_a_contiguous_layout_is_already_aligned_which_is_the_zero_code_prediction():
    """Balanced groups are contiguous, so group order IS forward order — full graph should already be
    correct there with no code, which makes it the cheap first box test of the hypothesis."""
    for n in (2, 4):
        assert _cross_group_pairings(bff_group_layout(_names(), balanced_groups=n, group_size=4)) == 0


# --- (2) the pure reorder ---------------------------------------------------------------------------
def test_forward_ordered_restores_forward_order_in_place():
    legacy = bff_group_layout(_names(), balanced_groups=0, group_size=4)
    meta = _group_ordered_meta(legacy)
    same_object = meta
    moved = patch.forward_ordered(meta, _names())
    assert list(meta) == _names(), "keys now follow capture order, so the zip pairs each layer with itself"
    assert meta is same_object, "in place: any holder of the runner's dict sees the fix"
    assert moved == 26, "only 0 and 1 were in place: 26,27 ahead of 2 shifts every later layer"
    group_of = {name: gi for gi, (names, _s) in enumerate(legacy) for name in names}
    assert all(meta[k] == f"meta-of-group-{group_of[k]}" for k in meta), "values travel with their keys"


def test_forward_ordered_is_the_identity_on_an_aligned_dict():
    meta = {k: i for i, k in enumerate(_names())}
    assert patch.forward_ordered(meta, _names()) == 0
    assert list(meta) == _names()


def test_forward_ordered_keeps_unknown_keys_at_the_end_in_their_order():
    meta = {"model.layers.1.self_attn.attn": 1, "extra.b": "b", "model.layers.0.self_attn.attn": 0,
            "extra.a": "a"}
    patch.forward_ordered(meta, _names(2))
    assert list(meta) == ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn",
                          "extra.b", "extra.a"]


# --- (3) the runner wrapper ---------------------------------------------------------------------------
def _install_fake_runner(monkeypatch, layout, returns_list=False):
    class NPUModelRunner:
        def _build_attention_metadata(self, *a, **k):
            meta = _group_ordered_meta(layout)
            return ([meta, _group_ordered_meta(layout)] if returns_list else meta), "spec-meta"

    mod = types.ModuleType("vllm_ascend.worker.model_runner_v1")
    mod.NPUModelRunner = NPUModelRunner
    for name in ("vllm_ascend", "vllm_ascend.worker"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.model_runner_v1", mod)
    monkeypatch.setattr(patch, "_FORWARD_ORDER_ANNOUNCED", [False])
    patch._patch_attn_metadata_forward_order()
    return NPUModelRunner


def _runner(cls, layout, full=True):
    r = cls()
    mode = types.SimpleNamespace(has_full_cudagraphs=lambda: full)
    r.vllm_config = types.SimpleNamespace(compilation_config=types.SimpleNamespace(
        cudagraph_mode=mode, static_forward_context={n: object() for n in _names()}))
    r.kv_cache_config = types.SimpleNamespace(kv_cache_groups=[object()] * len(layout))
    return r


def test_the_wrapper_reorders_under_full_graph_and_forwards_the_second_return(monkeypatch):
    legacy = bff_group_layout(_names(), balanced_groups=0, group_size=4)
    cls = _install_fake_runner(monkeypatch, legacy)
    meta, spec = _runner(cls, legacy)._build_attention_metadata()
    assert list(meta) == _names()
    assert spec == "spec-meta", "the spec-decode metadata must pass through untouched"
    assert patch._FORWARD_ORDER_ANNOUNCED[0], "and the run says whether the fix mattered"


def test_the_wrapper_leaves_piecewise_and_single_group_runs_byte_identical(monkeypatch):
    legacy = bff_group_layout(_names(), balanced_groups=0, group_size=4)
    cls = _install_fake_runner(monkeypatch, legacy)
    meta, _ = _runner(cls, legacy, full=False)._build_attention_metadata()
    assert list(meta) == list(_group_ordered_meta(legacy)), "PIECEWISE: untouched"

    one = [(_names(), 0)]
    cls1 = _install_fake_runner(monkeypatch, one)
    meta1, _ = _runner(cls1, one)._build_attention_metadata()
    assert list(meta1) == _names()
    assert patch._FORWARD_ORDER_ANNOUNCED[0] is False, "a stock single-group run is not even inspected"


def test_the_wrapper_handles_a_ubatch_list_and_wraps_once(monkeypatch):
    legacy = bff_group_layout(_names(), balanced_groups=0, group_size=4)
    cls = _install_fake_runner(monkeypatch, legacy, returns_list=True)
    wrapped = cls._build_attention_metadata
    patch._patch_attn_metadata_forward_order()
    assert cls._build_attention_metadata is wrapped, "wrap-once"
    metas, _ = _runner(cls, legacy)._build_attention_metadata()
    assert all(list(m) == _names() for m in metas)
