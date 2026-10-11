"""Pure partitioning of a hybrid (attention + Mamba) KV-cache group list. No vllm imports.

``_initialize_kv_caches`` rebuilds every KV group from ONE reference group and ONE spec so BFF can
split the attention layers into a warmup group plus fusion groups. That is correct for a model whose
layers all share a spec, and silently destroys a hybrid one: vLLM emits an attention group *and*
``MambaSpec`` groups, and rebuilding from ``groups[0]`` keeps the attention layers, drops (or
re-labels) every recurrent-state layer, and keeps the attention spec's ``page_size_padded`` — a
padding that exists only so an attention page can share an allocation with a Mamba page that is no
longer there. Nothing crashes at the rebuild; the model just stops computing correctly.

The rule this module encodes: **BFF may regroup attention layers; it must carry Mamba groups through
untouched.** Kept free of vllm so the rule is unit-testable on any box, which is the only place it
has ever been tested.
"""


def is_mamba_spec(spec) -> bool:
    """True for a ``MambaSpec`` without importing it (the class moved between vLLM versions).

    ``MambaSpec`` is the only spec carrying per-state ``shapes``/``dtypes`` tuples, which is the
    fallback for a subclass or a renamed class."""
    if spec is None:
        return False
    if any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__):
        return True
    return hasattr(spec, "shapes") and hasattr(spec, "dtypes")


def _spec_of(group):
    """A group's spec, unwrapping vLLM's ``UniformTypeKVCacheSpecs`` to one concrete layer spec."""
    spec = group.kv_cache_spec
    per_layer = getattr(spec, "kv_cache_specs", None)
    if isinstance(per_layer, dict) and per_layer:
        return next(iter(per_layer.values()))
    return spec


def _unstripe(groups) -> list[str]:
    """Undo vLLM's ``layers[i::num_groups]`` striping, recovering model order.

    ``_get_kv_cache_groups_uniform_page_size`` deals same-type layers round-robin across groups so
    a pipeline stage's layers spread evenly. BFF's layouts are *contiguous* ranges of the model, so
    they have to be applied to the original order, not the striped one. With a single group of the
    type — the Qwen3-Next case — this is the identity."""
    out: list[str] = []
    for j in range(max((len(g.layer_names) for g in groups), default=0)):
        for g in groups:
            if j < len(g.layer_names):
                out.append(g.layer_names[j])
    return out


def split_hybrid_groups(stock_groups):
    """``(attention_layers, attention_spec, mamba_groups)`` from vLLM's stock group list.

    ``attention_layers`` is in model order and ``attention_spec`` is one concrete per-layer spec,
    both ready for ``bff_group_layout``. ``mamba_groups`` are the stock group objects, unmodified,
    in their original order.

    Returns ``(None, None, [])`` when there are no Mamba groups: the caller keeps its legacy
    behaviour byte-for-byte, so no non-hybrid run moves. Raises if there are Mamba groups but no
    attention group to regroup — BFF has nothing to do for such a model and rebuilding it would
    only destroy it.
    """
    mamba_groups = [g for g in stock_groups if is_mamba_spec(_spec_of(g))]
    if not mamba_groups:
        return None, None, []
    attn_groups = [g for g in stock_groups if not is_mamba_spec(_spec_of(g))]
    if not attn_groups:
        raise ValueError(
            "the KV cache config has Mamba groups but no attention group; BFF regroups attention "
            "layers and has nothing to do here. Disable BFF for this model rather than rebuilding "
            "its groups.")
    return _unstripe(attn_groups), _spec_of(attn_groups[0]), mamba_groups
