"""Hybrid (attention + Mamba) KV geometry — the one place it is derived.

Qwen3-Next is the first model BFF runs where the KV pool is not uniform. 48 layers in a 3:1
linear:full pattern give 12 ``FullAttentionSpec`` layers and 36 ``MambaSpec`` layers, and vLLM's
``_get_kv_cache_groups_uniform_page_size`` splits that into **four** KV-cache groups of 12 layers
(1 attention + 3 mamba), backed by **twelve** shared allocations — one layer from every group per
allocation. Three assumptions BFF was built on all break at once:

* **One block id space.** Not any more. The pool hands out one id space, but the attention group's
  *tensor* is ``scale`` times longer than the pool, because ``vllm_ascend``'s hybrid config patch
  raises ``cache_config.block_size`` to a multiple of the attention backend's only supported kernel
  block size (128) so the attention page matches the mamba page. A pool block ``p`` of the attention
  group is therefore rows ``p*scale … p*scale+scale-1`` of that tensor, and the device block table
  holds those *logical* rows while the scheduler, the alias map and the free hook all speak *pool*
  ids. Mixing the two does not raise: it reads the wrong KV.
* **One block length.** Attention K, attention V, mamba conv and mamba ssm each have their own
  bytes-per-block.
* **One region per allocation.** Every one of those four is a *view* into the same shared int8
  buffer, so they all report base pointers inside one region.

``scale`` is config-derived — ``cache_config.block_size // 128``, computed in
``vllm_ascend/patch/platform/patch_mamba_config.py`` from the ssm page size — so it changes with the
model, the dtype and TP. **It is never a constant.** Every previous attempt to hand-roll it in a
call site produced a different answer, which is why this module cross-checks two independent
derivations and refuses to guess when they disagree.

Nothing here imports torch or ``vllm_ascend``; the tensors are duck-typed on ``.shape[0]``. That
keeps the whole geometry unit-testable on a GPU box, which matters because the failure mode it
guards against is silent on an NPU.
"""

from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger("vllm.bff_hybrid_layout")

# The only kernel block size AscendAttentionBackend/AscendMLABackend advertise
# (`get_supported_kernel_block_sizes() -> [128]`). Used only to sanity-check a derived scale
# against the configured block size, never to compute one.
ASCEND_KERNEL_BLOCK_SIZE = 128


class HybridLayoutError(RuntimeError):
    """The hybrid KV geometry could not be derived, or two derivations disagreed.

    Always fatal, for the same reason :class:`KVGroupLayoutError` is: a wrong scale does not raise
    downstream, it transfers and signs the wrong rows. Every stat still reads as healthy while the
    decode serves garbage, so there is no run to salvage by continuing."""


class RegroupedHybridError(HybridLayoutError):
    """The model is hybrid but its KV groups are not — something regrouped it as attention.

    A subclass so a caller that legitimately tolerates "groups not readable yet" (the pull
    connector's registration fallback) can catch the parent and still let this one through: the two
    look alike at the ``except`` and are opposite in meaning."""


@dataclass(frozen=True)
class GroupGeometry:
    """Per-KV-cache-group geometry, in both id spaces."""

    index: int
    is_mamba: bool
    #: kernel/logical rows per pool block. Always 1 for a mamba group — a mamba block holds one
    #: request's recurrent state, it is not a token window and is never subdivided.
    scale: int
    #: blocks the *pool* allocates. Identical for every group; the tensors are what differ.
    pool_num_blocks: int
    #: the group's own block size in tokens, as its spec reports it.
    block_size: int

    @property
    def tensor_rows(self) -> int:
        """Rows on dim 0 of this group's KV tensors — what ``signature_matrix`` indexes."""
        return self.pool_num_blocks * self.scale


class HybridLayout:
    """The derived geometry of every KV-cache group, and the two id-space conversions.

    ``to_kernel`` / ``to_pool`` are the only sanctioned way to cross between the scheduler's pool
    ids and a KV tensor's row index. Mamba groups pass through both unchanged, deliberately: their
    "scale" is 1 and treating them as divisible is how a recurrent state gets aliased to another
    request's.
    """

    def __init__(self, groups: list[GroupGeometry]):
        if not groups:
            raise HybridLayoutError("no KV-cache groups — nothing to derive geometry from")
        self.groups = groups
        self.pool_num_blocks = groups[0].pool_num_blocks

    # -- predicates ------------------------------------------------------------------------
    @property
    def is_hybrid(self) -> bool:
        """True when any group is mamba-backed, i.e. when any of this module applies."""
        return any(g.is_mamba for g in self.groups)

    @property
    def is_split(self) -> bool:
        """True when any attention group's tensor is longer than the pool (``scale > 1``)."""
        return any(g.scale > 1 for g in self.groups)

    def is_mamba(self, gi: int) -> bool:
        return self.groups[gi].is_mamba

    def scale(self, gi: int) -> int:
        return self.groups[gi].scale

    def tensor_rows(self, gi: int) -> int:
        """Row count to hand ``signature_matrix`` as ``num_blocks`` for this group.

        The global ``kv_cache_config.num_blocks`` is the *pool's* count and is wrong for any split
        attention group — passing it is what makes the layout guard fire and disable dedup for the
        whole run."""
        return self.groups[gi].tensor_rows

    # -- id-space conversion ---------------------------------------------------------------
    def to_kernel(self, gi: int, pool_ids) -> list[int]:
        """Pool block ids → the kernel/logical rows they own, in order.

        Mirrors ``MooncakeLayerwiseConnectorWorker._get_kernel_block_ids`` and the runner's
        ``BlockTable._convert_physical_to_logical_blocks``; kept here so the pull transport, which
        has no equivalent, can share one implementation with the layerwise one."""
        s = self.groups[gi].scale
        if s == 1:
            return [int(b) for b in pool_ids]
        return [int(b) * s + j for b in pool_ids for j in range(s)]

    def to_pool(self, gi: int, kernel_ids) -> list[int]:
        """Kernel rows → the pool blocks that own them, de-duplicated, order preserved.

        A partial run of rows still maps to its owning pool block: the caller gets the block back
        once, and it is the caller's job to decide whether owning *some* rows is enough (for a free
        it is not — see the all-of-N note in ``to_kernel``'s callers)."""
        s = self.groups[gi].scale
        if s == 1:
            return [int(b) for b in kernel_ids]
        out: list[int] = []
        seen: set[int] = set()
        for b in kernel_ids:
            p = int(b) // s
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # -- reporting -------------------------------------------------------------------------
    def describe(self) -> str:
        return ", ".join(
            f"g{g.index}={'mamba' if g.is_mamba else 'attn'}"
            f"(scale={g.scale}, block_size={g.block_size}, rows={g.tensor_rows})"
            for g in self.groups)

    def log_engagement(self, source: str) -> None:
        """Emit the one line a run is read by.

        Four runs on this project were lost to a diagnostic that was silently inert, so the rule is
        that every derived number appears once, greppably, before any of it is used."""
        logger.info(
            "BFF hybrid layout [%s]: %d groups, pool=%d blocks, hybrid=%s split=%s — %s",
            source, len(self.groups), self.pool_num_blocks, self.is_hybrid, self.is_split,
            self.describe())


# =================================================================================================
# derivation — two independent sources, cross-checked
# =================================================================================================
from kv_fast_fusion.hybrid_groups import is_mamba_spec as _is_mamba_spec  # one detector, not two


def model_declares_mamba(vllm_config) -> bool:
    """True when the MODEL has recurrent-state layers, read from its config rather than from the
    KV groups — because the groups are exactly what a regrouping bug corrupts, so they cannot be
    the witness. Covers Qwen3-Next / Qwen3.5-style configs (``layer_types`` containing
    ``linear_attention``, or a ``full_attention_interval``)."""
    try:
        cfg = vllm_config.model_config.hf_text_config
    except AttributeError:
        return False
    types = getattr(cfg, "layer_types", None) or ()
    if any("linear" in str(t) or "mamba" in str(t) for t in types):
        return True
    return getattr(cfg, "full_attention_interval", None) not in (None, 0, 1)


def _first_cache_rows(cache_or_caches) -> int | None:
    """Rows on dim 0 of a layer's KV, for both shapes the runner produces.

    Attention layers come back as a ``(k_cache, v_cache)`` tuple, mamba layers as a list of state
    tensors. A bare tensor is also accepted — but note that *iterating* one walks its block axis,
    not its caches, which is the bug behind ``_caches_per_layer`` in the pull connector. Hence this
    helper indexes explicitly rather than iterating."""
    if cache_or_caches is None:
        return None
    if isinstance(cache_or_caches, (list, tuple)):
        if not cache_or_caches:
            return None
        first = cache_or_caches[0]
    else:
        first = cache_or_caches
    shape = getattr(first, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def scales_from_tensors(kv_cache_groups, kv_caches, pool_num_blocks) -> dict[int, int]:
    """``group index -> scale``, read off the allocated tensors.

    This is the derivation the vendored layerwise connector already trusts
    (``block_size_scale = tensor_num_blocks // num_blocks``). It is authoritative because it
    measures what was actually allocated rather than what the config asked for."""
    out: dict[int, int] = {}
    for gi, group in enumerate(kv_cache_groups):
        for layer_name in getattr(group, "layer_names", ()):
            rows = _first_cache_rows(kv_caches.get(layer_name))
            if rows is None:
                continue
            if pool_num_blocks <= 0 or rows % pool_num_blocks != 0:
                raise HybridLayoutError(
                    f"group {gi} layer {layer_name!r}: tensor has {rows} rows, which is not a "
                    f"whole multiple of the pool's {pool_num_blocks} blocks. A connector block id "
                    "cannot index this tensor under any scale.")
            out[gi] = rows // pool_num_blocks
            break
    return out


def scales_from_block_tables(runner, n_groups) -> dict[int, int]:
    """``group index -> scale``, read off the runner's per-group block tables.

    The independent check: ``vllm_ascend``'s ``BlockTable`` computes ``blocks_per_phys_block``
    itself, from ``physical_block_size // logical_block_size``. If it disagrees with the tensors,
    the device block table and the KV tensors are indexed differently and *nothing* downstream is
    trustworthy. Returns whatever it can read — an empty dict just means no cross-check is
    available (the runner may not be reachable from this process)."""
    out: dict[int, int] = {}
    block_tables = getattr(getattr(runner, "input_batch", None), "block_table", None)
    if block_tables is None:
        return out
    for gi in range(n_groups):
        try:
            bt = block_tables[gi]
        except (IndexError, KeyError, TypeError):
            continue
        s = getattr(bt, "blocks_per_phys_block", None)
        if s is None:
            # Upstream vLLM's BlockTable spells the same quantity differently.
            s = getattr(bt, "blocks_per_kv_block", None)
        if s is not None:
            out[gi] = int(s)
    return out


def build_hybrid_layout(kv_cache_config, runner=None, kv_caches=None,
                        configured_block_size=None, expect_mamba: bool = False) -> HybridLayout:
    """Derive the geometry, cross-checking every scale that two sources can both see.

    ``kv_caches`` is the runner's ``layer_name -> caches`` map, as handed to
    ``register_kv_caches``. Without it only the block tables are consulted; without either, every
    group falls back to scale 1, which is correct for a non-hybrid model and for every mamba group.

    Disagreement is fatal rather than majority-voted. The two sources index different things — one
    the KV tensors, one the device block table — and if they differ, transfers and block-table
    redirects are already pointing at different rows."""
    # Same two sources, in the same order, as `resolve_kv_cache_groups`: the config the factory
    # handed the connector, else the live runner's. Which one is populated depends on import order,
    # not on anything the caller controls.
    source = kv_cache_config
    groups = getattr(source, "kv_cache_groups", None)
    if not groups:
        source = getattr(runner, "kv_cache_config", None)
        groups = getattr(source, "kv_cache_groups", None)
    if not groups:
        raise HybridLayoutError(
            "no kv_cache_groups on either the connector's kv_cache_config or the active runner — "
            "the hybrid geometry cannot be derived, and guessing it would transfer and sign the "
            "wrong rows silently.")

    pool_num_blocks = int(getattr(source, "num_blocks", 0) or 0)
    if pool_num_blocks <= 0:
        raise HybridLayoutError(
            f"the KV cache config reports num_blocks={pool_num_blocks!r}; every scale is derived "
            "by dividing a tensor row count by it.")

    from_tensors = (scales_from_tensors(groups, kv_caches, pool_num_blocks)
                    if kv_caches else {})
    from_tables = scales_from_block_tables(runner, len(groups)) if runner is not None else {}

    geometries: list[GroupGeometry] = []
    for gi, group in enumerate(groups):
        spec = getattr(group, "kv_cache_spec", None)
        is_mamba = _is_mamba_spec(spec)
        block_size = int(getattr(spec, "block_size", 0) or 0)

        t_scale = from_tensors.get(gi)
        b_scale = from_tables.get(gi)

        if is_mamba:
            # A mamba block is one request's recurrent state. The runner sets its kernel block size
            # to 0 ("no splitting, skip slot mapping"), so a block table may report a nonsense
            # scale here; the tensors are what matter, and they are always 1:1 with the pool.
            if t_scale not in (None, 1):
                raise HybridLayoutError(
                    f"group {gi} is a mamba group but its tensor has {t_scale}x the pool's rows. "
                    "Mamba state is not token-blocked and must never be subdivided.")
            scale = 1
        elif t_scale is not None and b_scale is not None and t_scale != b_scale:
            raise HybridLayoutError(
                f"group {gi}: the KV tensors say scale={t_scale} "
                f"({pool_num_blocks * t_scale} rows over a {pool_num_blocks}-block pool) but the "
                f"runner's block table says scale={b_scale}. The device block table and the KV "
                "tensors are indexed differently; transfers and redirects already disagree.")
        else:
            # The tensors win when both are readable: they measure what was allocated, while a
            # block table reports what the runner intends to index it with.
            scale = t_scale if t_scale is not None else (b_scale if b_scale is not None else 1)

        if scale < 1:
            raise HybridLayoutError(f"group {gi}: derived a scale of {scale}, which cannot index "
                                    "anything")

        # Cheap consistency check against the configured block size, when we have one. This is the
        # check that catches a hardcoded scale surviving a config change.
        if (not is_mamba and scale > 1 and configured_block_size
                and configured_block_size != scale * ASCEND_KERNEL_BLOCK_SIZE):
            logger.warning(
                "BFF hybrid layout: group %d derived scale=%d, which implies a block size of %d "
                "tokens, but cache_config.block_size is %d. Trusting the derived scale; if the "
                "attention backend's kernel block size is not %d this warning is expected.",
                gi, scale, scale * ASCEND_KERNEL_BLOCK_SIZE, configured_block_size,
                ASCEND_KERNEL_BLOCK_SIZE)

        geometries.append(GroupGeometry(
            index=gi,
            is_mamba=is_mamba,
            scale=scale,
            pool_num_blocks=pool_num_blocks,
            block_size=block_size,
        ))

    # The inert-by-omission guard. Every hybrid path in this package keys off `is_hybrid`, so a
    # regrouping bug that leaves attention-only groups does not fail — it makes ALL of them quietly
    # not run, and the model decodes against missing recurrent state. Two independent witnesses that
    # the model really is hybrid: the model config (`expect_mamba`), and the runner's own idiom of
    # naming recurrent layers `...linear_attn` among the registered caches.
    declared = expect_mamba or any("linear_attn" in ln for ln in (kv_caches or ()))
    if declared and not any(g.is_mamba for g in geometries):
        raise RegroupedHybridError(
            "the model has recurrent-state (Mamba / linear-attention) layers but none of the "
            f"{len(geometries)} KV-cache groups is a MambaSpec group. Something regrouped the layers "
            "as attention — in BFF that is `_initialize_kv_caches` rebuilding every group from "
            "groups[0]. Continuing would run every hybrid code path inert and decode against "
            "state that was never allocated.")

    return HybridLayout(geometries)


# =================================================================================================
# transfer geometry
# =================================================================================================
def mamba_state_index(n_blocks: int, num_speculative_tokens: int, align: bool) -> int:
    """Which of a Mamba group's blocks holds the request's state at the end of prefill.

    A Mamba layer keeps one recurrent state per request, not a per-token window. Under
    ``mamba_cache_mode="align"`` the group is still allocated a block per attention block so the
    block tables line up, and only the **last** holds the final state — the earlier ones are
    scratch the prefill has already moved past. Speculative decoding reserves lookahead blocks past
    it, hence the offset. Outside align mode the group holds one block and it is index 0.

    Verbatim rule from ``MooncakeLayerwiseConnectorWorker.get_transfer_meta``'s MambaSpec branch,
    which is the only implementation of it that has ever run."""
    if not align:
        return 0
    return max(0, n_blocks - int(num_speculative_tokens) - 1)


def narrow_mamba_groups(layout: HybridLayout | None, local_groups, remote_groups,
                        num_speculative_tokens: int = 0, align: bool = True):
    """Reduce every Mamba group to the single block holding the state; leave the rest alone.

    Pulling all of a Mamba group's blocks wastes the wire; pulling the wrong one hands the decode a
    mid-prompt state, which produces fluent, wrong text and no error. Returns the inputs unchanged
    for a non-hybrid model, which is the path every pre-Qwen3-Next run took."""
    if layout is None or not layout.is_hybrid:
        return local_groups, remote_groups
    out_local, out_remote = list(local_groups), list(remote_groups)
    for gi in range(min(len(out_local), len(out_remote), len(layout.groups))):
        if not layout.is_mamba(gi):
            continue
        loc, rem = list(out_local[gi]), list(out_remote[gi])
        if not loc or not rem:
            continue
        out_local[gi] = [loc[mamba_state_index(len(loc), num_speculative_tokens, align)]]
        out_remote[gi] = [rem[mamba_state_index(len(rem), num_speculative_tokens, align)]]
    return out_local, out_remote


def block_len_table(layout: HybridLayout | None, kv_caches, group_of_layer) -> list[int] | None:
    """Bytes per POOL block for each registered address, in ``kv_caches`` iteration order.

    The vendored pull connector derives ONE block length from the first layer and applies it to
    every address. On a hybrid model four different lengths are in play — attention K, attention V,
    mamba conv, mamba ssm — so that figure is wrong for whichever cache is not first, in both the
    region size and the per-block stride.

    An attention tensor holds ``scale`` rows per pool block and those rows are consecutive in a
    contiguous tensor, so a pool block is one span of ``scale * per_row`` bytes. That is why no
    part of the transfer path needs kernel block ids: the span the base already emits per block is
    simply ``scale`` times longer.

    ``group_of_layer`` maps a layer name to its group index (or None). Returns None for a
    non-hybrid model so that path keeps the base's arithmetic untouched."""
    if layout is None or not layout.is_hybrid:
        return None
    table: list[int] = []
    for layer_name, caches in kv_caches.items():
        gi = group_of_layer(layer_name)
        if gi is None:
            raise HybridLayoutError(
                f"layer {layer_name!r} has KV caches but no KV-cache group; its bytes per block "
                "cannot be derived and every transfer touching it would use another layer's "
                "stride.")
        scale = layout.scale(gi)
        for cache in (caches if isinstance(caches, (list, tuple)) else [caches]):
            per_row = cache.element_size() * _prod(cache.shape[1:])
            table.append(per_row * scale)
    return table


def _prod(dims) -> int:
    out = 1
    for d in dims:
        out *= int(d)
    return out


# =================================================================================================
# registration geometry
# =================================================================================================
def hybrid_registration_regions(kv_cache_tensors, layer_base_addrs) -> tuple[list[int], list[int]]:
    """One region per *allocation*, not per view — the hybrid registration recipe.

    Under a hybrid model every KV view is a slice of one shared int8 buffer::

        tensor1: [(kv_padding), conv           , ...]
        tensor2: [k           , ssm            , ...]
        tensor3: [v           , (mamba_padding), ...]

    so registering per view gives Mooncake a pile of overlapping extents and it refuses the lot
    ("Transfer Engine does not support overlapped memory region"). Registering *whole allocations*
    instead is both correct and simpler, and it disposes of the 2 MiB alignment failure for free:
    the runner already aligns each raw buffer to 2 MiB when a ``kv_transfer_config`` is present,
    while the views inside it (k at ``+conv_padding``, ssm at ``+conv_bytes``) are at arbitrary
    offsets and are exactly what CANN's ``rtsIpcMemGetExportKey`` rejects. Snapping view pointers
    down to their 2 MiB page would "work" by registering memory we do not own; this does not need
    to.

    This is the recipe ``MooncakeLayerwiseConnectorWorker.register_kv_caches`` already uses for
    ``use_attn_mamba_hybrid``. It lives here so the pull transport can use the same one.

    ``layer_base_addrs`` maps ``layer_name -> list of that layer's view base addresses``. The
    allocation base is the minimum over every view of every layer sharing it.
    """
    ptrs: list[int] = []
    sizes: list[int] = []
    for kv_cache_tensor in kv_cache_tensors:
        addrs: list[int] = []
        for layer_name in getattr(kv_cache_tensor, "shared_by", ()):
            addrs.extend(layer_base_addrs.get(layer_name, ()))
        if not addrs:
            raise HybridLayoutError(
                f"allocation shared by {list(getattr(kv_cache_tensor, 'shared_by', ()))!r} has no "
                "registered view addresses; it would go unregistered and every transfer touching "
                "those layers would fail or read unmapped memory.")
        base = min(addrs)
        size = int(kv_cache_tensor.size)
        if base % (2 * 1024 * 1024) != 0:
            raise HybridLayoutError(
                f"allocation base 0x{base:x} is not 2 MiB aligned. CANN's rtsIpcMemGetExportKey "
                "refuses unaligned regions ('driver error:internal error' / 'Failed to connect'). "
                "The runner aligns these only when a kv_transfer_config is set — check that the "
                "engine was launched with one.")
        if any(base < p + s and p < base + size for p, s in zip(ptrs, sizes)):
            raise HybridLayoutError(
                f"allocation [0x{base:x}, +{size}) overlaps one already registered; Mooncake will "
                "refuse the registration. Two KVCacheTensors should never share bytes.")
        ptrs.append(base)
        sizes.append(size)
    return ptrs, sizes
