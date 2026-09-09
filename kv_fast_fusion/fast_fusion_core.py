from vllm.config import ParallelConfig, VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
import time
import os
from copy import deepcopy
from vllm.logger import init_logger
logger = init_logger("vllm.patched_scheduler")
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)

def bff_group_layout(original_layers, balanced_groups, group_size):
    """Pure partition of the model's layer names into BFF storage groups.

    Returns ``list[(layer_names, spec_id)]``. ``spec_id`` selects one of the two page-identical
    specs the caller builds: ``0`` = ``per_layer_spec``, ``1`` = the ``sliding_window``-flipped
    ``warmup_spec``. Only two distinct spec VALUES are ever needed — get_kv_cache_config_from_groups
    preserves same-spec groups as separate groups (today's 6 fusion groups already share one spec);
    verify_and_split only requires that >1 distinct spec value be present overall.

    * ``balanced_groups >= 2`` (Update 17, dedup=0): ALL layers split into that many BALANCED
      contiguous groups (sizes differ by <=1), spec_id alternating so adjacent groups differ. This
      holds ``max_layers_per_group × num_groups`` at its floor (== total layers == baseline
      concurrency) while minimizing the per-step per-group block-table tax. N=2 is the optimum.
    * otherwise: today's layout — warmup group (first 2 + last 2 layers, alt spec) + the middle
      layers chunked by ``group_size`` (per_layer_spec). Byte-for-byte the prior behaviour.
    """
    n = len(original_layers)
    if balanced_groups and balanced_groups >= 2:
        g = max(2, min(balanced_groups, n))
        base, extra = divmod(n, g)
        layout = []
        start = 0
        for i in range(g):
            size = base + (1 if i < extra else 0)
            layout.append((original_layers[start:start + size], i % 2))
            start += size
        return layout
    warmup = original_layers[0:2] + original_layers[-2:]
    fused = original_layers[2:-2]
    layout = [(warmup, 1)]
    layout += [
        (fused[i:i + group_size], 0)
        for i in range(0, len(fused), group_size)
    ]
    return layout


def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> KVCacheConfig:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        if True: #vllm_config.kv_transfer_config.kv_role != "kv_producer":
            # Re-derive the KV cache config for the warmup/fusion group split.
            #
            # IMPORTANT: the previous version only swapped kv_cache_groups in place
            # and kept the single-group `num_blocks`/tensors from get_kv_cache_configs.
            # That froze the pool at `available/(page*num_layers)` while each request
            # then needed blocks in every group → effective concurrency ≈ vanilla/G,
            # with ~(G-1)/G of the per-layer rows stranded. We instead rebuild each
            # worker's config through the stock `get_kv_cache_config_from_groups`, so
            # `num_blocks = available/(page*max_layers_per_group)` (≈ G× larger) and
            # the shared-tensor layout is used — recovering ≈ vanilla concurrency plus
            # the fusion bonus. Mirrors the per-worker loop in get_kv_cache_configs.
            from dataclasses import replace
            from vllm.v1.kv_cache_interface import (
                KVCacheGroupSpec, UniformTypeKVCacheSpecs,
            )
            from vllm.v1.core.kv_cache_utils import (
                get_kv_cache_config_from_groups, _report_kv_cache_config,
            )
            from kv_fast_fusion.constants import (
                BFF_GROUP_SIZE, BFF_BALANCED_GROUPS,
            )

            # Reference (global) layer ordering + a concrete per-layer spec.
            ref_group = kv_cache_configs[0].kv_cache_groups[0]
            original_layers = list(ref_group.layer_names)
            ref_spec = ref_group.kv_cache_spec
            if isinstance(ref_spec, UniformTypeKVCacheSpecs):
                per_layer_spec = next(iter(ref_spec.kv_cache_specs.values()))
            else:
                per_layer_spec = ref_spec

            # first 2 + last 2 layers → warmup group; the rest → fusion.
            # NOTE: HybridKVCacheCoordinator.verify_and_split_kv_cache_groups groups
            # kv_cache_groups by spec *equality* and requires >1 resulting group. The
            # warmup group must therefore be a distinct spec value from the fusion
            # spec — but get_uniform_page_size requires all groups' page_size_bytes to
            # match. `sliding_window` is part of FullAttentionSpec/MLAAttentionSpec but
            # is NOT part of the page_size_bytes formula, so flipping only that field
            # gives us a distinct spec with an identical page size, and it keeps the
            # same manager class (FullAttentionManager, not SlidingWindowManager) so no
            # real windowed eviction is introduced. (A plain SlidingWindowSpec instead
            # would mismatch page size for MLA models: its formula assumes separate K/V
            # caches (2x) while MLAAttentionSpec's single-latent-cache formula has no
            # such factor, tripping `assert len(page_sizes) == 1` in
            # get_uniform_page_size.)
            warmup_spec = replace(per_layer_spec, sliding_window=8192)

            # Global group spec list. Default: warmup group first, then fusion chunks.
            # BFF_BALANCED_GROUPS>=2 (dedup=0): N balanced groups instead (see bff_group_layout).
            _specs = (per_layer_spec, warmup_spec)  # index by spec_id (0, 1)
            global_groups = [
                KVCacheGroupSpec(layer_names, _specs[spec_id])
                for layer_names, spec_id in bff_group_layout(
                    original_layers, BFF_BALANCED_GROUPS, BFF_GROUP_SIZE
                )
            ]
            if BFF_BALANCED_GROUPS >= 2 and os.environ.get("BFF_V2_DEDUP") == "1":
                logger.warning(
                    "BFF_BALANCED_GROUPS=%d with BFF_V2_DEDUP=1: balanced groups fuse boundary "
                    "layers and coarsen dedup granularity — supported combo is dedup=0.",
                    BFF_BALANCED_GROUPS,
                )

            # Rebuild each worker's config with the correctly-sized pool + tensors.
            rebuilt_configs = []
            for spec_one_worker, avail in zip(kv_cache_specs, available_gpu_memory):
                groups_one_worker = [
                    KVCacheGroupSpec(
                        [ln for ln in g.layer_names if ln in spec_one_worker],
                        g.kv_cache_spec,
                    )
                    for g in global_groups
                ]
                rebuilt_configs.append(
                    get_kv_cache_config_from_groups(
                        vllm_config, groups_one_worker, avail
                    )
                )

            # Unify num_blocks across workers (smallest) + shrink tensors, as stock
            # get_kv_cache_configs does, then report.
            min_num_blocks = min(c.num_blocks for c in rebuilt_configs)
            for c in rebuilt_configs:
                old = c.num_blocks
                c.num_blocks = min_num_blocks
                for tensor in c.kv_cache_tensors:
                    assert tensor.size % old == 0
                    tensor.size = tensor.size // old * min_num_blocks
                if len(c.kv_cache_groups) > 0:
                    _report_kv_cache_config(vllm_config, c)

            kv_cache_configs = rebuilt_configs
            original_spec = per_layer_spec  # used by the measurement log below
        ### sefi  end

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # v0.19.1 contract: _initialize_kv_caches returns the scheduler
        # KVCacheConfig and is itself responsible for syncing num_gpu_blocks /
        # block_size onto cache_config (the caller no longer unpacks a tuple).
        # Mirror stock EngineCore._initialize_kv_caches.
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        _sched_groups = scheduler_kv_cache_config.kv_cache_groups
        if _sched_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in _sched_groups
            )
        vllm_config.validate_block_size()

        # --- BFF measurement: log the POST-SPLIT config ---
        # The native "Maximum concurrency" log fires inside get_kv_cache_configs
        # BEFORE the group split (len(groups)==1), so it prints misleading
        # vanilla numbers. Re-derive the real numbers on the final G-group config.
        try:
            from math import ceil
            final_groups = kv_cache_configs[0].kv_cache_groups
            num_groups = len(final_groups)
            group_layer_counts = [len(g.layer_names) for g in final_groups]
            max_layers_per_group = max(group_layer_counts)
            page_size = original_spec.page_size_bytes
            block_size = original_spec.block_size
            available = available_gpu_memory[0]
            max_model_len = vllm_config.model_config.max_model_len
            blocks_per_req = ceil(max_model_len / block_size)
            eff_concurrency = num_gpu_blocks / (num_groups * blocks_per_req)
            num_blocks_if_regrouped = available // (page_size * max_layers_per_group)
            stranded_factor = num_blocks_if_regrouped / num_gpu_blocks if num_gpu_blocks else float("nan")
            logger.info(
                "BFF KV sizing | num_gpu_blocks=%d | groups=%d %s | "
                "page_size=%d B | available=%.2f GiB | block_size=%d | "
                "max_model_len=%d | blocks/req=%d | EFFECTIVE_CONCURRENCY=%.2fx | "
                "num_blocks_if_regrouped=%d | STRANDED_FACTOR=%.2fx (expect ~num_groups)",
                num_gpu_blocks, num_groups, group_layer_counts,
                page_size, available / (1024 ** 3), block_size,
                max_model_len, blocks_per_req, eff_concurrency,
                num_blocks_if_regrouped, stranded_factor,
            )
        except Exception as e:
            logger.warning("BFF KV sizing log failed: %s", e, exc_info=True)
        # --- end BFF measurement ---

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return scheduler_kv_cache_config

