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

def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
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
            from vllm.v1.kv_cache_interface import (
                SlidingWindowSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs,
            )
            from vllm.v1.core.kv_cache_utils import (
                get_kv_cache_config_from_groups, _report_kv_cache_config,
            )
            from kv_fast_fusion.kv_fast_fusion_graph_runner import BFF_GROUP_SIZE

            # Reference (global) layer ordering + a concrete per-layer spec.
            ref_group = kv_cache_configs[0].kv_cache_groups[0]
            original_layers = list(ref_group.layer_names)
            ref_spec = ref_group.kv_cache_spec
            if isinstance(ref_spec, UniformTypeKVCacheSpecs):
                per_layer_spec = next(iter(ref_spec.kv_cache_specs.values()))
            else:
                per_layer_spec = ref_spec

            # first 2 + last 2 layers → warmup (sliding window); the rest → fusion.
            warmup_layers_names = original_layers[0:2] + original_layers[-2:]
            fused_layers_names = original_layers[2:-2]
            fused_chunks = [
                fused_layers_names[i:i + BFF_GROUP_SIZE]
                for i in range(0, len(fused_layers_names), BFF_GROUP_SIZE)
            ]

            warmup_spec = SlidingWindowSpec(
                sliding_window=8192,
                block_size=per_layer_spec.block_size,
                num_kv_heads=per_layer_spec.num_kv_heads,
                head_size=per_layer_spec.head_size,
                dtype=per_layer_spec.dtype,
            )
            # Global group spec list (warmup first, then fusion chunks).
            global_groups = [KVCacheGroupSpec(warmup_layers_names, warmup_spec)]
            global_groups += [
                KVCacheGroupSpec(chunk, per_layer_spec) for chunk in fused_chunks
            ]

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
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

