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
            # Get original layers and spec  
            original_layers = deepcopy(kv_cache_configs[0].kv_cache_groups[0].layer_names)  
            original_spec = kv_cache_configs[0].kv_cache_groups[0].kv_cache_spec  
            
            # Split into warmup (sliding window) and fused (full attention) layers  
            warmup_layers_names = original_layers[0:2] + original_layers[-2:]  
            fused_layers_names = original_layers[2:-2]  
            
            # Create new configs with two attention types  
            tmp_config = []  
            
            # Group 0: Warmup layers with sliding window attention  
            from vllm.v1.kv_cache_interface import SlidingWindowSpec, AttentionSpec
            
            # attn_spec = AttentionSpec(
            #     block_size=original_spec.block_size,
            #       num_kv_heads= original_spec.num_kv_heads,
            #         head_size=original_spec.head_size,
            #           dtype=original_spec.dtype) 
            warmup_group = deepcopy(kv_cache_configs[0].kv_cache_groups[0])  
            warmup_group.layer_names = warmup_layers_names  
            warmup_group.kv_cache_spec = SlidingWindowSpec(
                sliding_window = 8192,  
                block_size=original_spec.block_size,
                  num_kv_heads= original_spec.num_kv_heads,
                    head_size=original_spec.head_size,
                      dtype=original_spec.dtype                      
            )  
            tmp_config.append(warmup_group)

            tmp_config.extend([deepcopy(kv_cache_configs[0].kv_cache_groups[0]) for _ in range(len(fused_layers_names))])
            # full_attention_group = deepcopy(kv_cache_configs[0].kv_cache_groups[0]) 
            for idx, layer_name in enumerate(fused_layers_names):
               tmp_config[idx+1].layer_names = [layer_name] 
            # full_attention_group.layer_names = fused_layers_names  
            # Keep the original FullAttentionSpec  
            # tmp_config.append(full_attention_group)  
            
            kv_cache_configs[0].kv_cache_groups = tmp_config  
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

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

