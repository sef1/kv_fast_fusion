import sys

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm.v1.core.sched.scheduler import Scheduler
from kv_fast_fusion.fast_fusion_scheduler import (
    _handle_block_merging_with_counts,
    _update_hashes_for_merged_blocks,
    update_from_output
)

from vllm.v1.engine.core import EngineCore
from kv_fast_fusion.fast_fusion_core import _initialize_kv_caches

from types import MethodType

import torch

from vllm.compilation import fusion_attn
from vllm.logger import init_logger

from vllm.v1.core.block_pool import BlockPool
from kv_fast_fusion.fast_fusion_block_pool import patched_free_blocks

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from kv_fast_fusion.fast_fusion_flash import patched_forward as patched_flash_forward

logger = init_logger("vllm.fast_fusion_patch")

from kv_fast_fusion.kv_fast_fusion_graph_runner import (
    _update_block_tables_after_compression,
    execute_model,
    _patched_build_attention_metadata,
    sample_tokens,
    _fill_norm_buffers,
    _run_post_forward_bff,
    BLOCK_SIZE,
)

import vllm.v1.core.kv_cache_utils as kv_utils
from kv_fast_fusion.fast_fusion_kv_hash_utils import hash_block_ids, get_request_block_hasher_by_ids


def apply_fast_fusion_graph_patch():
    """Apply the fast fusion graph-mode patch."""

    original_gpu_model_runner_init = GPUModelRunner.__init__

    def patched_gpu_model_runner_init(self, *args, **kwargs):
        self.execute_model = MethodType(execute_model, self)
        self._update_block_tables_after_compression = MethodType(
            _update_block_tables_after_compression, self)
        self.sample_tokens = MethodType(sample_tokens, self)
        self._build_attention_metadata = MethodType(
            _patched_build_attention_metadata, self)
        self._fill_norm_buffers = MethodType(_fill_norm_buffers, self)
        self._run_post_forward_bff = MethodType(_run_post_forward_bff, self)
        self.fused_requests = {}
        self._updated_block_tables = None

        original_gpu_model_runner_init(self, *args, **kwargs)

        # Allocate persistent norm buffers after model and KV caches are set up
        vllm_config = self.vllm_config
        num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        max_blocks_per_req = max(1, max_model_len // BLOCK_SIZE)

        self.norms_k_buf = torch.ones(
            num_layers, max_reqs, max_blocks_per_req,
            dtype=torch.bfloat16, device=self.device)
        self.norms_v_buf = torch.ones(
            num_layers, max_reqs, max_blocks_per_req,
            dtype=torch.bfloat16, device=self.device)

        # Cache warmup/max-layer constants so _patched_build_attention_metadata
        # doesn't have to recompute them every call.
        warmup = 2
        self._ff_warmup_layers = warmup
        self._ff_max_layer_idx = num_layers - warmup

        logger.info(
            "Fast fusion graph patch: allocated norm buffers "
            "[%d layers, %d reqs, %d blocks/req]",
            num_layers, max_reqs, max_blocks_per_req,
        )

    GPUModelRunner.__init__ = patched_gpu_model_runner_init

    # Patch Scheduler to handle block merging and update from output
    Scheduler._handle_block_merging_with_counts = _handle_block_merging_with_counts
    Scheduler.update_from_output = update_from_output

    # Patch BlockPool to use free_blocks with deduplication
    BlockPool.free_blocks = patched_free_blocks

    # Patch FlashAttentionImpl with static-buffer norm scaling
    FlashAttentionImpl.forward = patched_flash_forward

    # Patch EngineCore for custom KV cache initialization
    EngineCore._initialize_kv_caches = _initialize_kv_caches
