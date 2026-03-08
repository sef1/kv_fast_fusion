import sys

import vllm.forward_context as forward_context_module
from kv_fast_fusion.fast_fusion_context import patched_set_forward_context

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm.v1.core.sched.scheduler import Scheduler
from kv_fast_fusion.fast_fusion_scheduler import (
    _handle_block_merging_with_counts,
    update_from_output
)

from vllm.v1.engine.core import EngineCore
from kv_fast_fusion.fast_fusion_core import _initialize_kv_caches

from types import MethodType

from vllm.attention.layer import Attention

# from vllm.model_executor.custom_op import CustomOp  
from kv_fast_fusion.fast_fusion_layer import patched_unified_attention_with_output #, patched_forward as patched_layer_forward
# from vllm.config import CompilationConfig
# import vllm.attention.layer as attn_layer
import torch

from vllm.compilation import fusion_attn  
# import vllm.v1.outputs as outputs_module  
from vllm.logger import init_logger

from vllm.v1.core.block_pool import BlockPool
from kv_fast_fusion.fast_fusion_block_pool import patched_free_blocks  

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from kv_fast_fusion.fast_fustion_flash import patched_forward as patched_flash_forward 
logger = init_logger("vllm.fast_fusion_patch")

DO_COMPRESSION = True

from kv_fast_fusion.kv_fast_fusion_runner import (
    BlockCompressionHook,
    BlockCompressionHookSync,
    _update_block_tables_after_compression,
    execute_model,
    sample_tokens,    
    _dummy_run,
    # ExecuteModelState,
    # PatchedModelRunnerOutput
)

def apply_fast_fusion_patch():
    """Apply the fast fusion patch by registering the necessary hooks."""
    forward_context_module.set_forward_context = patched_set_forward_context
    

    # Patch GPUModelRunner to add compression hook and custom model execution
    original_gpu_model_runner_init = GPUModelRunner.__init__  
  
    def patched_gpu_model_runner_init(self, *args, **kwargs):  
        self.execute_model = MethodType(execute_model, self)
        self._update_block_tables_after_compression = MethodType(_update_block_tables_after_compression, self)         
        self.sample_tokens = MethodType(sample_tokens, self)
        self._dummy_run = MethodType(_dummy_run, self)       
        self.compression_hook = None  
        if DO_COMPRESSION:  
            vllm_config = args[0] if args else None
            self.compression_hook = BlockCompressionHookSync(vllm_config)
            self._updated_block_tables = None
            # self.hook_register = None
        original_gpu_model_runner_init(self, *args, **kwargs)  
  
    GPUModelRunner.__init__ = patched_gpu_model_runner_init  
        
    # Patch Scheduler to handle block merging and update from output    
    Scheduler._handle_block_merging_with_counts = _handle_block_merging_with_counts
    Scheduler.update_from_output = update_from_output

    #Patch BlockPool to use the new free_blocks with deduplication
    BlockPool.free_blocks = patched_free_blocks
    
     # Store the original op structure  
    original_op = torch.ops.vllm.unified_attention_with_output  
    original_default = original_op.default  
      
    # # Create a simple wrapper for the default method  
    # def patched_default(*args, **kwargs):  
    #     logger.info(f"🔥 PATCHED: unified_attention_with_output.default called")  
    #     return original_default(*args, **kwargs)  
    
    fusion_attn.ATTN_OP = original_default
    # Replace the op while maintaining its structure  
    ## for eager mode
    torch.ops.vllm.unified_attention_with_output = patched_unified_attention_with_output

    # Patch FlashAttentionBackend to use the new forward method
    ### for graph mode
    # FlashAttentionImpl.forward = patched_flash_forward

    
    # Patch EngineCore for custom KV cache initialization
    EngineCore._initialize_kv_caches = _initialize_kv_caches


    
    
    
    