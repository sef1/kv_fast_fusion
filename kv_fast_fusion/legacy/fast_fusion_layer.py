import torch
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.model_executor.custom_op import CustomOp
from vllm.attention.layer import Attention
# from vllm.model_executor.custom_op import CustomOp
# @CustomOp.register("fast_fusion_attention") 

@CustomOp.register_oot(name="Attention")  
class FastFusionAttention(Attention):  
        # def forward(self, query, key, value, output_shape=None):  

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """

         # Add this at the very beginning  
        print(f"🔥 FAST_FUSION_PATCH: Running patched_forward for layer {self.layer_name}")  
        import traceback  
        print("Call stack:", traceback.extract_stack()[-3].filename, traceback.extract_stack()[-3].lineno)  
        
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
        output_dtype = query.dtype
        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}

            # check if query quantization is supported
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            if output_shape is None:
                # Handle both 2D [num_tokens, hidden] and
                # 3D [num_tokens, heads, head_dim] query
                num_tokens = query.shape[0]
                output_shape = torch.Size(
                    (num_tokens, self.num_heads * self.head_size_v)
                )
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            hidden_size = output_shape[-1]
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size_v)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size_v)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(
                    self, query, key, value, self_kv_cache, attn_metadata, output=output
                )
            else:
                torch.ops.vllm.patched_unified_attention_with_output(  #sefi
                    query, key, value, output, self.layer_name
                )
            return output.view(-1, hidden_size)
        else:
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                return self.impl.forward(
                    self, query, key, value, self_kv_cache, attn_metadata
                )
            else:
                return torch.ops.vllm.unified_attention(
                    query, key, value, self.layer_name
                )

# from vllm.attention.layer import unified_attention_with_output, unified_attention_with_output_fake  
# from vllm.utils.torch_utils import direct_register_custom_op 

from vllm.logger import init_logger
logger = init_logger("vllm.patched_attn_layer")


# Store the original  
# original_unified_attention_with_output = unified_attention_with_output  
original_op = torch.ops.vllm.unified_attention_with_output

from vllm.attention.layer import get_attention_context
@maybe_transfer_kv_layer
def patched_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    
    ### sefi
    # forward_context: ForwardContext = get_forward_context()
    sf_k = None
    if hasattr(attn_metadata, 'sf_k'):
            sf_k = attn_metadata.sf_k
        # Direct lookup without intermediate attribute assignment  
        # layer_factors = scaling_factors[0].get(layer_name)  
        # if layer_factors is not None:  
            # Extract all needed data at once  
            # sf_k = layer_factors  
            sf_v = attn_metadata.sf_v
            idx = attn_metadata.req_idx
            
            # Ensure 2D shape in-place if needed (more efficient than unsqueeze_)  
            if sf_k.ndim == 1:  
                sf_k = sf_k[None, :]  # Creates view instead of copy  
                sf_v = sf_v[None, :]  
            
            # Apply scaling directly  
            key_cache, value_cache = kv_cache.unbind(0)  
            
            # Combine unsqueeze operations and apply scaling  
            sf_k_expanded = sf_k.view(*sf_k.shape, 1, 1, 1)  # More efficient than multiple unsqueeze  
            sf_v_expanded = sf_v.view(*sf_v.shape, 1, 1, 1)  
            
            # Index and scale in one operation  
            bt_idx = attn_metadata.block_table[idx, :sf_k.shape[1]]  
            key_cache[bt_idx] *= sf_k_expanded  
            value_cache[bt_idx] *= sf_v_expanded
   
    ####

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )

    ###sefi
    if sf_k is not None:            
        key_cache[bt_idx] /= sf_k_expanded
        value_cache[bt_idx] /= sf_v_expanded

    forward_context = get_forward_context() 
    compression_hook = forward_context.additional_kwargs.get('compression_hook')
    if compression_hook: # and forward_context.compression_hook is not None:
    # # Get the attention layer and KV cache  
        compression_hook.start_layer_compression(  
            layer_name,  
            kv_cache,  
            attn_metadata  
        )    
    ##



      
# Store the original op structure  
# original_op = torch.ops.vllm.unified_attention_with_output  
      
    # Create a wrapper that maintains the op structure  
# class PatchedOp:  
#     def __getattr__(self, name):  
#         original_attr = getattr(original_op, name)  
#         if name == 'default':  
#             def patched_default(*args, **kwargs):  
#                 logger.info(f"🔥 PATCHED: unified_attention_with_output.default called")  
#                 return original_attr(*args, **kwargs)  
#             return patched_default  
#         return original_attr  
    

# direct_register_custom_op(
#     op_name="unified_attention_with_output",
#     op_func=patched_unified_attention_with_output,
#     mutates_args=["output", "output_block_scale"],
#     fake_impl=unified_attention_with_output_fake,
# )
