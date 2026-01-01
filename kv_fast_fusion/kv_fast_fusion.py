import torch
import torch.nn.functional as F
from math import log2, floor
import os
from vllm.sequence import IntermediateTensors
from vllm.forward_context import set_forward_context
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.core.sched.output import SchedulerOutput


import vllm.envs as envs
from vllm.utils.math_utils import cdiv, round_up
# from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from math import floor, log2

from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, ExecuteModelState

from types import MethodType
import time
# from vllm.sequence import ExecuteModelRequest, IntermediateTensors
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, overload
from vllm.logger import init_logger
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import json
import numpy as np
logger = init_logger("vllm.vllm_patch")

from abc import ABC, abstractmethod  
from typing import Any  
import torch  

from copy import deepcopy

THRESHOLD = 0.75
BLOCK_SIZE = 128
NUM_LAST_CHUNKS_TO_COMPRESS = 4 
CHUNK_SIZE = 128
# VLLM_USE_V1  = (os.environ.get('VLLM_USE_V1') == '1')


class CompressionHook(ABC):  
    @abstractmethod  
    def start_layer_compression(  
        self,   
        layer_name: str,  
        kv_cache: torch.Tensor,  
        attn_metadata: Any,  
        **kwargs: Any  
    ) -> None:  
        """Start async compression for a specific layer."""  
        pass  
      
    @abstractmethod  
    def wait_for_layer_compression(self, layer_name: str) -> None:  
        """Wait for compression to complete for a layer."""  
        pass

def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)

def _apply_inv_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos + x2 * sin
    o2 = x2 * cos - x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class BlockCompressionHook(CompressionHook):  

    def __init__(self, vllm_config):  
        self.vllm_config = vllm_config  
        self.compression_stream = torch.cuda.Stream()
        # self.num_streams = 2  
        # self.compression_streams = [torch.cuda.Stream() for _ in range(self.num_streams)]  
        self.pending_compressions = {}  
        self.layer_stream_map = {}  
        self.compression_events = {}  
        self.fused_requests = [] 
        self.warmup_layers = 2
        self._block_size = BLOCK_SIZE
        self.thr = 0.7
        # self._should_compress_layer = [i for i in range(self.warmup_layers, self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config) - self.warmup_layers)] 
        #[f"model.layers.{i}.self_attn.attn" for i in range(self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config))][2:-2]
      
    def start_layer_compression(self, layer_name, kv_cache, attn_metadata):  
        """Start async compression immediately after layer populates KV cache."""  

        layer_idx = int(layer_name.split('.')[2])  # Extract layer number  

        if layer_idx == 0:
            B, num_blocks = attn_metadata.block_table.shape            
            self.idx__ = torch.arange(B*num_blocks, dtype=torch.int, device=kv_cache.device)
            self.b_idx = torch.arange(B, dtype=torch.int, device=kv_cache.device).tolist()
            return

        if layer_idx < self.warmup_layers or \
            layer_idx >= (self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config) - self.warmup_layers): #not in self._should_compress_layer:          
            return
          
        with torch.cuda.stream(self.compression_stream):  
            # Launch async compression for this specific layer  
            event = torch.cuda.Event()  
            self.layer_fast_fusion(layer_name, kv_cache, attn_metadata)
            event.record(self.compression_stream)  
            self.compression_events[layer_name] = event  
      
    def wait_for_layer_compression(self, layer_name):  
        """Wait for specific layer compression to complete."""  
        if layer_name in self.compression_events:  
            self.compression_events[layer_name].synchronize()  
            del self.compression_events[layer_name]  
      
    def wait_for_all_compressions(self):  
        """Wait for all pending compressions before next forward pass."""  
        for event in self.compression_events.values():  
            event.synchronize()  
        self.compression_events.clear()
    
    def layer_fast_fusion(self, layer_name, kv_layer, attention_metadata ):
           
        def fuse_all_above_thr(x, b_idx=None, thr=self.thr):
            """
            Recursively processes input tensor `x` to filter and combine elements based on a threshold `thr`.
            Args:
                x (torch.Tensor): A 3D tensor of shape (B, N, D), where B is the batch size, 
                    N is the number of elements per batch, and D is the feature dimension.
                b_idx (torch.Tensor, optional): A tensor containing batch indices. Defaults to None.
                thr (float): The threshold value used for filtering and combining elements.
            Returns:
                tuple: A tuple containing:
                    - combined_x (torch.Tensor): The processed and normalized tensor after combining elements 
                    based on the threshold.
                    - reverse_idx (list): A list of tensors representing the reverse mapping of indices 
                    after processing.
                    - fl_chain (list): A list of tuples containing intermediate results for debugging or 
                    further processing. Each tuple contains:
                        - b_idx (torch.Tensor): The batch indices.
                        - torch.Tensor: A tensor of shape (2, M) representing matched indices.
                        - list: Indices of unmatched elements in the left tensor.
                        - list: Indices of unmatched elements in the right tensor.
                    - shifts (list): A list of integers representing the cumulative shifts applied to indices 
                    during processing.
            Notes:
                - The function operates recursively, splitting the input tensor `x` into two halves and 
                processing them independently before merging the results.
                - The merging step involves filtering elements based on the threshold `thr` and combining 
                matched elements using their mean.
                - Unmatched elements from both halves are retained and concatenated with the combined results.
                - The function also tracks index mappings and shifts for reconstructing the original structure 
                if needed.
            """

            B, _, _ = x.shape
            
            if B == 1:            
                nz_blocks = x.shape[1] if is_chunks_fusion else (seq_lens[b_idx]//self._block_size).item()            
                return F.normalize(x[:,:nz_blocks], dim=-1, eps=1e-7), [self.idx__[:nz_blocks]], [], [nz_blocks]
                    
            xl, _idx_l, fl_chain, shifts_l = fuse_all_above_thr(x[:B//2],  b_idx[:B//2], thr=thr)
            xr, _idx_r, fr_chain, shifts_r = fuse_all_above_thr(x[B//2:],  b_idx[B//2:], thr=thr)

            nl = xl.shape[1]
            nr = xr.shape[1]
            idx_l = self.idx__[:nl]
            idx_r = self.idx__[:nr]
                
            idx_ll, idx_rr = (xl @ xr.mT > thr).nonzero(as_tuple=True)[1:]
            l_idx, c= torch.unique(idx_ll, return_counts=True)
            r_idx = idx_rr.split(tuple(c.tolist()))
            
            idx_ul = list(set(idx_l.tolist()) - set(idx_ll.tolist()))
            idx_ur = list(set(idx_r.tolist()) - set(idx_rr.tolist()))
            
            n_c = len(l_idx)
            n_ul = len(idx_ul)
            n_ur = len(idx_ur)        
            
            combined_tensors = [torch.cat([xl[:,l_idx[i]].unsqueeze(1),xr[:,r_idx[i]]] , dim=1).mean(1, keepdim=True) for i in range(n_c)]
            if combined_tensors != []:
                combined_x = F.normalize(torch.cat(combined_tensors, dim=1), dim = -1)
                combined_x = torch.cat([combined_x, xl[:,idx_ul], xr[:,idx_ur]], dim=1)
            else:
                combined_x = torch.cat([xl[:,idx_ul], xr[:,idx_ur]], dim=1)

            reverse_idx = torch.empty(nl+nr, device=x.device, dtype=torch.int)
            
            reverse_idx[l_idx.tolist()] = self.idx__[:n_c]
            for i in range(n_c):
                reverse_idx[(r_idx[i]+nl).tolist()] = self.idx__[:n_c][i]

            reverse_idx[idx_ul] = self.idx__[n_c:n_c + n_ul]#torch.arange(n_c, n_c + n_ul, device=idx_.device, dtype=torch.int)
            reverse_idx[list(map(lambda x: x + nl, idx_ur))] = self.idx__[n_c+ n_ul:n_c + n_ul+ n_ur]#torch.arange(n_c + n_ul, n_c + n_ul + n_ur, device=idx_.device, dtype=torch.int)
                    
            max_length = max(len(_idx_l), len(_idx_r))
            if len(_idx_l) < max_length:
                shifts_l += [shifts_l[-1]]*(max_length - len(_idx_l))
                _idx_l += [torch.tensor([], device=xl.device, dtype=torch.int) for _ in range(max_length - len(_idx_l))]
            
            chain = [torch.cat([_idx_l[i], _idx_r[i]+shifts_l[i]], dim=0) for i in range(max_length)]
            reverse_idx = [reverse_idx]
            reverse_idx +=chain
            fl_chain += fr_chain
            fl_chain += [(b_idx, torch.stack([idx_ll, idx_rr], dim = -1), idx_ul, idx_ur)]
            # print(b_idx)
            shifts = list(map(lambda x,y: x + y, shifts_l, shifts_r))
            shifts = [n_c+n_ul+n_ur] + shifts

            del x, xl, xr, idx_ll, idx_rr, l_idx, r_idx, idx_ul, idx_ur, n_c, n_ul, n_ur
            return combined_x, reverse_idx, fl_chain, shifts        
        
        def fuse_values_with_above_thr_idx(v, fwd_idx, b_idx):
            """
            Recursively combines and normalizes tensor blocks based on forward indices.
            This function performs a recursive combination of tensor blocks, normalizing
            the results at each step. It uses forward indices to determine how to combine
            the blocks and handles edge cases where the batch size is reduced to one.
            Args:
                v (torch.Tensor): A 3D tensor of shape (B, N, D), where B is the batch size,
                    N is the number of blocks, and D is the feature dimension.
                fwd_idx (list of tuples): A list of tuples containing forward indices and
                    related metadata for combining tensor blocks. Each tuple contains:
                    - idx_ (torch.Tensor): Indices for combining left and right blocks.
                    - idx_ul (torch.Tensor): Indices for unused left blocks.
                    - idx_ur (torch.Tensor): Indices for unused right blocks.
                b_idx (torch.Tensor): A tensor containing batch indices for the current
                    recursive step.
            Returns:
                torch.Tensor: A normalized tensor after recursively combining and processing
                the input tensor blocks.
            Notes:
                - The function assumes the presence of a global variable `seq_lens` and
                a constant `BLOCK_SIZE` for determining the number of non-zero blocks.
                - The `is_prefill` variable is used to determine whether to process all
                blocks or only a subset based on sequence lengths.
                - The function uses `torch.nn.functional.normalize` for normalization.
            """
            i = 0                     
            def recurssive_combining(v, b_idx):
                nonlocal i
                B,_,_ = v.shape
                if B == 1:                 
                    nz_blocks = v.shape[1] if is_chunks_fusion else (seq_lens[b_idx]//self._block_size).item()
                    return F.normalize(v[:, :nz_blocks], dim=-1, eps=1e-7)
                    # return F.normalize(v, dim=-1)

                vl = recurssive_combining(v[:B//2], b_idx[:B//2])
                vr = recurssive_combining(v[B//2:], b_idx[B//2:])
                
                _, idx_, idx_ul, idx_ur = fwd_idx[i]
                idx_ll, idx_rr = idx_.mT
                l_idx, c= torch.unique(idx_ll, return_counts=True)
                r_idx = idx_rr.split(tuple(c.tolist()))

                # combined_v = F.normalize((vl[:,idx_[:,0] ]+vr[:,idx_[:,1]])*0.5, dim=-1)
                # combined_v = torch.cat([combined_v, vl[:,idx_ul], vr[:,idx_ur]], dim=1)            
                combined_tensors = [torch.cat([vl[:,l_idx[i]].unsqueeze(1),vr[:,r_idx[i]]] , dim=1).mean(1, keepdim=True) for i in range(len(l_idx))]
                if combined_tensors != []:
                    combined_v = F.normalize(torch.cat(combined_tensors, dim=1), dim = -1)
                    combined_v = torch.cat([combined_v, vl[:,idx_ul], vr[:,idx_ur]], dim=1)
                else:
                    combined_v = torch.cat([vl[:,idx_ul], vr[:,idx_ur]], dim=1)

                i+=1
                del v, vl, vr, idx_ll, idx_rr, l_idx, r_idx, idx_ul, idx_ur
                return combined_v                       

            vv = recurssive_combining(v, b_idx)

            return vv     
                
        def restore_cache(x, idx, shape):
            """
            Restores a cache tensor by reshaping and reordering its elements based on the provided indices. 
            This decompresses the KV cache tensor, enabling compression ratio evaluation and accuracy analysis.

            Args:
                x (torch.Tensor): The input tensor to be restored. It is expected to have a shape 
                    compatible with the indices and the target shape.
                idx (list of lists): A list of index lists used to reorder the elements of the tensor. 
                    Each inner list specifies the indices for reordering at a particular step.
                shape (tuple): The target shape of the restored tensor. The third dimension of the shape 
                    is used to define the size of the last dimension of the output tensor.

            Returns:
                torch.Tensor: A tensor with the restored shape and reordered elements based on the 
                provided indices.

            Note:
                - The function assumes that the input tensor `x` and the indices in `idx` are compatible 
                with the target shape.
                - The device and data type of the output tensor match those of the input tensor `x`.
            """
            xx = torch.empty((1, len(idx[-2]), shape[2]), dtype=x.dtype, device=x.device)
            xx[:,:x.shape[1] ] = x 
            for idx_ in idx[:-1]:
                xx[:, :len(idx_)] = xx[:, idx_] 
                
            del x
            return xx
        
        def update_block_table(block_table, fwd_idx, b_idx):
            """
            Updates the block table by recursively combining blocks based on the provided forward index.
            Args:
                block_table (torch.Tensor): A tensor representing the block table. 
                    It is expected to have a shape where the first dimension represents 
                    the number of blocks (B).
                fwd_idx (list): A list of forward indices used to map and combine blocks.
                    Each element in the list is expected to be a tuple containing indices 
                    for combining blocks.
                b_idx (torch.Tensor): A tensor representing the block indices. It is used 
                    to determine the sequence lengths for each block.
            Returns:
                torch.Tensor: A tensor representing the updated block table after recursive 
                combination of blocks. The resulting tensor is squeezed along the first dimension.
            Notes:
                - The function uses a nested recursive helper function `blocks_recurssive_combining` 
                to perform the block combination.
                - The variable `i` is used as a nonlocal counter to track the current forward index 
                during the recursive process.
                - The sequence lengths for each block are assumed to be divisible by a constant 
                `BLOCK_SIZE`, which is used to calculate the number of non-zero blocks.
                - The function modifies the `br` tensor in-place during the recursive combination.
            """
            i = 0                     
            def blocks_recurssive_combining(bt, b_idx):
                nonlocal i
                B,_, = bt.shape
                if B == 1:
                    # nz_blocks = bt.shape[1] if is_prefill else (seq_lens[b_idx]//BLOCK_SIZE).item()
                    nz_blocks = (seq_lens[b_idx]//self._block_size).item()
                    return bt[:,:nz_blocks]

                bl = blocks_recurssive_combining(bt[:B//2], b_idx[:B//2])
                br = blocks_recurssive_combining(bt[B//2:], b_idx[B//2:])
            
                idx_ = fwd_idx[i][1]
                br.view(1,-1)[:,idx_[:,1]] = bl.view(1,-1)[:, idx_[:,0]]
                
                i+=1

                return torch.cat([bl,br], dim=-1)
            bt = blocks_recurssive_combining(block_table, b_idx)
            return bt.squeeze(0)
        
        thr = self.thr
        block_table = attention_metadata.block_table
        seq_lens = attention_metadata.seq_lens
        num_last_chunks_to_compress = 4
        is_chunks_fusion = False
        
        device = kv_layer.device
        B, num_blocks = block_table.shape
        
        mask = self.idx__[:num_blocks].repeat(B,1) < (seq_lens//self._block_size).unsqueeze(-1)
        block_table = block_table.to(device)
        kv_shape =kv_layer[0, block_table[mask]].shape
        blocks, block_sz, num_head, head_size =  kv_shape
        if blocks == 0:
            return   
        CHUNK_SIZE = 2**floor(log2(blocks*block_sz))
        blocks_to_keep = CHUNK_SIZE//block_sz
        
        compressed_ = []
        total_ = []
            
        kk = kv_layer[0, block_table]

        ######
        # 
        # 
        # if '17' in layer_name:
        #     with open("/data/users/sefi/vllm_logs/fusion_debug/kv_cache_sample.npy", 'ab') as f:
        #         np.save(f, kv_layer[0, block_table[mask]].float().cpu().numpy())
        #     # np.savez(f"/data/users/sefi/vllm_logs/fusion_debug/kv_samples.npz", kk[mask].float().cpu().numpy())   
             
        if is_chunks_fusion:
            kk = kk[mask]
            
            kk_cat = kk[:-blocks_to_keep]
            kk = kk[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            
            k_norms = kk.norm(2,-1)
        else:
            kk = kk.view(B,num_blocks, -1)
            k_norms = kk[mask].norm(2,-1)
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, self.b_idx, thr)

        # kk = restore_cache(_k, _idx, kk.shape)

        # if is_chunks_fusion:
        #     kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
        #     # kk*= k_norms.unsqueeze(-1)
        #     kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

        #     kv_layer[0, block_table[mask]] = (kk).view(kv_shape)
        # else:
        #     kv_layer[0, block_table[mask]] = kk.view(kv_shape) #(kk * k_norms.unsqueeze(-1)).view(kv_shape)

        del kk 

        vv = kv_layer[1, block_table]   

        if is_chunks_fusion:
            vv = vv[mask]

            vv_cat = vv[:-blocks_to_keep]
            vv = vv[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
        
            v_norms = vv.norm(2,-1)
        else:
            vv = vv.view(B,num_blocks, -1)
            v_norms = vv[mask].norm(2,-1)
        
        _v = fuse_values_with_above_thr_idx(vv,fwd_idx, self.b_idx)  
        
        
        update_block_table(block_table, fwd_idx, self.b_idx)
        
        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [num_last_chunks_to_compress*CHUNK_SIZE/self._block_size]
        else:
            total_ +=[blocks]
        
        del vv 
        
        # _total =  torch.tensor(total_)
        # _compressed = torch.tensor(compressed_)

        # if not hasattr(self, 'block_tables'):
        #     self.block_tables = {}
        #     self.slot_maps = {}
        #     self.compression_ratio = {}

        # self.block_tables[layer_name] = bt_clone
        # self.slot_maps[layer_name] = slot_maps
        # self.compression_ratio[layer_name] = (_total.sum().item() / _compressed.sum().item()) if _compressed.sum().item() > 0 else 0.0
        # compression_ratio = (_total.sum().item() / _compressed.sum().item()) if _compressed.sum().item() > 0 else 0.0 #, (_k, _v, _idx, k_norms, v_norms)
        # logger.info(f"Compression ratio: {compression_ratio}")

        # return _total.sum().item()/_compressed.sum().item(), _total, _compressed, bt_clone, slot_maps    
    

def _update_block_tables_after_compression(  
    self,   
    new_requests: list[str]  
) -> dict[str, dict[int, list[int]]]:  
    """Return final block lists instead of mappings."""  
    request_blocks = {}  
    for req_id in new_requests:  
        req_idx = self.input_batch.req_id_to_index[req_id]  
        req_state = self.requests[req_id]  
        
        for group_idx in range(len(self.input_batch.block_table.block_tables)):  
            if group_idx == 0:  
                continue  
                
            block_table_obj = self.input_batch.block_table.block_tables[group_idx]  
            new_table = block_table_obj.block_table[req_idx]#block_tables_[group_idx][req_idx]  
            num_blocks = block_table_obj.num_blocks_per_row[req_idx]    #sefi tset
            # Update state and block table  
            
            final_blocks = new_table[:num_blocks].tolist()
            if all(block_table_obj.block_table_np[req_idx, :num_blocks] == final_blocks) :
                continue
            if req_id not in request_blocks.keys():
                request_blocks[req_id] = {}
            req_state.block_ids[group_idx][:num_blocks] = final_blocks
            block_table_obj.block_table_np[req_idx, :num_blocks] = new_table[:num_blocks].cpu().numpy()  
            # block_table_obj.num_blocks_per_row[req_idx] = num_blocks  
            
            # Store final block list  
            request_blocks[req_id][group_idx] = final_blocks  
  
    return request_blocks


@torch.inference_mode()
def fast_fusion(kv_cache, block_tables, thr, is_chunks_fusion, num_last_chunks_to_compress=2, rotary_emb = None, seq_lens = None, restore_kv = True):
   
    def fuse_all_above_thr(x, b_idx=None, thr=thr):
        """
        Recursively processes input tensor `x` to filter and combine elements based on a threshold `thr`.
        Args:
            x (torch.Tensor): A 3D tensor of shape (B, N, D), where B is the batch size, 
                N is the number of elements per batch, and D is the feature dimension.
            b_idx (torch.Tensor, optional): A tensor containing batch indices. Defaults to None.
            thr (float): The threshold value used for filtering and combining elements.
        Returns:
            tuple: A tuple containing:
                - combined_x (torch.Tensor): The processed and normalized tensor after combining elements 
                  based on the threshold.
                - reverse_idx (list): A list of tensors representing the reverse mapping of indices 
                  after processing.
                - fl_chain (list): A list of tuples containing intermediate results for debugging or 
                  further processing. Each tuple contains:
                    - b_idx (torch.Tensor): The batch indices.
                    - torch.Tensor: A tensor of shape (2, M) representing matched indices.
                    - list: Indices of unmatched elements in the left tensor.
                    - list: Indices of unmatched elements in the right tensor.
                - shifts (list): A list of integers representing the cumulative shifts applied to indices 
                  during processing.
        Notes:
            - The function operates recursively, splitting the input tensor `x` into two halves and 
              processing them independently before merging the results.
            - The merging step involves filtering elements based on the threshold `thr` and combining 
              matched elements using their mean.
            - Unmatched elements from both halves are retained and concatenated with the combined results.
            - The function also tracks index mappings and shifts for reconstructing the original structure 
              if needed.
        """

        B, _, _ = x.shape
        
        if B == 1:            
            nz_blocks = x.shape[1] if is_chunks_fusion else (seq_lens[b_idx]//BLOCK_SIZE).item()            
            return F.normalize(x[:,:nz_blocks], dim=-1, eps=1e-6), [idx__[:nz_blocks]], [], [nz_blocks]
                
        xl, _idx_l, fl_chain, shifts_l = fuse_all_above_thr(x[:B//2],  b_idx[:B//2], thr=thr)
        xr, _idx_r, fr_chain, shifts_r = fuse_all_above_thr(x[B//2:],  b_idx[B//2:], thr=thr)

        nl = xl.shape[1]
        nr = xr.shape[1]
        idx_l = idx__[:nl]
        idx_r = idx__[:nr]
               
        idx_ll, idx_rr = (xl @ xr.mT > thr).nonzero(as_tuple=True)[1:]
        l_idx, c= torch.unique(idx_ll, return_counts=True)
        r_idx = idx_rr.split(tuple(c.tolist()))
        
        idx_ul = list(set(idx_l.tolist()) - set(idx_ll.tolist()))
        idx_ur = list(set(idx_r.tolist()) - set(idx_rr.tolist()))
        
        n_c = len(l_idx)
        n_ul = len(idx_ul)
        n_ur = len(idx_ur)        
        
        combined_tensors = [torch.cat([xl[:,l_idx[i]].unsqueeze(1),xr[:,r_idx[i]]] , dim=1).mean(1, keepdim=True) for i in range(n_c)]
        if combined_tensors != []:
            combined_x = F.normalize(torch.cat(combined_tensors, dim=1), dim = -1, eps=1e-6)
            combined_x = torch.cat([combined_x, xl[:,idx_ul], xr[:,idx_ur]], dim=1)
        else:
            combined_x = torch.cat([xl[:,idx_ul], xr[:,idx_ur]], dim=1)

        reverse_idx = torch.empty(nl+nr, device=x.device, dtype=torch.int)
        
        reverse_idx[l_idx.tolist()] = idx__[:n_c]
        for i in range(n_c):
            reverse_idx[(r_idx[i]+nl).tolist()] = idx__[:n_c][i]

        reverse_idx[idx_ul] = idx__[n_c:n_c + n_ul]#torch.arange(n_c, n_c + n_ul, device=idx_.device, dtype=torch.int)
        reverse_idx[list(map(lambda x: x + nl, idx_ur))] = idx__[n_c+ n_ul:n_c + n_ul+ n_ur]#torch.arange(n_c + n_ul, n_c + n_ul + n_ur, device=idx_.device, dtype=torch.int)
                
        max_length = max(len(_idx_l), len(_idx_r))
        if len(_idx_l) < max_length:
            shifts_l += [shifts_l[-1]]*(max_length - len(_idx_l))
            _idx_l += [torch.tensor([], device=xl.device, dtype=torch.int) for _ in range(max_length - len(_idx_l))]
        
        chain = [torch.cat([_idx_l[i], _idx_r[i]+shifts_l[i]], dim=0) for i in range(max_length)]
        reverse_idx = [reverse_idx]
        reverse_idx +=chain
        fl_chain += fr_chain
        fl_chain += [(b_idx, torch.stack([idx_ll, idx_rr], dim = -1), idx_ul, idx_ur)]
        # print(b_idx)
        shifts = list(map(lambda x,y: x + y, shifts_l, shifts_r))
        shifts = [n_c+n_ul+n_ur] + shifts
        
        return combined_x, reverse_idx, fl_chain, shifts        
    
    def fuse_values_with_above_thr_idx(v, fwd_idx, b_idx):
        """
        Recursively combines and normalizes tensor blocks based on forward indices.
        This function performs a recursive combination of tensor blocks, normalizing
        the results at each step. It uses forward indices to determine how to combine
        the blocks and handles edge cases where the batch size is reduced to one.
        Args:
            v (torch.Tensor): A 3D tensor of shape (B, N, D), where B is the batch size,
                N is the number of blocks, and D is the feature dimension.
            fwd_idx (list of tuples): A list of tuples containing forward indices and
                related metadata for combining tensor blocks. Each tuple contains:
                - idx_ (torch.Tensor): Indices for combining left and right blocks.
                - idx_ul (torch.Tensor): Indices for unused left blocks.
                - idx_ur (torch.Tensor): Indices for unused right blocks.
            b_idx (torch.Tensor): A tensor containing batch indices for the current
                recursive step.
        Returns:
            torch.Tensor: A normalized tensor after recursively combining and processing
            the input tensor blocks.
        Notes:
            - The function assumes the presence of a global variable `seq_lens` and
              a constant `BLOCK_SIZE` for determining the number of non-zero blocks.
            - The `is_prefill` variable is used to determine whether to process all
              blocks or only a subset based on sequence lengths.
            - The function uses `torch.nn.functional.normalize` for normalization.
        """
        i = 0                     
        def recurssive_combining(v, b_idx):
            nonlocal i
            B,_,_ = v.shape
            if B == 1:                 
                nz_blocks = v.shape[1] if is_chunks_fusion else (seq_lens[b_idx]//BLOCK_SIZE).item()
                return F.normalize(v[:, :nz_blocks], dim=-1, eps=1e-6)
                # return F.normalize(v, dim=-1)

            vl = recurssive_combining(v[:B//2], b_idx[:B//2])
            vr = recurssive_combining(v[B//2:], b_idx[B//2:])
            
            _, idx_, idx_ul, idx_ur = fwd_idx[i]
            idx_ll, idx_rr = idx_.mT
            l_idx, c= torch.unique(idx_ll, return_counts=True)
            r_idx = idx_rr.split(tuple(c.tolist()))

            # combined_v = F.normalize((vl[:,idx_[:,0] ]+vr[:,idx_[:,1]])*0.5, dim=-1)
            # combined_v = torch.cat([combined_v, vl[:,idx_ul], vr[:,idx_ur]], dim=1)            
            combined_tensors = [torch.cat([vl[:,l_idx[i]].unsqueeze(1),vr[:,r_idx[i]]] , dim=1).mean(1, keepdim=True) for i in range(len(l_idx))]
            if combined_tensors != []:
                combined_v = F.normalize(torch.cat(combined_tensors, dim=1), dim = -1, eps=1e-6)
                combined_v = torch.cat([combined_v, vl[:,idx_ul], vr[:,idx_ur]], dim=1)
            else:
                combined_v = torch.cat([vl[:,idx_ul], vr[:,idx_ur]], dim=1)

            i+=1

            return combined_v                       

        vv = recurssive_combining(v, b_idx)

        return vv     
            
    def restore_cache(x, idx, shape):
        """
        Restores a cache tensor by reshaping and reordering its elements based on the provided indices. 
        This decompresses the KV cache tensor, enabling compression ratio evaluation and accuracy analysis.

        Args:
            x (torch.Tensor): The input tensor to be restored. It is expected to have a shape 
                compatible with the indices and the target shape.
            idx (list of lists): A list of index lists used to reorder the elements of the tensor. 
                Each inner list specifies the indices for reordering at a particular step.
            shape (tuple): The target shape of the restored tensor. The third dimension of the shape 
                is used to define the size of the last dimension of the output tensor.

        Returns:
            torch.Tensor: A tensor with the restored shape and reordered elements based on the 
            provided indices.

        Note:
            - The function assumes that the input tensor `x` and the indices in `idx` are compatible 
              with the target shape.
            - The device and data type of the output tensor match those of the input tensor `x`.
        """
        xx = torch.empty((1, len(idx[-2]), shape[2]), dtype=x.dtype, device=x.device)
        xx[:,:x.shape[1] ] = x 
        for idx_ in idx[:-1]:
            xx[:, :len(idx_)] = xx[:, idx_]    

        return xx
    
    def update_block_table(block_table, fwd_idx, b_idx):
        """
        Updates the block table by recursively combining blocks based on the provided forward index.
        Args:
            block_table (torch.Tensor): A tensor representing the block table. 
                It is expected to have a shape where the first dimension represents 
                the number of blocks (B).
            fwd_idx (list): A list of forward indices used to map and combine blocks.
                Each element in the list is expected to be a tuple containing indices 
                for combining blocks.
            b_idx (torch.Tensor): A tensor representing the block indices. It is used 
                to determine the sequence lengths for each block.
        Returns:
            torch.Tensor: A tensor representing the updated block table after recursive 
            combination of blocks. The resulting tensor is squeezed along the first dimension.
        Notes:
            - The function uses a nested recursive helper function `blocks_recurssive_combining` 
              to perform the block combination.
            - The variable `i` is used as a nonlocal counter to track the current forward index 
              during the recursive process.
            - The sequence lengths for each block are assumed to be divisible by a constant 
              `BLOCK_SIZE`, which is used to calculate the number of non-zero blocks.
            - The function modifies the `br` tensor in-place during the recursive combination.
        """
        i = 0                     
        def blocks_recurssive_combining(bt, b_idx):
            nonlocal i
            B,_, = bt.shape
            if B == 1:
                # nz_blocks = bt.shape[1] if is_prefill else (seq_lens[b_idx]//BLOCK_SIZE).item()
                nz_blocks = (seq_lens[b_idx]//BLOCK_SIZE).item()
                return bt[:,:nz_blocks]

            bl = blocks_recurssive_combining(bt[:B//2], b_idx[:B//2])
            br = blocks_recurssive_combining(bt[B//2:], b_idx[B//2:])
           
            idx_ = fwd_idx[i][1]
            br.view(1,-1)[:,idx_[:,1]] = bl.view(1,-1)[:, idx_[:,0]]
            
            i+=1

            return torch.cat([bl,br], dim=-1)
        bt = blocks_recurssive_combining(block_table, b_idx)
        return bt.squeeze(0)
        
    device = kv_cache[0].device
    L = len(kv_cache)
    B, num_blocks = block_tables[0].shape
        
    b_idx = torch.arange(B,device=device).tolist()
    idx__ = torch.arange(B*num_blocks, dtype=torch.int, device=device)
    mask = idx__[:num_blocks].repeat(B,1) < (seq_lens//BLOCK_SIZE).unsqueeze(-1)
    kv_shape =kv_cache[0][0, block_tables[0][mask]].shape
    blocks, block_sz, num_head, head_size =  kv_shape    
    # block_tables_ = {}
    # slot_maps_ = {}
    # CHUNK_SIZE = 2**floor(log2(blocks*block_sz))
    # blocks_to_keep = CHUNK_SIZE//block_sz
    
    compressed_ = []
    total_ = []
    if rotary_emb:
        cos_sin = rotary_emb.cos_sin_cache.index_select(0, torch.arange(block_tables[0].shape[1]*block_sz, device=device))
        cos, sin = cos_sin.chunk(2, dim=-1)
    
    for l in range(L): # 2 warmup layers and 2 final layers
        
        if rotary_emb:
            kv_cache[l][0, block_tables[l]] = _apply_inv_rotary_emb(kv_cache[l][0, block_tables[l]].view(B,-1, num_head, head_size), cos, sin, rotary_emb.is_neox_style).view(B, -1, block_sz, num_head, head_size)
            
        kk = kv_cache[l][0, block_tables[l]]
         
        if is_chunks_fusion:
            kk = kk[mask]
            # CHUNK_SIZE = num_last_chunks_to_compress*block_sz
            chunks = (blocks*block_sz)//CHUNK_SIZE
            # blocks_to_keep = CHUNK_SIZE//block_sz
            # kk_cat = kk[:-blocks_to_keep]
            # kk = kk[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            to_keep = blocks // chunks * chunks 
            res = (blocks % chunks)
            # if res !=0:
            #     cat_chunks = 1
            kk_cat = kk[to_keep:]
            if res !=0:
                kk_cat = kk_cat.view(res, -1)
            kk = kk[:to_keep].view(chunks, blocks//chunks, -1)
            # kk_cat = kk.view(chunks,blocks//chunks, -1)[:-num_last_chunks_to_compress]
            # kk = kk.view(chunks,blocks//chunks, -1)[-num_last_chunks_to_compress:]
            k_norms = kk.norm(2,-1)
        else:
            kk = kk.view(B,num_blocks, -1)
            k_norms = kk[mask].norm(2,-1)
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, b_idx, thr)

        # if not VLLM_USE_V1
        if restore_kv:
            kk = restore_cache(_k, _idx, kk.shape)

            if is_chunks_fusion:
                kk =kk.view(chunks,blocks//chunks, -1)
                # kk =kk.view(num_last_chunks_to_compress,blocks//chunks, -1)
                # kk =kk.view(blocks - num_last_chunks_to_compress,blocks//chunks, -1)
                # kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
                kk*= k_norms.unsqueeze(-1)
                if res !=0:
                    kk = torch.cat([kk.view(to_keep, -1), kk_cat], dim=0)
                else:
                    kk = kk.view(to_keep, -1)
                # kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)
                kv_cache[l][0, block_tables[l][mask]] = (kk).view(kv_shape)
            else:
                kv_cache[l][0, block_tables[l][mask]] = (kk * k_norms.unsqueeze(-1)).view(kv_shape)

            if rotary_emb:
                kv_cache[l][0, block_tables[l]] = _apply_rotary_emb(kv_cache[l][0, block_tables[l]].view(B,-1, num_head, head_size), cos, sin, rotary_emb.is_neox_style).view(B, -1, block_sz, num_head, head_size)
            
        del _k, k_norms, kk

        vv = kv_cache[l][1, block_tables[l]]   

        if is_chunks_fusion:
            vv = vv[mask]
            chunks = (blocks*block_sz)//CHUNK_SIZE

            to_keep = blocks // chunks * chunks 
            res = (blocks % chunks)
            vv_cat = vv[to_keep:]
            if res !=0:
                vv_cat = vv_cat.view(res, -1)
            vv = vv[:to_keep].view(chunks, blocks//chunks, -1)
        
            # vv_cat = vv.view(chunks,blocks//chunks, -1)[:-num_last_chunks_to_compress]
            # vv = vv.view(chunks,blocks//chunks, -1)[-num_last_chunks_to_compress:]
            # vv_cat = vv[:-blocks_to_keep]
            # vv = vv[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            v_norms = vv.norm(2,-1)
        else:
            vv = vv.view(B,num_blocks, -1)
            v_norms = vv[mask].norm(2,-1)
        
        _v = fuse_values_with_above_thr_idx(vv,fwd_idx, b_idx)  
       
        if restore_kv:
            vv = restore_cache(_v, _idx, vv.shape) 
            
            if is_chunks_fusion:
                vv =vv.view(chunks,blocks//chunks, -1)
                # vv =vv.view(num_last_chunks_to_compress,blocks//chunks, -1)
                # vv =vv.view(blocks - num_last_chunks_to_compress,blocks//chunks, -1)
                # vv =vv.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
                vv*= v_norms.unsqueeze(-1)
                if res !=0:
                    vv = torch.cat([vv.view(to_keep, -1), vv_cat], dim=0)
                else:
                    vv = vv.view(to_keep, -1)
                # vv = torch.cat([vv_cat,vv.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)
                kv_cache[l][1, block_tables[l][mask]] = (vv).view(kv_shape)
            else:
                kv_cache[l][1, block_tables[l][mask]] = (vv * v_norms.unsqueeze(-1)).view(kv_shape)
        
        # bt_clone = block_tables[l].clone()
        # bt_clone[mask] = update_block_table(block_tables[l], fwd_idx, b_idx)
        # block_tables_[l] = bt_clone 
        if not restore_kv:
            update_block_table(block_tables[l], fwd_idx, b_idx)
        
        # blocks_idx = seq_lens//BLOCK_SIZE +  ((seq_lens % BLOCK_SIZE) > 0)
        # block_offsets = torch.arange(0, BLOCK_SIZE, device=seq_lens.device)
        # slot_maps = [(block_offsets.reshape((1, self._block_size)).to("cuda:0") + block_table[i, :nb].reshape(nb,1)*self._block_size).flatten()[:seq_lens[i]] for i,nb in enumerate(num_blocks)]
        # slot_maps_[l] = {f"{seq_lens[i]}":(block_offsets.reshape((1, BLOCK_SIZE)).to(vv.device) + block_tables_[l][i, :nb].reshape(nb,1)*BLOCK_SIZE).flatten()[:seq_lens[i]] for i,nb in enumerate(blocks_idx)}
        # slot_maps_[l] = torch.cat([(block_offsets.reshape((1, BLOCK_SIZE)).to(vv.device) + block_tables_[l][i, :nb].reshape(nb,1)*BLOCK_SIZE).flatten()[:seq_lens[i]] for i,nb in enumerate(blocks_idx)]).unique()
        
        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [to_keep]#[num_last_chunks_to_compress*CHUNK_SIZE/BLOCK_SIZE]
        else:
            total_ +=[blocks]
        del _v, v_norms, vv
    
    _total =  torch.tensor(total_)
    _compressed = torch.tensor(compressed_)
    return _total.sum().item()/_compressed.sum().item(), _total, _compressed, block_tables #, slot_maps_    

@torch.inference_mode()
def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if (
            self.use_async_scheduling
            and self.num_spec_tokens
            and self._draft_token_ids is None
        ):
            scheduler_output = deepcopy(scheduler_output)

        # is_decode = all(scheduler_output.num_scheduled_tokens[req_id] == 1 for req_id in self.input_batch.req_ids)
        # req_ids = scheduler_output.scheduled_cached_reqs.req_ids  
        # tokens = [scheduler_output.num_output_tokens[i] == 1 for i in req_ids]  
        # num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)  
  
        # Check if all are prefill (more than 1 token)  
        # is_decode = all([t > 1 for t in scheduler_output.scheduled_cached_reqs.num_output_tokens])

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("gpu_model_runner: preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        return make_empty_encoder_model_runner_output(scheduler_output)

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend
                        == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config
                    )
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

                (
                    logits_indices,
                    spec_decode_metadata,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = (
                    batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                )
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                logger.debug(
                    "ubatch_slices: %s, ubatch_slices_padded: %s",
                    ubatch_slices,
                    ubatch_slices_padded,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                (attn_metadata, spec_decode_common_attn_metadata) = (
                    self._build_attention_metadata(
                        num_tokens=num_tokens_unpadded,
                        num_tokens_padded=num_tokens_padded if pad_attn else None,
                        num_reqs=num_reqs,
                        num_reqs_padded=num_reqs_padded if pad_attn else None,
                        max_query_len=max_num_scheduled_tokens,
                        ubatch_slices=ubatch_slices_attn,
                        logits_indices=logits_indices,
                        use_spec_decode=use_spec_decode,
                        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    )
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output, num_tokens_padded, intermediate_tensors
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )            

        ########################## sefi
        if True:
            bs = self.input_batch.num_reqs
            # is_chunks_fusion = True
            if not hasattr(self, 'fused_requests'):
                self.fused_requests = []
            # req_to_compress = [req_id not in self.fused_requests for req_id in self.input_batch.req_ids]
            req_to_compress = [r_id for r_id,t in zip(scheduler_output.scheduled_cached_reqs.req_ids,scheduler_output.scheduled_cached_reqs.num_output_tokens) if t == 1 and r_id not in self.fused_requests]
            
            if len(req_to_compress) > 1: # and not is_decode:
                is_chunks_fusion = False
                logger.info(f"bff dealing with batch size %s ", len(req_to_compress))
                dict_key = next(iter(attn_metadata))
                req_idx = [self.input_batch.req_id_to_index[r] for r in req_to_compress]
                block_tables = [attn_metadata[k].block_table[req_idx] for k  in attn_metadata.keys()] #attn_metadata[dict_key].block_table
                seq_lens = attn_metadata[dict_key].seq_lens[req_idx]
                # NUM_LAST_CHUNKS_TO_COMPRESS = 4
                remove_position = False
                # m = self.model_runner.__dict__['model']
                rotary_emb = None ##self.model.model.layers[0].self_attn.rotary_emb
                # call 
                restore_kv = True
                cr, total, num_comp, block_tables_  = fast_fusion(self.kv_caches[2:-2], block_tables[4:],  THRESHOLD, is_chunks_fusion, NUM_LAST_CHUNKS_TO_COMPRESS, rotary_emb, seq_lens=seq_lens, restore_kv=restore_kv) 
                logger.info(f"compression {cr} | per layer compression {total/num_comp}")
                logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
                if not os.path.exists(f"compression_res"):
                    os.makedirs(f"compression_res")
                with open(f"compression_res/bff_thr_{THRESHOLD}_stress.jsonl", "a", encoding="utf-8") as f:
                    json.dump({"cr": cr, "per_layer": str((total/num_comp).tolist()), "num_comp_": str((num_comp).tolist())}, f, ensure_ascii=False)
                    f.write('\n')
                    # pd.read_json("/data/users/sefi/from_git/vllm_013/vllm/compression_res/cff_4_thr_0.8.jsonl", lines = True)              
 
                # Your logic to identify blocks to free  
                # new_requests = [req_id for req_id in self.input_batch.req_ids if req_id not in self.fused_requests]
                self.fused_requests.append(req_to_compress)
                # compressed_layers = total / num_comp > 1
                if cr > 1 and not restore_kv:  
                    updated_block_tables = _update_block_tables_after_compression(self, req_to_compress, block_tables_)
            # do cff
            elif len(req_to_compress) == 1: # and not is_decode:
                is_chunks_fusion = True
                if True: #hasattr(self, 'old_seq_len'):
                        # self.apply_cff_every_ = NUM_LAST_CHUNKS_TO_COMPRESS
                    dict_key = next(iter(attn_metadata))
                    req_idx = [self.input_batch.req_id_to_index[r] for r in req_to_compress]
                    seq_lens = attn_metadata[dict_key].seq_lens[req_idx]
                    if True: #seq_lens.item() - self.old_seq_len >= NUM_LAST_CHUNKS_TO_COMPRESS*CHUNK_SIZE:

                        # logger.info(f"{"D" if is_decode else "P"} dealing with batch size %s ", sum(req_to_compress))
                        
                        block_tables = [attn_metadata[k].block_table[req_idx] for k  in attn_metadata.keys()] #attn_metadata[dict_key].block_table
                        
                        # NUM_LAST_CHUNKS_TO_COMPRESS = 4
                        remove_position = False
                        # m = self.model_runner.__dict__['model']
                        rotary_emb = None ##self.model.model.layers[0].self_attn.rotary_emb
                        # call 
                        restore_kv = True
                        cr, total, num_comp, block_tables_  = fast_fusion(self.kv_caches[2:-2], block_tables[4:],  THRESHOLD, is_chunks_fusion, NUM_LAST_CHUNKS_TO_COMPRESS, rotary_emb, seq_lens=seq_lens, restore_kv=restore_kv) 
                        logger.info(f"cff compression {cr} | per layer compression {total/num_comp}")
                        logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
                        self.old_seq_len = seq_lens.item()
                        if not os.path.exists(f"compression_res"):
                            os.makedirs(f"compression_res")
                        with open(f"compression_res/cff_{NUM_LAST_CHUNKS_TO_COMPRESS}_thr_{THRESHOLD}.jsonl", "a", encoding="utf-8") as f:
                            json.dump({"cr": cr, "per_layer": str((total/num_comp).tolist()), "num_comp_": str((num_comp).tolist())}, f, ensure_ascii=False)
                            f.write('\n')
                            # pd.read_json("/data/users/sefi/from_git/vllm_013/vllm/compression_res/cff_4_thr_0.7.jsonl", lines = True)              

                        # Your logic to identify blocks to free  
                        # new_requests = [req_id for req_id in self.input_batch.req_ids if req_id not in self.fused_requests]
                        # self.fused_requests.append(new_requests)
                        # compressed_layers = total / num_comp > 1
                        if cr > 1 and not restore_kv:  
                            updated_block_tables = _update_block_tables_after_compression(self, new_requests, block_tables_)
                else:
                    self.old_seq_len = 0

            ######################
        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                    )
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_tokens_padded
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            # updated_block_tables,
        )
        self.kv_connector_output = kv_connector_output
        return None



