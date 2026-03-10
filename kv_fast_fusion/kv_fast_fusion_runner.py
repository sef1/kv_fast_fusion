import torch
import os

# import vllm.envs as envs

from math import floor, log2
from kv_fast_fusion.compression_hook import CompressionHook

import torch.nn.functional as F
from typing import Any, Optional, Union, TYPE_CHECKING, NamedTuple
from vllm.logger import init_logger
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import json
import numpy as np
logger = init_logger("vllm.vllm_patch")

# from vllm.config import get_layers_from_vllm_config  
# from vllm.attention.layer import Attention 

THRESHOLD = 0.75
BLOCK_SIZE = 128
NUM_LAST_CHUNKS_TO_COMPRESS = 4 
CHUNK_SIZE = 512
VLLM_USE_V1  = (os.environ.get('VLLM_USE_V1') == '1')

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
        self.warmup_layers = 2
        self.pending_compressions = {}  
        self.layer_stream_map = {}  
        self.compression_events = {}  
        self.norms_k = {}
        self.norms_v = {}
        self.block_tables = {}
        self.fused_requests = [] 
        
        self.req_idx = []
        self.reqs = []
        self._block_size = BLOCK_SIZE
        self.thr = 0.5
        # self._should_compress_layer = [i for i in range(self.warmup_layers, self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config) - self.warmup_layers)] 
        #[f"model.layers.{i}.self_attn.attn" for i in range(self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config))][2:-2]
      
    # @torch.compiler.disable
    def start_layer_compression(self, layer_name, kv_cache, attn_metadata):  
        """Start async compression immediately after layer populates KV cache."""  

        layer_idx = int(layer_name.split('.')[2])  # Extract layer number  

        if layer_idx == 0:
            B, num_blocks = attn_metadata.block_table[self.req_idx].shape            
            self.idx__ = torch.arange(B*num_blocks, dtype=torch.int, device=kv_cache.device)
            self.b_idx = torch.arange(B, dtype=torch.int, device=kv_cache.device).tolist()
            return

        if layer_idx < self.warmup_layers or \
            layer_idx >= (self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config) - self.warmup_layers): #not in self._should_compress_layer:          
            return
        
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.compression_stream):  
            # Ensure the attention kernel on the default stream is finished before we read the KV cache.
            self.compression_stream.wait_stream(default_stream)
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
        block_table = attention_metadata.block_table[self.req_idx]
        seq_lens = attention_metadata.seq_lens[self.req_idx]
        num_last_chunks_to_compress = 4
        is_chunks_fusion = False
        
        device = kv_layer.device
        B, num_blocks = block_table.shape
        
        mask = self.idx__[:num_blocks].repeat(B,1) < (seq_lens//self._block_size).unsqueeze(-1)
        mask_split = mask.sum(-1).tolist()
        max_split_len = max(mask_split)
        block_table = block_table.to(device)
        kv_shape =kv_layer[0, block_table[mask]].shape
        blocks, block_sz, num_head, head_size =  kv_shape
        if blocks == 0:
            return   
        # CHUNK_SIZE = 2**floor(log2(blocks*block_sz))
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
            # self.norms_k[layer_name] = k_norms
        else:
            kk = kk.view(B,num_blocks, -1)
            k_norms = kk[mask].norm(2,-1)
            splits_k = k_norms.split(mask_split)
            # max_len = max([len(s) for s in splits_k])
            # max_split_len = max(mask_split)
            splits_k = [F.pad(s, (0, max_split_len - len(s)), value=1.0) for s in splits_k]
            # for idx, req in enumerate(self.reqs):
            #     if req not in self.norms_k.keys():
            #         self.norms_k[req] = {}
                
            #     self.norms_k[req][layer_name] = splits_k[idx]

            # self.norms_k[layer_name] = k_norms.split(mask_split)
            # self.norms_k[layer_name] = [kk[i, mask[i]].norm(2, -1) for i in range(len(self.req_idx))]
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, self.b_idx, thr)

        kk = restore_cache(_k, _idx, kk.shape)

        if is_chunks_fusion:
            kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[0, block_table[mask]] = (kk).view(kv_shape)
        else:
            kv_layer[0, block_table[mask]] = kk.view(kv_shape) #(kk * k_norms.unsqueeze(-1)).view(kv_shape)

        del kk 

        vv = kv_layer[1, block_table]   

        if is_chunks_fusion:
            vv = vv[mask]

            vv_cat = vv[:-blocks_to_keep]
            vv = vv[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
        
            v_norms = vv.norm(2,-1)
            self.norms_v[layer_name] = v_norms
        else:
            vv = vv.view(B,num_blocks, -1)
            v_norms = vv[mask].norm(2,-1)
            splits_v = v_norms.split(mask_split)
            splits_v = [F.pad(s, (0, max_split_len - len(s)), value=1.0) for s in splits_v]
            # for idx in range(len(self.req_idx)):
            #     if f"{self.reqs[idx]}" not in self.norms_v.keys():
            #         self.norms_v[f"{self.reqs[idx]}"] = {}
                
            #     self.norms_v[f"{self.reqs[idx]}"][layer_name] = splits_v[idx]
            # self.norms_v[layer_name] = [vv[i, mask[i]].norm(2, -1) for i in range(len(self.req_idx))]

        _v = fuse_values_with_above_thr_idx(vv,fwd_idx, self.b_idx)  
        
        vv = restore_cache(_v, _idx, vv.shape)

        if is_chunks_fusion:
            vv =vv.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            vv = torch.cat([vv_cat,vv.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[1, block_table[mask]] = (vv).view(kv_shape)
        else:
            kv_layer[1, block_table[mask]] = vv.view(kv_shape) #(vv * v_norms.unsqueeze(-1)).view(kv_shape)

        update_block_table(block_table, fwd_idx, self.b_idx)
        attention_metadata.block_table[self.req_idx] = block_table
        
        # if layer_name not in self.block_tables.keys():
        #     self.block_tables[layer_name] = block_table
        for idx, req in enumerate(self.reqs):
            if req not in self.norms_k.keys():
                self.norms_k[req] = {}
                self.norms_v[req] = {}
                    # self.block_tables[req] = {}
                
            self.norms_k[req][layer_name] = splits_k[idx]
            self.norms_v[req][layer_name] = splits_v[idx]
        
        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [num_last_chunks_to_compress*CHUNK_SIZE/self._block_size]
        else:
            total_ +=[blocks]
        
        del vv 
        logger.info(f"Compression ratio in layer {layer_name}: {torch.tensor(total_).sum().item() / torch.tensor(compressed_).sum().item() if torch.tensor(compressed_).sum().item() > 0 else 0.0}")
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
    
class BlockCompressionHookSync(CompressionHook):  

    def __init__(self, vllm_config):  
        self.vllm_config = vllm_config  
        self.compression_stream = torch.cuda.Stream()
        self.warmup_layers = 2
        self.pending_compressions = {}  
        self.layer_stream_map = {}  
        self.compression_events = {}  
        self.norms_k = {}
        self.norms_v = {}
        self.block_tables = {}
        self.fused_requests = []
        self.req_idx = []
        self.reqs = []
        self._block_size = BLOCK_SIZE
        self.thr = 0.5
      
    def start_layer_compression(self, layer_name, kv_cache, attn_metadata):  
        """Start async compression immediately after layer populates KV cache."""  

        if len(self.req_idx)<2:
            return
        
        layer_idx = int(layer_name.split('.')[2])  # Extract layer number  

        if layer_idx == 0:
            B, num_blocks = attn_metadata.block_table[self.req_idx].shape            
            self.idx__ = torch.arange(B*num_blocks, dtype=torch.int, device=kv_cache.device)
            self.b_idx = torch.arange(B, dtype=torch.int, device=kv_cache.device).tolist()
            return

        if layer_idx < self.warmup_layers or \
            layer_idx >= (self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config) - self.warmup_layers): #not in self._should_compress_layer:          
            return
        
        # Run compression directly on current stream  
        self.layer_fast_fusion(layer_name, kv_cache, attn_metadata) 
      
    def wait_for_layer_compression(self, layer_name):  
        """Wait for specific layer compression to complete."""  
        pass
      
    def wait_for_all_compressions(self):  
        """Wait for all pending compressions before next forward pass."""  
        pass
    
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
        block_table = attention_metadata.block_table[self.req_idx]
        B, num_blocks = block_table.shape
        if B <=1:
            return
        seq_lens = attention_metadata.seq_lens[self.req_idx]
        num_last_chunks_to_compress = 4
        is_chunks_fusion = False
        
        device = kv_layer.device
        
        
        mask = self.idx__[:num_blocks].repeat(B,1) < (seq_lens//self._block_size).unsqueeze(-1)
        mask_split = mask.sum(-1).tolist()
        max_split_len = max(mask_split)
        block_table = block_table.to(device)
        kv_shape =kv_layer[0, block_table[mask]].shape
        blocks, block_sz, num_head, head_size =  kv_shape
        if blocks == 0:
            return   
        # CHUNK_SIZE = 2**floor(log2(blocks*block_sz))
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
            # self.norms_k[layer_name] = k_norms
        else:
            kk = kk.view(B,num_blocks, -1)
            k_norms = kk[mask].norm(2,-1)
            splits_k = k_norms.split(mask_split)
            # max_len = max([len(s) for s in splits_k])
            # max_split_len = max(mask_split)
            splits_k = [F.pad(s, (0, max_split_len - len(s)), value=1.0) for s in splits_k]
            # for idx, req in enumerate(self.reqs):
            #     if req not in self.norms_k.keys():
            #         self.norms_k[req] = {}
                
            #     self.norms_k[req][layer_name] = splits_k[idx]

            # self.norms_k[layer_name] = k_norms.split(mask_split)
            # self.norms_k[layer_name] = [kk[i, mask[i]].norm(2, -1) for i in range(len(self.req_idx))]
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, self.b_idx, thr)

        kk = restore_cache(_k, _idx, kk.shape)

        if is_chunks_fusion:
            kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[0, block_table[mask]] = (kk).view(kv_shape)
        else:
            kv_layer[0, block_table[mask]] = kk.view(kv_shape) #(kk * k_norms.unsqueeze(-1)).view(kv_shape)

        del kk 

        vv = kv_layer[1, block_table]   

        if is_chunks_fusion:
            vv = vv[mask]

            vv_cat = vv[:-blocks_to_keep]
            vv = vv[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
        
            v_norms = vv.norm(2,-1)
            self.norms_v[layer_name] = v_norms
        else:
            vv = vv.view(B,num_blocks, -1)
            v_norms = vv[mask].norm(2,-1)
            splits_v = v_norms.split(mask_split)
            splits_v = [F.pad(s, (0, max_split_len - len(s)), value=1.0) for s in splits_v]
            # for idx in range(len(self.req_idx)):
            #     if f"{self.reqs[idx]}" not in self.norms_v.keys():
            #         self.norms_v[f"{self.reqs[idx]}"] = {}
                
            #     self.norms_v[f"{self.reqs[idx]}"][layer_name] = splits_v[idx]
            # self.norms_v[layer_name] = [vv[i, mask[i]].norm(2, -1) for i in range(len(self.req_idx))]

        _v = fuse_values_with_above_thr_idx(vv,fwd_idx, self.b_idx)  
        
        vv = restore_cache(_v, _idx, vv.shape)

        if is_chunks_fusion:
            vv =vv.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            vv = torch.cat([vv_cat,vv.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[1, block_table[mask]] = (vv).view(kv_shape)
        else:
            kv_layer[1, block_table[mask]] = vv.view(kv_shape) #(vv * v_norms.unsqueeze(-1)).view(kv_shape)

        update_block_table(block_table, fwd_idx, self.b_idx)
        attention_metadata.block_table[self.req_idx] = block_table
        
        # if layer_name not in self.block_tables.keys():
        #     self.block_tables[layer_name] = block_table
        for idx, req in enumerate(self.reqs):
            if req not in self.norms_k.keys():
                self.norms_k[req] = {}
                self.norms_v[req] = {}
                    # self.block_tables[req] = {}
                
            self.norms_k[req][layer_name] = splits_k[idx]
            self.norms_v[req][layer_name] = splits_v[idx]
        
        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [num_last_chunks_to_compress*CHUNK_SIZE/self._block_size]
        else:
            total_ +=[blocks]
        
        del vv 
        logger.info(f"Compression ratio in layer {layer_name}: {torch.tensor(total_).sum().item() / torch.tensor(compressed_).sum().item() if torch.tensor(compressed_).sum().item() > 0 else 0.0}")
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
    

@torch.inference_mode()
def fast_fusion(kv_cache, block_tables, thr, is_chunks_fusion, num_last_chunks_to_compress=2, remove_position = True, rotary_emb = None, seq_lens = None):
   
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
    block_tables_ = {}
    slot_maps_ = {}
    CHUNK_SIZE = 2**floor(log2(blocks*block_sz))
    blocks_to_keep = CHUNK_SIZE//block_sz
    
    compressed_ = []
    total_ = []
    cos_sin = rotary_emb.cos_sin_cache.index_select(0, torch.arange(block_tables[0].shape[1]*block_sz, device=device))
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    for l in range(L): # 2 warmup layers and 2 final layers
        
        if remove_position:
            kv_cache[l][0, block_tables[l]] = _apply_inv_rotary_emb(kv_cache[l][0, block_tables[l]].view(B,-1, num_head, head_size), cos, sin, rotary_emb.is_neox_style).view(B, -1, block_sz, num_head, head_size)
            
        kk = kv_cache[l][0, block_tables[l]]
         
        if is_chunks_fusion:
            kk = kk[mask]
            # CHUNK_SIZE = num_last_chunks_to_compress*block_sz
            # chunks = (blocks*block_sz)//CHUNK_SIZE
            # blocks_to_keep = CHUNK_SIZE//block_sz
            kk_cat = kk[:-blocks_to_keep]
            kk = kk[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk_cat = kk.view(chunks,blocks//chunks, -1)[:-num_last_chunks_to_compress]
            # kk = kk.view(chunks,blocks//chunks, -1)[-num_last_chunks_to_compress:]
            k_norms = kk.norm(2,-1)
        else:
            kk = kk.view(B,num_blocks, -1)
            k_norms = kk[mask].norm(2,-1)
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, b_idx, thr)

        # if not VLLM_USE_V1
        if False:
            kk = restore_cache(_k, _idx, kk.shape)

            if is_chunks_fusion:
                # kk =kk.view(num_last_chunks_to_compress,blocks//chunks, -1)
                kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
                kk*= k_norms.unsqueeze(-1)
                kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)
                kv_cache[l][0, block_tables[mask]] = (kk).view(kv_shape)
            else:
                kv_cache[l][0, block_tables[mask]] = (kk * k_norms.unsqueeze(-1)).view(kv_shape)

            if remove_position:
                kv_cache[l][0, block_tables] = _apply_rotary_emb(kv_cache[l][0, block_tables].view(B,-1, num_head, head_size), cos, sin, rotary_emb.is_neox_style).view(B, -1, block_sz, num_head, head_size)
            
        del _k, k_norms, kk

        vv = kv_cache[l][1, block_tables[l]]   

        if is_chunks_fusion:
            vv = vv[mask]
            # chunks = (blocks*block_sz)//CHUNK_SIZE
        
            # vv_cat = vv.view(chunks,blocks//chunks, -1)[:-num_last_chunks_to_compress]
            # vv = vv.view(chunks,blocks//chunks, -1)[-num_last_chunks_to_compress:]
            vv_cat = vv[:-blocks_to_keep]
            vv = vv[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            v_norms = vv.norm(2,-1)
        else:
            vv = vv.view(B,num_blocks, -1)
            v_norms = vv[mask].norm(2,-1)
        
        _v = fuse_values_with_above_thr_idx(vv,fwd_idx, b_idx)  
       
        if False:
            vv = restore_cache(_v, _idx, vv.shape) 
            
            if is_chunks_fusion:
                # vv =vv.view(num_last_chunks_to_compress,blocks//chunks, -1)
                vv =vv.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
                vv*= v_norms.unsqueeze(-1)
                vv*= v_norms.unsqueeze(-1)
                vv = torch.cat([vv_cat,vv.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)
                kv_cache[l][1, block_tables[mask]] = (vv).view(kv_shape)
            else:
                kv_cache[l][1, block_tables[mask]] = (vv * v_norms.unsqueeze(-1)).view(kv_shape)
        
        # bt_clone = block_tables[l].clone()
        # bt_clone[mask] = update_block_table(block_tables[l], fwd_idx, b_idx)
        # block_tables_[l] = bt_clone 
        update_block_table(block_tables[l], fwd_idx, b_idx)
        
        # blocks_idx = seq_lens//BLOCK_SIZE +  ((seq_lens % BLOCK_SIZE) > 0)
        # block_offsets = torch.arange(0, BLOCK_SIZE, device=seq_lens.device)
        # slot_maps = [(block_offsets.reshape((1, self._block_size)).to("cuda:0") + block_table[i, :nb].reshape(nb,1)*self._block_size).flatten()[:seq_lens[i]] for i,nb in enumerate(num_blocks)]
        # slot_maps_[l] = {f"{seq_lens[i]}":(block_offsets.reshape((1, BLOCK_SIZE)).to(vv.device) + block_tables_[l][i, :nb].reshape(nb,1)*BLOCK_SIZE).flatten()[:seq_lens[i]] for i,nb in enumerate(blocks_idx)}
        # slot_maps_[l] = torch.cat([(block_offsets.reshape((1, BLOCK_SIZE)).to(vv.device) + block_tables_[l][i, :nb].reshape(nb,1)*BLOCK_SIZE).flatten()[:seq_lens[i]] for i,nb in enumerate(blocks_idx)]).unique()
        
        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [num_last_chunks_to_compress*CHUNK_SIZE/BLOCK_SIZE]
        else:
            total_ +=[blocks]
        del _v, v_norms, vv
    
    _total =  torch.tensor(total_)
    _compressed = torch.tensor(compressed_)
    return _total.sum().item()/_compressed.sum().item(), _total, _compressed, block_tables #, slot_maps_    

def _update_block_tables_after_compression_(  
    self,  
    new_requests: list[str]  
) -> dict[str, dict[int, list[int]]]:  
    """Return final block lists instead of mappings."""  
    request_blocks = {}  
      
    for req_id in new_requests:  
        if req_id not in self.input_batch.req_id_to_index:  
            continue  
              
        req_idx = self.input_batch.req_id_to_index[req_id]  
        req_state = self.requests[req_id]  
          
        # Handle both primary and modular runner block table structures  
        for group_idx, block_table_obj in enumerate(self.input_batch.block_table.block_tables):  
            if group_idx == 0:  
                continue  
              
            # Check which block table implementation we're using  
            if hasattr(block_table_obj, 'block_table'):  
                # Primary runner: uses CpuGpuBuffer  
                new_table = block_table_obj.block_table.gpu[req_idx]  
                num_blocks = block_table_obj.num_blocks_per_row[req_idx]  
                final_blocks = new_table[:num_blocks].tolist()  
                  
                # Update both GPU and CPU versions  
                if not all(block_table_obj.block_table.np[req_idx, :num_blocks] == final_blocks):  
                    req_state.block_ids[group_idx][:num_blocks] = final_blocks  
                    block_table_obj.block_table.np[req_idx, :num_blocks] = new_table[:num_blocks].cpu().numpy()  
                      
            else:  
                # Modular runner: uses StagedWriteTensor  
                # Note: StagedWriteTensor requires different handling  
                num_blocks = block_table_obj.num_blocks.np[group_idx, req_idx]  
                # Access through the staged write tensor  
                current_blocks = block_table_obj.block_tables[group_idx].gpu[req_idx, :num_blocks]  
                final_blocks = current_blocks.tolist()  
                  
                if req_id not in request_blocks:  
                    request_blocks[req_id] = {}  
                req_state.block_ids[group_idx][:num_blocks] = final_blocks  
                # Update requires staged write  
                block_table_obj.stage_write(req_idx, 0, final_blocks)  
              
            if req_id not in request_blocks:  
                request_blocks[req_id] = {}  
            request_blocks[req_id][group_idx] = final_blocks  
      
    # Apply staged writes if using modular runner  
    if hasattr(block_table_obj, 'apply_staged_writes'):  
        self.input_batch.block_table.apply_staged_writes()  
      
    self._updated_block_tables =  request_blocks

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
            new_table = block_table_obj.block_table.gpu[req_idx]#block_tables_[group_idx][req_idx]  
            num_blocks = block_table_obj.num_blocks_per_row[req_idx]    #sefi tset
            # Update state and block table  
            
            final_blocks = new_table[:num_blocks].tolist()
            if all(block_table_obj.block_table.np[req_idx, :num_blocks] == final_blocks) :
                continue
            if req_id not in request_blocks.keys():
                request_blocks[req_id] = {}
            req_state.block_ids[group_idx][:num_blocks] = final_blocks
            block_table_obj.block_table.np[req_idx, :num_blocks] = new_table[:num_blocks].cpu().numpy()  
            # block_table_obj.block_table[req_idx, :num_blocks] = new_table[:num_blocks]
            # block_table_obj.num_blocks_per_row[req_idx] = num_blocks  
            
            # Store final block list  
            request_blocks[req_id][group_idx] = final_blocks  
  
    self._updated_block_tables =  request_blocks

def _update_block_tables_after_compression_old(  
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
            # block_table_obj.block_table[req_idx, :num_blocks] = new_table[:num_blocks]
            # block_table_obj.num_blocks_per_row[req_idx] = num_blocks  
            
            # Store final block list  
            request_blocks[req_id][group_idx] = final_blocks  
  
    return request_blocks

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.sequence import IntermediateTensors
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
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.v1.worker.ubatch_utils import (
    UBatchSlices,
    check_ubatch_thresholds,
    maybe_create_ubatch_slices,
)
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.forward_context import (
    BatchDescriptor,
    # set_forward_context,
    get_forward_context,
)
from kv_fast_fusion.fast_fusion_context import patched_set_forward_context as set_forward_context
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
    if self.execute_model_state is not None:
        raise RuntimeError(
            "State error: sample_tokens() must be called "
            "after execute_model() returns None."
        )

    if self.vllm_config.model_config.enable_return_routed_experts:
        capturer = RoutedExpertsCapturer.get_instance()
        if capturer is not None:
            capturer.clear_buffer()  # noqa
        else:
            logger.error("RoutedExpertsCapturer not initialized.")

    if scheduler_output.preempted_req_ids and has_kv_transfer_group():
        get_kv_transfer_group().handle_preemptions(
            scheduler_output.preempted_req_ids
        )

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    with (
        record_function_or_nullcontext("gpu_model_runner: preprocess"),
        self.synchronize_input_prep(),
    ):
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
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

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

        logits_indices, spec_decode_metadata = self._prepare_inputs(
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

        attn_metadata, spec_decode_common_attn_metadata = (
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
    if not hasattr(self, 'fused_requests'):
            self.fused_requests = []
            
    # attn_metadata['scaling_factors'] = None
    if self.compression_hook:
        # req_to_compress = [req_id for req_id in self.input_batch.req_ids if scheduler_output.num_scheduled_tokens[req_id] > 1 and req_id not in self.fused_requests]
        req_to_compress = [  
            req_id for req_id in self.input_batch.req_ids  
            if (scheduler_output.num_scheduled_tokens.get(req_id, 0) == 1 and   
                req_id not in self.fused_requests)  
        ]
        fused_reqs = [req_id for req_id in self.input_batch.req_ids if req_id in self.fused_requests]
        req_idx_to_compress = [self.input_batch.req_id_to_index[r] for r in req_to_compress]
        scales = [(self.compression_hook.norms_k[fr], self.compression_hook.norms_v[fr]) for fr in fused_reqs] if self.compression_hook.norms_k != {} else None
        if scales is not None:
            req_scales_idx = [self.input_batch.req_id_to_index[r] for r in fused_reqs]
            for d1, d2 in scales:
                for key, val in zip(d1.items(), d2.values()):
                    if hasattr(attn_metadata[key[0]], 'sf_k'): #key[0] in sf_k:
                        shape_k1 = attn_metadata[key[0]].sf_k.shape[-1]
                        shape_k2 = key[1].shape[-1]
                        if shape_k1 > shape_k2:
                            # sf_k[key[0]] = torch.vstack((sf_k[key[0]], F.pad(key[1], (0, shape_k1 - shape_k2), value=1.0) )) 
                            # sf_v[key[0]] = torch.vstack((sf_v[key[0]], F.pad(val, (0, shape_k1 - shape_k2), value=1.0) ))
                            attn_metadata[key[0]].sf_k = torch.vstack((attn_metadata[key[0]].sf_k, F.pad(key[1], (0, shape_k1 - shape_k2), value=1.0) )) 
                            attn_metadata[key[0]].sf_v = torch.vstack((attn_metadata[key[0]].sf_v, F.pad(val, (0, shape_k1 - shape_k2), value=1.0) ))
                            attn_metadata[key[0]].req_idx = req_scales_idx
                        elif shape_k1 < shape_k2:
                            # sf_k[key[0]] = torch.vstack((F.pad(sf_k[key[0]], (0, shape_k2 - shape_k1), value=1.0), key[1])) 
                            # sf_v[key[0]] = torch.vstack((F.pad(sf_v[key[0]], (0, shape_k2 - shape_k1), value=1.0), val))
                            attn_metadata[key[0]].sf_k = torch.vstack((F.pad(attn_metadata[key[0]].sf_k, (0, shape_k2 - shape_k1), value=1.0), key[1])) 
                            attn_metadata[key[0]].sf_v = torch.vstack((F.pad(attn_metadata[key[0]].sf_v, (0, shape_k2 - shape_k1), value=1.0), val))
                            attn_metadata[key[0]].req_idx = req_scales_idx
                        else:
                            # sf_k[key[0]] = torch.vstack((sf_k[key[0]], key[1]))  
                            # sf_v[key[0]] = torch.vstack((sf_v[key[0]], val)) 
                            attn_metadata[key[0]].sf_k = torch.vstack((attn_metadata[key[0]].sf_k, key[1]))  
                            attn_metadata[key[0]].sf_v = torch.vstack((attn_metadata[key[0]].sf_v, val)) 
                            attn_metadata[key[0]].req_idx = req_scales_idx
                    else:
                        # sf_k[key[0]] = key[1]
                        # sf_v[key[0]] = val                    
                        attn_metadata[key[0]].sf_k = key[1]
                        attn_metadata[key[0]].sf_v = val
                        attn_metadata[key[0]].req_idx = req_scales_idx

            # attn_metadata['scaling_factors'] = (sf_k, sf_v,  [self.input_batch.req_id_to_index[r] for r in fused_reqs])            
            # attn_metadata['compression_hook'] = self.compression_hook
            
        self.compression_hook.req_idx = req_idx_to_compress
        self.compression_hook.reqs = req_to_compress

    self._updated_block_tables = None
    
    hook = None
    if self.compression_hook and len(req_to_compress) > 1:                 
                hook = self.compression_hook
                logger.info(f"bff is dealing with batch size {len(req_to_compress)}")
    with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                compression_hook= hook,            
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            # forward_context = get_forward_context()
            # forward_context.additional_kwargs['compression_hook'] = self.compression_hook
            
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

            if hook:
                self.compression_hook.wait_for_all_compressions()
                self.fused_requests.extend(req_to_compress)  
                self._update_block_tables_after_compression(req_to_compress) 

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
                return self._pool(
                    hidden_states,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    kv_connector_output,
                )

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

from vllm.v1.structured_output.utils import apply_grammar_bitmask
from copy import copy
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput 

@torch.inference_mode
def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None

    if self.execute_model_state is None:
        # Nothing to do (PP non-final rank case), output isn't used.
        if not kv_connector_output:
            return None  # type: ignore[return-value]

        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    updated_block_tables = getattr(self, '_updated_block_tables', None) #sefi
    # Unpack ephemeral state.
    (
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        ec_connector_output,
        cudagraph_stats,        
    ) = self.execute_model_state
    # Clear ephemeral state.
    self.execute_model_state = None

    # Apply structured output bitmasks if present.
    if grammar_output is not None:
        apply_grammar_bitmask(
            scheduler_output, grammar_output, self.input_batch, logits
        )

    with record_function_or_nullcontext("gpu_model_runner: sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    self._draft_token_ids = None
    self._draft_token_req_ids = None
    self.input_batch.prev_sampled_token_ids = None

    def propose_draft_token_ids(sampled_token_ids):
        assert spec_decode_common_attn_metadata is not None
        with record_function_or_nullcontext("gpu_model_runner: draft"):
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

    spec_config = self.speculative_config
    propose_drafts_after_bookkeeping = False
    if spec_config is not None:
        input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
            <= self.effective_drafter_max_model_len
        )
        if spec_config.use_eagle() and not spec_config.disable_padded_drafter_batch:
            # EAGLE speculative decoding can use the GPU sampled tokens
            # as inputs, and does not need to wait for bookkeeping to finish.
            assert isinstance(self.drafter, EagleProposer)
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                propose_draft_token_ids(sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                assert spec_decode_common_attn_metadata is not None
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        spec_decode_common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )
                # Since we couldn't run the drafter,
                # just use zeros for the draft tokens.
                self._draft_token_ids = torch.zeros(
                    1, device=self.device, dtype=torch.int32
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
        else:
            propose_drafts_after_bookkeeping = input_fits_in_drafter

    with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
            spec_decode_metadata,
        )

    if propose_drafts_after_bookkeeping:
        # ngram and other speculative decoding methods use the sampled
        # tokens on the CPU, so they are run after bookkeeping.
        propose_draft_token_ids(valid_sampled_token_ids)

    with record_function_or_nullcontext("gpu_model_runner: eplb"):
        self.eplb_step()

    with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
        if self.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output
            if self.supports_mm_inputs
            else None,
            num_nans_in_logits=num_nans_in_logits,
            cudagraph_stats=cudagraph_stats,
            # updated_block_tables=updated_block_tables,
        )

    output._updated_block_tables = updated_block_tables #sefi
    if not self.use_async_scheduling:
        return output

    with record_function_or_nullcontext(
        "gpu_model_runner: AsyncGPUModelRunnerOutput"
    ):
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
    with record_function_or_nullcontext(
        "gpu_model_runner: set_async_sampled_token_ids"
    ):
        # Save ref of sampled_token_ids CPU tensor if the batch contains
        # any requests with sampling params that require output ids.
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )

    return async_output

# from dataclasses import dataclass
# # Create new dataclass with additional field  
# @dataclass  
# class PatchedModelRunnerOutput(ModelRunnerOutput):  
#     updated_block_tables: dict[str, dict[int, list[int]]] | None = None  



from vllm.v1.worker.gpu_model_runner import (
    ExecuteModelState,
    PerLayerAttnMetadata, 
)
from vllm.model_executor.models.interfaces import (
    supports_mm_encoder_only,
)
from vllm.utils.math_utils import cdiv, round_up


def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
        """
        if supports_mm_encoder_only(self.model):
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile
                or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=activate_lora,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = (
            batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        )
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_padded, None, False
                )

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    compression_hook=self.compression_hook,                
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        self._register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

