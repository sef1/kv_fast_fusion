import torch
import os

# import vllm.envs as envs
from vllm.logger import init_logger
logger = init_logger("vllm.vllm_patch")

from math import floor, log2
from kv_fast_fusion.compression_hook import CompressionHook

import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, NamedTuple

from dataclasses import dataclass  
  
THRESHOLD = 0.7
BLOCK_SIZE = 128
NUM_LAST_CHUNKS_TO_COMPRESS = 4
CHUNK_SIZE = 512

# LSH deduplication constants
NUM_LSH_BITS = 256            # total bits per fingerprint
NUM_LSH_TABLES = 16           # independent sub-tables
LSH_BITS_PER_TABLE = 16       # bits per sub-table  (NUM_LSH_BITS // NUM_LSH_TABLES)
MAX_REGISTRY_PER_LAYER = 8192  # max registered blocks per fusion layer (hard cap)
LSH_MAX_HEAD_DIM = 512       # lsh_proj is pre-allocated up to this head_dim


class BlockCompressionHookGraph(CompressionHook):  

    def __init__(self, vllm_config, attn_metadata, fused_requests, max_batch_size = 16, warmup_layers=2):  
        self.warmup_layers = warmup_layers
        self.compression_stream = torch.cuda.Stream()
        self.max_layer_idx = (vllm_config.model_config.get_num_layers(vllm_config.parallel_config) - self.warmup_layers)
        self._block_size = BLOCK_SIZE
        self.thr = 0.0
        self.num_blocks = attn_metadata.block_table.shape[-1]
        req_idx = attn_metadata.req_idx_to_compress
        device = attn_metadata.block_table.device
        self.B = len(req_idx)
        self.block_table = attn_metadata.block_table[req_idx]        
        self.seq_lens_buffer = attn_metadata.seq_lens[req_idx].unsqueeze(-1)
        self.idx__=torch.arange(self.num_blocks, dtype=torch.int, device=device)
        self.b_idx = list(range(self.B))
        self.mask = self.idx__.repeat(self.B,1) < (self.seq_lens_buffer[:self.B] //self._block_size)
        self.block_table_masked = self.block_table[self.mask]
        self.nz_mask = self.mask.nonzero(as_tuple=True)
        mask_split = self.mask.sum(-1)
        self.mask_split_cumsum = torch.cumsum(mask_split, -1)
        self.mask_split = mask_split.tolist()
        self.max_split_len = mask_split.max()
        self.splits_k = torch.ones(self.B, self.num_blocks, dtype=torch.bfloat16)
        self.splits_v = torch.ones(self.B, self.num_blocks, dtype=torch.bfloat16)
        self.nz_blocks = (self.seq_lens_buffer // self._block_size).squeeze()
        self.fused_requests = fused_requests
            
    def start_layer_compression(self,
                                 layer_name: str,
                                   kv_cache,
                                     attn_metadata,                                       
                                        #  compression_state: dict | None = None
                                        ):  
        """Start async compression immediately after layer populates KV cache."""  

        # req_idx = attn_metadata.req_idx_to_compress
        # if len(req_idx)<2:
        #     return #compression_state or self._init_state(req_idx, attn_metadata.block_table.shape[1], kv_cache.device)  
        
        layer_idx = int(layer_name.split('.')[2])  # Extract layer number

       
        if layer_idx < self.warmup_layers or \
            layer_idx >= self.max_layer_idx: #not in self._should_compress_layer:          
            return

        # compression_state = self._init_state(req_idx, attn_metadata.block_table.shape[1], kv_cache.device) 
        
        # Run compression directly on current stream  
        # default_stream = torch.cuda.current_stream()
        # with torch.cuda.stream(self.compression_stream):  
        #     # Ensure the attention kernel on the default stream is finished before we read the KV cache.
        #     self.compression_stream.wait_stream(default_stream)
        #     # Launch async compression for this specific layer  
        #     event = torch.cuda.Event()  
        #     self.layer_fast_fusion(layer_name, kv_cache, attn_metadata)
        #     event.record(self.compression_stream)  
        #     self.compression_events[layer_name] = event  
        #     self.wait_for_layer_compression(layer_name)
        self.layer_fast_fusion(layer_name, kv_cache, attn_metadata) 

        return 
    
         
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
    
    def layer_fast_fusion(self,
                        layer_name: str,
                        kv_layer: torch.Tensor,
                        attn_metadata: Any,
                        ):
           
        def fuse_all_above_thr(x,
                               b_idx: list,                            
                               thr: float =self.thr):
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
                # nz_blocks = x.shape[1] if is_chunks_fusion else (seq_lens[b_idx]//self._block_size)[0,0]#.item()            
                nz_blocks = self.nz_blocks[b_idx[0]]
               
                return F.normalize( x[:, :nz_blocks] , dim=-1, eps=1e-7), [self.idx__[:nz_blocks]], [], [nz_blocks]
                    
            xl, _idx_l, fl_chain, shifts_l = fuse_all_above_thr(x[:B//2],  b_idx[:B//2], thr=thr)
            xr, _idx_r, fr_chain, shifts_r = fuse_all_above_thr(x[B//2:],  b_idx[B//2:], thr=thr)

            nl = xl.shape[1]
            nr = xr.shape[1]
            idx_l = self.idx__[:nl]
            idx_r = self.idx__[:nr]
                
            idx_ll, idx_rr = (xl @ xr.mT > thr).nonzero(as_tuple=True)[-2:]
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
                _idx_l += [torch.tensor([], device=xl.device, dtype=torch.int) 
                           for _ in range(max_length - len(_idx_l))]
            
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
        
        def fuse_values_with_above_thr_idx(v: torch.Tensor,
                                           fwd_idx: dict,
                                           b_idx: List[int],):
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
            def recurssive_combining(v: torch.Tensor,
                                     b_idx: List[int]):
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
            # xx = torch.empty((1, len(idx[-2]), shape[2]), dtype=x.dtype, device=x.device)
            xx = torch.empty((1, len(idx[-2]), x.shape[-1]), dtype=x.dtype, device=x.device)
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
        
        # Extract parameters  #block_table, seq_lens should be cloned? 
        thr = self.thr
        seq_lens = self.seq_lens_buffer
        req_idx = attn_metadata.req_idx_to_compress
        reqs = attn_metadata.req_ids_to_compress
               
        num_last_chunks_to_compress = 4
        is_chunks_fusion = False
        
            
       
        kv_shape =kv_layer[0, self.block_table_masked].shape
        blocks, block_sz, num_head, head_size =  kv_shape
        blocks_to_keep = CHUNK_SIZE//block_sz
        
        compressed_ = []
        total_ = []
            
        kk = kv_layer[0, self.block_table]

        ######
        # 
        # 
        # if '17' in layer_name:
        #     with open("/data/users/sefi/vllm_logs/fusion_debug/kv_cache_sample.npy", 'ab') as f:
        #         np.save(f, kv_layer[0, block_table[mask]].float().cpu().numpy())
        #     # np.savez(f"/data/users/sefi/vllm_logs/fusion_debug/kv_samples.npz", kk[mask].float().cpu().numpy())   
             
        if is_chunks_fusion:
            kk = kk[self.mask]
            
            kk_cat = kk[:-blocks_to_keep]
            kk = kk[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            k_norms = kk.norm(2,-1)
        else:
            kk = kk.view(self.B,self.num_blocks, -1)
            k_norms = kk[self.nz_mask].norm(2,-1)
            for i,k in enumerate(k_norms.split(self.mask_split)):
                self.splits_k[i,:k.shape[0]] = k
            # splits_k = k_norms.split(mask_split)
            # splits_k = [k_norms.as_strided((size,), k_norms.stride(), offset-size)  for offset,size in zip(self.mask_split_cumsum, self.mask_split)]
            # splits_k = [F.pad(k_norms.as_strided((size,), k_norms.stride(), offset-size), (0, max_split_len - len(k_norms.as_strided((size,), k_norms.stride(), offset-size))), value=1.0)  for offset,size in zip(torch.cumsum(mask_split, -1), mask_split)]
            # splits_k = [F.pad(s, (0, self.max_split_len - len(s)), value=1.0) for s in splits_k]
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, self.b_idx, thr)

        kk = restore_cache(_k, _idx, kk.shape)

        if is_chunks_fusion:
            kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[0, self.block_table_masked] = (kk).view(kv_shape)
        else:
            kv_layer[0, self.block_table_masked] = kk.view(kv_shape) #(kk * k_norms.unsqueeze(-1)).view(kv_shape)

        del kk, _k, 

        vv = kv_layer[1, self.block_table] 

        if is_chunks_fusion:
            vv = vv[self.mask]

            vv_cat = vv[:-blocks_to_keep]
            vv = vv[-blocks_to_keep:].view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
        
            v_norms = vv.norm(2,-1)
        else:
            vv = vv.view(self.B,self.num_blocks, -1)
            v_norms = vv[self.mask].norm(2,-1)
            # splits_v = v_norms.split(mask_split)
            # splits_v = [v_norms.as_strided((size,), v_norms.stride(), offset-size)  for offset,size in zip(self.mask_split_cumsum, self.mask_split)]
            # splits_v = [F.pad(s, (0, self.max_split_len - len(s)), value=1.0) for s in splits_v]
            for i,v in enumerate(v_norms.split(self.mask_split)):
                self.splits_v[i, :v.shape[0]] = v
            
        _v = fuse_values_with_above_thr_idx(vv,fwd_idx, self.b_idx)  
        
        vv = restore_cache(_v, _idx, vv.shape)

        if is_chunks_fusion:
            vv =vv.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            vv = torch.cat([vv_cat,vv.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[1, self.block_table_masked] = (vv).view(kv_shape)
        else:
            kv_layer[1, self.block_table_masked] = vv.view(kv_shape) #(vv * v_norms.unsqueeze(-1)).view(kv_shape)

        update_block_table(self.block_table, fwd_idx, self.b_idx)
        attn_metadata.block_table[req_idx] = self.block_table
        
        # if layer_name not in self.block_tables.keys():
        #     self.block_tables[layer_name] = block_table
        for idx, req in enumerate(reqs):
            if req not in self.fused_requests.keys():
                self.fused_requests[req] = {}
                self.fused_requests[req] = {}
                    # self.block_tables[req] = {}
                
            self.fused_requests[req][layer_name] = (self.splits_k[idx], self.splits_v[idx])
            # self.fused_requests[req][layer_name] = self.splits_v[idx]
        
        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [num_last_chunks_to_compress*CHUNK_SIZE/self._block_size]
        else:
            total_ +=[blocks]
        
        del vv, _v,  fwd_idx, _idx
        logger.info(f"Compression ratio in layer {layer_name}: {torch.tensor(total_).sum().item() / torch.tensor(compressed_).sum().item() if torch.tensor(compressed_).sum().item() > 0 else 0.0}")
        return 

import numpy as np


# ---------------------------------------------------------------------------
# Norm-buffer helpers (graph-mode decode scaling)
# ---------------------------------------------------------------------------

def _fill_norm_buffers(
    self,
    fused_reqs: list[str],
    fused_reqs_idx: list[int],
) -> None:
    """Reset norm buffers to 1.0 and fill with stored norms for fused requests."""
    if not hasattr(self, 'norms_k_buf') or self.norms_k_buf is None:
        return
    self.norms_k_buf.fill_(1.0)
    self.norms_v_buf.fill_(1.0)
    for req_id, req_idx in zip(fused_reqs, fused_reqs_idx):
        per_layer = self.fused_requests.get(req_id)
        if per_layer is None:
            continue
        for layer_name, (norms_k, norms_v) in per_layer.items():
            layer_idx = int(layer_name.split('.')[2])
            n = norms_k.shape[0]
            max_blocks = self.norms_k_buf.shape[2]
            n = min(n, max_blocks)
            self.norms_k_buf[layer_idx, req_idx, :n].copy_(norms_k[:n].to(self.norms_k_buf.device))
            self.norms_v_buf[layer_idx, req_idx, :n].copy_(norms_v[:n].to(self.norms_v_buf.device))


class BFFCompressor:
    """Encapsulates one BFF post-forward pass across all fusion layers.

    Expensive per-call-invariant state (mask, nz_mask, arange tensor, norm buffers)
    is computed once in __init__ instead of being recreated inside the layer loop.
    The four inner algorithms (fuse_all_above_thr, fuse_values, restore_cache,
    update_block_table_inplace) become class methods so Python does not create and
    garbage-collect closure objects on every iteration of the 28-layer loop.
    """

    __slots__ = (
        '_runner', '_req_to_compress', '_req_idx_to_compress', '_meta_dict',
        'B', '_b_idx', '_num_blocks_dim', '_device',
        '_nz_blocks', '_idx__', '_nz_mask', '_mask_split_list',
        '_splits_k', '_splits_v',
        '_total_valid', '_total_unique',
        # LSH fields — precomputed once, shared by run_lsh() and run_lsh_register()
        '_head_dim', '_lsh_proj', '_kv_dtype',
    )

    def __init__(self, runner, req_to_compress, req_idx_to_compress, attn_metadata_dict):
        self._runner            = runner
        self._req_to_compress   = req_to_compress
        self._req_idx_to_compress = req_idx_to_compress
        self.B                  = len(req_idx_to_compress)
        self._b_idx             = list(range(self.B))
        self._meta_dict         = (attn_metadata_dict
                                   if isinstance(attn_metadata_dict, dict)
                                   else attn_metadata_dict[0])
        self._total_valid  = 0
        self._total_unique = 0

        # Discover dimensions from the first available fusion-layer metadata.
        # seq_lens and block-table width are identical across all fusion layers.
        first_meta = None
        for kv_gid, kv_group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if kv_gid == 0:
                continue
            for ln in kv_group.layer_names:
                if ln in self._meta_dict:
                    first_meta = self._meta_dict[ln]
                    break
            if first_meta is not None:
                break

        if first_meta is None:
            self._num_blocks_dim  = 0
            self._device          = torch.device('cpu')
            self._nz_blocks       = None
            self._idx__           = None
            self._nz_mask         = None
            self._mask_split_list = []
            self._splits_k        = None
            self._splits_v        = None
            self._head_dim        = None
            self._lsh_proj        = None
            self._kv_dtype        = None
            return

        self._num_blocks_dim = first_meta.block_table.shape[1]
        self._device         = first_meta.block_table.device

        # All fusion layers present the same seq_lens — compute once.
        seq_lens = first_meta.seq_lens[req_idx_to_compress].unsqueeze(-1)  # [B, 1]
        self._nz_blocks = (seq_lens // BLOCK_SIZE).squeeze(-1)             # [B]

        self._idx__ = torch.arange(
            self._num_blocks_dim, dtype=torch.int, device=self._device
        )
        mask = self._idx__.repeat(self.B, 1) < self._nz_blocks.unsqueeze(-1)  # [B, P]
        self._nz_mask         = mask.nonzero(as_tuple=True)
        self._mask_split_list = mask.sum(-1).tolist()

        # Pre-allocated norm buffers — reset to 1.0 before each layer, no re-alloc.
        self._splits_k = torch.ones(
            self.B, self._num_blocks_dim, dtype=torch.bfloat16, device=self._device
        )
        self._splits_v = torch.ones(
            self.B, self._num_blocks_dim, dtype=torch.bfloat16, device=self._device
        )

        # LSH fields — head_dim and lsh_proj are constant across all fusion layers.
        self._head_dim = self._kv_dtype = self._lsh_proj = None
        for kv_gid2, kv_group2 in enumerate(runner.kv_cache_config.kv_cache_groups):
            if kv_gid2 == 0:
                continue
            if runner.kv_caches and kv_gid2 < len(runner.kv_caches):
                kv0 = runner.kv_caches[kv_gid2]
                self._head_dim = kv0.shape[-1]
                self._lsh_proj = runner.lsh_proj[:self._head_dim]  # [head_dim, NUM_LSH_BITS]
                self._kv_dtype = kv0.dtype
                break

    # ------------------------------------------------------------------
    # Merge algorithms — defined once as methods, not re-created per layer
    # ------------------------------------------------------------------

    def _fuse_all_above_thr(self, x, b_idx_local, thr=THRESHOLD):
        Bloc, _, _ = x.shape
        if Bloc == 1:
            nz = self._nz_blocks[b_idx_local[0]]
            return (F.normalize(x[:, :nz], dim=-1, eps=1e-7),
                    [self._idx__[:nz]], [], [nz])

        xl, _idx_l, fl_chain, shifts_l = self._fuse_all_above_thr(
            x[:Bloc // 2], b_idx_local[:Bloc // 2], thr)
        xr, _idx_r, fr_chain, shifts_r = self._fuse_all_above_thr(
            x[Bloc // 2:], b_idx_local[Bloc // 2:], thr)

        nl, nr = xl.shape[1], xr.shape[1]
        idx_ll, idx_rr = (xl @ xr.mT > thr).nonzero(as_tuple=True)[-2:]
        l_idx, c = torch.unique(idx_ll, return_counts=True)
        r_idx    = idx_rr.split(tuple(c.tolist()))

        idx_ul = list(set(range(nl)) - set(idx_ll.tolist()))
        idx_ur = list(set(range(nr)) - set(idx_rr.tolist()))
        n_c, n_ul, n_ur = len(l_idx), len(idx_ul), len(idx_ur)

        combined = [
            torch.cat([xl[:, l_idx[i]].unsqueeze(1), xr[:, r_idx[i]]], dim=1).mean(1, keepdim=True)
            for i in range(n_c)
        ]
        if combined:
            combined_x = F.normalize(torch.cat(combined, dim=1), dim=-1)
            combined_x = torch.cat([combined_x, xl[:, idx_ul], xr[:, idx_ur]], dim=1)
        else:
            combined_x = torch.cat([xl[:, idx_ul], xr[:, idx_ur]], dim=1)

        rev = torch.empty(nl + nr, device=self._device, dtype=torch.int)
        rev[l_idx.tolist()] = self._idx__[:n_c]
        for i in range(n_c):
            rev[(r_idx[i] + nl).tolist()] = self._idx__[:n_c][i]
        rev[idx_ul] = self._idx__[n_c:n_c + n_ul]
        rev[list(map(lambda v: v + nl, idx_ur))] = self._idx__[n_c + n_ul:n_c + n_ul + n_ur]

        max_len = max(len(_idx_l), len(_idx_r))
        if len(_idx_l) < max_len:
            shifts_l += [shifts_l[-1]] * (max_len - len(_idx_l))
            _idx_l   += [torch.tensor([], device=self._device, dtype=torch.int)] * (
                         max_len - len(_idx_l))

        chain    = [torch.cat([_idx_l[i], _idx_r[i] + shifts_l[i]]) for i in range(max_len)]
        rev      = [rev] + chain
        fl_chain = (fl_chain + fr_chain +
                    [(b_idx_local, torch.stack([idx_ll, idx_rr], dim=-1), idx_ul, idx_ur)])
        shifts   = ([n_c + n_ul + n_ur] +
                    list(map(lambda p, q: p + q, shifts_l, shifts_r)))
        return combined_x, rev, fl_chain, shifts

    def _fuse_values(self, v, fwd_idx, b_idx_local):
        nz_blocks = self._nz_blocks
        counter   = [0]

        def recurse(v, b_local):
            Bloc = v.shape[0]
            if Bloc == 1:
                nz = int(nz_blocks[b_local[0]])
                return F.normalize(v[:, :nz], dim=-1, eps=1e-7)
            vl = recurse(v[:Bloc // 2], b_local[:Bloc // 2])
            vr = recurse(v[Bloc // 2:], b_local[Bloc // 2:])
            _, idx_, idx_ul, idx_ur = fwd_idx[counter[0]]
            idx_ll, idx_rr = idx_.mT
            l_idx, c = torch.unique(idx_ll, return_counts=True)
            r_idx    = idx_rr.split(tuple(c.tolist()))
            combined = [
                torch.cat([vl[:, l_idx[i]].unsqueeze(1), vr[:, r_idx[i]]], dim=1).mean(1, keepdim=True)
                for i in range(len(l_idx))
            ]
            if combined:
                cv = F.normalize(torch.cat(combined, dim=1), dim=-1)
                cv = torch.cat([cv, vl[:, idx_ul], vr[:, idx_ur]], dim=1)
            else:
                cv = torch.cat([vl[:, idx_ul], vr[:, idx_ur]], dim=1)
            counter[0] += 1
            return cv

        return recurse(v, b_idx_local)

    @staticmethod
    def _restore_cache(x, rev_idx):
        xx = torch.empty((1, len(rev_idx[-2]), x.shape[-1]), dtype=x.dtype, device=x.device)
        xx[:, :x.shape[1]] = x
        for idx_ in rev_idx[:-1]:
            xx[:, :len(idx_)] = xx[:, idx_]
        return xx

    def _update_block_table_inplace(self, bt, fwd_idx, b_idx_local):
        """Redirect right-branch block IDs to the matched left-branch IDs in-place."""
        nz_blocks = self._nz_blocks
        counter   = [0]

        def recurse(bt_sub, b_local):
            Bloc = bt_sub.shape[0]
            if Bloc == 1:
                return bt_sub[:, :int(nz_blocks[b_local[0]])]
            bl = recurse(bt_sub[:Bloc // 2], b_local[:Bloc // 2])
            br = recurse(bt_sub[Bloc // 2:], b_local[Bloc // 2:])
            idx_ = fwd_idx[counter[0]][1]
            if idx_.numel() > 0:
                br.view(1, -1)[:, idx_[:, 1]] = bl.view(1, -1)[:, idx_[:, 0]]
            counter[0] += 1
            return torch.cat([bl, br], dim=-1)

        recurse(bt, b_idx_local)

    # ------------------------------------------------------------------
    # Per-layer compression (only genuinely layer-specific work here)
    # ------------------------------------------------------------------

    def _compress_layer(self, layer_name, layer_idx, layer_meta, kv_layer):
        """Run BFF for one fusion layer. Returns (n_valid_blocks, n_unique_blocks)."""
        _, block_sz, num_heads, head_dim = kv_layer.shape[1:]

        # Block-table entries differ per layer (each layer has its own block pool).
        block_table_layer = layer_meta.block_table[self._req_idx_to_compress].clone()
        bt_safe    = block_table_layer.clamp(min=0)
        flat_valid = bt_safe[self._nz_mask]

        # Reuse pre-allocated buffers — fill_(1.0) is cheaper than a new allocation.
        self._splits_k.fill_(1.0)
        self._splits_v.fill_(1.0)

        # ---- K cache ----
        kk     = kv_layer[0][bt_safe].view(self.B, self._num_blocks_dim, -1)
        k_norms = kk[self._nz_mask].norm(2, -1)
        for i, k in enumerate(k_norms.split(self._mask_split_list)):
            self._splits_k[i, :k.shape[0]] = k

        _k, rev_idx, fwd_idx, _ = self._fuse_all_above_thr(kk, self._b_idx)
        kk_restored = self._restore_cache(_k, rev_idx)
        kv_layer[0][flat_valid] = kk_restored.squeeze(0).view(-1, block_sz, num_heads, head_dim)

        # ---- V cache ----
        vv     = kv_layer[1][bt_safe].view(self.B, self._num_blocks_dim, -1)
        v_norms = vv[self._nz_mask].norm(2, -1)
        for i, v in enumerate(v_norms.split(self._mask_split_list)):
            self._splits_v[i, :v.shape[0]] = v

        _v = self._fuse_values(vv, fwd_idx, self._b_idx)
        vv_restored = self._restore_cache(_v, rev_idx)
        kv_layer[1][flat_valid] = vv_restored.squeeze(0).view(-1, block_sz, num_heads, head_dim)

        # ---- Block-table redirect and write-back ----
        self._update_block_table_inplace(block_table_layer, fwd_idx, self._b_idx)
        layer_meta.block_table[self._req_idx_to_compress] = block_table_layer

        # ---- Persist norms for decode scaling ----
        runner = self._runner
        for i, req_id in enumerate(self._req_to_compress):
            if req_id not in runner.fused_requests:
                runner.fused_requests[req_id] = {}
            runner.fused_requests[req_id][layer_name] = (
                self._splits_k[i].clone(), self._splits_v[i].clone()
            )

        if runner.norms_k_buf is not None:
            nb = min(self._num_blocks_dim, runner.norms_k_buf.shape[2])
            for i, req_idx in enumerate(self._req_idx_to_compress):
                runner.norms_k_buf[layer_idx, req_idx, :nb].copy_(self._splits_k[i, :nb])
                runner.norms_v_buf[layer_idx, req_idx, :nb].copy_(self._splits_v[i, :nb])

        return self._nz_mask[0].shape[0], _v.shape[1]

    # ------------------------------------------------------------------
    # LSH deduplication path (< 4 concurrent prefills)
    # ------------------------------------------------------------------

    def _lsh_dedup_layer(self, layer_name, layer_idx, layer_meta, kv_layer):
        """LSH dedup + normalisation for one fusion layer."""
        runner      = self._runner
        head_dim    = self._head_dim
        lsh_proj    = self._lsh_proj
        kv_dtype    = self._kv_dtype

        if layer_name not in runner.lsh_registry:
            runner.lsh_registry[layer_name] = [dict() for _ in range(NUM_LSH_TABLES)]
            runner.lsh_mean_k[layer_name]   = {}
        registry_tables = runner.lsh_registry[layer_name]
        mean_k_store    = runner.lsh_mean_k[layer_name]

        # Block table IS layer-specific; reuse pre-allocated norm buffers.
        block_table = layer_meta.block_table[self._req_idx_to_compress].clone()
        self._splits_k.fill_(1.0)
        self._splits_v.fill_(1.0)

        for i, req_id in enumerate(self._req_to_compress):
            nz_blocks = int(self._nz_blocks[i])   # precomputed, no per-layer division
            if nz_blocks == 0:
                continue

            raw_ids    = block_table[i, :nz_blocks]
            valid_mask = raw_ids >= 0
            valid_bids = raw_ids[valid_mask].long()
            valid_pos  = valid_mask.nonzero(as_tuple=True)[0]
            N = valid_bids.shape[0]
            if N == 0:
                continue

            # ---- Batched GPU: norms + mean-K + LSH fingerprints ----
            block_ks = kv_layer[0, valid_bids]   # [N, blk_sz, kv_heads, head_dim]
            block_vs = kv_layer[1, valid_bids]
            norms_k  = block_ks.float().view(N, -1).norm(dim=1)
            norms_v  = block_vs.float().view(N, -1).norm(dim=1)

            block_means      = block_ks.float().view(N, -1, head_dim).mean(dim=1)
            block_means_norm = F.normalize(block_means, dim=1).to(torch.float16)

            proj_all       = (block_means_norm @ lsh_proj).sign()
            proj_bits      = (proj_all > 0).to(torch.int32).cpu()
            proj_bytes     = proj_bits.view(N, NUM_LSH_TABLES, LSH_BITS_PER_TABLE)
            sub_hashes_all = (proj_bytes * _LSH_POWERS).sum(dim=2).tolist()

            # Write norms into pre-allocated buffers (GPU→GPU, no alloc)
            self._splits_k[i, valid_pos] = norms_k.clamp(min=1e-6).to(self._splits_k.dtype)
            self._splits_v[i, valid_pos] = norms_v.clamp(min=1e-6).to(self._splits_v.dtype)

            valid_bids_list = valid_bids.tolist()
            valid_pos_list  = valid_pos.tolist()

            # ---- Registry lookup (CPU loop; GPU matmul batched per block) ----
            to_register = []
            for k, (bid, pos) in enumerate(zip(valid_bids_list, valid_pos_list)):
                sub_hashes = sub_hashes_all[k]
                candidates = set()
                for t, h in enumerate(sub_hashes):
                    candidates.update(registry_tables[t].get(h, []))
                candidates.discard(bid)

                matched_bid = None
                if candidates:
                    cid_list = [c for c in candidates if c in mean_k_store]
                    if cid_list:
                        stored_stack       = torch.stack([mean_k_store[c] for c in cid_list])
                        sims               = (block_means_norm[k].float() @ stored_stack.T.float())
                        best_val, best_idx = sims.max(dim=0)
                        if best_val.item() > THRESHOLD:
                            matched_bid = cid_list[best_idx.item()]

                if matched_bid is not None:
                    block_table[i, pos] = matched_bid
                else:
                    to_register.append((k, bid, block_means_norm[k], sub_hashes))

            # ---- Batched GPU normalisation for unmatched blocks ----
            if to_register and len(mean_k_store) < MAX_REGISTRY_PER_LAYER:
                local_idxs         = [t[0] for t in to_register]
                unmatched_bids_gpu = valid_bids[local_idxs]
                norms_k_sub        = norms_k[local_idxs].view(-1, 1, 1, 1).clamp(min=1e-6)
                norms_v_sub        = norms_v[local_idxs].view(-1, 1, 1, 1).clamp(min=1e-6)
                kv_layer[0, unmatched_bids_gpu] = (block_ks[local_idxs] / norms_k_sub).to(kv_dtype)
                kv_layer[1, unmatched_bids_gpu] = (block_vs[local_idxs] / norms_v_sub).to(kv_dtype)

                for _, bid, mean_k, sub_hashes in to_register:
                    if len(mean_k_store) >= MAX_REGISTRY_PER_LAYER:
                        break
                    mean_k_store[bid] = mean_k
                    for t, h in enumerate(sub_hashes):
                        registry_tables[t].setdefault(h, []).append(bid)

            n_redir = nz_blocks - len(to_register)
            logger.info(
                "LSH dedup %s req=%s: %d/%d blocks redirected",
                layer_name, req_id, n_redir, nz_blocks,
            )

            if req_id not in runner.fused_requests:
                runner.fused_requests[req_id] = {}
            runner.fused_requests[req_id][layer_name] = (
                self._splits_k[i].clone(), self._splits_v[i].clone()
            )

        # Update norms_k_buf so patched_forward can scale this step's decode.
        if runner.norms_k_buf is not None:
            nb = min(self._num_blocks_dim, runner.norms_k_buf.shape[2])
            for i, req_idx in enumerate(self._req_idx_to_compress):
                runner.norms_k_buf[layer_idx, req_idx, :nb].copy_(self._splits_k[i, :nb])
                runner.norms_v_buf[layer_idx, req_idx, :nb].copy_(self._splits_v[i, :nb])

        # Write redirected block IDs back to the attention-metadata GPU buffer.
        layer_meta.block_table[self._req_idx_to_compress] = block_table

    def run_lsh(self):
        """Run LSH dedup + registration across all fusion layers."""
        if self._num_blocks_dim == 0 or self._lsh_proj is None:
            return
        runner = self._runner
        for kv_gid, kv_group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if kv_gid == 0:
                continue
            layer_names = kv_group.layer_names
            if not layer_names:
                continue
            layer_name = layer_names[0]
            if layer_name not in self._meta_dict:
                continue
            layer_idx  = int(layer_name.split('.')[2])
            layer_meta = self._meta_dict[layer_name]
            kv_layer   = runner.kv_caches[kv_gid]
            self._lsh_dedup_layer(layer_name, layer_idx, layer_meta, kv_layer)

    def _lsh_register_layer(self, layer_name, layer_meta, kv_layer):
        """Register BFF-merged (already normalised) blocks for one fusion layer."""
        runner      = self._runner
        head_dim    = self._head_dim
        lsh_proj    = self._lsh_proj

        if layer_name not in runner.lsh_registry:
            runner.lsh_registry[layer_name] = [dict() for _ in range(NUM_LSH_TABLES)]
            runner.lsh_mean_k[layer_name]   = {}
        registry_tables = runner.lsh_registry[layer_name]
        mean_k_store    = runner.lsh_mean_k[layer_name]

        for i, req_id in enumerate(self._req_to_compress):
            req_idx   = self._req_idx_to_compress[i]
            nz_blocks = int(self._nz_blocks[i])   # precomputed
            if nz_blocks == 0 or len(mean_k_store) >= MAX_REGISTRY_PER_LAYER:
                continue

            raw_ids    = layer_meta.block_table[req_idx, :nz_blocks]
            valid_mask = raw_ids >= 0
            valid_bids = raw_ids[valid_mask].long()
            if valid_bids.shape[0] == 0:
                continue

            bid_list = valid_bids.tolist()
            new_bids = [b for b in bid_list if b not in mean_k_store]
            if not new_bids:
                continue

            new_bids_t   = torch.tensor(new_bids, dtype=torch.long, device=kv_layer.device)
            block_ks     = kv_layer[0, new_bids_t]
            M            = len(new_bids)
            block_means  = block_ks.float().view(M, -1, head_dim).mean(dim=1)
            means_norm   = F.normalize(block_means, dim=1).to(torch.float16)

            proj_all       = (means_norm @ lsh_proj).sign()
            proj_bits      = (proj_all > 0).to(torch.int32).cpu()
            proj_bytes     = proj_bits.view(M, NUM_LSH_TABLES, LSH_BITS_PER_TABLE)
            sub_hashes_all = (proj_bytes * _LSH_POWERS).sum(dim=2).tolist()

            for k, bid in enumerate(new_bids):
                if len(mean_k_store) >= MAX_REGISTRY_PER_LAYER:
                    break
                mean_k_store[bid] = means_norm[k]
                for t, h in enumerate(sub_hashes_all[k]):
                    registry_tables[t].setdefault(h, []).append(bid)

    def run_lsh_register(self):
        """Register BFF blocks in the LSH registry across all fusion layers."""
        if self._num_blocks_dim == 0 or self._lsh_proj is None:
            return
        runner = self._runner
        for kv_gid, kv_group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if kv_gid == 0:
                continue
            layer_names = kv_group.layer_names
            if not layer_names:
                continue
            layer_name = layer_names[0]
            if layer_name not in self._meta_dict:
                continue
            layer_meta = self._meta_dict[layer_name]
            kv_layer   = runner.kv_caches[kv_gid]
            self._lsh_register_layer(layer_name, layer_meta, kv_layer)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        if self._num_blocks_dim == 0:
            return
        runner = self._runner
        for kv_gid, kv_group in enumerate(runner.kv_cache_config.kv_cache_groups):
            if kv_gid == 0:
                continue
            layer_names = kv_group.layer_names
            if not layer_names:
                continue
            layer_name = layer_names[0]
            if layer_name not in self._meta_dict:
                continue
            layer_idx  = int(layer_name.split('.')[2])
            layer_meta = self._meta_dict[layer_name]
            kv_layer   = runner.kv_caches[kv_gid]

            n_valid, n_unique = self._compress_layer(layer_name, layer_idx, layer_meta, kv_layer)
            self._total_valid  += n_valid
            self._total_unique += n_unique

            n_freed = n_valid - n_unique
            if n_freed > 0:
                logger.info(
                    "Post-forward BFF %s: %d valid → %d unique (%.2fx, freed %d)",
                    layer_name, n_valid, n_unique,
                    n_valid / n_unique if n_unique else 1.0, n_freed,
                )

        total_freed  = self._total_valid - self._total_unique
        overall_ratio = (self._total_valid / self._total_unique
                         if self._total_unique else 1.0)
        logger.info(
            "BFF summary: %d reqs, %d valid → %d unique across all layers "
            "(%.2fx, %d freed)",
            self.B, self._total_valid, self._total_unique, overall_ratio, total_freed,
        )


def _run_post_forward_bff(
    self,
    req_to_compress: list[str],
    req_idx_to_compress: list[int],
    attn_metadata_dict,
) -> None:
    """Run BFF on every fusion layer after the forward pass.  Delegates to BFFCompressor."""
    if len(req_idx_to_compress) < 2:
        return
    BFFCompressor(self, req_to_compress, req_idx_to_compress, attn_metadata_dict).run()


# ---------------------------------------------------------------------------
# LSH deduplication: per-request block normalisation + registry
# ---------------------------------------------------------------------------

_LSH_POWERS = torch.tensor([1 << b for b in range(LSH_BITS_PER_TABLE)], dtype=torch.int32)


def _lsh_fingerprint(mean_k_norm: torch.Tensor, lsh_proj: torch.Tensor) -> list:
    """Return 8 independent 8-bit sub-hashes for a single normalised mean-K vector.

    Args:
        mean_k_norm: [head_dim] float16, unit-norm.
        lsh_proj:    [head_dim, NUM_LSH_BITS] float16 random projections.
    Returns:
        list of NUM_LSH_TABLES ints in [0, 255].
    """
    proj_bits = ((mean_k_norm @ lsh_proj).sign() > 0).to(torch.int32)  # [64]
    packed = (proj_bits.view(NUM_LSH_TABLES, LSH_BITS_PER_TABLE) * _LSH_POWERS).sum(dim=1)
    return packed.tolist()


def _run_lsh_dedup_and_register(
    self,
    req_to_process: list,
    req_idx_to_process: list,
    attn_metadata_dict,
) -> None:
    """LSH dedup + registration for every fusion layer. Delegates to BFFCompressor."""
    if not req_to_process:
        return
    BFFCompressor(self, req_to_process, req_idx_to_process, attn_metadata_dict).run_lsh()


def _lsh_register_bff_blocks(
    self,
    req_to_compress: list,
    req_idx_to_compress: list,
    attn_metadata_dict,
) -> None:
    """Register BFF-merged blocks in the LSH registry. Delegates to BFFCompressor."""
    if not req_to_compress:
        return
    BFFCompressor(self, req_to_compress, req_idx_to_compress, attn_metadata_dict).run_lsh_register()


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
    set_forward_context,
    get_forward_context,
)
# from kv_fast_fusion.fast_fusion_context import patched_set_forward_context as set_forward_context
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.gpu_model_runner import (
    ExecuteModelState,
    PerLayerAttnMetadata, 
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.attention.backends.utils import (
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
    split_attn_metadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.spec_decode.eagle import EagleProposer
from copy import copy, deepcopy

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
  
    self._updated_block_tables =  request_blocks if request_blocks else None

def _patched_build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        fused_reqs_idx: list[int] | None = None,
        fused_reqs: list[str] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        # Attention metadata is not needed for attention free models
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        assert num_reqs_padded is not None and num_tokens_padded is not None

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        kv_cache_groups = self.kv_cache_config.kv_cache_groups

        def _get_block_table_and_slot_mapping(kv_cache_gid: int):
            assert num_reqs_padded is not None and num_tokens_padded is not None
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                blk_table_tensor = torch.zeros(
                    (num_reqs_padded, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
            blk_table_tensor[num_reqs:num_reqs_padded].fill_(-1)

            return blk_table_tensor, slot_mapping

        block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
        if self.model_config.enable_return_routed_experts:
            self.slot_mapping = slot_mapping_gid_0[:num_tokens].cpu().numpy()
        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
            seq_lens=self.seq_lens.gpu[:num_reqs_padded],
            _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
            _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
                :num_reqs_padded
            ],
            num_reqs=num_reqs_padded,
            num_actual_tokens=num_tokens_padded,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_gid_0,
            slot_mapping=slot_mapping_gid_0,
            causal=True,
        )

        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.cpu[num_reqs:].fill_(0)
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs_padded)

            cm_base.dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs_padded]
            cm_base.dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[
                :num_reqs_padded
            ]

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # Cache attention metadata builds across hybrid KV-cache groups
        # The only thing that changes between different hybrid KV-cache groups when the
        # same metadata builder and KVCacheSpec is the same is the block table, so we
        # can cache the attention metadata builds and just update the block table using
        # `builder.update_block_table` if the builder supports it.
        cached_attn_metadata: dict[
            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
        ] = {}

        def _build_attn_group_metadata(
            kv_cache_gid: int,
            attn_gid: int,
            common_attn_metadata: CommonAttentionMetadata,
            ubid: int | None = None,
        ) -> None:
            attn_group = self.attn_groups[kv_cache_gid][attn_gid]
            builder = attn_group.get_metadata_builder(ubid or 0)
            kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = kv_cache_spec.kv_cache_specs[attn_group.layer_names[0]]
            cache_key = (kv_cache_spec, type(builder))

            cascade_attn_prefix_len = (
                cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                if cascade_attn_prefix_lens
                else 0
            )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                assert ubid is None, "UBatching not supported with GDN yet"
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                    num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                        :num_reqs_padded
                    ],
                )

            if for_cudagraph_capture:
                attn_metadata_i = builder.build_for_cudagraph_capture(
                    common_attn_metadata
                )
            elif (
                cache_key in cached_attn_metadata
                and builder.supports_update_block_table
            ):
                attn_metadata_i = builder.update_block_table(
                    cached_attn_metadata[cache_key],
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                )
            else:
                attn_metadata_i = builder.build(
                    common_prefix_len=cascade_attn_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args,
                )
                if builder.supports_update_block_table:
                    cached_attn_metadata[cache_key] = attn_metadata_i


            # sefi — attach static norm-buffer views for graph-compatible decode scaling
            if len(attn_group.layer_names) == 1:
                layer_name = attn_group.layer_names[0]
                layer_idx = int(layer_name.split('.')[2])
                warmup = getattr(self, '_ff_warmup_layers', 2)
                max_layer = getattr(self, '_ff_max_layer_idx',
                                    self.vllm_config.model_config.get_num_layers(
                                        self.vllm_config.parallel_config) - warmup)
                is_fusion_layer = warmup <= layer_idx < max_layer

                if is_fusion_layer and hasattr(self, 'norms_k_buf') and self.norms_k_buf is not None:
                    nr_view = num_reqs_padded if num_reqs_padded is not None else num_reqs
                    max_blocks_buf = self.norms_k_buf.shape[2]
                    # Store views into the persistent buffer — same storage, graph-capturable
                    attn_metadata_i.norms_k_buf = self.norms_k_buf[layer_idx, :nr_view, :max_blocks_buf]
                    attn_metadata_i.norms_v_buf = self.norms_v_buf[layer_idx, :nr_view, :max_blocks_buf]
                    # Flag for patched_flash_forward: skip scaling when nothing is fused.
                    # Do NOT force True during graph capture — the warmup dummy run uses
                    # the maximum batch size (e.g. 256 reqs × 256 blocks) and the
                    # gather-multiply would try to allocate ~8 GiB → OOM.
                    # In PIECEWISE graph mode (the default) patched_flash_forward runs
                    # eagerly every step, so this Python flag is re-evaluated at each
                    # call and is safe to be False during warmup.
                    attn_metadata_i.has_fused_reqs = bool(fused_reqs)

            
            if ubid is None:
                assert isinstance(attn_metadata, dict)
                attn_metadata_dict = attn_metadata
            else:
                assert isinstance(attn_metadata, list)
                attn_metadata_dict = attn_metadata[ubid]

            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = attn_metadata_i

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            # Basically only the encoder seq_lens, block_table and slot_mapping change
            # for each kv_cache_group.
            cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs_padded,
            )
            if kv_cache_gid > 0:
                cm.block_table_tensor, cm.slot_mapping = (
                    _get_block_table_and_slot_mapping(kv_cache_gid)
                )

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                if ubatch_slices is not None:
                    for ubid, _cm in enumerate(split_attn_metadata(ubatch_slices, cm)):
                        _build_attn_group_metadata(kv_cache_gid, attn_gid, _cm, ubid)

                else:
                    _build_attn_group_metadata(kv_cache_gid, attn_gid, cm)

        if self.is_mm_prefix_lm:
            req_doc_ranges = {}
            for req_id in self.input_batch.req_ids:
                image_doc_ranges = []
                req_state = self.requests[req_id]
                for mm_feature in req_state.mm_features:
                    pos_info = mm_feature.mm_position
                    img_doc_range = pos_info.extract_embeds_range()
                    image_doc_ranges.extend(img_doc_range)
                req_idx = self.input_batch.req_id_to_index[req_id]
                req_doc_ranges[req_idx] = image_doc_ranges

            if isinstance(attn_metadata, list):
                for ub_metadata in attn_metadata:
                    for _metadata in ub_metadata.values():
                        _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
            else:
                for _metadata in attn_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

        if spec_decode_common_attn_metadata is not None and (
            num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
        ):
            # Currently the drafter still only uses piecewise cudagraphs (and modifies
            # the attention metadata in directly), and therefore does not want to use
            # padded attention metadata.
            spec_decode_common_attn_metadata = (
                spec_decode_common_attn_metadata.unpadded(num_tokens, num_reqs)
            )

        return attn_metadata, spec_decode_common_attn_metadata

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

            ## sefi
            self._updated_block_tables = None

            # Evict entries for requests that have left the batch (prevents unbounded growth)
            active_ids = set(self.input_batch.req_ids)
            for rid in [k for k in self.fused_requests if k not in active_ids]:
                del self.fused_requests[rid]

            req_to_compress = [
                req_id for req_id in self.input_batch.req_ids
                if (scheduler_output.num_scheduled_tokens.get(req_id, 0) > 1 and
                    req_id not in self.fused_requests)
            ]
            # Keep the full list; BFF still requires >=4, but the LSH path handles any count.
            req_idx_to_compress = [self.input_batch.req_id_to_index[r] for r in req_to_compress]
            fused_reqs = [req_id for req_id in self.input_batch.req_ids if req_id in self.fused_requests]
            fused_reqs_idx = [self.input_batch.req_id_to_index[r] for r in fused_reqs]

            # Skip norm-buffer fill/reset when no fused requests are present AND the
            # previous step also had none (avoids pointless fill_(1.0) every step).
            has_fused = bool(fused_reqs)
            if has_fused or getattr(self, '_had_fused_reqs_prev_step', False):
                self._fill_norm_buffers(fused_reqs, fused_reqs_idx)
            self._had_fused_reqs_prev_step = has_fused
            ##

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
                    fused_reqs_idx=fused_reqs_idx,
                    fused_reqs=fused_reqs,
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

        # Post-forward block fusion / deduplication — OUTSIDE set_forward_context.
        _n_fusion_layers = sum(
            1 for g in self.kv_cache_config.kv_cache_groups[1:] if g.layer_names
        )
        if False: # len(req_to_compress) > 1:
            # Full BFF: recursive merge across concurrent prefills, then register blocks.
            self._run_post_forward_bff(req_to_compress, req_idx_to_compress, attn_metadata)
            self._update_block_tables_after_compression(req_to_compress)
            # self._lsh_register_bff_blocks(req_to_compress, req_idx_to_compress, attn_metadata)
            logger.info(
                "BFF post-forward: compressed %d requests across %d fusion layers",
                len(req_to_compress), _n_fusion_layers,
            )
        elif len(req_to_compress) >= 1:
            # LSH dedup: normalise + look up registry for each individual prefill.
            self._run_lsh_dedup_and_register(req_to_compress, req_idx_to_compress, attn_metadata)
            self._update_block_tables_after_compression(req_to_compress)
            logger.info(
                "LSH dedup post-forward: processed %d requests across %d fusion layers",
                len(req_to_compress), _n_fusion_layers,
            )

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
        )
        self.kv_connector_output = kv_connector_output
        return None

from vllm.v1.structured_output.utils import apply_grammar_bitmask
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


