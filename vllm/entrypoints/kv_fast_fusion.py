import torch
import os
from vllm.sequence import ExecuteModelRequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.worker.worker_base import LocalOrDistributedWorkerBase, get_pp_group, get_tp_group
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.forward_context import set_forward_context
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.eagle import EagleProposer

import vllm.envs as envs
from vllm.utils import round_up
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from math import floor, log2


from types import MethodType
import time
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, overload
from vllm.logger import init_logger
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import json
import numpy as np
logger = init_logger("vllm.vllm_patch")

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


def execute_model_v0(
    self,
    execute_model_req: Optional[ExecuteModelRequest] = None,
) -> Optional[List[SamplerOutput]]:
    
    start_time = time.perf_counter()

    inputs = self.prepare_input(execute_model_req)
    if inputs is None:
        return None

    model_input, worker_input, kwargs = inputs
    num_steps = worker_input.num_steps

    self.execute_worker(worker_input)

    # If there is no input, we don't need to execute the model.
    if worker_input.num_seq_groups == 0:
        return []

    intermediate_tensors = None
    orig_model_execute_time = 0.0
    if not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group()))
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time):
            orig_model_execute_time = intermediate_tensors.tensors.get(
                "model_execute_time", torch.tensor(0)).item()
    
    # Batch Fast Fusion # 
    if hasattr(self, 'previous_is_prompt'):        
        if not model_input.is_prompt and self.previous_is_prompt:
            # if we are here, it means that we start decoing phase.
                
            bs = model_input.attn_metadata.block_tables.shape[0]          
            logger.info("BFF dealing with batch size %s ", bs)
            if bs > 1:
                block_tables = model_input.attn_metadata.block_tables
                seq_lens = torch.tensor(model_input.__dict__['seq_lens'], device=block_tables.device)
                NUM_LAST_CHUNKS_TO_COMPRESS = 8
                remove_position = False
                m = self.model_runner.__dict__['model']
                rotary_emb = m.model.layers[0].self_attn.rotary_emb 
                # call 
                cr, total, num_comp, _ = fast_fusion(self.kv_cache[worker_input.virtual_engine], model_input.attn_metadata.block_tables,  THRESHOLD, model_input.is_prompt, NUM_LAST_CHUNKS_TO_COMPRESS, remove_position, rotary_emb, seq_lens=seq_lens) 
                logger.info(f"compression {cr} | per layer compression {total/num_comp}")
                logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
                if not os.path.exists(f"compression_res"):
                    os.makedirs(f"compression_res")
                with open(f"compression_res/{bs}_batchsz_thr_{THRESHOLD}.jsonl", "a", encoding="utf-8") as f:
                    json.dump({"cr": cr, "per_layer": str((total/num_comp).tolist()), "num_comp_": str((num_comp).tolist())}, f, ensure_ascii=False)
                    f.write('\n')              
 
    # Execute the model with fused KV cache blocks #
    output = self.model_runner.execute_model(
        model_input=model_input,
        kv_caches=self.kv_cache[worker_input.virtual_engine]
        if self.kv_cache is not None else None,
        intermediate_tensors=intermediate_tensors,
        num_steps=num_steps,
        **kwargs,
    )

    self.previous_is_prompt = model_input.is_prompt

    bs = model_input.attn_metadata.block_tables.shape[0]
    # Chunks Fast Fusion - here we evaluate on single requrest #
    if bs == 1:
        if model_input.is_prompt:
            #If we are here, it means that we start prefill phase.
            seq_lens = torch.tensor(model_input.__dict__['seq_lens'], device=model_input.attn_metadata.block_tables.device)
            
            if hasattr(self, 'old_seq_len'):                
                if seq_lens.item() - self.old_seq_len == NUM_LAST_CHUNKS_TO_COMPRESS*CHUNK_SIZE:
                    m = self.model_runner.__dict__['model']
                    rotary_emb = m.model.layers[0].self_attn.rotary_emb 
                    remove_position = False
                    cr, total, num_comp, _ = fast_fusion(self.kv_cache[worker_input.virtual_engine], model_input.attn_metadata.block_tables,  THRESHOLD, model_input.is_prompt, NUM_LAST_CHUNKS_TO_COMPRESS, remove_position, rotary_emb, seq_lens=seq_lens) 
                    self.total_ += total
                    self.num_comp_ += num_comp
                    self.old_seq_len = seq_lens.item() 
            else: 
                self.old_seq_len = 0
                comopressed_layers_ = len(self.kv_cache[worker_input.virtual_engine]) - 4 # we have 2 warmup layers and 2 final layers 
                self.total_ = torch.zeros(comopressed_layers_)
                self.num_comp_ = torch.zeros(comopressed_layers_)
        else: # for the last prefill step
            self.old_seq_len = 0
            if self.total_.sum() > 1:
                logger.info(f"compression {self.total_.sum()/self.num_comp_.sum()} | per layer compression {self.total_/self.num_comp_}")
                logger.info(f"total blocks per layer {self.total_}| blocks per layer after compression {self.num_comp_}")
                if not os.path.exists(f"compression_res"):
                    os.makedirs(f"compression_res")
                with open(f"compression_res/{NUM_LAST_CHUNKS_TO_COMPRESS}_chunks_thr_{THRESHOLD}_wo_strip_pos.jsonl", "a", encoding="utf-8") as f:
                    json.dump({"cr": (self.total_.sum()/self.num_comp_.sum()).item(), "per_layer": str((self.total_/self.num_comp_).tolist()), "num_comp_": str((self.num_comp_).tolist())}, f, ensure_ascii=False)
                    f.write('\n')
                    #analyze the compression ratio by:
                    #data = pd.read_json("compression_res/4_chunks.jsonl", lines = True)
            comopressed_layers_ = len(self.kv_cache[worker_input.virtual_engine]) - 4 # we have 2 warmup layers and 2 final layers 
            self.total_ = torch.zeros(comopressed_layers_)
            self.num_comp_ = torch.zeros(comopressed_layers_)
           
    
    

    model_execute_time = time.perf_counter() - start_time
    if not get_pp_group().is_last_rank:
        # output is IntermediateTensors
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time):
            output.tensors["model_execute_time"] = torch.tensor(
                model_execute_time + orig_model_execute_time)
        get_pp_group().send_tensor_dict(output.tensors,
                                        all_gather_group=get_tp_group())
        return [None]
    if (self.observability_config is not None
            and self.observability_config.collect_model_execute_time
            and output is not None):
        for o in output:
            o.model_execute_time = (orig_model_execute_time +
                                    model_execute_time)

    # output is List[SamplerOutput]
    return output



def _update_block_tables_after_compression(  
    self,   
    new_requests: list[str],  
    block_tables_: list[torch.Tensor]  
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
            new_table = block_tables_[group_idx - 1][req_idx]  
            num_blocks = block_table_obj.num_blocks_per_row[req_idx]    #sefi tset
            # Update state and block table  
            
            final_blocks = new_table[:num_blocks].tolist()
            if all(block_table_obj.block_table_np[req_idx, :num_blocks] == final_blocks) :
                continue
            if req_id not in request_blocks.keys():
                request_blocks[req_id] = {}
            req_state.block_ids[group_idx][:num_blocks] = final_blocks[:num_blocks]  
            block_table_obj.block_table_np[req_idx, :num_blocks] = new_table[:num_blocks].cpu().numpy()  
            # block_table_obj.num_blocks_per_row[req_idx] = num_blocks  
            
            # Store final block list  
            request_blocks[req_id][group_idx] = final_blocks  
  
    return request_blocks

@torch.inference_mode()
def execute_model_v1(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:        
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)
            
        is_decode = all(scheduler_output.num_scheduled_tokens[req_id] == 1 for req_id in self.input_batch.req_ids)
        # if any(scheduler_output.num_scheduled_tokens[req_id] < 1 for req_id in self.input_batch.req_ids):
        #     raise ValueError("Error: num_scheduled_tokens less than 1")
        
            #  self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        
        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        per_layer_block_table = None
        if has_kv_transfer_group():

            # if is_decode:
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=False,
            ):
                # get_kv_transfer_group().set_input_batch(self.input_batch)``
                self.maybe_setup_bt_connector(scheduler_output)
                per_layer_block_table = get_kv_transfer_group().get_per_layer_block_table()

        if per_layer_block_table is not None: 
            # Prepare the decoder inputs.
            
            (attn_metadata, attention_cuda_graphs, logits_indices,
             spec_decode_metadata, num_scheduled_tokens_np,
              spec_decode_common_attn_metadata) = (
                self._prepare_inputs_with_block_tables(scheduler_output, per_layer_block_table)
                # self._prepare_inputs(scheduler_output)
            )
        else:
            (attn_metadata, attention_cuda_graphs, logits_indices,
            spec_decode_metadata, num_scheduled_tokens_np,
               spec_decode_common_attn_metadata) = (
             self._prepare_inputs(scheduler_output))        
        

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]

            model_kwargs = self._init_model_kwargs_for_multimodal_model(
                scheduler_output=scheduler_output)
            inputs_embeds = self.model.get_input_embeddings(
                input_ids=input_ids,
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = {}
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)
                

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        is_last_prefill = all(self.requests[req_id].num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id] <= self.requests[req_id].num_tokens 
                              for req_id in self.input_batch.req_ids)        
        
        # is_decode = all(scheduler_output.num_scheduled_tokens[req_id] == 1 for req_id in self.input_batch.req_ids)
        if is_decode:
            logger.info(f"D dealing with batch size {self.input_batch.num_reqs}")
        else:
            logger.info(f"P dealing with batch size {self.input_batch.num_reqs}")
            
        # is_last_prefill = all(scheduler_output.num_scheduled_tokens[req_id] == self.requests[req_id].num_tokens and self.requests[req_id].num_computed_tokens == 0
                                # for req_id in self.input_batch.req_ids)    
    
        # if not hasattr(self, 'fused_requests'):
        #     self.fused_requests = set()
        # request_ids = {req_id for req_id in self.input_batch.req_ids}
        # if not request_ids.issubset(self.fused_requests):
        if False: #is_last_prefill and not is_decode:        
            bs = self.input_batch.num_reqs
            is_chunks_fusion = True
            if bs > 1:
                is_chunks_fusion = False
            logger.info("P dealing with batch size %s ", bs)
            dict_key = next(iter(attn_metadata))
            block_tables = attn_metadata[dict_key].block_table
            seq_lens = attn_metadata[dict_key].seq_lens
            NUM_LAST_CHUNKS_TO_COMPRESS = 4
            remove_position = False
            # m = self.model_runner.__dict__['model']
            rotary_emb = self.model.model.layers[0].self_attn.rotary_emb
            # call 
            cr, total, num_comp, block_tables_, slot_maps_ = fast_fusion(self.kv_caches, block_tables,  0.7, is_chunks_fusion, NUM_LAST_CHUNKS_TO_COMPRESS, remove_position, rotary_emb, seq_lens=seq_lens) 
            logger.info(f"compression {cr} | per layer compression {total/num_comp}")
            logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
            for i,v in enumerate(attn_metadata.values()):
                if i >= len(block_tables_):  # skip the last two layers
                    break
                v.block_table = block_tables_[i+2]
                v.slot_mapping = slot_maps_[i+2]

            if not os.path.exists(f"compression_res"):
                os.makedirs(f"compression_res")
            with open(f"compression_res/{bs}_batchsz_thr_{THRESHOLD}.jsonl", "a", encoding="utf-8") as f:
                json.dump({"cr": cr, "per_layer": str((total/num_comp).tolist()), "num_comp_": str((num_comp).tolist())}, f, ensure_ascii=False)
                f.write('\n')     
    
        #sefi
        # if has_kv_transfer_group():
            # get_kv_transfer_group().set_scheduler_output(scheduler_output)
            # get_kv_transfer_group().set_attn_metadata(attn_metadata.copy())
            # get_kv_transfer_group().set_papare_func(self._prepare_inputs)
            
            
            # get_kv_transfer_group().set_input_batch(self.input_batch)
            # get_kv_transfer_group().set_fwd_context_params((self.vllm_config, num_input_tokens, num_tokens_across_dp, skip_cuda_graphs))

        
        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ):
            self.maybe_setup_kv_connector(scheduler_output)        

            # if is_decode:
            #     # block_table_ = get_kv_transfer_group()._block_table
            #     # num_updates = block_table_["model.layers.0.self_attn.attn"].shape[0]
            #     # num_reqs = attn_metadata["model.layers.0.self_attn.attn"].block_table.shape[0]
            #     # for i in range(2,30):
            #     #     layer_name = f"model.layers.{i}.self_attn.attn"
            #     #     attn_metadata[layer_name].block_table[-num_updates:] = block_table_[layer_name][-num_reqs:]  
            #     attn_metadata =  get_kv_transfer_group().get_attn_metadata()


            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **MultiModalKwargs.as_kwargs(
                    model_kwargs,
                    device=self.device,
                ),
            )
            
            

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        ######## sefi - nov
        
        blocks_count = {}  # Your logic to count shared blocks  
        blocks_to_free = []
        updated_block_tables = {}
        if False:
            bs = self.input_batch.num_reqs
            # is_chunks_fusion = True
            if not hasattr(self, 'fused_requests'):
                self.fused_requests = []
            req_to_compress = [req_id not in self.fused_requests for req_id in self.input_batch.req_ids]
            

            
            if sum(req_to_compress) > 1 and not is_decode:
                is_chunks_fusion = False
                logger.info(f"{"D" if is_decode else "P"} dealing with batch size %s ", sum(req_to_compress))
                dict_key = next(iter(attn_metadata))
                block_tables = [attn_metadata[k].block_table[req_to_compress] for k  in attn_metadata.keys()] #attn_metadata[dict_key].block_table
                seq_lens = attn_metadata[dict_key].seq_lens
                NUM_LAST_CHUNKS_TO_COMPRESS = 4
                remove_position = False
                # m = self.model_runner.__dict__['model']
                rotary_emb = self.model.model.layers[0].self_attn.rotary_emb
                # call 
                cr, total, num_comp, block_tables_  = fast_fusion(self.kv_caches[2:-2], block_tables[4:],  0.5, is_chunks_fusion, NUM_LAST_CHUNKS_TO_COMPRESS, remove_position, rotary_emb, seq_lens=seq_lens) 
                logger.info(f"compression {cr} | per layer compression {total/num_comp}")
                logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
                # Your logic to identify blocks to free  
                new_requests = [req_id for req_id in self.input_batch.req_ids if req_id not in self.fused_requests]
                self.fused_requests.append(new_requests)
                compressed_layers = total / num_comp > 1
                if cr > 1:
                    orig_blocks = set()
                    new_blocks = set()
                    # num_blocks = block_tables[0].shape[1]
                    # idx__ = torch.arange(bs*num_blocks, dtype=torch.int, device=seq_lens.device)
                    # count_mask = idx__[:num_blocks].to(seq_lens.device).repeat(bs,1) < (seq_lens//BLOCK_SIZE + ((seq_lens % BLOCK_SIZE) > 0)).unsqueeze(-1)
                    for req_id in new_requests:
                        for group_idx in range(len(block_tables[4:])):
                    # if compressed_layers[idx]:                        
                            
                            req_idx = self.input_batch.req_id_to_index[req_id]  
                            nz_idx = block_tables_[group_idx][req_idx].nonzero(as_tuple=True)[0] 
                            num_blocks = len(nz_idx)

                            block_table_obj = self.input_batch.block_table.block_tables[group_idx+1]  

                            orig_blocks.update(set(block_table_obj.block_table_cpu[req_idx].numpy()))
                            new_blocks.update(set(block_tables_[group_idx][req_idx].cpu().numpy()))
                            block_table_obj.block_table_cpu[req_idx, :num_blocks] = block_tables_[group_idx][req_idx][nz_idx].cpu()

                            
                            block_table_obj.num_blocks_per_row[req_idx] = num_blocks
                                                        
                            req_state = self.requests[req_id]
                            req_state.block_ids[group_idx+1][:] = block_tables_[group_idx][req_idx][nz_idx].tolist()

                            # blocks_to_free += list(set(self.input_batch.block_table.block_tables[group_idx+1].block_table_cpu[req_idx].numpy()) - set(block_tables_[group_idx][req_idx].cpu().numpy()))
                            new_bt_vals, new_bt_cnts = block_tables_[group_idx][req_idx].unique(return_counts=True)
                        # orig_bt_vals,orig_bt_cnts = self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs][count_mask.cpu()].unique(return_counts=True)
                            for entry in zip(new_bt_vals[new_bt_cnts > 1].tolist(), new_bt_cnts[new_bt_cnts > 1].tolist()):
                                blocks_count[entry[0]] = entry[1] 
                    
                    blocks_to_free = list(orig_blocks - new_blocks)

        if True:
            bs = self.input_batch.num_reqs
            # is_chunks_fusion = True
            if not hasattr(self, 'fused_requests'):
                self.fused_requests = []
            req_to_compress = [req_id not in self.fused_requests for req_id in self.input_batch.req_ids]
            
            if sum(req_to_compress) > 1 and not is_decode:
                is_chunks_fusion = False
                logger.info(f"{"D" if is_decode else "P"} dealing with batch size %s ", sum(req_to_compress))
                dict_key = next(iter(attn_metadata))
                block_tables = [attn_metadata[k].block_table[req_to_compress] for k  in attn_metadata.keys()] #attn_metadata[dict_key].block_table
                seq_lens = attn_metadata[dict_key].seq_lens
                NUM_LAST_CHUNKS_TO_COMPRESS = 4
                remove_position = False
                # m = self.model_runner.__dict__['model']
                rotary_emb = self.model.model.layers[0].self_attn.rotary_emb
                # call 
                cr, total, num_comp, block_tables_  = fast_fusion(self.kv_caches[2:-2], block_tables[4:],  0.7, is_chunks_fusion, NUM_LAST_CHUNKS_TO_COMPRESS, remove_position, rotary_emb, seq_lens=seq_lens) 
                logger.info(f"compression {cr} | per layer compression {total/num_comp}")
                logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
                # Your logic to identify blocks to free  
                new_requests = [req_id for req_id in self.input_batch.req_ids if req_id not in self.fused_requests]
                self.fused_requests.append(new_requests)
                compressed_layers = total / num_comp > 1
                if cr > 1:  
                    updated_block_tables = _update_block_tables_after_compression(self, new_requests, block_tables_)
                    # request_blocks = {}  
                    # for req_id in new_requests:  
                    #     req_idx = self.input_batch.req_id_to_index[req_id]  
                    #     req_state = self.requests[req_id]  
                        
                    #     for group_idx in range(len(self.input_batch.block_table.block_tables)):  
                    #         if group_idx == 0:  
                    #             continue  
                                
                    #         block_table_obj = self.input_batch.block_table.block_tables[group_idx]  
                    #         new_table = block_tables_[group_idx - 1][req_idx]  
                    #         # nz_idx = new_table.nonzero(as_tuple=True)[0]  
                    #         num_blocks = block_table_obj.num_blocks_per_row[req_idx] #len(nz_idx)  
                    #         # Update state and block table  
                    #         final_blocks = new_table[:num_blocks].tolist()
                    #         if all(block_table_obj.block_table_np[req_idx, :num_blocks] == final_blocks) :
                    #             continue
                    #         if req_id not in request_blocks.keys():
                    #             request_blocks[req_id] = {}
                    #         req_state.block_ids[group_idx][:] = final_blocks  
                    #         block_table_obj.block_table_np[req_idx, :num_blocks] = new_table[:num_blocks].cpu().numpy()  
                    #         # block_table_obj.num_blocks_per_row[req_idx] = num_blocks  
                            
                    #         # Store final block list  
                    #         request_blocks[req_id][group_idx] = final_blocks  
                    
                    # updated_block_tables = request_blocks
                                    # for req_id in new_requests:    
                    #     req_idx = self.input_batch.req_id_to_index[req_id]   
                           
                    #     req_state = self.requests[req_id]      
                        
                    #     # Process ALL KV cache groups (not just starting from 4)  
                    #     for group_idx in range(len(self.input_batch.block_table.block_tables)):  
                    #         # Skip group 0 if it's not used for KV cache  
                    #         if group_idx == 0:  
                    #             continue  
                                
                    #         block_table_obj = self.input_batch.block_table.block_tables[group_idx]    
                            
                    #         # Get old and new tables  
                    #         old_table = block_table_obj.block_table_np[req_idx]  # Use .np for numpy array  
                    #         new_table = block_tables_[group_idx - 1][req_idx]  # Adjust index since we skip group 0  
                    #         nz_idx = new_table.nonzero(as_tuple=True)[0]    
                            
                    #         # Update CachedRequestState    
                    #         req_state.block_ids[group_idx][:] = new_table[nz_idx].tolist()    
                            
                    #         # Update the block table's numpy array (THIS is the critical update)  
                    #         num_blocks = len(nz_idx)    
                            
                    #         # Track mapping for scheduler    
                    #         if updated_block_tables.get(req_id) is None:    
                    #                 updated_block_tables[req_id] = {}     
                    #         if updated_block_tables[req_id].get(group_idx) is None:    
                    #                 updated_block_tables[req_id][group_idx] = {}  
                    #         updated_block_tables[req_id][group_idx][tuple(old_table[:num_blocks].tolist())] = new_table[nz_idx].cpu().numpy()    
                    #         # for o, n in zip(old_table[:num_blocks], new_table[nz_idx].cpu().numpy()):    
                    #         #     if o != n:
                    #         #         if updated_block_tables.get(req_id) is None:    
                    #         #             updated_block_tables[req_id] = {} 
                    #         #         if updated_block_tables[req_id].get(group_idx) is None:    
                    #         #             updated_block_tables[req_id][group_idx] = {}    
                    #         #         updated_block_tables[req_id][group_idx][int(o)] = int(n)

                    #         block_table_obj.block_table_np[req_idx, :num_blocks] = new_table[nz_idx].cpu().numpy()  
                    #         block_table_obj.num_blocks_per_row[req_idx] = num_blocks
                    #         # block_table_obj.block_table_cpu[req_idx, :num_blocks] = new_table[nz_idx].cpu()    
                    #         # block_table_obj.block_table_gpu[req_idx, :num_blocks] = new_table[nz_idx].cuda()
                                        
                                        
                    #                     # for req_id in (new_requests):  
                    #                     #     req_idx = self.input_batch.req_id_to_index[req_id]  
                    #                     #     if updated_block_tables.get(req_id) is None:  
                    #                     #         updated_block_tables[req_id] = {}  
                    #                     #     req_state = self.requests[req_id]    
                                            
                    #                     #     for group_idx in range(len(block_tables[4:])):  
                    #                     #         # Get the BlockTable object for this group  
                    #                     #         block_table_obj = self.input_batch.block_table.block_tables[group_idx+1]  
                                                
                    #                     #         old_table = block_table_obj.block_table_cpu[req_idx]  # or .np if you need numpy  
                    #                     #         new_table = block_tables_[group_idx][req_idx]  
                    #                     #         nz_idx = block_tables_[group_idx][req_idx].nonzero(as_tuple=True)[0]  
                                                
                    #                     #         # Update CachedRequestState  
                    #                     #         req_state.block_ids[group_idx+1][:] = new_table[nz_idx].tolist()  
                                                
                    #                     #         # Update the block table's numpy array  
                    #                     #         num_blocks = len(nz_idx)  
                    #                     #         # block_table_obj.block_table.np[req_idx, :num_blocks] = new_table[nz_idx].cpu().numpy()                              
                                                
                    #                     #         # Track mapping for scheduler  
                    #                     #         if updated_block_tables[req_id].get(group_idx) is None:  
                    #                     #             updated_block_tables[req_id][group_idx] = {}  
                    #                     #         for o, n in zip(old_table[:num_blocks], new_table[nz_idx]):  
                    #                     #             if o.item() != n.item():  
                    #                     #                 updated_block_tables[req_id][group_idx][o.item()] = n.item()
                                                
                    #                     #         block_table_obj.block_table_cpu[req_idx, :num_blocks] = new_table[nz_idx].cpu()
                    #                     #         block_table_obj.np[req_idx, :num_blocks] = new_table[nz_idx].cpu().numpy()
                    #                     #         block_table_obj.num_blocks_per_row[req_idx] = num_blocks  
                                                
                                    
        #     # if cr > 1:
        #         for req_id in (new_requests):
        #             req_idx = self.input_batch.req_id_to_index[req_id]
        #             if updated_block_tables.get(req_id) is None:
        #                 updated_block_tables[req_id] = {}
        #             req_state = self.requests[req_id]  
        #             for group_idx in range(len(block_tables[4:])):  
        #                 old_table = self.input_batch.block_table.block_tables[group_idx+1].block_table_cpu[req_idx]  
        #                 new_table = block_tables_[group_idx][req_idx]  
        #                 nz_idx = block_tables_[group_idx][req_idx].nonzero(as_tuple=True)[0]  
                        
        #                 # Update CachedRequestState  
        #                 req_state.block_ids[group_idx+1][:] = new_table[nz_idx].tolist()  
                        
        #                 # CRITICAL: Also update InputBatch.block_table  
        #                 block_table = self.input_batch.block_table[group_idx+1]  
        #                 num_blocks = len(nz_idx)  
        #                 block_table.block_table.np[req_idx, :num_blocks] = new_table[nz_idx].cpu().numpy()  
        #                 block_table.num_blocks_per_row[req_idx] = num_blocks  
        #                 # self.input_batch.block_table.block_tables[group_idx+1].block_table_cpu[:bs][req_to_compress] = block_tables_[group_idx].cpu()
        #                 # Track mapping for scheduler  
        #                 if updated_block_tables[req_id].get(group_idx) is None:  
        #                     updated_block_tables[req_id][group_idx] = {}  
        #                 for o, n in zip(old_table, new_table):  
        #                     if o.item() != n.item():  
        #                         updated_block_tables[req_id][group_idx][o.item()] = n.item()
        #                                 # for group_idx in range(len(block_tables[4:])):
        #             #     old_table = self.input_batch.block_table.block_tables[group_idx+1].block_table_cpu[req_idx]
        #             #     new_table = block_tables_[group_idx][req_idx]
        #             #     nz_idx = block_tables_[group_idx][req_idx].nonzero(as_tuple=True)[0]
        #             #     req_state.block_ids[group_idx+1][:] = new_table[nz_idx].tolist()
        #             #     if updated_block_tables[req_id].get(group_idx) is None:
        #             #         updated_block_tables[req_id][group_idx] = {}
        #             #     for o,n in zip(old_table, new_table):
        #             #         if o.item() != n.item():
        #             #             updated_block_tables[req_id][group_idx][o.item()] = n.item()
        #         # num_blocks = block_tables[0].shape[1]
        #         # idx__ = torch.arange(bs*num_blocks, dtype=torch.int, device=seq_lens.device)
        #         # count_mask = idx__[:num_blocks].to(seq_lens.device).repeat(bs,1) < (seq_lens//BLOCK_SIZE + ((seq_lens % BLOCK_SIZE) > 0)).unsqueeze(-1)
        #         # # blocks_count = {}  # Your logic to count shared blocks  
        #         # # blocks_to_free = []
        #         # # orig_blocks = set().union(*(set(self.input_batch.block_table.block_tables[i + 1].block_table_cpu[:bs].view(-1).numpy()) for i in range(len(block_tables[4:]))))
        #         # # new_blocks = set().union(*(set(block_tables_[idx].view(-1).cpu().numpy()) for i in range(len(block_tables[4:]))))
        #         # orig_blocks = set()
        #         # new_blocks = set()
        #         # for idx in range(len(block_tables[4:])):
        #         #     # if compressed_layers[idx]:
        #         #     orig_blocks.update(set(self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs].view(-1).numpy()))
        #         #     new_blocks.update(set(block_tables_[idx].view(-1).cpu().numpy()))
        #         #     new_bt_vals, new_bt_cnts = block_tables_[idx][count_mask].unique(return_counts=True)
        #         #     # orig_bt_vals,orig_bt_cnts = self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs][count_mask.cpu()].unique(return_counts=True)

        #         #     for entry in zip(new_bt_vals[new_bt_cnts > 1].tolist(), new_bt_cnts[new_bt_cnts > 1].tolist()):
        #         #         blocks_count[entry[0]] = entry[1]    
        #         #     # free_mask = ~torch.isin(self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs].view(-1), block_tables_[idx].view(-1).cpu())
        #         #     # blocks_to_free += self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs].view(-1)[free_mask].tolist()
        #         #     # blocks_to_free += list(set(self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs][count_mask.cpu()].numpy()) - set(vals.cpu().numpy()))
        #         #     # blocks_to_free += list(set(self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs].view(-1).numpy()) - set(block_tables_[idx].view(-1).cpu().numpy()))


        #         #     self.input_batch.block_table.block_tables[idx+1].block_table_cpu[:bs][req_to_compress] = block_tables_[idx].cpu()
        #         # blocks_to_free = list(orig_blocks - new_blocks)
        #         # # for idx in range(len(new_requests)):
        #         # #     updated_block_tables[new_requests[idx]] = [bt[idx] for bt in block_tables_]
        # ###############
        # else:
        #     blocks_count = {}  # Your logic to count shared blocks  
        #     blocks_to_free = []

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                if finished_sending or finished_recving:
                    hidden_states.finished_sending = finished_sending
                    hidden_states.finished_recving = finished_recving
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, finished_sending,
                                  finished_recving)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            blocks_count=blocks_count,  
            blocks_to_free=blocks_to_free,
            updated_block_table=updated_block_tables, 
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )



@torch.inference_mode()
def ___execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)

        # Prepare the decoder inputs.
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata, num_scheduled_tokens_np,
         spec_decode_common_attn_metadata) = (
             self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]

            model_kwargs = self._init_model_kwargs_for_multimodal_model(
                scheduler_output=scheduler_output)
            inputs_embeds = self.model.get_input_embeddings(
                input_ids=input_ids,
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = {}
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        
        #sefi
        # is_prefill = False
        # if self.vllm_config.kv_transfer_config.kv_role=='kv_producer':
        #     is_prefill = True
    
        # if not is_prefill:
        #     bs = self.input_batch.num_reqs
        #     if bs > 1:
        #         logger.info("BFF dealing with batch size %s ", bs)
        #         dict_key = next(iter(attn_metadata))
        #         block_tables = attn_metadata[dict_key].block_table
        #         seq_lens = attn_metadata[dict_key].seq_lens
        #         NUM_LAST_CHUNKS_TO_COMPRESS = 8
        #         remove_position = False
        #         # m = self.model_runner.__dict__['model']
        #         rotary_emb = self.model.model.layers[0].self_attn.rotary_emb
        #         # call 
        #         cr, total, num_comp, block_tables_ = fast_fusion(self.kv_caches, block_tables,  THRESHOLD, is_prefill, NUM_LAST_CHUNKS_TO_COMPRESS, remove_position, rotary_emb, seq_lens=seq_lens) 
        #         logger.info(f"compression {cr} | per layer compression {total/num_comp}")
        #         logger.info(f"total blocks per layer {total}| blocks per layer after compression {num_comp}")
        #         for i,v in enumerate(attn_metadata.values()):
        #             if i >= len(block_tables_):  # skip the last two layers
        #                 break
        #             v.block_table = block_tables_[i+2]

        #         if not os.path.exists(f"compression_res"):
        #             os.makedirs(f"compression_res")
        #         with open(f"compression_res/{bs}_batchsz_thr_{THRESHOLD}.jsonl", "a", encoding="utf-8") as f:
        #             json.dump({"cr": cr, "per_layer": str((total/num_comp).tolist()), "num_comp_": str((num_comp).tolist())}, f, ensure_ascii=False)
        #             f.write('\n')              
        
        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ):
            self.maybe_setup_kv_connector(scheduler_output)

            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **MultiModalKwargs.as_kwargs(
                    model_kwargs,
                    device=self.device,
                ),
            )

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                if finished_sending or finished_recving:
                    hidden_states.finished_sending = finished_sending
                    hidden_states.finished_recving = finished_recving
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, finished_sending,
                                  finished_recving)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )

def replace_excute_model_with_compressed_excute_model(args, is_v1=False):     
    global THRESHOLD, NUM_LAST_CHUNKS_TO_COMPRESS, BLOCK_SIZE, CHUNK_SIZE
    BLOCK_SIZE = args.block_size
    CHUNK_SIZE = args.max_num_batched_tokens
    THRESHOLD = args.thr
    NUM_LAST_CHUNKS_TO_COMPRESS = args.num_chunks_to_compress       
    logger.info(f"Similarity threshold set to {THRESHOLD}")
    
    if not is_v1:
        LocalOrDistributedWorkerBase.execute_model = execute_model_v0
        logger.info("use compressed_excute_model v0")
        # GPUModelRunner.execute_model = execute_model
        # from types import MethodType
        # GPUModelRunner.execute_model = MethodType(execute_model, GPUModelRunner)
        # from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    #     GPUModelRunner.execute_model = execute_model_v1
    #     logger.info("use compressed_excute_model v1")
    # else:
       
    