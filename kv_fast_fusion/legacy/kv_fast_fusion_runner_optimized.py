#fast_fusion_graph_runner_async.py  
import torch  
import os  
import queue  
import threading  
import copy  
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, NamedTuple  
from dataclasses import dataclass  
  
from vllm.logger import init_logger  
logger = init_logger("vllm.vllm_patch")  
  
from kv_fast_fusion.compression_hook import CompressionHook  
import torch.nn.functional as F  
  
THRESHOLD = 0.75  
BLOCK_SIZE = 128  
NUM_LAST_CHUNKS_TO_COMPRESS = 4   
CHUNK_SIZE = 512  
  
class AsyncCompressionWorker:  
    """Background worker for async KV cache compression using recursive method"""  
    def __init__(self, device, thr, fused_requests):  
        self.device = device  
        self.compression_queue = queue.Queue()  
        self.compression_stream = torch.cuda.Stream()
        self.thr = thr 
        self.fused_requests = fused_requests  
        self._block_size = BLOCK_SIZE  
      
        # CRITICAL: Initialize shutdown_event BEFORE starting thread  
        self.shutdown_event = threading.Event()  
          
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)  
        self.worker_thread.start()  
        self.compression_results = {}

    def _setup_masks(self, attn_metadata):  
        """Setup masks and indices in worker thread"""  
        req_idx = attn_metadata.req_idx_to_compress  
        self.B = len(req_idx)  
        self.num_blocks = attn_metadata.block_table.shape[-1]  
        self.device = attn_metadata.block_table.device  
        self.block_table = attn_metadata.block_table[req_idx]          
        self.seq_lens_buffer = attn_metadata.seq_lens[req_idx].unsqueeze(-1)  
        self.idx__ = torch.arange(self.num_blocks, dtype=torch.int, device=self.device)  
        self.b_idx = list(range(self.B))  
        self.mask = self.idx__.repeat(self.B,1) < (self.seq_lens_buffer[:self.B] // BLOCK_SIZE)  
        self.block_table_masked = self.block_table[self.mask]  
        self.nz_mask = self.mask.nonzero(as_tuple=True)  
        mask_split = self.mask.sum(-1)  
        self.mask_split_cumsum = torch.cumsum(mask_split, -1)  
        self.mask_split = mask_split.tolist()  
        self.max_split_len = mask_split.max()  
        self.splits_k = torch.ones(self.B, self.num_blocks, dtype=torch.bfloat16)  
        self.splits_v = torch.ones(self.B, self.num_blocks, dtype=torch.bfloat16)  
        self.nz_blocks = (self.seq_lens_buffer // BLOCK_SIZE).squeeze()

    def _worker_loop(self):  
        """Background thread processing compression tasks"""  
        with torch.cuda.stream(self.compression_stream):  
            while not self.shutdown_event.is_set():  
                try:  
                    task = self.compression_queue.get(timeout=0.1)  
                    if task is None:  # Shutdown signal  
                        break  
                    
                    layer_name, kv_cache, attn_metadata, callback = task  
                    
                    # Validate inputs  
                    if kv_cache is None or attn_metadata is None:  
                        logger.warning(f"Invalid compression task for layer {layer_name}")  
                        continue  
                    
                    result = self._process_compression_recursive(layer_name, kv_cache, attn_metadata)  
                    
                    # Store result for main thread to retrieve  
                    if result is not None:  
                        self.compression_results[layer_name] = result  
                    
                    if callback:  
                        try:  
                            callback()  
                        except Exception as e:  
                            logger.error(f"Callback failed for layer {layer_name}: {e}")  
                            
                    self.compression_queue.task_done()  
                    
                except queue.Empty:  
                    continue  
                except Exception as e:  
                    logger.error(f"Worker thread error: {e}")  
                    continue
        
    def _process_compression_recursive(self, layer_name, kv_cache, attn_metadata):  
        """Process compression using recursive method without circular dependency"""  
        try:  
            # Setup masks directly in worker  
            self._setup_masks(attn_metadata)  
            
            # Run recursive fusion with worker's state  
            result = self.layer_fast_fusion_recursive(layer_name, kv_cache, attn_metadata) 
            return result 
        except Exception as e:  
            logger.error(f"Compression failed for layer {layer_name}: {e}")  
            return None    
    
    # def layer_fast_fusion_recursive(self, layer_name, kv_layer, attn_metadata):  
    #     """recursive fusion method (unchanged)"""  
          
    #     def fuse_all_above_thr(x,  
    #                            b_idx: list,                              
    #                            thr: float = self.thr):  
    #         """recursive method"""  
    #         B, _, _ = x.shape  
              
    #         if B == 1:              
    #             nz_blocks = self.nz_blocks[b_idx[0]]  
    #             return F.normalize(x[:, :nz_blocks], dim=-1, eps=1e-7), [self.idx__[:nz_blocks]], [], [nz_blocks]  
                  
    #         xl, _idx_l, fl_chain, shifts_l = fuse_all_above_thr(x[:B//2], b_idx[:B//2], thr=thr)  
    #         xr, _idx_r, fr_chain, shifts_r = fuse_all_above_thr(x[B//2:], b_idx[B//2:], thr=thr)  
  
    #         nl = xl.shape[1]  
    #         nr = xr.shape[1]  
    #         idx_l = self.idx__[:nl]  
    #         idx_r = self.idx__[:nr]  
                  
    #         idx_ll, idx_rr = (xl @ xr.mT > thr).nonzero(as_tuple=True)[-2:]  
    #         l_idx, c = torch.unique(idx_ll, return_counts=True)  
    #         r_idx = idx_rr.split(tuple(c.tolist()))  
              
    #         idx_ul = list(set(idx_l.tolist()) - set(idx_ll.tolist()))  
    #         idx_ur = list(set(idx_r.tolist()) - set(idx_rr.tolist()))  
              
    #         n_c = len(l_idx)  
    #         n_ul = len(idx_ul)  
    #         n_ur = len(idx_ur)          
              
    #         combined_tensors = [torch.cat([xl[:,l_idx[i]].unsqueeze(1),xr[:,r_idx[i]]], dim=1).mean(1, keepdim=True) for i in range(n_c)]  
    #         if combined_tensors != []:  
    #             combined_x = F.normalize(torch.cat(combined_tensors, dim=1), dim=-1)  
    #             combined_x = torch.cat([combined_x, xl[:,idx_ul], xr[:,idx_ur]], dim=1)  
    #         else:  
    #             combined_x = torch.cat([xl[:,idx_ul], xr[:,idx_ur]], dim=1)  
  
    #         reverse_idx = torch.empty(nl+nr, device=x.device, dtype=torch.int)  
              
    #         reverse_idx[l_idx.tolist()] = self.idx__[:n_c]  
    #         for i in range(n_c):  
    #             reverse_idx[(r_idx[i]+nl).tolist()] = self.idx__[:n_c][i]  
  
    #         reverse_idx[idx_ul] = self.idx__[n_c:n_c + n_ul]  
    #         reverse_idx[list(map(lambda x: x + nl, idx_ur))] = self.idx__[n_c + n_ul:n_c + n_ul + n_ur]  
                      
    #         max_length = max(len(_idx_l), len(_idx_r))  
    #         if len(_idx_l) < max_length:  
    #             shifts_l += [shifts_l[-1]] * (max_length - len(_idx_l))  
    #             _idx_l += [torch.tensor([], device=xl.device, dtype=torch.int)   
    #                        for _ in range(max_length - len(_idx_l))]  
              
    #         chain = [torch.cat([_idx_l[i], _idx_r[i] + shifts_l[i]], dim=0) for i in range(max_length)]  
    #         reverse_idx = [reverse_idx]  
    #         reverse_idx += chain  
    #         fl_chain += fr_chain  
    #         fl_chain += [(b_idx, torch.stack([idx_ll, idx_rr], dim=-1), idx_ul, idx_ur)]  
              
    #         shifts = list(map(lambda x, y: x + y, shifts_l, shifts_r))  
    #         shifts = [n_c + n_ul + n_ur] + shifts  
  
    #         del x, xl, xr, idx_ll, idx_rr, l_idx, r_idx, idx_ul, idx_ur, n_c, n_ul, n_ur  
    #         return combined_x, reverse_idx, fl_chain, shifts  
          
    #     def fuse_values_with_above_thr_idx(v: torch.Tensor,  
    #                                        fwd_idx: dict,  
    #                                        b_idx: List[int],):  
    #         """Your original value fusion method"""  
    #         i = 0                       
    #         def recurssive_combining(v: torch.Tensor,  
    #                                  b_idx: List[int]):  
    #             nonlocal i  
    #             B, _, _ = v.shape  
    #             if B == 1:                   
    #                 nz_blocks = v.shape[1]  
    #                 return F.normalize(v[:, :nz_blocks], dim=-1, eps=1e-7)  
  
    #             vl = recurssive_combining(v[:B//2], b_idx[:B//2])  
    #             vr = recurssive_combining(v[B//2:], b_idx[B//2:])  
                  
    #             _, idx_, idx_ul, idx_ur = fwd_idx[i]  
    #             idx_ll, idx_rr = idx_.mT  
    #             l_idx, c = torch.unique(idx_ll, return_counts=True)  
    #             r_idx = idx_rr.split(tuple(c.tolist()))  
  
    #             combined_tensors = [torch.cat([vl[:,l_idx[i]].unsqueeze(1), vr[:,r_idx[i]]], dim=1).mean(1, keepdim=True) for i in range(len(l_idx))]  
    #             if combined_tensors != []:  
    #                 combined_v = F.normalize(torch.cat(combined_tensors, dim=1), dim=-1)  
    #                 combined_v = torch.cat([combined_v, vl[:,idx_ul], vr[:,idx_ur]], dim=1)  
    #             else:  
    #                 combined_v = torch.cat([vl[:,idx_ul], vr[:,idx_ur]], dim=1)  
  
    #             i += 1  
    #             del v, vl, vr, idx_ll, idx_rr, l_idx, r_idx, idx_ul, idx_ur  
    #             return combined_v                         
  
    #         vv = recurssive_combining(v, b_idx)  
    #         return vv       
          
    #     def restore_cache(x, idx, shape):  
    #         """Your original restore method"""  
    #         x = x.view(shape[0], shape[1], -1)  
    #         for i in range(1, len(idx)):  
    #             x = torch.cat([x, idx[i].view(shape[0], -1, x.shape[-1])], dim=1)  
    #         return x  
          
    #     # Extract and reshape KV cache  
    #     kv_shape = kv_layer[0, self.block_table_masked].shape  
    #     blocks, block_sz, num_head, head_size = kv_shape  
          
    #     # Process key cache  
    #     kk = kv_layer[0, self.block_table].view(self.B, self.num_blocks, -1)  
    #     k_norms = kk[self.nz_mask].norm(2, -1)  
          
    #     for i, k in enumerate(k_norms.split(self.mask_split)):  
    #         self.splits_k[i, :k.shape[0]] = k  
          
    #     # Run recursive fusion  
    #     # kk_masked = kk[self.mask]
    #     kk.view(self.B,self.num_blocks, -1)  
    #     compressed_k, reverse_idx, fl_chain, shifts = fuse_all_above_thr(kk_masked, self.b_idx, thr=self.thr)  
          
    #     # Restore cache structure  
    #     kk_restored = restore_cache(compressed_k, reverse_idx, kk_masked.shape)  
    #     kv_layer[0, self.block_table_masked] = kk_restored.view(kv_shape)  
          
    #     # Process value cache  
    #     vv = kv_layer[1, self.block_table].view(self.B, self.num_blocks, -1)  
    #     v_norms = vv[self.nz_mask].norm(2, -1)  
          
    #     for i, v in enumerate(v_norms.split(self.mask_split)):  
    #         self.splits_v[i, :v.shape[0]] = v  
          
    #     vv_masked = vv[self.mask]  
    #     compressed_v = fuse_values_with_above_thr_idx(vv_masked, fl_chain, self.b_idx)  
    #     vv_restored = restore_cache(compressed_v, reverse_idx, vv_masked.shape)  
    #     kv_layer[1, self.block_table_masked] = vv_restored.view(kv_shape)  
          
    #     # Update block tables  
    #     self._update_block_table(fl_chain, shifts)  
          
    #     # Store compression metadata  
    #     for idx, req in enumerate(attn_metadata.req_ids_to_compress):  
    #         if req not in self.fused_requests:  
    #             self.fused_requests[req] = {}  
    #         self.fused_requests[req][layer_name] = (  
    #             self.splits_k[idx], self.splits_v[idx])  
          
    #     return kv_layer  
    def layer_fast_fusion_recursive(self,
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
            
        kk = kv_layer[0, self.block_table]#.clone()

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
        
        _k, _idx, fwd_idx, _  = fuse_all_above_thr(kk, self.b_idx, thr)

        kk = restore_cache(_k, _idx, kk.shape)

        if is_chunks_fusion:
            kk =kk.view(num_last_chunks_to_compress,blocks_to_keep//num_last_chunks_to_compress, -1)
            # kk*= k_norms.unsqueeze(-1)
            kk = torch.cat([kk_cat,kk.view(blocks_to_keep, block_sz, num_head, head_size)], dim=0)

            kv_layer[0, self.block_table_masked] = (kk).view(kv_shape)
        # else:
        #     kv_layer[0, self.block_table_masked] = kk.view(kv_shape) #(kk * k_norms.unsqueeze(-1)).view(kv_shape)

        del _k 

        vv = kv_layer[1, self.block_table]#.clone() 

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
        # else:
            # kv_layer[1, self.block_table_masked] = vv.view(kv_shape) #(vv * v_norms.unsqueeze(-1)).view(kv_shape)

        #    
        # from vllm.forward_context import get_forward_context
        # forward_context = get_forward_context()  
        # attention_layer = forward_context.no_compile_layers[layer_name] 
        # compressed_kv = torch.stack([kk.view(kv_shape), vv.view(kv_shape)]) 
        # attention_layer.kv_cache[forward_context.virtual_engine] = compressed_kv

        compressed_ += [_v.shape[1]]
        if is_chunks_fusion:
            total_ += [num_last_chunks_to_compress*CHUNK_SIZE/self._block_size]
        else:
            total_ +=[blocks]
        
        # del vv, _v,  fwd_idx, _idx
        logger.info(f"Compression ratio in layer {layer_name}: {torch.tensor(total_).sum().item() / torch.tensor(compressed_).sum().item() if torch.tensor(compressed_).sum().item() > 0 else 0.0}")
        return #kk, vv, kv_shape

    def _recursive_fusion_with_state(self, layer_name, kv_cache, req_idx, block_table, seq_lens):  
        """Perform recursive fusion without creating new hook instances"""  
        try:  
            # Setup state directly without creating new hook  
            # B = len(req_idx)  
            # num_blocks = block_table.shape[-1]  
            # device = kv_cache.device  
            
            # Create masks  
            # seq_lens_buffer = seq_lens.unsqueeze(-1)  
            # idx_ = torch.arange(num_blocks, dtype=torch.int, device=device)  
            # mask = idx_.repeat(B, 1) < (seq_lens_buffer[:B] // BLOCK_SIZE)  
            
            # Extract KV blocks  
            kk = kv_cache[0].view(self.B, self.num_blocks, -1)  
            kk_masked = kk[self.mask]  
            
            # Your original recursive fusion logic  
            def fuse_all_above_thr(x, b_idx, thr=0.5):  
                B = x.shape[0]  
                if B == 1:  
                    return x, [torch.arange(x.shape[1], device=x.device)], [], [0]  
                
                xl, _idx_l, fl_chain, shifts_l = fuse_all_above_thr(x[:B//2], b_idx[:B//2], thr=thr)  
                xr, _idx_r, fr_chain, shifts_r = fuse_all_above_thr(x[B//2:], b_idx[B//2:], thr=thr)  
                
                nl = xl.shape[1]  
                nr = xr.shape[1]  
                idx_l = torch.arange(nl, device=x.device)  
                idx_r = torch.arange(nr, device=x.device)  
                
                idx_ll, idx_rr = (xl @ xr.mT > thr).nonzero(as_tuple=True)[-2:]  
                l_idx, c = torch.unique(idx_ll, return_counts=True)  
                r_idx = idx_rr.split(tuple(c.tolist()))  
                
                idx_ul = list(set(idx_l.tolist()) - set(idx_ll.tolist()))  
                idx_ur = list(set(idx_r.tolist()) - set(idx_rr.tolist()))  
                
                n_c = len(l_idx)  
                n_ul = len(idx_ul)  
                n_ur = len(idx_ur)  
                
                combined_tensors = [torch.cat([xl[:,l_idx[i]].unsqueeze(1),xr[:,r_idx[i]]], dim=1).mean(1, keepdim=True) for i in range(n_c)]  
                if combined_tensors != []:  
                    combined_x = F.normalize(torch.cat(combined_tensors, dim=1), dim=-1)  
                    combined_x = torch.cat([combined_x, xl[:,idx_ul], xr[:,idx_ur]], dim=1)  
                else:  
                    combined_x = torch.cat([xl[:,idx_ul], xr[:,idx_ur]], dim=1)  
                
                reverse_idx = torch.empty(nl+nr, device=x.device, dtype=torch.int)  
                reverse_idx[l_idx.tolist()] = torch.arange(n_c, device=x.device)  
                for i in range(n_c):  
                    reverse_idx[(r_idx[i]+nl).tolist()] = i  
                
                reverse_idx[idx_ul] = torch.arange(n_c, n_c + n_ul, device=x.device)  
                reverse_idx[list(map(lambda x: x + nl, idx_ur))] = torch.arange(n_c + n_ul, n_c + n_ul + n_ur, device=x.device)  
                
                max_length = max(len(_idx_l), len(_idx_r))  
                if len(_idx_l) < max_length:  
                    shifts_l += [shifts_l[-1]] * (max_length - len(_idx_l))  
                    _idx_l += [torch.tensor([], device=xl.device, dtype=torch.int) for _ in range(max_length - len(_idx_l))]  
                
                chain = [torch.cat([_idx_l[i], _idx_r[i] + shifts_l[i]], dim=0) for i in range(max_length)]  
                reverse_idx = [reverse_idx]  
                reverse_idx += chain  
                fl_chain += fr_chain  
                fl_chain += [(b_idx, torch.stack([idx_ll, idx_rr], dim=-1), idx_ul, idx_ur)]  
                
                shifts = list(map(lambda x, y: x + y, shifts_l, shifts_r))  
                shifts = [n_c + n_ul + n_ur] + shifts  
                
                return combined_x, reverse_idx, fl_chain, shifts  
            
            # Run fusion  
            compressed_k, reverse_idx, fl_chain, shifts = fuse_all_above_thr(kk_masked, list(range(self.B)), thr=0.5)  
            
            # Update KV cache  
            kv_shape = kv_cache[0, self.mask].shape  
            kk_restored = self._restore_cache(compressed_k, reverse_idx, kk_masked.shape)  
            kv_cache[0, self.mask] = kk_restored.view(kv_shape)  
            
            # Similar for value cache...  
            # (omitted for brevity)  
            
            return {  
                'compressed_kv': kv_cache,  
                'fl_chain': fl_chain,  
                'shifts': shifts  
            }  
            
        except Exception as e:  
            logger.error(f"Recursive fusion failed: {e}")  
            return None  
    
    def _restore_cache(self, x, idx, shape):  
        """Helper to restore cache structure"""  
        x = x.view(shape[0], shape[1], -1)  
        for i in range(1, len(idx)):  
            x = torch.cat([x, idx[i].view(shape[0], -1, x.shape[-1])], dim=1)  
        return x

    def enqueue_compression(self, layer_name, kv_cache, attn_metadata, callback=None):  
        """Queue compression task"""  
        if kv_cache is None or attn_metadata is None:  
            logger.warning("Skipping compression task with None inputs")  
            return  
          
        # Clone KV cache to avoid modifications during async processing  
        kv_clone = kv_cache#.clone() if kv_cache is not None else None  
        self.compression_queue.put((layer_name, kv_clone, attn_metadata, callback))  
      
    def get_result(self, layer_name):  
        """Get compression result if available"""  
        return self.compression_results.pop(layer_name, None)  
      
    def shutdown(self):  
        """Shutdown worker thread"""  
        self.shutdown_event.set()  
        self.compression_queue.put(None)  
        self.worker_thread.join(timeout=1.0)  
  
class BlockCompressionHookGraphAsync(CompressionHook):  
    """Async version of your BlockCompressionHookGraph using recursive method"""  
      
    def __init__(self, vllm_config, attn_metadata, fused_requests,   
                 max_batch_size=16, warmup_layers=2):  
        self.warmup_layers = warmup_layers  
        self.thr = 0.5  
        self._block_size = BLOCK_SIZE  
        # self.fused_requests = fused_requests  
        self.compression_events = {}  
        self.max_layer_idx = (vllm_config.model_config.get_num_layers(  
                vllm_config.parallel_config) - self.warmup_layers) 
          
        # Initialize async worker  
        self.async_worker = AsyncCompressionWorker(attn_metadata.block_table.device, 
                                                   self.thr,
                                                   fused_requests)  
      
    def start_layer_compression(self, layer_name: str, kv_cache, attn_metadata):  
        """Start async compression using recursive method"""  
        layer_idx = int(layer_name.split('.')[2])  
          
        if layer_idx < self.warmup_layers or layer_idx >= self.max_layer_idx:  
            return  
          
        # Setup masks if not done yet  
        # self._setup_masks(attn_metadata)  
          
        # Enqueue compression instead of immediate execution  
        self.async_worker.enqueue_compression(  
            layer_name, kv_cache, attn_metadata,  
            callback=lambda: self._on_compression_complete(layer_name)  
        )  
      
    def _on_compression_complete(self, layer_name):  
        """Callback when compression completes"""  
        event = torch.cuda.Event()  
        event.record(self.async_worker.compression_stream)  
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
      
    def __del__(self):  
        """Cleanup async worker"""  
        if hasattr(self, 'async_worker'):  
            self.async_worker.shutdown()