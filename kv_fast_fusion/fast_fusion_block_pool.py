def patched_free_blocks(self, ordered_blocks):  
    """Free a list of blocks with deduplication by block_id."""  
    # Materialize the iterable to allow multiple passes.  
    seen = {}  
    blocks_list = list(ordered_blocks)  
    for block in blocks_list:  
        block.ref_cnt -= 1  
        seen[block.block_id] = block  
  
    unique_blocks = list(seen.values())  
    # self.free_block_queue.append_n([  
    #     block for block in unique_blocks  
    #     if block.ref_cnt == 0 and not block.is_null  
    # ])  
    for block in unique_blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.append(block)
            self._maybe_evict_cached_block(block) 

  
#   def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
#         """Free a list of blocks. The blocks should be ordered by their
#         eviction priority, where the first block will be evicted first.

#         Args:
#             ordered_blocks: A list of blocks to free ordered by their eviction
#                 priority.
#         """
#         # Materialize the iterable to allow multiple passes.
#         seen = {}
#         blocks_list = list(ordered_blocks)
#         for block in blocks_list:
#             block.ref_cnt -= 1
#             seen[block.block_id] = block

#         unique_blocks = list(seen.values())
#         ######
#         self.free_block_queue.append_n([
#             block for block in unique_blocks
#             if block.ref_cnt == 0 and not block.is_null
#         ])
