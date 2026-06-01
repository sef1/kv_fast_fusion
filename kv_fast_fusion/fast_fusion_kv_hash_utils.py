import hashlib  
from typing import Callable, Any  
from vllm.v1.core.kv_cache_utils import (  
    hash_block_tokens,   
    get_request_block_hasher,  
    BlockHash,  
    make_block_hash_with_group_id  
)  
from collections.abc import Callable, Iterable, Iterator, Sequence

def hash_block_ids(  
    hash_function: Callable[[Any], bytes],  
    parent_block_hash: BlockHash | None,  # Add this parameter  
    curr_block_token_ids: Sequence[int],  # Add this parameter    
    extra_keys: tuple[Any, ...] | None = None,  
) -> BlockHash:  
    """Hash based on block IDs instead of tokens."""  
    # Extract block IDs from the token_ids parameter  
    # Assuming you're passing block IDs through curr_block_token_ids  
    block_ids = list(curr_block_token_ids) if curr_block_token_ids else []  
      
    # Sort for consistency (optional)  
    sorted_ids = tuple(sorted(block_ids))  
    hash_bytes = hash_function((sorted_ids, extra_keys))  
    block_hash = BlockHash(hash_bytes)  
      
    # Use group_id 0 by default  
    return make_block_hash_with_group_id(block_hash, 1)

# def hash_block_ids(  
#     hash_function: Callable[[Any], bytes],  
#     block_ids: list[int],  
#     extra_keys: tuple[Any, ...] | None = None,  
# ) -> BlockHash:  
#     """Hash based on block IDs instead of tokens."""  
#     # Sort for consistency (remove if order matters)  
#     sorted_ids = tuple(sorted(block_ids))  
#     hash_bytes = hash_function((sorted_ids, extra_keys))  
#     return make_block_hash_with_group_id(BlockHash(hash_bytes), 0)  
  
def get_request_block_hasher_by_ids(  
    block_size: int,  
    caching_hash_fn: Callable[[Any], bytes],  
) -> Callable[[Any], list[BlockHash]]:  
    """Request hasher that uses block IDs instead of tokens."""  
      
    def request_block_hasher(request) -> list[BlockHash]:  
        # Get block IDs from the request's block mapping  
        # This requires access to the scheduler/manager  
        if not hasattr(request, '_merged_block_ids'):  
            return []  
          
        block_ids = request._merged_block_ids  
        if not block_ids:  
            return []  
              
        # Create hash from block IDs  
        block_hash = hash_block_ids(caching_hash_fn, block_ids)  
        return [block_hash]  
      
    return request_block_hasher  
  