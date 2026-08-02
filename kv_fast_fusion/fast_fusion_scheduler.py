import atexit
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from vllm.logger import init_logger
logger = init_logger("vllm.patched_scheduler")

# Decode-side GROUND TRUTH for fusion compression. The producer can only count redirect rows it
# SHIPPED, which overstates the win (a row frees a block only if the decode side resolves it: rep
# still resident, owner resident, merge not dropped). The block-pool delta below is what actually
# happened, so the benchmark reports a measurement instead of an upper bound. Dumped only when
# BFF_PD_STATS_DIR is set; the bff_decode_stats_* prefix keeps the producers' bff_stats_* globs
# from matching it. This module is shared with the NCCL/legacy patches, hence the env gate.
_PD_STATS_DIR = os.environ.get("BFF_PD_STATS_DIR")
_PD_STATS_EVERY = int(os.environ.get("BFF_PD_STATS_EVERY", "50"))
# Wall-clock backstop for the dump, mirroring the producer side. The event cadence alone silently
# truncates the ground truth: a con32 run produced 38 merge events, so only event 1 was ever written
# and the file reported blocks_freed_total=16 when the real figure was 785. Every derived compression
# number (the harness's "1.008x smaller ... realized 4.3%") was wrong by ~50x as a result.
_PD_STATS_MAX_AGE_S = float(os.environ.get("BFF_PD_STATS_MAX_AGE_S", "30"))
_bff_decode_stats = {"blocks_freed_total": 0, "merge_events": 0}
_bff_decode_dump_state = {"last_t": 0.0, "atexit": False}


# Per-merge freed-block-id trace. Gated: it appends to a JSONL file on EVERY merge event, which is
# the scheduler's hot path — it must not ride on _PD_STATS_DIR (which is always set when the
# benchmark collects stats) or it becomes unconditional per-merge file I/O.
_FF_AUDIT = os.environ.get("BFF_FF_AUDIT", "0") == "1"


def _bff_record_decode_free(freed: int, freed_ids: list | None = None) -> None:
    """Accumulate one merge event's real block-pool delta and periodically dump it.

    ``freed_ids`` (only used under BFF_FF_AUDIT) are the block ids handed to ``free_blocks`` this
    event — an upper bound, since the dedup in ``patched_free_blocks`` may skip some decrements —
    for tracing premature frees at high concurrency."""
    s = _bff_decode_stats
    s["blocks_freed_total"] += int(freed)
    s["merge_events"] += 1
    if not _PD_STATS_DIR:
        return
    if _FF_AUDIT and freed_ids:
        try:
            audit = os.path.join(_PD_STATS_DIR, f"bff_free_audit_{os.getpid()}.jsonl")
            with open(audit, "a") as f:
                f.write(json.dumps({
                    "event": s["merge_events"],
                    "freed": freed,
                    "block_ids": freed_ids[:20],       # cap: this is a trace, not a ledger
                    "total_freed_so_far": s["blocks_freed_total"],
                }) + "\n")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("BFF: could not write free audit log: %s", e)
    st = _bff_decode_dump_state
    if not st["atexit"]:
        st["atexit"] = True
        atexit.register(_bff_dump_decode_stats)
    now = time.monotonic()
    # Event cadence OR wall-clock backstop. A run whose merge count never reaches _PD_STATS_EVERY
    # would otherwise leave the event-1 snapshot on disk as if it were the final total.
    if (s["merge_events"] != 1 and s["merge_events"] % _PD_STATS_EVERY
            and now - st["last_t"] < _PD_STATS_MAX_AGE_S):
        return
    st["last_t"] = now
    _bff_dump_decode_stats()


def _bff_dump_decode_stats() -> None:
    """Write the cumulative decode-side free ledger. Also the atexit hook, so a clean shutdown always
    leaves the true totals on disk regardless of where the event cadence landed."""
    if not _PD_STATS_DIR:
        return
    try:
        path = os.path.join(_PD_STATS_DIR, f"bff_decode_stats_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pid": os.getpid(), **_bff_decode_stats}, f)
        os.replace(tmp, path)   # atomic — the reader never sees a half-written file
    except Exception as e:  # pragma: no cover - defensive (must never break scheduling)
        logger.warning("BFF: could not dump decode fuse stats: %s", e)

# Diagnostic toggle (orthogonal to BFF_SCALE_MODE): when set, evict fusion TARGET
# (shared) blocks from the prefix cache after a merge so future prefills can't
# prefix-hit and re-pin them. Tests whether prefix-retention of fused targets is what
# drives the KV-pool cliff at scale. See ROUND 19 in the plan file.
BFF_EVICT_FUSED = os.environ.get("BFF_EVICT_FUSED", "0") == "1"
# ROUND 39 (lever 3): alias a merged request's redirected fusion-block hash to the live
# representative block, so on resume from preemption the request prefix-hits the rep
# (which outlives its merge-orphan) instead of recomputing. raw mode only (KV unmutated).
_BFF_RAW = os.environ.get("BFF_SCALE_MODE", "raw") == "raw"
# Free redundant blocks for post-KV-load consumers that are not yet RUNNING (still WAITING behind the
# max_num_seqs cap). Those requests hold their receive blocks until they finish, so only freeing at
# RUNNING never relieves the receive-buffer backlog. Safe when keyed on num_computed_tokens>0 +
# num_cached_block membership (NOT status): the assert-crash the old RUNNING guard protected against
# only fires for genuinely preempted-and-freed requests (num_computed_tokens==0). Off → legacy
# RUNNING-only behavior (NCCL default until validated).
_BFF_FREE_PRERUNNING = os.environ.get("BFF_FREE_PRERUNNING", "1") == "1"
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, make_block_hash_with_group_id
def _handle_block_merging_with_counts_o(  
        self,   
        request_blocks: dict[str, dict[int, list[int]]],      
    ) -> None:  
        """Apply reference changes using pre-calculated counts."""  
        # if not request_blocks:
        #     return
            
        before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
        
        # Using defaultdict for counting block IDs
        old_ref_counts = defaultdict(int)
        new_ref_counts = defaultdict(int)

        for req_id, group_blocks in request_blocks.items():  
            if req_id not in self.requests or not group_blocks:
                continue
            
            for group_idx, new_block_ids in group_blocks.items():  
                if group_idx == 0:  
                    continue  
                    
                manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
                req_blocks = manager.req_to_blocks.get(req_id, [])  
                
                # Count old references directly
                for block in req_blocks:
                    if not block.is_null and block.block_id != -1:
                        old_ref_counts[block.block_id] += 1

                # Build new blocks while filtering valid IDs
                new_req_blocks = []
                block_pool = self.kv_cache_manager.coordinator.block_pool
                
                for block_id in new_block_ids:    
                    block = block_pool.blocks[block_id]    
                    new_req_blocks.append(block)
                    if not block.is_null and block.block_id != -1:
                        new_ref_counts[block.block_id] += 1  # Count new references

                manager.req_to_blocks[req_id] = new_req_blocks

        # Calculate reference changes
        block_ref_changes = {}
        all_keys = set(new_ref_counts) | set(old_ref_counts)  # Combine keys efficiently

        for key in all_keys:
            block_ref_changes[key] = new_ref_counts[key] - old_ref_counts[key]

        blocks_to_touch = []  
        blocks_to_free = []  

        for block_id, ref_change in block_ref_changes.items():  
            block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  # Access block once
            
            if ref_change > 0:  
                blocks_to_touch.extend([block] * ref_change)  
            elif ref_change < 0:  
                # if block.prev_free_block is not None:  # Check if already in free queue
                #     logger.warning(f"Block {block_id} already in free queue, skipping free")  
                # else:
                blocks_to_free.extend([block] * abs(ref_change)) 

        # Apply batch operations if necessary
        if blocks_to_touch:
            self.kv_cache_manager.coordinator.block_pool.touch(blocks_to_touch)
        if blocks_to_free:
            self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free)
        
        after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
        logger.info(f"Block merging freed {after_free - before_free} blocks")  

def _handle_block_merging_with_counts____(    
    self,     
    request_blocks: dict[str, dict[int, list[int]]],        
) -> None:    
    """Apply reference changes using pre-calculated counts."""    
    before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()    
      
    blocks_to_touch = []    
    blocks_to_free = []    
    block_pool = self.kv_cache_manager.coordinator.block_pool  
      
    for req_id, group_blocks in request_blocks.items():    
        if req_id not in self.requests or not group_blocks:  
            continue  
          
        request = self.requests[req_id]  
          
        for group_idx, new_block_ids in group_blocks.items():    
            if group_idx == 0:    
                continue  
                  
            manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]    
            old_blocks = manager.req_to_blocks.get(req_id, [])  
              
            # Build new blocks and calculate ref changes in single pass  
            new_blocks = []  
            old_block_ids = set()  
              
            # Process old blocks  
            for block in old_blocks:  
                if not block.is_null and block.block_id != -1:  
                    old_block_ids.add(block.block_id)  
              
            # Process new blocks and calculate ref changes  
            new_block_ids_set = set()  
            for block_id in new_block_ids:      
                block = block_pool.blocks[block_id]      
                new_blocks.append(block)  
                new_block_ids_set.add(block_id)  
                  
                # Calculate ref change on the fly  
                if block_id not in old_block_ids:  
                    # New block - need to touch  
                    if not block.is_null and block.block_id != -1:  
                        blocks_to_touch.append(block)  
              
            # Find blocks to free (in old but not in new)  
            for block_id in old_block_ids - new_block_ids_set:  
                block = block_pool.blocks[block_id]  
                if not block.is_null and block.block_id != -1:  
                    blocks_to_free.append(block)  
              
            # Reset hashes only for blocks that changed  
            blocks_changed = (set(old_block_ids) != new_block_ids_set)  
            if blocks_changed:  
                for block in new_blocks:  
                    if not block.is_null:  
                        block.reset_hash()  
                  
                # Recompute hashes only if blocks actually changed  
                # num_full_blocks = len([b for b in new_blocks if not b.is_null])  
                # if num_full_blocks > 0:  
                    # manager.cache_blocks(request, num_full_blocks * manager.block_size)  
                manager.cache_blocks(request, request.num_tokens)
              
            manager.req_to_blocks[req_id] = new_blocks  
      
    # Apply batch operations  
    if blocks_to_touch:  
        block_pool.touch(blocks_to_touch)  
    if blocks_to_free:  
        block_pool.free_blocks(blocks_to_free)  
      
    after_free = block_pool.get_num_free_blocks()    
    logger.info(f"Block merging freed {after_free - before_free} blocks")

def _handle_block_merging_with_counts(self, request_blocks: dict[str, dict[int, list[int]]]) -> None:  
    """Apply reference changes using pre-calculated counts."""  
    if not request_blocks:  
        return  
          
    block_pool = self.kv_cache_manager.coordinator.block_pool  
    before_free = block_pool.get_num_free_blocks()  
      
    old_ref_counts = defaultdict(int)  
    new_ref_counts = defaultdict(int)  
    block_cache = {}  # Cache block objects to avoid repeated lookups

    for req_id, group_blocks in request_blocks.items():
        if req_id not in self.requests or not group_blocks:
            continue
        # The worker computed this redirect for the batch the request was in. With the
        # pipelined batch queue, by the time this runs the request may have been preempted
        # (freed → num_cached_block popped, num_computed_tokens reset to 0) or finished.
        # Re-applying req_to_blocks/num_cached_block to such a request desyncs the prefix
        # cache: on its re-schedule from WAITING it does a prefix lookup (non-empty
        # new_computed_blocks) while still in num_cached_block → the
        # `assert len(new_computed_blocks) == 0` crash in single_type_kv_cache_manager.
        req = self.requests[req_id]
        if _BFF_FREE_PRERUNNING:
            # Also free for post-KV-load consumers still WAITING behind the max_num_seqs cap (they
            # pin the receive-buffer backlog). Safe iff the request has computed tokens (so the
            # scheduler skips the prefix lookup → new_computed_blocks stays empty → assert can't
            # fire) and its KV load has finished (exclude WAITING_FOR_REMOTE_KVS). The per-group
            # num_cached_block check below confirms the cache is populated.
            if (req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                    or req.num_computed_tokens == 0):
                continue
        elif req.status != RequestStatus.RUNNING:
            # Legacy: only mutate scheduler state for requests that are still actively RUNNING.
            continue


        for group_idx, new_block_ids in group_blocks.items():
            if group_idx == 0:
                continue

            manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]
            # Skip if this group's cache isn't populated for the request (e.g. load not fully cached);
            # mutating req_to_blocks/num_cached_block without it risks the prefix-cache desync.
            if _BFF_FREE_PRERUNNING and req_id not in manager.num_cached_block:
                continue
            req_blocks = manager.req_to_blocks.get(req_id, [])
              
            # Count old references and cache blocks  
            for block in req_blocks:  
                if not block.is_null and block.block_id != -1:  
                    old_ref_counts[block.block_id] += 1  
                    block_cache[block.block_id] = block  
  
            # Build new blocks and count new references  
            new_req_blocks = []  
            for block_id in new_block_ids:  
                if block_id not in block_cache:  
                    block_cache[block_id] = block_pool.blocks[block_id]  
                block = block_cache[block_id]  
                new_req_blocks.append(block)  
                if not block.is_null and block.block_id != -1:
                    new_ref_counts[block.block_id] += 1

            manager.req_to_blocks[req_id] = new_req_blocks
            # After merge/dedup a block may already carry a hash (it now points at a
            # block another request cached). Mark the whole current block set as
            # cached so cache_full_blocks won't try to re-hash it and trip
            # `assert blk.block_hash is None`. New decode blocks (beyond this count)
            # still get cached normally.
            manager.num_cached_block[req_id] = len(new_req_blocks)

            # ROUND 39 (lever 3): for every REDIRECTED position, register the representative
            # block under THIS request's own prefix-cache key. On resume from preemption the
            # request's prefix lookup then hits the live rep (which outlives the merge-orphan)
            # instead of recomputing. raw mode only; the staleness guard lives in
            # fast_fusion_block_pool.patched_maybe_evict_cached_block (drops the alias the
            # instant the rep is recycled/evicted).
            if _BFF_RAW and getattr(block_pool, "enable_caching", False):
                try:
                    from kv_fast_fusion import fast_fusion_block_pool as _bp
                    if _bp._ALIAS_ENABLED:
                        bhs = self.requests[req_id].block_hashes
                        for i, blk in enumerate(new_req_blocks):
                            if i >= len(req_blocks) or i >= len(bhs):
                                break
                            # Skip nulls and unchanged (not-redirected) positions — only a
                            # redirect points at another request's still-live rep block.
                            if blk.is_null or blk.block_id == req_blocks[i].block_id:
                                continue
                            key = make_block_hash_with_group_id(bhs[i], group_idx)
                            _bp.add_block_alias(block_pool, key, blk)
                except Exception as e:
                    logger.warning("BFF alias-fused failed for %s: %s", req_id, e)

    # Calculate reference changes and prepare operations
    blocks_to_touch = []  
    blocks_to_free = []  
      
    all_keys = set(new_ref_counts) | set(old_ref_counts)  
    for block_id in all_keys:  
        ref_change = new_ref_counts[block_id] - old_ref_counts[block_id]  
        if ref_change > 0:  
            blocks_to_touch.extend([block_cache[block_id]] * ref_change)  
        elif ref_change < 0:  
            blocks_to_free.extend([block_cache[block_id]] * abs(ref_change))  
  
    # Apply batch operations
    if blocks_to_touch:
        block_pool.touch(blocks_to_touch)
    freed_ids = None
    if blocks_to_free:
        # Snapshot ids BEFORE freeing (the objects are recycled after) — audit only.
        if _FF_AUDIT:
            freed_ids = [b.block_id for b in blocks_to_free]
        block_pool.free_blocks(blocks_to_free)

    # Optionally evict the fusion TARGET (shared) blocks from the prefix cache so future
    # prefills can't prefix-hit and re-pin them (drops discoverability only; does NOT free
    # them — current holders read them correctly via req_to_blocks). ROUND 19 diagnostic.
    if BFF_EVICT_FUSED and blocks_to_touch:
        tgt_ids = {b.block_id for b in blocks_to_touch}
        block_pool.evict_blocks(tgt_ids)
        logger.info(f"BFF evicted {len(tgt_ids)} fusion-target blocks from prefix cache")

    after_free = block_pool.get_num_free_blocks()
    freed = after_free - before_free
    logger.info(f"Block merging freed {freed} blocks")
    _bff_record_decode_free(freed, freed_ids)

def _handle_block_merging_with_counts_(
    self,
    request_blocks: dict[str, dict[int, list[int]]],  
) -> None:  
    """Apply reference changes using pre-calculated counts and block ID hashing."""  
    if not request_blocks:  
        return  
          
    before_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
      
    # Track all changes  
    old_ref_counts = defaultdict(int)  
    new_ref_counts = defaultdict(int)  
    merged_requests = []  
      
    # First pass: update block mappings and track changes  
    for req_id, group_blocks in request_blocks.items():  
        if req_id not in self.requests or not group_blocks:  
            continue  
              
        merged_requests.append(req_id)  
          
        for group_idx, new_block_ids in group_blocks.items():  
            if group_idx == 0:  
                continue  
                  
            manager = self.kv_cache_manager.coordinator.single_type_managers[group_idx]  
            req_blocks = manager.req_to_blocks.get(req_id, [])  
              
            # Count old references  
            for block in req_blocks:  
                if not block.is_null and block.block_id != -1:  
                    old_ref_counts[block.block_id] += 1  
  
            # Build new blocks and count new references  
            new_req_blocks = []  
            block_pool = self.kv_cache_manager.coordinator.block_pool  
              
            for block_id in new_block_ids:  
                block = block_pool.blocks[block_id]  
                new_req_blocks.append(block)  
                if not block.is_null and block.block_id != -1:  
                    new_ref_counts[block.block_id] += 1  
  
            manager.req_to_blocks[req_id] = new_req_blocks  
      
    # Calculate and apply reference changes  
    block_ref_changes = {}  
    all_keys = set(new_ref_counts) | set(old_ref_counts)  
      
    for key in all_keys:  
        block_ref_changes[key] = new_ref_counts[key] - old_ref_counts[key]  
  
    blocks_to_touch = []  
    blocks_to_free = []  
      
    for block_id, ref_change in block_ref_changes.items():  
        block = self.kv_cache_manager.coordinator.block_pool.blocks[block_id]  
          
        if ref_change > 0:  
            blocks_to_touch.extend([block] * ref_change)  
        elif ref_change < 0:  
            blocks_to_free.extend([block] * abs(ref_change))  
      
    # Apply block operations  
    if blocks_to_touch:  
        self.kv_cache_manager.coordinator.block_pool.touch(blocks_to_touch)  
    if blocks_to_free:  
        self.kv_cache_manager.coordinator.block_pool.free_blocks(blocks_to_free)  
      
    # Update hashes for merged blocks  
    self._update_hashes_for_merged_blocks(request_blocks, merged_requests)  
      
    after_free = self.kv_cache_manager.coordinator.block_pool.get_num_free_blocks()  
    logger.info(f"Block merging freed {after_free - before_free} blocks")  
  
def _update_hashes_for_merged_blocks(  
    self,  
    request_blocks: dict[str, dict[int, list[int]]],  
    merged_requests: list[str]  
) -> None:  
    """Update block hashes using block ID-based hashing for merged blocks."""  
    block_pool = self.kv_cache_manager.coordinator.block_pool  
      
    for req_id in merged_requests:  
        request = self.requests[req_id]  
          
        for group_idx, new_block_ids in request_blocks[req_id].items():  
            if group_idx == 0 or not new_block_ids:  
                continue  
                  
            # Create hash from block IDs  
            block_id_hash = _create_block_id_hash(new_block_ids, group_idx)  
              
            # Update all blocks in the group  
            for i, block_id in enumerate(new_block_ids):  
                block = block_pool.blocks[block_id]  
                if not block.is_null:  
                    # Reset existing hash and set new block ID-based hash  
                    block.reset_hash()  
                    block.block_hash = block_id_hash  
                    block_pool.cached_block_hash_to_block.insert(  
                        block.block_hash, block  
                    )  
                  
                # Update request's block hashes  
                from vllm.v1.core.kv_cache_utils import get_block_hash  
                hash_value = get_block_hash(block_id_hash)  
                  
                if i < len(request.block_hashes):  
                    request.block_hashes[i] = hash_value  
                else:  
                    request.block_hashes.append(hash_value)  
  
def _create_block_id_hash(block_ids: list[int], group_idx: int = 0) -> BlockHashWithGroupId:  
    """Create a hash from block IDs."""  
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id  
    import hashlib  
      
    # Sort block IDs for consistency  
    sorted_ids = tuple(sorted(block_ids))  
    hash_bytes = hashlib.sha256(str(sorted_ids).encode()).digest()  
    block_hash = BlockHash(hash_bytes)  
      
    # Use group_id 0 by default  
    return make_block_hash_with_group_id(block_hash, group_idx)

from vllm.v1.core.sched.output import (
    SchedulerOutput,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.request import Request, RequestStatus
import numpy as np
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
import time

def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats
        # sefi
        # block_merge_mapping = model_runner_output.updated_block_table
        block_merge_mapping = getattr(model_runner_output, "_updated_block_tables", None)
        if not block_merge_mapping:
            # P/D path: the connector wrote the redirect map straight onto the runner. Under
            # async scheduling the sample_tokens output is a wrapper that drops the attached
            # attr, so read it from _ACTIVE_RUNNER directly (worker+scheduler share this process
            # at TP=1) and CONSUME it so it applies exactly once. Single-machine is unaffected:
            # its output carries the attr (this branch isn't taken) and is None when no fusion.
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                _runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                if _runner is not None:
                    block_merge_mapping = getattr(_runner, "_updated_block_tables", None)
                    if block_merge_mapping:
                        _runner._updated_block_tables = None
                        logger.info("BFF: block-merge via runner | reqs=%d",
                                    len(block_merge_mapping))
            except Exception as e:
                logger.warning("BFF runner block-merge fallback failed: %s", e)
        if not block_merge_mapping and kv_connector_output is not None:
            # TP>1 path (ROUND 52): worker and scheduler are SEPARATE processes, so the map can't
            # ride the in-process _ACTIVE_RUNNER global — it arrives on the connector stats carrier
            # (BFFMergeStats, set by the FF worker's get_kv_connector_stats). Every TP rank shipped
            # the identical all-reduced map, so the aggregated rank-0 copy is authoritative.
            try:
                _stats = getattr(kv_connector_output, "kv_connector_stats", None)
                _data = getattr(_stats, "data", None)
                if _data and _data.get("bff_merges"):
                    block_merge_mapping = _data["bff_merges"]
                    logger.info("BFF: block-merge via connector stats (TP>1) | reqs=%d",
                                len(block_merge_mapping))
            except Exception as e:
                logger.warning("BFF connector-stats block-merge fallback failed: %s", e)
        if block_merge_mapping:
            try:
                self._handle_block_merging_with_counts(block_merge_mapping)
            except Exception as e:
                logger.error("BFF block merging failed — skipping this step: %s", e, exc_info=True)
        ###
        # --- BFF measurement: periodic runtime scheduling snapshot ---
        # Separates "no free blocks" (static capacity cap) from "free blocks exist
        # but batch stays small" (runtime / preemption). K=50 steps.
        try:
            # Preemption gate (ROUND 38): each preemption resets num_computed_tokens=0, so the
            # request must re-prefill on resume minus whatever survives the prefix cache. BFF
            # eager-evicts merged blocks, so resume recovers little → recompute. Count it.
            _preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
            self._bff_preempt_step = len(_preempted)
            self._bff_preempt_total = getattr(self, "_bff_preempt_total", 0) + len(_preempted)
            _k = 50
            self._bff_step = getattr(self, "_bff_step", 0) + 1
            if self._bff_step % _k == 0:
                _bp = self.kv_cache_manager.coordinator.block_pool
                _free = _bp.get_num_free_blocks()
                _total = getattr(_bp, "num_gpu_blocks", None) or len(getattr(_bp, "blocks", []))
                _usage = (1.0 - _free / _total) if _total else float("nan")
                # ROUND 40: alias-fire counters disambiguate "no fusion → no aliases" (A) from
                # "aliases fire but warmup truncates recovery" (B).
                from kv_fast_fusion import fast_fusion_block_pool as _ffbp
                logger.info(
                    "BFF sched | step=%d | running=%d | waiting=%d | "
                    "free_blocks=%d / %d | block_usage=%.1f%% | preempt(cum)=%d | preempt(step)=%d "
                    "| alias_ins=%d | alias_drop=%d | hash_skip=%d",
                    self._bff_step, len(self.running), len(self.waiting),
                    _free, _total, _usage * 100,
                    self._bff_preempt_total, self._bff_preempt_step,
                    _ffbp._alias_inserts, _ffbp._alias_drops, _ffbp._alias_hash_skips,
                )
                # ROUND 41: per-group resume recovery (only when BFF_HIT_DEBUG). avg_recovered_tok
                # is the overall (post cross-group-min) hit; raw_group_prefix_blocks is each
                # group's INDEPENDENT contiguous prefix hit (group 0 = warmup) — if group 0 ≈ 0
                # while fusion groups > 0, the warmup group is what pins the min to 0.
                if _ffbp._HIT_DEBUG and _ffbp._resume_lookups:
                    _nl = _ffbp._resume_lookups
                    _avg_tok = _ffbp._resume_recovered_tok / _nl
                    _raw_grp = {gi: round(c / _nl, 2)
                                for gi, c in sorted(_ffbp._resume_raw_group_blocks.items())}
                    logger.info(
                        "BFF resume-recovery | preempted_lookups=%d | avg_recovered_tok=%.1f "
                        "| raw_group_prefix_blocks=%s",
                        _nl, _avg_tok, _raw_grp,
                    )
        except Exception as e:
            logger.warning("BFF sched log failed: %s", e, exc_info=True)
        # --- end BFF measurement ---
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            if stopped:
                if self.vllm_config.model_config.enable_return_routed_experts:
                    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
                    block_ids = kv_blocks.get_block_ids()[0]
                    num_tokens = request.num_tokens - 1

                    # compute slot mapping
                    block_ids_array = np.array(block_ids, dtype=np.int32)
                    num_blocks = len(block_ids)
                    block_size = self.block_size

                    # generate block offsets
                    block_offsets = np.arange(0, block_size)

                    # compute slot mapping: slot = block_id * block_size + offset
                    slot_mapping = (
                        block_offsets.reshape((1, block_size))
                        + block_ids_array.reshape((num_blocks, 1)) * block_size
                    ).flatten()[:num_tokens]

                    routed_experts = self.routed_experts_reader.get_routed_experts(
                        indices=slot_mapping
                    )
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

