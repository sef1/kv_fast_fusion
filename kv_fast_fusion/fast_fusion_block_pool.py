import os
from typing import Any

from vllm.distributed.kv_events import MEDIUM_GPU, BlockStored
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    ExternalBlockHash,
    generate_block_hash_extra_keys,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)

# Set by the GPUModelRunner __init__ patch so the free path (scheduler side, same
# EngineCore process for TP=1) can evict freed block IDs from the LSH dedup registry.
_ACTIVE_RUNNER = None

# When set, do NOT eagerly evict a freed block from the prefix cache on ref-0 free — keep it
# cached (stock vLLM does this lazy eviction) so a preempted request can recover it on resume
# instead of recomputing the prefill. Safe in raw/ratio (KV is not mutated, so the cached block
# stays correct); UNSAFE in norm (mutated KV must not be re-hit). Default therefore tracks the
# scale mode: keep in raw/ratio, eager-evict in norm. Env BFF_KEEP_FUSED_HASH overrides.
# NOTE (ROUND 45): recovery from this works (~144 tok/preempted-req, ~9% median TTFT) but does NOT
# move req/s on a capacity-bound workload — preemption recompute is not the throughput bottleneck.
_SCALE_MODE = os.environ.get("BFF_SCALE_MODE", "raw")
_KEEP_FUSED_HASH = os.environ.get(
    "BFF_KEEP_FUSED_HASH", "0" if _SCALE_MODE == "norm" else "1"
) == "1"

# ROUND 39 (lever 3): alias a merged request's fusion-group block hash to the LIVE
# representative block so a preempted request recovers its KV on resume via a prefix-hit
# (the rep outlives the merge-orphan), instead of recomputing the prefill. raw mode only
# (KV is unmutated → the rep's bytes are exactly what the merged request reads). Gated.
_ALIAS_ENABLED = os.environ.get("BFF_ALIAS_FUSED", "1") == "1"
# rep_block_id -> {alias keys (BlockHashWithGroupId) that point at this block}
_BLOCK_ALIASES: dict = {}
# Set once at patch time to the original BlockPool._maybe_evict_cached_block.
_ORIG_MAYBE_EVICT = None
# Debug counters (alias inserts vs drops) — confirm aliases are invalidated, not leaked.
_alias_inserts = 0
_alias_drops = 0

# ROUND 40 diagnostic (gated by BFF_HIT_DEBUG): per-group prefix-cache recovery for requests
# that have been preempted at least once. Distinguishes "no fusion → no aliases" (A) from
# "aliases fire but the un-aliased sliding-window warmup group truncates the cross-group min
# to 0" (B). Accumulators are summarized periodically by the scheduler's BFF sched log.
_HIT_DEBUG = os.environ.get("BFF_HIT_DEBUG", "0") == "1"
_ORIG_GET_COMPUTED = None
_resume_lookups = 0
_resume_recovered_tok = 0
# ROUND 41: PRE-min raw per-group contiguous-prefix hit (independent of the cross-group min),
# group_idx -> summed contiguous hit-block count over resumed lookups. group 0 = warmup.
_resume_raw_group_blocks: dict = {}


def patched_get_computed_blocks(self, request):
    """Wrap KVCacheManager.get_computed_blocks to record per-group prefix-cache recovery for
    resumed (previously-preempted) requests. Pure passthrough when BFF_HIT_DEBUG is off."""
    cb, n = _ORIG_GET_COMPUTED(self, request)
    if _HIT_DEBUG and getattr(request, "num_preemptions", 0) > 0:
        global _resume_lookups, _resume_recovered_tok
        _resume_lookups += 1
        _resume_recovered_tok += int(n)
        # PRE-min raw per-group probe: count each group's INDEPENDENT contiguous prefix hit so
        # warmup-vs-fusion truncation is visible (the returned cb is already trimmed to the
        # cross-group min, which hides which group is the bottleneck).
        try:
            bp = self.coordinator.block_pool
            ngroups = len(self.coordinator.kv_cache_config.kv_cache_groups)
            bhs = request.block_hashes
            for g in range(ngroups):
                cnt = 0
                for bh in bhs:
                    if bp.get_cached_block(bh, [g]):
                        cnt += 1
                    else:
                        break
                _resume_raw_group_blocks[g] = _resume_raw_group_blocks.get(g, 0) + cnt
        except Exception:
            pass
    return cb, n


def add_block_alias(block_pool, key, rep_block) -> None:
    """Register rep_block under an extra prefix-cache key and remember the alias so it
    can be dropped the instant rep_block is recycled/evicted (see
    patched_maybe_evict_cached_block)."""
    global _alias_inserts
    block_pool.cached_block_hash_to_block.insert(key, rep_block)
    _BLOCK_ALIASES.setdefault(rep_block.block_id, set()).add(key)
    _alias_inserts += 1


def patched_maybe_evict_cached_block(self, block) -> bool:
    """Wrap BlockPool._maybe_evict_cached_block: when a block is about to be recycled or
    evicted, first drop ANY alias keys that pointed at it (keyed by block_id, so this fires
    even after block.block_hash was already reset to None) — the mandatory staleness guard
    that prevents a later prefix-hit from reading a recycled representative. Then run the
    original eviction (which drops the block's own hash)."""
    global _alias_drops
    if _ALIAS_ENABLED:
        aliases = _BLOCK_ALIASES.pop(block.block_id, None)
        if aliases:
            for key in aliases:
                self.cached_block_hash_to_block.pop(key, block.block_id)
                _alias_drops += 1
    return _ORIG_MAYBE_EVICT(self, block)


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
    freed_ids = []
    for block in unique_blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.append(block)
            # Eagerly drop the prefix-cache hash UNLESS keeping it for preemption recovery
            # (A/B). Lazy eviction still happens at reuse time in get_new_blocks either way.
            if not _KEEP_FUSED_HASH:
                self._maybe_evict_cached_block(block)
            freed_ids.append(block.block_id)

    # Drop these now-recycled block IDs from the LSH dedup registry so later
    # prefills never redirect to (and read) stale KV. See evict_lsh_blocks.
    if freed_ids and _ACTIVE_RUNNER is not None:
        try:
            _ACTIVE_RUNNER.evict_lsh_blocks(freed_ids)
        except Exception:
            pass

  
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


def patched_cache_full_blocks(
    self,
    request,
    blocks,
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
) -> None:
    """Verbatim copy of stock BlockPool.cache_full_blocks (vllm/v1/core/block_pool.py),
    except the `assert blk.block_hash is None` becomes a skip. Stock vLLM assumes every
    block reaching this loop is fresh/exclusively-owned; BFF's block aliasing/fusion
    (add_block_alias, BFF_ALIAS_FUSED) deliberately shares physical blocks across
    requests, so a resumed (preempted-then-retried) request can prefix-hit an aliased
    representative block that already carries its original owner's hash. Skipping is
    correct here: the block is already in the prefix cache and reachable through its
    block table, so a fused/shared block shouldn't be re-registered under a different
    request's exact-token hash. The caller (cache_blocks) still sets
    num_cached_block = num_full regardless, so accounting stays consistent."""
    if num_cached_blocks >= num_full_blocks:
        return
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    assert len(request.block_hashes) >= num_full_blocks
    if block_size == self.hash_block_size:
        # Common case.
        block_hashes: BlockHashList = request.block_hashes
    else:
        # block_size is a multiple of hash_block_size. This happens when
        # different KV cache groups have different block sizes.
        assert block_size % self.hash_block_size == 0
        # Recalculate block_hashes at the granularity of block_size, using
        # the original block_hashes (at the granularity of hash_block_size).
        block_hashes = BlockHashListWithBlockSize(
            request.block_hashes, self.hash_block_size, block_size
        )

    new_block_hashes = block_hashes[num_cached_blocks:]
    new_hashes: list[ExternalBlockHash] | None = (
        [] if self.enable_kv_cache_events else None
    )
    for i, blk in enumerate(new_full_blocks):
        # Some blocks may be null blocks when enabling sparse attention like
        # sliding window attention, or Mamba models with prefix-caching in
        # align mode. We skip null blocks here.
        if blk.is_null:
            continue
        if blk.block_hash is not None:
            continue   # BFF: fused/shared block already cached — don't re-hash it
        block_hash = new_block_hashes[i]

        # Update and added the full block to the cache.
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        blk.block_hash = block_hash_with_group_id
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
        if new_hashes is not None:
            new_hashes.append(maybe_convert_block_hash(block_hash))

    if self.enable_kv_cache_events:
        if num_cached_blocks == 0:
            parent_block_hash: ExternalBlockHash | None = None
        else:
            parent_block_hash = maybe_convert_block_hash(
                block_hashes[num_cached_blocks - 1]
            )

        # Calculate token range for the blocks being cached
        start_token_idx = num_cached_blocks * block_size
        end_token_idx = num_full_blocks * block_size

        # Generate extra keys for each block individually.
        # Each block may have different extra_keys (e.g., different MM
        # features, or cache_salt only for the first block).
        # Skip null blocks to match the length of new_hashes.
        extra_keys_list: list[tuple[Any, ...] | None] = []
        curr_mm_idx = 0
        for i in range(num_cached_blocks, num_full_blocks):
            if blocks[i].is_null:
                continue
            block_start = i * block_size
            block_end = block_start + block_size
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)

        self.kv_event_queue.append(
            BlockStored(
                block_hashes=new_hashes,
                parent_block_hash=parent_block_hash,
                token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                block_size=block_size,
                lora_id=request.lora_request.adapter_id
                if request.lora_request
                else None,
                medium=MEDIUM_GPU,
                lora_name=request.lora_request.name
                if request.lora_request
                else None,
                extra_keys=extra_keys_list if extra_keys_list else None,
            )
        )
