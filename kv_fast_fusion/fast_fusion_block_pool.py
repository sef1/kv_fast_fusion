import os

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
