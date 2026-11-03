"""Who writes into a block somebody else is reading? (BFF_V2_SHARED_WRITE_CHECK)

With free-at-promotion on, a swapped request's victim goes back to the pool and the request points
at its representative, a block it now SHARES with the owner. Writing into a shared block is never
legitimate: prefix-cache sharing only covers full, already-computed blocks, and every forward write
lands at ``num_computed_tokens`` onward. A write into a block with ``ref_cnt > 1`` therefore
corrupts every other holder. One benign look-alike: on a FULL prefix-cache hit vLLM recomputes the
last prompt token into the shared last block, rewriting identical KV — a single-token span
``tokens=[N-1,N)`` on a request with ``roles=none``.

GLM Magicoder-4096, build 2026-11-03a: nothing-freed run 0/121 garbage, free-at-promotion 11/121.
Owners whose representatives were shared with 5-6 requests went bad while all of those readers stayed
fine, so the shared block was intact when they read it and something touched it afterwards. This
names the writer, at the step it is scheduled. It also logs every preemption and every
kv-consumer recompute (which frees without calling ``_preempt_request``) of a request in a swap,
since a resumed request rewriting its prompt is the leading way such a write could happen.

Pure functions plus the scheduler hooks; installed from ``fast_fusion_ascend_patch``.
"""

import os

from vllm.logger import init_logger

logger = init_logger("vllm.bff_shared_write_check")

# On by default: one dict lookup per scheduled request per group per step, a scan only on a hit.
ENABLED = os.environ.get("BFF_V2_SHARED_WRITE_CHECK", "1") == "1"
# Each (request, group, block) is reported once; a writer stuck on a shared block would otherwise
# log every step of its decode.
_SEEN_CAP = 65536


def write_span(num_computed: int, num_scheduled: int, block_size: int) -> range:
    """Block indices a step writes into: tokens ``[num_computed, num_computed + num_scheduled)``."""
    if num_scheduled <= 0 or block_size <= 0:
        return range(0)
    first = int(num_computed) // block_size
    last = (int(num_computed) + int(num_scheduled) - 1) // block_size
    return range(first, last + 1)


def shared_writes(blocks, num_computed: int, num_scheduled: int, block_size: int) -> list:
    """``[(pos, block)]`` for every block this step writes that another request also holds.

    ``blocks`` is the request's ``req_to_blocks`` row for one group. Positions past the row (not yet
    allocated) and null blocks are skipped; neither can be shared."""
    out = []
    for pos in write_span(num_computed, num_scheduled, block_size):
        if pos >= len(blocks):
            break
        blk = blocks[pos]
        if getattr(blk, "is_null", False) or getattr(blk, "ref_cnt", 0) <= 1:
            continue
        out.append((pos, blk))
    return out


def holders_of(managers, gi: int, block_id: int, exclude: str) -> list:
    """Every OTHER request whose group-``gi`` row holds ``block_id``, as ``"rid@pos"``."""
    m = managers[gi]
    out = []
    for rid, row in m.req_to_blocks.items():
        if rid == exclude:
            continue
        for pos, b in enumerate(row):
            if b.block_id == block_id:
                out.append(f"{rid}@{pos}")
    return out


def _roles(scheduler, rid: str) -> str:
    from kv_fast_fusion_ascend.fast_fusion_ascend_patch import _ext_of
    got = scheduler.__dict__.get("_bff_swap_roles", {}).get(_ext_of(rid))
    return ",".join(sorted(got)) if got else "none"


def check_step(scheduler, scheduler_output) -> int:
    """Log every shared-block write ``scheduler_output`` is about to perform. Returns the count.

    Must run BEFORE ``_update_after_schedule`` advances ``num_computed_tokens``."""
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    seen = scheduler.__dict__.setdefault("_bff_shared_write_seen", set())
    n = 0
    for rid, n_sched in scheduler_output.num_scheduled_tokens.items():
        req = scheduler.requests.get(rid)
        if req is None:
            continue
        for gi, m in enumerate(managers):
            row = m.req_to_blocks.get(rid)
            if not row:
                continue
            bs = int(getattr(m, "block_size", 0) or 0)
            for pos, blk in shared_writes(row, req.num_computed_tokens, n_sched, bs):
                n += 1
                key = (rid, gi, blk.block_id)
                if key in seen:
                    continue
                if len(seen) >= _SEEN_CAP:
                    seen.clear()
                seen.add(key)
                logger.error(
                    "BFF shared-block write | req=%s g=%d pos=%d block=%d ref_cnt=%d tokens=[%d,%d) "
                    "roles=%s | other holders: %s", rid, gi, pos, blk.block_id, blk.ref_cnt,
                    req.num_computed_tokens, req.num_computed_tokens + n_sched,
                    _roles(scheduler, rid),
                    " ".join(holders_of(managers, gi, blk.block_id, rid)) or "none")
    for info in getattr(scheduler_output, "recomputed_reqs", None) or ():
        rid = getattr(info, "request_id", None)
        if rid is not None:
            logger.info("BFF recompute | req=%s roles=%s", rid, _roles(scheduler, rid))
            # Its next prefix lookup is a re-arrival; fast_fusion_block_pool logs that lookup.
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            from kv_fast_fusion_ascend.fast_fusion_ascend_patch import _ext_of
            _bp._RECOMPUTED_EXT.add(_ext_of(rid))
    return n


def landed_anomalies(own_blocks, consensus: dict, blocks) -> list:
    """``[(gi, victim, ref_cnt, in_own_table)]`` for every landed copy destination that is NOT a
    block only this request holds.

    The landing copy writes the representative into each victim on the worker, before promotion,
    so the scheduler-side write check cannot see it. ``consensus`` is ``{gi: {victim: (rep,
    owner)}}`` — what every rank copied into. A victim shared with another request (``ref_cnt >
    1``), or one the request no longer holds at all, means that copy overwrote someone else's KV."""
    out = []
    for gi, vs in consensus.items():
        mine = set(own_blocks[gi]) if gi < len(own_blocks) else set()
        for victim in vs:
            ref = blocks[victim].ref_cnt
            if ref > 1 or victim not in mine:
                out.append((int(gi), int(victim), int(ref), victim in mine))
    return out


def check_landed(scheduler, request, consensus) -> int:
    """Log :func:`landed_anomalies` for ``request`` at promotion. Returns the count."""
    if not ENABLED or not consensus:
        return 0
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    own = [[b.block_id for b in m.req_to_blocks.get(request.request_id, [])] for m in managers]
    blocks = scheduler.kv_cache_manager.coordinator.block_pool.blocks
    bad = landed_anomalies(own, consensus, blocks)
    for gi, victim, ref, mine in bad:
        logger.error("BFF landed-into-shared | req=%s g=%d victim=%d ref_cnt=%d in_own_table=%s "
                     "roles=%s | other holders: %s", request.request_id, gi, victim, ref, mine,
                     _roles(scheduler, request.request_id),
                     " ".join(holders_of(managers, gi, victim, request.request_id)) or "none")
    return len(bad)


def check_pull_targets(request, blocks) -> int:
    """Log any block the RDMA pull will write that another request also holds.

    ``blocks`` is the ``KVCacheBlocks`` handed to ``update_state_after_alloc``; the pull targets
    are its unhashed blocks. A hashed (prefix-hit) block is never a target, so a hit through the
    alias cannot be written this way — this line says whether that holds on the box."""
    if not ENABLED:
        return 0
    n = 0
    for gi, row in enumerate(getattr(blocks, "blocks", None) or ()):
        for pos, blk in enumerate(row):
            if (getattr(blk, "is_null", False) or blk.block_hash is not None
                    or blk.ref_cnt <= 1):
                continue
            n += 1
            logger.error("BFF pull-into-shared | req=%s g=%d pos=%d block=%d ref_cnt=%d",
                         request.request_id, gi, pos, blk.block_id, blk.ref_cnt)
    return n


def install(scheduler_cls) -> bool:
    """Wrap ``_update_after_schedule`` and ``_preempt_request`` on ``scheduler_cls``. Idempotent.

    The base class only: the async and recompute schedulers reach both through ``super()`` or
    inheritance."""
    if not ENABLED:
        return False
    sentinel = "_bff_shared_write_wrapped"
    orig_update = scheduler_cls._update_after_schedule
    if getattr(orig_update, sentinel, False):
        return False

    def _update(self, scheduler_output, _orig=orig_update):
        try:
            check_step(self, scheduler_output)
        except Exception as e:  # noqa: BLE001 - a diagnostic must never break scheduling
            logger.warning("BFF shared-write check failed: %s", e)
        return _orig(self, scheduler_output)

    orig_preempt = scheduler_cls._preempt_request

    def _preempt(self, request, *args, _orig=orig_preempt, **kwargs):
        try:
            logger.info("BFF preempt | req=%s computed=%d roles=%s", request.request_id,
                        request.num_computed_tokens, _roles(self, request.request_id))
        except Exception:  # noqa: BLE001
            pass
        return _orig(self, request, *args, **kwargs)

    setattr(_update, sentinel, True)
    setattr(_preempt, sentinel, True)
    scheduler_cls._update_after_schedule = _update
    scheduler_cls._preempt_request = _preempt
    return True
