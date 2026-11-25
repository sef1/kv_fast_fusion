"""Shared-block read audit: how much decode attention re-reads blocks another request also reads.

Opt-in with ``BFF_SHARE_AUDIT=N`` (audit every N steps; 0 = off). Pure CPU numpy on the runner's
block tables, so it never touches the GPU stream. Answers the cascade-attention ceiling question:

  redundant = sum over groups of L_g * (refs_g - unique_g)  / sum_g L_g * refs_g
      -> the fraction of KV block reads a perfect read-once kernel would remove;
  pairable  = same, but only counting overlaps of a greedy DISJOINT pairing of requests
      -> what a two-request grouped cascade (shared pass + own pass + merge) can capture.

Only decode requests (1 scheduled token) are counted; prefill chunks take the normal path.
"""
import os
from collections import defaultdict
from itertools import combinations

import numpy as np

from vllm.logger import init_logger

logger = init_logger("vllm.bff_share_audit")

SHARE_AUDIT_EVERY = int(os.environ.get("BFF_SHARE_AUDIT", "0"))
_MAX_LIST = 16  # cap pair enumeration per block (a block shared by >16 decode reqs is rare)

_totals = defaultdict(float)


def _group_stats(bt_np, nblocks, rows):
    """refs, unique, greedy-disjoint-pair saving, max refcount for one KV-cache group."""
    owners = defaultdict(list)
    refs = 0
    for r in rows:
        n = int(nblocks[r])
        refs += n
        for b in bt_np[r, :n].tolist():
            if b > 0:  # 0 = null block
                owners[b].append(r)
    unique = len(owners)
    pair_overlap = defaultdict(int)
    max_ref = 1
    for lst in owners.values():
        k = len(lst)
        if k < 2:
            continue
        max_ref = max(max_ref, k)
        for a, b in combinations(lst[:_MAX_LIST], 2):
            pair_overlap[(a, b)] += 1
    used, saving = set(), 0
    for (a, b), ov in sorted(pair_overlap.items(), key=lambda kv: -kv[1]):
        if a in used or b in used:
            continue
        used.add(a); used.add(b); saving += ov
    return refs, unique, saving, max_ref, len(used)


def maybe_audit(runner, scheduler_output, step: int) -> None:
    if not SHARE_AUDIT_EVERY or step % SHARE_AUDIT_EVERY:
        return
    try:
        ib = runner.input_batch
        rows = [ib.req_id_to_index[rid] for rid in ib.req_ids
                if scheduler_output.num_scheduled_tokens.get(rid, 0) == 1]
        if len(rows) < 2:
            return
        groups = runner.kv_cache_config.kv_cache_groups
        reads = redundant = pairable = 0.0
        max_ref, paired = 1, 0
        for g, bt in enumerate(ib.block_table.block_tables):
            L = len(groups[g].layer_names)
            refs, uniq, save, mref, npaired = _group_stats(bt.block_table.np, bt.num_blocks_per_row, rows)
            reads += L * refs
            redundant += L * (refs - uniq)
            pairable += L * save
            if g > 0:
                max_ref = max(max_ref, mref); paired = max(paired, npaired)
        if reads <= 0:
            return
        for k, v in (("reads", reads), ("redundant", redundant), ("pairable", pairable), ("n", 1)):
            _totals[k] += v
        logger.info(
            "BFF share-audit | step=%d | decode_reqs=%d | redundant=%.1f%% | pairable=%.1f%% | "
            "max_refcount=%d | paired_reqs=%d | cum_redundant=%.1f%% | cum_pairable=%.1f%%",
            step, len(rows), 100 * redundant / reads, 100 * pairable / reads, max_ref, paired,
            100 * _totals["redundant"] / _totals["reads"], 100 * _totals["pairable"] / _totals["reads"])
    except Exception as e:  # diagnostics must never break serving
        logger.warning("BFF share-audit failed: %s", e)
