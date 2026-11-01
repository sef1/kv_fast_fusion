"""Free materialized aliases at promotion — the pull-v2 half of BFF's capacity gain under full graph.

MATERIALIZE copies a representative's KV into the request's OWN block at landing, which keeps every
read correct but frees nothing. Redirecting instead (point the table at the representative, free the
copy) needs a block-table write after graph capture, which is why it is PIECEWISE-only — unless the
redirect is made BEFORE the worker ever sees the request's table. That window exists: a remote-KV
request first reaches the worker when it is first scheduled, and promotion out of
``WAITING_FOR_REMOTE_KVS`` happens in the same ``schedule()`` pass, so rewriting ``req_to_blocks``
there ships the representative's id as the request's own initial table. No device write, nothing for
a replayed graph to miss, and the victim really returns to the pool.

Because the victim already holds a valid copy, a swap that cannot be PROVEN safe is simply skipped —
the request keeps its block. Proof needs two things:

* **Every TP rank chose the same alias.** Each rank plans from its own signature shard, and the
  scheduler owns ONE block table for all of them; a swap only one rank agreed to would point the
  others at KV they never chose.
* **The scheduler still sees the representative where the worker did**: its owner is alive and
  loaded, the block is in the owner's table, and the owner has finished writing it (the owner's last
  block keeps receiving its decode's tokens).

Pure functions only; the scheduler glue lives in ``fast_fusion_ascend_patch``.
"""

import os

# Opt-in until a run confirms it: ship landed aliases worker→scheduler and swap them at promotion.
FREE_AT_PROMOTION = os.environ.get("BFF_V2_FREE_AT_PROMOTION", "0") == "1"

# Why a landed alias was not swapped. One counter each, so a run that frees nothing says why.
# ``incomplete_ranks`` includes a rank that landed NO alias for the request (it ships no map): the
# consensus would be empty then anyway, so nothing that could be freed is lost to it.
REFUSALS = ("incomplete_ranks", "no_consensus", "owner_gone", "owner_loading", "rep_moved",
            "rep_hot", "victim_moved", "victim_hot")


def consensus_landed(per_rank: dict, n_ranks: int):
    """``({gi: {victim: (rep, owner)}}, n_disagreed)``, or ``(None, 0)`` until every rank reported.

    ``per_rank`` is ``{rank: {gi: {victim: (rep, owner)}}}``. An alias survives only if every one of
    the ``n_ranks`` ranks landed exactly it; ``n_disagreed`` counts the aliases some rank had and the
    consensus dropped."""
    if n_ranks <= 0 or len(per_rank) < n_ranks:
        return None, 0
    maps = [per_rank[r] for r in sorted(per_rank)][:n_ranks]
    out: dict[int, dict[int, tuple]] = {}
    seen = set()
    for m in maps:
        for gi, vs in m.items():
            for v, ro in vs.items():
                seen.add((int(gi), int(v), int(ro[0]), ro[1]))
    for gi, vs in maps[0].items():
        for v, ro in vs.items():
            ro = (int(ro[0]), ro[1])
            if all(_landed_at(m, gi, v) == ro for m in maps[1:]):
                out.setdefault(int(gi), {})[int(v)] = ro
    agreed = sum(len(vs) for vs in out.values())
    return out, len(seen) - agreed


def _landed_at(m, gi, v):
    ro = (m.get(gi) or m.get(int(gi)) or {}).get(v)
    return None if ro is None else (int(ro[0]), ro[1])


def promotion_swaps(own_blocks, consensus: dict, owner_state, block_size, own_computed=None):
    """``({gi: new_block_ids}, {reason: n})`` — this request's tables with every provable swap made.

    ``own_blocks`` is the request's ``[[block ids] per group]``. ``owner_state(owner)`` returns
    ``None`` (gone), ``"loading"`` (still WAITING_FOR_REMOTE_KVS — its KV is incomplete) or
    ``(blocks_by_group, num_computed_tokens)``. A representative is usable only at a position below
    ``num_computed_tokens // block_size`` in its owner's table: a full block the owner will never
    write again. ``block_size`` is an int or ``block_size(gi)``, since groups may differ.

    ``own_computed`` is the request's own ``num_computed_tokens``: a victim at or past
    ``own_computed // block_size`` is the block its decode writes next, so sharing it would send the
    request's new tokens into the owner's rep. Such a victim keeps its own (already copied) block.
    None disables the check, for callers that cannot know it."""
    refused = {r: 0 for r in REFUSALS}
    merged: dict[int, list[int]] = {}
    for gi, vs in consensus.items():
        if gi >= len(own_blocks):
            refused["victim_moved"] += len(vs)
            continue
        bs = int(block_size(gi) if callable(block_size) else block_size)
        table = list(own_blocks[gi])
        pos = {int(b): i for i, b in enumerate(table)}
        changed = False
        for victim, (rep, owner) in vs.items():
            if victim not in pos:
                refused["victim_moved"] += 1
                continue
            if own_computed is not None and (bs <= 0 or pos[victim] >= int(own_computed) // bs):
                refused["victim_hot"] += 1
                continue
            st = owner_state(owner)
            if st is None:
                refused["owner_gone"] += 1
                continue
            if st == "loading":
                refused["owner_loading"] += 1
                continue
            o_blocks, o_computed = st
            o_table = o_blocks[gi] if gi < len(o_blocks) else []
            try:
                rpos = [int(b) for b in o_table].index(int(rep))
            except ValueError:
                refused["rep_moved"] += 1
                continue
            if bs <= 0 or rpos >= int(o_computed) // bs:
                refused["rep_hot"] += 1
                continue
            table[pos[victim]] = int(rep)
            changed = True
        if changed:
            merged[int(gi)] = table
    return merged, refused
