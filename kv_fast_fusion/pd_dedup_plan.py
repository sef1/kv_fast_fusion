"""Decode-side dedup planning for Mooncake BFF v2: decide which blocks NOT to pull.

The v1 connector had the PRODUCER choose which blocks are duplicates and ship a redirect map for
the decode to apply after the KV had already arrived. Two things went wrong with that, both
measured:

* the producer has no idea what is resident on the decode, so **71% of its redirects resolved to
  nothing** (7,714 applied against 18,726 unresolved) — the decode simply kept its own block;
* every block was transferred regardless, including the ones the decode discarded milliseconds
  later, so the producer paid full wire cost and full transfer latency for them. Under Mooncake's
  pull model the producer cannot free a block until the decode has taken it, so those wasted
  transfers are exactly what wedged the producer at 99.5% cache for an entire run.

This module inverts that. The producer ships **signatures** — cheap per-block fingerprints — and the
decode, which is the side that actually owns the memory and knows what is resident, decides. A block
it can satisfy locally is never requested at all, so the bytes are never sent.

The property that makes it correct: the decode only ever aliases to a block **it holds**, so
"unresolved" stops being a possible outcome rather than being a number to minimise.

Device-, transport- and vLLM-free, so it is testable on CPU. Reuses :mod:`kv_fast_fusion.pd_lsh`
unchanged for the index itself.
"""

import os
from dataclasses import dataclass, field

import torch

from kv_fast_fusion import pd_lsh

# Blocks held per group in the decode's dedup index. pd_lsh's default owner cap is sized for the
# producer, where an owner is a whole REQUEST; here an owner is a single block, so the default would
# hold 512 blocks instead of 512 requests' worth. Sized to the decode's block count.
MAX_RESIDENT = int(os.environ.get("BFF_V2_MAX_RESIDENT", "32768"))


@dataclass(frozen=True)
class IncomingBlock:
    """One block the decode has allocated and is about to request from a producer."""

    req_id: str
    group: int
    slot: int                # position within this request's per-group block list
    block_id: int            # the decode's own physical block id
    row: int                 # index into the signature matrix passed alongside


@dataclass
class DedupPlan:
    """What to pull and what to alias instead.

    ``keep[(req_id, group)]`` is the slot list still worth requesting; every slot absent from it is
    in ``alias``, mapped to the decode-side block id that will serve it."""

    keep: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    alias: dict[tuple[str, int], dict[int, int]] = field(default_factory=dict)
    n_resident: int = 0      # satisfied by a block already on D
    n_batch: int = 0         # satisfied by another block in this same pull
    # Cleared the cosine bar, rejected by the substitution-error budget: the merges the norm
    # prevented. A cosine-only bar would have taken every one of these.
    rejected_by_rel_err: int = 0
    accept_cos: list = field(default_factory=lambda: [0] * len(pd_lsh.ACCEPT_COS_LABELS))
    accept_rel_err: list = field(default_factory=lambda: [0] * len(pd_lsh.REL_ERR_LABELS))

    def n_dropped(self) -> int:
        return sum(len(v) for v in self.alias.values())


class DedupPlanner:
    """Per-group index of the blocks THIS decode currently holds, plus the planning pass.

    Ownership semantics differ from the producer's use of :class:`pd_lsh.LshIndex`. There, a
    candidate had to come from a different request, because merging a request onto its own block was
    meaningless across the wire. Here a request's own duplicate blocks are a perfectly good saving,
    so every row is registered under a unique owner (its block id) to disable that exclusion."""

    def __init__(self) -> None:
        self._idx: dict[int, pd_lsh.LshIndex] = {}
        self._resident: dict[int, set[int]] = {}     # group -> block ids in the index

    # -- residency -------------------------------------------------------------------
    def register(self, group: int, sigs, hashes, norms, block_ids) -> None:
        """Record blocks that are now resident on this decode and safe to alias to."""
        if not block_ids:
            return
        idx = self._idx.get(group)
        if idx is None:
            idx = self._idx[group] = pd_lsh.LshIndex(max_owners=MAX_RESIDENT)
        live = self._resident.setdefault(group, set())
        rows = [(i, int(b), 0, int(b)) for i, b in enumerate(block_ids)]
        idx.register(sigs, hashes, rows, norms)
        live.update(int(b) for b in block_ids)

    def forget(self, group: int, block_ids) -> int:
        """Drop blocks the decode has freed. Owner == block id, so eviction is per block."""
        idx = self._idx.get(group)
        if idx is None:
            return 0
        live = self._resident.setdefault(group, set())
        gone = [int(b) for b in block_ids if int(b) in live]
        live.difference_update(gone)
        return idx.evict_owners(gone)

    def size(self, group: int) -> int:
        idx = self._idx.get(group)
        return 0 if idx is None else idx.size()

    def is_resident(self, group: int, block_id: int) -> bool:
        """Is this block still a legal alias target?

        The check the apply step needs, and the reason it is exact: a block leaves this set the
        moment D frees it, so "still resident" answers "has this block been recycled since the
        decision was taken" — which is the only way an alias can go wrong."""
        return int(block_id) in self._resident.get(group, ())

    def forget_any(self, block_ids) -> int:
        """Drop blocks without knowing their group. Block ids are disjoint across KV-cache groups
        (each group owns a slice of the shared block axis), so this is unambiguous; it exists
        because the block pool frees by id and does not know about groups."""
        return sum(self.forget(g, block_ids) for g in list(self._idx))

    # -- planning --------------------------------------------------------------------
    def plan(self, group: int, incoming: list, sigs, hashes, norms, threshold) -> DedupPlan:
        """Decide, for one group, which of ``incoming`` still needs to be pulled.

        ``sigs`` is the [N, d] L2-normalised signature matrix the producer sent, ``hashes[i]`` block
        i's bucket ids and ``norms[i]`` its pre-normalisation norm; ``incoming[i].row`` indexes them.

        A block is dropped when it matches, above ``threshold``, either a resident block or an
        earlier block in this same pull that is being KEPT. Chaining is not allowed — a rep must be
        a block that will actually be written — so an aliased block never becomes a candidate for a
        later one."""
        plan = DedupPlan()
        idx = self._idx.get(group)
        # Blocks kept so far in this pull, as a second candidate source. They are not resident yet;
        # the caller applies every alias only after the transfer completes, which is what makes
        # aliasing to them safe.
        batch_rows: list[int] = []
        batch_ids: list[int] = []

        for blk in incoming:
            keep = plan.keep.setdefault((blk.req_id, blk.group), [])
            row = blk.row
            cur = sigs[row:row + 1]
            hit = None

            if idx is not None and idx.size():
                matched, hits = idx.probe(cur, [hashes[row]], [blk.block_id], threshold,
                                          [norms[row]])
                if matched[0] and hits:
                    hit = (int(hits[0][1]), True)

            if hit is None and batch_rows:
                # Same-pull candidates: a small exact comparison, no index needed. Ranked by the
                # substitution error, exactly as pd_lsh.probe ranks resident ones — otherwise the
                # two halves of the same decision would use different rules.
                cand = sigs[torch.as_tensor(batch_rows, dtype=torch.long)]
                sims = (cand @ sigs[row]).tolist()
                own = float(norms[row]) or 1.0
                errs = [pd_lsh.rel_err(c, float(norms[batch_rows[k]]) / own)
                        for k, c in enumerate(sims)]
                j = min(range(len(errs)), key=errs.__getitem__)
                v, e = sims[j], errs[j]
                if v > threshold and e <= pd_lsh.MAX_REL_ERR:
                    hit = (batch_ids[j], False)
                    self._bin(plan, v, norms[batch_rows[j]], norms[row])
                elif v > threshold:
                    plan.rejected_by_rel_err += 1

            if hit is None:
                keep.append(blk.slot)
                batch_rows.append(row)
                batch_ids.append(blk.block_id)
                continue

            rep_id, from_resident = hit
            plan.alias.setdefault((blk.req_id, blk.group), {})[blk.slot] = rep_id
            if from_resident:
                plan.n_resident += 1
            else:
                plan.n_batch += 1

        if idx is not None:
            # The index binned its own accepted cosines during probe; fold them in and reset so the
            # plan's histogram covers exactly this pull.
            for i, c in enumerate(idx.accept_cos):
                plan.accept_cos[i] += c
            for i, c in enumerate(idx.accept_rel_err):
                plan.accept_rel_err[i] += c
            plan.rejected_by_rel_err += idx.rejected_by_rel_err
            idx.accept_cos = [0] * len(pd_lsh.ACCEPT_COS_LABELS)
            idx.accept_rel_err = [0] * len(pd_lsh.REL_ERR_LABELS)
            idx.rejected_by_rel_err = 0
        return plan

    @staticmethod
    def _bin(plan: DedupPlan, cos: float, rep_norm, own_norm) -> None:
        plan.accept_cos[pd_lsh._bin(cos, pd_lsh.ACCEPT_COS_BINS)] += 1
        own = float(own_norm) or 1.0
        e = pd_lsh.rel_err(cos, float(rep_norm) / own)
        plan.accept_rel_err[pd_lsh._bin(e, pd_lsh.REL_ERR_BINS)] += 1
