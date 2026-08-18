"""SimHash LSH index for cross-request P/D block fusion.

The connector's default cross-request backend ("matrix") compares each step's blocks against a
dense FIFO window of the last ``BFF_PD_ENCODED_BATCH_SIZE`` requests' representatives. That is
O(N·R·D) per group and the window has to stay small, so a block only ever matches a handful of
recent requests.

This module is the alternative: a banded SimHash index. Each representative is hashed into
``BFF_LSH_TABLES`` buckets of ``BFF_LSH_BITS_PER_TABLE`` bits (sign bits of a fixed-seed
random-hyperplane projection). Probing a block looks up its own buckets and only compares against
what collides, so the pool can hold ``BFF_LSH_MAX_ENTRIES`` representatives per group instead of a
request window, and the probe cost stays roughly O(N).

**A bucket hit is a candidate, not a match.** Every candidate's exact cosine is verified against
``THRESHOLD`` and the best one wins. This is load-bearing, not defensive: the Ascend port records
that a looser configuration (16 tables × 10 bits) let dissimilar blocks collide and merge, which
corrupted decode output at high concurrency — fixed by both the wider 20-bit bands and this verify.
Redirecting on a raw bucket hit would reintroduce exactly that failure, and it surfaces as an
accuracy drop rather than an error.

Pure torch: no device, transport or vLLM dependency, so it is unit-testable on CPU. Ported from
``kv_fast_fusion_ascend/connectors/mooncake_layerwise_connector_ff.py`` (which keeps its own copy so
the working NPU path is untouched); that one was in turn ported from the legacy single-instance
runner in ``kv_fast_fusion/legacy/kv_fast_fusion_graph_runner.py``.
"""

import itertools
import os
from collections import Counter, OrderedDict

import torch

# Banded SimHash config: TABLES sub-hashes of BITS each, from sign() of a fixed-seed random
# hyperplane projection. 16 x 20 keeps ~87% recall at cos 0.95 while random-pair collisions sit
# ~1000x below the old 16 x 10 (1.5e-2 -> 1.5e-5): near-duplicates still merge, dissimilar blocks
# do not.
LSH_TABLES = int(os.environ.get("BFF_LSH_TABLES", "16"))
LSH_BITS = int(os.environ.get("BFF_LSH_BITS_PER_TABLE", "20"))
# Max representative entries per fusion group before LRU-evicting the oldest half. A hard cap, not
# the working bound — see LSH_MAX_OWNERS.
LSH_MAX_ENTRIES = int(os.environ.get("BFF_LSH_MAX_ENTRIES", "50000"))
# Max distinct OWNING REQUESTS held at once, evicted oldest-request-first. This, not the row cap, is
# what keeps the pool honest: a representative is only usable if its request is still resident on the
# decode instance, and a request's blocks all die together. Sized to the decode's concurrency
# ceiling. It is a backstop — the connector evicts exactly, on notification from D — but without it
# a row-only cap lets the index keep serving reps whose KV was freed long ago (measured: redirects
# resolving on D decayed 56% -> 7% across a single run).
LSH_MAX_OWNERS = int(os.environ.get("BFF_LSH_MAX_LIVE_OWNERS", "512"))
# Cap on candidates verified per block. Candidates/block grows as TABLES*MAX_ENTRIES/2**BITS, so a
# low BITS against a large index makes the verify dominate. >0 keeps only the top-K candidates by
# table-collision count (multi-probe ranking); 0 = verify every bucket candidate.
LSH_MAX_CAND = int(os.environ.get("BFF_LSH_MAX_CANDIDATES", "0"))
_LSH_POWERS = (2 ** torch.arange(LSH_BITS, dtype=torch.int64)).tolist()

# Accepted-cosine histogram bins. `probe` already pays the .item() sync for the accept decision, so
# binning the value is free. The distribution answers, from ONE run, whether BFF_THRESHOLD is the
# quality lever (mass sitting near the threshold → raising it trims exactly that mass) or the merges
# are already near-identical (mass > 0.95 → any accuracy cost is elsewhere, e.g. JL projection error).
ACCEPT_COS_BINS = (0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.01)
ACCEPT_COS_LABELS = tuple(
    f"{ACCEPT_COS_BINS[i]:.2f}-{min(ACCEPT_COS_BINS[i + 1], 1.0):.2f}"
    for i in range(len(ACCEPT_COS_BINS) - 1))

# Relative substitution error histogram. Cosine is scale-free, so two blocks with the same DIRECTION
# and different magnitudes score identically while swapping one for the other injects real error.
# What the decode actually suffers when a redirect replaces k_owner with k_rep is
#
#     rel_err = ||k_owner - k_rep|| / ||k_owner|| = sqrt(1 + r^2 - 2*r*cos),   r = ||k_rep||/||k_owner||
#
# which needs only the norm ratio and the cosine the probe already computed. This — not the
# compression factor — is the quantity a threshold sweep should be plotted against.
REL_ERR_BINS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, float("inf"))
REL_ERR_LABELS = tuple(
    (f">{REL_ERR_BINS[i]:.2f}" if REL_ERR_BINS[i + 1] == float("inf")
     else f"{REL_ERR_BINS[i]:.2f}-{REL_ERR_BINS[i + 1]:.2f}")
    for i in range(len(REL_ERR_BINS) - 1))


# Hard ceiling on the relative substitution error a merge may inject. This, not the cosine bar, is
# the quantity that governs accuracy: cosine is scale-free, so a pair can clear any cosine threshold
# and still be a terrible substitution because the magnitudes differ. 1.0 (default) is inert —
# rel_err can only exceed 1 when the rep is more than twice the owner's norm — so unsetting it
# reproduces the pure-cosine behaviour every earlier run used.
#
# Worth knowing before tuning it: minimising rel_err over the norm ratio gives r = cos and
#
#     min_r rel_err = sqrt(1 - cos^2)
#
# an error floor no norm can beat. So a budget of 0.20 implies cos >= 0.9798 whatever the norms do;
# the cosine bar is the binding constraint and the norm can only make a given pair worse.
MAX_REL_ERR = float(os.environ.get("BFF_MAX_REL_ERR", "1.0"))


def rel_err(cos: float, norm_ratio: float) -> float:
    """Relative L2 error of substituting a rep for the owner block. See :data:`REL_ERR_BINS`."""
    return float(max(0.0, 1.0 + norm_ratio * norm_ratio - 2.0 * norm_ratio * cos)) ** 0.5


def min_rel_err(cos: float) -> float:
    """The smallest substitution error this cosine can possibly give, at the best norm ratio.

    Lets a caller reject a candidate on cosine alone, before it knows (or needs) the norms."""
    return float(max(0.0, 1.0 - cos * cos)) ** 0.5


def _bin(value: float, bins) -> int:
    """Index of the bin `value` falls in (clamped to the last)."""
    for j in range(len(bins) - 2, 0, -1):
        if value >= bins[j]:
            return j
    return 0


def get_proj(holder: list, d: int, device) -> torch.Tensor:
    """Fixed-seed random-hyperplane projection ``[d, LSH_TABLES*LSH_BITS]`` for SimHash, cached in
    ``holder`` per feature width (deterministic across restarts, so P and D agree)."""
    m = holder[0]
    if m is None or m.shape[0] != d:
        g = torch.Generator(device="cpu")
        g.manual_seed(20240517)
        m = torch.randn(d, LSH_TABLES * LSH_BITS, generator=g, dtype=torch.float32).to(device)
        holder[0] = m
    return m


def sub_hashes_device(vecs_norm: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Banded SimHash bucket ids ``[M, LSH_TABLES]`` from the sign bits of ``vecs_norm @ proj``,
    left ON ``vecs_norm``'s device.

    Packs before any transfer: the host only ever needs the ``[M, T]`` bucket ids, so copying the
    unpacked ``[M, T*B]`` bits would move LSH_BITS× more bytes. The copy is the caller's, so it
    chooses when to pay the sync."""
    bits = (vecs_norm.float() @ proj > 0).to(torch.int64)               # [M, T*B] (on device)
    powers = torch.tensor(_LSH_POWERS, dtype=torch.int64, device=bits.device)
    return (bits.view(-1, LSH_TABLES, LSH_BITS) * powers).sum(dim=2)    # [M, T] (on device)


def sub_hashes(vecs_norm: torch.Tensor, proj: torch.Tensor) -> list:
    """Blocking convenience wrapper over :func:`sub_hashes_device`: a length-M list of length-T
    int lists."""
    return sub_hashes_device(vecs_norm, proj).cpu().tolist()


class LshIndex:
    """One fusion group's representative index.

    Representative vectors live in ONE contiguous ``mat`` ``[cap, d]`` grown by doubling, rather than
    per-entry tensors, so a probe's verify is a single ``index_select`` + mv instead of a per-block
    ``torch.stack`` over dict values (~5-7× faster on the Ascend measurements), and storing a copy
    avoids pinning a whole step's ``[N, d]`` buffer alive through an index view.

    Identity is opaque to this module: ``register`` takes ``(rep_key, slot, owner)`` per row and
    ``probe`` returns them back. The connector supplies its own P/D-stable ids (for Mooncake, the
    hashed ``transfer_id`` and the owning request's ``transfer_id``)."""

    def __init__(self, max_owners: int | None = None):
        # An "owner" is whatever granularity the caller's reps die at: the owning REQUEST for the
        # producer index (a request's blocks are freed together), but the individual BLOCK for the
        # decode-side dedup index, where blocks are freed one at a time. The default is sized for
        # the former, so the latter passes its own cap — see pd_dedup_plan.DedupPlanner.
        self.max_owners = LSH_MAX_OWNERS if max_owners is None else int(max_owners)
        self.tables = [dict() for _ in range(LSH_TABLES)]   # bucket_hash -> [row]
        self.meta = {}                                      # row -> (rep_key, slot, owner)
        self.row_hashes = {}                                # row -> sub_hashes (for eviction)
        self.lru = OrderedDict()                            # row -> None (oldest first)
        # Owner (request) -> its rows, plus insertion order. Eviction is per REQUEST because that is
        # the granularity at which the reps actually stop existing on the decode side.
        self.by_owner: dict = {}
        self.owner_lru = OrderedDict()                      # owner -> None (oldest first)
        self.row_norm = {}                                  # row -> ||rep|| before normalisation
        self.mat = None                                     # [cap, d] float32 rep vectors
        self.n_rows = 0
        self.accept_cos = [0] * len(ACCEPT_COS_LABELS)
        self.accept_rel_err = [0] * len(REL_ERR_LABELS)
        # Candidates that cleared the cosine bar and were then rejected by MAX_REL_ERR — i.e. the
        # merges the norm saved us from, which a cosine-only bar would have taken.
        self.rejected_by_rel_err = 0

    def size(self) -> int:
        return self.n_rows

    def n_owners(self) -> int:
        return len(self.by_owner)

    def _ensure_cap(self, need: int, d: int) -> None:
        """Grow ``mat`` by doubling until it holds ``need`` rows (never allocates the full
        LSH_MAX_ENTRIES × d up front)."""
        if self.mat is None:
            self.mat = torch.empty(max(256, need), d, dtype=torch.float32)
            return
        if self.mat.shape[0] >= need:
            return
        cap = self.mat.shape[0]
        while cap < need:
            cap *= 2
        new = torch.empty(cap, d, dtype=torch.float32)
        new[:self.n_rows] = self.mat[:self.n_rows]
        self.mat = new

    def probe(self, cur, hashes, owners, threshold, norms=None):
        """Probe for each current block; return ``(matched: list[bool], hits: list[(i, rep_key,
        rep_slot)])``.

        A hit requires a bucket candidate from a DIFFERENT owner whose **exact cosine** with the
        current block is > ``threshold``; the best candidate wins. A raw bucket collision is never
        enough — see the module docstring for why that distinction is load-bearing.

        ``cur`` is the caller's ``[N, d]`` L2-normalized reps on the same device as ``mat``,
        ``hashes[i]`` is block i's length-T bucket-id list, ``owners[i]`` its owning request.
        ``norms[i]``, when given, is block i's norm BEFORE normalisation, which turns the accepted
        cosine into the relative substitution error the decode will actually see."""
        n = cur.shape[0]
        matched = [False] * n
        hits: list[tuple[int, object, int]] = []
        if not self.n_rows:
            return matched, hits
        for i in range(n):
            owner_i = owners[i]
            if LSH_MAX_CAND > 0:
                # Multi-probe ranking: prefer candidates colliding in the MOST tables (more likely
                # to be similar), then keep at most LSH_MAX_CAND of them.
                counts: Counter = Counter()
                for t, h in enumerate(hashes[i]):
                    counts.update(self.tables[t].get(h, ()))
                cand_rows = [r for r, _ in counts.most_common()
                             if self.meta[r][2] != owner_i][:LSH_MAX_CAND]
            else:
                cand: set = set()
                for t, h in enumerate(hashes[i]):
                    cand.update(self.tables[t].get(h, ()))
                # Rows ARE the index into mat, so there is no id→row lookup — and buckets only ever
                # hold live rows (evict+compact run together), so no liveness check either.
                cand_rows = [r for r in cand if self.meta[r][2] != owner_i]
            if not cand_rows:
                continue
            rows = torch.tensor(cand_rows, dtype=torch.long, device=self.mat.device)
            sims = self.mat.index_select(0, rows) @ cur[i]              # [C]
            own = float(norms[i]) if norms is not None else None
            if own is not None and MAX_REL_ERR < 1.0:
                # Rank by the error the decode will actually suffer, not by cosine. The two
                # disagree whenever the norms differ: a candidate 1.3x the owner's magnitude is a
                # bad substitution however well aligned it is, and picking max(cosine) would take
                # it over a slightly-less-aligned candidate of the right size.
                own = own or 1.0
                cand_cos = sims.tolist()
                errs = [rel_err(c, float(self.row_norm.get(r, own)) / own)
                        for c, r in zip(cand_cos, cand_rows)]
                best_j = min(range(len(errs)), key=errs.__getitem__)
                v, e = cand_cos[best_j], errs[best_j]
                if v <= threshold or e > MAX_REL_ERR:
                    # Charge the rejection to whichever bar actually stopped it, so a run says how
                    # much the error budget removed BEYOND the cosine threshold.
                    if v > threshold:
                        self.rejected_by_rel_err += 1
                    continue
                row = cand_rows[best_j]
            else:
                best_val, best_j = sims.max(dim=0)
                v = float(best_val.item())
                if v <= threshold:
                    continue
                row = cand_rows[int(best_j.item())]
                e = (rel_err(v, float(self.row_norm.get(row, own or 1.0)) / (own or 1.0))
                     if own is not None else None)
            rep_key, rep_slot, _ = self.meta[row]
            hits.append((i, rep_key, rep_slot))
            matched[i] = True
            self.lru.move_to_end(row)                                   # LRU touch
            self.accept_cos[_bin(v, ACCEPT_COS_BINS)] += 1
            if e is not None:
                self.accept_rel_err[_bin(e, REL_ERR_BINS)] += 1
        return matched, hits

    def register(self, cur, hashes, rows_to_add, norms=None) -> None:
        """Insert this step's unmatched representatives; LRU-evict the oldest half when over the
        per-group cap. ``rows_to_add`` is ``[(flat_idx, rep_key, slot, owner), ...]``. ``cur`` is the
        same normalized host/device matrix passed to :meth:`probe_with_hashes` — sharing it means one
        sync per group, not two."""
        if not rows_to_add:
            return
        if self.n_rows >= LSH_MAX_ENTRIES:
            for row in list(itertools.islice(self.lru.keys(), max(1, LSH_MAX_ENTRIES // 2))):
                self.evict(row)
            self.compact()
        self._ensure_cap(self.n_rows + len(rows_to_add), cur.shape[1])
        for flat_idx, rep_key, slot, owner in rows_to_add:
            row = self.n_rows
            self.n_rows += 1
            self.mat[row] = cur[flat_idx]        # copy, not a view into this step's buffer
            self.meta[row] = (rep_key, int(slot), owner)
            sh = hashes[flat_idx]
            self.row_hashes[row] = sh
            if norms is not None:
                self.row_norm[row] = float(norms[flat_idx])
            self.lru[row] = None
            self.by_owner.setdefault(owner, []).append(row)
            self.owner_lru[owner] = None
            for t, h in enumerate(sh):
                self.tables[t].setdefault(h, []).append(row)
        # Owner bound last, so this step's own reps are never the ones dropped.
        if len(self.by_owner) > self.max_owners:
            stale = list(itertools.islice(self.owner_lru.keys(),
                                          len(self.by_owner) - self.max_owners))
            self.evict_owners(stale)

    def evict_owners(self, owners) -> int:
        """Drop every representative belonging to ``owners`` (one compaction for the batch).

        This is the eviction that matters: the connector calls it when the decode instance reports
        a request finished, so the index stops offering reps whose KV D has already freed. Returns
        the number of rows dropped."""
        dropped = 0
        for owner in owners:
            for row in self.by_owner.pop(owner, ()):
                self.evict(row)
                dropped += 1
            self.owner_lru.pop(owner, None)
        if dropped:
            self.compact()
        return dropped

    def evict_owner(self, owner) -> int:
        return self.evict_owners([owner])

    def evict(self, row) -> None:
        """Drop one row's metadata + bucket memberships. The ``mat`` row itself is reclaimed by the
        :meth:`compact` that always follows a batch evict."""
        sh = self.row_hashes.pop(row, None)
        entry = self.meta.pop(row, None)
        self.row_norm.pop(row, None)
        self.lru.pop(row, None)
        if entry is not None:
            # Keep the owner view consistent even when a caller evicts rows directly (the
            # LSH_MAX_ENTRIES path does), so evict_owners can never resurrect a dead row.
            rows = self.by_owner.get(entry[2])
            if rows is not None:
                if row in rows:
                    rows.remove(row)
                if not rows:
                    del self.by_owner[entry[2]]
                    self.owner_lru.pop(entry[2], None)
        if sh is not None:
            for t, h in enumerate(sh):
                bucket = self.tables[t].get(h)
                if bucket:
                    try:
                        bucket.remove(row)
                    except ValueError:
                        pass
                    if not bucket:
                        del self.tables[t][h]

    def compact(self) -> None:
        """Renumber surviving rows to 0..n-1 after a batch evict: gather them in ``mat`` and rebuild
        meta/owner/lru/tables under the new row ids (LRU order preserved)."""
        survivors = [r for r in range(self.n_rows) if r in self.meta]
        if not survivors:
            self.tables = [dict() for _ in range(LSH_TABLES)]
            self.meta, self.row_hashes, self.row_norm = {}, {}, {}
            self.lru = OrderedDict()
            self.by_owner = {}
            self.owner_lru = OrderedDict()
            self.mat = None
            self.n_rows = 0
            return
        remap = {old: new for new, old in enumerate(survivors)}
        self.mat = self.mat.index_select(
            0, torch.tensor(survivors, dtype=torch.long, device=self.mat.device)).contiguous()
        self.meta = {remap[r]: v for r, v in self.meta.items()}
        self.row_hashes = {remap[r]: v for r, v in self.row_hashes.items()}
        self.row_norm = {remap[r]: v for r, v in self.row_norm.items()}
        self.lru = OrderedDict((remap[r], None) for r in self.lru)
        by_owner: dict = {}
        for r, v in self.meta.items():
            by_owner.setdefault(v[2], []).append(r)
        self.by_owner = by_owner
        # Keep insertion order for the surviving owners, so the bound still evicts oldest-first.
        self.owner_lru = OrderedDict((o, None) for o in self.owner_lru if o in by_owner)
        tables = [dict() for _ in range(LSH_TABLES)]
        for r, sh in self.row_hashes.items():
            for t, h in enumerate(sh):
                tables[t].setdefault(h, []).append(r)
        self.tables = tables
        self.n_rows = len(survivors)
