"""BFF v2 core: the producer ships signatures, the decode decides what not to fetch.

v1 put the merge decision on the PRODUCER: it clustered blocks during its forward pass, shipped a
redirect map, and the decode rewrote its block table after the KV had already landed. Two
measurements condemned that arrangement:

* **73% of redirects resolved to nothing** — the producer cannot see what is resident on the decode,
  so most of its merge decisions were guesses about another process's memory;
* **every block was transferred anyway**, including the ones freed milliseconds after arrival.

v2 inverts the roles. The producer computes a cheap per-block **signature** and nothing else; the
decode, which owns the memory and knows exactly what is resident, decides which blocks it does not
need and never asks for them. An alias therefore always points at a block the decode holds
("unresolved" stops being a possible outcome), and a deduplicated block is never written, so the
saving lands on the wire.

**Everything here is transport- and device-free**, so one implementation serves the GPU Mooncake
pull connector and the Ascend NPU push connectors. What a transport must supply is only:

1. a way to get signatures from P to D (who *initiates* that exchange is irrelevant — what matters
   is that **D decides**);
2. a way for D to express "do not send me this block" — every Mooncake transport pairs P's and D's
   block lists POSITIONALLY, so that is a :data:`SENTINEL` in place of the id, never a shorter list;
3. a signal that a request's KV has landed, so its aliases become appliable (see
   :meth:`DedupEngine.release`).

**The one hazard v2 introduces, and how it is contained.** In v1 a failed apply was harmless: the
block held its own correct KV and you merely lost compression. Here a dropped block is *never
written*, so an alias that cannot be applied would leave the decode reading whatever was in that
block before. Every such case is routed into vLLM's KV-load-failure path (``note_failed_blocks`` →
``get_block_ids_with_load_errors`` → ``Scheduler._handle_invalid_blocks``), which recomputes the
request locally. Slower, never wrong. :class:`AliasApplier` names the four ways it can happen.
"""

import json
import os
import time
from typing import Any

import torch

from kv_fast_fusion import pd_lsh
from kv_fast_fusion.pd_dedup_plan import DedupPlanner, IncomingBlock

# Signature width. The decode verifies an exact cosine against these, so this is the only place
# fidelity is traded for wire size: 128 fp16 dims is ~256 B against a block of KV that is ~1.6 MB,
# i.e. 0.02%. Shared JL projection, fixed seed, so P and D agree without negotiating.
SIG_DIM = int(os.environ.get("BFF_SIG_DIM", "128"))
# Master switch. Off disables the ENTIRE mechanism, signature exchange included — the transfer
# reverts to what the stock connector would do, which is the "BFF group split, no fusion" control
# arm, NOT a measurement of what the exchange costs. (Two runs with this off differ only by noise,
# which is how the ±5% throughput / ±0.011 F1 band was established.) To price the exchange itself,
# leave this on and set BFF_THRESHOLD to 1.01 so every candidate is rejected.
V2_ENABLED = os.environ.get("BFF_V2_DEDUP", "1") == "1"
# Alias to blocks left over from EARLIER transfers, not just duplicates inside the current one. This
# is where cross-request reuse comes from; it is safe only because the decode's block pool calls
# back on every release (fast_fusion_block_pool.on_blocks_freed), so a block leaves the index
# strictly before it can be handed to another request.
V2_RESIDENT = os.environ.get("BFF_V2_RESIDENT", "1") == "1"
# Marks a block the decode decided not to fetch. It rides in place of the block id rather than being
# removed, because every Mooncake transport pairs P's and D's block lists positionally — a shortened
# list would pair the surviving blocks against the wrong source and silently write the wrong KV.
SENTINEL = -1
# Forward steps an alias map may wait for its owner to be batched before its blocks are declared
# unwritten. Shared with v1's knob so both connectors expire on the same clock.
APPLY_MAX_AGE = int(os.environ.get("BFF_FF_APPLY_MAX_AGE", "16"))
# How often the stats file is refreshed (wall clock; the step rate varies too much between arms for
# a count-only rule, which once pinned the file at its first snapshot for a whole run).
STATS_DUMP_SEC = float(os.environ.get("BFF_PD_STATS_DUMP_SEC", "10"))
STATS_DIR = os.environ.get("BFF_PD_STATS_DIR", ".")

# Why an alias could not be applied. One counter per distinct cause, because the first v2 run
# reported a single number for all four and the root cause (maps staged a whole transfer too early)
# had to be inferred from the step rate instead of read off the run.
FAIL_REASONS = ("owner_never_batched", "rep_not_resident", "victim_not_in_table",
                "block_table_write_refused", "owner_id_ambiguous", "rep_recycled")

# Why an exchange did not happen at all. A connector that never ASKS produces exactly the same
# all-zero stats as one that asked and found nothing, and the first Ascend run was the former for
# an entire benchmark before a log dive found it. Silence is not an acceptable report.
SKIP_REASONS = ("no_kv_tensors", "no_signature", "length_mismatch", "kv_layout_refused",
                "no_peer", "empty_group")


def threshold_for(gi: int) -> float:
    """Cosine bar for a group (``BFF_THRESHOLD`` / ``BFF_THRESHOLD_G``).

    At matched norms this is an error budget, not a similarity preference: 0.75 authorises replacing
    a block with one that differs by 0.707 of its own magnitude. The real budget is
    ``pd_lsh.MAX_REL_ERR``; this stays for continuity with every earlier run."""
    from kv_fast_fusion.constants import THRESHOLD
    from kv_fast_fusion.connectors.mooncake_connector_ff import _THRESHOLD_G
    return _THRESHOLD_G.get(gi, float(THRESHOLD))


class KVLayoutError(ValueError):
    """The per-layer cache is not a shape this can index blocks out of.

    Raised rather than guessed at: a mis-indexed signature is worse than no signature, because it
    does not fail — it merges blocks that are not alike."""


def key_blocks(kv, idx: "torch.Tensor", is_mla: bool) -> "torch.Tensor":
    """Select the K rows for ``idx`` from one layer's cache entry, whatever shape it arrives in.

    The two transports genuinely differ, and the difference is invisible until it corrupts:

    * **GPU Mooncake** hands one stacked tensor per layer, ``[2, num_blocks, block, heads, dim]``,
      where dim 0 selects K vs V — so blocks are ``kv[0, idx]``.
    * **Ascend layerwise** hands a ``(K, V)`` list/tuple of separate tensors, each with the block
      dim at 0 — so blocks are ``K[idx]``. Applying the GPU form here would index *inside block 0*
      and silently return the wrong content.
    * **MLA** has a single latent cache with the block dim at 0 — ``kv[idx]``.
    """
    if isinstance(kv, (list, tuple)):
        if not kv:
            raise KVLayoutError("empty per-layer cache entry")
        k = kv[0]                       # Ascend: [K, V], each [num_blocks, ...]
    elif is_mla:
        k = kv                          # single latent cache, block dim 0
    elif kv.ndim >= 4 and kv.shape[0] == 2:
        return kv[0].index_select(0, idx)   # GPU stacked [2, num_blocks, ...]
    else:
        k = kv                          # already a bare K tensor with block dim 0
    if not hasattr(k, "index_select"):
        raise KVLayoutError(f"per-layer cache is a {type(k).__name__}, not a tensor")
    return k.index_select(0, idx)


def signature_matrix(kv_layers: list, block_ids: list[int], is_mla: bool, jl_holder: list,
                     num_blocks: int | None = None):
    """Concatenate the given layers' per-block K and project to :data:`SIG_DIM`.

    Mirrors ``FFProducerFusion.block_repr`` + the concat the v1 clustering used, so a v2 signature
    means the same thing a v1 representation did and the two runs stay comparable. ``kv_layers`` is
    whichever layers of the group are available — all of them on a whole-request transport, possibly
    only the first on a layerwise one that streams a layer the moment it is computed.

    ``num_blocks``, when given, is the block count the *connector* speaks. Ascend allows a cache
    tensor whose block count is an integer multiple of it (``block_size_scale``), in which case a
    connector block id does not index the tensor and the only safe answer is to refuse.

    Returns ``(normalised [N, d], norms [N])`` on the caller's device."""
    if not block_ids or not kv_layers:
        return None, None
    first = kv_layers[0]
    probe = first[0] if isinstance(first, (list, tuple)) else first
    if num_blocks is not None and getattr(probe, "shape", None) is not None:
        rows = int(probe.shape[1] if (not isinstance(first, (list, tuple)) and not is_mla
                                      and probe.ndim >= 4 and probe.shape[0] == 2)
                   else probe.shape[0])
        if rows != int(num_blocks):
            raise KVLayoutError(
                f"cache holds {rows} blocks but the connector addresses {num_blocks} "
                "(block_size_scale != 1); a connector block id does not index this tensor")
    idx = torch.as_tensor(block_ids, dtype=torch.long, device=probe.device)
    parts = []
    for kv in kv_layers:
        blk = key_blocks(kv, idx, is_mla).float()
        parts.append(blk.reshape(idx.shape[0], -1))
    full = torch.cat(parts, dim=1)
    if jl_holder[0] is None or jl_holder[0].shape[0] != full.shape[1]:
        g = torch.Generator(device="cpu")
        g.manual_seed(20240517)
        jl_holder[0] = torch.randn(full.shape[1], SIG_DIM, generator=g,
                                   dtype=torch.float32).to(full.device)
    sig = full @ jl_holder[0]
    norms = sig.norm(dim=1).clamp(min=1e-6)
    return sig / norms.unsqueeze(1), norms


class SignatureCodec:
    """Pack/unpack the per-block signatures that cross the wire.

    fp16 for the vectors (the decode verifies a cosine, which does not need fp32) and int64 bucket
    ids, both as plain Python types so any msgpack-based control plane carries them without a new
    encoder."""

    @staticmethod
    def encode(sig: torch.Tensor, norms: torch.Tensor, hashes: list) -> dict:
        return {"sig": sig.to(torch.float16).cpu().numpy().tobytes(),
                "dim": int(sig.shape[1]),
                "norms": [float(x) for x in norms.detach().cpu().tolist()],
                "hashes": hashes}

    @staticmethod
    def decode(payload: dict):
        import numpy as np
        dim = int(payload["dim"])
        arr = np.frombuffer(payload["sig"], dtype=np.float16).reshape(-1, dim)
        return (torch.from_numpy(arr.copy()).float(),
                payload["norms"], payload["hashes"])


class DedupStats:
    """What v2 did, in the units the collector prints.

    Per group wherever v1 was per group: the threshold question is a per-group question (each
    group's blocks have their own similarity floor — one run measured g1 deduping 96.5% against an
    index of 138 while g4 managed 25%), and a run that reports one pooled histogram cannot answer
    it."""

    def __init__(self) -> None:
        self.planned: dict[int, int] = {}            # blocks considered
        self.dropped_resident: dict[int, int] = {}   # served by a block already on D
        self.dropped_batch: dict[int, int] = {}      # served by another block in the same transfer
        self.applied = 0            # aliases that reached the block table
        self.recomputed = 0         # aliases that could not be applied -> local recompute
        # Wall time spent inside AliasApplier.apply, which runs on the decode's critical path once
        # per forward step. Measured because the 2026-08-19 A/B showed the current connector paying
        # ~15% more TPOT than the legacy one AT MATCHED CONCURRENCY (80.6ms vs 94.3ms at ~101
        # running), and static diffing twice failed to explain it. This separates the two possible
        # homes for that cost: our Python here, or the GPU work the resulting block tables imply.
        # If apply_ms stays far below the step budget, the cost is not in this file.
        self.apply_ms = 0.0
        self.apply_calls = 0
        self.fail_reasons = dict.fromkeys(FAIL_REASONS, 0)
        self.sig_phase_failed = 0   # exchanges that fell back to a full transfer
        self.exchanges = 0          # exchanges actually attempted
        self.skip_reasons = dict.fromkeys(SKIP_REASONS, 0)
        # Producer-side only: blocks the decode told it to skip. An independent cross-check that
        # the two sides agree — it should track the decode's blocks_not_requested, and a divergence
        # means the sentinel list and the alias map have come apart.
        self.blocks_withheld = 0
        self.accept_cos: dict[int, list] = {}
        self.accept_rel_err: dict[int, list] = {}
        self.rejected_by_rel_err: dict[int, int] = {}
        # Cosines of what the error budget turned away, per group. The count alone cannot say
        # whether a rejection was salvageable; the distribution can, because
        # sqrt(1 - MAX_REL_ERR^2) is a hard floor (0.954 at the usual 0.30). One run measured g1
        # supplying 4,453 of 5,221 rejections, which this separates into "near-duplicates the norm
        # ratio spoiled" and "unrelated blocks that a shared common component floated to cos ~0.9".
        self.reject_cos: dict[int, list] = {}
        self.index_blocks: dict[int, int] = {}
        self._last_dump = 0.0

    def absorb(self, gi: int, plan) -> None:
        """Fold in one group's plan histograms.

        Deliberately NOT ``plan.n_resident`` / ``plan.n_batch``: a block counts as saved only once
        the sentinel is actually written into the request, and the caller can still decline a merge
        the planner proposed. Counting the proposal is how v1 came to report 34,812 merges of which
        7,171 were real."""
        cos = self.accept_cos.setdefault(gi, [0] * len(pd_lsh.ACCEPT_COS_LABELS))
        err = self.accept_rel_err.setdefault(gi, [0] * len(pd_lsh.REL_ERR_LABELS))
        rej = self.reject_cos.setdefault(gi, [0] * len(pd_lsh.ACCEPT_COS_LABELS))
        for i, c in enumerate(plan.accept_cos):
            cos[i] += c
        for i, c in enumerate(plan.accept_rel_err):
            err[i] += c
        for i, c in enumerate(getattr(plan, "reject_cos", ())):
            rej[i] += c
        self.rejected_by_rel_err[gi] = (self.rejected_by_rel_err.get(gi, 0)
                                        + plan.rejected_by_rel_err)

    def note_failure(self, reason: str, n: int = 1) -> None:
        self.recomputed += n
        self.fail_reasons[reason] = self.fail_reasons.get(reason, 0) + n

    def note_skip(self, reason: str, n: int = 1) -> None:
        """Record that an exchange did NOT happen, and why. See :data:`SKIP_REASONS`."""
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + n

    def is_inert(self) -> bool:
        """True when v2 is installed but has never once asked the decode anything."""
        return self.exchanges == 0 and any(self.skip_reasons.values())

    def should_dump(self, step: int) -> bool:
        now = time.monotonic()
        if step <= 3 or now - self._last_dump >= STATS_DUMP_SEC:
            self._last_dump = now
            return True
        return False

    def dump(self, stats_dir: str = STATS_DIR) -> None:
        """Atomically write to ``<stats_dir>/bff_stats_<pid>.json``, the file the collector reads."""
        try:
            path = os.path.join(stats_dir, f"bff_stats_{os.getpid()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.stats_dict(), f)
            os.replace(tmp, path)   # atomic — the reader never sees a half-written file
        except Exception as e:  # pragma: no cover - defensive (never break the transfer)
            from vllm.logger import init_logger
            init_logger("vllm.pd_dedup_v2").warning("BFF v2: could not dump stats: %s", e)

    def stats_dict(self) -> dict:
        planned = sum(self.planned.values())
        res, batch = sum(self.dropped_resident.values()), sum(self.dropped_batch.values())
        dropped = res + batch
        groups = sorted(set(self.planned) | set(self.dropped_resident) | set(self.dropped_batch))
        return {
            "pid": os.getpid(),
            "bff_version": 2,
            "blocks_planned": planned,
            "blocks_not_requested": dropped,
            # v1 reported what the producer CLAIMED; there is no such gap here, because a block that
            # is not requested is not transferred and its alias always resolves.
            "blocks_not_requested_resident": res,
            "blocks_not_requested_same_pull": batch,
            "wire_saving_pct": (100.0 * dropped / planned) if planned else 0.0,
            "wire_saving_per_group": {
                str(g): {
                    "planned": self.planned.get(g, 0),
                    "not_requested": (self.dropped_resident.get(g, 0)
                                      + self.dropped_batch.get(g, 0)),
                    "pct": (100.0 * (self.dropped_resident.get(g, 0)
                                     + self.dropped_batch.get(g, 0)) / self.planned[g])
                    if self.planned.get(g) else 0.0,
                } for g in groups},
            "apply_ms_total": round(self.apply_ms, 1),
            "apply_calls": self.apply_calls,
            "apply_us_mean": round(self.apply_ms * 1000 / self.apply_calls, 1) if self.apply_calls else 0,
            "aliases_applied": self.applied,
            "aliases_recomputed": self.recomputed,
            "alias_failure_reasons": dict(self.fail_reasons),
            "signature_phase_failed": self.sig_phase_failed,
            # An all-zero saving means one of two very different things. These separate them:
            # exchanges>0 with no drops = nothing worth merging; exchanges==0 = v2 never ran.
            "exchanges": self.exchanges,
            "exchange_skip_reasons": dict(self.skip_reasons),
            "blocks_withheld": self.blocks_withheld,
            "inert": self.is_inert(),
            "dedup_index_blocks": dict(sorted(self.index_blocks.items())),
            "lsh_accept_cos": {str(g): dict(zip(pd_lsh.ACCEPT_COS_LABELS, v))
                               for g, v in sorted(self.accept_cos.items())},
            "lsh_accept_rel_err": {str(g): dict(zip(pd_lsh.REL_ERR_LABELS, v))
                                   for g, v in sorted(self.accept_rel_err.items())},
            "threshold": threshold_for(0),
            "thresholds_per_group": {str(g): threshold_for(g) for g in groups},
            # What the substitution-error budget rejected on top of the cosine bar — i.e. what
            # taking the norm into account actually bought.
            "max_rel_err": pd_lsh.MAX_REL_ERR,
            "rejected_by_rel_err": {str(g): n
                                    for g, n in sorted(self.rejected_by_rel_err.items()) if n},
            # Where those rejections sat on the cosine axis. Anything at or below the bin containing
            # sqrt(1 - max_rel_err^2) is unreachable by ANY norm ratio, so this is what says whether
            # the budget is discarding near-duplicates or filtering noise.
            "min_cos_for_budget": pd_lsh.min_cos_for_budget(pd_lsh.MAX_REL_ERR),
            "lsh_reject_cos": {str(g): dict(zip(pd_lsh.ACCEPT_COS_LABELS, v))
                               for g, v in sorted(self.reject_cos.items()) if any(v)},
        }


class DedupEngine:
    """The decode's decision state: what to skip, what may be aliased, and what is still resident.

    Thread-safe because the decision is taken on a transport thread while the apply runs on the
    forward thread."""

    def __init__(self, lock=None, resident: bool | None = None) -> None:
        import threading
        self.lock = lock or threading.Lock()
        self.stats = DedupStats()
        self._planner = DedupPlanner()
        self._resident_enabled = V2_RESIDENT if resident is None else resident
        # Decided, transfer still in flight: {req_id: {group: {victim_block: (rep, rep_owner)}}}.
        # Keyed by the victim's BLOCK ID, not its slot, because a transfer covers only a request's
        # unhashed tail while the runner's table is the full list — matching by value removes the
        # offset entirely and rejects a stale entry for free.
        self._pending_alias: dict[str, dict[int, dict[int, tuple[int, str]]]] = {}
        # Decided AND landed. Only entries here are handed to the apply path.
        #
        # The distinction is the whole reason the first v2 run applied 22 of 26,531 aliases: the
        # apply path expires a map that has waited APPLY_MAX_AGE forward steps (~1.2 s) for its
        # owner to be batched, but an owner cannot be batched until its KV lands, which is the
        # entire remote round trip. Staging at decision time started that clock a whole transfer too
        # early, so essentially every alias expired and its never-written block went to recompute.
        self._alias_ready: dict[str, dict[int, dict[int, tuple[int, str]]]] = {}
        # Signatures of the blocks a transfer is actually fetching, held until it lands: only then
        # is the KV real and the block a legal alias target.
        self._pending_resident: dict[str, dict[int, tuple]] = {}
        # (group, block id) -> the request that brought it in, for the apply-time check.
        self._resident_owner: dict[tuple[int, int], str] = {}
        if self._resident_enabled:
            self._install_free_hook()

    # -- residency invalidation ----------------------------------------------------------
    def _install_free_hook(self) -> None:
        """Drop freed blocks from the index at the block pool's single release point.

        This is what makes cross-transfer aliasing safe. Preemption (61 of them in the run that
        motivated v2) frees a request's blocks without finishing it, so an index keyed off request
        completion would keep offering blocks that had already been reallocated."""
        try:
            from kv_fast_fusion import fast_fusion_block_pool as _bp
            _bp.on_blocks_freed(self.on_blocks_freed)
        except Exception as e:  # pragma: no cover - defensive
            from vllm.logger import init_logger
            init_logger("vllm.pd_dedup_v2").warning(
                "BFF v2: no block-free hook (%s) — disabling resident aliasing, which cannot be "
                "trusted without one.", e)
            self._resident_enabled = False

    def on_blocks_freed(self, freed_ids) -> None:
        with self.lock:
            self._planner.forget_any(freed_ids)
            for b in freed_ids:
                for key in [k for k in self._resident_owner if k[1] == int(b)]:
                    self._resident_owner.pop(key, None)

    # -- deciding ------------------------------------------------------------------------
    def plan(self, req_blocks: dict, signatures: dict, threshold=None) -> dict:
        """Replace the blocks this decode can satisfy locally with :data:`SENTINEL`.

        ``req_blocks[req_id]`` and the return value are per-group block id lists of the SAME length;
        ``signatures[req_id][group]`` is the producer's payload for that request's blocks in slot
        order. Aliases are held until the transfer lands, because a representative from this same
        transfer is not readable until then."""
        if not V2_ENABLED or not signatures:
            return req_blocks
        planned = {rid: [list(g) for g in groups] for rid, groups in req_blocks.items()}
        groups = sorted({int(gi) for per in signatures.values() for gi in per})

        for gi in groups:
            thr = threshold_for(gi) if threshold is None else threshold
            incoming: list[IncomingBlock] = []
            mats, hashes, norms = [], [], []
            row = 0
            for rid, per_group in req_blocks.items():
                payload = (signatures.get(rid) or {}).get(gi)
                if payload is None:
                    continue
                ids = per_group[gi] if gi < len(per_group) else []
                sig, nrm, hsh = SignatureCodec.decode(payload)
                if sig.shape[0] != len(ids):
                    # P and D disagree about this request's block count. Never guess: leave the
                    # request alone and it transfers exactly as vanilla would.
                    from vllm.logger import init_logger
                    init_logger("vllm.pd_dedup_v2").warning(
                        "BFF v2: %s group %d signature/block mismatch (%d vs %d) — "
                        "transferring it in full.", rid, gi, sig.shape[0], len(ids))
                    continue
                mats.append(sig)
                norms.extend(nrm)
                hashes.extend(hsh)
                for slot, bid in enumerate(ids):
                    incoming.append(IncomingBlock(rid, gi, slot, int(bid), row))
                    row += 1
            if not incoming:
                continue

            sigs = torch.cat(mats, dim=0)
            with self.lock:
                plan = self._planner.plan(gi, incoming, sigs, hashes, norms, thr)
                self.stats.planned[gi] = self.stats.planned.get(gi, 0) + len(incoming)
                self.stats.absorb(gi, plan)
                self._record(gi, plan, incoming, sigs, hashes, norms, planned)
                self.stats.index_blocks[gi] = self._planner.size(gi)
        return planned

    def _record(self, gi, plan, incoming, sigs, hashes, norms, planned) -> None:
        """Turn one group's plan into sentinels, pending aliases and pending residency."""
        by_slot = {(b.req_id, b.slot): b for b in incoming}
        owner_of = {b.block_id: b.req_id for b in incoming}

        for (rid, g), slots in plan.alias.items():
            for slot, rep in slots.items():
                victim = by_slot[(rid, slot)].block_id
                same_transfer = owner_of.get(int(rep))
                rep_owner = same_transfer or self._resident_owner.get((g, int(rep)))
                if rep_owner is None:
                    continue      # cannot name an owner → do not risk it; fetch the block
                planned[rid][g][slot] = SENTINEL
                self._pending_alias.setdefault(rid, {}).setdefault(g, {})[victim] = (
                    int(rep), rep_owner)
                d = self.stats.dropped_batch if same_transfer else self.stats.dropped_resident
                d[g] = d.get(g, 0) + 1

        for (rid, g), keep_slots in plan.keep.items():
            if not keep_slots:
                continue
            rows = [by_slot[(rid, s)].row for s in keep_slots]
            self._pending_resident.setdefault(rid, {})[g] = (
                sigs[torch.as_tensor(rows, dtype=torch.long)],
                [hashes[r] for r in rows],
                [norms[r] for r in rows],
                [by_slot[(rid, s)].block_id for s in keep_slots])

    # -- landing -------------------------------------------------------------------------
    def release(self, req_id: str) -> None:
        """This request's KV has landed: its aliases become appliable and its blocks become legal
        alias targets. Call this from the transport's completion signal, never earlier."""
        with self.lock:
            got = self._pending_alias.pop(req_id, None)
            if got:
                self._alias_ready.setdefault(req_id, {}).update(got)
            pend = self._pending_resident.pop(req_id, None)
        if not pend or not self._resident_enabled:
            return
        with self.lock:
            for gi, (sig, hsh, nrm, ids) in pend.items():
                self._planner.register(gi, sig, hsh, nrm, ids)
                for b in ids:
                    self._resident_owner[(gi, int(b))] = req_id
                self.stats.index_blocks[gi] = self._planner.size(gi)

    def forget(self, req_ids) -> None:
        """A transfer that failed leaves nothing written, so its aliases must not be applied."""
        with self.lock:
            for rid in req_ids:
                self._pending_alias.pop(rid, None)
                self._alias_ready.pop(rid, None)
                self._pending_resident.pop(rid, None)

    def drain_ready(self) -> dict:
        """Aliases whose KV has landed. See ``_alias_ready`` for why this is not ``_pending_alias``."""
        with self.lock:
            out, self._alias_ready = self._alias_ready, {}
            return out

    def is_resident(self, group: int, block_id: int) -> bool:
        return self._planner.is_resident(group, block_id)

    def resident_owner(self, group: int, block_id: int):
        """Which request registered this block, or None if it is not currently a legal target.

        Identity, not just presence. A block id can be freed and handed to a different request
        inside the apply deadline, and the index would then hold the SAME id with entirely
        different KV behind it. Checking residency alone cannot tell those apart, so any retry
        across time must compare this against the owner recorded when the alias was decided."""
        return self._resident_owner.get((group, int(block_id)))

    # -- test/introspection helpers ------------------------------------------------------
    def note_resident(self, group: int, sigs, hashes, norms, block_ids, owner="") -> None:
        if not self._resident_enabled:
            return
        self._planner.register(group, sigs, hashes, norms, block_ids)
        for b in block_ids:
            self._resident_owner[(group, int(b))] = owner
        self.stats.index_blocks[group] = self._planner.size(group)

    def pending_alias(self, req_id: str):
        return self._pending_alias.get(req_id)


class AliasApplier:
    """Point each aliased slot at its representative, then let the scheduler free the orphan.

    Held until the owner is batched because writing the runner's block table needs its
    ``input_batch`` row — still strictly before the request's first decode forward reads it. The
    failure handling is what differs from v1: an alias that cannot be applied means a block nobody
    wrote, so it goes to the KV-load-failure path rather than being quietly dropped.

    ``write_block_table(runner, rid, gi, blocks) -> bool`` is injected because each connector already
    owns a copy of it.

    ``normalize_req_id`` bridges two ID SPACES that are not interchangeable. The engine is keyed by
    whatever id the two nodes agreed on over the wire; the runner is keyed by this node's own
    EngineCore request id. On CUDA (with the stable-id patch) they are the same string and the
    default identity is correct. On Ascend they are not: vLLM PR #27987 appends a per-EngineCore
    9-character suffix, so the transport talks in ``get_external_request_id(rid) == rid[:-9]`` while
    ``runner.input_batch.req_id_to_index`` holds the full local id. Matching one against the other
    finds nothing, every alias ages out, and the whole run reports ``owner_never_batched`` — which
    is exactly what the first working NPU run did, 5216 of 5216. Normalising the RUNNER's ids into
    the engine's space (and writing back through the local id we came from) is what keeps the two
    joined."""

    def __init__(self, engine: DedupEngine, write_block_table, note_failed_blocks,
                 normalize_req_id=None) -> None:
        self._engine = engine
        self._write = write_block_table
        self._note_failed = note_failed_blocks
        self._norm = normalize_req_id or (lambda rid: rid)
        self.pending: dict[str, tuple[dict, int]] = {}
        self.pending_merges: dict | None = None
        self.step = 0

    def apply(self, runner) -> None:
        """Timed wrapper; the work is in :meth:`_apply`. See ``DedupStats.apply_ms``."""
        t0 = time.perf_counter()
        try:
            return self._apply(runner)
        finally:
            s = self._engine.stats
            s.apply_ms += (time.perf_counter() - t0) * 1000.0
            s.apply_calls += 1

    def _apply(self, runner) -> None:
        engine, stats = self._engine, self._engine.stats
        self.pending_merges = None
        self.step += 1
        for rid, by_group in engine.drain_ready().items():
            prev = self.pending.get(rid)
            if prev is None:
                self.pending[rid] = (by_group, self.step)
            else:
                for gi, m in by_group.items():
                    prev[0].setdefault(gi, {}).update(m)
        if not self.pending or runner is None:
            return
        batched = getattr(getattr(runner, "input_batch", None), "req_id_to_index", None)
        if batched is None:
            return
        # Keyed in the ENGINE's id space (see _norm); rid2local carries the way back, because the
        # block-table write and the merge channel both address the runner/scheduler by local id.
        rid2blocks: dict[str, Any] = {}
        rid2local: dict[str, str] = {}
        # Normalisation only has to be injective ACROSS THIS BATCH, and the transport's own keying
        # already assumes that. If it ever is not, the alias map cannot be attributed to a request
        # and resolving it anyway would rewrite the wrong request's block table — the one failure in
        # this file that is silent and unrecoverable. Refuse the key instead, and say so.
        ambiguous: set[str] = set()
        for rid_r in batched:
            st = getattr(runner, "requests", {}).get(rid_r)
            bids = getattr(st, "block_ids", None) if st is not None else None
            if bids is not None:
                key = self._norm(rid_r)
                if key in rid2blocks or key in ambiguous:
                    ambiguous.add(key)
                    rid2blocks.pop(key, None)
                    rid2local.pop(key, None)
                    continue
                rid2blocks[key] = bids
                rid2local[key] = rid_r

        updated: dict[str, dict[int, list[int]]] = {}
        failed: set[int] = set()
        n_applied = 0
        done: list[str] = []
        for rid, (by_group, first_step) in self.pending.items():
            if rid in ambiguous:
                done.append(rid)
                for m in by_group.values():
                    failed.update(m)
                    stats.note_failure("owner_id_ambiguous", len(m))
                continue
            if rid not in rid2blocks:
                if self.step - first_step > APPLY_MAX_AGE:
                    # It never came back. Those blocks hold nothing. Reaching this in bulk means
                    # the maps are being staged before the transfer lands — see DedupEngine.release.
                    done.append(rid)
                    for m in by_group.values():
                        failed.update(m)
                        stats.note_failure("owner_never_batched", len(m))
                continue
            local_rid = rid2local[rid]
            # Mutated in place, so a group that is retried keeps this request's original deadline.
            for gi, mapping in list(by_group.items()):
                new_blocks, why = self._substitute(rid2blocks, rid, gi, mapping)
                # Free ONLY when the device table was really rewritten — and recompute whenever it
                # was not, because unlike v1 the victim block holds nothing.
                if why is None and self._write(runner, local_rid, int(gi), new_blocks):
                    updated.setdefault(local_rid, {})[int(gi)] = new_blocks
                    n_applied += len(mapping)
                    del by_group[gi]
                    continue
                # "Not resident YET" is not "gone". A representative from the SAME transfer becomes
                # resident only when ITS request's release() runs, and nothing orders that before
                # the victim's — the two arrive in whatever order ok_reqs lists them, possibly in
                # different response messages. Treating the miss as terminal turned that race into a
                # full re-prefill: 601-765 requests per run, 100% of them rep_not_resident. Retry on
                # the deadline owner_never_batched already uses; a rep that is genuinely gone still
                # fails, just at the deadline instead of on the first look.
                if why == "rep_not_resident" and self.step - first_step <= APPLY_MAX_AGE:
                    continue
                del by_group[gi]
                failed.update(mapping)
                stats.note_failure(why or "block_table_write_refused", len(mapping))
            if not by_group:
                done.append(rid)
        for rid in done:
            self.pending.pop(rid, None)

        if failed:
            self._note_failed(failed)
        if updated:
            runner._updated_block_tables = updated
            self.pending_merges = updated
        if n_applied:
            stats.applied += n_applied
        return

    def _substitute(self, rid2blocks, rid, gi, mapping):
        """Build this request's new per-group block list, or name the reason it cannot.

        Refuses unless every representative is *still resident* — the index drops a block the moment
        the pool frees it, so this is an exact answer to "was it recycled since the decision", which
        is the only way an alias can point at the wrong KV.

        Returns ``(blocks, None)`` on success, ``(None, reason)`` otherwise."""
        groups = rid2blocks[rid]
        if gi >= len(groups):
            return None, "victim_not_in_table"
        blocks = [int(b) for b in groups[gi]]
        pos = {b: i for i, b in enumerate(blocks)}
        for victim, (rep, owner) in mapping.items():
            if victim not in pos:
                return None, "victim_not_in_table"
            if not self._engine.is_resident(gi, rep):
                return None, "rep_not_resident"
            # Residency says a block with this id is a legal target; it does not say it is the
            # block we chose. Between decision and apply the id can be freed and reissued to
            # another request, and applying then would point the victim at unrelated KV — silently,
            # which is the one failure mode in this file that no counter would catch. The owner
            # recorded at decision time is what distinguishes them.
            if self._engine.resident_owner(gi, rep) != owner:
                return None, "rep_recycled"
            blocks[pos[victim]] = int(rep)
        return blocks, None
