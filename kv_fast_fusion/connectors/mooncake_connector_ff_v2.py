"""Mooncake BFF v2 — the producer ships signatures, the decode decides what to pull.

v1 put the merge decision on the PRODUCER: it clustered blocks during its forward pass, shipped a
redirect map on the transfer ACK, and the decode rewrote its block table after the KV had already
landed. Two measurements condemned that arrangement:

* **73% of redirects resolved to nothing** (7,171 applied against 19,834 unresolved). The producer
  cannot see what is resident on the decode, so most of its merge decisions were guesses about
  another process's memory.
* **Every block was transferred anyway**, including the ones freed milliseconds after arrival. Under
  Mooncake's pull model the producer cannot release a block until the decode has taken it, so those
  wasted transfers are what pinned the producer at 99.6% KV for an entire run (2928/2928 samples,
  3 free blocks of 18617, nothing running) while the decode sat at 56% — compression relieving a
  resource that was never scarce, and starving the one that was.

v2 inverts the roles. The producer computes a cheap per-block **signature** and nothing else; the
decode, which owns the memory and knows exactly what is resident, decides which blocks it does not
need and simply never asks for them. Two consequences:

* an alias always points at a block the decode holds, so "unresolved" stops being a possible
  outcome rather than a number to minimise;
* a deduplicated block is never written, so the saving lands on the wire and on the producer's
  residency — the actual bottleneck.

The producer also loses its forward-path hook entirely. Signatures are computed on demand from
``device_kv_caches`` (populated once in ``register_kv_caches``), so v2 needs no ``save_kv_layer``
work, no chunked-prefill accumulation, and no PIECEWISE cudagraph constraint.

**The one hazard v2 introduces, and how it is contained.** In v1 a failed apply was harmless: the
block held its own correct KV and you merely lost compression. Here a dropped block is *never
written*, so an alias that cannot be applied would leave the decode reading whatever was in that
block before. Every such case is therefore routed into vLLM's KV-load-failure path
(``_ff_failed_blocks`` → ``get_block_ids_with_load_errors`` → ``Scheduler._handle_invalid_blocks``),
which recomputes the request locally. Slower, never wrong. The three places this can happen — the
owner never gets batched, the block table write fails, the representative was freed in the meantime
— are each handled explicitly in ``_ff_consumer_apply`` below.

Registered as ``MooncakeConnectorFFv2`` alongside v1 so both can be run against the same benchmark.
Everything hard-won about the HMA/group-aware transfer path is inherited from v1 unchanged: this
subclasses its worker and connector rather than restating them.
"""

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Any

import torch
import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket

from kv_fast_fusion import pd_lsh
from kv_fast_fusion.connectors import mooncake_connector_ff as v1
from kv_fast_fusion.pd_dedup_plan import DedupPlanner, IncomingBlock

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = init_logger("vllm.mooncake_connector_ff_v2")

# Signature width. The decode verifies an exact cosine against these, so this is the only place
# fidelity is traded for wire size: 128 fp16 dims is ~256 B against a block of KV that is ~1.6 MB,
# i.e. 0.02%. Shared JL projection, fixed seed, so P and D agree without negotiating.
_SIG_DIM = int(os.environ.get("BFF_SIG_DIM", "128"))
# Master switch. Off disables the ENTIRE mechanism, signature phase included — the pull reverts to
# a single round trip and this is the "BFF group split, no fusion" control arm, not a measurement
# of what the signature exchange costs. (Two runs with this off differ only by noise, which is how
# the ±5% throughput / ±0.011 F1 noise band was established.) To price the exchange itself, leave
# this on and set BFF_THRESHOLD to 1.01 so every candidate is rejected.
_V2_ENABLED = os.environ.get("BFF_V2_DEDUP", "1") == "1"
# Alias to blocks left over from EARLIER pulls, not just duplicates inside the current one. This is
# where cross-request reuse would come from; it is safe only because the decode's block pool calls
# forget_resident on every release (fast_fusion_block_pool.on_blocks_freed), so a block leaves the
# index strictly before it can be handed to another request.
_V2_RESIDENT = os.environ.get("BFF_V2_RESIDENT", "1") == "1"
# Phase 1 is best-effort: a producer that cannot answer within these seconds gets pulled in full.
# Both are far below the transfer timeouts — a slow signature phase must never become a stall.
_SIG_PHASE_TIMEOUT = float(os.environ.get("BFF_V2_SIG_TIMEOUT", "10"))
_SIG_READY_TIMEOUT = float(os.environ.get("BFF_V2_READY_TIMEOUT", "10"))
# Marks a block the decode decided not to pull. It rides in place of the block id rather than being
# removed, because the P↔D pairing is POSITIONAL (mooncake_connector_ff._build_transfer_params tail-
# aligns and zips): a shortened list would pair the surviving blocks against the wrong source.
_SENTINEL = -1


def _signature_matrix(kv_layers: list, block_ids: list[int], is_mla: bool, jl_holder: list):
    """Concatenate the group's per-layer K for ``block_ids`` and project to ``_SIG_DIM``.

    Mirrors ``FFProducerFusion.block_repr`` + the concat the clustering used, so a v2 signature
    means the same thing a v1 representation did and the two runs stay comparable. Returns
    ``(normalised [N, d], norms [N])`` on the caller's device."""
    if not block_ids:
        return None, None
    idx = torch.as_tensor(block_ids, dtype=torch.long, device=kv_layers[0].device)
    parts = []
    for kv in kv_layers:
        blk = (kv[idx] if is_mla else kv[0, idx]).float()
        parts.append(blk.reshape(idx.shape[0], -1))
    full = torch.cat(parts, dim=1)
    if jl_holder[0] is None or jl_holder[0].shape[0] != full.shape[1]:
        g = torch.Generator(device="cpu")
        g.manual_seed(20240517)
        jl_holder[0] = torch.randn(full.shape[1], _SIG_DIM, generator=g,
                                   dtype=torch.float32).to(full.device)
    sig = full @ jl_holder[0]
    norms = sig.norm(dim=1).clamp(min=1e-6)
    return sig / norms.unsqueeze(1), norms


class SignatureCodec:
    """Pack/unpack the per-block signatures that cross the wire.

    fp16 for the vectors (the decode verifies a cosine, which does not need fp32) and int64 bucket
    ids, both as plain lists so the existing msgpack structs carry them without a new encoder."""

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


# Why an alias could not be applied. One counter per distinct cause, because the first v2 run
# reported a single number for all four and the root cause had to be inferred from the step rate.
FAIL_REASONS = ("owner_never_batched", "rep_not_resident", "victim_not_in_table",
                "block_table_write_refused")


class DedupStats:
    """What v2 did, in the same units the collector already prints for v1.

    Per group wherever v1 was per group: the threshold question is a per-group question (each
    group's blocks have their own similarity floor), and a run that reports one pooled histogram
    cannot answer it."""

    def __init__(self) -> None:
        self.planned: dict[int, int] = {}            # blocks considered
        self.dropped_resident: dict[int, int] = {}   # served by a block already on D
        self.dropped_batch: dict[int, int] = {}      # served by another block in the same pull
        self.applied = 0            # aliases that reached the block table
        self.recomputed = 0         # aliases that could not be applied -> local recompute
        self.fail_reasons = dict.fromkeys(FAIL_REASONS, 0)
        self.sig_phase_failed = 0   # pulls that fell back to a full request
        self.accept_cos: dict[int, list] = {}
        self.accept_rel_err: dict[int, list] = {}
        self.rejected_by_rel_err: dict[int, int] = {}
        self.index_blocks: dict[int, int] = {}
        self._last_dump = 0.0

    def absorb(self, gi: int, plan) -> None:
        """Fold in one group's plan histograms.

        Deliberately NOT plan.n_resident / plan.n_batch: a block is only counted as saved once the
        sentinel is actually written into the request, and the caller can still decline a merge the
        planner proposed. Counting the proposal is how v1 ended up reporting 34,812 merges of which
        7,171 were real."""
        cos = self.accept_cos.setdefault(gi, [0] * len(pd_lsh.ACCEPT_COS_LABELS))
        err = self.accept_rel_err.setdefault(gi, [0] * len(pd_lsh.REL_ERR_LABELS))
        for i, c in enumerate(plan.accept_cos):
            cos[i] += c
        for i, c in enumerate(plan.accept_rel_err):
            err[i] += c
        self.rejected_by_rel_err[gi] = (self.rejected_by_rel_err.get(gi, 0)
                                        + plan.rejected_by_rel_err)

    def note_failure(self, reason: str, n: int = 1) -> None:
        self.recomputed += n
        self.fail_reasons[reason] = self.fail_reasons.get(reason, 0) + n

    def should_dump(self, step: int) -> bool:
        """Same cadence rule as v1's fusion engine: step count OR wall clock, because the step rate
        varies enough between arms that a count-only rule pinned the file at its first snapshot."""
        now = time.monotonic()
        if step <= 3 or now - self._last_dump >= v1._PD_STATS_DUMP_SEC:
            self._last_dump = now
            return True
        return False

    def dump(self, stats_dir: str = v1._PD_STATS_DIR) -> None:
        """Atomically write to ``<stats_dir>/bff_stats_<pid>.json``, the file the collector reads."""
        try:
            path = os.path.join(stats_dir, f"bff_stats_{os.getpid()}.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.stats_dict(), f)
            os.replace(tmp, path)   # atomic — the reader never sees a half-written file
        except Exception as e:  # pragma: no cover - defensive (never break the transfer)
            logger.warning("BFF v2: could not dump stats: %s", e)

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
            # Per group, so the threshold can be set per group. A group whose saving is high AND
            # whose accepted cosines hug the bar is matching noise, not finding duplicates.
            "wire_saving_per_group": {
                str(g): {
                    "planned": self.planned.get(g, 0),
                    "not_requested": (self.dropped_resident.get(g, 0)
                                      + self.dropped_batch.get(g, 0)),
                    "pct": (100.0 * (self.dropped_resident.get(g, 0)
                                     + self.dropped_batch.get(g, 0)) / self.planned[g])
                    if self.planned.get(g) else 0.0,
                } for g in groups},
            "aliases_applied": self.applied,
            "aliases_recomputed": self.recomputed,
            "alias_failure_reasons": dict(self.fail_reasons),
            "signature_phase_failed": self.sig_phase_failed,
            "dedup_index_blocks": dict(sorted(self.index_blocks.items())),
            "lsh_accept_cos": {str(g): dict(zip(pd_lsh.ACCEPT_COS_LABELS, v))
                               for g, v in sorted(self.accept_cos.items())},
            "lsh_accept_rel_err": {str(g): dict(zip(pd_lsh.REL_ERR_LABELS, v))
                                   for g, v in sorted(self.accept_rel_err.items())},
            "threshold": _threshold(0),
            "thresholds_per_group": {str(g): _threshold(g) for g in groups},
            # What the substitution-error budget rejected on top of the cosine bar — i.e. what
            # taking the norm into account actually bought.
            "max_rel_err": pd_lsh.MAX_REL_ERR,
            "rejected_by_rel_err": {str(g): n
                                    for g, n in sorted(self.rejected_by_rel_err.items()) if n},
        }


def _threshold(gi: int) -> float:
    """Same knob as v1 (``BFF_THRESHOLD`` / ``BFF_THRESHOLD_G``), so a v2 run is directly
    comparable with a v1 one at the same setting. At matched norms it is an error budget, not a
    similarity preference: 0.75 authorises replacing a block with one that differs by 0.707 of its
    own magnitude — see ``pd_lsh.rel_err``."""
    from kv_fast_fusion.constants import THRESHOLD
    return v1._THRESHOLD_G.get(gi, float(THRESHOLD))


if v1._MOONCAKE_AVAILABLE:      # the same gate v1 uses

    import vllm.envs as envs
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeXferResponseStatus,
    )

    class MooncakeConnectorWorkerFFv2(v1.MooncakeConnectorWorkerFF):
        """Adds signature service on the producer and dedup planning on the decode.

        Inherits v1's ``_build_transfer_params`` unchanged — the group-aware pairing is orthogonal
        to who decides, and it is the part that took the longest to get right."""

        def __init__(self, vllm_config, engine_id):
            super().__init__(vllm_config, engine_id)
            self._jl: list = [None]
            self._planner = DedupPlanner()
            self._dedup = DedupStats()
            # Decided at pull time, applied once the transfer has completed and the owner is
            # batched: {req_id: {group: {victim_block_id: (rep_block_id, rep_owner_req_id)}}}.
            # Keyed by the victim's BLOCK ID, not its slot, because the pull request covers only a
            # request's unhashed tail while the runner's table is the full list — matching by value
            # removes the offset entirely and rejects a stale entry for free.
            self._pending_alias: dict[str, dict[int, dict[int, tuple[int, str]]]] = {}
            # Decided AND transferred. Only entries here are handed to the apply path.
            #
            # The distinction is the whole reason the first v2 run applied 22 of 26,531 aliases: the
            # apply path expires a map that has waited _FF_APPLY_MAX_AGE (16) forward steps ~= 1.2 s
            # for its owner to be batched, but an owner cannot be batched until its KV lands, which
            # is the entire remote round trip. Staging at decision time started that clock a whole
            # transfer too early, so essentially every alias expired and its never-written block went
            # to recompute. v1 never hit this because its redirect rows rode the transfer ACK.
            self._alias_ready: dict[str, dict[int, dict[int, tuple[int, str]]]] = {}
            # Signatures of the blocks a pull is actually fetching, held until it completes: only
            # then is the KV real and the block a legal alias target.
            self._pending_resident: dict[str, dict[int, tuple]] = {}
            # (group, block id) -> the request that brought it in, for the apply-time check.
            self._resident_owner: dict[tuple[int, int], str] = {}
            if _V2_RESIDENT:
                self._install_free_hook()

        # -- residency invalidation ------------------------------------------------------
        def _install_free_hook(self) -> None:
            """Drop freed blocks from the dedup index at the block pool's single release point.

            This is what makes cross-pull aliasing safe. Preemption (61 of them in the run that
            motivated v2) frees a request's blocks without finishing it, so a residency index keyed
            off request completion would keep offering blocks that had already been reallocated."""
            try:
                from kv_fast_fusion import fast_fusion_block_pool as _bp
                _bp.on_blocks_freed(self._on_blocks_freed)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF v2: no block-free hook (%s) — disabling resident aliasing, "
                               "which cannot be trusted without one.", e)
                globals()["_V2_RESIDENT"] = False

        def _on_blocks_freed(self, freed_ids) -> None:
            with self._ff_lock:
                self._planner.forget_any(freed_ids)

        # -- producer: answer with signatures, never with decisions ---------------------
        def signatures_for(self, block_ids_by_group: dict[int, list[int]]) -> dict:
            """Compute this producer's signatures for the requested blocks, per fusion group.

            Reads straight out of ``device_kv_caches``, so it costs the producer nothing on the
            forward path — the reason v2 needs no ``save_kv_layer`` hook and no PIECEWISE
            cudagraph constraint."""
            out: dict[int, dict] = {}
            caches = getattr(self, "device_kv_caches", None) or {}
            for gi, block_ids in block_ids_by_group.items():
                layers = [caches[ln] for ln in sorted(self._group_layers.get(gi, ()))
                          if ln in caches]
                if not layers or not block_ids:
                    continue
                is_mla = layers[0].ndim == 3
                sig, norms = _signature_matrix(layers, block_ids, is_mla, self._jl)
                if sig is None:
                    continue
                proj = pd_lsh.get_proj([None], sig.shape[1], sig.device)
                hashes = pd_lsh.sub_hashes_device(sig, proj).cpu().tolist()
                out[gi] = SignatureCodec.encode(sig, norms, hashes)
            return out

        async def send_kv_to_decode(self, identity, sock, meta):
            """Phase 1 answers with signatures and writes nothing; phase 2 is the stock transfer."""
            if getattr(meta, "want_signatures", False):
                return await self._send_signatures(identity, sock, meta)
            return await super().send_kv_to_decode(identity, sock, meta)

        async def _send_signatures(self, identity, sock, meta) -> None:
            """Reply with per-block signatures for exactly the blocks phase 2 would write.

            The rows are computed over ``local_ids[-len(remote_ids):]`` — the same tail alignment
            ``_build_transfer_params`` uses — so signature row *i* is the decode's slot *i*."""
            out: dict[str, dict[int, dict]] = {}
            marked = []
            try:
                for d_req_id, (transfer_id, remote_groups) in meta.req_blocks.items():
                    send_meta = self.reqs_need_send.get(transfer_id)
                    if send_meta is None:
                        continue
                    try:
                        await asyncio.wait_for(send_meta.ready.wait(), _SIG_READY_TIMEOUT)
                    except (TimeoutError, asyncio.TimeoutError):
                        continue          # not prefilled yet → D pulls this one whole
                    if transfer_id not in self.reqs_need_send:
                        continue          # expired between the wait and here
                    # Hold off the abort-timeout sweep for the duration: it only reaps entries with
                    # sending == 0, and this run showed that sweep firing 16 times.
                    send_meta.sending += 1
                    marked.append(send_meta)
                    local_groups = send_meta.local_block_ids
                    per_group: dict[int, list[int]] = {}
                    for gi, remote_ids in enumerate(remote_groups):
                        if gi <= 0 or not remote_ids:
                            continue      # group 0 is the warmup group — never fused
                        local_ids = local_groups[gi] if gi < len(local_groups) else []
                        if len(local_ids) < len(remote_ids):
                            continue      # P has fewer blocks than D wants; phase 2 will error
                        per_group[gi] = list(local_ids[-len(remote_ids):])
                    sigs = self.signatures_for(per_group)
                    if sigs:
                        out[d_req_id] = sigs
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF v2: signature phase failed (%s) — D will pull in full.", e)
                out = {}
            finally:
                for sm in marked:
                    sm.sending -= 1
            response = v1.MooncakeXferResponseFF(
                status=MooncakeXferResponseStatus.FINISH, signatures=out or None)
            await sock.send_multipart((identity, self._encoder.encode(response)))

        # -- decode: plan the pull ------------------------------------------------------
        def plan_pull(self, req_blocks: dict, signatures: dict, threshold=None) -> dict:
            """Replace the blocks this decode can satisfy locally with ``_SENTINEL``.

            ``req_blocks[req_id]`` and the return value are per-group block id lists of the SAME
            length; ``signatures[req_id][group]`` is the producer's payload for that request's
            blocks in slot order. The aliases are held until the transfer completes, because a
            representative from this same pull is not readable until then."""
            if not _V2_ENABLED or not signatures:
                return req_blocks
            planned = {rid: [list(g) for g in groups] for rid, groups in req_blocks.items()}
            groups = sorted({int(gi) for per in signatures.values() for gi in per})

            for gi in groups:
                thr = _threshold(gi) if threshold is None else threshold
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
                        # request alone and it is pulled exactly as vanilla would pull it.
                        logger.warning("BFF v2: %s group %d signature/block mismatch (%d vs %d) — "
                                       "pulling it in full.", rid, gi, sig.shape[0], len(ids))
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
                with self._ff_lock:
                    plan = self._planner.plan(gi, incoming, sigs, hashes, norms, thr)
                    self._dedup.planned[gi] = self._dedup.planned.get(gi, 0) + len(incoming)
                    self._dedup.absorb(gi, plan)
                    self._record_plan(gi, plan, incoming, sigs, hashes, norms, planned)
                    self._dedup.index_blocks[gi] = self._planner.size(gi)
            return planned

        def _record_plan(self, gi, plan, incoming, sigs, hashes, norms, planned) -> None:
            """Turn one group's plan into sentinels, pending aliases and pending residency."""
            by_slot = {(b.req_id, b.slot): b for b in incoming}
            owner_of = {b.block_id: b.req_id for b in incoming}

            for (rid, g), slots in plan.alias.items():
                for slot, rep in slots.items():
                    victim = by_slot[(rid, slot)].block_id
                    same_pull = owner_of.get(int(rep))
                    rep_owner = same_pull or self._resident_owner.get((g, int(rep)))
                    if rep_owner is None:
                        continue      # cannot name an owner → do not risk it; pull the block
                    planned[rid][g][slot] = _SENTINEL
                    self._pending_alias.setdefault(rid, {}).setdefault(g, {})[victim] = (
                        int(rep), rep_owner)
                    d = self._dedup.dropped_batch if same_pull else self._dedup.dropped_resident
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

        def take_pending_alias(self, req_id: str):
            with self._ff_lock:
                return self._pending_alias.pop(req_id, None)

        def release_alias(self, req_id: str) -> None:
            """Promote a request's aliases to appliable, now that its transfer has completed."""
            with self._ff_lock:
                got = self._pending_alias.pop(req_id, None)
                if got:
                    self._alias_ready.setdefault(req_id, {}).update(got)

        def drain_pending_alias(self) -> dict:
            """Only aliases whose KV has landed. See ``_alias_ready`` for why this matters."""
            with self._ff_lock:
                out, self._alias_ready = self._alias_ready, {}
                return out

        def note_resident(self, group: int, sigs, hashes, norms, block_ids, owner="") -> None:
            """Blocks whose transfer completed are now valid alias targets."""
            if not _V2_RESIDENT:
                return
            self._planner.register(group, sigs, hashes, norms, block_ids)
            for b in block_ids:
                self._resident_owner[(group, int(b))] = owner
            self._dedup.index_blocks[group] = self._planner.size(group)

        def forget_resident(self, group: int, block_ids) -> int:
            n = self._planner.forget(group, block_ids)
            for b in block_ids:
                self._resident_owner.pop((group, int(b)), None)
            self._dedup.index_blocks[group] = self._planner.size(group)
            return n

        def is_resident(self, group: int, block_id: int) -> bool:
            return self._planner.is_resident(group, block_id)

        # -- decode: the two-phase pull --------------------------------------------------
        async def receive_kv_from_single_worker(self, worker_addr, pull_metas):
            """Ask for signatures, decide, then request only what is left.

            Replaces the stock body (which builds one metadata and sends it) rather than wrapping
            it, because the deduplicated block list has to go into the message it builds. The
            transfer half below is the stock loop verbatim."""
            req_ids = set(pull_metas)
            base = {rid: [list(g) for g in pm.local_block_ids] for rid, pm in pull_metas.items()}
            planned = base

            if _V2_ENABLED:
                signatures = await self._request_signatures(worker_addr, pull_metas, base)
                if signatures:
                    try:
                        planned = self.plan_pull(base, signatures)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning("BFF v2: planning failed (%s) — pulling in full.", e)
                        self._forget_pending(req_ids)
                        planned = base

            metadata = v1.MooncakeXferMetadataFF(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks={rid: (pm.transfer_id, planned[rid])
                            for rid, pm in pull_metas.items()},
                kv_caches_base_addr=self.kv_caches_base_addr,
            )
            try:
                with make_zmq_socket(
                    self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
                ) as sock:
                    sock.setsockopt(
                        zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000)
                    await sock.send(self._encoder.encode(metadata))
                    while True:
                        ret_msg = await sock.recv()
                        response = self._xfer_resp_decoder.decode(ret_msg)
                        if response.status == MooncakeXferResponseStatus.ERROR:
                            logger.error("Error transferring kvcache for %s: %s",
                                         req_ids, response.err_msg)
                            self._forget_pending(req_ids)
                            return
                        self.process_pulling_result(response, pull_metas)
                        if response.status == MooncakeXferResponseStatus.FINISH:
                            break
            except zmq.ContextTerminated:
                self._forget_pending(req_ids)
                logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
            except Exception as e:
                self._forget_pending(req_ids)
                logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)

        async def _request_signatures(self, worker_addr, pull_metas, base) -> dict:
            """Phase 1, on its own short-lived socket.

            Best-effort by construction: on any failure or timeout this returns nothing and the
            caller pulls the full block list, which is exactly what vanilla would have sent. A slow
            producer must cost compression, never a stall — hence the separate socket, so a late
            reply can never be mistaken for a phase-2 response."""
            meta = v1.MooncakeXferMetadataFF(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks={rid: (pm.transfer_id, base[rid]) for rid, pm in pull_metas.items()},
                kv_caches_base_addr=self.kv_caches_base_addr,
                want_signatures=True,
            )
            try:
                with make_zmq_socket(
                    self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
                ) as sock:
                    sock.setsockopt(zmq.RCVTIMEO, int(_SIG_PHASE_TIMEOUT * 1000))
                    await sock.send(self._encoder.encode(meta))
                    raw = await asyncio.wait_for(sock.recv(), _SIG_PHASE_TIMEOUT)
                    return self._xfer_resp_decoder.decode(raw).signatures or {}
            except Exception as e:
                self._dedup.sig_phase_failed += 1
                logger.warning("BFF v2: signature phase unavailable (%s) — pulling in full.", e)
                return {}

        def _forget_pending(self, req_ids) -> None:
            """A pull that failed leaves nothing written, so its aliases must not be applied."""
            with self._ff_lock:
                for rid in req_ids:
                    self._pending_alias.pop(rid, None)
                    self._alias_ready.pop(rid, None)
                    self._pending_resident.pop(rid, None)

        def process_pulling_result(self, response, pull_metas):
            """v1's failed-pull recovery, plus the two things that become true only now that the
            KV has actually landed: this request's aliases may be applied, and the blocks it did
            pull may serve as representatives for later ones."""
            for rid in (response.err_reqs or []):
                self._forget_pending([rid])
            super().process_pulling_result(response, pull_metas)
            for rid in (response.ok_reqs or []):
                self.release_alias(rid)
                with self._ff_lock:
                    pend = self._pending_resident.pop(rid, None)
                if not pend:
                    continue
                with self._ff_lock:
                    for gi, (sig, hsh, nrm, ids) in pend.items():
                        self.note_resident(gi, sig, hsh, nrm, ids, owner=rid)

        def note_failed_blocks(self, block_ids) -> None:
            """Route blocks that were never written into vLLM's KV-load-failure path."""
            if not block_ids:
                return
            with self._ff_lock:
                self._ff_failed_blocks |= {int(b) for b in block_ids}

    class MooncakeConnectorFFv2(v1.MooncakeConnectorFF):
        """v1's connector with the merge decision moved to the decode.

        The producer-side fusion engine, the redirect-row wire format and the whole
        resolve/hold/expire path on the consumer are simply not used: v2 never emits a redirect that
        might not resolve, so there is nothing to resolve, hold, or expire."""

        _WORKER_CLS = MooncakeConnectorWorkerFFv2

        @classmethod
        def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
            """v1 demanded PIECEWISE because it ran real Python per layer inside ``save_kv_layer``.
            v2 does no forward-path work at all, so a full cudagraph is fine again."""
            return False

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
            """No-op: signatures are computed on demand from the registered KV cache."""

        def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
            # Deliberately skips v1's start_load_kv body (which drives the producer fusion engine
            # and the redirect apply) and calls the transport base directly.
            v1.MooncakeConnector.start_load_kv(self, forward_context, **kwargs)
            self._ff_step += 1
            if self.is_producer:
                return
            self._ff_consumer_apply()
            stats = getattr(self.connector_worker, "_dedup", None)
            if stats is not None and stats.should_dump(self._ff_step):
                stats.dump()

        def get_kv_connector_stats(self):
            merges = self._ff_pending_merges
            self._ff_pending_merges = None
            if merges and self._tp_group() is not None:
                return v1.BFFMergeStats(data={"bff_merges": merges})
            return v1.MooncakeConnector.get_kv_connector_stats(self)

        # -- consumer apply -------------------------------------------------------------
        def _ff_consumer_apply(self) -> None:
            """Point each aliased slot at its representative, then let the scheduler free the
            orphan.

            Held until the owner is batched because ``write_runner_block_table`` needs its
            ``input_batch`` row — the same timing v1 used, and still strictly before the request's
            first decode forward reads the table. What is new is the failure handling: an alias
            that cannot be applied means a block nobody wrote, so it goes to the KV-load-failure
            path rather than being dropped."""
            self._ff_pending_merges = None
            try:
                worker = self.connector_worker
                for rid, by_group in worker.drain_pending_alias().items():
                    prev = self._ff_pending.get(rid)
                    if prev is None:
                        self._ff_pending[rid] = (by_group, self._ff_step)
                    else:
                        for gi, m in by_group.items():
                            prev[0].setdefault(gi, {}).update(m)
                if not self._ff_pending:
                    return

                from kv_fast_fusion import fast_fusion_block_pool as _bp
                runner = getattr(_bp, "_ACTIVE_RUNNER", None)
                if runner is None:
                    return
                batched = getattr(getattr(runner, "input_batch", None), "req_id_to_index", None)
                if batched is None:
                    return
                rid2blocks: dict[str, Any] = {}
                for rid_r in batched:
                    st = getattr(runner, "requests", {}).get(rid_r)
                    bids = getattr(st, "block_ids", None) if st is not None else None
                    if bids is not None:
                        rid2blocks[rid_r] = bids

                updated: dict[str, dict[int, list[int]]] = {}
                failed: set[int] = set()
                n_applied = 0
                done: list[str] = []
                for rid, (by_group, first_step) in self._ff_pending.items():
                    if rid not in rid2blocks:
                        if self._ff_step - first_step > v1._FF_APPLY_MAX_AGE:
                            # It never came back. Those blocks hold nothing. Reaching this in bulk
                            # means the maps are being staged too early — see _alias_ready.
                            done.append(rid)
                            for m in by_group.values():
                                failed.update(m)
                                worker._dedup.note_failure("owner_never_batched", len(m))
                        continue
                    for gi, mapping in by_group.items():
                        new_blocks, why = self._substitute(worker, rid2blocks, rid, gi, mapping)
                        # Free ONLY when the device table was really rewritten — and recompute
                        # whenever it was not, because unlike v1 the victim block holds nothing.
                        if why is None and v1.write_runner_block_table(
                                runner, rid, int(gi), new_blocks):
                            updated.setdefault(rid, {})[int(gi)] = new_blocks
                            n_applied += len(mapping)
                        else:
                            failed.update(mapping)
                            worker._dedup.note_failure(
                                why or "block_table_write_refused", len(mapping))
                    done.append(rid)
                for rid in done:
                    self._ff_pending.pop(rid, None)

                if failed:
                    worker.note_failed_blocks(failed)
                    reasons = " ".join(f"{k}={v}" for k, v in worker._dedup.fail_reasons.items()
                                       if v)
                    logger.warning(
                        "BFF v2: %d aliased block(s) could not be redirected — their KV was never "
                        "written, so the owning requests go to local recompute. "
                        "cumulative causes: %s", len(failed), reasons or "none")
                if updated:
                    runner._updated_block_tables = updated
                    self._ff_pending_merges = updated
                if n_applied:
                    worker._dedup.applied += n_applied
                    self._ff_applied += n_applied
                    logger.info("BFF v2 apply | aliases_applied=%d | recompute=%d",
                                n_applied, len(failed))
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("BFF v2 consumer apply failed: %s", e)

        @staticmethod
        def _substitute(worker, rid2blocks, rid, gi, mapping):
            """Build this request's new per-group block list, or name the reason it cannot.

            Refuses unless every representative is *still resident* — the index drops a block the
            moment the pool frees it, so this is an exact answer to "was it recycled since the
            decision", which is the only way an alias can point at the wrong KV.

            Returns ``(blocks, None)`` on success, ``(None, reason)`` otherwise."""
            groups = rid2blocks[rid]
            if gi >= len(groups):
                return None, "victim_not_in_table"
            blocks = [int(b) for b in groups[gi]]
            pos = {b: i for i, b in enumerate(blocks)}
            for victim, (rep, _owner) in mapping.items():
                if victim not in pos:
                    return None, "victim_not_in_table"
                if not worker.is_resident(gi, rep):
                    return None, "rep_not_resident"
                blocks[pos[victim]] = int(rep)
            return blocks, None


def register_mooncake_connector_ff_v2() -> None:
    """Register ``MooncakeConnectorFFv2`` (idempotent), by path + name so importing this module's
    Mooncake half is not required on a box without the package."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    if "MooncakeConnectorFFv2" in KVConnectorFactory._registry:
        return
    KVConnectorFactory.register_connector(
        "MooncakeConnectorFFv2",
        "kv_fast_fusion.connectors.mooncake_connector_ff_v2",
        "MooncakeConnectorFFv2",
    )
    logger.info("Fast fusion P/D patch: registered MooncakeConnectorFFv2.")
