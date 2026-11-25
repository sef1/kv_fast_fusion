"""Online (cross-step), stall-free BFF fusion for the single-instance raw path (`BFF_ONLINE=1`).

Worker side, per fusion call (requests whose prefill finished this step), per fused KV group:
  1. fingerprint each new prompt block (bf16 JL projection of the group's K, unit-normalized);
  2. match against a REGISTRY of resident representative blocks (requests fused earlier and still
     alive): cos > THRESHOLD -> redirect to that block;
  3. nr_tree among the call's remaining blocks, sync-free: node composition is host-known (nodes keep
     full size, redirected blocks are masked out), and a block that became a target can never be
     redirected itself -> every member is compared directly with its final representative;
  4. append the call's surviving blocks to the registry;
  5. copy result codes to pinned memory + record an event. Nothing on the CPU waits.

Commit protocol (safe under async scheduling, where a target's owner may finish concurrently):
  worker PROPOSES (member, pos, dup, target, owner, owner_pos) when the event completes (a later step)
  -> scheduler CONFIRMS in update_from_output (member still holds dup at pos, owner still RUNNING and
  holds target at owner_pos), touches target, frees dup -> next SchedulerOutput carries the confirmed
  redirects -> worker APPLIES them to its block table before that step's forward. The member reads its
  own (still allocated) block until then, so a rejected or late proposal is always exact.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger("vllm.bff_online")

ONLINE_ENABLED = os.environ.get("BFF_ONLINE", "0") == "1"
THRESHOLD = float(os.environ.get("BFF_THRESHOLD", "0.75"))
# Registry matches take the max over thousands of candidates, so a fixed bar is exceeded far more often
# than in-call matching (extreme-value effect); give them their own, stricter bar.
REG_THRESHOLD = float(os.environ.get("BFF_REG_THRESHOLD", str(THRESHOLD)))
REG_SLOTS = int(os.environ.get("BFF_REG_SLOTS", "8192"))       # registry capacity per fused group
USE_REGISTRY = os.environ.get("BFF_ONLINE_REGISTRY", "1") == "1"
LOG_EVERY = int(os.environ.get("BFF_ONLINE_LOG_EVERY", "50"))  # fusion calls between stat lines


# ----------------------------------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------------------------------
class _Registry:
    def __init__(self, dim, device):
        R = REG_SLOTS
        self.x = torch.zeros((R, dim), dtype=torch.float16, device=device)
        self.valid = torch.zeros((R,), dtype=torch.bool, device=device)
        self.bid = np.zeros(R, np.int64)
        self.owner = np.full(R, -1, np.int64)
        self.pos = np.zeros(R, np.int64)
        self.used = 0            # high-water mark of written slots
        self.next = 0            # ring pointer
        self.slots_of = defaultdict(list)

    def alloc(self, n):
        R = REG_SLOTS
        n = min(n, R)
        s = (self.next + np.arange(n)) % R
        self.next = int((self.next + n) % R)
        self.used = min(R, max(self.used, int(s.max()) + 1)) if n else self.used
        return s


class OnlineFuser:
    def __init__(self, runner):
        self.r = runner
        self.dev = runner.device
        self.bs = runner.cache_config.block_size
        self.reg = {}
        self.h_of, self.req_of_h, self._next_h = {}, {}, 0
        self.pending = []
        self.proposals = None
        self.calls = 0
        self.stats = defaultdict(int)
        self._jl = None
        self._sched = {}  # tree schedule cache keyed by sizes tuple

    # -- helpers ------------------------------------------------------------------------------
    def _h(self, req_id):
        h = self.h_of.get(req_id)
        if h is None:
            h = self.h_of[req_id] = self._next_h
            self.req_of_h[h] = req_id
            self._next_h += 1
        return h

    def _groups(self):
        for g, grp in enumerate(self.r.kv_cache_config.kv_cache_groups):
            if g == 0 or not grp.layer_names:
                continue
            yield g, [int(ln.split('.')[2]) for ln in grp.layer_names]

    def _jl_mat(self, d_full):
        if self._jl is None:
            from kv_fast_fusion.legacy.kv_fast_fusion_graph_runner import (
                LSH_PROJ_DIM, _get_jl_matrix)
            self._jl = _get_jl_matrix(self.r, d_full, LSH_PROJ_DIM, self.dev).to(torch.bfloat16)
        return self._jl

    def _repr(self, layer_idxs, ids_t):
        parts = []
        for li in layer_idxs:
            k = self.r.kv_caches[li][0].index_select(0, ids_t)     # [N, bs, Hk, D]
            k = k.view(k.shape[0], -1).to(torch.bfloat16)
            parts.append(k @ self._jl_mat(k.shape[1]))
        rep = torch.cat(parts, dim=1).float() if len(parts) > 1 else parts[0].float()
        return F.normalize(rep, dim=1).to(torch.float16)

    def _tree_pairs(self, sizes):
        """Host-side nr_tree schedule: nodes keep full size (no shrinking), so all pair index lists
        are known up front. Returns concatenated (L, R) index arrays + per-pair offsets."""
        key = tuple(sizes)
        if key in self._sched:
            return self._sched[key]
        offs = np.concatenate([[0], np.cumsum(sizes)])
        nodes = [np.arange(offs[i], offs[i + 1]) for i in range(len(sizes)) if sizes[i] > 0]
        Ls, Rs, meta = [], [], []
        while len(nodes) > 1:
            nodes.sort(key=len)
            nxt = []
            for k in range(0, len(nodes) - (len(nodes) & 1), 2):
                L, R = nodes[k], nodes[k + 1]
                meta.append((sum(map(len, Ls)), len(L), sum(map(len, Rs)), len(R)))
                Ls.append(L); Rs.append(R)
                nxt.append(np.concatenate([L, R]))
            if len(nodes) & 1:
                nxt.append(nodes[-1])
            nodes = nxt
        cat = lambda a: np.concatenate(a) if a else np.zeros(0, np.int64)
        res = (cat(Ls), cat(Rs), meta)
        if len(self._sched) < 256:
            self._sched[key] = res
        return res

    # -- step hooks ------------------------------------------------------------------------------
    def begin_step(self, so):
        """After _update_states: apply confirmed redirects, invalidate gone owners, harvest results."""
        r = self.r
        conf = getattr(so, "bff_confirmed", None)
        if conf:
            ib = r.input_batch
            n_app = 0
            for rid, groups in conf.items():
                row = ib.req_id_to_index.get(rid)
                st = r.requests.get(rid)
                if row is None or st is None:
                    continue
                for g, items in groups.items():
                    tbl = ib.block_table.block_tables[g].block_table.np
                    for pos, dup, tgt in items:
                        if tbl[row, pos] == dup:
                            tbl[row, pos] = tgt
                            st.block_ids[g][pos] = tgt
                            n_app += 1
            self.stats["applied"] += n_app
        gone = set(so.finished_req_ids or ())
        gone |= set(getattr(so, "preempted_req_ids", None) or ())
        if gone and self.reg:
            for g, reg in self.reg.items():
                kill = []
                for rid in gone:
                    h = self.h_of.get(rid)
                    if h is None:
                        continue
                    sl = reg.slots_of.pop(h, None)
                    if sl:
                        sl = np.asarray(sl)
                        kill.append(sl[reg.owner[sl] == h])
                if kill:
                    k = np.concatenate(kill)
                    if k.size:
                        reg.owner[k] = -1
                        reg.valid.index_fill_(0, torch.from_numpy(k).to(self.dev, non_blocking=True), False)
        for rid in gone:
            h = self.h_of.pop(rid, None)
            if h is not None:
                self.req_of_h.pop(h, None)
        self._harvest()

    def _harvest(self):
        props = defaultdict(lambda: defaultdict(list))
        keep = []
        for p in self.pending:
            if not p["ev"].query():
                keep.append(p); continue
            codes = p["codes"].numpy()
            N = len(p["ids"])
            for i in np.nonzero(codes >= 0)[0]:
                c = int(codes[i])
                member = self.req_of_h.get(int(p["own"][i]))
                if member is None:
                    continue
                if c < N:
                    tgt, oh, opos = int(p["ids"][c]), int(p["own"][c]), int(p["pos"][c])
                    self.stats["m_tree"] += 1
                else:
                    s = c - N
                    tgt, oh, opos = int(p["snap_bid"][s]), int(p["snap_owner"][s]), int(p["snap_pos"][s])
                    self.stats["m_reg"] += 1
                owner = self.req_of_h.get(oh)
                if owner is None:
                    continue
                props[member][p["g"]].append((int(p["pos"][i]), int(p["ids"][i]), tgt, owner, opos))
            self.stats["blocks"] += N
        self.pending = keep
        self.proposals = {m: dict(gs) for m, gs in props.items()} if props else None

    def launch(self, req_ids):
        """Enqueue fusion for requests whose prefill finished this step. No host sync."""
        r = self.r
        ib = r.input_batch
        reqs = []
        for rid in req_ids:
            row = ib.req_id_to_index.get(rid)
            st = r.requests.get(rid)
            if row is None or st is None:
                continue
            n = st.num_prompt_tokens // self.bs
            r.fused_requests.setdefault(rid, {})
            if n > 0:
                reqs.append((rid, row, n))
        if not reqs:
            return
        self.calls += 1
        sizes = [n for _, _, n in reqs]
        Ls, Rs, meta = self._tree_pairs(sizes)
        N = int(sum(sizes))
        own = np.concatenate([np.full(n, self._h(rid), np.int64) for rid, _, n in reqs])
        pos = np.concatenate([np.arange(n, dtype=np.int64) for _, _, n in reqs])
        L_t = torch.from_numpy(Ls.astype(np.int64)).to(self.dev, non_blocking=True)
        R_t = torch.from_numpy(Rs.astype(np.int64)).to(self.dev, non_blocking=True)
        ar = torch.arange(N, device=self.dev)
        for g, layer_idxs in self._groups():
            tbl = ib.block_table.block_tables[g].block_table.np
            ids = np.concatenate([tbl[row, :n] for _, row, n in reqs]).astype(np.int64)
            ids_t = torch.from_numpy(ids).to(self.dev, non_blocking=True)
            X = self._repr(layer_idxs, ids_t)                                  # [N, D] fp16
            reg = self.reg.get(g)
            if reg is None and USE_REGISTRY:
                reg = self.reg[g] = _Registry(X.shape[1], self.dev)
            alive = ids_t > 0
            code = torch.full((N,), -1, dtype=torch.int64, device=self.dev)
            snap = None
            if USE_REGISTRY and reg.used > 0:
                S = X @ reg.x[:reg.used].T                                     # [N, used]
                S.masked_fill_(~reg.valid[:reg.used].unsqueeze(0), float("-inf"))
                best, arg = S.max(dim=1)
                m = (best.float() > REG_THRESHOLD) & alive
                code = torch.where(m, N + arg, code)
                alive = alive & ~m
                snap = (reg.bid.copy(), reg.owner.copy(), reg.pos.copy())
                del S
            # sync-free nr_tree among this call's remaining blocks
            parent = ar.clone()
            is_tgt = torch.zeros(N, dtype=torch.int8, device=self.dev)
            for (lo, ln, ro, rn) in meta:
                Li = L_t[lo:lo + ln]; Ri = R_t[ro:ro + rn]
                S = X.index_select(0, Ri) @ X.index_select(0, Li).T            # [rn, ln]
                S.masked_fill_(~alive.index_select(0, Li).unsqueeze(0), float("-inf"))
                best, arg = S.max(dim=1)
                m = ((best.float() > THRESHOLD) & alive.index_select(0, Ri)
                     & (is_tgt.index_select(0, Ri) == 0))
                tgt = Li.index_select(0, arg)
                parent.index_copy_(0, Ri, torch.where(m, tgt, parent.index_select(0, Ri)))
                alive.index_copy_(0, Ri, alive.index_select(0, Ri) & ~m)
                is_tgt.scatter_reduce_(0, tgt, m.to(torch.int8), reduce="amax")
            code = torch.where(parent != ar, parent, code)
            if USE_REGISTRY:
                slots = reg.alloc(N)
                k = len(slots)
                sl_t = torch.from_numpy(slots).to(self.dev, non_blocking=True)
                reg.x.index_copy_(0, sl_t, X[:k])
                reg.valid.index_copy_(0, sl_t, alive[:k])
                reg.bid[slots] = ids[:k]; reg.owner[slots] = own[:k]; reg.pos[slots] = pos[:k]
                for h in np.unique(own[:k]):
                    reg.slots_of[int(h)].extend(slots[own[:k] == h].tolist())
            codes_cpu = torch.empty(N, dtype=torch.int64, pin_memory=True)
            codes_cpu.copy_(code, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            p = {"g": g, "ev": ev, "codes": codes_cpu, "ids": ids, "own": own, "pos": pos}
            if snap is not None:
                p["snap_bid"], p["snap_owner"], p["snap_pos"] = snap
            self.pending.append(p)
        if LOG_EVERY and self.calls % LOG_EVERY == 0:
            s = self.stats
            logger.info(
                "BFF online | calls=%d | blocks=%d | matched tree=%d registry=%d (%.1f%%) | "
                "applied=%d | confirm ok=%d rejected=%d | reg_used=%s",
                self.calls, s["blocks"], s["m_tree"], s["m_reg"],
                100 * (s["m_tree"] + s["m_reg"]) / max(1, s["blocks"]), s["applied"],
                _SCHED_STATS["ok"], _SCHED_STATS["rej"],
                {g: reg.used for g, reg in self.reg.items()})


def get_fuser(runner) -> OnlineFuser:
    f = runner.__dict__.get("_bff_online")
    if f is None:
        f = runner.__dict__["_bff_online"] = OnlineFuser(runner)
    return f


# ----------------------------------------------------------------------------------------------
# scheduler side
# ----------------------------------------------------------------------------------------------
_SCHED_STATS = {"ok": 0, "rej": 0}


def _confirm(sched, proposals) -> None:
    from vllm.v1.request import RequestStatus
    pool = sched.kv_cache_manager.coordinator.block_pool
    mgrs = sched.kv_cache_manager.coordinator.single_type_managers
    confirmed = getattr(sched, "_bff_confirmed", None)
    if confirmed is None:
        confirmed = sched._bff_confirmed = {}
    to_touch, to_free = [], []
    ok = rej = 0
    for member, groups in proposals.items():
        req = sched.requests.get(member)
        if req is None or req.status != RequestStatus.RUNNING:
            rej += sum(len(v) for v in groups.values()); continue
        for g, items in groups.items():
            mgr = mgrs[g]
            mblocks = mgr.req_to_blocks.get(member)
            if not mblocks:
                rej += len(items); continue
            for pos, dup, tgt, owner, opos in items:
                oreq = sched.requests.get(owner)
                oblocks = mgr.req_to_blocks.get(owner) if oreq is not None else None
                if (pos >= len(mblocks) or mblocks[pos].block_id != dup
                        or oreq is None or oreq.status != RequestStatus.RUNNING
                        or not oblocks or opos >= len(oblocks) or oblocks[opos].block_id != tgt):
                    rej += 1
                    continue
                old = mblocks[pos]
                new = pool.blocks[tgt]
                mblocks[pos] = new
                to_touch.append(new); to_free.append(old)
                confirmed.setdefault(member, {}).setdefault(g, []).append((pos, dup, tgt))
                ok += 1
    if to_touch:
        pool.touch(to_touch)
    if to_free:
        pool.free_blocks(to_free)
    _SCHED_STATS["ok"] += ok
    _SCHED_STATS["rej"] += rej


def install_scheduler_hooks() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    if getattr(Scheduler, "_bff_online_hooked", False):
        return
    prev_update = Scheduler.update_from_output
    prev_schedule = Scheduler.schedule

    def update_from_output(self, scheduler_output, model_runner_output):
        props = getattr(model_runner_output, "_bff_proposals", None)
        if props:
            try:
                _confirm(self, props)
            except Exception as e:  # never break serving
                logger.warning("BFF online confirm failed: %s", e, exc_info=True)
        return prev_update(self, scheduler_output, model_runner_output)

    def schedule(self, *a, **kw):
        out = prev_schedule(self, *a, **kw)
        conf = getattr(self, "_bff_confirmed", None)
        if conf:
            out.bff_confirmed = conf
            self._bff_confirmed = {}
        return out

    Scheduler.update_from_output = update_from_output
    Scheduler.schedule = schedule
    Scheduler._bff_online_hooked = True
