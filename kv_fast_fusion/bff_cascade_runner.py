"""Runner-side bookkeeping for BFF pair-cascade decode (`BFF_CASCADE=1`, single-instance raw mode).

* `record_pairs` (after each fusion call): per fused KV group, greedily pair the just-fused requests by
  how many prompt blocks they share, and remember for each pair the shared multiset S, each side's
  static own blocks (its prompt blocks minus S), and a snapshot of its prompt row (detects a later
  rewrite, e.g. preemption + resume).
* `fill` (every uniform-decode metadata build, incl. FULL-graph capture): stage the own-pass block table
  (= the normal row, or own_static + decode tail for paired rows whose partner is also decoding), the
  shared tables and the gather maps, then commit them non-blocking into the persistent GPU buffers
  that the captured graph reads.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from vllm.logger import init_logger

from kv_fast_fusion.bff_cascade import PairCascadeBuffers

logger = init_logger("vllm.bff_cascade")

CASCADE_ENABLED = os.environ.get("BFF_CASCADE", "0") == "1"
_LOG_EVERY = int(os.environ.get("BFF_CASCADE_LOG_EVERY", "500"))
_MAX_OWNERS = 16
_DUMP = os.environ.get("BFF_CASCADE_DUMP", "")
_DUMP_STEP = int(os.environ.get("BFF_CASCADE_DUMP_STEP", "300"))


@dataclass
class _Pair:
    a: str
    b: str
    shared: np.ndarray       # int32 block ids (multiset intersection of the two prompt rows)
    own_a: np.ndarray        # a's prompt blocks minus shared
    own_b: np.ndarray
    npf_a: int               # a's full prompt blocks at fusion time (row prefix covered by own_a+shared)
    npf_b: int
    snap_a: np.ndarray       # a's row[:npf_a] at fusion time
    snap_b: np.ndarray


def _state(runner):
    st = runner.__dict__.get("_bff_casc")
    if st is None:
        st = {"pairs": defaultdict(dict), "bufs": {}, "step": 0,
              "tok_total": 0, "tok_shared": 0, "rows": 0, "rows_paired": 0, "dropped": 0}
        runner.__dict__["_bff_casc"] = st
    return st


def _multiset_minus(row: np.ndarray, remove: Counter) -> np.ndarray:
    left = Counter(remove)
    keep = []
    for x in row.tolist():
        if left.get(x, 0) > 0:
            left[x] -= 1
        else:
            keep.append(x)
    return np.asarray(keep, dtype=np.int32)


def record_pairs(runner, req_ids: list[str]) -> None:
    if not CASCADE_ENABLED or len(req_ids) < 2:
        return
    st = _state(runner)
    ib = runner.input_batch
    bs = runner.cache_config.block_size
    for g, bt in enumerate(ib.block_table.block_tables):
        if g == 0:
            continue  # warmup group is never fused
        bt_np = bt.block_table.np
        rows, npf = {}, {}
        for rid in req_ids:
            if rid not in ib.req_id_to_index or rid not in runner.requests:
                continue
            r = ib.req_id_to_index[rid]
            n = runner.requests[rid].num_prompt_tokens // bs
            rows[rid] = bt_np[r, :n].copy()
            npf[rid] = n
        owners = defaultdict(list)
        for rid, row in rows.items():
            for b in set(row.tolist()):
                if b > 0:
                    owners[b].append(rid)
        overlap = Counter()
        for lst in owners.values():
            if len(lst) > 1:
                for x, y in combinations(lst[:_MAX_OWNERS], 2):
                    overlap[(x, y)] += 1
        used = set()
        for (x, y), _ in overlap.most_common():
            if x in used or y in used:
                continue
            shared_ms = Counter(rows[x].tolist()) & Counter(rows[y].tolist())
            shared_ms.pop(0, None)
            if not shared_ms:
                continue
            used.add(x); used.add(y)
            shared = np.asarray(list(shared_ms.elements()), dtype=np.int32)
            p = _Pair(a=x, b=y, shared=shared,
                      own_a=_multiset_minus(rows[x], shared_ms), own_b=_multiset_minus(rows[y], shared_ms),
                      npf_a=npf[x], npf_b=npf[y], snap_a=rows[x], snap_b=rows[y])
            st["pairs"][g][x] = p
            st["pairs"][g][y] = p


def _bufs(runner, st, g, bt):
    b = st["bufs"].get(g)
    if b is None:
        mc = runner.model_config
        pc = runner.parallel_config
        b = PairCascadeBuffers(
            max_reqs=bt.block_table.np.shape[0], max_blocks=bt.block_table.np.shape[1],
            num_heads=mc.get_num_attention_heads(pc), num_kv_heads=mc.get_num_kv_heads(pc),
            device=runner.device)
        st["bufs"][g] = b
    return b


def fill(runner, g: int, num_reqs: int, num_reqs_padded: int, capture: bool) -> PairCascadeBuffers:
    st = _state(runner)
    ib = runner.input_batch
    bt = ib.block_table.block_tables[g]
    bufs = _bufs(runner, st, g, bt)
    bs = runner.cache_config.block_size
    bt_np, nbr = bt.block_table.np, bt.num_blocks_per_row
    seq = runner.seq_lens.np
    maxc = int(nbr[:num_reqs].max()) if num_reqs else 1
    maxc = max(maxc, 1)
    own_bt = bufs.own_bt_cpu.numpy()
    own_len = bufs.own_len_cpu.numpy()
    own_bt[:num_reqs, :maxc] = bt_np[:num_reqs, :maxc]
    own_len[:num_reqs] = seq[:num_reqs]
    own_len[num_reqs:num_reqs_padded] = 0
    sh_bt = bufs.sh_bt_cpu.numpy()
    sh_len = bufs.sh_len_cpu.numpy()
    sh_len[:] = 0
    a_rows, b_rows = [], []
    pairs = st["pairs"].get(g)
    if pairs and not capture:
        idx = ib.req_id_to_index
        done = set()
        for rid in ib.req_ids[:num_reqs]:
            p = pairs.get(rid)
            if p is None or id(p) in done:
                continue
            done.add(id(p))
            ra, rb = idx.get(p.a), idx.get(p.b)
            if ra is None or rb is None or ra >= num_reqs or rb >= num_reqs:
                continue  # partner not decoding this step -> both run unpaired
            if not (np.array_equal(bt_np[ra, :p.npf_a], p.snap_a)
                    and np.array_equal(bt_np[rb, :p.npf_b], p.snap_b)):
                pairs.pop(p.a, None); pairs.pop(p.b, None); st["dropped"] += 1
                continue
            k = len(a_rows)
            ns = len(p.shared)
            for r, own, npf in ((ra, p.own_a, p.npf_a), (rb, p.own_b, p.npf_b)):
                tail = bt_np[r, npf:nbr[r]]
                no = len(own)
                own_bt[r, :no] = own
                own_bt[r, no:no + len(tail)] = tail
                own_len[r] = seq[r] - ns * bs
            sh_bt[k, :ns] = p.shared
            sh_len[k] = ns * bs
            maxc = max(maxc, ns)
            a_rows.append(ra); b_rows.append(rb)
            st["tok_shared"] += 2 * ns * bs
        st["tok_total"] += int(seq[:num_reqs].sum())
        st["rows"] += num_reqs
        st["rows_paired"] += 2 * len(a_rows)
    bufs.set_pairs(np.asarray(a_rows, np.int64), np.asarray(b_rows, np.int64), num_reqs)
    bufs.max_cols = maxc
    bufs.commit(num_reqs_padded)
    if _DUMP and not capture and a_rows and st["step"] >= _DUMP_STEP and g not in st.setdefault("dumped", set()):
        st["dumped"].add(g)
        np.savez(f"{_DUMP}_g{g}.npz", bt=bt_np[:num_reqs, :maxc].copy(), nbr=nbr[:num_reqs].copy(),
                 seq=seq[:num_reqs].copy(), own_bt=own_bt[:num_reqs, :maxc].copy(),
                 own_len=own_len[:num_reqs].copy(), sh_bt=sh_bt[:, :maxc].copy(), sh_len=sh_len.copy(),
                 a_rows=np.asarray(a_rows), b_rows=np.asarray(b_rows), block_size=bs)
        logger.info("BFF cascade dump -> %s_g%d.npz (pairs=%d rows=%d)", _DUMP, g, len(a_rows), num_reqs)
    return bufs


def gc_and_log(runner) -> None:
    """Drop pair records of requests that left the runner; periodic stats."""
    if not CASCADE_ENABLED:
        return
    st = _state(runner)
    st["step"] += 1
    if st["step"] % 64 == 0:
        live = runner.requests
        for g, pairs in st["pairs"].items():
            for rid in [r for r, p in pairs.items() if p.a not in live or p.b not in live]:
                pairs.pop(rid, None)
    if _LOG_EVERY and st["step"] % _LOG_EVERY == 0 and st["tok_total"]:
        logger.info(
            "BFF cascade | step=%d | shared_tok_frac=%.1f%% (fused groups) | rows_paired=%.1f%% | "
            "live_pairs=%s | dropped=%d", st["step"], 100 * st["tok_shared"] / st["tok_total"],
            100 * st["rows_paired"] / max(1, st["rows"]),
            {g: len(p) // 2 for g, p in st["pairs"].items()}, st["dropped"])
