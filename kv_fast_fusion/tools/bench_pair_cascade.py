"""Correctness + CUDA-graph timing of bff_cascade.pair_cascade_decode vs plain FA decode.

Synthetic paged cache: B decode requests of ~CTX tokens; requests are paired and B's prompt blocks
at a fraction `share` of positions are redirected to A's blocks (raw BFF). Baseline = one FA decode
over the real (redirected) block tables. Usage:
  CUDA_VISIBLE_DEVICES=1 python -m kv_fast_fusion.tools.bench_pair_cascade
"""
import argparse
import random

import numpy as np
import torch

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
from kv_fast_fusion.bff_cascade import PairCascadeBuffers, pair_cascade_decode


def build(B, ctx, share, bs, nblocks, adjacent, seed=0):
    rng = random.Random(seed)
    free = list(range(1, nblocks))
    rng.shuffle(free)
    tables, lens = [], []
    pairs = [(2 * i, 2 * i + 1) for i in range(B // 2)]
    rows = {}
    for a, b in pairs:
        la = ctx + rng.randint(-200, 200)
        lb = ctx + rng.randint(-200, 200)
        na, nb = (la + bs - 1) // bs, (lb + bs - 1) // bs
        ta = [free.pop() for _ in range(na)]
        tb = [free.pop() for _ in range(nb)]
        # redirect a fraction of B's FULL prompt blocks (not its last, partial one) to A's full blocks
        full_b, full_a = nb - 1, na - 1
        k = int(share * min(full_a, full_b))
        pos_b = rng.sample(range(full_b), k)
        pos_a = rng.sample(range(full_a), k)
        for pb, pa in zip(pos_b, pos_a):
            tb[pb] = ta[pa]
        rows[a] = (ta, la); rows[b] = (tb, lb)
    order = list(range(B))
    if not adjacent:
        rng.shuffle(order)
    slot = {r: i for i, r in enumerate(order)}
    for r in order:
        tables.append(rows[r][0]); lens.append(rows[r][1])
    pairs = [(slot[a], slot[b]) for a, b in pairs]
    return tables, lens, pairs


def fill(bufs, tables, lens, pairs, bs):
    bufs.own_bt_cpu.zero_(); bufs.sh_bt_cpu.zero_(); bufs.sh_len_cpu.zero_()
    maxc = 1
    for r, (t, l) in enumerate(zip(tables, lens)):
        bufs.own_bt_cpu[r, :len(t)] = torch.tensor(t, dtype=torch.int32)
        bufs.own_len_cpu[r] = l
        maxc = max(maxc, len(t))
    a_rows, b_rows = [], []
    for a, b in pairs:
        ta, tb = tables[a], tables[b]
        sa = set(ta)
        shared = sorted({x for x in tb[:-1] if x in sa and x != ta[-1]})
        if not shared:
            continue
        sset = set(shared)
        own_a = [x for x in ta if x not in sset]
        own_b, seen = [], set()   # drop ONE occurrence of each shared block from B
        for x in tb:
            if x in sset and x not in seen:
                seen.add(x); continue
            own_b.append(x)
        for r, own in ((a, own_a), (b, own_b)):
            bufs.own_bt_cpu[r].zero_(); bufs.own_bt_cpu[r, :len(own)] = torch.tensor(own, dtype=torch.int32)
            bufs.own_len_cpu[r] = lens[r] - len(shared) * bs
        p = len(a_rows)
        bufs.sh_bt_cpu[p, :len(shared)] = torch.tensor(shared, dtype=torch.int32)
        bufs.sh_len_cpu[p] = len(shared) * bs
        a_rows.append(a); b_rows.append(b)
    bufs.set_pairs(np.array(a_rows, np.int64), np.array(b_rows, np.int64), len(tables))
    bufs.max_cols = maxc
    bufs.commit(len(tables))


def time_graph(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        g.replay()
    st.record()
    for _ in range(iters):
        g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1e3  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--Hk", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--shares", default="0,0.2,0.4,0.6,0.8")
    ap.add_argument("--maxk", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    nblocks = args.B * (args.ctx + 400) // args.bs + 64
    kc = torch.randn(nblocks, args.bs, args.Hk, args.D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    q = torch.randn(args.B, args.H, args.D, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(args.B + 1, device=dev, dtype=torch.int32)
    fa = get_flash_attn_version()
    scale = args.D ** -0.5
    maxk = args.maxk or (args.ctx + 512)
    print(f"FA version {fa} | B={args.B} ctx~{args.ctx} bs={args.bs} H={args.H} Hk={args.Hk}")
    print(f"{'share':>5} {'order':>9} {'redund%':>7} {'base us':>8} {'casc us':>8} {'speedup':>7} {'max|err|':>9}")
    for share in [float(x) for x in args.shares.split(",")]:
        for adjacent in (True, False):
            tables, lens, pairs = build(args.B, args.ctx, share, args.bs, nblocks, adjacent)
            M = max(len(t) for t in tables)
            bt = torch.zeros(args.B, M, dtype=torch.int32)
            for r, t in enumerate(tables):
                bt[r, :len(t)] = torch.tensor(t, dtype=torch.int32)
            bt = bt.to(dev)
            seqused = torch.tensor(lens, dtype=torch.int32, device=dev)
            refs = sum(len(t) for t in tables); uniq = len({x for t in tables for x in t})
            out_b = torch.empty_like(q)
            def base():
                flash_attn_varlen_func(q=q, k=kc, v=vc, out=out_b, cu_seqlens_q=cu, max_seqlen_q=1,
                                       seqused_k=seqused, max_seqlen_k=maxk, softmax_scale=scale,
                                       causal=True, block_table=bt, fa_version=fa)
            bufs = PairCascadeBuffers(args.B, M, args.H, args.Hk, dev)
            fill(bufs, tables, lens, pairs, args.bs)
            torch.cuda.synchronize()
            out_c = torch.empty_like(q)
            def casc():
                pair_cascade_decode(out_c, q, kc, vc, bufs, cu, maxk, scale, fa)
            base(); casc(); torch.cuda.synchronize()
            err = (out_b.float() - out_c.float()).abs().max().item()
            tb_, tc_ = time_graph(base), time_graph(casc)
            print(f"{share:5.1f} {'adjacent' if adjacent else 'shuffled':>9} {100*(1-uniq/refs):7.1f} "
                  f"{tb_:8.1f} {tc_:8.1f} {tb_/tc_:7.3f} {err:9.2e}")




def replay(path, H=32, Hk=8, D=128, maxk=16384):
    """Replay one real decode step dumped by BFF_CASCADE_DUMP (block tables + cascade buffers)."""
    z = np.load(path)
    bt, seq, bs = z["bt"], z["seq"], int(z["block_size"])
    B, M = bt.shape
    dev = torch.device("cuda")
    nblocks = int(max(bt.max(), z["sh_bt"].max(), z["own_bt"].max())) + 1
    kc = torch.randn(nblocks, bs, Hk, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    fa = get_flash_attn_version(); scale = D ** -0.5
    bt_g = torch.from_numpy(bt.astype(np.int32)).to(dev)
    seq_g = torch.from_numpy(seq.astype(np.int32)).to(dev)
    bufs = PairCascadeBuffers(B, M, H, Hk, dev)
    bufs.own_bt_cpu[:, :M] = torch.from_numpy(z["own_bt"].astype(np.int32))
    bufs.own_len_cpu[:] = torch.from_numpy(z["own_len"].astype(np.int32))
    P1 = bufs.max_pairs + 1
    n_sh = min(P1, z["sh_bt"].shape[0])
    bufs.sh_bt_cpu[:n_sh, :M] = torch.from_numpy(z["sh_bt"][:n_sh].astype(np.int32))
    bufs.sh_len_cpu[:n_sh] = torch.from_numpy(z["sh_len"][:n_sh].astype(np.int32))
    bufs.set_pairs(z["a_rows"].astype(np.int64), z["b_rows"].astype(np.int64), B)
    bufs.max_cols = M
    bufs.commit(B); torch.cuda.synchronize()
    nb = [(int(s) + bs - 1) // bs for s in seq]
    refs = sum(nb); uniq = len({int(x) for r in range(B) for x in bt[r, :nb[r]]})
    pair_saved = int(z["sh_len"].sum()) // bs
    from collections import Counter
    cnt = Counter(int(x) for r in range(B) for x in bt[r, :nb[r]])
    print(f"{path}: B={B} refs={refs} unique={uniq} redundant={100*(1-uniq/refs):.1f}% "
          f"pair-saved={100*pair_saved/refs:.1f}% pairs={len(z['a_rows'])} max_refcount={max(cnt.values())} "
          f"blocks_shared_by>2={sum(1 for v in cnt.values() if v > 2)}")
    out_b = torch.empty_like(q); out_c = torch.empty_like(q)
    def base():
        flash_attn_varlen_func(q=q, k=kc, v=vc, out=out_b, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seq_g,
                               max_seqlen_k=maxk, softmax_scale=scale, causal=True, block_table=bt_g, fa_version=fa)
    def casc():
        pair_cascade_decode(out_c, q, kc, vc, bufs, cu, maxk, scale, fa)
    def own_only():
        flash_attn_varlen_func(q=q, k=kc, v=vc, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=bufs.own_len[:B],
                               max_seqlen_k=maxk, softmax_scale=scale, causal=True, block_table=bufs.own_bt[:B],
                               return_softmax_lse=True, fa_version=fa)
    def shared_only():
        qs = q.reshape(B * H, D).index_select(0, bufs.qidx).view(P1, 2 * H, D)
        flash_attn_varlen_func(q=qs, k=kc, v=vc, cu_seqlens_q=bufs.cu_pairs, max_seqlen_q=1, seqused_k=bufs.sh_len,
                               max_seqlen_k=maxk, softmax_scale=scale, causal=False, block_table=bufs.sh_bt,
                               return_softmax_lse=True, fa_version=fa)
    base(); casc(); torch.cuda.synchronize()
    err = (out_b.float() - out_c.float()).abs().max().item()
    tb, tc, to, ts = (time_graph(f) for f in (base, casc, own_only, shared_only))
    print(f"   base {tb:.1f}us  cascade {tc:.1f}us (x{tb/tc:.3f})  own {to:.1f}us  shared {ts:.1f}us  err {err:.2e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--replay":
        for f in sys.argv[2:]:
            replay(f)
    else:
        main()
