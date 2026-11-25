"""Grouped (pair) cascade attention for BFF raw-mode decode.

Raw BFF makes two requests reference the same physical KV block. Stock paged attention still reads
that block once per request. For a fused PAIR (A, B) with shared block set S this computes:

  shared pass : one FA decode "sequence" per pair whose single query token carries BOTH requests'
                heads, ordered [kv_head, request, group] -> GQA ratio 2G, so FA's seqlen_q==1 path packs
                both requests into one row tile and reads each block of S once;
  own pass    : a normal FA decode over each request's remaining blocks (unpaired rows: all blocks);
  merge       : fused softmax-LSE merge reading the pair's shared result in place.

Exact in raw mode: keys are stored post-RoPE, so decode attention is a permutation-invariant softmax over
the request's blocks, and splitting them into two disjoint sets is lossless. Every tensor has a static
shape for a given padded batch, so the sequence is CUDA-graph (FULL) capturable; the per-step pairing
lives only in buffer CONTENTS. Glue is one flat gather (q pack) and one fused Triton merge that reads the shared result in place;
the maps are built on the CPU by `PairCascadeBuffers.set_pairs`.
"""
from __future__ import annotations

import numpy as np
import torch

from vllm.triton_utils import tl, triton

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


@triton.jit
def _merge_pair_shared_kernel(
    out_ptr, own_ptr, own_lse_ptr, sh_ptr, sh_lse_ptr, pair_ptr, side_ptr,
    B, P1, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
):
    """out[r,h] = LSE-merge(own[r,h], shared[pair(r), packed_head(side(r), h)]); pair < 0 -> own."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    own = tl.load(own_ptr + (r * H + h) * D + d).to(tl.float32)
    o_lse = tl.load(own_lse_ptr + h * B + r)
    o_lse = tl.where(o_lse == float("inf"), float("-inf"), o_lse)  # FA early exit (empty) -> -inf
    p = tl.load(pair_ptr + r)
    if p >= 0:
        t = tl.load(side_ptr + r)
        hp = (h // G) * 2 * G + t * G + h % G
        sh = tl.load(sh_ptr + (p * 2 * H + hp) * D + d).to(tl.float32)
        s_lse = tl.load(sh_lse_ptr + hp * P1 + p)
        s_lse = tl.where(s_lse == float("inf"), float("-inf"), s_lse)
        m = tl.maximum(o_lse, s_lse)
        m = tl.where(m == float("-inf"), 0.0, m)
        w_o = tl.exp(o_lse - m)
        w_s = tl.exp(s_lse - m)
        den = w_o + w_s
        den = tl.where(den == 0.0, 1.0, den)
        res = (own * w_o + sh * w_s) / den
    else:
        res = own
    tl.store(out_ptr + (r * H + h) * D + d, res.to(out_ptr.dtype.element_ty))


class PairCascadeBuffers:
    """Persistent per-KV-group buffers (GPU + pinned CPU staging), sized for the max batch.

    Pair slots [0, P) hold real pairs; slot P (= max_pairs) is spare (zero shared length). Rows with
    pair == -1 take their own pass verbatim in the fused merge kernel."""

    def __init__(self, max_reqs: int, max_blocks: int, num_heads: int, num_kv_heads: int,
                 device: torch.device):
        self.max_reqs = B = max_reqs
        self.max_pairs = P = max(1, max_reqs // 2)
        self.max_blocks = M = max_blocks
        self.H, self.Hk = H, Hk = num_heads, num_kv_heads
        self.G = G = H // Hk
        P1 = P + 1
        pin = dict(pin_memory=True)
        i32, i64 = torch.int32, torch.int64
        self.own_bt_cpu = torch.zeros((B, M), dtype=i32, **pin)
        self.own_len_cpu = torch.zeros((B,), dtype=i32, **pin)
        self.sh_bt_cpu = torch.zeros((P1, M), dtype=i32, **pin)
        self.sh_len_cpu = torch.zeros((P1,), dtype=i32, **pin)
        self.qidx_cpu = torch.zeros((P1 * 2 * H,), dtype=i64, **pin)
        self.pair_cpu = torch.full((B,), -1, dtype=i32, **pin)
        self.side_cpu = torch.zeros((B,), dtype=i32, **pin)
        g = dict(device=device)
        self.own_bt = torch.zeros((B, M), dtype=i32, **g)
        self.own_len = torch.zeros((B,), dtype=i32, **g)
        self.sh_bt = torch.zeros((P1, M), dtype=i32, **g)
        self.sh_len = torch.zeros((P1,), dtype=i32, **g)
        self.qidx = torch.zeros((P1 * 2 * H,), dtype=i64, **g)
        self.pair = torch.full((B,), -1, dtype=i32, **g)
        self.side = torch.zeros((B,), dtype=i32, **g)
        self.cu_pairs = torch.arange(P1 + 1, dtype=i32, **g)
        # static head maps: packed head h' = kvh*2G + t*G + g  <->  request t, head kvh*G + g
        hp = np.arange(2 * H)
        self._t_of_hp = (hp % (2 * G)) // G
        self._h_of_hp = (hp // (2 * G)) * G + hp % G
        h = np.arange(H)
        self._hp_of_h = [(h // G) * 2 * G + t * G + h % G for t in (0, 1)]
        self.n_pairs = 0
        self.max_cols = 1
        self.set_pairs(np.zeros(0, np.int64), np.zeros(0, np.int64), 0)

    def set_pairs(self, a_rows: np.ndarray, b_rows: np.ndarray, num_rows: int) -> None:
        """Build the flat gather maps for pairs (a_rows[p], b_rows[p]); every other row in
        [0, max_reqs) points at the dummy slot. sh_bt/sh_len/own_* are filled by the caller."""
        H, P = self.H, self.max_pairs
        n = len(a_rows)
        self.n_pairs = n
        qidx = np.zeros((P + 1, 2 * H), np.int64)
        if n:
            src = np.where(self._t_of_hp[None, :] == 0, a_rows[:, None], b_rows[:, None])
            qidx[:n] = src * H + self._h_of_hp[None, :]
        self.qidx_cpu.numpy()[:] = qidx.reshape(-1)
        pair = self.pair_cpu.numpy(); side = self.side_cpu.numpy()
        pair[:] = -1; side[:] = 0
        if n:
            pair[a_rows] = np.arange(n); pair[b_rows] = np.arange(n)
            side[b_rows] = 1

    def commit(self, num_reqs_padded: int) -> None:
        """Copy staged CPU contents to the persistent GPU buffers (non-blocking)."""
        B, c = num_reqs_padded, self.max_cols
        self.own_bt[:B, :c].copy_(self.own_bt_cpu[:B, :c], non_blocking=True)
        self.own_len[:B].copy_(self.own_len_cpu[:B], non_blocking=True)
        self.sh_bt[:, :c].copy_(self.sh_bt_cpu[:, :c], non_blocking=True)
        self.sh_len.copy_(self.sh_len_cpu, non_blocking=True)
        self.qidx.copy_(self.qidx_cpu, non_blocking=True)
        self.pair[:B].copy_(self.pair_cpu[:B], non_blocking=True)
        self.side[:B].copy_(self.side_cpu[:B], non_blocking=True)


def pair_cascade_decode(
    out: torch.Tensor,          # [B, H, D]  (B = padded decode batch, one token per request)
    query: torch.Tensor,        # [B, H, D]
    key_cache: torch.Tensor,    # [num_blocks, block_size, Hk, D]
    value_cache: torch.Tensor,
    bufs: PairCascadeBuffers,
    cu_seqlens_q: torch.Tensor,  # [B+1] = arange (uniform decode)
    max_seqlen_k: int,
    softmax_scale: float,
    fa_version: int,
    num_splits: int = 0,
) -> torch.Tensor:
    B, H, D = query.shape
    P1 = bufs.max_pairs + 1

    # shared pass: one token per pair carrying both requests' heads (GQA ratio 2G)
    qs = query.reshape(B * H, D).index_select(0, bufs.qidx).view(P1, 2 * H, D)
    sh_out, sh_lse = flash_attn_varlen_func(
        q=qs, k=key_cache, v=value_cache,
        cu_seqlens_q=bufs.cu_pairs, max_seqlen_q=1,
        seqused_k=bufs.sh_len, max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=False,
        block_table=bufs.sh_bt, return_softmax_lse=True,
        fa_version=fa_version, num_splits=num_splits,
    )

    # own pass: normal decode over each row's non-shared blocks
    own_out, own_lse = flash_attn_varlen_func(
        q=query, k=key_cache, v=value_cache,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1,
        seqused_k=bufs.own_len[:B], max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale, causal=True,
        block_table=bufs.own_bt[:B], return_softmax_lse=True,
        fa_version=fa_version, num_splits=num_splits,
    )
    _merge_pair_shared_kernel[(B, H)](
        out, own_out, own_lse, sh_out, sh_lse, bufs.pair, bufs.side,
        B, P1, H=H, G=H // bufs.Hk, D=D,
    )
    return out
