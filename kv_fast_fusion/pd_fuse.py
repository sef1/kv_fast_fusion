"""Pure (NCCL-free) helpers for connector-level P/D fusion (plan ROUND 21).

The producer connector buffers a fusion group's G layers as they stream through
`save_kv_layer`, then calls `concat_cosine_cc_labels` to cluster the co-prefilling
requests' blocks by the **G-layer concatenation cosine** (same similarity as the cc/lsh/
tree/nr_tree paths — no cross-layer terms), and `build_group_redirect` to turn the labels
into a per-(request, slot) redirect that is shipped to D. Both are layout-agnostic and fully
unit-testable on CPU; the connector supplies the gathered tensors.
"""

from __future__ import annotations

import torch


def concat_cosine_cc_labels(
    k_per_layer: list[torch.Tensor],
    req_of_block: torch.Tensor,
    threshold: float,
    cc_iters: int = 32,
    tp_group=None,
) -> torch.Tensor:
    """Connected-components clustering of N blocks by the G-layer concatenation cosine.

    Args:
        k_per_layer: list of G tensors, each ``[N, D]`` — the per-layer K representation of
            the same N blocks (flattened block; any BFF_LSH_REPR can be pre-applied by the
            caller). The concatenation cosine is ``(Σ_g⟨v_g,u_g⟩)/(‖cat‖‖cat‖)`` computed as a
            sum of per-layer Gram matrices with one joint normalization — no cross-layer terms.
        req_of_block: ``[N]`` int, the owning request index per block. Edges are only allowed
            between blocks of DIFFERENT requests (never merge within one request).
        threshold: cosine threshold for an edge.
        cc_iters: label-propagation iteration cap.
        tp_group: optional ``torch.distributed`` process group (the tensor-parallel group). When
            given (TP>1), each rank holds only a HEAD SHARD of K, so its local ``cross``/``sq`` are
            partial; ``all_reduce(SUM)`` them over the group BEFORE normalizing so every rank
            computes the identical FULL-vector cosine (``⟨a,b⟩_full = Σ_ranks⟨shard,shard⟩``,
            ``‖a‖²_full = Σ_ranks‖shard‖²``) → identical labels → coherent global block table,
            numerically equal to the single-GPU decision. ``None`` (TP=1) is the original path.

    Returns:
        ``labels`` ``[N]`` where ``labels[i]`` is the representative (smallest member index)
        of i's component. A singleton has ``labels[i] == i``.
    """
    N = int(req_of_block.shape[0])
    dev = req_of_block.device
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=dev)

    cross = torch.zeros(N, N, device=dev, dtype=torch.float32)
    sq = torch.zeros(N, device=dev, dtype=torch.float32)
    for Kg in k_per_layer:
        Kg = Kg.float()
        cross += Kg @ Kg.T
        sq += (Kg * Kg).sum(1)
    if tp_group is not None:
        # Reconstruct the full-vector statistics from this rank's head-shard partials.
        import torch.distributed as dist
        dist.all_reduce(cross, op=dist.ReduceOp.SUM, group=tp_group)
        dist.all_reduce(sq, op=dist.ReduceOp.SUM, group=tp_group)
    d = sq.sqrt().clamp(min=1e-6)
    S = cross / (d[:, None] * d[None, :])

    A = (S > threshold) & (req_of_block[:, None] != req_of_block[None, :])
    A |= torch.eye(N, dtype=torch.bool, device=dev)

    labels = torch.arange(N, device=dev)
    big = torch.full((N,), N, device=dev, dtype=labels.dtype)
    for _ in range(cc_iters):
        nb_min = torch.where(A, labels[None, :], big).min(dim=1).values
        new = torch.minimum(labels, nb_min)
        if torch.equal(new, labels):
            break
        labels = new
    return labels


def concat_cosine_nr_tree_labels(
    k_per_layer: list[torch.Tensor],
    req_of_block: torch.Tensor,
    threshold: float,
    jump_iters: int = 32,
) -> torch.Tensor:
    """Non-recursive ('butterfly') tree clustering of N blocks by the G-layer concatenation
    cosine — a state-free port of the runner's ``_compress_group_nr_tree`` for the connector.

    Same inputs/return contract as :func:`concat_cosine_cc_labels` (so ``build_group_redirect``
    is unchanged). Instead of the dense N×N CC graph, a request's blocks travel together as one
    node; each level sorts active nodes by block-count ASCENDING (shorter request → left =
    representative), pairs adjacent nodes, and redirects each right block to its best-matching
    left block when cosine > ``threshold`` (so the longer request sheds blocks). Cross-request
    only by construction (paired nodes never share a request). 'full' precision: the caller
    passes the full per-layer block-K, concatenated + unit-normalized here.
    """
    N = int(req_of_block.shape[0])
    dev = req_of_block.device
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=dev)

    # Unit-normalized G-layer concatenation → cosine(i, j) = Xn[i] · Xn[j].
    Xn = torch.cat([Kg.float() for Kg in k_per_layer], dim=1)
    Xn = Xn / Xn.norm(dim=1, keepdim=True).clamp(min=1e-6)

    parent = torch.arange(N, device=dev)
    nodes = []
    for r in req_of_block.unique().tolist():
        idxs = (req_of_block == r).nonzero(as_tuple=True)[0]
        if idxs.numel():
            nodes.append(idxs)

    while len(nodes) > 1:
        nodes.sort(key=lambda t: t.numel())            # shorter → left (representative)
        nxt = []
        for k in range(0, len(nodes) - (len(nodes) & 1), 2):
            L, R = nodes[k], nodes[k + 1]              # |L| <= |R|
            sim = Xn[R] @ Xn[L].T                      # [|R|, |L|]
            best_val, best_l = sim.max(dim=1)
            match = best_val > threshold
            if bool(match.any()):
                parent[R[match]] = L[best_l[match]]    # union matched R → L
                nxt.append(torch.cat([L, R[~match]]))  # carry L + unmatched R
            else:
                nxt.append(torch.cat([L, R]))
        if len(nodes) & 1:
            nxt.append(nodes[-1])                      # odd node carries up unchanged
        nodes = nxt

    labels = parent
    for _ in range(jump_iters):                        # pointer-jump chains → roots
        nl = parent[labels]
        if torch.equal(nl, labels):
            break
        labels = nl
    return labels


def concat_cosine_cross_match(
    cur_per_layer: list[torch.Tensor],
    reg_vecs: torch.Tensor,
    reg_sq: torch.Tensor,
    threshold: float,
    tp_group=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match each of M current blocks against R registered rep blocks by the G-layer concatenation
    cosine (cross-batch P/D fusion, plan ROUND 58).

    Unlike the within-batch clustering, this compares two DIFFERENT sets: the current step's blocks
    vs a rolling registry of earlier requests' rep blocks. The registry is stored as the RAW
    concatenated vector (so TP head-shards stay reconstructable) plus its FULL squared norm.

    Args:
        cur_per_layer: list of G tensors ``[M, D]`` — current blocks' per-layer repr (this rank's
            head shard under TP). Concatenated here into ``[M, G·D]``.
        reg_vecs: ``[R, G·D]`` registry raw concat vectors (same rank's head shard).
        reg_sq:   ``[R]`` registry FULL (already all-reduced at registration) squared concat norm.
        threshold: cosine threshold for a match.
        tp_group: optional TP process group. Under TP each rank holds a head shard, so the local
            ``cur@reg.T`` cross terms and ``cur_sq`` are partial; ``all_reduce(SUM)`` them so every
            rank computes the identical FULL-vector cosine → identical match → coherent block table
            (mirrors :func:`concat_cosine_cc_labels`). ``reg_sq`` is already full.

    Returns:
        best_idx:  ``[M]`` long, the matched registry row per current block, or ``-1`` if none > thr.
        best_score:``[M]`` the matched cosine (0 where unmatched).
        cur_sq:    ``[M]`` the current blocks' FULL squared concat norm (reused for registration).
        cur_concat:``[M, G·D]`` the current concat vectors (this rank's shard; stored on register).
    """
    cur_concat = torch.cat([Kg.float() for Kg in cur_per_layer], dim=1)   # [M, G·D]
    M = cur_concat.shape[0]
    dev = cur_concat.device
    cur_sq = (cur_concat * cur_concat).sum(1)                              # [M] (partial under TP)
    if reg_vecs is None or reg_vecs.numel() == 0 or M == 0:
        if tp_group is not None:
            import torch.distributed as dist
            dist.all_reduce(cur_sq, op=dist.ReduceOp.SUM, group=tp_group)
        return (torch.full((M,), -1, dtype=torch.long, device=dev),
                torch.zeros(M, device=dev), cur_sq, cur_concat)

    cross = cur_concat @ reg_vecs.float().T                               # [M, R] (partial under TP)
    if tp_group is not None:
        import torch.distributed as dist
        dist.all_reduce(cross, op=dist.ReduceOp.SUM, group=tp_group)
        dist.all_reduce(cur_sq, op=dist.ReduceOp.SUM, group=tp_group)
    dcur = cur_sq.sqrt().clamp(min=1e-6)
    dreg = reg_sq.sqrt().clamp(min=1e-6)
    S = cross / (dcur[:, None] * dreg[None, :])                           # full-vector cosine
    best_score, best_idx = S.max(dim=1)
    best_idx = torch.where(best_score > threshold, best_idx,
                           torch.full_like(best_idx, -1))
    return best_idx, best_score, cur_sq, cur_concat


def build_group_redirect(
    labels: torch.Tensor,
    flat_req_idx: list[int],
    flat_slot: list[int],
) -> tuple[list[int], dict[int, list[tuple[int, int, int, int]]]]:
    """Turn flat-block ``labels`` into (unique-representative blocks, per-owner redirects).

    Args:
        labels: ``[N]`` representative flat index per block (from concat_cosine_cc_labels).
        flat_req_idx: length-N, the owner request's batch index for each flat block.
        flat_slot: length-N, the block slot within that request's (per-group) block table.

    Returns:
        unique_flat: the flat indices that are representatives of a cluster of size>1 OR
            singletons — i.e. the blocks whose KV must actually be sent (everything that is
            not redirected away). Concretely: indices ``i`` with ``labels[i] == i``.
        redirects: ``{owner_req_idx: [(slot, rep_req_idx, rep_slot, rep_flat, owner_flat), ...]}``
            for the blocks that are redirected away (``labels[i] != i``) — the consumer points
            ``(owner, slot)`` at the representative's physical block and frees its own. ``rep_flat``
            and ``owner_flat`` are the flat indices of the rep / owner blocks (used by the connector
            in `ratio` mode to look up per-block ‖owner‖/‖rep‖ K/V norms).
    """
    labels_l = labels.tolist()
    unique_flat = [i for i in range(len(labels_l)) if labels_l[i] == i]
    redirects: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for i, rep in enumerate(labels_l):
        if rep == i:
            continue
        owner = flat_req_idx[i]
        redirects.setdefault(owner, []).append(
            (flat_slot[i], flat_req_idx[rep], flat_slot[rep], rep, i)
        )
    return unique_flat, redirects
