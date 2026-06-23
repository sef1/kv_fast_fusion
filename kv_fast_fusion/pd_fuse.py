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
