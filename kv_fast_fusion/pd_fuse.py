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
        redirects: ``{owner_req_idx: [(slot, rep_req_idx, rep_slot, rep_flat), ...]}`` for the
            blocks that are redirected away (``labels[i] != i``) — i.e. the consumer should
            point ``(owner, slot)`` at the representative's physical block and free its own.
    """
    labels_l = labels.tolist()
    unique_flat = [i for i in range(len(labels_l)) if labels_l[i] == i]
    redirects: dict[int, list[tuple[int, int, int, int]]] = {}
    for i, rep in enumerate(labels_l):
        if rep == i:
            continue
        owner = flat_req_idx[i]
        redirects.setdefault(owner, []).append(
            (flat_slot[i], flat_req_idx[rep], flat_slot[rep], rep)
        )
    return unique_flat, redirects
