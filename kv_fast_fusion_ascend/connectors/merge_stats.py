"""The worker→scheduler carrier for BFF's D-side block-merge map, on Ascend.

At TP=1 the worker and the scheduler share a process, so the connector publishes the map by
assigning ``runner._updated_block_tables`` and the patched ``update_from_output`` reads it straight
off ``_ACTIVE_RUNNER``. At TP>1 they are **separate processes** and that channel is a write into an
object the scheduler cannot see — so the aliases apply on the device, the block tables are
rewritten, ``aliases_applied`` counts up, and not one block is ever freed. Every gain BFF exists to
produce is on the scheduler's side of that boundary.

``KVConnectorOutput.kv_connector_stats`` is the only serializable connector→scheduler slot, so the
map rides it: ``data["bff_merges"] = {req_id: {group_idx: [block_ids]}}``.

The producer's piggybacked signatures cross the same boundary the same way:
``data["bff_sigs"] = {ext_id: {tp_rank: {group_idx: payload}}}``. Unlike the merge map these are
per-rank (each rank signs its own head shard), so :meth:`aggregate` unions them across ranks.

This is a straight port of the GPU connectors' ``BFFMergeStats``
(``kv_fast_fusion/connectors/mooncake_connector_ff.py``), lifted into its own module because Ascend
has **two** v1 transports — the pull connector and the layerwise one — and their v2 subclasses both
need it. A third copy of the class would be a third place for the aggregate semantics to drift.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats


class BFFMergeStats(KVConnectorStats):
    """Carries the D-side block-merge map worker→scheduler under TP>1.

    Every TP rank produces the identical (all-reduced) map, so the cross-rank :meth:`aggregate`
    keeps the accumulator's copy rather than concatenating — double-applying a merge would corrupt
    the block ref counts, which is a worse failure than losing one."""

    def reset(self):
        self.data = {}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        other_data = getattr(other, "data", None) or {}
        # Ranks are coherent → identical maps; keep the accumulator's, never concatenate.
        if not self.data.get("bff_merges") and other_data.get("bff_merges"):
            self.data["bff_merges"] = other_data["bff_merges"]
        # Signatures differ per rank: union them, request by request.
        if other_data.get("bff_sigs"):
            mine = self.data.setdefault("bff_sigs", {})
            for ext_id, by_rank in other_data["bff_sigs"].items():
                mine.setdefault(ext_id, {}).update(by_rank)
        return self

    def reduce(self) -> dict:
        return {"bff_merge_reqs": len(self.data.get("bff_merges", {}) or {}),
                "bff_sig_reqs": len(self.data.get("bff_sigs", {}) or {})}

    def is_empty(self) -> bool:
        return not self.data.get("bff_merges") and not self.data.get("bff_sigs")


def tp_group_if_sharded():
    """The tensor-parallel process group when TP>1, else ``None``.

    The carrier is only needed when the scheduler is out of process, which is exactly TP>1. At TP=1
    returning None keeps the in-process channel as the single source and avoids shipping the same
    map twice — the scheduler's apply is consume-once, and two arrivals would be two applies."""
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return None
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        return tp.device_group if tp.world_size > 1 else None
    except Exception:       # noqa: BLE001 - absence of a TP group is a normal TP=1 answer
        return None
