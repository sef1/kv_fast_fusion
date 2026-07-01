"""Ascend/NPU-specific BFF fast-fusion patches.

Kept separate from the CUDA/NCCL package (`kv_fast_fusion`) so the two runtimes never patch each
other's device-specific classes. `kv_fast_fusion/__init__.py` calls `apply_fast_fusion_ascend_patch`
when the vllm_ascend stack is present; otherwise the CUDA path (`apply_fast_fusion_pd_patch`) runs.
The device-agnostic patches (KV-cache group split, scheduler block-merge handler, block-pool
free/evict) are imported from `kv_fast_fusion` and reused unchanged.
"""

from kv_fast_fusion_ascend.fast_fusion_ascend_patch import apply_fast_fusion_ascend_patch

__all__ = ["apply_fast_fusion_ascend_patch"]
