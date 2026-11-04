"""Request-id helpers, in their own module so the patch and its checks can share them without a cycle."""


def _ext_of(rid: str) -> str:
    """External (cross-P/D stable) id of a request id. Uses vllm_ascend's canonical helper on the
    NPU stack; falls back to stripping the per-server random ``-<8hex>`` suffix (the same transform)
    off-NPU so the promotion apply is unit-testable."""
    try:
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
            get_external_request_id,
        )
        return get_external_request_id(rid)
    except ImportError:
        return rid.rsplit("-", 1)[0]
