from kv_fast_fusion.connectors.p2p_nccl_connector_ff import P2pNcclConnectorFF

__all__ = ["P2pNcclConnectorFF"]

# The Mooncake variant needs the `mooncake` transfer-engine package; keep it optional so the
# NCCL connector still imports on a box without it. (vLLM's factory registers it by module
# path anyway, so this re-export is a convenience, not the wiring.)
try:
    from kv_fast_fusion.connectors.mooncake_connector_ff import MooncakeConnectorFF

    __all__.append("MooncakeConnectorFF")
except ImportError:  # pragma: no cover - optional dependency
    pass
