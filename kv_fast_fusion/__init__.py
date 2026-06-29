# =============================================================================
# Fast Fusion patch selection — toggle ONE of the two blocks below.
# =============================================================================

# --- Single-instance BFF (default) -------------------------------------------
# Post-forward LSH/cc/tree fusion + per-block norm kernel, one server.
# try:
#     from kv_fast_fusion.kv_fast_fusion_graph_patch import apply_fast_fusion_graph_patch
#     apply_fast_fusion_graph_patch()
#     print("Fast fusion graph patch applied successfully.")
# except ModuleNotFoundError:  # pragma: no cover - optional dependency
#     apply_fast_fusion_graph_patch = None  # type: ignore

# --- P/D disaggregated (toggle: comment the block above, uncomment below) -----
# Lean raw-mode patch + connector-level fusion (BFF_PD_FUSE=1). Registers
# P2pNcclConnectorFF and the consume-time recv-buffer free itself.
try:
    from kv_fast_fusion.fast_fusion_pd_patch import apply_fast_fusion_pd_patch
    apply_fast_fusion_pd_patch()
    print("Fast fusion P/D patch applied successfully.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    apply_fast_fusion_pd_patch = None  # type: ignore
