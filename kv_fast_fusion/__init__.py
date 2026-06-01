try:
    from kv_fast_fusion.kv_fast_fusion_patch import apply_fast_fusion_patch
    apply_fast_fusion_patch()
    # from kv_fast_fusion.kv_fast_fusion_graph_patch import apply_fast_fusion_graph_patch
    # apply_fast_fusion_graph_patch()
    print("Fast fusion graph patch applied successfully.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    apply_fast_fusion_graph_patch = None  # type: ignore
