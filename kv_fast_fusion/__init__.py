try:
    from kv_fast_fusion.kv_fast_fusion_patch import apply_fast_fusion_patch
    apply_fast_fusion_patch()
    print("Fast fusion patch applied successfully.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    apply_threshold_filter_patch = None  # type: ignore
