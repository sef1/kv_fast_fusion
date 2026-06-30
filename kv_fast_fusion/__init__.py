try:
    from kv_fast_fusion.fast_fusion_pd_patch import apply_fast_fusion_pd_patch
    apply_fast_fusion_pd_patch()
    print("Fast fusion P/D patch applied successfully.")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    apply_fast_fusion_pd_patch = None  # type: ignore
