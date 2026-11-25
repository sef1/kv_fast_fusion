"""BFF fast-fusion activation. Branches on the runtime stack:

  * Ascend/NPU present  → apply the Ascend patch (`kv_fast_fusion_ascend`): NPUModelRunner init,
    scheduler `update_from_output` wrapper, reused device-agnostic patches, and registration of
    `MooncakeLayerwiseConnectorFF`. The CUDA/NCCL patch is NOT applied (it would double-patch the
    runner init via the `NPUModelRunner(GPUModelRunner)` base and clobber the recompute scheduler).
  * Otherwise (CUDA)    → apply the existing P/D patch (`fast_fusion_pd_patch`), unchanged.

Ascend detection: try importing `NPUModelRunner`. Guarded so a non-Ascend box falls through cleanly.
"""


def _ascend_stack_available() -> bool:
    try:
        import vllm_ascend.worker.model_runner_v1  # noqa: F401
        return True
    except Exception:
        return False


if _ascend_stack_available():
    try:
        from kv_fast_fusion_ascend import apply_fast_fusion_ascend_patch
        apply_fast_fusion_ascend_patch()
        print("Fast fusion Ascend patch applied successfully.")
    except Exception as e:  # pragma: no cover - defensive
        print(f"Fast fusion Ascend patch failed: {e}")
        apply_fast_fusion_ascend_patch = None  # type: ignore
elif __import__("os").environ.get("BFF_SINGLE", "0") == "1":
    # Single-instance (no P/D, no connector) BFF: the legacy graph runner fuses co-finishing
    # prefills post-forward and frees duplicates. BFF_SINGLE_FUSE=0 keeps the layout, skips merges.
    from kv_fast_fusion.legacy.kv_fast_fusion_graph_patch import apply_fast_fusion_graph_patch
    apply_fast_fusion_graph_patch()
    from kv_fast_fusion.bff_online_fuse import ONLINE_ENABLED as _online, install_scheduler_hooks
    if _online:
        install_scheduler_hooks()
    print("Fast fusion single-instance (graph) patch applied successfully."
          + (" [online fusion]" if _online else ""))
else:
    try:
        from kv_fast_fusion.fast_fusion_pd_patch import apply_fast_fusion_pd_patch
        apply_fast_fusion_pd_patch()
        print("Fast fusion P/D patch applied successfully.")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        apply_fast_fusion_pd_patch = None  # type: ignore
