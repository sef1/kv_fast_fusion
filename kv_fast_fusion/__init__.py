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
else:
    try:
        from kv_fast_fusion.fast_fusion_pd_patch import apply_fast_fusion_pd_patch
        apply_fast_fusion_pd_patch()
        print("Fast fusion P/D patch applied successfully.")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        apply_fast_fusion_pd_patch = None  # type: ignore
