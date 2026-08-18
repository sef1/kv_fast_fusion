"""The CUDA and Ascend patches must install the same scheduler surgery.

Both `kv_fast_fusion.fast_fusion_pd_patch` and `kv_fast_fusion_ascend.fast_fusion_ascend_patch`
rebind methods on `vllm.v1.core.sched.Scheduler`. They are two hand-maintained lists, so they
drift — and when they drift the NPU inherits a stock method that BFF's layout breaks.

That is not hypothetical. `_update_requests_with_invalid_blocks` was replaced on CUDA (stock
unpacks a 1-tuple from `get_block_ids`, which BFF's 7 KV-cache groups blow up) but the same line
was never added to the Ascend patch. It stayed invisible until v2 started reporting failed KV
loads on the NPU, at which point the very first one killed EngineCore with
`ValueError: too many values to unpack (expected 1)`.

The test reads both files rather than applying them: `apply_*` mutates global vLLM classes and
imports `vllm_ascend`, neither of which belongs in a unit test. What it checks is exactly the
thing that went wrong — a name patched on one backend and forgotten on the other.
"""

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CUDA = _ROOT / "kv_fast_fusion" / "fast_fusion_pd_patch.py"
_ASCEND = _ROOT / "kv_fast_fusion_ascend" / "fast_fusion_ascend_patch.py"

# Names the Ascend patch installs differently ON PURPOSE, with the reason it is allowed to differ.
# Anything not listed here must be patched on both or neither.
_DELIBERATELY_DIFFERENT = {
    # vllm_ascend's RecomputeScheduler/AsyncRecomputeScheduler OVERRIDE update_from_output, so a
    # wholesale replacement would be shadowed by them (or would clobber their recompute logic).
    # The Ascend patch wraps every class that defines it instead — see
    # _wrap_scheduler_update_from_output.
    "update_from_output",
}


def _scheduler_attrs_assigned(path):
    """Names `X` in any `Scheduler.X = ...` statement in the file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "Scheduler"
            ):
                found.add(target.attr)
    return found


def test_the_two_patch_files_exist_and_patch_the_scheduler():
    """Guard the guard: if a refactor renames or moves these, the parity check below would pass
    vacuously on two empty sets."""
    for path in (_CUDA, _ASCEND):
        assert path.is_file(), f"{path} moved; this test's paths need updating"
    assert _scheduler_attrs_assigned(_CUDA), "found no Scheduler.X = ... in the CUDA patch"
    assert _scheduler_attrs_assigned(_ASCEND), "found no Scheduler.X = ... in the Ascend patch"


def test_ascend_installs_every_scheduler_patch_cuda_does():
    cuda = _scheduler_attrs_assigned(_CUDA)
    ascend = _scheduler_attrs_assigned(_ASCEND)
    missing = cuda - ascend - _DELIBERATELY_DIFFERENT

    assert not missing, (
        "these scheduler methods are patched on CUDA but not on Ascend: "
        f"{sorted(missing)}. Either install them in apply_fast_fusion_ascend_patch or add them "
        "to _DELIBERATELY_DIFFERENT with the reason. Leaving stock in place on the NPU means "
        "vLLM's single-KV-cache-group assumptions meet BFF's multi-group layout at runtime."
    )


@pytest.mark.parametrize("path", [_CUDA, _ASCEND], ids=["cuda", "ascend"])
def test_the_hma_invalid_blocks_replacement_is_installed_on_both(path):
    """The specific crash: without this, the first failed KV load takes EngineCore down.

    Pinned by name on both backends because it is the one whose absence is silent right up to the
    moment a transfer fails, which on a healthy run can be many minutes in.
    """
    assert "_update_requests_with_invalid_blocks" in _scheduler_attrs_assigned(path)
