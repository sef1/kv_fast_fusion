"""The live Ascend path must not need the GPU connector package.

``pd_dedup_v2.threshold_for`` used to read ``_THRESHOLD_G`` out of the GPU Mooncake connector. That
imported ``kv_fast_fusion.connectors``, whose ``__init__`` imports the NCCL connector, so a dedup
threshold lookup depended on vLLM's p2p module being present.
"""

import subprocess
import sys

from kv_fast_fusion import constants
from kv_fast_fusion.connector_selection import _selected_connector_names


def test_threshold_for_does_not_import_the_gpu_connectors():
    code = ("import sys\n"
            "from kv_fast_fusion import pd_dedup_v2\n"
            "before = 'kv_fast_fusion.connectors' in sys.modules\n"
            "pd_dedup_v2.threshold_for(1)\n"
            "after = 'kv_fast_fusion.connectors' in sys.modules\n"
            "assert before or not after, 'threshold_for imported the GPU connector package'\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_threshold_for_reads_the_per_group_override(monkeypatch):
    from kv_fast_fusion import pd_dedup_v2

    monkeypatch.setattr(constants, "THRESHOLD_G", {2: 0.9})
    monkeypatch.setattr(constants, "THRESHOLD", 0.75)
    assert pd_dedup_v2.threshold_for(2) == 0.9
    assert pd_dedup_v2.threshold_for(1) == 0.75


def test_parse_thresholds_keeps_good_entries_and_drops_bad_ones():
    assert constants.parse_thresholds(" 1:0.97 , ") == {1: 0.97}
    assert constants.parse_thresholds(None) == {} and constants.parse_thresholds("") == {}
    assert constants.parse_thresholds("junk,3:x,4:0.5") == {4: 0.5}


def test_selected_connector_names_reads_argv(monkeypatch):
    cfg = '{"kv_connector":"MooncakeConnectorFFv2","kv_role":"kv_consumer"}'
    monkeypatch.setattr(sys, "argv", ["vllm", "--kv-transfer-config", cfg])
    assert _selected_connector_names() == {"MooncakeConnectorFFv2"}
    monkeypatch.setattr(sys, "argv", ["vllm"])
    assert _selected_connector_names() == set()
