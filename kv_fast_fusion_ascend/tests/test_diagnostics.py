"""The diagnostics mixins run on CPU now, so these execute them instead of reading their source."""

import types

from kv_fast_fusion_ascend import diagnostics as d
from kv_fast_fusion_ascend.connectors import mooncake_connector_ff_v2 as v2


def test_the_connector_re_exports_the_same_objects():
    for name in ("VerifySampler", "compare_signature_rows", "hot_block_collisions", "write_slot",
                 "slot_trace_fault", "cross_match", "classify_mismatch", "_sync_device"):
        assert getattr(v2, name) is getattr(d, name)


def test_every_diagnostic_is_off_by_default():
    assert d.VERIFY_TRANSFER == 0 and d.XFER_AUDIT == 0
    assert d.TRACE_SLOTS == 0 and d.AUDIT_HOT_BLOCKS is False


class _Recv(d.TransferDiagnostics):
    local_sigs = None
    local_sigs_multi = None
    dedup_engine = None
    last_descriptors = None
    region_map = None


def test_recompute_prefers_the_cross_group_entry_point():
    r = _Recv()
    assert r._recompute_local({}) == {}
    r.local_sigs_multi = lambda plan: ({1: ["multi"]}, {})
    assert r._recompute_local({1: ([7], [1])}) == {1: "multi"}
    r.local_sigs_multi = None
    r.local_sigs = lambda gi, ids, lengths: ["single"]
    assert r._recompute_local({1: ([7], [1])}) == {1: "single"}


def test_the_audit_and_the_verification_do_nothing_when_off():
    r = _Recv()
    # dedup_engine is None and XFER_AUDIT is 0: either would raise if these got past their gates.
    r._audit_self_consistency({})
    assert d.TransferDiagnostics._audit_seen == 0


def test_the_audit_buckets_are_kept_per_instance():
    r = _Recv()
    assert r._audit_buckets is r._audit_buckets
    assert _Recv()._audit_buckets is not r._audit_buckets


def test_an_unknown_block_buckets_by_group():
    assert _Recv()._bucket_for(3, 9) == "group3"


class _Conn(d.TraceDiagnostics):
    def __init__(self):
        self._traced = {}
        self._hot_collisions = 0

    def _block_size(self):
        return 128


def _runner(**requests):
    return types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_id_to_index={r: i for i, r in enumerate(requests)}),
        requests={r: types.SimpleNamespace(num_computed_tokens=n, block_ids=b)
                  for r, (n, b) in requests.items()})


def test_the_hot_block_audit_counts_a_shared_write_frontier(monkeypatch):
    monkeypatch.setattr(d, "AUDIT_HOT_BLOCKS", True)
    monkeypatch.setattr(d.TraceDiagnostics, "_logged_hot_collision", False)
    c = _Conn()
    c._audit_hot_blocks(_runner(a=(0, [[5]]), b=(0, [[6]])))
    assert c._hot_collisions == 0
    c._audit_hot_blocks(_runner(a=(0, [[5]]), b=(0, [[5]])))
    assert c._hot_collisions == 1
    assert d.TraceDiagnostics._logged_hot_collision is True


def test_the_hot_block_audit_is_silent_when_off():
    c = _Conn()
    c._audit_hot_blocks(_runner(a=(0, [[5]]), b=(0, [[5]])))
    assert c._hot_collisions == 0


def test_the_slot_trace_follows_consecutive_tokens(monkeypatch):
    monkeypatch.setattr(d, "TRACE_SLOTS", 1)
    # No block table on the fake runner: the mirror check must give up quietly, not raise.
    monkeypatch.setattr(d.TraceDiagnostics, "_logged_mirror_drift", False)
    c = _Conn()
    c._trace_slots(_runner(a=(300, [[0], [1, 2, 7]])))
    assert c._traced["a"] == (300, 7 * 128 + 44)
    c._trace_slots(_runner(a=(301, [[0], [1, 2, 7]])))
    assert c._traced["a"] == (301, 7 * 128 + 45)
