"""Unit tests for the GPU Mooncake BFF connector (`kv_fast_fusion.connectors.mooncake_connector_ff`).

Everything here runs on CPU with fake runners / fake scheduler outputs — no GPU, no NPU, no
Transfer Engine, no P/D topology. The point is to pin the parts that are easy to get silently
wrong and expensive to debug on a live run:

  * the producer fusion actually clusters duplicate blocks and emits redirect rows keyed by the
    P/D-stable transfer id,
  * the consumer resolve/apply only rewrites what it can prove, and never frees a block table it
    did not write,
  * `_build_transfer_params` pairs blocks PER KV-cache group (the single-group collapse is exactly
    the bug this connector exists to fix),
  * the wire structs round-trip per-group block ids and the redirect side-payload.

Run:  .venv/bin/python -m pytest kv_fast_fusion/tests/test_mooncake_connector_ff.py -q
"""

import asyncio
import types

import pytest
import torch

from kv_fast_fusion.connectors import mooncake_connector_ff as mc


# =====================================================================================
# helpers
# =====================================================================================
def _blocks(*rows):
    """Per-group block-id table: _blocks([1,2], [7,8]) -> [[1,2],[7,8]] (group 0, group 1)."""
    return [list(r) for r in rows]


class _FakeBlockTable:
    def __init__(self, nreq, ncol):
        self.np = torch.zeros((nreq, ncol), dtype=torch.int32).numpy()
        self.gpu = torch.zeros((nreq, ncol), dtype=torch.int32)


class _FakeGroupTable:
    def __init__(self, nreq, ncol):
        self.block_table = _FakeBlockTable(nreq, ncol)
        self.num_blocks_per_row = [ncol] * nreq


class _FakeRunner:
    """The subset of GPUModelRunner state the apply path touches."""

    def __init__(self, req_ids, ngroups=2, ncol=4):
        self.input_batch = types.SimpleNamespace(
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            block_table=types.SimpleNamespace(
                block_tables=[_FakeGroupTable(len(req_ids), ncol) for _ in range(ngroups)]),
        )
        self.requests = {}
        self._updated_block_tables = None


# =====================================================================================
# _tid_hash
# =====================================================================================
def test_tid_hash_is_stable_positive_and_distinct():
    a, b = mc._tid_hash("xfer-abc"), mc._tid_hash("xfer-abc")
    assert a == b, "must be reproducible across calls (and across the P and D processes)"
    assert 0 < a < 2 ** 63
    assert mc._tid_hash("xfer-abc") != mc._tid_hash("xfer-abd")


# =====================================================================================
# resolve_redirect_rows
# =====================================================================================
def test_resolve_points_owner_at_rep_block():
    rid2blocks = {"own": _blocks([9, 9], [100, 101]), "rep": _blocks([9, 9], [200, 201])}
    hash2rid = {mc._tid_hash("t-rep"): "rep"}
    new, applied, unresolved = mc.resolve_redirect_rows(
        rid2blocks, hash2rid, "own", 1, [(1, mc._tid_hash("t-rep"), 0)])
    assert new == [100, 200] and applied == 1 and unresolved == 0


def test_resolve_unresolved_when_rep_not_resident():
    rid2blocks = {"own": _blocks([9], [100, 101])}
    new, applied, unresolved = mc.resolve_redirect_rows(
        rid2blocks, {}, "own", 1, [(1, mc._tid_hash("gone"), 0)])
    assert new is None and applied == 0 and unresolved == 1


def test_resolve_owner_missing_counts_all_rows():
    new, applied, unresolved = mc.resolve_redirect_rows({}, {}, "own", 1, [(0, 1, 0), (1, 2, 0)])
    assert new is None and applied == 0 and unresolved == 2


def test_resolve_rejects_out_of_range_slots_and_sentinels():
    rid2blocks = {"own": _blocks([9], [100, 101]), "rep": _blocks([9], [200])}
    h = mc._tid_hash("t-rep")
    hash2rid = {h: "rep"}
    rows = [(-1, -1, -1),          # sentinel row → skipped silently
            (5, h, 0),             # owner slot out of range
            (0, h, 7)]             # rep slot out of range
    new, applied, unresolved = mc.resolve_redirect_rows(rid2blocks, hash2rid, "own", 1, rows)
    assert new is None and applied == 0 and unresolved == 2


# =====================================================================================
# write_runner_block_table
# =====================================================================================
def test_write_block_table_updates_np_gpu_and_request_state():
    runner = _FakeRunner(["a"], ngroups=2, ncol=3)
    runner.requests["a"] = types.SimpleNamespace(block_ids=[[0, 0, 0], [10, 11, 12]])
    assert mc.write_runner_block_table(runner, "a", 1, [10, 99, 12]) is True
    bt = runner.input_batch.block_table.block_tables[1].block_table
    assert list(bt.np[0]) == [10, 99, 12]
    assert bt.gpu[0].tolist() == [10, 99, 12]
    assert runner.requests["a"].block_ids[1] == [10, 99, 12]


def test_write_block_table_returns_false_when_request_not_batched():
    """The free must be coupled to this: rewriting nothing but freeing anyway would leave the
    request pointing at freed-then-reallocated KV."""
    runner = _FakeRunner(["a"])
    assert mc.write_runner_block_table(runner, "not-batched", 1, [1, 2]) is False


# =====================================================================================
# FFProducerFusion
# =====================================================================================
def _kv_layer(vectors):
    """Fake FlashAttention KV cache [2, nblocks, block_sz, heads, dim] holding `vectors` as K."""
    n = len(vectors)
    k = torch.zeros(n + 1, 1, 1, len(vectors[0]))          # +1 → block id 0 stays the null block
    for i, v in enumerate(vectors):
        k[i + 1, 0, 0] = torch.tensor(v, dtype=torch.float32)
    return torch.stack([k, k.clone()])


def test_fusion_emits_redirect_for_duplicate_blocks_across_requests():
    """Two requests whose fusion-group blocks are identical → the second redirects to the first."""
    fusion = mc.FFProducerFusion({1: {"l0", "l1"}})
    dup = [1.0, 0.0, 0.0]
    kv = _kv_layer([dup, dup])                             # block 1 (req A), block 2 (req B)
    reqs = [("t-A", _blocks([], [1])), ("t-B", _blocks([], [2]))]

    assert fusion.on_layer(1, "l0", kv, reqs, step_key=1, is_mla=False) is None, "group incomplete"
    rows = fusion.on_layer(1, "l1", kv, reqs, step_key=1, is_mla=False)

    assert rows, "identical blocks must produce a redirect"
    (owner_tid, owner_rows), = rows.items()
    assert owner_tid == "t-B", "the later request redirects to the earlier representative"
    (owner_slot, rep_hash, rep_slot), = owner_rows
    assert (owner_slot, rep_hash, rep_slot) == (0, mc._tid_hash("t-A"), 0)
    assert fusion.redir_total[1] == 1 and fusion.blk_total[1] == 2


def test_fusion_emits_nothing_for_dissimilar_blocks():
    fusion = mc.FFProducerFusion({1: {"l0"}})
    kv = _kv_layer([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])     # orthogonal → cosine 0
    reqs = [("t-A", _blocks([], [1])), ("t-B", _blocks([], [2]))]
    assert fusion.on_layer(1, "l0", kv, reqs, step_key=1, is_mla=False) == {}
    assert fusion.redir_total[1] == 0


def test_fusion_skips_null_block_zero():
    """Block id 0 is the null block — it must never enter the flat set (it holds no request KV)."""
    fusion = mc.FFProducerFusion({1: {"l0"}})
    kv = _kv_layer([[1.0, 0.0]])
    reqs = [("t-A", _blocks([], [0, 1]))]
    fusion.on_layer(1, "l0", kv, reqs, step_key=1, is_mla=False)
    assert fusion.blk_total[1] == 1, "only the real block counts"


def test_fusion_resets_partial_group_on_new_step():
    """A group that never completed must not leak its buffer into the next step's clustering."""
    fusion = mc.FFProducerFusion({1: {"l0", "l1"}})
    kv = _kv_layer([[1.0, 0.0]])
    reqs = [("t-A", _blocks([], [1]))]
    assert fusion.on_layer(1, "l0", kv, reqs, step_key=1, is_mla=False) is None
    assert fusion.on_layer(1, "l0", kv, reqs, step_key=2, is_mla=False) is None
    assert len(fusion._buf[1]["k_layers"]) == 1, "step 2 started from a clean buffer"


def test_fusion_unknown_group_is_a_noop():
    fusion = mc.FFProducerFusion({1: {"l0"}})
    kv = _kv_layer([[1.0, 0.0]])
    assert fusion.on_layer(7, "lx", kv, [("t", _blocks([], [1]))], 1, False) is None


def test_fusion_stats_schema_matches_the_nccl_connector():
    """The launch script's bff_stats_*.json merge step reads these exact keys."""
    fusion = mc.FFProducerFusion({1: {"l0"}})
    dup = [1.0, 0.0]
    fusion.on_layer(1, "l0", _kv_layer([dup, dup]),
                    [("t-A", _blocks([], [1])), ("t-B", _blocks([], [2]))], 1, False)
    s = fusion.stats_dict()
    for key in ("pid", "is_producer", "steps", "overhead_avg_group_dedup_ms", "total_blocks",
                "freed", "compression_avg_factor", "compression_per_group",
                "encoded_batch_size", "cross_batch_redirects", "within_batch_redirects",
                "registry_blocks"):
        assert key in s, key
    assert s["total_blocks"] == 2 and s["freed"] == 1
    assert s["compression_avg_factor"] == pytest.approx(2.0), "half the blocks → 2x smaller"


def test_fusion_dump_stats_writes_atomically(tmp_path):
    import json
    fusion = mc.FFProducerFusion({1: {"l0"}})
    fusion.dump_stats(str(tmp_path))
    files = list(tmp_path.glob("bff_stats_*.json"))
    assert len(files) == 1 and not list(tmp_path.glob("*.tmp"))
    assert json.loads(files[0].read_text())["is_producer"] is True


def test_fusion_cross_batch_registry_matches_an_earlier_step(monkeypatch):
    """With BFF_PD_ENCODED_BATCH_SIZE>0 a block may redirect to a rep from a PREVIOUS step."""
    monkeypatch.setattr(mc, "_PD_ENCODED_BATCH", 8)
    fusion = mc.FFProducerFusion({1: {"l0"}})
    vec = [1.0, 0.0, 0.0]

    # Each step gets its own fake cache tensor holding one block (id 1); the registry has to carry
    # A's representation across the step boundary for B to match it.
    step1 = fusion.on_layer(1, "l0", _kv_layer([vec]), [("t-A", _blocks([], [1]))], 1, False)
    assert step1 == {}, "nothing to redirect to yet; A is registered as a rep"
    step2 = fusion.on_layer(1, "l0", _kv_layer([vec]), [("t-B", _blocks([], [1]))], 2, False)

    assert step2 == {"t-B": [[0, mc._tid_hash("t-A"), 0]]}
    assert fusion.cross_redir_total == 1 and fusion.within_redir_total == 0


def test_fusion_registry_evicts_beyond_the_window(monkeypatch):
    monkeypatch.setattr(mc, "_PD_ENCODED_BATCH", 2)
    fusion = mc.FFProducerFusion({1: {"l0"}})
    for step in range(5):
        v = [0.0] * 5
        v[step] = 1.0                                       # all mutually orthogonal → all reps
        fusion.on_layer(1, "l0", _kv_layer([v]), [(f"t-{step}", _blocks([], [1]))], step, False)
    assert fusion.registry_size(1) == 2, "window of 2 requests"


# =====================================================================================
# wire structs
# =====================================================================================
requires_mooncake = pytest.mark.skipif(
    not mc._MOONCAKE_AVAILABLE, reason="mooncake transfer engine not installed")


@requires_mooncake
def test_xfer_metadata_round_trips_per_group_block_ids():
    """The decode encodes the STOCK struct with per-group lists in it (msgspec serializes the
    object it is given); the prefill decodes with the FF struct. That asymmetry is what lets us
    avoid overriding the decode side's send path at all — so pin it."""
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeXferMetadata,
    )
    stock = MooncakeXferMetadata(
        remote_hostname="h", remote_port=1, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"d0": ("t0", [[1, 2], [7, 8]])}, kv_caches_base_addr=[10, 20])
    out = msgspec.msgpack.Decoder(mc.MooncakeXferMetadataFF).decode(
        msgspec.msgpack.Encoder().encode(stock))
    assert out.req_blocks["d0"] == ("t0", [[1, 2], [7, 8]])


@requires_mooncake
def test_xfer_response_round_trips_redirects_and_tolerates_absence():
    import msgspec
    enc, dec = msgspec.msgpack.Encoder(), msgspec.msgpack.Decoder(mc.MooncakeXferResponseFF)

    with_rows = mc.MooncakeXferResponseFF(
        status=mc.MooncakeXferResponseStatus.FINISH, ok_reqs=["d0"],
        ff_redirects={"d0": {1: [[0, 12345, 2]]}})
    assert dec.decode(enc.encode(with_rows)).ff_redirects == {"d0": {1: [[0, 12345, 2]]}}

    plain = mc.MooncakeXferResponse(status=mc.MooncakeXferResponseStatus.FINISH, ok_reqs=["d0"])
    assert dec.decode(enc.encode(plain)).ff_redirects is None, "a stock ACK must still decode"


@requires_mooncake
def test_response_encoder_attaches_pending_rows_once():
    """The encoder shim is the whole delivery mechanism — it must attach rows for the acked
    requests, translate transfer id → decode request id, and never ship the same map twice."""
    worker = types.SimpleNamespace(
        _ff_rows={"t0": {1: [[0, 5, 1]]}},
        _ff_dreq2tid={"d0": "t0"},
    )
    worker.pop_ff_rows = mc.MooncakeConnectorWorkerFF.pop_ff_rows.__get__(worker)
    encoder = mc._FFResponseEncoder(worker)

    import msgspec
    dec = msgspec.msgpack.Decoder(mc.MooncakeXferResponseFF)
    resp = mc.MooncakeXferResponse(status=mc.MooncakeXferResponseStatus.FINISH, ok_reqs=["d0"])

    first = dec.decode(encoder.encode(resp))
    assert first.ff_redirects == {"d0": {1: [[0, 5, 1]]}}
    second = dec.decode(encoder.encode(resp))
    assert second.ff_redirects is None, "popped — a second ACK must not re-ship the map"


# =====================================================================================
# failed-pull recovery — the 2026-08-13 hang
# =====================================================================================
def _recv_worker():
    """A real MooncakeConnectorWorkerFF instance with only the state process_pulling_result
    touches. Built via __new__ because __init__ would stand up a Transfer Engine — but it must be a
    genuine instance, since the method's zero-arg super() call needs one."""
    import threading
    w = mc.MooncakeConnectorWorkerFF.__new__(mc.MooncakeConnectorWorkerFF)
    w._ff_lock = threading.Lock()
    w._ff_recv_rows = {}
    w._ff_failed_blocks = set()
    w.finished_recving_reqs = set()
    # State the patched base needs when super() is NOT stubbed out.
    w._failed_load_block_ids = set()
    return w


def _pull_meta(d_req_id, blocks):
    return types.SimpleNamespace(d_req_id=d_req_id, transfer_id=f"t-{d_req_id}",
                                 local_block_ids=blocks, pull_tasks_count=1)


@requires_mooncake
def test_failed_pull_releases_the_request_and_reports_bad_blocks(monkeypatch):
    """A pull that fails must NOT leave the request in WAITING_FOR_REMOTE_KVS forever — that is
    the hang: 81 failed transfers wedged the engine at Running: 0 / Waiting: 86."""
    w = _recv_worker()
    monkeypatch.setattr(mc.MooncakeConnectorWorker, "process_pulling_result",
                        lambda self, response, pull_metas: None)
    metas = {"d0": _pull_meta("d0", [[1, 2], [7, 8]])}
    resp = mc.MooncakeXferResponseFF(
        status=mc.MooncakeXferResponseStatus.FINISH, err_reqs=["d0"],
        err_msg="Mooncake transfer engine returned -1")

    w.process_pulling_result(resp, metas)

    assert w.finished_recving_reqs == {"d0"}, "released, so the scheduler can act on it"
    assert w.take_block_ids_with_load_errors() == {1, 2, 7, 8}, "all groups' blocks are invalid"
    assert w.take_block_ids_with_load_errors() == set(), "drained exactly once"


@requires_mooncake
def test_failed_pull_runs_through_the_real_base_without_stubbing():
    """The tests above stub out `super().process_pulling_result` — which is precisely why they
    missed the 2026-08-13 crash. The patched STOCK method does `set.update(local_block_ids)`,
    correct for its own flat list[int] but not for the per-group list[list[int]] the FF subclass
    hands it: "unhashable type: 'list'", raised inside the recv coroutine and swallowed, aborting
    the rest of that pull batch. Exercise the genuine two-layer path here.

    Skipped when the vLLM patch is not applied (patch/vllm/apply_mooncake_load_failure.sh)."""
    if not hasattr(mc.MooncakeConnectorWorker, "take_block_ids_with_load_errors"):
        pytest.skip("vLLM mooncake load-failure patch not applied")
    w = _recv_worker()
    metas = {"d0": _pull_meta("d0", [[1, 2], [7, 8]])}
    resp = mc.MooncakeXferResponseFF(
        status=mc.MooncakeXferResponseStatus.FINISH, err_reqs=["d0"],
        err_msg="Mooncake transfer engine returned -1")

    w.process_pulling_result(resp, metas)          # must not raise

    assert w.finished_recving_reqs == {"d0"}
    assert w.take_block_ids_with_load_errors() == {1, 2, 7, 8}, (
        "both layers report the same flattened blocks; double handling is idempotent")


@requires_mooncake
def test_failed_pull_discards_any_redirect_map_for_that_request(monkeypatch):
    """The KV never landed, so applying its redirects would point other requests at garbage."""
    w = _recv_worker()
    monkeypatch.setattr(mc.MooncakeConnectorWorker, "process_pulling_result",
                        lambda self, response, pull_metas: None)
    w._ff_recv_rows["d0"] = {1: [[0, 5, 1]]}
    resp = mc.MooncakeXferResponseFF(
        status=mc.MooncakeXferResponseStatus.FINISH, err_reqs=["d0"],
        ff_redirects={"d0": {1: [[0, 5, 1]]}})
    w.process_pulling_result(resp, {"d0": _pull_meta("d0", [[1]])})
    assert "d0" not in w._ff_recv_rows


@requires_mooncake
def test_successful_pull_reports_no_load_errors(monkeypatch):
    w = _recv_worker()
    monkeypatch.setattr(mc.MooncakeConnectorWorker, "process_pulling_result",
                        lambda self, response, pull_metas: None)
    resp = mc.MooncakeXferResponseFF(
        status=mc.MooncakeXferResponseStatus.FINISH, ok_reqs=["d0"])
    w.process_pulling_result(resp, {"d0": _pull_meta("d0", [[1, 2]])})
    assert w.take_block_ids_with_load_errors() == set()
    assert w.finished_recving_reqs == set(), "the base handles the ok path, not us"


@requires_mooncake
def test_stash_ff_rows_is_bounded(monkeypatch):
    """An aborted request's map is never ACKed; without a cap they accumulate for the whole run."""
    monkeypatch.setattr(mc, "_FF_ROWS_MAX_PENDING", 3)
    worker = types.SimpleNamespace(_ff_rows={})
    stash = mc.MooncakeConnectorWorkerFF.stash_ff_rows.__get__(worker)
    for i in range(10):
        stash(f"t{i}", 1, [[0, 5, 1]])
    assert list(worker._ff_rows) == ["t7", "t8", "t9"], "oldest maps evicted, newest kept"


@requires_mooncake
def test_response_encoder_passes_through_error_responses():
    worker = types.SimpleNamespace(_ff_rows={}, _ff_dreq2tid={})
    worker.pop_ff_rows = mc.MooncakeConnectorWorkerFF.pop_ff_rows.__get__(worker)
    encoder = mc._FFResponseEncoder(worker)
    resp = mc.MooncakeXferResponse(status=mc.MooncakeXferResponseStatus.ERROR, err_msg="boom")
    import msgspec
    assert msgspec.msgpack.Decoder(mc.MooncakeXferResponseFF).decode(
        encoder.encode(resp)).err_msg == "boom"


# =====================================================================================
# _build_transfer_params — the group-awareness that is the reason this connector exists
# =====================================================================================
@requires_mooncake
def _fake_worker(addr_groups, local_base=(1000, 2000), remote_base=(5000, 6000), block_len=16):
    w = types.SimpleNamespace(
        kv_caches_base_addr=list(local_base),
        base_addr_groups=[list(g) for g in addr_groups],
        block_len=block_len,
        _ff_dreq2tid={},
    )
    w._build_transfer_params = mc.MooncakeConnectorWorkerFF._build_transfer_params.__get__(w)
    return w


@requires_mooncake
def _run_params(worker, local_groups, remote_groups):
    send_meta = types.SimpleNamespace(local_block_ids=local_groups, transfer_id="t0")
    agent_meta = types.SimpleNamespace(
        req_blocks={"d0": ("t0", remote_groups)},
        kv_caches_base_addr=[5000, 6000],
        remote_hostname="h", remote_port=1)
    return asyncio.run(worker._build_transfer_params([("d0", send_meta)], agent_meta))


@requires_mooncake
def test_transfer_params_index_each_layer_by_its_own_group():
    """base addr 0 is a group-0 layer, base addr 1 is a group-1 layer; each must use its OWN
    group's block ids. Collapsing onto one list is precisely the corruption bug."""
    worker = _fake_worker(addr_groups=[[0], [1]])
    src, dst, lengths, err = _run_params(worker, [[3], [4]], [[30], [40]])
    assert not err
    assert src == [1000 + 3 * 16, 2000 + 4 * 16]
    assert dst == [5000 + 30 * 16, 6000 + 40 * 16]
    assert lengths == [16, 16]


@requires_mooncake
def test_transfer_params_fills_every_group_sharing_an_allocation():
    """THE regression. A base address is an allocation, not a layer: vLLM's hybrid allocator packs
    one layer from EVERY KV-cache group into each tensor ("layers of different groups ... use
    different parts of the shared Tensor"). Emitting only one group's blocks per allocation left the
    other groups' KV untransferred and overwrote their regions — F1 0.69 -> 0.28."""
    worker = _fake_worker(addr_groups=[[0, 1, 2]], local_base=(1000,), remote_base=(5000,))
    send_meta = types.SimpleNamespace(local_block_ids=[[3], [4], [5]], transfer_id="t0")
    agent_meta = types.SimpleNamespace(
        req_blocks={"d0": ("t0", [[30], [40], [50]])}, kv_caches_base_addr=[5000],
        remote_hostname="h", remote_port=1)
    src, dst, lengths, err = asyncio.run(
        worker._build_transfer_params([("d0", send_meta)], agent_meta))

    assert not err
    assert src == [1000 + 3 * 16, 1000 + 4 * 16, 1000 + 5 * 16], "all three groups, one allocation"
    assert dst == [5000 + 30 * 16, 5000 + 40 * 16, 5000 + 50 * 16]
    assert lengths == [16, 16, 16]


@requires_mooncake
def test_transfer_params_two_layers_in_the_same_group_share_a_block_list():
    worker = _fake_worker(addr_groups=[[1], [1]])
    src, dst, _lengths, err = _run_params(worker, [[], [4]], [[], [40]])
    assert not err
    assert src == [1000 + 4 * 16, 2000 + 4 * 16], "both layers of group 1 use group 1's blocks"
    assert dst == [5000 + 40 * 16, 6000 + 40 * 16]


@requires_mooncake
def test_transfer_params_coalesces_contiguous_blocks_per_group():
    worker = _fake_worker(addr_groups=[[1], [1]])
    src, _dst, lengths, err = _run_params(worker, [[], [4, 5, 6]], [[], [40, 41, 42]])
    assert not err
    assert lengths == [3 * 16, 3 * 16], "contiguous runs collapse into one transfer per layer"
    assert src == [1000 + 4 * 16, 2000 + 4 * 16]


@requires_mooncake
def test_transfer_params_truncates_local_tail_on_partial_prefix_hit():
    """D already had a prefix cached, so it asks for fewer blocks than P holds: P must send its
    TAIL, per group."""
    worker = _fake_worker(addr_groups=[[1]])
    src, _dst, _lengths, err = _run_params(worker, [[], [4, 5, 6]], [[], [50]])
    assert not err
    assert src == [1000 + 6 * 16], "the last local block pairs with the single remote block"


@requires_mooncake
def test_transfer_params_errors_when_a_group_is_short_on_the_producer():
    worker = _fake_worker(addr_groups=[[1]])
    src, _dst, _lengths, err = _run_params(worker, [[], [4]], [[], [50, 51]])
    assert err == ["d0"] and src == []


@requires_mooncake
def test_transfer_params_records_the_transfer_id_for_the_ack():
    """Without this mapping the redirect rows (keyed by transfer id) can never be attached to the
    ACK (keyed by decode request id)."""
    worker = _fake_worker(addr_groups=[[1]])
    _run_params(worker, [[], [4]], [[], [40]])
    assert worker._ff_dreq2tid == {"d0": "t0"}


# =====================================================================================
# BFF_FF_GROUPS — restrict fusion to the groups that actually pay
# =====================================================================================
def test_parse_groups_forms():
    assert mc._parse_groups(None) is None, "unset = all eligible groups"
    assert mc._parse_groups("") is None
    assert mc._parse_groups("   ") is None
    assert mc._parse_groups(",,") is None, "parses to nothing -> all, never 'disable everything'"
    assert mc._parse_groups("1,2,3") == {1, 2, 3}
    assert mc._parse_groups("1, 2,") == {1, 2}, "blanks tolerated"


def test_parse_groups_none_is_an_empty_set_not_none():
    """The control arm: keep the group split and every BFF patch, select NO fusion groups. It has
    to be distinguishable from unset, which means all groups."""
    for raw in ("none", "off", "NONE", " Off "):
        assert mc._parse_groups(raw) == frozenset()
        assert mc._parse_groups(raw) is not None


@requires_mooncake
def _fusion_spy(monkeypatch, selected):
    """A connector wired just enough for save_kv_layer, with on_layer instrumented."""
    monkeypatch.setattr(mc, "_FF_GROUPS", selected)
    conn = mc.MooncakeConnectorFF.__new__(mc.MooncakeConnectorFF)
    conn._ff_fuse = True
    conn.is_producer = True
    conn._ff_groups_logged = False
    conn._ff_tp = None
    calls = []
    fusion = types.SimpleNamespace(group_layers={1: {"l0"}, 2: {"l0"}, 3: {"l0"}})
    fusion.on_layer = lambda *a, **k: calls.append(a[0]) or None
    fusion.evict_owners = lambda tids: 0
    conn._fusion = fusion
    worker = types.SimpleNamespace(
        _group_layers={0: {"w"}, 1: {"l0"}, 2: {"l0"}, 3: {"l0"}},
        group_of=lambda ln: int(ln[1]),          # "g2.l0" -> 2
        take_done_tids=lambda: set(),
        stash_ff_rows=lambda *a: None)
    conn.connector_worker = worker
    conn._get_connector_metadata = lambda: types.SimpleNamespace(
        fuse_reqs=[("r0", "t0", [[], [1], [1], [1]])])
    return conn, calls


@requires_mooncake
def test_excluded_group_never_reaches_the_fusion_engine(monkeypatch):
    """The whole point of the knob: an excluded group must not compute a block repr, buffer a layer,
    cluster, or register — so the guard has to sit BEFORE on_layer, not inside it."""
    conn, calls = _fusion_spy(monkeypatch, {1, 2})
    kv = torch.zeros(2, 2, 1, 1, 4)

    for gi in (1, 2, 3):
        conn.save_kv_layer(f"g{gi}.l0", kv, None)

    assert calls == [1, 2], "group 3 was excluded and did no work"


@requires_mooncake
def test_unset_ff_groups_fuses_every_eligible_group(monkeypatch):
    conn, calls = _fusion_spy(monkeypatch, None)
    kv = torch.zeros(2, 2, 1, 1, 4)
    for gi in (1, 2, 3):
        conn.save_kv_layer(f"g{gi}.l0", kv, None)
    assert calls == [1, 2, 3]


@requires_mooncake
def test_ff_groups_none_fuses_nothing(monkeypatch):
    """Control arm: group split stays active, zero clustering, zero redirects."""
    conn, calls = _fusion_spy(monkeypatch, frozenset())
    kv = torch.zeros(2, 2, 1, 1, 4)
    for gi in (1, 2, 3):
        conn.save_kv_layer(f"g{gi}.l0", kv, None)
    assert calls == []


@requires_mooncake
def test_warmup_group_is_never_fused_regardless(monkeypatch):
    conn, calls = _fusion_spy(monkeypatch, {0, 1})
    conn.save_kv_layer("g0.l0", torch.zeros(2, 2, 1, 1, 4), None)
    assert calls == [], "group 0 is the warmup group"


# =====================================================================================
# base-address -> group mapping (register_kv_caches)
# =====================================================================================
def _addr_group_worker(groups, tensors, n_addr, split_k_and_v=True):
    w = mc.MooncakeConnectorWorkerFF.__new__(mc.MooncakeConnectorWorkerFF)
    w._layer_group = {ln: gi for gi, g in enumerate(groups) for ln in g}
    w._group_layers = {gi: set(g) for gi, g in enumerate(groups)}
    w._warned_layers = set()
    w.kv_topo = types.SimpleNamespace(split_k_and_v=split_k_and_v)
    w.kv_caches_base_addr = [0] * n_addr
    return w


def _install_runner_config(monkeypatch, groups, tensors):
    from kv_fast_fusion import fast_fusion_block_pool as bp
    cfg = types.SimpleNamespace(
        kv_cache_tensors=[types.SimpleNamespace(shared_by=t) for t in tensors],
        kv_cache_groups=[types.SimpleNamespace(layer_names=g) for g in groups])
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER",
                        types.SimpleNamespace(kv_cache_config=cfg), raising=False)


@requires_mooncake
def test_base_addr_groups_matches_the_real_hybrid_layout(monkeypatch):
    """The exact shape the failing run had: BFF's 7 groups x 4 layers become 4 shared tensors (one
    layer from EVERY group each), x2 for split K/V = 8 base addresses. Every address must map to all
    7 groups. The old code reported 'across 1 KV-cache groups' here — that was the corruption."""
    groups = [[f"g{g}.l{i}" for i in range(4)] for g in range(7)]
    tensors = [[groups[j][i] for j in range(7)] for i in range(4)]
    _install_runner_config(monkeypatch, groups, tensors)

    out = _addr_group_worker(groups, tensors, n_addr=8)._build_base_addr_groups()

    assert len(out) == 8
    assert all(gs == list(range(7)) for gs in out), "every allocation holds all 7 groups"


@requires_mooncake
def test_base_addr_groups_handles_unsplit_kv(monkeypatch):
    groups = [["g0.l0"], ["g1.l0"]]
    tensors = [["g0.l0", "g1.l0"]]
    _install_runner_config(monkeypatch, groups, tensors)
    out = _addr_group_worker(groups, tensors, n_addr=1,
                             split_k_and_v=False)._build_base_addr_groups()
    assert out == [[0, 1]]


@requires_mooncake
def test_base_addr_groups_refuses_to_serve_on_an_inconsistent_map(monkeypatch):
    """Silently degrading to one group is what produced a wrong-but-running server (F1 0.69 ->
    0.28). An incomplete map must abort startup instead."""
    groups = [["g0.l0"], ["g1.l0"]]
    tensors = [["g0.l0"]]                       # group 1 covered by no allocation
    _install_runner_config(monkeypatch, groups, tensors)
    with pytest.raises(RuntimeError, match="Refusing to serve"):
        _addr_group_worker(groups, tensors, n_addr=2)._build_base_addr_groups()


@requires_mooncake
def test_base_addr_groups_refuses_when_the_runner_config_is_unavailable(monkeypatch):
    from kv_fast_fusion import fast_fusion_block_pool as bp
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", None, raising=False)
    with pytest.raises(RuntimeError, match="cannot read kv_cache_config"):
        _addr_group_worker([["l0"]], [["l0"]], n_addr=2)._build_base_addr_groups()


# =====================================================================================
# scheduler side: fuse_reqs collection
# =====================================================================================
def _sched_output(new_reqs, cached=None, scheduled_tokens=None):
    return types.SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached or types.SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()),
        num_scheduled_tokens=scheduled_tokens or {},
    )


def _new_req(req_id, tid, prompt_len, blocks, computed=0):
    """A REAL NewRequestData. Deliberately not a SimpleNamespace: the previous fake invented a
    `kv_transfer_params` attribute that this dataclass does not have, so the tests passed while
    `_collect_fuse_reqs` skipped every request in production and fusion never ran at all. The
    transfer id therefore has to arrive the way it really does — via `_ff_tid_of`."""
    from vllm.v1.core.sched.output import NewRequestData
    return NewRequestData(
        req_id=req_id, prompt_token_ids=list(range(prompt_len)), mm_features=[],
        sampling_params=None, pooling_params=None, block_ids=tuple(tuple(g) for g in blocks),
        num_computed_tokens=computed, lora_request=None)


@requires_mooncake
def _fake_scheduler(tids=None):
    s = types.SimpleNamespace(_ff_chunked={}, _ff_tid_of=dict(tids or {}))
    s._collect_fuse_reqs = mc.MooncakeConnectorSchedulerFF._collect_fuse_reqs.__get__(s)
    return s


@requires_mooncake
def test_new_request_data_really_has_no_kv_transfer_params():
    """Pins the fact the original bug hinged on. If a future vLLM adds the field, this fails and we
    can simplify — until then `_collect_fuse_reqs` must not reach for it."""
    from vllm.v1.core.sched.output import NewRequestData
    assert not hasattr(_new_req("r", "t", 4, [[1]]), "kv_transfer_params")
    assert "kv_transfer_params" not in NewRequestData.__dataclass_fields__


@requires_mooncake
def test_update_state_after_alloc_records_the_producer_transfer_id():
    """The only place the transfer id is reachable — via the real Request object."""
    # A real instance (via __new__, skipping the engine-touching __init__): the method delegates to
    # super() for the producer branch, and zero-arg super() requires one.
    sched = mc.MooncakeConnectorSchedulerFF.__new__(mc.MooncakeConnectorSchedulerFF)
    sched._ff_tid_of = {}
    sched.is_kv_producer, sched.is_kv_consumer = True, False
    sched._reqs_need_send, sched._reqs_need_recv = {}, {}
    req = types.SimpleNamespace(
        request_id="r0", kv_transfer_params={"do_remote_decode": True, "transfer_id": "t0"})
    sched.update_state_after_alloc(req, blocks=None, num_external_tokens=0)
    assert sched._ff_tid_of == {"r0": "t0"}
    assert "r0" in sched._reqs_need_send, "stock producer bookkeeping still runs"


@requires_mooncake
def test_fuse_reqs_collected_for_a_single_step_prefill(monkeypatch):
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", True)
    sched, meta = _fake_scheduler({"r0": "t0"}), mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(
        _sched_output([_new_req("r0", "t0", 8, [[1], [2]])], scheduled_tokens={"r0": 8}), meta)
    assert meta.fuse_reqs == [("r0", "t0", [[1], [2]])]
    assert not sched._ff_chunked


@requires_mooncake
def test_fuse_reqs_defers_chunked_prefill_until_the_last_chunk(monkeypatch):
    """Reading K from blocks that have not been written yet would cluster on garbage, so a prompt
    spanning several steps must accumulate and emit exactly once, with ALL its blocks."""
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", True)
    sched, meta = _fake_scheduler({"r0": "t0"}), mc.MooncakeConnectorMetadataFF()

    sched._collect_fuse_reqs(
        _sched_output([_new_req("r0", "t0", 10, [[1], [2]])], scheduled_tokens={"r0": 6}), meta)
    assert meta.fuse_reqs == [] and "r0" in sched._ff_chunked

    cached = types.SimpleNamespace(req_ids=["r0"], new_block_ids=[[[3], [4]]],
                                   num_computed_tokens=[6], resumed_req_ids=set())
    meta2 = mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(_sched_output([], cached, {"r0": 4}), meta2)
    assert meta2.fuse_reqs == [("r0", "t0", [[1, 3], [2, 4]])]
    assert not sched._ff_chunked, "entry cleared once emitted"


@requires_mooncake
def test_fuse_reqs_chunk_with_no_new_blocks_carries_the_accumulation(monkeypatch):
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", True)
    sched, meta = _fake_scheduler({"r0": "t0"}), mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(
        _sched_output([_new_req("r0", "t0", 10, [[1], [2]])], scheduled_tokens={"r0": 6}), meta)
    cached = types.SimpleNamespace(req_ids=["r0"], new_block_ids=[None],
                                   num_computed_tokens=[6], resumed_req_ids=set())
    meta2 = mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(_sched_output([], cached, {"r0": 4}), meta2)
    assert meta2.fuse_reqs == [("r0", "t0", [[1], [2]])]


@requires_mooncake
def test_fuse_reqs_resumed_request_restarts_its_accumulation(monkeypatch):
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", True)
    sched, meta = _fake_scheduler({"r0": "t0"}), mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(
        _sched_output([_new_req("r0", "t0", 10, [[1], [2]])], scheduled_tokens={"r0": 6}), meta)
    cached = types.SimpleNamespace(req_ids=["r0"], new_block_ids=[[[9], [8]]],
                                   num_computed_tokens=[6], resumed_req_ids={"r0"})
    meta2 = mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(_sched_output([], cached, {"r0": 4}), meta2)
    assert meta2.fuse_reqs == [("r0", "t0", [[9], [8]])], "preemption discards the stale blocks"


@requires_mooncake
def test_fuse_reqs_skips_requests_without_a_transfer_id(monkeypatch):
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", True)
    sched, meta = _fake_scheduler(), mc.MooncakeConnectorMetadataFF()   # no transfer id recorded
    sched._collect_fuse_reqs(
        _sched_output([_new_req("r0", "t0", 4, [[1]])], scheduled_tokens={"r0": 4}), meta)
    assert meta.fuse_reqs == [], "a request the producer never registered is not fusable"


@requires_mooncake
def test_fuse_reqs_noop_when_fusion_disabled(monkeypatch):
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", False)
    sched, meta = _fake_scheduler({"r0": "t0"}), mc.MooncakeConnectorMetadataFF()
    sched._collect_fuse_reqs(
        _sched_output([_new_req("r0", "t0", 4, [[1]])], scheduled_tokens={"r0": 4}), meta)
    assert meta.fuse_reqs == []


# =====================================================================================
# consumer apply lifecycle
# =====================================================================================
class _StubWorker:
    def __init__(self, arrived, hash2rid):
        self._arrived = arrived
        self._hash2rid = hash2rid

    def drain_ff_rows(self):
        out, self._arrived = self._arrived, {}
        return out, dict(self._hash2rid)


def _connector_for_apply(monkeypatch, runner, arrived, hash2rid):
    """A MooncakeConnectorFF with just enough state wired for _ff_consumer_apply, without
    constructing the real Mooncake worker (which would need a Transfer Engine)."""
    from kv_fast_fusion import fast_fusion_block_pool as bp
    monkeypatch.setattr(bp, "_ACTIVE_RUNNER", runner, raising=False)
    conn = mc.MooncakeConnectorFF.__new__(mc.MooncakeConnectorFF)
    conn.connector_worker = _StubWorker(arrived, hash2rid)
    conn._ff_step = 1
    conn._ff_pending = {}
    conn._ff_pending_merges = None
    conn._ff_applied = conn._ff_unresolved = 0
    conn._ff_tp = None
    return conn


@requires_mooncake
def test_apply_rewrites_block_table_and_stages_the_merge(monkeypatch):
    runner = _FakeRunner(["own", "rep"], ngroups=2, ncol=2)
    runner.requests = {
        "own": types.SimpleNamespace(block_ids=[[0, 0], [100, 101]]),
        "rep": types.SimpleNamespace(block_ids=[[0, 0], [200, 201]]),
    }
    conn = _connector_for_apply(
        monkeypatch, runner,
        arrived={"own": {1: [[1, mc._tid_hash("t-rep"), 0]]}},
        hash2rid={mc._tid_hash("t-rep"): "rep"})

    conn._ff_consumer_apply()

    assert runner._updated_block_tables == {"own": {1: [100, 200]}}
    assert conn._ff_pending_merges == {"own": {1: [100, 200]}}
    assert runner.requests["own"].block_ids[1] == [100, 200]
    assert conn._ff_pending == {}, "applied maps are dropped"


@requires_mooncake
def test_apply_holds_a_map_until_its_owner_is_batched(monkeypatch):
    """Under the pull model the recv completes between steps, so the owner is not in input_batch
    when its map lands. The map must wait, not be discarded."""
    runner = _FakeRunner(["rep"], ngroups=2, ncol=2)          # "own" NOT batched yet
    runner.requests = {"rep": types.SimpleNamespace(block_ids=[[0, 0], [200, 201]])}
    conn = _connector_for_apply(
        monkeypatch, runner,
        arrived={"own": {1: [[1, mc._tid_hash("t-rep"), 0]]}},
        hash2rid={mc._tid_hash("t-rep"): "rep"})

    conn._ff_consumer_apply()
    assert "own" in conn._ff_pending and runner._updated_block_tables is None

    # Next step the scheduler batches the owner → the held map applies.
    runner.input_batch.req_id_to_index["own"] = 1
    runner.requests["own"] = types.SimpleNamespace(block_ids=[[0, 0], [100, 101]])
    runner.input_batch.block_table.block_tables = [_FakeGroupTable(2, 2) for _ in range(2)]
    conn._ff_step += 1
    conn._ff_consumer_apply()
    assert runner._updated_block_tables == {"own": {1: [100, 200]}}


@requires_mooncake
def test_apply_drops_a_map_whose_owner_never_gets_batched(monkeypatch):
    runner = _FakeRunner(["other"])
    conn = _connector_for_apply(monkeypatch, runner, arrived={"own": {1: [[0, 1, 0]]}}, hash2rid={})
    conn._ff_consumer_apply()
    conn._ff_step += mc._FF_APPLY_MAX_AGE + 1
    conn._ff_consumer_apply()
    assert conn._ff_pending == {}, "stale maps must not accumulate forever"


@requires_mooncake
def test_apply_does_not_stage_a_merge_it_could_not_write(monkeypatch):
    """The rep is unresolvable, so nothing is rewritten — and therefore nothing may be freed."""
    runner = _FakeRunner(["own"], ngroups=2, ncol=2)
    runner.requests = {"own": types.SimpleNamespace(block_ids=[[0, 0], [100, 101]])}
    conn = _connector_for_apply(
        monkeypatch, runner, arrived={"own": {1: [[1, 999999, 0]]}}, hash2rid={})
    conn._ff_consumer_apply()
    assert runner._updated_block_tables is None and conn._ff_pending_merges is None


@requires_mooncake
def test_apply_survives_a_missing_runner(monkeypatch):
    conn = _connector_for_apply(monkeypatch, None, arrived={"own": {1: [[0, 1, 0]]}}, hash2rid={})
    conn._ff_consumer_apply()          # must not raise
    assert conn._ff_pending_merges is None


# =====================================================================================
# construction
# =====================================================================================
def _fake_vllm_config(kv_role="kv_producer", engine_id="prefill-0"):
    return types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(
            kv_role=kv_role, engine_id=engine_id, kv_connector_extra_config={}))


@requires_mooncake
def test_scheduler_role_builds_the_ff_scheduler_and_no_worker():
    """The worker is what owns the Transfer Engine and the bootstrap server, so the scheduler role
    must never construct one — and the scheduler it does build must be the FF subclass, or the
    per-group block ids silently collapse back to stock behavior."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    conn = mc.MooncakeConnectorFF(
        _fake_vllm_config(), KVConnectorRole.SCHEDULER, kv_cache_config=None)
    assert isinstance(conn.connector_scheduler, mc.MooncakeConnectorSchedulerFF)
    assert conn.connector_worker is None
    assert conn.is_producer is True
    assert conn.engine_id == "prefill-0"


@requires_mooncake
def test_construction_requires_an_explicit_engine_id():
    """engine_id defaults to a fresh uuid per process, which the proxy cannot know — so an
    unset one has to fail loudly at startup rather than at the first transfer."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    with pytest.raises(AssertionError, match="engine_id"):
        mc.MooncakeConnectorFF(
            _fake_vllm_config(engine_id=None), KVConnectorRole.SCHEDULER, None)


@requires_mooncake
def test_consumer_role_scheduler_is_not_a_producer():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    conn = mc.MooncakeConnectorFF(
        _fake_vllm_config("kv_consumer", "decode-0"), KVConnectorRole.SCHEDULER, None)
    assert conn.is_producer is False
    assert conn.connector_scheduler.is_kv_consumer is True


@requires_mooncake
def test_piecewise_cudagraph_is_demanded_exactly_when_fusion_is_on(monkeypatch):
    """save_kv_layer does real work here; inside a full CUDA graph it would be skipped on replay
    and fusion would silently stop."""
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", True)
    assert mc.MooncakeConnectorFF.requires_piecewise_for_cudagraph({}) is True
    monkeypatch.setattr(mc, "_BFF_PD_FUSE", False)
    assert mc.MooncakeConnectorFF.requires_piecewise_for_cudagraph({}) is False


@requires_mooncake
def test_save_kv_layer_is_inert_on_the_consumer():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    conn = mc.MooncakeConnectorFF(
        _fake_vllm_config("kv_consumer", "decode-0"), KVConnectorRole.SCHEDULER, None)
    conn.save_kv_layer("l0", torch.zeros(2, 2, 1, 1, 1), None)   # must not raise


# =====================================================================================
# stats carrier
# =====================================================================================
def test_merge_stats_keeps_one_ranks_map_and_never_concatenates():
    a = mc.BFFMergeStats(data={})
    b = mc.BFFMergeStats(data={"bff_merges": {"r": {1: [1, 2]}}})
    assert a.is_empty() and not b.is_empty()
    assert a.aggregate(b).data == b.data
    # Aggregating a second identical rank must not double the map (that would corrupt ref counts).
    c = mc.BFFMergeStats(data={"bff_merges": {"r": {1: [1, 2]}}})
    assert a.aggregate(c).data["bff_merges"] == {"r": {1: [1, 2]}}
    assert b.reduce() == {"bff_merge_reqs": 1}


# =====================================================================================
# overhead accounting + stats-dump cadence
#
# The metric these pin down is the one that reads as "fusion is slow". It has two ways of lying:
# billing fusion for GPU work it merely waited behind, and freezing the JSON at a cold-start
# snapshot when the step rate drops (which is exactly what BFF_FF_GROUPS=1 does).
# =====================================================================================
def test_overhead_separates_fusion_cost_from_the_queue_drain():
    fusion = mc.FFProducerFusion({1: {"l0"}})
    drained = []
    fusion._drain = lambda _buf: drained.append(True)       # stands in for the device sync
    kv = _kv_layer([[1.0, 0.0], [1.0, 0.0]])
    fusion.on_layer(1, "l0", kv, [("t-A", _blocks([], [1])), ("t-B", _blocks([], [2]))], 1, False)

    assert drained, "the drain runs before the timed region, not inside it"
    assert fusion.steps == 1
    assert len(fusion.step_ms) == 1 and fusion.step_ms[0] == fusion.ms
    s = fusion.stats_dict()
    assert "overhead_avg_queue_drain_ms" in s
    assert s["overhead_ms_pct"]["n"] == 1, "percentiles report their own sample count"


def test_dump_cadence_survives_a_low_step_rate(monkeypatch):
    """BFF_FF_GROUPS=1 accrues ~6x fewer steps per forward pass, so a run can finish below the
    step-count threshold. Without the wall-clock backstop the file keeps the step-1 snapshot and
    the reported mean is one cold-start sample."""
    monkeypatch.setattr(mc, "_PD_FUSE_LOG_EVERY", 50)
    monkeypatch.setattr(mc, "_PD_STATS_DUMP_SEC", 10.0)
    now = [1000.0]
    monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
    fusion = mc.FFProducerFusion({1: {"l0"}})

    assert not fusion.should_dump(), "nothing to dump before the first step"
    fusion.steps = 1
    assert fusion.should_dump(), "first steps always dump"
    fusion.steps = 12
    assert not fusion.should_dump(), "no step milestone, no time elapsed"
    now[0] += 11.0
    assert fusion.should_dump(), "wall clock forces a refresh"
    now[0] += 1.0
    assert not fusion.should_dump(), "and the clock resets on dump"
    fusion.steps = 50
    assert fusion.should_dump(), "step milestone still works"


def test_drain_is_a_noop_off_cuda():
    """CPU tensors have no stream to synchronize; the unit tests all run this path."""
    mc.FFProducerFusion._drain({"k_layers": [torch.zeros(2, 2)]})
    mc.FFProducerFusion._drain({})


# =====================================================================================
# liveness feedback (D -> P) and the residency guard
#
# A merge representative is only usable while its KV is still resident on the decode instance.
# Nothing on the producer knows that, so D reports finished transfer ids back on the pull request
# it is already sending. Without this the index kept serving reps D had freed: 63% of all redirects
# resolved to nothing, decaying 56% -> 7% over one run.
# =====================================================================================
def _consumer_worker():
    import threading as _t
    w = mc.MooncakeConnectorWorkerFF.__new__(mc.MooncakeConnectorWorkerFF)
    w._ff_lock = _t.Lock()
    w._ff_recv_rows, w._ff_tid2rid, w._ff_rid2tid = {}, {}, {}
    w._ff_done_pending = mc.OrderedDict()
    w._ff_done_from_d = set()
    return w


@requires_mooncake
def test_finished_decode_requests_are_advertised_to_producers():
    w = _consumer_worker()
    w.note_pull_ids({"eng": {"d0": _pull_meta("d0", [[1]]), "d1": _pull_meta("d1", [[2]])}})
    assert w.peek_done_tids() is None, "nothing finished yet"

    w.forget_ff_ids({"d0"})

    assert w.peek_done_tids() == ["t-d0"]
    assert w.peek_done_tids() == ["t-d0"], "peek, not pop — every producer must see it"


@requires_mooncake
def test_advertised_ids_expire(monkeypatch):
    w = _consumer_worker()
    w.note_pull_ids({"eng": {"d0": _pull_meta("d0", [[1]])}})
    w.forget_ff_ids({"d0"})
    assert w.peek_done_tids() == ["t-d0"]

    monkeypatch.setattr(mc, "_FF_DONE_TID_TTL", -1.0)     # everything is now expired
    assert w.peek_done_tids() is None, "the list cannot grow without bound over a long run"


@requires_mooncake
def test_pull_request_carries_done_ids_and_the_producer_evicts_them():
    """Full round trip over the real structs: D encodes, P decodes, P's index drops those reps."""
    consumer = _consumer_worker()
    consumer.note_pull_ids({"eng": {"d0": _pull_meta("d0", [[1]])}})
    consumer.forget_ff_ids({"d0"})

    meta = mc.MooncakeXferMetadata(
        remote_hostname="h", remote_port=1, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"d1": ("t-d1", [[3]])}, kv_caches_base_addr=[0])
    wire = mc._FFResponseEncoder(consumer).encode(meta)

    producer = _consumer_worker()
    decoded = mc._FFMetadataDecoder(producer).decode(wire)

    assert decoded.done_tids == ["t-d0"]
    assert producer.take_done_tids() == {"t-d0"}
    assert producer.take_done_tids() == set(), "drained once"


@requires_mooncake
def test_fusion_evicts_reps_whose_owner_finished(monkeypatch):
    monkeypatch.setattr(mc, "_PD_CROSS_INDEX", "lsh")
    monkeypatch.setattr(mc, "_PD_ENCODED_BATCH", 0)
    f = mc.FFProducerFusion({1: {"l0"}})
    v = [1.0, 0.0, 0.0, 0.5]
    f.on_layer(1, "l0", _kv_layer([v]), [("t-A", _blocks([], [1]))], 1, False)
    assert f._lsh[1].size() == 1

    assert f.evict_owners({"t-A"}) == 1
    assert f._lsh[1].size() == 0 and f.lsh_evicted == 1
    # And the next request no longer matches the dead rep — it becomes a rep itself.
    assert f.on_layer(1, "l0", _kv_layer([v]), [("t-B", _blocks([], [1]))], 2, False) == {}
    assert f._lsh[1].size() == 1


@requires_mooncake
def test_apply_ignores_a_rep_that_is_resident_but_not_batched(monkeypatch):
    """A PREEMPTED request keeps its runner.requests entry while the scheduler has freed and
    reallocated its blocks. Redirecting to it would alias a live owner onto another request's KV,
    so residency has to mean 'batched this step' — the same test the owner passes."""
    runner = _FakeRunner(["own"], ngroups=2, ncol=2)          # "rep" NOT in input_batch
    runner.requests = {
        "own": types.SimpleNamespace(block_ids=[[0, 0], [100, 101]]),
        "rep": types.SimpleNamespace(block_ids=[[0, 0], [200, 201]]),   # stale
    }
    conn = _connector_for_apply(
        monkeypatch, runner,
        arrived={"own": {1: [[1, mc._tid_hash("t-rep"), 0]]}},
        hash2rid={mc._tid_hash("t-rep"): "rep"})

    conn._ff_consumer_apply()

    assert runner._updated_block_tables is None, "no rewrite against a non-batched rep"
    assert runner.requests["own"].block_ids[1] == [100, 101]
    assert conn._ff_unresolved == 1


# =====================================================================================
# per-group thresholds + the similarity audit
# =====================================================================================
def test_threshold_per_group_parsing():
    assert mc._parse_thresholds("1:0.97,2:0.90") == {1: 0.97, 2: 0.9}
    assert mc._parse_thresholds(" 1:0.97 , ") == {1: 0.97}
    assert mc._parse_thresholds(None) == {} and mc._parse_thresholds("") == {}
    assert mc._parse_thresholds("junk,3:x,4:0.5") == {4: 0.5}, "bad entries dropped, good kept"


def test_per_group_threshold_overrides_the_global_one(monkeypatch):
    """Same pair of blocks, two groups: the one with the raised bar must not merge them."""
    monkeypatch.setattr(mc, "_THRESHOLD_G", {2: 0.999})
    near = [1.0, 0.0, 0.0]
    other = [0.94, 0.34, 0.0]                       # cosine ~0.94: over 0.75, under 0.999
    f = mc.FFProducerFusion({1: {"l0"}, 2: {"l0"}})
    reqs = [("t-A", _blocks([], [1], [1])), ("t-B", _blocks([], [2], [2]))]

    assert f.on_layer(1, "l0", _kv_layer([near, other]), reqs, 1, False), "g1 uses BFF_THRESHOLD"
    assert f.on_layer(2, "l0", _kv_layer([near, other]), reqs, 1, False) == {}


def test_audit_samples_random_cross_request_pairs(monkeypatch):
    monkeypatch.setattr(mc, "_PD_AUDIT", True)
    monkeypatch.setattr(mc, "_PD_AUDIT_STEPS", 1)
    f = mc.FFProducerFusion({1: {"l0"}})
    orth = [[1.0, 0.0], [0.0, 1.0]]
    reqs = [("t-A", _blocks([], [1])), ("t-B", _blocks([], [2]))]

    f.on_layer(1, "l0", _kv_layer(orth), reqs, 1, False)
    q = f.stats_dict()["audit_random_pair_cos"]["1"]
    assert q["n"] > 0 and abs(q["p50"]) < 1e-5, "orthogonal blocks -> a floor at cosine 0"

    f.on_layer(1, "l0", _kv_layer(orth), reqs, 2, False)
    assert f.audit_steps[1] == 1, "sampling stops after BFF_PD_AUDIT_STEPS"


def test_stats_report_the_layer_ceiling(monkeypatch):
    """Fusion can only dedup the layers it runs on, so a factor above that share is impossible."""
    monkeypatch.setattr(mc, "_FF_GROUPS", {1})
    f = mc.FFProducerFusion({0: {"w0", "w1"}, 1: {"a", "b"}, 2: {"c", "d"}})
    s = f.stats_dict()
    assert s["layers_total"] == 6
    assert s["layers_fused"] == 2, "group 0 is never fused and BFF_FF_GROUPS excluded group 2"


# =====================================================================================
# producer hold window
# =====================================================================================
# What these pin: the producer pins a finished request's KV from `record_send_reqs` (prefill done,
# abort timer started) until `fetch_finished_sending_reqs` reports it (pull complete, or the 480 s
# timeout). That duration x the pinned block count is the ceiling on every "cancel the pull and free
# P's blocks early" design, and it had never been measured. Both wrapped methods are stock with no
# other FF override, so a break here is silent — the run simply reports nothing, which is exactly
# what an unrelated failed run looks like.
def _hold_worker(monkeypatch):
    import asyncio as _a
    w = mc.MooncakeConnectorWorkerFF.__new__(mc.MooncakeConnectorWorkerFF)
    w._ff_hold, w._ff_hold_done, w._ff_hold_logged = {}, [], 0.0
    base = mc.MooncakeConnectorWorker

    async def _noop_record(self, metadata):
        return None

    async def _noop_fetch(self):
        return self._test_finished

    monkeypatch.setattr(base, "record_send_reqs", _noop_record, raising=False)
    monkeypatch.setattr(base, "fetch_finished_sending_reqs", _noop_fetch, raising=False)
    return w, _a


@requires_mooncake
def test_the_hold_window_is_measured_from_prefill_done_to_release(monkeypatch, caplog):
    """A released request reports a duration and its block count, per-GROUP lists summed."""
    w, aio = _hold_worker(monkeypatch)
    meta = types.SimpleNamespace(reqs_to_send={"p0": ("t0", [[1, 2], [3, 4, 5]])})

    aio.run(w.record_send_reqs(meta))
    assert set(w._ff_hold) == {"p0"} and w._ff_hold["p0"][1] == 5, "per-group lists must be summed"

    w._test_finished = {"p0"}
    with caplog.at_level("INFO"):
        aio.run(w.fetch_finished_sending_reqs())

    assert w._ff_hold == {}, "a released request must stop counting as in-flight"
    assert "BFF hold |" in caplog.text
    assert "released=1" in caplog.text and "released_blocks=5" in caplog.text


@requires_mooncake
def test_a_flat_block_list_is_counted_too(monkeypatch):
    """Stock passes a flat list, FF passes per-group lists. Counting only one shape would silently
    report zero pinned blocks on the other, which reads identically to 'nothing is pinned'."""
    w, aio = _hold_worker(monkeypatch)
    aio.run(w.record_send_reqs(
        types.SimpleNamespace(reqs_to_send={"p0": ("t0", [1, 2, 3])})))
    assert w._ff_hold["p0"][1] == 3


@requires_mooncake
def test_a_preregistered_request_is_not_a_hold(monkeypatch):
    """`record_send_reqs` is also called with an empty block list from update_state_after_alloc,
    before the request has finished prefilling. Counting that as a hold would start the clock long
    before the blocks are actually pinned and inflate every duration."""
    w, aio = _hold_worker(monkeypatch)
    aio.run(w.record_send_reqs(types.SimpleNamespace(reqs_to_send={"p0": ("t0", [])})))
    assert w._ff_hold == {}


@requires_mooncake
def test_instrumentation_never_breaks_the_transfer(monkeypatch):
    """The wrapped methods are on the transfer's critical path. A malformed metadata must cost the
    measurement, never the send."""
    w, aio = _hold_worker(monkeypatch)
    aio.run(w.record_send_reqs(types.SimpleNamespace(reqs_to_send={"p0": "not-a-tuple"})))
    assert w._ff_hold == {}, "bad input is dropped, not raised"

    w._test_finished = {"p0"}
    assert aio.run(w.fetch_finished_sending_reqs()) == {"p0"}, "the release set still passes through"
