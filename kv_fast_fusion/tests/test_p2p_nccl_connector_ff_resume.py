"""Preempt/resume correctness for P2pNcclConnectorFF.

A decode request receives its prompt-1 KV once from the producer, which then FREES it. If the request is
later preempted (KV saturation) and resumed, the base `get_num_new_matched_tokens` would again advertise
external tokens → the connector re-fetches KV the producer no longer holds → stale/missing context → the
request rambles (F1 collapse observed only once preemption starts). The fix (BFF_PD_RESUME_LOCAL, default
on) makes a request that has already loaded recompute the prompt LOCALLY on resume — return 0 external
tokens — so it never re-enters the remote-load set.

Runs off-cluster: bypasses the heavy __init__ (which builds a real NCCL engine) via object.__new__ and
sets only the attributes the scheduler-side method touches. Run with:
    PYTHONPATH=<repo root> pytest kv_fast_fusion/tests/test_p2p_nccl_connector_ff_resume.py
"""
import types

import kv_fast_fusion.connectors.p2p_nccl_connector_ff as m

Conn = m.P2pNcclConnectorFF


def _req(rid, prompt_len):
    return types.SimpleNamespace(request_id=rid, prompt_token_ids=list(range(prompt_len)))


def _consumer():
    c = object.__new__(Conn)          # skip __init__ (no NCCL engine off-cluster)
    c.is_producer = False
    c._pd_loaded_once = set()
    return c


def test_unseen_request_uses_base_remote_load():
    # First load: not yet loaded-once → base advertises len(prompt)-1-num_computed external tokens.
    c = _consumer()
    assert c.get_num_new_matched_tokens(_req("r1", 10), 0) == (9, False)


def test_loaded_once_forces_local_recompute_on_resume():
    c = _consumer()
    c._pd_loaded_once.add("r1")                      # simulate the committed initial load
    assert c.get_num_new_matched_tokens(_req("r1", 10), 0) == (0, False)
    # an unrelated fresh request is unaffected
    assert c.get_num_new_matched_tokens(_req("r2", 10), 0) == (9, False)


def test_flag_off_restores_old_refetch_behavior():
    saved = m._PD_RESUME_LOCAL
    m._PD_RESUME_LOCAL = False
    try:
        c = _consumer()
        c._pd_loaded_once.add("r1")
        # with the fix disabled, a loaded-once req still advertises external tokens (buggy re-fetch path)
        assert c.get_num_new_matched_tokens(_req("r1", 10), 0) == (9, False)
    finally:
        m._PD_RESUME_LOCAL = saved


def test_producer_goes_through_base_not_the_guard():
    c = _consumer()
    c.is_producer = True
    c._pd_loaded_once.add("r1")
    assert c.get_num_new_matched_tokens(_req("r1", 10), 0) == (0, False)


if __name__ == "__main__":
    test_unseen_request_uses_base_remote_load()
    test_loaded_once_forces_local_recompute_on_resume()
    test_flag_off_restores_old_refetch_behavior()
    test_producer_goes_through_base_not_the_guard()
    print("ALL PASS")
