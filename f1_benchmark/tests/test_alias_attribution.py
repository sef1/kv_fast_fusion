"""The join that attributes a BFF run's F1 change to the prompts dedup actually touched.

The client saves each completion's id (chatcmpl-<X-Request-Id>); the P/D proxy sends the decode the
same uuid, and the decode's ledger is keyed by its external request id. CPU only, no server.
"""

import asyncio
import json

from f1_benchmark import alias_attribution as aa
from f1_benchmark.f1_main import _parse_stream

U1 = "0a1b2c3d-0000-4000-8000-000000000001"
U2 = "0a1b2c3d-0000-4000-8000-000000000002"
U3 = "0a1b2c3d-0000-4000-8000-000000000003"


class _Lines:
    def __init__(self, lines):
        self._it = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Resp:
    def __init__(self, chunks):
        self.content = _Lines([("data: " + json.dumps(c) + "\n").encode() for c in chunks]
                              + [b"data: [DONE]\n"])


def test_the_stream_parser_keeps_the_completion_id():
    rid = f"chatcmpl-{U1}"
    chunks = [{"id": rid, "choices": [{"delta": {"content": "hi"}}]},
              {"id": rid, "choices": [{"delta": {}, "finish_reason": "stop"}]}]
    rec = asyncio.run(_parse_stream(_Resp(chunks), 0.0))
    assert rec["response_id"] == rid and rec["generated_text"] == "hi"


def test_ids_join_on_the_proxy_uuid_whatever_wraps_it():
    assert aa.uuid_of(f"chatcmpl-{U1}") == U1
    assert aa.uuid_of(f"chatcmpl-{U1}-abcdef12") == U1      # the engine's local suffix
    assert aa.uuid_of(f"chatcmpl-{U1.upper()}") == U1
    assert aa.uuid_of("cmpl-no-uuid-here") is None
    assert aa.uuid_of(None) is None


def test_buckets_follow_the_substituted_fraction():
    assert aa.bucket_of(None) == "0%"
    assert aa.bucket_of({"declined": 0, "planned": 10}) == "0%"
    assert aa.bucket_of({"declined": 9, "planned": 10, "capped": True}) == "0%"   # read in full
    assert aa.bucket_of({"declined": 0.5, "planned": 10}) == "(0,10%)"
    assert aa.bucket_of({"declined": 3, "planned": 10}) == "[10,50%)"
    assert aa.bucket_of({"declined": 10, "planned": 10}) == "[50,100%]"
    assert aa.bucket_of({"declined": 3, "planned": 10, "failed": 1}) == "recomputed"


def test_join_pairs_each_prompt_with_its_ledger_entry_and_the_baseline(tmp_path):
    bff = [{"prompt": "p1", "generated_text": "a b c", "response_id": f"chatcmpl-{U1}"},
           {"prompt": "p2", "generated_text": "x y", "response_id": f"chatcmpl-{U2}"},
           {"prompt": "p3", "generated_text": "q", "response_id": None}]
    base = [{"prompt": "p1", "generated_text": "a b c"},
            {"prompt": "p2", "generated_text": "x y z"},
            {"prompt": "p3", "generated_text": "q"}]
    (tmp_path / "bff_stats_1.json").write_text(json.dumps(
        {"request_ledger": {f"chatcmpl-{U2}": {"declined": 8, "planned": 10, "capped": False,
                                               "failed": 0},
                            f"chatcmpl-{U3}": {"declined": 1, "planned": 10}}}))
    (tmp_path / "bff_stats_2.json").write_text(json.dumps({"request_ledger": {}}))
    ledger = aa.load_ledger(str(tmp_path))
    assert set(ledger) == {U2, U3}

    refs = {"p1": "a b c", "p2": "x y z", "p3": "q"}
    rows, unmatched = aa.join(bff, base, ledger, refs)
    assert unmatched == 1                                  # p3 has no response_id
    by_bucket = {bk: (a, b) for bk, a, b in rows}
    assert by_bucket["0%"] == (1.0, 1.0)                   # p1: not in the ledger, unchanged
    a, b = by_bucket["[50,100%]"]                          # p2: 80% substituted, lost a token
    assert a < b == 1.0
