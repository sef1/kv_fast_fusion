"""PROXY_REORDER: the proxy holds requests before prefill and releases the one that shares most with what is
already there, without starving anyone, and lets evicted requests straight back."""

import asyncio
import glob
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "examples/online_serving/disaggregated_serving_mooncake_ascend_xpyd")
spec = importlib.util.spec_from_file_location("bff_proxy_reorder", os.path.join(DIR, "bff_proxy_reorder.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


def keys(*blocks):
    return [(s, b) for s, b in enumerate(blocks)]


def run(coro):
    return asyncio.run(coro)


def test_block_keys_are_full_blocks_at_their_positions():
    ids = list(range(300))
    k = pr.block_keys(ids)
    assert len(k) == 2 and [p for p, _ in k] == [0, 1]
    assert pr.block_keys(list(range(128)) + [9] * 172)[0] == k[0], "same tokens, same position, same key"
    assert pr.block_keys([5] + list(range(127)))[0] != k[0]


def test_only_slots_requests_are_at_the_prefill_at_once():
    async def go():
        g = pr.Gate(slots=2, log=lambda m: None)
        tasks = [asyncio.create_task(g.enter(keys(i))) for i in range(5)]
        await asyncio.sleep(0)
        assert sum(t.done() for t in tasks) == 2 and len(g.waiting) == 3
        g.done()
        await asyncio.sleep(0)
        assert sum(t.done() for t in tasks) == 3
    run(go())


def test_the_held_request_that_shares_most_goes_first_and_the_others_count_a_skip():
    async def go():
        g = pr.Gate(slots=1, max_skip=5, log=lambda m: None)
        g.add_present(keys("p0", "p1", "p2"))           # a request decoding now
        first = asyncio.create_task(g.enter(keys("x")))  # takes the only slot
        await asyncio.sleep(0)
        b = asyncio.create_task(g.enter(keys("y0", "y1")))           # older, shares nothing
        c = asyncio.create_task(g.enter(keys("p0", "p1", "z")))      # newer, shares two blocks
        await asyncio.sleep(0)
        assert first.done() and not b.done() and not c.done()
        g.done()
        await asyncio.sleep(0)
        assert c.done() and not b.done()
        assert g.waiting[0]["skips"] == 1
    run(go())


def test_with_nothing_to_share_the_order_is_first_come_first_served():
    async def go():
        g = pr.Gate(slots=1, log=lambda m: None)
        order = []

        async def req(name):
            await g.enter(keys(name))
            order.append(name)

        tasks = [asyncio.create_task(req(n)) for n in "abcd"]
        for _ in range(4):
            await asyncio.sleep(0)
            g.done()
        await asyncio.gather(*tasks)
        assert order == list("abcd")
    run(go())


def test_a_request_passed_over_max_skip_times_goes_next_whatever_matches():
    async def go():
        g = pr.Gate(slots=1, max_skip=1, log=lambda m: None)
        g.add_present(keys("p"))
        hold = asyncio.create_task(g.enter(keys("x")))
        await asyncio.sleep(0)
        old = asyncio.create_task(g.enter(keys("q")))
        m1 = asyncio.create_task(g.enter(keys("p")))
        m2 = asyncio.create_task(g.enter(keys("p")))
        await asyncio.sleep(0)
        g.done()                                    # m1 matches: old skipped once
        await asyncio.sleep(0)
        assert m1.done() and not old.done()
        g.done()                                    # old has hit the cap: it goes, although m2 matches
        await asyncio.sleep(0)
        assert old.done() and not m2.done()
        assert g.stats["forced"] == 1
    run(go())


class FakeTok:
    """Token ids are the words of the message; enough to give blocks real positions."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        return [hash(w) % 50000 for w in " ".join(m["content"] for m in messages).split()]

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [hash(w) % 50000 for w in text.split()]
        return R()


def body(words):
    return {"messages": [{"role": "user", "content": " ".join(words)}]}


def test_install_holds_new_requests_lets_evicted_ones_through_and_balances_presence():
    async def go():
        g = pr.Gate(slots=1, log=lambda m: None)
        prefill_started = []
        release_prefill = asyncio.Event()

        async def select(api, req_data, n):
            prefill_started.append(id(req_data))
            await release_prefill.wait()
            return "instance"

        async def stream(client, api, req_data, **kw):
            yield b"chunk"

        sel, strm = pr.install(select, stream, tokenizer=FakeTok(), gate=g)
        a, b = body(f"w{i}" for i in range(300)), body(f"v{i}" for i in range(300))
        ta = asyncio.create_task(sel("/v1", a, 0))
        tb = asyncio.create_task(sel("/v1", b, 0))
        for _ in range(20):
            await asyncio.sleep(0.01)
        assert prefill_started == [id(a)], "one slot: the second request is held"
        release_prefill.set()
        await asyncio.gather(ta, tb)
        assert prefill_started == [id(a), id(b)]
        for req in (a, b):                          # both decode and finish
            async for _ in strm(None, "/v1", req):
                pass
        assert not +g.present, "every block added was removed again"
        # a is evicted and comes back: straight to prefill even with the slot taken
        g.at_prefill = g.slots
        prefill_started.clear()
        await asyncio.wait_for(sel("/v1", a, 0), timeout=1)
        assert prefill_started == [id(a)]
        async for _ in strm(None, "/v1", a):
            pass
        assert not +g.present
    run(go())


def test_a_failed_prefill_does_not_leave_its_blocks_present():
    async def go():
        g = pr.Gate(slots=1, log=lambda m: None)

        async def select(api, req_data, n):
            raise RuntimeError("prefill refused")

        async def stream(client, api, req_data, **kw):
            yield b""

        sel, _ = pr.install(select, stream, tokenizer=FakeTok(), gate=g)
        req = body(f"w{i}" for i in range(300))
        for _ in range(2):                          # the first call, then a retry of the same request
            with pytest.raises(RuntimeError):
                await sel("/v1", req, 0)
        assert not +g.present and g.at_prefill == 0
    run(go())


# ------------------------------------------------------------------------------ the real stock proxy
TOK = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))


@pytest.mark.skipif(not TOK, reason="no cached tokenizer")
def test_the_stock_proxy_runs_with_reordering_and_still_continues_evicted_answers(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_proxy_continue_answer as t

    if not os.path.exists(t.STOCK):
        pytest.skip("no vllm-ascend checkout")
    monkeypatch.setenv("PROXY_REORDER", "1")
    monkeypatch.setenv("PROXY_REORDER_TOKENIZER", TOK[-1])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    backend = t.Backend([[("aaa ", None), ("", "recomputed")], [(" bbb", "stop")]])
    try:
        gen = t.run_proxy.__wrapped__(tmp_path) if hasattr(t.run_proxy, "__wrapped__") else None
        start = next(gen) if gen else None
        port = start(backend)
        assert t._ask(port) == "aaa bbb"
        log = (tmp_path / "proxy.txt").read_text()
        assert "prefill reordering on" in log
        assert log.count("BFF proxy reorder | released") == 1, "the evicted request is not held a second time"
        assert len(backend.prefill_bodies) == 2, "the evicted request went back to prefill once"
        assert backend.prefill_bodies[1]["messages"][-1] == {"role": "assistant", "content": "aaa "}
    finally:
        backend.close()
        if gen:
            gen.close()
