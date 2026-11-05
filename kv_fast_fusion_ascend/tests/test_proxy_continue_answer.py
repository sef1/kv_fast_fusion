"""An evicted answer must be continued, not restarted: the real proxy against fake P and D servers."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WRAPPER = os.path.join(ROOT, "examples/online_serving/disaggregated_serving_mooncake_ascend_xpyd",
                       "proxy_continue_answer.py")
STOCK = os.path.join(ROOT, "vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py")

spec = importlib.util.spec_from_file_location("proxy_continue_answer", WRAPPER)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def test_a_changed_proxy_stops_instead_of_running_unpatched():
    with pytest.raises(SystemExit, match="installed proxy differs"):
        pc.patched("import json\n")


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Backend:
    """One fake prefill and one fake decode. ``script`` is the decode's streams, one per request."""

    def __init__(self, script, refuse_continuation=False):
        self.script, self.refuse = list(script), refuse_continuation
        self.prefill_bodies, self.decode_bodies = [], []
        self.pport, self.dport = _port(), _port()
        self.servers = [self._serve(self._prefill_app(), self.pport), self._serve(self._decode_app(), self.dport)]

    def _serve(self, app, port):
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        while not srv.started:
            time.sleep(0.01)
        return srv

    def _prefill_app(self):
        app = fastapi.FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(req: fastapi.Request):
            body = await req.json()
            self.prefill_bodies.append(body)
            if self.refuse and body.get("continue_final_message"):
                return fastapi.responses.JSONResponse({"error": "no"}, status_code=400)
            return {"kv_transfer_params": {"remote_engine_id": "p"}, "choices": []}

        return app

    def _decode_app(self):
        app = fastapi.FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(req: fastapi.Request):
            self.decode_bodies.append(await req.json())
            pieces = self.script.pop(0)

            async def stream():
                for text, stop in pieces:
                    yield ("data: " + json.dumps({"choices": [{"delta": {"content": text}, "stop_reason": stop}]})
                           + "\n\n").encode()
                yield b"data: [DONE]\n\n"

            return fastapi.responses.StreamingResponse(stream(), media_type="text/event-stream")

        return app

    def close(self):
        for s in self.servers:
            s.should_exit = True


@pytest.fixture
def run_proxy(tmp_path):
    if not os.path.exists(STOCK):
        pytest.skip("no vllm-ascend checkout")
    procs = []

    def start(backend):
        port = _port()
        log = open(tmp_path / "proxy.txt", "w")
        p = subprocess.Popen(
            [sys.executable, WRAPPER, STOCK, "--prefiller-hosts", "127.0.0.1", "--prefiller-ports", str(backend.pport),
             "--decoder-hosts", "127.0.0.1", "--decoder-ports", str(backend.dport), "--host", "127.0.0.1",
             "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
        procs.append(p)
        for _ in range(400):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return port
            except OSError:
                if p.poll() is not None:
                    raise RuntimeError((tmp_path / "proxy.txt").read_text())
                time.sleep(0.05)
        raise RuntimeError("proxy did not start")

    yield start
    for p in procs:
        p.kill()


def _ask(port):
    body = {"model": "m", "stream": True, "max_tokens": 50, "messages": [{"role": "user", "content": "Write it"}]}
    out = []
    with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=20) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                d = json.loads(line[6:])["choices"][0]
                if d.get("stop_reason") != "recomputed":
                    out.append(d["delta"].get("content") or "")
                else:
                    out.append(d["delta"].get("content") or "")
    return "".join(out)


def test_the_partial_answer_is_continued_not_pasted_into_the_question(run_proxy):
    b = Backend([[("def f():", None), ("\n    ", None), ("", "recomputed")],
                 [("\n    return", None), (" 1", None), ("\n```", "stop")]])
    try:
        text = _ask(run_proxy(b))
    finally:
        b.close()
    second = b.prefill_bodies[1]
    assert second["messages"] == [{"role": "user", "content": "Write it"},
                                  {"role": "assistant", "content": "def f():\n    "}]
    assert second["continue_final_message"] is True and second["add_generation_prompt"] is False
    assert b.decode_bodies[1]["continue_final_message"] is True
    assert text == "def f():\n    return 1\n```", "the whitespace the model repeats is not sent twice"


def test_a_second_eviction_continues_from_all_of_it(run_proxy):
    b = Backend([[("aaa ", None), ("", "recomputed")], [(" bbb", None), ("", "recomputed")], [(" ccc", "stop")]])
    try:
        text = _ask(run_proxy(b))
    finally:
        b.close()
    assert b.prefill_bodies[2]["messages"][-1]["content"] == "aaa bbb"
    assert len(b.prefill_bodies[2]["messages"]) == 2, "one assistant message, not one per eviction"
    assert text == "aaa bbb ccc"


def test_an_eviction_before_any_text_just_resends_the_question(run_proxy):
    b = Backend([[("", "recomputed")], [("hello", "stop")]])
    try:
        text = _ask(run_proxy(b))
    finally:
        b.close()
    assert b.prefill_bodies[1]["messages"] == [{"role": "user", "content": "Write it"}]
    assert "continue_final_message" not in b.prefill_bodies[1]
    assert text == "hello"


def test_a_refused_continuation_falls_back_and_keeps_the_request(run_proxy):
    b = Backend([[("abc", None), ("", "recomputed")], [("xyz", "stop")]], refuse_continuation=True)
    try:
        text = _ask(run_proxy(b))
    finally:
        b.close()
    last = b.prefill_bodies[-1]
    assert "continue_final_message" not in last and last["messages"] == [{"role": "user", "content": "Write itabc"}]
    assert text == "abcxyz", "the request still completes"
