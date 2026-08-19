# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated-serving proxy for the GPU Mooncake connector (nPmD).

Why a second proxy: the P2P-NCCL proxy pairs P and D by rewriting the *request id* with
``___prefill_addr_.../___decode_addr_...`` and discovers instances over ZMQ. Mooncake pairs them
completely differently — through ``kv_transfer_params`` and a per-prefill HTTP **bootstrap server**:

    1. the proxy mints a ``transfer_id`` and sends the prompt to a prefill with
       ``kv_transfer_params={"do_remote_decode": true, "transfer_id": ...}`` and ``max_tokens=1``;
    2. it then sends the same prompt to a decode with ``do_remote_prefill`` plus the chosen
       prefill's ``remote_engine_id`` and ``remote_bootstrap_addr``;
    3. the decode looks that engine up in the prefill's bootstrap server and **pulls** the KV over
       the Mooncake Transfer Engine. The proxy never touches KV.

So instances are configured statically here (there is no ZMQ registration to listen for): the
launch script knows each prefill's HTTP address, engine id and bootstrap address, and passes them
in via env. Requests are round-robined across prefills and decodes independently, so any n×m works.

Env:
    PREFILL_ADDRS        comma-separated ``host:port`` of the prefill HTTP servers
    PREFILL_ENGINE_IDS   comma-separated engine ids, SAME order (must match each server's
                         ``--kv-transfer-config`` ``engine_id``)
    PREFILL_BOOTSTRAPS   comma-separated ``http://host:port`` bootstrap addresses, SAME order
    DECODE_ADDRS         comma-separated ``host:port`` of the decode HTTP servers
    PROXY_HOST/PROXY_HTTP_PORT   where this proxy listens (default 0.0.0.0:10001)
"""

import itertools
import os
import sys
import traceback
import uuid

import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def _csv(name: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


PREFILL_ADDRS = _csv("PREFILL_ADDRS")
PREFILL_ENGINE_IDS = _csv("PREFILL_ENGINE_IDS")
PREFILL_BOOTSTRAPS = _csv("PREFILL_BOOTSTRAPS")
DECODE_ADDRS = _csv("DECODE_ADDRS")

if not PREFILL_ADDRS or not DECODE_ADDRS:
    raise SystemExit("PREFILL_ADDRS and DECODE_ADDRS must be set (comma-separated host:port).")
if not (len(PREFILL_ADDRS) == len(PREFILL_ENGINE_IDS) == len(PREFILL_BOOTSTRAPS)):
    raise SystemExit(
        f"PREFILL_ADDRS ({len(PREFILL_ADDRS)}), PREFILL_ENGINE_IDS "
        f"({len(PREFILL_ENGINE_IDS)}) and PREFILL_BOOTSTRAPS ({len(PREFILL_BOOTSTRAPS)}) "
        "must be the same length and in the same order.")

# Independent cycles: a 2P1D topology must keep both prefills busy against the one decode.
_prefill_rr = itertools.cycle(range(len(PREFILL_ADDRS)))
_decode_rr = itertools.cycle(range(len(DECODE_ADDRS)))

print(f"Mooncake disagg proxy: {len(PREFILL_ADDRS)}P x {len(DECODE_ADDRS)}D")
for a, e, b in zip(PREFILL_ADDRS, PREFILL_ENGINE_IDS, PREFILL_BOOTSTRAPS):
    print(f"  P {a}  engine_id={e}  bootstrap={b}")
for a in DECODE_ADDRS:
    print(f"  D {a}")


async def _post_and_drain(url: str, data: dict, request_id: str) -> None:
    """Run the prefill to completion and discard its body (only its KV matters)."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                   "X-Request-Id": request_id}
        async with session.post(url=url, json=data, headers=headers) as response:
            body = await response.read()
            if response.status != 200:
                print(f"prefill returned {response.status}: {body[:512]!r}")


async def _stream(url: str, data: dict, request_id: str):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                   "X-Request-Id": request_id}
        async with session.post(url=url, json=data, headers=headers) as response:
            async for chunk in response.content.iter_chunked(1024):
                yield chunk


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original = await request.get_json()
        transfer_id = uuid.uuid4().hex
        request_id = transfer_id

        p = next(_prefill_rr)
        d = next(_decode_rr)
        prefill_addr = PREFILL_ADDRS[p]
        decode_addr = DECODE_ADDRS[d]

        # ---- 1. prefill: one token, KV kept for the decode to pull -------------------
        prefill_request = dict(original)
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        # Force non-streaming: we only need the prefill to finish, and a streamed body would
        # just be parsed and thrown away.
        prefill_request["stream"] = False
        prefill_request.pop("stream_options", None)
        prefill_request["kv_transfer_params"] = {
            "do_remote_decode": True,
            "transfer_id": transfer_id,
        }
        await _post_and_drain(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id)

        # ---- 2. decode: pull that KV, then generate ---------------------------------
        decode_request = dict(original)
        decode_request["kv_transfer_params"] = {
            "do_remote_prefill": True,
            "transfer_id": transfer_id,
            "remote_engine_id": PREFILL_ENGINE_IDS[p],
            "remote_bootstrap_addr": PREFILL_BOOTSTRAPS[p],
        }
        response = await make_response(
            _stream(f"http://{decode_addr}{request.path}", decode_request, request_id))
        response.timeout = None
        return response

    except Exception as e:  # keep the proxy alive; one bad request must not kill the run
        print("Error occurred in the Mooncake disagg proxy")
        print(e)
        print("".join(traceback.format_exception(*sys.exc_info())))
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host=os.environ.get("PROXY_HOST", "0.0.0.0"),
            port=int(os.environ.get("PROXY_HTTP_PORT", "10001")))
