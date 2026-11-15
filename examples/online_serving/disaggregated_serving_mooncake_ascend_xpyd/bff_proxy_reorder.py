"""Similarity-aware prefill order for the P/D proxy (PROXY_REORDER=1).

The stock proxy forwards every request to prefill at once, so the queue lives inside the prefill node and
runs first come, first served. With this on, the proxy keeps at most PROXY_REORDER_PREFILL_SLOTS requests at
the prefill and holds the rest. When a slot frees, it sends the held request that shares the most 128-token
blocks, at the same positions, with requests now decoding or at the prefill: those it can share KV with under
BFF. A held request passed over PROXY_REORDER_MAX_SKIP times goes next, so none waits forever. A request coming
back after an eviction is let straight through.

It is arm-neutral: it changes only the order requests reach the prefill, so run the baseline with it too.
Queue simulation (families unsorted, limit 24): sharing 10% -> 16%, BFF about +3% over a reordered baseline.

Installed by proxy_continue_answer.py, which wraps the two stock functions:
  _handle_select_instance            sends a request to prefill; returns when its prefill is done
  stream_service_response_with_retry the decode stream; while it runs, the request is decoding
"""

import asyncio
import collections
import itertools
import os
import time

BLOCK = 128


def block_keys(token_ids) -> list:
    """(position, block content) for each full block: what BFF can match at the same position."""
    return [(s, hash(tuple(token_ids[s * BLOCK:(s + 1) * BLOCK]))) for s in range(len(token_ids) // BLOCK)]


class Gate:
    """The hold queue. Pure asyncio, no I/O: keys go in, release order comes out."""

    def __init__(self, slots: int = 4, max_skip: int = 20, log=print):
        self.slots, self.max_skip, self.log = slots, max_skip, log
        self.waiting = []                       # entries in arrival order
        self.at_prefill = 0
        self.present = collections.Counter()    # block keys of requests at the prefill or decoding
        self.seq = itertools.count()
        self.stats = collections.Counter()
        self.waits = []

    def _score(self, keys) -> int:
        return sum(1 for k in keys if self.present[k])

    def add_present(self, keys):
        self.present.update(keys)

    def remove_present(self, keys):
        self.present.subtract(keys)
        for k in keys:
            if self.present[k] <= 0:
                del self.present[k]

    async def enter(self, keys):
        """Wait until this request may go to prefill. Returns once released; call done() after prefill."""
        entry = {"seq": next(self.seq), "keys": keys, "skips": 0, "t": time.monotonic(),
                 "fut": asyncio.get_running_loop().create_future()}
        self.waiting.append(entry)
        self._dispatch()
        await entry["fut"]
        return entry

    def done(self):
        """One request finished its prefill: free its slot."""
        self.at_prefill -= 1
        self._dispatch()

    def _pick(self):
        oldest = self.waiting[0]
        if oldest["skips"] >= self.max_skip:
            self.stats["forced"] += 1
            return oldest, self._score(oldest["keys"])
        scored = [(self._score(e["keys"]), -e["seq"], e) for e in self.waiting]
        score, _, best = max(scored, key=lambda x: (x[0], x[1]))
        if score == 0:
            return oldest, 0
        for e in self.waiting:
            if e["seq"] < best["seq"]:
                e["skips"] += 1
        return best, score

    def _dispatch(self):
        while self.waiting and self.at_prefill < self.slots:
            entry, score = self._pick()
            self.waiting.remove(entry)
            self.at_prefill += 1
            self.add_present(entry["keys"])
            waited = time.monotonic() - entry["t"]
            self.waits.append(waited)
            self.stats["released"] += 1
            self.stats["by_score"] += score > 0
            self.log(f"BFF proxy reorder | released score={score}/{len(entry['keys'])} skips={entry['skips']} "
                     f"waited={waited:.1f}s held={len(self.waiting)} at_prefill={self.at_prefill}")
            if self.stats["released"] % 50 == 0:
                w = sorted(self.waits)
                self.log(f"BFF proxy reorder summary | released={self.stats['released']} "
                         f"chosen_by_similarity={self.stats['by_score']} forced_by_skip_cap={self.stats['forced']} "
                         f"wait p50={w[len(w) // 2]:.1f}s max={w[-1]:.1f}s")
            entry["fut"].set_result(None)


def _tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def install(handle_select_instance, stream_service_response_with_retry, *, tokenizer=None, gate=None):
    """Wrap the stock proxy's two functions. Returns the replacements, in the same order."""
    path = os.environ.get("PROXY_REORDER_TOKENIZER") or os.environ.get("MODEL")
    tok = tokenizer or _tokenizer(path)
    gate = gate or Gate(int(os.environ.get("PROXY_REORDER_PREFILL_SLOTS", "4")),
                        int(os.environ.get("PROXY_REORDER_MAX_SKIP", "20")),
                        log=lambda m: print(m, flush=True))
    known = {}      # id(req_data) -> (req_data, keys); req_data kept alive so its id is not reused
    print(f"BFF proxy: prefill reordering on (slots={gate.slots}, max_skip={gate.max_skip}, tokenizer={path})",
          flush=True)

    def tokens(req_data):
        if "messages" in req_data:
            kw = req_data.get("chat_template_kwargs") or {}
            return tok.apply_chat_template(req_data["messages"], tokenize=True, add_generation_prompt=True, **kw)
        return tok(req_data.get("prompt") or "", add_special_tokens=False).input_ids

    # Presence is counted: every select adds the request's blocks once (at release, or on a retry), and every
    # decode stream that ends removes them once. The stock proxy breaks out of a stream on an eviction and
    # retries at once, while the old stream's cleanup runs a moment later; counting keeps that order harmless.
    async def gated_select(api, req_data, request_length):
        item = known.get(id(req_data))
        if item is None:                        # a new request: hold it until the gate releases it
            keys = block_keys(await asyncio.to_thread(tokens, req_data))
            known[id(req_data)] = (req_data, keys)
            await gate.enter(keys)              # release adds its blocks
            try:
                return await handle_select_instance(api, req_data, request_length)
            except BaseException:
                gate.remove_present(keys)
                raise
            finally:
                gate.done()
        keys = item[1]                          # back after an eviction: no holding
        gate.add_present(keys)
        try:
            return await handle_select_instance(api, req_data, request_length)
        except BaseException:
            gate.remove_present(keys)
            raise

    async def tracked_stream(client, api, req_data, *args, **kwargs):
        try:
            async for chunk in stream_service_response_with_retry(client, api, req_data, *args, **kwargs):
                yield chunk
        finally:
            item = known.get(id(req_data))
            if item is not None:
                gate.remove_present(item[1])

    return gated_select, tracked_stream
