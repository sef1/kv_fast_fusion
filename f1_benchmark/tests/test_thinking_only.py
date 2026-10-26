"""Reasoning models stream their thinking as `reasoning_content`; F1 scores `content` only.

On GLM-5.1 (served with --reasoning-parser glm45) 8 of 60 baseline and 17 of 60 BFF requests spent
all of max_tokens thinking, returned empty `content`, and were reported as plain failures. These tests
pin that the stream parser tells the two apart. CPU only, no server.
"""

import asyncio
import json

from f1_benchmark.f1_main import _parse_stream


class _FakeResponse:
    def __init__(self, deltas, finish="length"):
        lines = []
        for i, d in enumerate(deltas):
            choice = {"delta": d}
            if i == len(deltas) - 1:
                choice["finish_reason"] = finish
            lines.append(("data: " + json.dumps({"choices": [choice]}) + "\n").encode())
        lines.append(b"data: [DONE]\n")
        self.content = _Lines(lines)


class _Lines:
    def __init__(self, lines):
        self._lines = lines

    def __aiter__(self):
        self._it = iter(self._lines)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _parse(deltas, finish="length"):
    return asyncio.run(_parse_stream(_FakeResponse(deltas, finish), 0.0))


def test_a_stream_of_only_thinking_is_thinking_only_with_no_answer():
    rec = _parse([{"reasoning_content": "Let me think"}, {"reasoning_content": " more..."}])
    assert rec["generated_text"] == ""
    assert rec["thinking_only"] is True
    assert rec["reasoning_chars"] == len("Let me think more...")


def test_thinking_then_an_answer_keeps_the_answer_and_is_not_thinking_only():
    rec = _parse([{"reasoning_content": "plan"}, {"content": "def f(): pass"}], finish="stop")
    assert rec["generated_text"] == "def f(): pass"
    assert rec["thinking_only"] is False


def test_a_plain_model_is_unaffected():
    rec = _parse([{"content": "hello"}], finish="stop")
    assert rec["generated_text"] == "hello" and rec["reasoning_chars"] == 0
    assert rec["thinking_only"] is False


def test_an_empty_stream_is_not_mistaken_for_thinking():
    rec = _parse([{}])
    assert rec["generated_text"] == "" and rec["thinking_only"] is False
