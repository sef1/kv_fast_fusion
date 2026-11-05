#!/usr/bin/env python3
"""Run vllm-ascend's P/D proxy with evicted answers continued instead of restarted.

When the decode evicts a request it stops with ``stop_reason="recomputed"`` and the proxy resends the
request. For chat requests the stock proxy pastes the partial answer into the USER message, so to the
model the answer has not begun: it starts again from the top, and the client receives both pieces glued
together (``__tablename__ =```python``). About 12% of answers on the families scenario, in every arm,
F1 0.51 against 0.64.

This sends the partial as an unfinished assistant message with ``continue_final_message``. The chat
template strips trailing whitespace from it, so the model emits that whitespace again; it is removed from
the resumed stream because the client already has it. If the prefill refuses the continuation, the request
falls back to the stock behaviour rather than failing.

The proxy file on the box is patched in memory, so this tracks whatever version is installed. If an
expected line is missing this stops with an error instead of running an unpatched proxy under this name.

    python proxy_continue_answer.py STOCK_PROXY.py [proxy args...]
"""

import sys

EDITS = [
    # State for one client request.
    ('            generated_token = ""\n',
     '            generated_token = ""\n'
     '            strip_resume = False\n'),
    # The model repeats whitespace the template stripped from the partial; the client already has it.
    ('                        generated_token += content\n',
     '                        rewritten = False\n'
     '                        if strip_resume and content:\n'
     '                            kept = content.lstrip()\n'
     '                            strip_resume = not kept\n'
     '                            if kept != content:\n'
     '                                rewritten = True\n'
     '                                content = kept\n'
     '                                for holder, key in ((delta, "content"), (message, "content"), (choice, "text")):\n'
     '                                    if holder.get(key):\n'
     '                                        holder[key] = kept\n'
     '                        generated_token += content\n'),
    # Continue the answer. Nothing generated yet means there is nothing to continue.
    ('                                messages[0]["content"] = origin_prompt + generated_token\n',
     '                                continuing = bool(generated_token.strip())\n'
     '                                if continuing:\n'
     '                                    req_data["messages"] = list(messages) + [\n'
     '                                        {"role": "assistant", "content": generated_token}]\n'
     '                                    req_data["add_generation_prompt"] = False\n'
     '                                    req_data["continue_final_message"] = True\n'
     '                                    strip_resume = generated_token != generated_token.rstrip()\n'
     '                                    print(f"BFF proxy continued answer | partial_chars="\n'
     '                                          f"{len(generated_token)} retry={retry_count}", flush=True)\n'),
    # A continuation the prefill rejects must not cost the request.
    ('                            instance_info = await _handle_select_instance(api, req_data, tmp_request_length)\n'
     '                            break\n',
     '                            try:\n'
     '                                instance_info = await _handle_select_instance(api, req_data, tmp_request_length)\n'
     '                            except httpx.HTTPStatusError as e:\n'
     '                                if not req_data.pop("continue_final_message", False):\n'
     '                                    raise\n'
     '                                req_data.pop("add_generation_prompt", None)\n'
     '                                strip_resume = False\n'
     '                                messages[0]["content"] = origin_prompt + generated_token\n'
     '                                req_data["messages"] = messages\n'
     '                                print(f"BFF proxy continuation refused, restarting | {e}", flush=True)\n'
     '                                instance_info = await _handle_select_instance(api, req_data, tmp_request_length)\n'
     '                            break\n'),
    # A rewritten chunk has to be serialized again.
    ('                            chunk = json.dumps(chunk_json).encode("utf-8")\n'
     '                        yield chunk\n',
     '                            chunk = json.dumps(chunk_json).encode("utf-8")\n'
     '                        elif rewritten and stream_flag:\n'
     '                            chunk = ("data: " + json.dumps(chunk_json) + "\\n\\n").encode("utf-8")\n'
     '                        elif rewritten:\n'
     '                            chunk = json.dumps(chunk_json).encode("utf-8")\n'
     '                        yield chunk\n'),
]


def patched(source: str) -> str:
    """The proxy's source with every edit applied. Raises if any anchor is not there exactly once."""
    for old, new in EDITS:
        found = source.count(old)
        if found != 1:
            raise SystemExit(f"proxy_continue_answer: expected this line once, found {found}. The installed "
                             f"proxy differs, so it was not started:\n{old}")
        source = source.replace(old, new)
    return source


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        code = compile(patched(f.read()), path, "exec")
    sys.argv = [path] + sys.argv[2:]
    print("BFF proxy: evicted answers are continued, not restarted", flush=True)
    exec(code, {"__name__": "__main__", "__file__": path})


if __name__ == "__main__":
    main()
