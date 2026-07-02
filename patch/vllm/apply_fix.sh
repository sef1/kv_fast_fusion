#!/bin/bash
# Wiki patch: wrap Prometheus counter .inc() calls in loggers.py with try/except
# ValueError (avoids a hard crash when a counter receives a value it rejects).
# Verified to match vLLM v0.19.1 verbatim. Idempotent (self-detects prior patch).
# Override the target with VLLM_DIR (default /vllm-workspace/vllm).

VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
LOGGERS_PATH="${VLLM_DIR}/vllm/v1/metrics/loggers.py"

if [ ! -f "$LOGGERS_PATH" ]; then
    echo "Error: $LOGGERS_PATH not found!"
    exit 1
fi

if [ ! -f "${LOGGERS_PATH}.bak2" ]; then
    cp "$LOGGERS_PATH" "${LOGGERS_PATH}.bak2"
    echo "Created backup at ${LOGGERS_PATH}.bak2"
fi

LOGGERS_PATH="$LOGGERS_PATH" python3 << 'PYTHON_EOF'
import os

LOGGERS_PATH = os.environ["LOGGERS_PATH"]

with open(LOGGERS_PATH) as f:
    content = f.read()

if "Failed to increment prompt_tokens" in content:
    print("File appears to already be patched!")
    raise SystemExit(0)

old_prompt = '''        for source in PromptTokenStats.ALL_SOURCES:
            self.counter_prompt_tokens_by_source[source][engine_idx].inc(
                pts.get_by_source(source)
            )'''
new_prompt = '''        for source in PromptTokenStats.ALL_SOURCES:
            try:
                self.counter_prompt_tokens_by_source[source][engine_idx].inc(
                    pts.get_by_source(source))
            except ValueError as e:
                logger.warning(
                    "Failed to increment prompt_tokens counter for "
                    "source %s, engine %d: %s", source, engine_idx, e)'''

old_prompt_tokens = '''        self.counter_prompt_tokens[engine_idx].inc(iteration_stats.num_prompt_tokens)'''
new_prompt_tokens = '''        try:
            self.counter_prompt_tokens[engine_idx].inc(iteration_stats.num_prompt_tokens)
        except ValueError as e:
            logger.warning(
                "Failed to increment counter_prompt_tokens for engine %d: %s",
                engine_idx, e)'''

old_gen_tokens = '''        self.counter_generation_tokens[engine_idx].inc(
            iteration_stats.num_generation_tokens
        )'''
new_gen_tokens = '''        try:
            self.counter_generation_tokens[engine_idx].inc(
                iteration_stats.num_generation_tokens)
        except ValueError as e:
            logger.warning(
                "Failed to increment counter_generation_tokens for engine %d: %s",
                engine_idx, e)'''

for old, new, label in (
    (old_prompt, new_prompt, "prompt_tokens_by_source"),
    (old_prompt_tokens, new_prompt_tokens, "counter_prompt_tokens"),
    (old_gen_tokens, new_gen_tokens, "counter_generation_tokens"),
):
    if old in content:
        content = content.replace(old, new)
        print(f"Applied fix for {label}")
    else:
        print(f"Warning: Could not find {label} pattern")

with open(LOGGERS_PATH, "w") as f:
    f.write(content)

print("\nSuccessfully applied the loggers fix!")
PYTHON_EOF
