#!/bin/bash
# Wiki patches 0005 (GLM chat template) + 0006/0011 (GLM reasoning parser), rebased to vLLM v0.19.1.
# Delivered as an idempotent string-replace script because the container's vLLM is a plain install
# (not a git tree we can format-patch against). Override target with VLLM_DIR (default /vllm-workspace/vllm).
#
# NOTE: These are GLM-model fixes. For Qwen (the current benchmark model) they are inert; included
# for completeness per request.
set -e
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHAT="${VLLM_DIR}/vllm/entrypoints/chat_utils.py"
REASON_INIT="${VLLM_DIR}/vllm/reasoning/__init__.py"
GLM_DST="${VLLM_DIR}/vllm/reasoning/glm_reasoning_parser.py"

for f in "$CHAT" "$REASON_INIT"; do
    [ -f "$f" ] || { echo "Error: $f not found (set VLLM_DIR)"; exit 1; }
done

# 1) Bundle the GLM reasoning parser (0006 + 0011 folded).
cp "${HERE}/glm_reasoning_parser.py" "$GLM_DST"
echo "Installed $GLM_DST"

CHAT="$CHAT" REASON_INIT="$REASON_INIT" python3 << 'PYTHON_EOF'
import os

# --- 0005: preserve string content for tool messages (chat_utils.py) ---
chat = os.environ["CHAT"]
with open(chat) as f:
    c = f.read()
old = ('            if "tool_call_id" in parsed_msg:\n'
       '                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]\n')
ins = ('            if "tool_call_id" in parsed_msg:\n'
       '                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]\n'
       '            # Preserve string content as-is for tool messages\n'
       '            # (don\'t apply OpenAI list-of-dicts transformation)\n'
       '            if isinstance(message.get("content"), str):\n'
       '                result_msg["content"] = message["content"]\n')
if "Preserve string content as-is for tool messages" in c:
    print("chat_utils.py: already patched")
elif old in c:
    with open(chat, "w") as f:
        f.write(c.replace(old, ins, 1))
    print("chat_utils.py: applied 0005")
else:
    print("chat_utils.py: WARNING anchor not found — apply 0005 manually")

# --- 0006: point glm45 reasoning parser at the new GLMReasoningParser ---
ri = os.environ["REASON_INIT"]
with open(ri) as f:
    c = f.read()
old = ('    "glm45": (\n'
       '        "deepseek_v3_reasoning_parser",\n'
       '        "DeepSeekV3ReasoningWithThinkingParser",\n'
       '    ),')
new = ('    "glm45": (\n'
       '        "glm_reasoning_parser",\n'
       '        "GLMReasoningParser",\n'
       '    ),')
if '"GLMReasoningParser"' in c:
    print("reasoning/__init__.py: already patched")
elif old in c:
    with open(ri, "w") as f:
        f.write(c.replace(old, new, 1))
    print("reasoning/__init__.py: applied 0006 mapping")
else:
    print("reasoning/__init__.py: WARNING anchor not found — apply 0006 manually")
PYTHON_EOF

echo "Done (GLM chat + reasoning)."
