#!/bin/bash
# Apply the v0.19.1-rebased wiki patch set. See README.md.
#   Core P/D transfer fixes always; SSD (ENABLE_SSD=1) and GLM serving (ENABLE_GLM=1) optional.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
MOONCAKE_DIR="${MOONCAKE_DIR:-/vllm-workspace/Mooncake}"
ENABLE_SSD="${ENABLE_SSD:-0}"
ENABLE_GLM="${ENABLE_GLM:-0}"

echo "== vllm-ascend core (P/D transfer) =="
git -C "$VLLM_ASCEND_DIR" am \
    "$HERE"/vllm-ascend/0001-*.patch \
    "$HERE"/vllm-ascend/0002-*.patch \
    "$HERE"/vllm-ascend/0003-*.patch

echo "== vLLM loggers guard =="
VLLM_DIR="$VLLM_DIR" bash "$HERE/vllm/apply_fix.sh"

echo "== vLLM p2p stable request id + chunked continuation (local fix, required for P/D) =="
VLLM_DIR="$VLLM_DIR" bash "$HERE/vllm/apply_p2p_stable_id.sh"

if [ "$ENABLE_GLM" = "1" ]; then
    echo "== vLLM GLM chat + reasoning (optional) =="
    VLLM_DIR="$VLLM_DIR" bash "$HERE/vllm/apply_glm_and_chat.sh"
fi

if [ "$ENABLE_SSD" = "1" ]; then
    echo "== Mooncake SSD metrics (optional — rebuild Mooncake after this) =="
    git -C "$MOONCAKE_DIR" am "$HERE"/mooncake/0001-*.patch
    echo "== vllm-ascend SSD offload (optional) =="
    git -C "$VLLM_ASCEND_DIR" am \
        "$HERE"/vllm-ascend/0004-*.patch \
        "$HERE"/vllm-ascend/0005-*.patch
fi

echo "Done. Restart the P/D servers. (If a hunk rejected, retry that repo with 'git am -3'.)"
