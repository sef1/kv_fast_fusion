#!/usr/bin/env bash
# Build a test environment on the vLLM version the Ascend box runs. Takes about 25 minutes the first
# time, because vLLM pulls torch and CUDA. The NPU-only code paths still do not run here.
#
#   bash kv_fast_fusion/tools/make_test_env.sh
#   .venv-vllm018/bin/python -m pytest -q kv_fast_fusion kv_fast_fusion_ascend f1_benchmark
set -euo pipefail

VENV=${VENV:-.venv-vllm018}
uv venv --python 3.12 "$VENV"
# msgpack: vLLM's p2p engine imports it without declaring it.
uv pip install --python "$VENV/bin/python" "vllm==0.18.0" pytest msgpack
