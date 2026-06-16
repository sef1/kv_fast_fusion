#!/bin/bash

# =============================================================================
# BFF (KV-Cache Fast Fusion) Disaggregated Serving - P2P NCCL XpYd
# =============================================================================
# Specialized version of disagg_example_p2p_nccl_xpyd.sh for this fork's
# Fast-Fusion P/D setup (plan ROUND 20-24). Differences from the stock example:
#
#   * Servers launch via `kv_fast_fusion.fast_fusion_main serve` (NOT `vllm
#     serve`) so kv_fast_fusion/__init__.py runs the BFF patches AND registers
#     the group-aware `P2pNcclConnectorFF` connector.
#   * `--kv_connector` is `P2pNcclConnectorFF` (per-KV-cache-group block tables;
#     stock P2pNcclConnector is single-group → corrupts BFF fusion-layer KV).
#   * `--block-size 128` (required by the BFF algorithm) + `--enable-prefix-caching`.
#   * Hybrid KV-cache manager forced on (BFF is genuinely hybrid: warmup
#     SlidingWindow group + fusion FullAttention groups).
#   * BFF env: BFF_SCALE_MODE=raw (transfer-safe, no KV mutation/scales) and
#     BFF_PD_FUSE=1 (connector-level layer-streamed fusion, sharing propagated to D).
#
# Topology (default 1P1D): proxy (HTTP 10001 / ZMQ 30001) -> 1 prefill -> 1 decode.
# The benchmark targets the proxy HTTP port (10001), never a server directly
# (a direct hit lacks the proxy-injected ___decode_addr_ and skips transfer).
#
# Override via env vars:
#   MODEL, PREFILL_GPUS, DECODE_GPUS, PREFILL_PORTS, DECODE_PORTS, PROXY_PORT,
#   KV_IP, BFF_SCALE_MODE, BFF_PD_FUSE, BFF_GROUP_SIZE, TIMEOUT_SECONDS
# =============================================================================

# ---- Model / topology --------------------------------------------------------
MODEL=${MODEL:-zai-org/glm-4-9b-chat-hf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}      # ZMQ service-discovery port (matches proxy_port in connector cfg)
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10001}   # proxy HTTP serving port (benchmark target)
KV_IP=${KV_IP:-10.10.10.174}

# Default 1P1D (one prefill GPU, one decode GPU)
PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-20003}
DECODE_PORTS=${DECODE_PORTS:-20005}

# ---- BFF knobs ---------------------------------------------------------------
BFF_SCALE_MODE=${BFF_SCALE_MODE:-raw}   # raw is required for P/D (no KV mutation, no scales to ship)
BFF_PD_FUSE=${BFF_PD_FUSE:-1}           # connector-level fusion + redirect propagation to D
BFF_GROUP_SIZE=${BFF_GROUP_SIZE:-4}     # fusion layers packed per KV cache group

# ---- GPU memory / recv-buffer tuning -----------------------------------------
# P only SENDS, so its recv-buffer threshold (kv_buffer_size) can be tiny (1e1).
# D's kv_buffer_size IS the threshold for GPU-RESIDENT recv KV before it spills to
# the CPU pool. It MUST be < D's free GPU memory after the KV cache, or D OOMs on
# the recv torch.empty (the "Peer Out Of Memory/Threshold, response:1" on P).
PREFILL_GPU_UTIL=${PREFILL_GPU_UTIL:-0.9}
DECODE_GPU_UTIL=${DECODE_GPU_UTIL:-0.6}     # headroom for transient recv buffers / NCCL / CPU-pool staging
PREFILL_KV_BUFFER=${PREFILL_KV_BUFFER:-1e1} # producer never receives → tiny
DECODE_KV_BUFFER=${DECODE_KV_BUFFER:-4e9}   # spill to CPU pool early; keep GPU-resident recv small

# ---- Required BFF / HF environment (CLAUDE.md) -------------------------------
REPO_ROOT=${REPO_ROOT:-/data/users/sefi/from_git/vllm_013/vllm}
export HF_HOME=${HF_HOME:-/data/models/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/data/models/huggingface/hub}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_V1=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

echo "Warning: P2P NCCL disaggregated prefill XpYd for vLLM v1 is experimental."
echo ""
echo "BFF Disaggregated Configuration:"
echo "  Model:        $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS  (KV ports 21001+)"
echo "  Decode GPUs:  $DECODE_GPUS, Ports: $DECODE_PORTS  (KV ports 22001+)"
echo "  Proxy:        HTTP $PROXY_HTTP_PORT / ZMQ $PROXY_PORT   KV_IP $KV_IP"
echo "  Connector:    P2pNcclConnectorFF"
echo "  BFF:          SCALE_MODE=$BFF_SCALE_MODE  PD_FUSE=$BFF_PD_FUSE  GROUP_SIZE=$BFF_GROUP_SIZE"
echo "  GPU util:     P=$PREFILL_GPU_UTIL  D=$DECODE_GPU_UTIL    kv_buffer: P=$PREFILL_KV_BUFFER  D=$DECODE_KV_BUFFER"
echo ""

PIDS=()

# Switch to the directory of the current script (so the proxy file is found).
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    if [[ ! -f "disagg_proxy_p2p_nccl_xpyd.py" ]]; then
        echo "Required file disagg_proxy_p2p_nccl_xpyd.py not found in $(pwd)"
        exit 1
    fi
}

check_num_gpus() {
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    fi
    echo "Found $num_gpus GPUs."
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM
    pkill -9 -f "disagg_proxy_p2p_nccl_xpyd.py"
    kill -- -$$
    wait
    exit 0
}

wait_for_server() {
    local port=$1
    local start_time=$(date +%s)
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s "localhost:${port}/v1/completions" > /dev/null; then
            echo "Server on port $port is ready."
            return 0
        fi
        local now=$(date +%s)
        if (( now - start_time >= TIMEOUT_SECONDS )); then
            echo "Timeout waiting for server on port $port"
            return 1
        fi
        sleep 1
    done
}

# Common vLLM args shared by P and D.
common_args() {
    echo "--host 0.0.0.0 \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --trust-remote-code \
        --block-size 128 \
        --enable-prefix-caching \
        --no-disable-hybrid-kv-cache-manager \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 256"
}

main() {
    check_required_files
    check_num_gpus

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching BFF disaggregated serving components..."
    echo "Logs: prefill*.log / decode*.log / proxy.log"

    # ---- Proxy ----------------------------------------------------------------
    echo ""
    echo "Starting proxy server (HTTP $PROXY_HTTP_PORT / ZMQ $PROXY_PORT)..."
    python3 disagg_proxy_p2p_nccl_xpyd.py > proxy.log 2>&1 &
    PIDS+=($!)

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # ---- Prefill servers (producers) -----------------------------------------
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))

        echo "  Prefill $((i+1)): GPU $gpu_id, HTTP $port, KV $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id \
        BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_FUSE=$BFF_PD_FUSE BFF_GROUP_SIZE=$BFF_GROUP_SIZE \
        python3 -m kv_fast_fusion.fast_fusion_main serve $MODEL \
        $(common_args) \
        --port $port \
        --gpu-memory-utilization $PREFILL_GPU_UTIL \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnectorFF\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"$PREFILL_KV_BUFFER\",\"kv_ip\":\"$KV_IP\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$KV_IP\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
        > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # ---- Decode servers (consumers) ------------------------------------------
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))

        echo "  Decode $((i+1)): GPU $gpu_id, HTTP $port, KV $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id \
        BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_FUSE=$BFF_PD_FUSE BFF_GROUP_SIZE=$BFF_GROUP_SIZE \
        python3 -m kv_fast_fusion.fast_fusion_main serve $MODEL \
        $(common_args) \
        --port $port \
        --gpu-memory-utilization $DECODE_GPU_UTIL \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnectorFF\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"$DECODE_KV_BUFFER\",\"kv_ip\":\"$KV_IP\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$KV_IP\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
        > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # ---- Wait for all servers ------------------------------------------------
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done

    echo ""
    echo "All servers up. Running benchmark against proxy HTTP $PROXY_HTTP_PORT..."

    # ---- Benchmark (targets the PROXY, not a server) -------------------------
    vllm bench serve --host $KV_IP --port $PROXY_HTTP_PORT --seed $(date +%s) \
        --model $MODEL \
        --dataset-name random --random-input-len 7500 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 2 | tee benchmark.log

    echo "Benchmarking done. Cleaning up..."
    cleanup
}

main
