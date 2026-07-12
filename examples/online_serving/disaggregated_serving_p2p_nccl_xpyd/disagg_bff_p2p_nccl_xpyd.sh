#!/bin/bash

#example usage: NUM_PREFILL=2 NUM_DECODE=1 ./examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh
# example 2:
# NUM_PREFILL=2 NUM_DECODE=1 BFF_PD_MERGE=cc     BFF_THRESHOLD=0.75 BFF_GROUP_SIZE=4 \
#   ./examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh
#example 3:
# NUM_PREFILL=2 NUM_DECODE=1 BFF_PD_MERGE=nr_tree BFF_THRESHOLD=0.85 BFF_GROUP_SIZE=4 \
#   ./examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh
### Full BFF
# NUM_PREFILL=2 NUM_DECODE=1 BFF_PD_MERGE=cc BFF_THRESHOLD=0.75 \
#  ./examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh
### Fusion ablation (BFF layout, no merge) — the fully-fair "what does fusion buy" baseline
#NUM_PREFILL=2 NUM_DECODE=1 BFF_PD_FUSE=0 \
#  ./examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh
### True vanilla — stock vLLM P/D, single group (end-to-end reference, exposes layout cost)
# NUM_PREFILL=2 NUM_DECODE=1 BASELINE=vanilla \
#  ./examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh
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
#   * BFF env: BFF_SCALE_MODE=raw (transfer-safe, no KV mutation/scales) and
#     BFF_PD_FUSE=1 (connector-level layer-streamed fusion, sharing propagated to D).
#
# Topology (nPmD): proxy (HTTP 10001 / ZMQ 30001) -> n prefill -> m decode. The
# proxy round-robins requests across all registered prefill/decode instances, so
# n,m > 1 work with no proxy change. Default 1P1D. Example: NUM_PREFILL=2 NUM_DECODE=1
# (2 prefill GPUs feeding 1 decode → makes the DECODE instance the bottleneck, where
# BFF's freed-KV capacity benefit shows up). The benchmark targets the proxy HTTP port
# (10001), never a server directly (a direct hit lacks the proxy-injected
# ___decode_addr_ and skips transfer).
#
# Override via env vars:
#   MODEL, NUM_PREFILL (n), NUM_DECODE (m), HTTP_PORT_BASE, PROXY_PORT, KV_IP,
#   BFF_SCALE_MODE, BFF_PD_FUSE, BFF_GROUP_SIZE, TIMEOUT_SECONDS.
#   For a custom GPU/port mapping, override PREFILL_GPUS/DECODE_GPUS/PREFILL_PORTS/
#   DECODE_PORTS (comma-separated lists) directly — they win over NUM_*.
# =============================================================================

# ---- Model / topology --------------------------------------------------------
MODEL=${MODEL:-zai-org/GLM-4.7-Flash} #{MODEL:-NousResearch/Hermes-3-Llama-3.1-8B} #{MODEL:-zai-org/glm-4-9b-chat-hf}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}      # ZMQ service-discovery port (matches proxy_port in connector cfg)
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10001}   # proxy HTTP serving port (benchmark target)
KV_IP=${KV_IP:-10.10.10.174}
HF_HOME=${HF_HOME:-"/data/models/huggingface"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"/data/models/huggingface/hub"}

# ---- Topology: n prefill (P) x m decode (D) ----------------------------------
# Set NUM_PREFILL (n) and NUM_DECODE (m); GPUs and HTTP ports are auto-derived:
#   P  → GPUs [0 .. n-1]              D → GPUs [n .. n+m-1]
#   P  HTTP ports [HTTP_PORT_BASE .. +n-1]   D HTTP ports [HTTP_PORT_BASE+n .. +m-1]
#   KV ports (set in the launch loops) → P 21001+i, D 22001+i
# To customize the mapping, override PREFILL_GPUS/DECODE_GPUS/PREFILL_PORTS/DECODE_PORTS
# directly (comma-separated lists) — those win over the NUM_*-derived defaults.
NUM_PREFILL=${NUM_PREFILL:-1}        # n
NUM_DECODE=${NUM_DECODE:-1}          # m
TP=${TP:-4}                          # tensor-parallel size PER instance (each P/D gets TP GPUs)
HTTP_PORT_BASE=${HTTP_PORT_BASE:-20003}

# Build "start,start+1,...,start+count-1".
_seq_csv() { local start=$1 count=$2 out="" k; for ((k=0; k<count; k++)); do out+="$((start+k)),"; done; echo "${out%,}"; }

# GPUs are allocated TP-per-instance, packed contiguously: prefill i → [i*TP .. i*TP+TP-1],
# decode j → [NUM_PREFILL*TP + j*TP .. +TP-1]. For NUM_PREFILL=2 NUM_DECODE=1 TP=2 → P1=0,1
# P2=2,3 D=4,5 (same-NUMA TP pairs on the dual-socket box; cross-NUMA intra-TP crashed — plan ROUND 53).
PREFILL_GPUS=${PREFILL_GPUS:-$(_seq_csv 0 "$((NUM_PREFILL * TP))")}
DECODE_GPUS=${DECODE_GPUS:-$(_seq_csv "$((NUM_PREFILL * TP))" "$((NUM_DECODE * TP))")}
PREFILL_PORTS=${PREFILL_PORTS:-$(_seq_csv "$HTTP_PORT_BASE" "$NUM_PREFILL")}
DECODE_PORTS=${DECODE_PORTS:-$(_seq_csv "$((HTTP_PORT_BASE + NUM_PREFILL))" "$NUM_DECODE")}

# ---- Baseline mode -----------------------------------------------------------
# BASELINE=bff (default)  → BFF launcher + group-aware P2pNcclConnectorFF (the system under test).
# BASELINE=vanilla        → stock `vllm serve` + stock P2pNcclConnector, single KV-cache group, NO
#                           BFF patches/group-split — the true end-to-end reference. (It must launch
#                           via vllm.entrypoints.cli.main, since fast_fusion_main's import
#                           unconditionally applies the BFF group split. Chunked prefill defaults ON
#                           for both baselines now — some hybrid Mamba/attention models, e.g.
#                           Qwen3.5, hard-require it whenever prefix caching is enabled; see the
#                           ENABLE_CHUNKED comment below for why that's safe for this benchmark.)
# NOTE: BFF_PD_FUSE=0 is the *fusion ablation* (BFF layout, no merge) — NOT vanilla. Use FUSE=1 vs
# FUSE=0 to isolate fusion (fully fair); use BASELINE=vanilla for the layout-cost / end-to-end ref.
BASELINE=${BASELINE:-bff}
if [[ "$BASELINE" == "vanilla" ]]; then
    LAUNCHER="vllm.entrypoints.cli.main"
    CONNECTOR="P2pNcclConnector"
    HYBRID_FLAG=""                      # stock single-group default
    # Some hybrid Mamba/attention models (e.g. Qwen3.5) hard-require chunked prefill whenever
    # prefix caching is enabled (vLLM auto-sets mamba_cache_mode="align" when the model doesn't
    # support mamba_cache_mode="all", which asserts enable_chunked_prefill) -- so vanilla can no
    # longer default to chunked-off unconditionally. The stock P2pNcclConnector's own
    # chunked-prefill limitation only trips when a request's prompt genuinely spans multiple
    # scheduler steps, which cannot happen here since MAX_NUM_BATCHED_TOKENS==MAX_MODEL_LEN caps
    # every prompt in one step -- so defaulting to chunked ON is safe for this benchmark. Override
    # with ENABLE_CHUNKED=0 if testing a model/config where a request can genuinely span steps.
    ENABLE_CHUNKED=${ENABLE_CHUNKED:-1}
else
    LAUNCHER="kv_fast_fusion.fast_fusion_main"
    CONNECTOR="P2pNcclConnectorFF"
    HYBRID_FLAG="--no-disable-hybrid-kv-cache-manager"
    ENABLE_CHUNKED=${ENABLE_CHUNKED:-1}
fi
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
if [[ "$ENABLE_CHUNKED" == "1" ]]; then
    CHUNKED_FLAG="--enable-chunked-prefill"
else
    CHUNKED_FLAG="--no-enable-chunked-prefill"
    # With chunked prefill OFF, vLLM rejects max_num_batched_tokens < max_model_len (a single
    # prefill step must fit the whole sequence). Raise the batched-token cap to max_model_len so
    # vanilla starts. (F1 math prompts are short → they still complete in one step.)
    if (( MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
        MAX_NUM_BATCHED_TOKENS=$MAX_MODEL_LEN
    fi
fi

# ---- BFF knobs ---------------------------------------------------------------
# BFF_MERGE=${BFF_MERGE:-cc}           # merge fusion layers into a single KV cache group (cc, nr_tree)
# BFF_LSH_REPR=${BFF_LSH_REPR:-proj}     # full (default) or proj (LSH) representation for the fusion-layer KV
BFF_PD_MERGE=${BFF_PD_MERGE:-cc}         # merge P/D fusion layers into a single KV cache group (cc, nr_tree)
BFF_SCALE_MODE=${BFF_SCALE_MODE:-raw}   # raw is required for P/D (no KV mutation, no scales to ship)
BFF_PD_REPR=${BFF_PD_REPR:-proj}         # full (default) or proj (LSH) representation for the fusion-layer KV
BFF_PD_FUSE=${BFF_PD_FUSE:-1}           # connector-level fusion + redirect propagation to D
BFF_GROUP_SIZE=${BFF_GROUP_SIZE:-4}     # fusion layers packed per KV cache group
BFF_THRESHOLD=${BFF_THRESHOLD:-0.75}       # BFF fusion threshold (0.0-1.0, 0.75 default)
# ROUND 58: cross-batch fusion window. 0 = within-batch only (per-prefill-step, today's behavior).
# >0 = also match each prefill batch against a rolling registry of the last N requests' rep blocks
# (frees more on D → compression can exceed ~2×). Set N near the decode-resident request count;
# proj repr recommended to bound registry memory.
BFF_PD_ENCODED_BATCH_SIZE=${BFF_PD_ENCODED_BATCH_SIZE:-32}
# ---- GPU memory / recv-buffer tuning -----------------------------------------
# P only SENDS, so its recv-buffer threshold (kv_buffer_size) can be tiny (1e1).
# P only SENDS, so its recv-buffer threshold (kv_buffer_size) can be tiny (1e1).
# D's kv_buffer_size IS the threshold for GPU-RESIDENT recv KV before it spills to
# the CPU pool. It MUST be < D's free GPU memory after the KV cache, or D OOMs on
# the recv torch.empty (the "Peer Out Of Memory/Threshold, response:1" on P).
PREFILL_GPU_UTIL=${PREFILL_GPU_UTIL:-0.85}  # P frees blocks fast (low KV residency) but needs headroom
                                            # for the prefill activation spike — 0.95 OOMs the forward
DECODE_GPU_UTIL=${DECODE_GPU_UTIL:-0.75}     # headroom for transient recv buffers / NCCL / CPU-pool staging
PREFILL_KV_BUFFER=${PREFILL_KV_BUFFER:-1e1} # producer never receives → tiny
DECODE_KV_BUFFER=${DECODE_KV_BUFFER:-8e9}   # spill to CPU pool early; keep GPU-resident recv small

# ---- F1 accuracy + latency benchmark knobs -----------------------------------
# The run targets the PROXY (f1_main streams, so it captures TTFT/ITL/TPOT + throughput,
# and computes F1 against the HF dataset). Results are saved to a config-tagged JSON so a
# sweep over BFF_PD_MERGE × BFF_THRESHOLD × BFF_GROUP_SIZE is easy to tabulate.
F1_DATASET=${F1_DATASET:-m-a-p/CodeFeedback-Filtered-Instruction} #nvidia/OpenMathInstruct-2}
F1_SPLIT=${F1_SPLIT:-train}
F1_INPUT_KEY=${F1_INPUT_KEY:-query}
F1_OUTPUT_KEY=${F1_OUTPUT_KEY:-answer}
NUM_PROMPTS=${NUM_PROMPTS:-500}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
REQUEST_RATE=${REQUEST_RATE:-300}      # arrivals/s (stress test). 'inf' = fire all at once (cap by MAX_CONCURRENCY)
BURSTINESS=${BURSTINESS:-0.3}          # gamma shape: <1 burstier (spiky), 1=Poisson, >1 more uniform
MIN_TOKENS=${MIN_TOKENS:-512}            # skip prompts shorter than this many input tokens (0=off)
MAX_TOKENS=${MAX_TOKENS:-4096}         # per-request generation budget (must be < max_model_len - prompt)
# Guard: max_tokens >= max_model_len leaves no room for the prompt → the server rejects EVERY request
# ('max_tokens too large'). Clamp an over-large value (with headroom for the prompt) and warn loudly.
if (( MAX_TOKENS >= MAX_MODEL_LEN )); then
    _safe=$(( MAX_MODEL_LEN - 8192 )); (( _safe < 1024 )) && _safe=1024
    echo "  WARNING: MAX_TOKENS ($MAX_TOKENS) >= max_model_len ($MAX_MODEL_LEN): the server rejects"
    echo "           EVERY request (no room for the prompt). Clamping MAX_TOKENS to $_safe."
    echo "           Set MAX_TOKENS < max_model_len - longest_prompt to silence this."
    MAX_TOKENS=$_safe
fi
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1200}
RESULT_DIR=${RESULT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/f1_results}
# Tag each run by its config so summaries don't overwrite across the sweep.
if [[ "$BASELINE" == "vanilla" ]]; then
    RUN_TAG=${RUN_TAG:-vanilla_con_${MAX_CONCURRENCY}_${NUM_PREFILL}Px${NUM_DECODE}D}
else
    RUN_TAG=${RUN_TAG:-${BFF_PD_MERGE}_${BFF_SCALE_MODE}_${BFF_PD_REPR}_thr${BFF_THRESHOLD}_gs${BFF_GROUP_SIZE}_eb${BFF_PD_ENCODED_BATCH_SIZE}_con_${MAX_CONCURRENCY}_${NUM_PREFILL}Px${NUM_DECODE}D}
fi

# ---- Required BFF / HF environment (CLAUDE.md) -------------------------------
REPO_ROOT=${REPO_ROOT:-/data/users/sefi/from_git/vllm_013/vllm_ff}
export HF_HOME=${HF_HOME:-/data/models/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/data/models/huggingface/hub}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_V1=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

# The P2P NCCL engine loads "libnccl.so.2" by bare name. The venv ships
# nvidia-nccl 2.28.9+cuda13.0 (needs driver >=580); this host runs driver
# 575.51.03 (CUDA 12.9). Pin the engine to the system 2.28.3+cuda12.9 build,
# which inits a comm cleanly under this driver.
export VLLM_NCCL_SO_PATH=${VLLM_NCCL_SO_PATH:-/lib/x86_64-linux-gnu/libnccl.so.2}

# P↔D KV transfer runs over NCCL between GPUs in DISJOINT per-instance
# CUDA_VISIBLE_DEVICES namespaces. The crash needs TWO things together:
#   (1) a transfer rank r>0 — at r=0 the local device ordinal is 0, always valid
#       across namespaces, so P2P IPC succeeds (only Worker_TP{r>0} ever died); and
#   (2) that rank's transfer GPU pair is topologically "close" — same NUMA / PCIe
#       switch (PIX/PXB/PHB/NODE/NVLink in `nvidia-smi topo -m`), so NCCL attempts
#       direct-GPU P2P transport. Cross-socket (SYS) pairs already fall back to SHM.
# When both hold NCCL dies: "transport/p2p.cc Cuda failure 101 'invalid device
# ordinal'". So probe the actual topology of the rank>0 pairs THIS run will use
# and force host-staged (SHM) transport only when needed. Process-global because
# NCCL caches the param + it must be set before any NCCL init. Override by
# exporting NCCL_P2P_DISABLE yourself.
_p2p_needs_disable() {
    python3 - "$PREFILL_GPUS" "$DECODE_GPUS" "$TP" <<'PY' 2>/dev/null
import subprocess, sys, re
prefill = [int(x) for x in sys.argv[1].split(",") if x != ""]
decode  = [int(x) for x in sys.argv[2].split(",") if x != ""]
tp = int(sys.argv[3])
out = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
# Parse the GPUk x GPUk link-type matrix. The header row is wrapped in ANSI
# escapes by nvidia-smi, so strip them; match GPU labels strictly (GPU\d+) to
# avoid the trailing "GPU NUMA ID" header column.
ansi = re.compile(r"\x1b\[[0-9;]*m")
gpu = re.compile(r"GPU(\d+)$")
cols, link = None, {}
for line in out.splitlines():
    toks = ansi.sub("", line).split()
    if not toks:
        continue
    if cols is None and toks[0] == "GPU0" and "X" not in toks:
        cols = [int(m.group(1)) for t in toks if (m := gpu.match(t))]
        continue
    if cols and gpu.match(toks[0]) and "X" in toks:
        g = int(gpu.match(toks[0]).group(1))
        for c, v in zip(cols, toks[1:1 + len(cols)]):
            link[(g, c)] = v
# rank-matched pairs: prefill (i,r) <-> decode (j,r); r=0 never triggers it.
need = False
for r in range(1, tp):
    pg = [prefill[i * tp + r] for i in range(len(prefill) // tp)]
    dg = [decode[j * tp + r]  for j in range(len(decode) // tp)]
    for a in pg:
        for b in dg:
            if link.get((a, b), "SYS") not in ("SYS", "X"):
                need = True
print("1" if need else "0")
PY
}
_NEED_P2P_DISABLE=$(_p2p_needs_disable)
if [[ "$_NEED_P2P_DISABLE" == "1" ]]; then
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
elif [[ -z "$_NEED_P2P_DISABLE" && "$TP" -gt 1 ]]; then
    # Probe failed (no nvidia-smi/python) → conservatively disable for TP>1.
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
fi
[[ -n "${NCCL_P2P_DISABLE:-}" ]] && echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE (transfer-pair topo probe: ${_NEED_P2P_DISABLE:-probe-failed})"

echo "Warning: P2P NCCL disaggregated prefill XpYd for vLLM v1 is experimental."
echo ""
echo "BFF Disaggregated Configuration:"
echo "  Model:        $MODEL"
echo "  Topology:     ${NUM_PREFILL}P x ${NUM_DECODE}D   (TP=$TP per instance)"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS  (KV ports 21001+)"
echo "  Decode GPUs:  $DECODE_GPUS, Ports: $DECODE_PORTS  (KV ports 22001+)"
echo "  Proxy:        HTTP $PROXY_HTTP_PORT / ZMQ $PROXY_PORT   KV_IP $KV_IP"
echo "  Baseline:     $BASELINE   (launcher=$LAUNCHER  connector=$CONNECTOR  chunked_prefill=$ENABLE_CHUNKED)"
echo "  Traffic:      MAX_CONCURRENCY=$MAX_CONCURRENCY  REQUEST_RATE=$REQUEST_RATE  BURSTINESS=$BURSTINESS"
if [[ "$BASELINE" == "vanilla" ]]; then
echo "  BFF:          (disabled — stock vLLM single-group reference)"
else
echo "  BFF:          BFF_PD_MERGE=$BFF_PD_MERGE  BFF_SCALE_MODE=$BFF_SCALE_MODE  BFF_PD_REPR=$BFF_PD_REPR  BFF_PD_FUSE=$BFF_PD_FUSE BFF_THRESHOLD=$BFF_THRESHOLD  BFF_GROUP_SIZE=$BFF_GROUP_SIZE  ENCODED_BATCH=$BFF_PD_ENCODED_BATCH_SIZE"
fi
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
    local need=$(((NUM_PREFILL + NUM_DECODE) * TP))
    if [ "$num_gpus" -lt "$need" ]; then
        echo "You need at least $need GPUs ((${NUM_PREFILL}P + ${NUM_DECODE}D) × TP=$TP); found $num_gpus."
        exit 1
    fi
    echo "Found $num_gpus GPUs (using $need: (${NUM_PREFILL}P + ${NUM_DECODE}D) × TP=$TP)."
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

# Common vLLM args shared by P and D. $HYBRID_FLAG / $CHUNKED_FLAG depend on BASELINE (vanilla
# omits the hybrid flag and disables chunked prefill for the stock connector).
common_args() {
    echo "--host 0.0.0.0 \
        --tensor-parallel-size $TP \
        --seed 1024 \
        --trust-remote-code \
        --block-size 128 \
        --enable-prefix-caching \
        $HYBRID_FLAG \
        $CHUNKED_FLAG \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-seqs $MAX_CONCURRENCY"
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
    # Producers dump their cumulative fuse overhead + compression to bff_stats_<pid>.json here
    # (read back post-run — replaces the old periodic-log scrape). Clean stale files first.
    mkdir -p "$RESULT_DIR"
    rm -f "$RESULT_DIR"/bff_stats_*.json
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for ((i=0; i<${#PREFILL_PORT_ARRAY[@]}; i++)); do
        local gpu_id=$(IFS=','; echo "${PREFILL_GPU_ARRAY[*]:i*TP:TP}")   # TP GPUs for this instance
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i*TP))   # space by TP: each rank binds kv_port+rank

        echo "  Prefill $((i+1)): GPU $gpu_id, HTTP $port, KV $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id \
        BFF_PD_MERGE=$BFF_PD_MERGE BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_REPR=$BFF_PD_REPR BFF_PD_FUSE=$BFF_PD_FUSE \
        BFF_GROUP_SIZE=$BFF_GROUP_SIZE BFF_THRESHOLD=$BFF_THRESHOLD \
        BFF_PD_ENCODED_BATCH_SIZE=$BFF_PD_ENCODED_BATCH_SIZE \
        BFF_PD_STATS_DIR="$RESULT_DIR" \
        python3 -m $LAUNCHER serve $MODEL \
        $(common_args) \
        --port $port \
        --gpu-memory-utilization $PREFILL_GPU_UTIL \
        --kv-transfer-config \
        "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"$PREFILL_KV_BUFFER\",\"kv_ip\":\"$KV_IP\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$KV_IP\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
        > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # ---- Decode servers (consumers) ------------------------------------------
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for ((i=0; i<${#DECODE_PORT_ARRAY[@]}; i++)); do
        local gpu_id=$(IFS=','; echo "${DECODE_GPU_ARRAY[*]:i*TP:TP}")   # TP GPUs for this instance
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i*TP))   # space by TP: each rank binds kv_port+rank

        echo "  Decode $((i+1)): GPU $gpu_id, HTTP $port, KV $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id \
        BFF_PD_MERGE=$BFF_PD_MERGE BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_REPR=$BFF_PD_REPR BFF_PD_FUSE=$BFF_PD_FUSE \
        BFF_GROUP_SIZE=$BFF_GROUP_SIZE BFF_THRESHOLD=$BFF_THRESHOLD \
        python3 -m $LAUNCHER serve $MODEL \
        $(common_args) \
        --port $port \
        --gpu-memory-utilization $DECODE_GPU_UTIL \
        --kv-transfer-config \
        "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"$DECODE_KV_BUFFER\",\"kv_ip\":\"$KV_IP\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"$KV_IP\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" \
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
    echo "All servers up. Running F1 benchmark against proxy HTTP $PROXY_HTTP_PORT..."
    echo "  run tag: $RUN_TAG   →   $RESULT_DIR/f1_${RUN_TAG}.{json,log}"
    mkdir -p "$RESULT_DIR"

    # ---- F1 + latency benchmark (targets the PROXY, not a server) ------------
    # Streams, so it reports TTFT / ITL / TPOT + throughput, computes F1, and saves a
    # config-tagged summary JSON (accuracy + throughput + end-to-end time + latency).
    python3 -m f1_benchmark.f1_main \
        --model $MODEL --host $KV_IP --port $PROXY_HTTP_PORT \
        --dataset-path $F1_DATASET --hf-split $F1_SPLIT \
        --input-key $F1_INPUT_KEY --output-key $F1_OUTPUT_KEY \
        --num-prompts $NUM_PROMPTS --max-concurrency $MAX_CONCURRENCY \
        --request-rate $REQUEST_RATE --burstiness $BURSTINESS \
        --min-tokens $MIN_TOKENS --max-tokens $MAX_TOKENS \
        --request-timeout $REQUEST_TIMEOUT \
        --compute-f1 --result-dir "$RESULT_DIR" \
        --result-file "$RESULT_DIR/f1_${RUN_TAG}.json" \
        --label "$RUN_TAG" 2>&1 | tee "$RESULT_DIR/f1_${RUN_TAG}.log"

    # ---- Collect BFF metrics into the JSON (all POST-run → ZERO impact on throughput/elapsed) --
    # Producer overhead + compression: read from the per-process bff_stats_*.json the producers
    # dump in $RESULT_DIR (always-current cumulative totals — no log flood, no throttled-line scrape
    # that a prefill-only producer would miss). Scheduler/consumer metrics still scraped from logs:
    #   "BFF sched | ... free_blocks=F / T | block_usage=U% | running=.. | preempt(cum)=.."
    #   "Block merging freed N blocks"            (D net free)   "redirects_applied=N | reps_unresolved=M"
    # Merged under bff_overhead / bff_compression / bff_sched / bff_blocks_freed / bff_redirects_applied.
    python3 - "$RESULT_DIR/f1_${RUN_TAG}.json" "$RESULT_DIR" prefill*.log decode*.log <<'PY'
import glob, json, os, re, sys
result_file, stats_dir, logs = sys.argv[1], sys.argv[2], sys.argv[3:]
sched_pat = re.compile(r"BFF sched \| step=\d+ \| running=(\d+) \| waiting=(\d+) \| "
                       r"free_blocks=(\d+) / (\d+) \| block_usage=([\d.]+)% \| preempt\(cum\)=(\d+)")
freed_pat = re.compile(r"Block merging freed (-?\d+) blocks")
redir_pat = re.compile(r"redirects_applied=(\d+) \| reps_unresolved=(\d+)")

def stats(xs):
    return None if not xs else {"min": min(xs), "mean": sum(xs) / len(xs),
                                "max": max(xs), "last": xs[-1]}

# Producer fuse stats (overhead + compression) from the dumped per-process JSON files.
ov_per, cm_per = {}, {}
for sf in sorted(glob.glob(os.path.join(stats_dir, "bff_stats_*.json"))):
    try:
        with open(sf) as f:
            s = json.load(f)
    except Exception:
        continue
    name = os.path.basename(sf)
    if s.get("steps"):
        ov_per[name] = {"avg_group_dedup_ms": s["overhead_avg_group_dedup_ms"],
                        "groups": s["steps"]}
    if s.get("total_blocks"):
        cm_per[name] = {"avg_factor": s["compression_avg_factor"],
                        "total_blocks": s["total_blocks"], "freed": s["freed"],
                        "per_group": {int(gi): r for gi, r
                                      in s.get("compression_per_group", {}).items()}}

sched_per, freed_per, redir_per = {}, {}, {}
for lg in logs:
    runs, waits, frees, usages, total, preempt_last = [], [], [], [], None, None
    freed_sum = freed_cnt = redir_app = redir_unres = redir_cnt = 0
    try:
        with open(lg) as f:
            for line in f:
                s = sched_pat.search(line)
                if s:
                    runs.append(int(s.group(1))); waits.append(int(s.group(2)))
                    frees.append(int(s.group(3))); total = int(s.group(4))
                    usages.append(float(s.group(5))); preempt_last = int(s.group(6))
                fr = freed_pat.search(line)
                if fr:
                    freed_sum += int(fr.group(1)); freed_cnt += 1
                rd = redir_pat.search(line)
                if rd:
                    redir_app += int(rd.group(1)); redir_unres += int(rd.group(2)); redir_cnt += 1
    except FileNotFoundError:
        continue
    if frees:
        sched_per[lg] = {"total_blocks": total, "free_blocks": stats(frees),
                         "block_usage_pct": stats(usages), "running": stats(runs),
                         "waiting": stats(waits), "preempt_cum": preempt_last}
    if freed_cnt:
        freed_per[lg] = {"net_blocks_freed": freed_sum, "merge_events": freed_cnt}
    if redir_cnt:
        redir_per[lg] = {"redirects_applied": redir_app, "reps_unresolved": redir_unres,
                         "apply_calls": redir_cnt}

try:
    with open(result_file) as f:
        data = json.load(f)
except Exception:
    data = {}

if ov_per:
    avg = sum(v["avg_group_dedup_ms"] for v in ov_per.values()) / len(ov_per)
    data["bff_overhead"] = {"producer_avg_group_dedup_ms": avg, "per_prefill": ov_per}
    print(f"  bff overhead: producer avg group dedup {avg:.3f} ms")
else:
    print("  bff overhead: no bff_stats_*.json with steps>0 "
          "(BFF_PD_FUSE off, or producer ran no fusion groups)")

if cm_per:
    # Compression FACTOR = total/(total-freed) = how many× smaller the KV cache gets. Overall is
    # block-weighted across producers (ΣB / Σ(B-freed)); per-group is the mean factor across them.
    B = sum(v["total_blocks"] for v in cm_per.values())
    F = sum(v["freed"] for v in cm_per.values())
    avg_factor = B / max(1, B - F)
    gids = sorted({gi for v in cm_per.values() for gi in v["per_group"]})
    per_group = {gi: sum(v["per_group"][gi] for v in cm_per.values() if gi in v["per_group"])
                     / sum(1 for v in cm_per.values() if gi in v["per_group"]) for gi in gids}
    data["bff_compression"] = {"avg_factor": avg_factor, "per_group": per_group, "per_prefill": cm_per}
    print(f"  bff compression (x smaller): avg_factor={avg_factor:.4f} | per_group "
          + " ".join(f"g{gi}={r:.4f}" for gi, r in per_group.items()))
else:
    print("  bff compression: no bff_stats_*.json with total_blocks>0")

if sched_per:
    data["bff_sched"] = sched_per
    for lg, v in sched_per.items():
        fb, us = v["free_blocks"], v["block_usage_pct"]
        print(f"  bff sched [{lg}]: free_blocks min={fb['min']} mean={fb['mean']:.0f} "
              f"last={fb['last']} / {v['total_blocks']} | block_usage% max={us['max']:.1f} "
              f"mean={us['mean']:.1f} | running mean={v['running']['mean']:.0f} "
              f"max={v['running']['max']} | preempt(cum)={v['preempt_cum']}")
if freed_per:
    data["bff_blocks_freed"] = freed_per
    for lg, v in freed_per.items():
        print(f"  bff blocks freed [{lg}]: net={v['net_blocks_freed']} "
              f"over {v['merge_events']} merge events")
if redir_per:
    data["bff_redirects_applied"] = redir_per
    for lg, v in redir_per.items():
        print(f"  bff redirects applied [{lg}]: {v['redirects_applied']} "
              f"(reps_unresolved={v['reps_unresolved']})")

if ov_per or cm_per or sched_per or freed_per or redir_per:
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → merged into {result_file}")

# ---- one consolidated screen summary (accuracy + throughput + latency + BFF) ----
def _lat(d):
    return (f"mean={d['mean']:.1f} med={d['median']:.1f} p99={d['p99']:.1f}"
            if isinstance(d, dict) else "n/a")
f1 = data.get("mean_f1")
rps = data.get("request_throughput_rps")
otps = data.get("output_throughput_toks_s")
el = data.get("elapsed_s")
print(f"\n===== SUMMARY [{data.get('label', '')}] =====")
print(f"  accuracy: F1={f1:.4f}" if isinstance(f1, (int, float)) else "  accuracy: F1=n/a")
print(f"  throughput: {rps:.2f} req/s"
      + (f" | {otps:.1f} output tok/s" if isinstance(otps, (int, float)) else "")
      + (f" | elapsed {el:.1f}s" if isinstance(el, (int, float)) else ""))
print(f"  latency ms: TTFT[{_lat(data.get('ttft_ms'))}] "
      f"TPOT[{_lat(data.get('tpot_ms'))}] ITL[{_lat(data.get('itl_ms'))}]")
if "bff_compression" in data:
    print(f"  bff: compression {data['bff_compression']['avg_factor']:.3f}x smaller"
          + (f" | fusion overhead {data['bff_overhead']['producer_avg_group_dedup_ms']:.3f} ms/group"
             if "bff_overhead" in data else ""))
print("=" * (len(data.get("label", "")) + 18))
PY

    # Reference throughput sweep (random dataset, no F1) — uncomment to use instead:
    # vllm bench serve --host $KV_IP --port $PROXY_HTTP_PORT --seed $(date +%s) \
    #     --model $MODEL --dataset-name random --random-input-len 7500 --random-output-len 200 \
    #     --num-prompts 500 --burstiness 100 --request-rate 200 | tee "$RESULT_DIR/bench_${RUN_TAG}.log"

    echo "Benchmarking done. Cleaning up..."
    cleanup
}

main
